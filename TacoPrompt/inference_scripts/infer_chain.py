import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) 


import argparse
import collections
import os
import time
import warnings
from functools import partial
from math import ceil

import numpy as np
import json
import torch
import transformers

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import trainer as trainer_arch
import wandb
from parse_config import ConfigParser

from tqdm import tqdm
import more_itertools as mit
from torch.cuda.amp import autocast as autocast
from collections import defaultdict
import pickle
import logging
import itertools
MAX_CANDIDATE_NUM = 100000
import random
import datetime


def load_data(config):
    test_data_loader = config.initialize('train_data_loader', module_data, config['data_path'])
    return test_data_loader

def load_model(model_path,config,test_dataset):
    state = torch.load(model_path)
    state_dict = state['state_dict']
    model = config.initialize('arch', module_arch,test_dataset)
    model.load_state_dict(state_dict,strict=False)
    model.eval()
    return model

def rearrange(energy_scores, candidate_position_idx, true_position_idx):
    if true_position_idx == []:
        true_position_idx = [[]]
    tmp = np.array([[x == y for x in candidate_position_idx] for y in true_position_idx]).any(0)
    correct = np.where(tmp)[0]
    incorrect = np.where(~tmp)[0]
    labels = torch.cat((torch.ones(len(correct)), torch.zeros(len(incorrect)))).int()
    energy_scores = torch.cat((energy_scores[correct], energy_scores[incorrect]))
    return energy_scores, labels

class Tester:
    def __init__(self, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None, hit_metrics=None, recall_metrics=None):
        super(Tester, self).__init__()
        self.model = model.cuda()
        self.metrics = metrics
        self.hit_metrics = hit_metrics
        self.recall_metrics = recall_metrics
        self.pre_metric = pre_metric
        self.test_batch_size = config['trainer']['test_batch_size']
        self.device = torch.device('cuda' if config["n_gpu"] > 0 else 'cpu')

        self.config = config
        self.train_data_loader = None

        self.data_loader = data_loader
        dataset = self.data_loader.dataset
        self.candidate_positions = data_loader.dataset.all_edges
        if len(self.candidate_positions) > MAX_CANDIDATE_NUM:
            valid_pos = set(itertools.chain.from_iterable(dataset.valid_node2pos.values()))
            valid_neg = list(set(self.candidate_positions).difference(valid_pos))
            valid_sample_size = max(MAX_CANDIDATE_NUM - len(valid_pos), 0)
            self.valid_candidate_positions = random.sample(valid_neg, valid_sample_size) + list(valid_pos)
        else:
            self.valid_candidate_positions = self.candidate_positions
        self.valid_candidate_positions = self.candidate_positions

    def infer(self):
        mode = 'test'
        logger = logging.getLogger("1")
        total_metrics, leaf_metrics, nonleaf_metrics = self._test(mode)
        for i, mtr in enumerate(self.metrics):
            logger.info('    {:15s}: {:.3f}'.format('test_' + mtr.__name__, total_metrics[i]))
        for i, mtr in enumerate(self.metrics):
            logger.info('    {:15s}: {:.3f}'.format('test_leaf_' + mtr.__name__, leaf_metrics[i]))
        for i, mtr in enumerate(self.metrics):
            logger.info('    {:15s}: {:.3f}'.format('test_nonleaf_' + mtr.__name__, nonleaf_metrics[i]))


    def _test(self, mode, gpu=True, q_val = 100):
        assert mode in ['test', 'validation']
        model = self.model if gpu else self.model.cpu()

        para = isinstance(model, torch.nn.DataParallel)
        if para:
            model = model.module

        batch_size = self.test_batch_size

        model.eval()
        with torch.no_grad():
            dataset = self.data_loader.dataset
            if mode == 'test':
                queries = dataset.test_node_list
                node2pos = dataset.test_node2pos
                graph = dataset.test_holdout_subgraph
                candidate_positions = self.candidate_positions
                taxon2id = dataset.test_taxon2id
                id2taxon = dataset.test_id2taxon
            else:
                queries = dataset.valid_node_list
                node2pos = dataset.valid_node2pos
                graph = dataset.valid_holdout_subgraph
                candidate_positions = self.valid_candidate_positions
                taxon2id = dataset.valid_taxon2id
                id2taxon = dataset.valid_id2taxon
            taxon2id.keys()
            # id for test
            taxon2allemb_id = model.train_taxon2allemb_id  
            taxon2all_node_id = model.taxon2id  
            all_node_id_2all_embed_id = model.all_node_id_2all_embed_id
            
            pseudo_leaf_id = taxon2allemb_id[dataset.pseudo_leaf_node]
            pseudo_root_id = taxon2allemb_id[dataset.pseudo_root_node]
            
            all_ranks = []
            leaf_queries = []
            leaf_ranks, nonleaf_ranks = [], []
            
            # begin per query prediction
            eval_queries = queries
            print(len(eval_queries))
            
            q_id = torch.tensor([taxon2all_node_id[query] for i, query in enumerate(eval_queries)])

            # find leaf and nonleaf
            # redefine nonleaf
            num = 0 
            for i, query in enumerate(eval_queries):
                poses = node2pos[query]
                flag = True
                for pos in poses:
                    if pos[1] != dataset.pseudo_leaf_node:
                        flag = False
                        break
                if flag:
                    leaf_queries.append(query)
                    num+=1
            print(num)

            for i, query in tqdm(enumerate(eval_queries), desc=mode, total=len(eval_queries)):
                batched_energy_scores = []
                query_id = taxon2all_node_id[query]
                for edges in mit.sliced(candidate_positions, batch_size):
                    edges = list(edges)
                    ps, cs = zip(*edges)

                    with autocast():
                        p_idx = torch.tensor([taxon2allemb_id[n] for n in ps]).to(self.device)
                        c_idx = torch.tensor([taxon2allemb_id[n] for n in cs]).to(self.device)
                        pe, ce = map(lambda x: torch.logical_and(x != pseudo_leaf_id, x != pseudo_root_id),
                                             [p_idx, c_idx])
                        q_idx = torch.tensor([q_id[i] for j in range(p_idx.shape[0])])
                        pseudo_pct = torch.tensor([1 for j in range(p_idx.shape[0])])
                        _, p_ans = model.scorer(p_idx, q_idx, c_idx, pseudo_pct, pseudo_pct, pseudo_pct, pe=pe, ce=ce, subtask='p')
                        _, c_ans = model.scorer(p_idx, q_idx, c_idx, pseudo_pct, pseudo_pct, pseudo_pct, pe=pe, ce=ce, p_ans=p_ans, subtask='c')
                        scores, _ = model.scorer(p_idx, q_idx, c_idx, pseudo_pct, pseudo_pct, pseudo_pct, pe=pe, ce=ce, p_ans=p_ans, c_ans=c_ans, subtask='a')
                        scores = scores.squeeze(1)
                    batched_energy_scores.append(scores)
                batched_energy_scores_cat = torch.cat(batched_energy_scores)

                # Total
                batched_energy_scores_cat, labels = rearrange(batched_energy_scores_cat, candidate_positions,
                                                              node2pos[query])
                ranks = self.pre_metric(batched_energy_scores_cat, labels)
                all_ranks.extend(ranks)

                if query in leaf_queries:
                    leaf_ranks.extend(ranks)
                else:
                    nonleaf_ranks.extend(ranks)

            print(all_ranks)
            total_metrics = [metric(all_ranks) for metric in self.metrics]
            leaf_metrics = [metric(leaf_ranks) for metric in self.metrics]
            nonleaf_metrics = [metric(nonleaf_ranks) for metric in self.metrics]

            print(f'leaf_metrics:{leaf_metrics}')
            print(f'nonleaf_metrics:{nonleaf_metrics}')
        return total_metrics, leaf_metrics, nonleaf_metrics

    

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='inference')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--suffix', default="", type=str, help='suffix indicating this run (default: None)')
    args.add_argument('-n', '--n_trials', default=1, type=int, help='number of trials (default: 1)')
    args.add_argument('-m', '--model_path', default=None, type=str, help='path of saved checkpoint (default: None)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--exp'], type=str, target=('exp_name',)),
        # Data loader (self-supervision generation)
        CustomArgs(['--train_data'], type=str, target=('train_data_loader', 'args', 'data_path')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('train_data_loader', 'args', 'batch_size')),
        CustomArgs(['--ns', '--negative_size'], type=int, target=('train_data_loader', 'args', 'negative_size')),
        CustomArgs(['--ef', '--expand_factor'], type=int, target=('train_data_loader', 'args', 'expand_factor')),
        CustomArgs(['--crt', '--cache_refresh_time'], type=int,
                   target=('train_data_loader', 'args', 'cache_refresh_time')),
        CustomArgs(['--nw', '--num_workers'], type=int, target=('train_data_loader', 'args', 'num_workers')),
        CustomArgs(['--sm', '--sampling_mode'], type=int, target=('train_data_loader', 'args', 'sampling_mode')),
        # Trainer & Optimizer
        CustomArgs(['--mode'], type=str, target=('mode',)),
        CustomArgs(['--loss'], type=str, target=('loss',)),
        CustomArgs(['--ep', '--epochs'], type=int, target=('trainer', 'epochs')),
        CustomArgs(['--es', '--early_stop'], type=int, target=('trainer', 'early_stop')),
        CustomArgs(['--tbs', '--test_batch_size'], type=int, target=('trainer', 'test_batch_size')),
        CustomArgs(['--v', '--verbose_level'], type=int, target=('trainer', 'verbosity')),
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--wd', '--weight_decay'], type=float, target=('optimizer', 'args', 'weight_decay')),
        CustomArgs(['--l1'], type=float, target=('trainer', 'l1')),
        CustomArgs(['--l2'], type=float, target=('trainer', 'l2')),
        CustomArgs(['--l3'], type=float, target=('trainer', 'l3')),
        # Model architecture
        CustomArgs(['--pm', '--propagation_method'], type=str, target=('arch', 'args', 'propagation_method')),
        CustomArgs(['--rm', '--readout_method'], type=str, target=('arch', 'args', 'readout_method')),
        CustomArgs(['--mm', '--matching_method'], type=str, target=('arch', 'args', 'matching_method')),
        CustomArgs(['--k'], type=int, target=('arch', 'args', 'k')),
        CustomArgs(['--in_dim'], type=int, target=('arch', 'args', 'in_dim')),
        CustomArgs(['--hidden_dim'], type=int, target=('arch', 'args', 'hidden_dim')),
        CustomArgs(['--out_dim'], type=int, target=('arch', 'args', 'out_dim')),
        CustomArgs(['--pos_dim'], type=int, target=('arch', 'args', 'pos_dim')),
        CustomArgs(['--num_heads'], type=int, target=('arch', 'args', 'heads', 0)),
        CustomArgs(['--feat_drop'], type=float, target=('arch', 'args', 'feat_drop')),
        CustomArgs(['--attn_drop'], type=float, target=('arch', 'args', 'attn_drop')),
        CustomArgs(['--hidden_drop'], type=float, target=('arch', 'args', 'hidden_drop')),
        CustomArgs(['--out_drop'], type=float, target=('arch', 'args', 'out_drop')),
        # TC added
        CustomArgs(['--lp'], type=float, target=('trainer', 'lp')),
        CustomArgs(['--lc'], type=float, target=('trainer', 'lc')),
        CustomArgs(['--lb'], type=float, target=('trainer', 'lb')),
        CustomArgs(['--lw'], type=float, target=('trainer', 'lw')),
    ]
    config = ConfigParser(args, options)
    args = args.parse_args()

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    setup_seed(47)

    test_data_loader = load_data(config)
    loss = getattr(module_loss, config['loss'])
    if config['loss'].startswith("FocalLoss"):
        loss = loss()
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    hit_metrics = [getattr(module_metric, met) for met in config['hit_metrics']]
    recll_metrics = [getattr(module_metric, met) for met in config['recall_metrics']]
    
    if config['loss'].startswith("info_nce") or config['loss'].startswith("bce_loss"):
        pre_metric = partial(module_metric.obtain_ranks, mode=1)  # info_nce_loss
    else:
        pre_metric = partial(module_metric.obtain_ranks, mode=0)
    
    model_path = args.model_path
    assert model_path != None,"Please enter model path!"
    model = load_model(model_path,config=config,test_dataset=test_data_loader.dataset)

    test_bs = config["trainer"]["test_batch_size"]
    tester = Tester(model, loss, metrics, pre_metric, optimizer=None, config=config, data_loader=test_data_loader, lr_scheduler=None,hit_metrics=hit_metrics,recall_metrics=recll_metrics)
    tester.infer()
    
