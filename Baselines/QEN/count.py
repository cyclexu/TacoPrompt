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



def load_data(config):
    test_data_loader = config.initialize('train_data_loader', module_data, config['data_path'])
    return test_data_loader

def load_model(model_path,config):
    # state = torch.load(model_path)
    # state_dict = state['state_dict']
    model = config.initialize('arch', module_arch)
    # model.load_state_dict(state_dict,strict=False)
    model.eval()
    return model

class Tester:
    def __init__(self,model,data_loader,test_bs,config):
        super(Tester, self).__init__()
        self.model = model
        self.test_batch_size = test_bs
        self.device = torch.device('cuda' if config["n_gpu"] > 0 else 'cpu')
        self.data_loader = data_loader
        self.candidate_positions = data_loader.dataset.all_edges
        self.valid_candidate_positions = data_loader.dataset.all_edges
        self.get_candidate_position_info()
        self.all_taxon2node_id = self.data_loader.dataset.taxon2id
        self.query2pos2score = {}

    def get_candidate_position_info(self):
        candidate_position2_posid = {}
        posid2_candidate_position = []
        taxon2node_id = self.data_loader.dataset.taxon2id
        for taxon_p, taxon_c in self.candidate_positions:
            node_id_p = taxon2node_id[taxon_p]
            node_id_c = taxon2node_id[taxon_c]
            # 候选位置->候选位置id
            if (node_id_p, node_id_c) not in candidate_position2_posid:
                candidate_position2_posid[(node_id_p, node_id_c)] = len(posid2_candidate_position)
                posid2_candidate_position.append((node_id_p, node_id_c))
        self.candidate_position2_posid = candidate_position2_posid
        self.posid2_candidate_position = posid2_candidate_position
        print(self.candidate_position2_posid)

    def test_to_get_json(self):
        for mode in ['test', 'validation']:
            self._test(mode)
        with open("query2pos2score.json",'w') as f:
            json.dump(self.query2pos2score,f)

    def _test(self,mode,gpu=True):
        assert mode in ['test', 'validation']

        model = self.model.cuda() if gpu else self.model.cpu()

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
            pseudo_leaf_id = taxon2id[dataset.pseudo_leaf_node]
            pseudo_root_id = taxon2id[dataset.pseudo_root_node]
            # self.logger.info(f'number of candidate positions: {len(candidate_positions)}')

            # generate code representations for all seed nodes
            debug = False
            batched_codes = []
            for node_idx in tqdm(mit.sliced(list(range(len(graph.nodes()))), batch_size)):
                descs = [id2taxon[i].description for i in node_idx]
                descs = self.data_loader.tokenizer(descs, return_tensors='pt', padding=True, truncation=True,
                                                   max_length=64)
                if debug:
                    print(descs)
                descs = {k: v.to(self.device) for k, v in descs.items()}
                with autocast():
                    codes = model.poly_encoder(descs, debug=debug)
                if debug:
                    print(codes)
                batched_codes.append(codes)
                debug = False
            codes = torch.cat(batched_codes, dim=0)  # nodes * m * dim
            # print(codes.shape)

            # start per query prediction
            all_ranks = []
            leaf_queries = []
            leaf_ranks, nonleaf_ranks = [], []

            # begin per query prediction
            debug = False
            eval_queries = queries

            # find leaf and nonleaf
            temp = []
            for i, query in enumerate(eval_queries):
                if node2pos[query][0][1] == dataset.pseudo_leaf_node:
                    leaf_queries.append(query)
                    temp.append(True)
                else:
                    temp.append(False)

            # print(temp)
            # exit(0)

            for i, query in tqdm(enumerate(eval_queries), desc=mode, total=len(eval_queries)):
                now_query_id = self.all_taxon2node_id[query]
                if now_query_id in self.query2pos2score.keys():
                    continue
                batched_energy_scores = []
                query_index = torch.tensor(taxon2id[query]).to(self.device)
                query_code = torch.index_select(codes, 0, query_index)  # 1 * m * dim

                descs = [id2taxon[i].description + id2taxon[i].description for i in node_idx]
                descs = self.data_loader.tokenizer(descs, return_tensors='pt', padding=True)
                descs = {k: v.to(self.device) for k, v in descs.items()}
                temp = model.poly_encoder(descs, debug=debug)
                temp += 1

                # find best/worst siblings in seed taxonomy
                s = time.time()
                best_worsts = {
                    k: [taxon2id[n] for n in dataset.get_best_worst_siblings(k, query, graph, train=False)] for k in
                    dataset.core_subgraph.nodes()}  # taxon: best and worst's ids
                best_worst_time = time.time() - s

                # compute
                s = time.time()
                tmp_dic = {}
                for edges in mit.sliced(candidate_positions, batch_size):
                    edges = list(edges)
                    ps, cs = zip(*edges)

                    with autocast():
                        p_idx = torch.tensor([taxon2id[n] for n in ps]).to(self.device)
                        c_idx = torch.tensor([taxon2id[n] for n in cs]).to(self.device)
                        b_idx = torch.tensor([best_worsts[p][0] for p in ps]).to(self.device)
                        w_idx = torch.tensor([best_worsts[p][1] for p in ps]).to(self.device)

                        pe, ce, be, we = map(lambda x: torch.logical_and(x != pseudo_leaf_id, x != pseudo_root_id),
                                             [p_idx, c_idx, b_idx, w_idx])
                        pt, ct, bt, wt = map(lambda x: torch.index_select(input=codes, dim=0, index=x),
                                             [p_idx, c_idx, b_idx, w_idx])
                        # if debug:
                        #     print(f'widx: {w_idx}')
                        #     print(pt, wt)
                        qt = query_code.expand(pt.shape[0], -1, -1)
                        scores, _, _, _, _ = model.poly_scorer(qt, pt, ct, bt, wt, pe, ce, be, we, debug=debug)
                        for _ in range(p_idx.shape[0]):
                            # pos_id = self.candidate_position2_posid[(p_idx[_].item(),c_idx[_].item())]
                            # tmp_dic[pos_id] = scores[_].item()
                            pos_id = self.candidate_position2_posid[(self.all_taxon2node_id[id2taxon[p_idx[_].item()]],self.all_taxon2node_id[id2taxon[c_idx[_].item()]])]
                            assert pos_id not in tmp_dic.keys()
                            tmp_dic[pos_id] = scores[_].item()
                    if debug:
                        print(scores.shape)
                        print(scores.squeeze())
                    debug = False
                    batched_energy_scores.append(scores)
                
                self.query2pos2score[self.all_taxon2node_id[query]] = tmp_dic
                batched_energy_scores_cat = torch.cat(batched_energy_scores)
                calc_time = time.time() - s

        
        

    


if __name__ == '__main__':
    # args = argparse.ArgumentParser(description='Training taxonomy expansion model')
    # args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    # args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    # args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    # args.add_argument('-s', '--suffix', default="", type=str, help='suffix indicating this run (default: None)')
    # args.add_argument('-n', '--n_trials', default=1, type=int, help='number of trials (default: 1)')
    # args.add_argument('-m', '--model_path', default=None, type=str, help='number of trials (default: 1)')
    # # custom cli options to modify configuration from default values given in json file.
    # CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    # options = [
    #     CustomArgs(['--exp'], type=str, target=('exp_name',)),
    #     # Data loader (self-supervision generation)
    #     CustomArgs(['--train_data'], type=str, target=('train_data_loader', 'args', 'data_path')),
    #     CustomArgs(['--bs', '--batch_size'], type=int, target=('train_data_loader', 'args', 'batch_size')),
    #     CustomArgs(['--ns', '--negative_size'], type=int, target=('train_data_loader', 'args', 'negative_size')),
    #     CustomArgs(['--ef', '--expand_factor'], type=int, target=('train_data_loader', 'args', 'expand_factor')),
    #     CustomArgs(['--crt', '--cache_refresh_time'], type=int,
    #                target=('train_data_loader', 'args', 'cache_refresh_time')),
    #     CustomArgs(['--nw', '--num_workers'], type=int, target=('train_data_loader', 'args', 'num_workers')),
    #     CustomArgs(['--sm', '--sampling_mode'], type=int, target=('train_data_loader', 'args', 'sampling_mode')),
    #     # Trainer & Optimizer
    #     CustomArgs(['--mode'], type=str, target=('mode',)),
    #     CustomArgs(['--loss'], type=str, target=('loss',)),
    #     CustomArgs(['--ep', '--epochs'], type=int, target=('trainer', 'epochs')),
    #     CustomArgs(['--es', '--early_stop'], type=int, target=('trainer', 'early_stop')),
    #     CustomArgs(['--tbs', '--test_batch_size'], type=int, target=('trainer', 'test_batch_size')),
    #     CustomArgs(['--v', '--verbose_level'], type=int, target=('trainer', 'verbosity')),
    #     CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
    #     CustomArgs(['--wd', '--weight_decay'], type=float, target=('optimizer', 'args', 'weight_decay')),
    #     CustomArgs(['--l1'], type=float, target=('trainer', 'l1')),
    #     CustomArgs(['--l2'], type=float, target=('trainer', 'l2')),
    #     CustomArgs(['--l3'], type=float, target=('trainer', 'l3')),
    #     # Model architecture
    #     CustomArgs(['--pm', '--propagation_method'], type=str, target=('arch', 'args', 'propagation_method')),
    #     CustomArgs(['--rm', '--readout_method'], type=str, target=('arch', 'args', 'readout_method')),
    #     CustomArgs(['--mm', '--matching_method'], type=str, target=('arch', 'args', 'matching_method')),
    #     CustomArgs(['--k'], type=int, target=('arch', 'args', 'k')),
    #     CustomArgs(['--in_dim'], type=int, target=('arch', 'args', 'in_dim')),
    #     CustomArgs(['--hidden_dim'], type=int, target=('arch', 'args', 'hidden_dim')),
    #     CustomArgs(['--out_dim'], type=int, target=('arch', 'args', 'out_dim')),
    #     CustomArgs(['--pos_dim'], type=int, target=('arch', 'args', 'pos_dim')),
    #     CustomArgs(['--num_heads'], type=int, target=('arch', 'args', 'heads', 0)),
    #     CustomArgs(['--feat_drop'], type=float, target=('arch', 'args', 'feat_drop')),
    #     CustomArgs(['--attn_drop'], type=float, target=('arch', 'args', 'attn_drop')),
    #     CustomArgs(['--hidden_drop'], type=float, target=('arch', 'args', 'hidden_drop')),
    #     CustomArgs(['--out_drop'], type=float, target=('arch', 'args', 'out_drop')),
    #     # TC added
    #     CustomArgs(['--lp'], type=float, target=('trainer', 'lp')),
    #     CustomArgs(['--lc'], type=float, target=('trainer', 'lc')),
    #     CustomArgs(['--lb'], type=float, target=('trainer', 'lb')),
    #     CustomArgs(['--lw'], type=float, target=('trainer', 'lw')),
    # ]
    # config = ConfigParser(args, options)
    # args = args.parse_args()

    # test_data_loader = load_data(config)
    # dataset = test_data_loader.dataset
    # full_graph = dataset.full_graph
    # taxon2id = dataset.taxon2id
    # print(len(full_graph.nodes()))
    # print(len(list(taxon2id.keys())))
    # test_to_get_json(test_data_loader,model)

    # with open("query2pos2score.json") as f:
    #     dic = json.load(f)
    # print(len(dic))
    import pickle
    # dic = {
    #     1:2,
    #     2:3
    # }
    with open("query2pos2score.pkl",'rb') as f:
        dic1 = pickle.load(f)
    print(len(dic1))
    print(len(dic1[0]))
