import itertools
import random
import time

import wandb
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch

from base import BaseTrainer
from model.loss import *
from functools import partial

MAX_CANDIDATE_NUM = 100000


def rearrange(energy_scores, candidate_position_idx, true_position_idx):
    tmp = np.array([[x == y for x in candidate_position_idx] for y in true_position_idx]).any(0)
    correct = np.where(tmp)[0]
    incorrect = np.where(~tmp)[0]
    labels = torch.cat((torch.ones(len(correct)), torch.zeros(len(incorrect)))).int()
    energy_scores = torch.cat((energy_scores[correct], energy_scores[incorrect]))
    return energy_scores, labels


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.test_batch_size = config['trainer']['test_batch_size']
        self.is_infonce_training = config['loss'].startswith("info_nce")
        self.is_focal_loss = config['loss'].startswith("FocalLoss")
        self.data_loader = data_loader
        self.do_validation = True
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_mode = self.config['lr_scheduler']['args']['mode']  # "min" or "max"
        self.log_step = len(data_loader) // 10
        # self.log_step = 1
        self.pre_metric = pre_metric
        self.writer.add_text('Text', 'Model Architecture: {}'.format(self.config['arch']), 0)
        self.writer.add_text('Text', 'Training Data Loader: {}'.format(self.config['train_data_loader']), 0)
        self.writer.add_text('Text', 'Loss Function: {}'.format(self.config['loss']), 0)
        self.writer.add_text('Text', 'Optimizer: {}'.format(self.config['optimizer']), 0)

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        all_ranks = self.pre_metric(output, target)
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(all_ranks)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics


class TrainerTEMP(Trainer):
    """
    Trainer class, for our proposed model on taxonomy completion task

    Note:
        Inherited from Trainer and overrides training/evaluation methods.
    """

    def __init__(self, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(TrainerTEMP, self).__init__(model, loss, metrics, pre_metric, optimizer, config, data_loader,
                                          lr_scheduler)
        self.config = config
        self.train_data_loader = None
        self.l_p = config['trainer']['lp']
        self.l_c = config['trainer']['lc']
        self.l_b = config['trainer']['lb']
        self.l_w = config['trainer']['lw']
        self.l_angle = config['trainer']['l_angle']
        self.anneal = config['trainer']['anneal']
        self.accumulation_iters = 2
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

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        scaler = GradScaler()
        self.optimizer.zero_grad()
        cached_hgcn_emb = False
        for batch_idx, batch in enumerate(self.data_loader):
            qs, ps, cs, bs, ws, pct, pt, ct, bt, wt, pe, ce, be, we, pit, qit, cit = batch
            qs = {k: v.to(self.device) for k, v in qs.items()}
            ps = {k: v.to(self.device) for k, v in ps.items()}
            cs = {k: v.to(self.device) for k, v in cs.items()}
            bs = {k: v.to(self.device) for k, v in bs.items()}
            ws = {k: v.to(self.device) for k, v in ws.items()}
            pe = pe.to(self.device)
            ce = ce.to(self.device)
            be = be.to(self.device)
            we = we.to(self.device)
            pit = pit.to(self.device)
            qit = qit.to(self.device)
            cit = cit.to(self.device)
            with autocast():
                score = self.model(qs, ps, cs, bs, ws, pe, ce, be, we, pit, qit, cit)
                
                pct = pct.to(self.device)
                pt = pt.to(self.device)
                ct = ct.to(self.device)
                bt = bt.to(self.device)
                wt = wt.to(self.device)
                loss_label = self.loss[2](score.squeeze(1), pct)
                loss = loss_label / self.accumulation_iters

            scaler.scale(loss).backward()
            if (batch_idx + 1) % self.accumulation_iters == 0 or batch_idx == len(self.data_loader):
                total_loss += loss.item()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader) - 1:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Total: {:.6f} Loss: {:.6f}'.format(
                        epoch, batch_idx * self.data_loader.batch_size, self.data_loader.n_samples,
                               100.0 * batch_idx / len(self.data_loader), total_loss, loss.item()))
                wandb.log(
                    {'total_loss': total_loss,
                     'loss': loss.item(),
                     'step': (epoch - 1) * len(self.data_loader) + batch_idx})

        log = {'epoch avg loss': total_loss / len(self.data_loader)}

        # Validation stage
        if self.do_validation and epoch >= 40:
            if epoch == 40:
                self._save_checkpoint(epoch=epoch, save_best=False)
            val, leaf_val, nonleaf_val = self._test('validation')
            val_log = {'val_metrics': val}
            leaf_val_log = {'leaf_val_metrics': leaf_val}
            nonleaf_val_log = {'nonleaf_val_metrics': nonleaf_val}
            log = {**log, **val_log, **leaf_val_log, **nonleaf_val_log}
            wandb_log = {k.__name__: v for (k, v) in zip(self.metrics, val_log['val_metrics'])}
            wandb_log.update({f'leaf_{k.__name__}': v for (k, v) in zip(self.metrics, leaf_val)})
            wandb_log.update({f'nonleaf_{k.__name__}': v for (k, v) in zip(self.metrics, nonleaf_val)})
            wandb_log.update({'epoch': epoch})
            wandb.log(wandb_log)

        return log

    def _test(self, mode, gpu=True):
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
            pseudo_leaf_id = taxon2id[dataset.pseudo_leaf_node]
            pseudo_root_id = taxon2id[dataset.pseudo_root_node]
            taxon2allemb_id = model.train_taxon2allemb_id  
            taxon2all_node_id = model.taxon2id  
            self.logger.info(f'number of candidate positions: {len(candidate_positions)}')

            # start per query prediction
            all_ranks = []
            leaf_queries = []
            leaf_ranks, nonleaf_ranks = [], []

            # begin per query prediction
            debug = False
            eval_queries = queries[:100] if mode == 'validation' else queries
            # generate desc representations for all query nodes
            q_id = torch.tensor([taxon2all_node_id[query] for i, query in enumerate(eval_queries)])

            # find leaf and nonleaf
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
                    num += 1
            print(num)

            for i, query in tqdm(enumerate(eval_queries), desc=mode, total=len(eval_queries)):
                batched_energy_scores = []
                for edges in mit.sliced(candidate_positions, batch_size):
                    edges = list(edges)
                    ps, cs = zip(*edges)

                    with autocast():
                        p_idx = torch.tensor([taxon2allemb_id[n] for n in ps]).to(self.device)
                        c_idx = torch.tensor([taxon2allemb_id[n] for n in cs]).to(self.device)
                        q_idx = torch.tensor([q_id[i] for j in range(p_idx.shape[0])])
                        scores = model.scorer(p_idx, q_idx, c_idx, test=True).squeeze(1)
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


class TrainerHidden(Trainer):
    """
    Trainer class, for our proposed model on taxonomy completion task

    Note:
        Inherited from Trainer and overrides training/evaluation methods.
    """

    def __init__(self, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(TrainerHidden, self).__init__(model, loss, metrics, pre_metric, optimizer, config, data_loader,
                                                    lr_scheduler)
        self.config = config
        self.train_data_loader = None
        self.l_main = config['trainer']['lm']
        self.l_p = config['trainer']['lp']
        self.l_c = config['trainer']['lc']
        self.accumulation_iters = 2
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

    def _inconsistency_loss(self, pc_score, p_score, c_score, pe, ce):
        pc_prob = torch.sigmoid(pc_score)
        p_prob = torch.sigmoid(p_score)
        c_prob = torch.sigmoid(c_score)

        all = pe & ce
        p_only = pe & ~ce
        c_only = ce & ~pe
        
        pc_prob_all = torch.masked_select(pc_prob, all)
        p_prob_all = torch.masked_select(p_prob, all)
        c_prob_all = torch.masked_select(c_prob, all)
        inconsistency_loss_all = torch.abs(p_prob_all * c_prob_all - pc_prob_all) * (all.shape[0] / pc_score.shape[0])

        pc_prob_p_only = torch.masked_select(pc_prob, p_only)
        p_prob_p_only = torch.masked_select(p_prob, p_only)
        inconsistency_loss_p_only = torch.abs(p_prob_p_only - pc_prob_p_only) * (p_only.shape[0] / pc_score.shape[0])

        pc_prob_c_only = torch.masked_select(pc_prob, c_only)
        c_prob_c_only = torch.masked_select(c_prob, c_only)
        inconsistency_loss_c_only = torch.abs(c_prob_c_only - pc_prob_c_only) * (c_only.shape[0] / pc_score.shape[0])
        
        inconsistency_loss = torch.cat([inconsistency_loss_all, inconsistency_loss_p_only, inconsistency_loss_c_only])
        return torch.mean(inconsistency_loss)

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        scaler = GradScaler()
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(self.data_loader):
            qs, ps, cs, bs, ws, pct, pt, ct, bt, wt, pe, ce, be, we, pit, qit, cit = batch
            qs = {k: v.to(self.device) for k, v in qs.items()}
            ps = {k: v.to(self.device) for k, v in ps.items()}
            cs = {k: v.to(self.device) for k, v in cs.items()}
            bs = {k: v.to(self.device) for k, v in bs.items()}
            ws = {k: v.to(self.device) for k, v in ws.items()}
            pe = pe.to(self.device)
            ce = ce.to(self.device)
            be = be.to(self.device)
            we = we.to(self.device)
            pit = pit.to(self.device)
            qit = qit.to(self.device)
            cit = cit.to(self.device)
            pct = pct.to(self.device)
            pt = pt.to(self.device)
            ct = ct.to(self.device)
            with autocast():
                score = self.model(qs, ps, cs, bs, ws, pe, ce, be, we, pit, qit, cit, pct, pt, ct)
                pc_score = score[:, 0]
                p_score = score[:, 1]
                c_score = score[:, 2]
                loss_pc = self.loss(pc_score, pct)
                loss_p = self.loss(torch.masked_select(p_score, pe),
                                   torch.masked_select(pt, pe))
                loss_c = self.loss(torch.masked_select(c_score, ce),
                                  torch.masked_select(ct, ce))
                mt_loss = (self.l_main * loss_pc + self.l_p * loss_p + self.l_c * loss_c) / self.accumulation_iters
                loss = mt_loss
            scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_iters == 0 or batch_idx == len(self.data_loader):
                total_loss += loss.item()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader) - 1:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        epoch, batch_idx * self.data_loader.batch_size, self.data_loader.n_samples,
                               100.0 * batch_idx / len(self.data_loader), loss.item()
                    ))
                wandb.log(
                    {'total_loss': total_loss,
                     'loss': loss.item(),
                     'step': (epoch - 1) * len(self.data_loader) + batch_idx})
        log = {'epoch avg loss': total_loss / len(self.data_loader)}

        # Validation stage
        if epoch in [40]:
            self._save_checkpoint(epoch=epoch, save_best=False)
        if self.do_validation and epoch >= 40:
            if epoch == 40:
                self._save_checkpoint(epoch=epoch, save_best=False)
            val, leaf_val, nonleaf_val = self._test('validation')
            val_log = {'val_metrics': val}
            leaf_val_log = {'leaf_val_metrics': leaf_val}
            nonleaf_val_log = {'nonleaf_val_metrics': nonleaf_val}
            log = {**log, **val_log, **leaf_val_log, **nonleaf_val_log}
            wandb_log = {k.__name__: v for (k, v) in zip(self.metrics, val_log['val_metrics'])}
            wandb_log.update({f'leaf_{k.__name__}': v for (k, v) in zip(self.metrics, leaf_val)})
            wandb_log.update({f'nonleaf_{k.__name__}': v for (k, v) in zip(self.metrics, nonleaf_val)})
            wandb_log.update({'epoch': epoch})
            wandb.log(wandb_log)
        return log

    def _test(self, mode, gpu=True):
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
            pseudo_leaf_id = taxon2id[dataset.pseudo_leaf_node]
            pseudo_root_id = taxon2id[dataset.pseudo_root_node]
            # id for test
            taxon2allemb_id = model.train_taxon2allemb_id 
            taxon2all_node_id = model.taxon2id  
            self.logger.info(f'number of candidate positions: {len(candidate_positions)}')

            all_ranks = []
            leaf_queries = []
            leaf_ranks, nonleaf_ranks = [], []

            eval_queries = queries[:100] if mode == 'validation' else queries
            # generate desc representations for all query nodes
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
                for edges in mit.sliced(candidate_positions, batch_size):
                    edges = list(edges)
                    ps, cs = zip(*edges)

                    with autocast():
                        p_idx = torch.tensor([taxon2allemb_id[n] for n in ps]).to(self.device)
                        c_idx = torch.tensor([taxon2allemb_id[n] for n in cs]).to(self.device)
                        q_idx = torch.tensor([q_id[i] for j in range(p_idx.shape[0])])
                        pseudo_pct = torch.tensor([1 for j in range(p_idx.shape[0])])
                        scores = model.scorer(p_idx, q_idx, c_idx, pseudo_pct, pseudo_pct, pseudo_pct)
                        scores = scores[:, 0]
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
    
    
class TrainerChain(Trainer):
    """
    Trainer class, for our proposed model on taxonomy completion task

    Note:
        Inherited from Trainer and overrides training/evaluation methods.
    """

    def __init__(self, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(TrainerChain, self).__init__(model, loss, metrics, pre_metric, optimizer, config, data_loader,
                                                    lr_scheduler)
        self.config = config
        self.train_data_loader = None
        self.l_main = config['trainer']['lm']
        self.l_p = config['trainer']['lp']
        self.l_c = config['trainer']['lc']
        self.accumulation_iters = 2
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

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        scaler = GradScaler()
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(self.data_loader):
            qs, ps, cs, bs, ws, pct, pt, ct, bt, wt, pe, ce, be, we, pit, qit, cit = batch
            qs = {k: v.to(self.device) for k, v in qs.items()}
            ps = {k: v.to(self.device) for k, v in ps.items()}
            cs = {k: v.to(self.device) for k, v in cs.items()}
            bs = {k: v.to(self.device) for k, v in bs.items()}
            ws = {k: v.to(self.device) for k, v in ws.items()}
            pe = pe.to(self.device)
            ce = ce.to(self.device)
            be = be.to(self.device)
            we = we.to(self.device)
            pit = pit.to(self.device)
            qit = qit.to(self.device)
            cit = cit.to(self.device)
            pct = pct.to(self.device)
            pt = pt.to(self.device)
            ct = ct.to(self.device)
            mt_loss = 0
            with autocast():
                p_score, p_ans = self.model(qs, ps, cs, bs, ws, pe, ce, be, we, pit, qit, cit, pct, pt, ct, subtask='p')
                loss_p = self.l_p * self.loss(p_score.squeeze(1),
                                   pt) / self.accumulation_iters
                mt_loss += loss_p.item()
                scaler.scale(loss_p).backward()

                c_score, c_ans = self.model(qs, ps, cs, bs, ws, pe, ce, be, we, pit, qit, cit, pct, pt, ct, p_ans=p_ans.squeeze(1), subtask='c')
                loss_c = self.l_c * self.loss(c_score.squeeze(1),
                                   ct) / self.accumulation_iters
                mt_loss += loss_c.item()
                scaler.scale(loss_c).backward()

                pc_score, pc_ans = self.model(qs, ps, cs, bs, ws, pe, ce, be, we, pit, qit, cit, pct, pt, ct, p_ans=p_ans.squeeze(1), c_ans=c_ans.squeeze(1), subtask='a')
                loss_pc = self.l_main * self.loss(pc_score.squeeze(1), pct) / self.accumulation_iters
                mt_loss += loss_pc.item()
                scaler.scale(loss_pc).backward()
                
            if (batch_idx + 1) % self.accumulation_iters == 0 or batch_idx == len(self.data_loader):
                total_loss += mt_loss
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader) - 1:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        epoch, batch_idx * self.data_loader.batch_size, self.data_loader.n_samples,
                               100.0 * batch_idx / len(self.data_loader),  mt_loss
                    ))
                wandb.log(
                    {'total_loss': total_loss,
                     'loss': mt_loss,
                     'step': (epoch - 1) * len(self.data_loader) + batch_idx})
        log = {'epoch avg loss': total_loss / len(self.data_loader)}

        # Validation stage
        if self.do_validation and epoch >= 40:
            if epoch == 40:
                self._save_checkpoint(epoch=epoch, save_best=False)
            val, leaf_val, nonleaf_val = self._test('validation')
            val_log = {'val_metrics': val}
            leaf_val_log = {'leaf_val_metrics': leaf_val}
            nonleaf_val_log = {'nonleaf_val_metrics': nonleaf_val}
            log = {**log, **val_log, **leaf_val_log, **nonleaf_val_log}
            wandb_log = {k.__name__: v for (k, v) in zip(self.metrics, val_log['val_metrics'])}
            wandb_log.update({f'leaf_{k.__name__}': v for (k, v) in zip(self.metrics, leaf_val)})
            wandb_log.update({f'nonleaf_{k.__name__}': v for (k, v) in zip(self.metrics, nonleaf_val)})
            wandb_log.update({'epoch': epoch})
            wandb.log(wandb_log)
        return log

    def _test(self, mode, q_val=100, gpu=True):
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
            
            pseudo_leaf_id = taxon2allemb_id[dataset.pseudo_leaf_node]
            pseudo_root_id = taxon2allemb_id[dataset.pseudo_root_node]
            self.logger.info(f'number of candidate positions: {len(candidate_positions)}')

            
            all_ranks = []
            leaf_queries = []
            leaf_ranks, nonleaf_ranks = [], []

            eval_queries = queries[:q_val] if mode == 'validation' else queries
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