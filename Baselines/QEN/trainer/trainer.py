import itertools
import random
import time

import wandb
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from base import BaseTrainer
from model.loss import *

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
        self.log_step = len(data_loader) // 100
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


class TrainerS(Trainer):
    """
    Trainer class, for one-to-one matching methods on taxonomy completion task

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, mode, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(TrainerS, self).__init__(model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler)
        self.mode = mode

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

        if 'g' in mode:
            self.all_nodes = sorted(list(dataset.core_subgraph.nodes))
            self.edge2subgraph = {e: dataset._get_subgraph_and_node_pair(-1, e[0], e[1]) for e in
                                  tqdm(self.candidate_positions, desc='collecting nodegraph')}

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(self.data_loader):
            nf, label, u, v, bgu, bgv, bpu, bpv, lens = batch

            self.optimizer.zero_grad()
            prediction = self.model(nf, u, v, bgu, bgv, bpu, bpv, lens)
            label = label[:, 0].to(self.device)
            if self.is_infonce_training:
                n_batches = label.sum().detach()
                prediction = prediction.reshape(n_batches, -1)
                target = torch.zeros(n_batches, dtype=torch.long).to(self.device)
                loss = self.loss(prediction, target)
            else:
                loss = self.loss(prediction, label)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader) - 1:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        loss.item()))

        log = {'loss': total_loss / len(self.data_loader)}

        ## Validation stage
        if self.do_validation:
            val, val_leaf, val_nonleaf = self._test('validation')
            val_log = {'val_metrics': self._test('validation')}
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.lr_scheduler_mode == "min":
                    self.lr_scheduler.step(log['val_metrics'][0])
                else:
                    self.lr_scheduler.step(log['val_metrics'][-1])
            else:
                self.lr_scheduler.step()

        return log

    def _test(self, mode, gpu=True):
        assert mode in ['test', 'validation']
        torch.cuda.empty_cache()
        model = self.model if gpu else self.model.cpu()

        batch_size = self.test_batch_size

        model.eval()
        with torch.no_grad():
            dataset = self.data_loader.dataset
            node_features = dataset.node_features
            if mode == 'test':
                vocab = dataset.test_node_list
                node2pos = dataset.test_node2pos
                candidate_positions = self.candidate_positions
                self.logger.info(f'number of candidate positions: {len(candidate_positions)}')
            else:
                vocab = dataset.valid_node_list
                node2pos = dataset.valid_node2pos
                candidate_positions = self.valid_candidate_positions
            batched_model = []  # save the CPU graph representation
            batched_positions = []
            for edges in tqdm(mit.sliced(candidate_positions, batch_size), desc="Generating graph encoding ..."):
                edges = list(edges)
                us, vs, bgu, bgv, bpu, bpv, lens = None, None, None, None, None, None, None
                if 'r' in self.mode:
                    us, vs = zip(*edges)
                    us = torch.tensor(us)
                    vs = torch.tensor(vs)
                if 'g' in self.mode:
                    bgs = [self.edge2subgraph[e] for e in edges]
                    bgu, bgv = zip(*bgs)
                if 'p' in self.mode:
                    bpu, bpv, lens = dataset._get_batch_edge_node_path(edges)
                    bpu = bpu
                    bpv = bpv
                    lens = lens

                ur, vr = self.model.forward_encoders(us, vs, bgu, bgv, bpu, bpv, lens)
                batched_model.append((ur.detach().cpu(), vr.detach().cpu()))
                batched_positions.append(len(edges))

            # start per query prediction
            all_ranks = []
            leaf_queries = []
            leaf_ranks, nonleaf_ranks = [], []

            # leaf and nonleaf
            for i, query in enumerate(vocab):
                if node2pos[query][0][1] == self.data_loader.dataset.pseudo_leaf_node:
                    leaf_queries.append(query)

            # for i, query in tqdm(enumerate(vocab[:200] if mode == 'validation' else vocab), desc='testing'):
            for i, query in tqdm(enumerate(vocab), desc='testing'):
                batched_energy_scores = []
                nf = node_features[query, :].to(self.device)
                for (ur, vr), n_position in zip(batched_model, batched_positions):
                    expanded_nf = nf.expand(n_position, -1)
                    ur = ur.to(self.device)
                    vr = vr.to(self.device)
                    energy_scores = model.match(ur, vr, expanded_nf)
                    batched_energy_scores.append(energy_scores)
                batched_energy_scores = torch.cat(batched_energy_scores)

                # Total
                batched_energy_scores, labels = rearrange(batched_energy_scores, candidate_positions, node2pos[query])
                ranks = self.pre_metric(batched_energy_scores, labels)
                all_ranks.extend(ranks)

                # Leaf
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


class TrainerT(TrainerS):
    """
    Trainer class, for TMN on taxonomy completion task

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, mode, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(TrainerT, self).__init__(mode, model, loss, metrics, pre_metric, optimizer, config, data_loader,
                                       lr_scheduler)
        self.train_data_loader = None
        self.l1 = config['trainer']['l1']
        self.l2 = config['trainer']['l2']
        self.l3 = config['trainer']['l3']

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(self.data_loader):
            nf, label, u, v, bgu, bgv, bpu, bpv, lens = batch

            self.optimizer.zero_grad()
            scores, scores_p, scores_c, scores_e = self.model(nf, u, v, bgu, bgv, bpu, bpv, lens)
            label = label.to(self.device)
            loss_p = self.loss(scores_p, label[:, 1])
            loss_c = self.loss(scores_c, label[:, 2])
            loss_e = self.loss(scores_e, label[:, 0])
            loss = self.loss(scores, label[:, 0]) + self.l1 * loss_p + self.l2 * loss_c + self.l3 * loss_e
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # if True:
            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader) - 1:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} ELoss: {:.6f} PLoss: {:.6f} CLoss: {:.6f}'.format(
                        epoch, batch_idx * self.data_loader.batch_size, self.data_loader.n_samples,
                               100.0 * batch_idx / len(self.data_loader),
                        loss.item(), loss_e.item(), loss_p.item(), loss_c.item()))
                wandb.log({'loss': loss.item(), 'lossP': loss_p.item(), 'lossC': loss_c.item(),
                           'step': (epoch - 1) * len(self.data_loader) + batch_idx})

            # for debugging
            # if batch_idx > 2:
            #     break

        log = {'loss': total_loss / len(self.data_loader)}

        ## Validation stage
        if self.do_validation:
            val, leaf_val, nonleaf_val = self._test('validation')
            val_log = {'val_metrics': val}
            log = {**log, **val_log}
            wandb_log = {k.__name__: v for (k, v) in zip(self.metrics, val_log['val_metrics'])}
            wandb_log.update({f'leaf_{k.__name__}': v for (k, v) in zip(self.metrics, leaf_val)})
            wandb_log.update({f'nonleaf_{k.__name__}': v for (k, v) in zip(self.metrics, nonleaf_val)})
            wandb_log.update({'epoch': epoch})
            wandb.log(wandb_log)

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.lr_scheduler_mode == "min":
                    self.lr_scheduler.step(log['val_metrics'][0])
                else:
                    self.lr_scheduler.step(log['val_metrics'][-1])
            else:
                self.lr_scheduler.step()

        return log


class TrainerTC(Trainer):
    """
    Trainer class, for our proposed model on taxonomy completion task

    Note:
        Inherited from Trainer and overrides training/evaluation methods.
    """

    def __init__(self, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(TrainerTC, self).__init__(model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler)
        self.train_data_loader = None
        self.l_p = config['trainer']['lp']
        self.l_c = config['trainer']['lc']
        self.l_b = config['trainer']['lb']
        self.l_w = config['trainer']['lw']
        self.l_main = 1
        self.anneal = config['trainer']['anneal']
        self.accumulation_iters = config['trainer']['accumulation_iters']
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
        # if epoch < 3:
        #     self.l_main = 0
        # else:
        #     self.l_main = 1
        for batch_idx, batch in enumerate(self.data_loader):
            qs, ps, cs, bs, ws, pct, pt, ct, bt, wt, pe, ce, be, we = batch
            qs = {k: v.to(self.device) for k, v in qs.items()}
            ps = {k: v.to(self.device) for k, v in ps.items()}
            cs = {k: v.to(self.device) for k, v in cs.items()}
            bs = {k: v.to(self.device) for k, v in bs.items()}
            ws = {k: v.to(self.device) for k, v in ws.items()}
            pe = pe.to(self.device)
            ce = ce.to(self.device)
            be = be.to(self.device)
            we = we.to(self.device)
            with autocast():
                score, p_parent_score, c_child_score, b_sibling_score, w_sibling_score = self.model(qs, ps, cs, bs, ws,
                                                                                                    pe, ce, be, we)
                pct = pct.to(self.device)
                pt = pt.to(self.device)
                ct = ct.to(self.device)
                bt = bt.to(self.device)
                wt = wt.to(self.device)
                loss_pc = self.loss(score.squeeze(1), pct)
                loss_p = self.loss(torch.masked_select(p_parent_score.squeeze(1), pe),
                                   torch.masked_select(pt, pe))
                loss_c = self.loss(torch.masked_select(c_child_score.squeeze(1), ce),
                                   torch.masked_select(ct, ce))
                loss_b = self.loss(torch.masked_select(b_sibling_score.squeeze(1), be),
                                   torch.masked_select(bt, be))
                loss_w = self.loss(torch.masked_select(w_sibling_score.squeeze(1), we),
                                   torch.masked_select(wt, we))

                loss = self.l_main * loss_pc + self.l_p * loss_p + self.l_c * loss_c + self.l_b * loss_b + self.l_w * loss_w

            scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_iters == 0:
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            total_loss += loss.item()

            # if True:
            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader) - 1:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} PCLoss: {:.6f} PLoss: {:.6f} CLoss: {:.6f} BLoss: {:.6f} WLoss: {:.6f}'.format(
                        epoch, batch_idx * self.data_loader.batch_size, self.data_loader.n_samples,
                               100.0 * batch_idx / len(self.data_loader), loss.item(), loss_pc.item(), loss_p.item(),
                        loss_c.item(), loss_b.item(), loss_w.item()))
                wandb.log(
                    {'loss': loss.item(), 'lossP': loss_p.item(), 'lossC': loss_c.item(), 'lossPC': loss_pc.item(),
                     'lossB': loss_b.item(), 'lossW': loss_w.item(),
                     'step': (epoch - 1) * len(self.data_loader) + batch_idx})

            # for debugging
            # if batch_idx > 1:
            #     break

        self.l_b *= self.anneal
        self.l_c *= self.anneal
        self.l_b *= self.anneal
        self.l_w *= self.anneal

        log = {'loss': total_loss / len(self.data_loader)}

        ## Validation stage
        if self.do_validation:
            # val, leaf_val, nonleaf_val = self._test('validation')
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

        # if self.lr_scheduler is not None:
        #     if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #         if self.lr_scheduler_mode == "min":
        #             self.lr_scheduler.step(log['val_metrics'][0])
        #         else:
        #             self.lr_scheduler.step(log['val_metrics'][-1])
        #     else:
        #         self.lr_scheduler.step()

        return log

    def _test(self, mode, gpu=True):
        assert mode in ['test', 'validation']
        model = self.model if gpu else self.model.cpu()

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
            self.logger.info(f'number of candidate positions: {len(candidate_positions)}')

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
            eval_queries = queries[:100] if mode == 'validation' else queries

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
                    num+=1
                # if node2pos[query][0][1] == dataset.pseudo_leaf_node:
                #     leaf_queries.append(query)
                #     num+=1
            print(num)

            # print(temp)
            # exit(0)

            for i, query in tqdm(enumerate(eval_queries), desc=mode, total=len(eval_queries)):
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
                    if debug:
                        print(scores.shape)
                        print(scores.squeeze())
                    debug = False
                    batched_energy_scores.append(scores)
                batched_energy_scores_cat = torch.cat(batched_energy_scores)
                calc_time = time.time() - s

                # Total
                batched_energy_scores_cat, labels = rearrange(batched_energy_scores_cat, candidate_positions,
                                                              node2pos[query])
                ranks = self.pre_metric(batched_energy_scores_cat, labels)
                all_ranks.extend(ranks)

                if query in leaf_queries:
                    leaf_ranks.extend(ranks)
                else:
                    nonleaf_ranks.extend(ranks)

            print(best_worst_time / calc_time)

            print(all_ranks)
            total_metrics = [metric(all_ranks) for metric in self.metrics]
            leaf_metrics = [metric(leaf_ranks) for metric in self.metrics]
            nonleaf_metrics = [metric(nonleaf_ranks) for metric in self.metrics]

            print(f'leaf_metrics:{leaf_metrics}')
            print(f'nonleaf_metrics:{nonleaf_metrics}')
        return total_metrics, leaf_metrics, nonleaf_metrics


class TrainerTExpan(Trainer):
    """
    Trainer class, for TMN on taxonomy expansion task
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, mode, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(TrainerTExpan, self).__init__(model, loss, metrics, pre_metric, optimizer, config, data_loader,
                                            lr_scheduler)
        self.mode = mode

        self.l1 = config['trainer']['l1']
        self.l2 = config['trainer']['l2']
        self.l3 = config['trainer']['l3']

        dataset = self.data_loader.dataset
        self.pseudo_leaf = dataset.pseudo_leaf_node
        self.candidate_positions = list(
            set([(p, c) for (p, c) in data_loader.dataset.all_edges if c == self.pseudo_leaf]))
        self.valid_node2pos = {node: set([(p, c) for (p, c) in pos_l if c == self.pseudo_leaf]) for node, pos_l in
                               dataset.valid_node2pos.items()}
        self.test_node2pos = {node: set([(p, c) for (p, c) in pos_l if c == self.pseudo_leaf]) for node, pos_l in
                              dataset.test_node2pos.items()}
        self.valid_vocab = [node for node, pos in self.valid_node2pos.items() if len(pos)]
        self.test_vocab = [node for node, pos in self.test_node2pos.items() if len(pos)]

        if 'g' in mode:
            self.all_nodes = sorted(list(dataset.core_subgraph.nodes))
            self.edge2subgraph = {e: dataset._get_subgraph_and_node_pair(-1, e[0], e[1]) for e in
                                  tqdm(self.candidate_positions, desc='collecting nodegraph')}

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(self.data_loader):
            nf, label, u, v, bgu, bgv, bpu, bpv, lens = batch

            self.optimizer.zero_grad()
            scores, scores_p, scores_c, scores_e = self.model(nf, u, v, bgu, bgv, bpu, bpv, lens)
            label = label.to(self.device)
            loss_p = self.loss(scores_p, label[:, 1])
            loss_c = self.loss(scores_c, label[:, 2])
            loss_e = self.loss(scores_e, label[:, 0])
            loss = self.loss(scores, label[:, 0]) + self.l1 * loss_p + self.l2 * loss_c + self.l3 * loss_e
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader) - 1:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} ELoss: {:.6f} PLoss: {:.6f} CLoss: {:.6f}'
                        .format(epoch, batch_idx * self.data_loader.batch_size, self.data_loader.n_samples,
                                100.0 * batch_idx / len(self.data_loader),
                                loss.item(), loss_e.item(), loss_p.item(), loss_c.item()))

        log = {'loss': total_loss / len(self.data_loader)}

        ## Validation stage
        if self.do_validation:
            val_log = {'val_metrics': self._test('validation')}
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.lr_scheduler_mode == "min":
                    self.lr_scheduler.step(log['val_metrics'][0])
                else:
                    self.lr_scheduler.step(log['val_metrics'][-1])
            else:
                self.lr_scheduler.step()

        return log

    def _test(self, mode, gpu=True):
        assert mode in ['test', 'validation']
        torch.cuda.empty_cache()
        model = self.model if gpu else self.model.cpu()

        batch_size = self.test_batch_size

        model.eval()
        with torch.no_grad():
            dataset = self.data_loader.dataset
            node_features = dataset.node_features
            if mode == 'test':
                vocab = self.test_vocab
                node2pos = self.test_node2pos
            else:
                vocab = self.valid_vocab
                node2pos = self.valid_node2pos
            candidate_positions = self.candidate_positions
            batched_model = []  # save the CPU graph representation
            batched_positions = []
            for edges in tqdm(mit.sliced(candidate_positions, batch_size), desc="Generating graph encoding ..."):
                edges = list(edges)
                us, vs, bgu, bgv, bpu, bpv, lens = None, None, None, None, None, None, None
                if 'r' in self.mode:
                    us, vs = zip(*edges)
                    us = torch.tensor(us)
                    vs = torch.tensor(vs)
                if 'g' in self.mode:
                    bgs = [self.edge2subgraph[e] for e in edges]
                    bgu, bgv = zip(*bgs)
                if 'p' in self.mode:
                    bpu, bpv, lens = dataset._get_batch_edge_node_path(edges)
                    bpu = bpu
                    bpv = bpv
                    lens = lens

                ur, vr = self.model.forward_encoders(us, vs, bgu, bgv, bpu, bpv, lens)
                batched_model.append((ur.detach().cpu(), vr.detach().cpu()))
                batched_positions.append(len(edges))

            # start per query prediction
            all_ranks = []
            for i, query in tqdm(enumerate(vocab), desc='testing'):
                batched_energy_scores = []
                nf = node_features[query, :].to(self.device)
                for (ur, vr), n_position in zip(batched_model, batched_positions):
                    expanded_nf = nf.expand(n_position, -1)
                    ur = ur.to(self.device)
                    vr = vr.to(self.device)
                    energy_scores = model.match(ur, vr, expanded_nf)
                    batched_energy_scores.append(energy_scores)
                batched_energy_scores = torch.cat(batched_energy_scores)
                batched_energy_scores, labels = rearrange(batched_energy_scores, candidate_positions, node2pos[query])
                all_ranks.extend(self.pre_metric(batched_energy_scores, labels))
            total_metrics = [metric(all_ranks) for metric in self.metrics]

        return total_metrics


class TrainerExpan(Trainer):
    """
    Trainer class, for one-to-one matching methods on taxonomy expansion task

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, mode, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(TrainerExpan, self).__init__(model, loss, metrics, pre_metric, optimizer, config, data_loader,
                                           lr_scheduler)
        self.mode = mode

        dataset = self.data_loader.dataset
        self.candidate_positions = dataset.all_nodes
        self.valid_node2pos = dataset.valid_node2pos
        self.test_node2pos = dataset.test_node2pos
        self.valid_vocab = dataset.valid_node_list
        self.test_vocab = dataset.test_node_list

        if 'g' in mode:
            self.all_nodes = sorted(list(dataset.core_subgraph.nodes))
            self.node2subgraph = {node: dataset._get_subgraph_and_node_pair(-1, node) for node in
                                  tqdm(self.all_nodes, desc='collecting nodegraph')}

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(self.data_loader):
            nf, label, u, graphs, paths, lens = batch

            self.optimizer.zero_grad()
            scores = self.model(nf, u, graphs, paths, lens)
            label = label.to(self.device)
            if self.is_infonce_training:
                n_batches = label.sum().detach()
                prediction = scores.reshape(n_batches, -1)
                target = torch.zeros(n_batches, dtype=torch.long).to(self.device)
                loss = self.loss(prediction, target)
            else:
                loss = self.loss(scores, label)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader) - 1:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'
                        .format(epoch, batch_idx * self.data_loader.batch_size, self.data_loader.n_samples,
                                100.0 * batch_idx / len(self.data_loader),
                                loss.item()))

        log = {'loss': total_loss / len(self.data_loader)}

        ## Validation stage
        if self.do_validation:
            val_log = {'val_metrics': self._test('validation')}
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.lr_scheduler_mode == "min":
                    self.lr_scheduler.step(log['val_metrics'][0])
                else:
                    self.lr_scheduler.step(log['val_metrics'][-1])
            else:
                self.lr_scheduler.step()

        return log

    def _test(self, mode, gpu=True):
        assert mode in ['test', 'validation']
        torch.cuda.empty_cache()
        model = self.model if gpu else self.model.cpu()

        batch_size = self.test_batch_size

        model.eval()
        with torch.no_grad():
            dataset = self.data_loader.dataset
            node_features = dataset.node_features
            if mode == 'test':
                # vocab = self.test_vocab
                # node2pos = self.test_node2pos
                node2pos = dataset.test_node2parent
                vocab = list(node2pos.keys())
            else:
                vocab = self.valid_vocab
                node2pos = self.valid_node2pos
            candidate_positions = self.candidate_positions
            batched_model = []  # save the CPU graph representation
            batched_positions = []
            for us_l in tqdm(mit.sliced(candidate_positions, batch_size), desc="Generating graph encoding ..."):

                bgu, bpu, lens = None, None, None
                if 'r' in self.mode:
                    us = torch.tensor(us_l)
                if 'g' in self.mode:
                    bgu = [self.node2subgraph[e] for e in us_l]
                if 'p' in self.mode:
                    bpu, lens = dataset._get_batch_edge_node_path(us_l)
                    bpu = bpu
                    lens = lens
                ur = self.model.forward_encoders(us, bgu, bpu, lens)
                batched_model.append(ur.detach().cpu())
                batched_positions.append(len(us))

            # start per query prediction
            all_ranks = []
            for i, query in tqdm(enumerate(vocab), desc='testing'):
                batched_energy_scores = []
                nf = node_features[query, :].to(self.device)
                for ur, n_position in zip(batched_model, batched_positions):
                    expanded_nf = nf.expand(n_position, -1)
                    ur = ur.to(self.device)
                    energy_scores = model.match(ur, expanded_nf)
                    batched_energy_scores.append(energy_scores)
                batched_energy_scores = torch.cat(batched_energy_scores)
                batched_energy_scores, labels = rearrange(batched_energy_scores, candidate_positions, node2pos[query])
                all_ranks.extend(self.pre_metric(batched_energy_scores, labels))
            total_metrics = [metric(all_ranks) for metric in self.metrics]

        return total_metrics


# Xv:
