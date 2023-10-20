import time

import torch
import torch.nn as nn

from transformers import AutoModel, DistilBertModel, BertForMaskedLM, DistilBertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM, ElectraForMaskedLM, AutoTokenizer
from base import BaseModel
from .model_zoo import *
import scipy.sparse as sp
from gensim.models import KeyedVectors
import numpy as np


class AbstractPathModel(nn.Module):
    def __init__(self):
        super(AbstractPathModel, self).__init__()

    def init(self, **options):
        self.hidden_size = options['out_dim']
        in_dim = options['in_dim']
        out_dim = options['out_dim']
        self.p_lstm = nn.LSTM(input_size=in_dim, hidden_size=self.hidden_size, batch_first=True)
        self.c_lstm = nn.LSTM(input_size=in_dim, hidden_size=self.hidden_size, batch_first=True)
        self.p_control = nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.ReLU())
        self.c_control = nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.ReLU())

    def init_hidden(self, batch_size, device):
        hidden = (torch.randn(1, batch_size, self.hidden_size).to(device),
                  torch.randn(1, batch_size, self.hidden_size).to(device))
        return hidden

    def encode_parent_path(self, p, lens):
        batch_size, seq_len = p.size()
        hidden = self.init_hidden(batch_size, self.device)
        p = self.embedding(p)
        c = self.p_control(p[:, 0, :]).view(batch_size, 1, -1)
        X = torch.nn.utils.rnn.pack_padded_sequence(p, lens, batch_first=True, enforce_sorted=False)
        X, hidden = self.p_lstm(X, hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = (c * X).max(dim=1)[0]
        return X

    def encode_child_path(self, p, lens):
        batch_size, seq_len = p.size()
        hidden = self.init_hidden(batch_size, self.device)
        p = self.embedding(p)
        c = self.c_control(p[:, 0, :]).view(batch_size, 1, -1)
        X = torch.nn.utils.rnn.pack_padded_sequence(p, lens, batch_first=True, enforce_sorted=False)
        X, hidden = self.c_lstm(X, hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = (c * X).max(dim=1)[0]
        return X

    def forward_path_encoders(self, pu, pv, lens):
        pu = pu.to(self.device)
        pv = pv.to(self.device)
        lens = lens.to(self.device)
        hpu = self.encode_parent_path(pu, lens[:, 0])
        hpv = self.encode_child_path(pv, lens[:, 1])
        return hpu, hpv


class AbstractGraphModel(nn.Module):
    def __init__(self):
        super(AbstractGraphModel, self).__init__()

    def init(self, **options):
        propagation_method = options['propagation_method']
        readout_method = options['readout_method']
        options = options
        if propagation_method == "GCN":
            self.parent_graph_propagate = GCN(
                options["in_dim"], options["hidden_dim"], options["out_dim"], num_layers=options["num_layers"],
                activation=F.leaky_relu, in_dropout=options["feat_drop"], hidden_dropout=options["hidden_drop"],
                output_dropout=options["out_drop"])
            self.child_graph_propagate = GCN(
                options["in_dim"], options["hidden_dim"], options["out_dim"], num_layers=options["num_layers"],
                activation=F.leaky_relu, in_dropout=options["feat_drop"], hidden_dropout=options["hidden_drop"],
                output_dropout=options["out_drop"])
        elif propagation_method == "PGCN":
            self.parent_graph_propagate = PGCN(
                options["in_dim"], options["hidden_dim"], options["out_dim"], options["pos_dim"],
                num_layers=options["num_layers"], activation=F.leaky_relu, in_dropout=options["feat_drop"],
                hidden_dropout=options["hidden_drop"], output_dropout=options["out_drop"])
            self.child_graph_propagate = PGCN(
                options["in_dim"], options["hidden_dim"], options["out_dim"], options["pos_dim"],
                num_layers=options["num_layers"], activation=F.leaky_relu, in_dropout=options["feat_drop"],
                hidden_dropout=options["hidden_drop"], output_dropout=options["out_drop"])
        elif propagation_method == "GAT":
            self.parent_graph_propagate = GAT(
                options["in_dim"], options["hidden_dim"], options["out_dim"], num_layers=options["num_layers"],
                heads=options["heads"], activation=F.leaky_relu, feat_drop=options["feat_drop"],
                attn_drop=options["attn_drop"])
            self.child_graph_propagate = GAT(
                options["in_dim"], options["hidden_dim"], options["out_dim"], num_layers=options["num_layers"],
                heads=options["heads"], activation=F.leaky_relu, feat_drop=options["feat_drop"],
                attn_drop=options["attn_drop"])
        elif propagation_method == "PGAT":
            self.parent_graph_propagate = PGAT(
                options["in_dim"], options["hidden_dim"], options["out_dim"], options["pos_dim"],
                num_layers=options["num_layers"], heads=options["heads"], activation=F.leaky_relu,
                feat_drop=options["feat_drop"], attn_drop=options["attn_drop"])
            self.child_graph_propagate = PGAT(
                options["in_dim"], options["hidden_dim"], options["out_dim"], options["pos_dim"],
                num_layers=options["num_layers"], heads=options["heads"], activation=F.leaky_relu,
                feat_drop=options["feat_drop"], attn_drop=options["attn_drop"])
        else:
            assert f"Unacceptable Graph Propagation Method: {self.propagation_method}"

        if readout_method == "MR":
            self.p_readout = MeanReadout()
            self.c_readout = MeanReadout()
        elif readout_method == "WMR":
            self.p_readout = WeightedMeanReadout()
            self.c_readout = WeightedMeanReadout()
        else:
            assert f"Unacceptable Readout Method: {self.readout_method}"

    def encode_parent_graph(self, g):
        h = self.embedding(g.ndata['_id'].to(self.device))
        pos = g.ndata['pos'].to(self.device)
        g.ndata['h'] = self.parent_graph_propagate(g, h)
        h = self.p_readout(g, pos)
        return h

    def encode_child_graph(self, g):
        h = self.embedding(g.ndata['_id'].to(self.device))
        pos = g.ndata['pos'].to(self.device)
        g.ndata['h'] = self.child_graph_propagate(g, h)
        h = self.c_readout(g, pos)
        return h

    def forward_graph_encoders(self, gu, gv):
        hgu = self.encode_parent_graph(gu)
        hgv = self.encode_child_graph(gv)
        return hgu, hgv


class MatchModel(BaseModel, AbstractPathModel, AbstractGraphModel):
    def __init__(self, mode, **options):
        super(MatchModel, self).__init__()
        self.options = options
        self.mode = mode

        l_dim = 0
        if 'r' in self.mode:
            l_dim += options["in_dim"]
        if 'g' in self.mode:
            l_dim += options["out_dim"]
            AbstractGraphModel.init(self, **options)
        if 'p' in self.mode:
            l_dim += options["out_dim"]
            AbstractPathModel.init(self, **options)
        self.l_dim = l_dim
        self.r_dim = options["in_dim"]

        if options['matching_method'] == "MLP":
            self.match = MLP(self.l_dim, self.r_dim, 100, options["k"])
        elif options['matching_method'] == "SLP":
            self.match = SLP(self.l_dim, self.r_dim, 100)
        elif options['matching_method'] == "DST":
            self.match = DST(self.l_dim, self.r_dim)
        elif options['matching_method'] == "LBM":
            self.match = LBM(self.l_dim, self.r_dim)
        elif options['matching_method'] == "BIM":
            self.match = BIM(self.l_dim, self.r_dim)
        elif options['matching_method'] == "Arborist":
            self.match = Arborist(self.l_dim, self.r_dim, options["k"])
        elif options['matching_method'] == "NTN":
            self.match = NTN(self.l_dim, self.r_dim, options["k"])
        elif options['matching_method'] == "CNTN":
            self.match = CNTN(self.l_dim, self.r_dim, options["k"])
        elif options['matching_method'] == "TMN":
            self.match = TMN(self.l_dim, self.r_dim, options["k"])
        else:
            assert f"Unacceptable Matching Method: {options['matching_method']}"

    def forward_encoders(self, u=None, v=None, gu=None, gv=None, pu=None, pv=None, lens=None):
        ur, vr = [], []
        if 'r' in self.mode:
            hu = self.embedding(u.to(self.device))
            hv = self.embedding(v.to(self.device))
            ur.append(hu)
            vr.append(hv)
        if 'g' in self.mode:
            gu = dgl.batch(gu)
            gv = dgl.batch(gv)
            hgu, hgv = self.forward_graph_encoders(gu, gv)
            ur.append(hgu)
            vr.append(hgv)
        if 'p' in self.mode:
            hpu, hpv = self.forward_path_encoders(pu, pv, lens)
            ur.append(hpu)
            vr.append(hpv)
        ur = torch.cat(ur, -1)
        vr = torch.cat(vr, -1)
        return ur, vr

    def forward(self, q, *inputs):
        qf = self.embedding(q.to(self.device))
        ur, vr = self.forward_encoders(*inputs)
        scores = self.match(ur, vr, qf)
        return scores


class ExpanMatchModel(BaseModel, AbstractPathModel, AbstractGraphModel):
    def __init__(self, mode, **options):
        super(ExpanMatchModel, self).__init__()
        self.options = options
        self.mode = mode

        l_dim = 0
        if 'r' in self.mode:
            l_dim += options["in_dim"]
        if 'g' in self.mode:
            l_dim += options["out_dim"]
            AbstractGraphModel.init(self, **options)
        if 'p' in self.mode:
            l_dim += options["out_dim"]
            AbstractPathModel.init(self, **options)
        self.l_dim = l_dim
        self.r_dim = options["in_dim"]

        if options['matching_method'] == "NTN":
            self.match = RawNTN(self.l_dim, self.r_dim, options["k"])
        if options['matching_method'] == "BIM":
            self.match = RawBIM(self.l_dim, self.r_dim)
        if options['matching_method'] == "MLP":
            self.match = RawMLP(self.l_dim, self.r_dim, 100, options["k"])
        elif options['matching_method'] == "ARB":
            self.match = RawArborist(self.l_dim, self.r_dim, options["k"])
        else:
            assert f"Unacceptable Matching Method: {options['matching_method']}"

    def forward_encoders(self, u=None, gu=None, pu=None, lens=None):
        ur = []
        if 'r' in self.mode:
            hu = self.embedding(u.to(self.device))
            ur.append(hu)
        if 'g' in self.mode:
            gu = dgl.batch(gu)
            hgu = self.encode_parent_graph(gu)
            ur.append(hgu)
        if 'p' in self.mode:
            pu = pu.to(self.device)
            lens = lens.to(self.device)
            hpu = self.encode_parent_path(pu, lens)
            ur.append(hpu)
        ur = torch.cat(ur, -1)
        return ur

    def forward(self, q, us, graphs, paths, lens):
        qf = self.embedding(q.to(self.device))
        ur = self.forward_encoders(us, graphs, paths, lens)
        scores = self.match(ur, qf)
        return scores


class QENModel(BaseModel):
    def __init__(self, **options):
        super(BaseModel, self).__init__()
        self.options = options
        self.dropout = nn.Dropout(p=options['dropout'])
        self.poly_encoder = PolyEncoder(**options)
        self.poly_scorer = PolyScorer(**options)
        self.parent_scorer = nn.Sequential(
            nn.Linear(options['code_dim'], options['scorer_hidden_dim']),
            nn.ReLU(),
            self.dropout,
            nn.Linear(options['scorer_hidden_dim'], 1)
        )
        self.sibling_scorer = nn.Sequential(
            nn.Linear(options['code_dim'], options['scorer_hidden_dim']),
            nn.ReLU(),
            self.dropout,
            nn.Linear(options['scorer_hidden_dim'], 1)
        )
        # self.parent_scorer = nn.Linear(options['code_dim'], 1)
        # self.sibling_scorer = nn.Linear(options['code_dim'], 1)

    def forward(self, qs, ps, cs, bs, ws, pe, ce, be, we, debug=False):
        q_code, p_code, c_code, b_code, w_code = map(self.poly_encoder, [qs, ps, cs, bs, ws])
        # p_code, c_code, b_code, w_code = map(self.poly_encoder, [ps, cs, bs, ws])
        score, pq_parental, qc_parental, qb_sibling, qw_sibling = self.poly_scorer(q_code, p_code, c_code, b_code,
                                                                                   w_code, pe, ce, be, we)

        # auxiliary scores
        p_parent_score = self.parent_scorer(pq_parental)
        c_child_score = self.parent_scorer(qc_parental)
        b_sibling_score = self.sibling_scorer(qb_sibling)
        w_sibling_score = self.sibling_scorer(qw_sibling)

        return score, p_parent_score, c_child_score, b_sibling_score, w_sibling_score


class TEMPCompletion(BaseModel):
    def __init__(self, raw_dataset, **options):
        super(BaseModel, self).__init__()
        self.raw_dataset = raw_dataset
        self.n_anchors = len(self.raw_dataset.train_allemb_id2taxon)
        self.taxon2id = self.raw_dataset.taxon2id
        self.id2taxon = self.raw_dataset.id2taxon
        self.train_allemb_id2taxon = self.raw_dataset.train_allemb_id2taxon
        self.train_taxon2allemb_id = self.raw_dataset.train_taxon2allemb_id
        self.pseudo_leaf_allemb_id = self.taxon2id[raw_dataset.pseudo_leaf_node]
        self.tokenizer = AutoTokenizer.from_pretrained(options['plm'])
        self.plm = AutoModel.from_pretrained(options['plm'])
        self.mlp = nn.Sequential(nn.Linear(768, 256), nn.Dropout(0.2), nn.Linear(256, 1))

        self.options = options
        self.desc = [node.description for node_id, node in self.id2taxon.items()]
        self.query_name = [node.norm_name for node_id, node in self.id2taxon.items()]
        self.parent2path = self.raw_dataset.parent2paths

        self._init_idx_map()

    def _init_idx_map(self):
        self.all_embed_id2all_node_id = {}
        for id, taxon in self.train_allemb_id2taxon.items():
            self.all_embed_id2all_node_id[id] = self.taxon2id[taxon]
        self.all_node_id_2all_embed_id = {v: k for k, v in self.all_embed_id2all_node_id.items()}

    def convert_id(self, id):
        return self.all_embed_id2all_node_id[id]

    def lm_forward(self, text):
        repr = self.plm(**text)
        repr = repr['last_hidden_state'][:, 0, :].squeeze(1)
        return repr

    def get_text_emb(self, pit, qid, cit, batch=32, test=False):
        if test:
            batch = 4096
        text_prompt = "{} [SEP] {}"
        text = [text_prompt.format(self.parent2path[self.train_allemb_id2taxon[pit[i].item()]] + ',{},{}'.format(
            self.query_name[qid[i]],self.train_allemb_id2taxon[cit[i].item()].norm_name), 
            self.desc[qid[i]]) for i in range(pit.shape[0])]

        text_emb = torch.zeros(pit.shape[0], 768)
        total_times = math.ceil(pit.shape[0] / batch)
        for i in range(total_times - 1):
            start = i * batch
            text_ = [text[start + j] for j in range(batch)]
            text_pt = self.tokenizer(text_, return_tensors='pt', padding=True, truncation=True, max_length=256)
            text_pt = {k: v.to("cuda") for k, v in text_pt.items()}
            text_emb[start: start + batch] = self.lm_forward(text_pt)
        start = (total_times - 1) * batch
        end_step = pit.shape[0] - start
        text_ = [text[start + j] for j in range(end_step)]
        text_pt = self.tokenizer(text_, return_tensors='pt', padding=True, truncation=True, max_length=256)
        text_pt = {k: v.to("cuda") for k, v in text_pt.items()}
        text_emb[start: start + end_step] = self.lm_forward(text_pt)
        return text_emb.to('cuda')

    def scorer(self, pit, qid, cit, test=False):
        text_emb = self.get_text_emb(pit, qid, cit, test=test)
        score = self.mlp(text_emb)
        return score

    def forward(self, qs, ps, cs, bs, ws, pe, ce, be, we, pit, qit, cit):
        qid = [self.convert_id(qit[i].item()) for i in range(qit.shape[0])]
        return self.scorer(pit, qid, cit)


class TacoPromptHidden(BaseModel):
    def __init__(self, raw_dataset, **options):
        super(BaseModel, self).__init__()
        self.raw_dataset = raw_dataset
        self.n_anchors = len(self.raw_dataset.train_allemb_id2taxon)
        self.taxon2id = self.raw_dataset.taxon2id
        self.id2taxon = self.raw_dataset.id2taxon
        self.train_allemb_id2taxon = self.raw_dataset.train_allemb_id2taxon
        self.train_taxon2allemb_id = self.raw_dataset.train_taxon2allemb_id
        self.pseudo_leaf_allemb_id = self.taxon2id[raw_dataset.pseudo_leaf_node]

        self.model = BertForMaskedLM.from_pretrained(options['plm'])
        self.tokenizer = AutoTokenizer.from_pretrained(options['plm'])
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('yes'))[0]
        self.no_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('no'))[0]
        self.mask_token_id = self.tokenizer.mask_token_id
        self.max_seq_length = 256

        self.options = options
        self.desc = [node.description for node_id, node in self.id2taxon.items()]
        self.device = "cuda"

        self._init_idx_map()

    def _init_idx_map(self):
        # all_emb_id <-> all_node_id
        self.all_embed_id2all_node_id = {}
        for id, taxon in self.train_allemb_id2taxon.items():
            self.all_embed_id2all_node_id[id] = self.taxon2id[taxon]
        self.all_node_id_2all_embed_id = {v: k for k, v in self.all_embed_id2all_node_id.items()}

    def convert_id(self, id):
        return self.all_embed_id2all_node_id[id]

    def get_lm_inputs(self, pit, qid, cit, label, pl, cl, tc=False):
        max_seq_length = self.max_seq_length
        tokenizer = self.tokenizer
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        pad_token_id = tokenizer.pad_token_id
        mask_token = tokenizer.mask_token
        mask_token_id = tokenizer.mask_token_id

        sentences_prompt = []
        sentences_ground_truth = []

        for i in range(label.shape[0]):
            a_prompt = "All: {} Parent: {} Chlid: {}. <Parent>: {} <Child>: {}".format(
                mask_token, mask_token, mask_token, self.desc[self.convert_id(pit[i].item())], self.desc[self.convert_id(cit[i].item())])
            b_prompt = "<Query>: {}".format(self.desc[qid[i]])
            sentence_prompt = a_prompt + sep_token + b_prompt
            sentences_prompt.append(sentence_prompt)

            label_token = 'yes' if label[i].item() else 'no'
            pl_token = 'yes' if pl[i].item() else 'no'
            cl_token = 'yes' if cl[i].item() else 'no'

            a_prompt = "All: {} Parent: {} Child: {}. <Parent>: {} <Child>: {}".format(
                label_token, pl_token, cl_token, self.desc[self.convert_id(pit[i].item())], self.desc[self.convert_id(cit[i].item())])
            sentence_prompt_gt = a_prompt + sep_token + b_prompt
            sentences_ground_truth.append(sentence_prompt_gt)

        sentences_prompt_pt = tokenizer(sentences_prompt, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
        sentences_prompt_gt_pt = tokenizer(sentences_ground_truth, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = sentences_prompt_pt.input_ids
        attention_mask = sentences_prompt_pt.attention_mask
        labels = sentences_prompt_gt_pt.input_ids
        label_ids = torch.where(input_ids == mask_token_id, labels, -100)

        return input_ids, label_ids, attention_mask

    def scorer(self, pit, qid, cit, pct, pl, cl):
        batch_input_ids, batch_labels, batch_attention_mask = self.get_lm_inputs(pit, qid, cit, pct, pl, cl)
        batch_lm_inputs = {'input_ids': batch_input_ids,
                           'attention_mask': batch_attention_mask}
        batch_lm_inputs = {k: v.to(self.device) for k, v in batch_lm_inputs.items()}
        outputs = self.model(**batch_lm_inputs)
        score = outputs[0][batch_lm_inputs['input_ids'] == self.mask_token_id]
        pred_scores = score[:, self.yes_token_id] - score[:, self.no_token_id]
        pred_scores = pred_scores.view(pit.shape[0], -1)
        return pred_scores

    def forward(self, qs, ps, cs, bs, ws, pe, ce, be, we, pit, qit, cit, pct, pt, ct):
        qid = [self.convert_id(qit[i].item()) for i in range(qit.shape[0])]
        return self.scorer(pit, qid, cit, pct, pt, ct)
    

class TacoPromptChain(BaseModel):
    def __init__(self, raw_dataset, **options):
        super(BaseModel, self).__init__()
        self.raw_dataset = raw_dataset
        self.n_anchors = len(self.raw_dataset.train_allemb_id2taxon)
        self.taxon2id = self.raw_dataset.taxon2id
        self.id2taxon = self.raw_dataset.id2taxon
        self.train_allemb_id2taxon = self.raw_dataset.train_allemb_id2taxon
        self.train_taxon2allemb_id = self.raw_dataset.train_taxon2allemb_id
        self.pseudo_leaf_allemb_id = self.taxon2id[raw_dataset.pseudo_leaf_node]

        self.model = BertForMaskedLM.from_pretrained(options['plm'])
        self.tokenizer = AutoTokenizer.from_pretrained(options['plm'])
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('yes'))[0]
        self.no_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('no'))[0]
        self.mask_token_id = self.tokenizer.mask_token_id
        self.max_seq_length = 256

        self.options = options
        self.desc = [node.description for node_id, node in self.id2taxon.items()]
        self.device = "cuda"

        self._init_idx_map()

    def _init_idx_map(self):
        # all_emb_id <-> all_node_id
        self.all_embed_id2all_node_id = {}
        for id, taxon in self.train_allemb_id2taxon.items():
            self.all_embed_id2all_node_id[id] = self.taxon2id[taxon]
        self.all_node_id_2all_embed_id = {v: k for k, v in self.all_embed_id2all_node_id.items()}

    def convert_id(self, id):
        return self.all_embed_id2all_node_id[id]

    def get_lm_inputs(self, pit, qid, cit, label, pl, cl, pe, ce, p_ans=None, c_ans=None, subtask='a', tc=False):
        max_seq_length = self.max_seq_length
        tokenizer = self.tokenizer
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        pad_token_id = tokenizer.pad_token_id
        mask_token = tokenizer.mask_token
        mask_token_id = tokenizer.mask_token_id

        sentences_prompt = []
        sentences_ground_truth = []

        for i in range(label.shape[0]):
            if subtask == 'a':
                p_answer = 'yes' if p_ans[i] else 'no'
                c_answer = 'yes' if c_ans[i] else 'no'

                a_prompt = "All: {} Parent: {}  Child: {}. <Parent>: {} <Child>: {}".format(
                    mask_token, p_answer, c_answer, self.desc[self.convert_id(pit[i].item())], self.desc[self.convert_id(cit[i].item())])
            
            elif subtask == 'p':
                a_prompt = "Parent: {}. <Parent>: {}".format(
                    mask_token, self.desc[self.convert_id(pit[i].item())])
                
            elif subtask == 'c':
                p_answer = 'yes' if p_ans[i] else 'no'
                a_prompt = "Child: {} Parent: {}. <Child>: {}".format(
                    mask_token, p_answer, self.desc[self.convert_id(cit[i].item())])
            b_prompt = "<Query>: {}".format(self.desc[qid[i]])
            sentence_prompt = a_prompt + sep_token + b_prompt
            sentences_prompt.append(sentence_prompt)


            label_token = 'yes' if label[i].item() else 'no'
            pl_token = 'yes' if pl[i].item() else 'no'
            cl_token = 'yes' if cl[i].item() else 'no'
            
            if subtask == 'a':
                p_answer = 'yes' if p_ans[i] else 'no'
                c_answer = 'yes' if c_ans[i] else 'no'

                a_prompt = "All: {} Parent: {}  Child: {}. <Parent>: {} <Child>: {}".format(
                    label_token, p_answer, c_answer, self.desc[self.convert_id(pit[i].item())], self.desc[self.convert_id(cit[i].item())])
                
            elif subtask == 'p':
                a_prompt = "Parent: {}. <Parent>: {}".format(
                    pl_token, self.desc[self.convert_id(pit[i].item())])
                
            elif subtask == 'c':
                p_answer = 'yes' if p_ans[i] else 'no'
                a_prompt = "Child: {} Parent: {}. <Child>: {}".format(
                    cl_token, p_answer, self.desc[self.convert_id(cit[i].item())])
            sentence_prompt_gt = a_prompt + sep_token + b_prompt

            sentences_ground_truth.append(sentence_prompt_gt)

        sentences_prompt_pt = tokenizer(sentences_prompt, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
        sentences_prompt_gt_pt = tokenizer(sentences_ground_truth, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = sentences_prompt_pt.input_ids
        attention_mask = sentences_prompt_pt.attention_mask
        labels = sentences_prompt_gt_pt.input_ids
        label_ids = torch.where(input_ids == mask_token_id, labels, -100)

        return input_ids, label_ids, attention_mask

    def scorer(self, pit, qid, cit, pct, pl, cl, pe, ce, p_ans=None, c_ans=None, subtask='a'):
        batch_input_ids, batch_labels, batch_attention_mask = self.get_lm_inputs(pit, qid, cit, pct, pl, cl, pe, ce, p_ans=p_ans, c_ans=c_ans, subtask=subtask)
        # batch_lm_inputs = {'input_ids': batch_input_ids,
        #                    'attention_mask': batch_attention_mask,
        #                    'labels': batch_labels}
        batch_lm_inputs = {'input_ids': batch_input_ids,
                           'attention_mask': batch_attention_mask}
        batch_lm_inputs = {k: v.to(self.device) for k, v in batch_lm_inputs.items()}
        outputs = self.model(**batch_lm_inputs)
        score = outputs[0][batch_lm_inputs['input_ids'] == self.mask_token_id]
        pred_scores = score[:, self.yes_token_id] - score[:, self.no_token_id]
        pred_scores = pred_scores.view(pit.shape[0], -1)
        
        # results
        ans = score[:, self.yes_token_id] > score[:, self.no_token_id]
        ans = ans.view(pit.shape[0], -1)
        return pred_scores, ans

    def forward(self, qs, ps, cs, bs, ws, pe, ce, be, we, pit, qit, cit, pct, pt, ct, p_ans=None, c_ans=None, subtask='a'):
        qid = [self.convert_id(qit[i].item()) for i in range(qit.shape[0])]
        return self.scorer(pit, qid, cit, pct, pt, ct, pe, ce, p_ans=p_ans, c_ans=c_ans, subtask=subtask)