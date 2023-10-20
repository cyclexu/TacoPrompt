import pdb
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Iterable, Dict, List
from operator import itemgetter
from model.utils import MixedLinear, MixedDropout
from model.sbert import SentenceTransformer
import pdb
EPSILON = 1e10

#taken from https://github.com/UKPLab/sentence-transformers/blob/46a149433fe9af0851f7fa6f9bf37b5ffa2c891c/sentence_transformers/losses/CosineSimilarityLoss.py
#changed the transformer model to nn.Sequential model
class HierarchySimilarityLoss(nn.Module):
    def __init__(self, loss_strategy:int, input_size: int, nclasses: int, hiddenunits: List[int], drop_prob: float, propagation: nn.Module, model: SentenceTransformer, nodes_list: List, trainInputLevel: Dict, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity(), bias=False):
        super(HierarchySimilarityLoss, self).__init__()
        self.loss_strategy = loss_strategy
        self.model = model
        self.loss_fct = loss_fct
        self.nclasses = nclasses
        self.cos_score_transformation = cos_score_transformation
        self.nodes_list = nodes_list
        self.sorter = self.nodes_list.argsort()
        self.trainInputLevel = trainInputLevel
        fcs = [MixedLinear(input_size, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            fcs.append(nn.Linear(hiddenunits[i - 1], hiddenunits[i], bias=bias))
        fcs.append(nn.Linear(hiddenunits[-1], nclasses, bias=bias))
        self.fcs = nn.ModuleList(fcs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._target_device = torch.device(device)
        self.reg_params = list(self.fcs[0].parameters())

        if drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)
        self.act_fn = nn.ReLU()
        self.propagation = propagation

    def _transform_features(self, attr_matrix: torch.sparse.FloatTensor):
        layer_inner = self.act_fn(self.fcs[0](self.dropout(attr_matrix)))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = self.fcs[-1](self.dropout(layer_inner))
        return res

    def get_node_indices(self,indices):
        queries_id = []
        cand_id = []
        for idx in indices:
            query_idx = int(idx.split('_')[0])
            can_idx = int(idx.split('_')[1])
            queries_id.append(query_idx)
            cand_id.append(can_idx)
        query_idx_adj = self.sorter[np.searchsorted(self.nodes_list, queries_id, sorter=self.sorter)]
        cand_idx_adj = self.sorter[np.searchsorted(self.nodes_list, cand_id, sorter=self.sorter)]
        return query_idx_adj,cand_idx_adj

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, indices: List):
        loss = 0
        # m = nn.Sigmoid()
        if self.loss_strategy == 0:
            embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
            output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
            loss = self.loss_fct(output, labels[:,0].view(-1))
            pdb.set_trace()
        elif self.loss_strategy == 1:
            embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
            query_idx_adj, cand_idx_adj = self.get_node_indices(indices)
            # e1_l = self.propagation(embeddings[0], query_idx_adj, cand_idx_adj)
            # e2_l = self.propagation(embeddings[1], cand_idx_adj, query_idx_adj)
            output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
            # pdb.set_trace()
            output_2 = self.propagation(output, query_idx_adj, cand_idx_adj)
            # pdb.set_trace()
            loss = self.loss_fct(output_2, labels[:,0].view(-1))
        return loss
    # def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, indices: List):
    #     embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
    #     output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
    #     loss_sim = self.loss_fct(output, labels[:, 0].view(-1))
    #     loss_1 = 0
    #     loss_2 = 0
    #     if self.loss_strategy>0:
    #         root = ' '
    #         query_idx_adj, cand_idx_adj = self.get_node_indices(indices)
    #         embedding_root = Tensor(self.model.encode(root)).repeat(embeddings[0].shape[0], 1).to(self._target_device)
    #         e1_l = self.propagation(embeddings[0],query_idx_adj, cand_idx_adj)
    #         output_1 = self.cos_score_transformation(torch.cosine_similarity(embedding_root, e1_l))
    #         e2_l = self.propagation(embeddings[1], cand_idx_adj, query_idx_adj)
    #         output_2 = self.cos_score_transformation(torch.cosine_similarity(embedding_root, e2_l))
    #         loss_1 = self.loss_fct(output_1, 1/(labels[:, 1].view(-1)+1))
    #         loss_2 = self.loss_fct(output_2, 1/(labels[:, 2].view(-1)+1))
    #     loss = loss_sim+loss_1+loss_2
    #     return loss,loss_sim,loss_1,loss_2