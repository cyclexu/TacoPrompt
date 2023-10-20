#code from https://github.com/UKPLab/sentence-transformers
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from operator import itemgetter
from typing import Iterable, Dict, List
from model.sbert import SentenceTransformer
from model.class_level import LevelClass
from model.utils import MixedLinear, MixedDropout
import pdb

class CosineSimilarityLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, indices: List):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
        return self.loss_fct(output, labels[:,0].view(-1))


class LevelSimilarityLoss(nn.Module):
    def __init__(self, model: LevelClass):
        super(LevelSimilarityLoss, self).__init__()
        self.model = model


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, indices: List):
        pdb.set_trace()
        log_preds = [self.model(sentence_feature,indices) for sentence_feature in sentence_features]
        cross_entropy_mean_e1 = F.nll_loss(log_preds[0], labels[:,1].long())
        cross_entropy_mean_e2 = F.nll_loss(log_preds[1], labels[:,2].long())
        return cross_entropy_mean_e1+cross_entropy_mean_e2