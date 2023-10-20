#code from https://github.com/UKPLab/sentence-transformers
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch import nn, Tensor
from operator import itemgetter
from typing import Iterable, Dict, List
from model.sbert import SentenceTransformer
from model.utils import MixedLinear, MixedDropout
from model.sbert.util import fullname
import pdb
import os
import json


class LevelClass(nn.Module):
    def __init__(self, input_size: int, nclasses: int, hiddenunits: List[int], drop_prob: float, propagation, bias=False):
        super(LevelClass, self).__init__()

        self.input_size = input_size
        self.nclasses = nclasses
        self.hiddenunits =hiddenunits
        self.drop_prob =drop_prob
        self.bias = bias
        fcs = [MixedLinear(input_size, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            fcs.append(nn.Linear(hiddenunits[i - 1], hiddenunits[i], bias=bias))
        fcs.append(nn.Linear(hiddenunits[-1], nclasses, bias=bias))
        self.fcs = nn.ModuleList(fcs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._target_device = torch.device(device)
        self.reg_params = list(self.fcs[0].parameters())

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.act_fn = nn.ReLU()
        self.propagation = propagation

    def _transform_features(self, attr_matrix: torch.sparse.FloatTensor):
        layer_inner = self.act_fn(self.fcs[0](self.dropout(attr_matrix)))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = self.fcs[-1](self.dropout(layer_inner))
        return res

    def forward(self, features: Dict[str, Tensor]):
        #first model
        features.update({'sentence_embedding': self._transform_features(features['sentence_embedding'])})
        features.update({'sentence_embedding': self.propagation(features['sentence_embedding'])})
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.embeddings_dimension

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'level_class_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    def get_config_dict(self):
        return {'input_size': self.input_size, 'nclasses': self.nclasses, 'hiddenunits':self.hiddenunits, 'bias': self.bias, 'drop_prob': self.drop_prob}

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'level_class_config.json'), 'r') as fIn:
            config = json.load(fIn)

        weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        model = LevelClass(**config)
        model.load_state_dict(weights)
        return model