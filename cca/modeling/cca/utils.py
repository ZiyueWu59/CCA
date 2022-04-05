import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from cca.modeling.cca.con_graph_modules import l2norm

class Permute(nn.Module):
    
    def __init__(self, *idx):
        super(Permute, self).__init__()
        self.idx_num = len(idx)
        self.idx = idx

    def forward(self, x):
        return x.permute(self.idx)

class FuseAttention(nn.Module):
    def __init__(self, hidden_dim, concept_dim, norm=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.concept_dim = concept_dim
        self.query = nn.Linear(self.hidden_dim, self.concept_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.norm = norm
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, feat, concept):

        query = self.query(feat)
        key = self.key(concept)
        value = self.value(concept)

        attention_scores = torch.matmul(query, key.transpose(1, 0))
        attention_scores = nn.Softmax(dim=1)(attention_scores * 10)
        attention_scores = self.dropout(attention_scores)

        out = torch.matmul(attention_scores, value)

        if self.norm:
            out = l2norm(out + feat)

        return out
