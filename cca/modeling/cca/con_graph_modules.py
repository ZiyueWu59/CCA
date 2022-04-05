import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.nn import Parameter
from .util_C_GCN import *


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, which shared the weight between two separate graphs
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj['adj_all'], support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class C_GCN(nn.Module):

    def __init__(self, num_classes, in_channel=300, t=0, embed_size=None, adj_file=None, norm_func='sigmoid', num_path=None, com_path=None):
        super(C_GCN, self).__init__()

        self.num_classes = num_classes
        self.gc1 = GraphConvolution(in_channel, embed_size // 2)
        self.gc2 = GraphConvolution(embed_size // 2,  embed_size)
        self.relu = nn.LeakyReLU(0.2)

        # concept correlation mat generation
        _adj = gen_A_concept(num_classes, t, adj_file, num_path=num_path, com_path=com_path)

        self.adj_all = Parameter(torch.from_numpy(_adj['adj_all']).float())

        self.norm_func = norm_func
        self.softmax = nn.Softmax(dim=1)
        self.joint_att_emb = nn.Linear(embed_size, embed_size)
        self.embed_size = embed_size
        self.init_weights()

    def init_weights(self):
        """Xavier initialization"""
        r = np.sqrt(6.) / np.sqrt(self.embed_size + self.embed_size)
        self.joint_att_emb.weight.data.uniform_(-r, r)
        self.joint_att_emb.bias.data.fill_(0)


    def forward(self, inp):

        inp = inp[0]

        adj_all = gen_adj(self.adj_all).detach()

        adj = {}

        adj['adj_all'] = adj_all

        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        concept_feature = x
        concept_feature = l2norm(concept_feature)

        return concept_feature


def l2norm(input, axit=-1):
    norm = torch.norm(input, p=2, dim=-1, keepdim=True) + 1e-12
    output = torch.div(input, norm)
    return output
