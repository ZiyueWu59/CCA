import math
import pickle
from urllib.request import urlretrieve
import torch
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F
import os


def gen_A_concept(num_classes, t, adj_file, num_path=None, com_path=None):
    import pickle
    result = pickle.load(open(adj_file, 'rb')).numpy()
    for idx in range(result.shape[0]):
        result[idx][idx] = 0

    _nums = get_num(num_path)

    _A_adj = {}
    
    _adj_all = result
    _adj_all = _adj_all / _nums

    _adj_all = rescale_adj_matrix(_adj_all)
    _adj_all[_adj_all < t] = 0
    _adj_all[_adj_all >= t] = 1 
    _adj_all = generate_com_weight(_adj_all, com_path)
    _adj_all = _adj_all * 0.25 / (_adj_all.sum(0, keepdims=True) + 1e-6)
    _adj_all = _adj_all + np.identity(num_classes, np.int)  # identity square matrix
    _A_adj['adj_all'] = _adj_all

    return _A_adj


def rescale_adj_matrix(adj_mat, t=5, p=0.02):

    adj_mat_smooth = np.power(t, adj_mat - p) - np.power(t,  -p)
    return adj_mat_smooth


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


def get_num(path=None):
    concept_dict = pickle.load(open(path, 'rb'))
    num = len(concept_dict)
    _num = np.zeros([num, 1], dtype=np.int32)
    key_list = list(concept_dict.keys())
    for idx in range(len(key_list)):
        _num[idx][0] = concept_dict[key_list[idx]]
    return _num

def generate_com_weight(_adj_all, com_path):

    com_weight = pickle.load(open(com_path, 'rb'))
    train_length = _adj_all.shape[0]
    com_length = com_weight.shape[0]
    all_length = train_length + com_length
    _adj = np.zeros([all_length, all_length], dtype=np.int32)
    _adj[:train_length, :train_length] = _adj_all
    _adj[train_length:, :] = com_weight
    _adj[:, train_length:] = np.transpose(com_weight)
    return _adj

