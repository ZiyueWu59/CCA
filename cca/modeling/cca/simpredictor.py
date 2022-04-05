import torch
from torch import nn
from torch.functional import F

def mask2weight(mask2d, mask_kernel, padding=0):
    weight = torch.conv2d(mask2d[None,None,:,:].float(),
                          mask_kernel, padding=padding)[0, 0]
    weight[weight > 0] = 1 / weight[weight > 0]
    return weight

class SimPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, k, num_stack_layers, mask2d, feat_hidden_size, query_input_size, query_hidden_size,
                 bidirectional, num_layers):
        super(SimPredictor, self).__init__()

        if bidirectional:
            query_hidden_size //= 2
        self.lstm = nn.LSTM(
            query_input_size, query_hidden_size, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True
        )
        if bidirectional:
            query_hidden_size *= 2
        self.fc_full = nn.Linear(query_hidden_size, feat_hidden_size)

        self.conv = nn.Conv2d(hidden_size, feat_hidden_size, 5, padding=2)
        self.bn = nn.BatchNorm2d(feat_hidden_size)
        self.conv1 = nn.Conv2d(feat_hidden_size, feat_hidden_size, 3, padding=1)

    def encode_query(self, queries, wordlens):

        self.lstm.flatten_parameters()
        queries = self.lstm(queries)[0]
        queries_start = queries[range(queries.size(0)),0]
        queries_end = queries[range(queries.size(0)), wordlens.long() - 1]
        full_queries = (queries_start + queries_end)/2

        return self.fc_full(full_queries)

    def forward(self, batch_queries, wordlens, map2d):

        queries = self.encode_query(batch_queries, wordlens)
        map2d = self.conv(map2d)
        map2d = F.tanh(self.bn(map2d))
        map2d = self.conv1(map2d)
        return map2d, queries


def build_simpredictor(cfg, mask2d):
    input_size = cfg.MODEL.CCA.FEATPOOL.HIDDEN_SIZE
    hidden_size = cfg.MODEL.CCA.FEATPOOL.HIDDEN_SIZE
    kernel_size = cfg.MODEL.CCA.PREDICTOR.KERNEL_SIZE
    num_stack_layers = cfg.MODEL.CCA.PREDICTOR.NUM_STACK_LAYERS
    feat_hidden_size = cfg.MODEL.CCA.FEATPOOL.HIDDEN_SIZE
    query_input_size = cfg.INPUT.PRE_QUERY_SIZE
    query_hidden_size = cfg.MODEL.CCA.INTEGRATOR.QUERY_HIDDEN_SIZE
    bidirectional = cfg.MODEL.CCA.INTEGRATOR.LSTM.BIDIRECTIONAL
    num_layers = cfg.MODEL.CCA.INTEGRATOR.LSTM.NUM_LAYERS

    return SimPredictor(
        input_size, hidden_size, kernel_size, num_stack_layers, mask2d, feat_hidden_size, query_input_size, query_hidden_size,
        bidirectional, num_layers
    )

