from numpy.core.shape_base import stack
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from cca.structures import TLGBatch


class BatchCollator(object):
    """
    Collect batch for dataloader
    """

    def __init__(self, ):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        # [xxx, xxx, xxx], [xxx, xxx, xxx] ......
        # feats, queries, wordlens, ious2d, attribute_label, label_mask, attri_input_emb, idxs = transposed_batch
        # return TLGBatch(
        #     feats=torch.stack(feats).float(),
        #     queries=pad_sequence(queries).transpose(0, 1),
        #     wordlens=torch.tensor(wordlens),
        # ), torch.stack(ious2d), torch.stack(attribute_label), torch.stack(label_mask), torch.stack(attri_input_emb), idxs
        feats, queries, wordlens, ious2d, attri_input_emb, idxs = transposed_batch
        return TLGBatch(
            feats=torch.stack(feats).float(),
            queries=pad_sequence(queries).transpose(0, 1),
            wordlens=torch.tensor(wordlens),
        ), torch.stack(ious2d), torch.stack(attri_input_emb), idxs
