from email.policy import default
import os
from os.path import join, dirname
import json
import logging
import h5py
import torch
import torchtext
import numpy as np
from torch.functional import F
import pickle
from .utils import moment_to_iou2d, embedding

class TACoSDataset(torch.utils.data.Dataset):

    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    
    def __init__(self, root, ann_file, feat_file, num_pre_clips, num_clips, pre_query_size, attr_inp_emb, com_emb=None):
        super(TACoSDataset, self).__init__()

        with open(ann_file,'r') as f:
            annos = json.load(f)

        self.annos =  []
        self.qid = 0
        logger = logging.getLogger("tan.trainer")
        logger.info("Preparing data, please wait...")
        annos = pickle.load(open(ann_file.split('.')[0] + '_anns.pkl', 'rb'))
        self.annos = annos
        self.qids = []
        for qid in annos.keys():
            sent = annos[qid]['sentence']
            annos[qid]['query'] = embedding(sent)
            self.qids.append(int(qid))

        self.max_words_in_sent = 30

        self.feat_file = feat_file
        self.num_pre_clips = num_pre_clips
        attribute_input_emb = pickle.load(open(attr_inp_emb, 'rb'))
        if com_emb is not None:
            com_dict = pickle.load(open(com_emb, 'rb'))
            com_vectors = []
            for k in com_dict.keys():
                com_vectors.append(com_dict[k])
            com_vectors = np.array(com_vectors)
            attribute_input_emb = np.concatenate([attribute_input_emb, com_vectors], 0)
        self.attribute_input_emb = attribute_input_emb


    def __getitem__(self, idx):
        qid = self.qids[idx]
        anno = self.annos[str(qid)]
        vid = anno['vid']
        attribute_input_emb = torch.Tensor(self.attribute_input_emb)
        
        feat = get_features(self.feat_file, vid, self.num_pre_clips, "tacos")

        return feat, anno['query'], anno['wordlen'], anno['iou2d'], attribute_input_emb, idx

    def __len__(self):
        return len(self.annos)

    def get_duration(self, idx):
        qid = self.qids[idx]
        return self.annos[str(qid)]['duration']

    def get_sentence(self, idx):
        qid = self.qids[idx]
        return self.annos[str(qid)]['sentence']

    def get_moment(self, idx):
        qid = self.qids[idx]
        return self.annos[str(qid)]['moment']

    def get_vid(self, idx):
        qid = self.qids[idx]
        return self.annos[str(qid)]['vid']

def get_features(feat_file, vid, num_pre_clips, dataset_name):
    # assert exists(feat_file)
    with h5py.File(feat_file, 'r') as f:
        if dataset_name == "activitynet":
            feat = f[vid]['c3d_features'][:]
        elif dataset_name == 'tacos':
            if vid not in f.keys():
                feat = f[vid.split('.')[0]][:]
            else:
                feat = f[vid][:]
        feat = F.normalize(torch.from_numpy(feat),dim=1)
        vid_feats = avgfeats(feat, num_pre_clips)
    return vid_feats

def avgfeats(feats, num_pre_clips):
    # Produce the feature of per video into fixed shape (e.g. 256*4096)
    # Input Example: feats (torch.tensor, ?x4096); num_pre_clips (256)
    num_src_clips = feats.size(0)
    idxs = torch.arange(0, num_pre_clips+1, 1.0) / num_pre_clips * num_src_clips
    idxs = idxs.round().long().clamp(max=num_src_clips-1)
    # To prevent a empty selection, check the idxs
    meanfeats = []
    for i in range(num_pre_clips):
        s, e = idxs[i], idxs[i+1]
        if s < e:
            meanfeats.append(feats[s:e].mean(dim=0))
        else:
            meanfeats.append(feats[s])
    return torch.stack(meanfeats)


