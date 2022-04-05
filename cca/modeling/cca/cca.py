import torch
from torch import nn

from .featpool import build_featpool
from .feat2d import build_feat2d
from .loss import build_ccaloss
from .simpredictor import build_simpredictor
import torch.nn.functional as F
from .con_graph_modules import *
from .utils import FuseAttention

class CCA(nn.Module):
    def __init__(self, cfg):
        super(CCA, self).__init__()
        self.device = cfg.MODEL.DEVICE
        self.featpool = build_featpool(cfg) 
        self.feat2d = build_feat2d(cfg)
        self.simpredictor = build_simpredictor(cfg, self.feat2d.mask2d)
        self.ccaloss = build_ccaloss(cfg, self.feat2d.mask2d)
        self.T_fuse_attn = FuseAttention(512, 512, True)
        self.C_GCN = C_GCN(cfg.num_attribute, in_channel=cfg.input_channel, t=0.3, embed_size=cfg.embed_size, adj_file=cfg.adj_file,
                                norm_func=cfg.norm_func_type, num_path=cfg.num_path, com_path=cfg.com_concept)
        
        self.v_t_param = nn.Parameter(torch.FloatTensor([0.5]))

        self.concept_dim = cfg.num_attribute
        self.V_TransformerLayer = nn.TransformerEncoderLayer(cfg.MODEL.CCA.NUM_CLIPS + self.concept_dim, 8)
        self.cut_dim = cfg.MODEL.CCA.NUM_CLIPS
    
    def forward(self, batches, concept_input_embs, ious2d=None):

        concept_basis = self.C_GCN(concept_input_embs)
        feats = self.featpool(batches.feats)

        feats = torch.cat([feats, concept_basis.unsqueeze(0).repeat(feats.size(0), 1, 1).permute(0, 2, 1)], dim=2)
        feats = self.V_TransformerLayer(feats)[:, :, :self.cut_dim]
        map2d = self.feat2d(feats)

        map2d_fused, queries = self.simpredictor(batches.queries, batches.wordlens, map2d)

        queries_fused = self.T_fuse_attn(queries, concept_basis)

        v2t_map2d = queries[:, :, None, None] * map2d_fused
        v2t_scores2d = torch.sum(F.normalize(v2t_map2d), dim=1).squeeze_()
        t2v_map2d = queries_fused[:, :, None, None] * map2d
        t2v_scores2d = torch.sum(F.normalize(t2v_map2d), dim=1).squeeze_()
        
        original_scores2d = self.v_t_param * v2t_scores2d + (1 - self.v_t_param) * t2v_scores2d

        if self.training:
            original_loss = self.ccaloss(original_scores2d, ious2d)

            return original_loss

        else:

            return original_scores2d.sigmoid_() * self.feat2d.mask2d
