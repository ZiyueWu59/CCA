import torch
from torch.functional import F 

class CCALoss(object):
    def __init__(self, min_iou, max_iou, mask2d):
        self.min_iou, self.max_iou = min_iou, max_iou
        self.mask2d = mask2d

    def scale(self, iou):
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    def __call__(self, scores2d, ious2d):
        ious2d = self.scale(ious2d).clamp(0, 1) 
        return F.binary_cross_entropy_with_logits(
            scores2d.masked_select(self.mask2d), 
            ious2d.masked_select(self.mask2d)
        )
        
def build_ccaloss(cfg, mask2d):
    min_iou = cfg.MODEL.CCA.LOSS.MIN_IOU 
    max_iou = cfg.MODEL.CCA.LOSS.MAX_IOU
    return CCALoss(min_iou, max_iou, mask2d) 


class MLLoss(object):
    def __init__(self, mask2d):
        self.mask2d = mask2d

    def __call__(self, pred_labels, gt_labels):
        _, _, w, h = pred_labels.size()
        gt_labels = gt_labels.unsqueeze(1).unsqueeze(1).repeat(1, w, h, 1)
        loss = F.binary_cross_entropy_with_logits(pred_labels.permute(0, 2, 3, 1), gt_labels, reduce='none') * self.mask2d
        loss_value = torch.sum(loss)/ torch.sum(self.masks2d)

        return loss_value
