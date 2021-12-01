import numpy as np
from empanada.array_utils import rle_iou

__all__ = ['iou']

def iou(
    gt_rle,
    pred_rle
):
    if len(gt_rle) == 0 and len(pred_rle) == 0:
        return 1
    elif len(gt_rle) == 0 or len(pred_rle) == 0:
        return 0
    
    return rle_iou(gt_rle[:, 0], gt_rle[:, 1], pred_rle[:, 0], pred_rle[:, 1])
