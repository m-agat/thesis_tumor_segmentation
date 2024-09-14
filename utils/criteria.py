import torch
import numpy as np
from torch import nn
from torch.nn.modules.loss import _Loss
from scipy.spatial.distance import directed_hausdorff

'''
Adapted from https://github.com/RosalindFok/3D-BrainTumorSegmentat4MIA/blob/main/criteria.py
'''

class SoftDiceLoss(_Loss):
    def __init__(self, classes=5): 
        super(SoftDiceLoss, self).__init__()
        self.classes = classes

    def forward(self, y_pred, y_true, eps=1e-8):
        dice_loss = 0
        for cls in range(self.classes):  # Calculate dice for each class
            pred = y_pred[:, cls]
            true = y_true[:, cls]
            intersection = torch.sum(pred * true)
            union = torch.sum(pred * pred) + torch.sum(true * true) + eps
            dice = 2 * intersection / union
            dice_loss += 1 - dice  # Maximize dice, minimize loss
        return dice_loss / self.classes


class CombinedLoss(_Loss):
    def __init__(self, k1=0.1, k2=0.1, classes=5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = SoftDiceLoss(classes=classes)
        self.l2_loss = nn.MSELoss()

    def forward(self, seg_y_pred, seg_y_true, rec_y_pred=None, rec_y_true=None, y_mid=None):
        dice_loss = self.dice_loss(seg_y_pred, seg_y_true)
        l2_loss = self.l2_loss(rec_y_pred, rec_y_true) if rec_y_pred is not None else 0
        combined_loss = dice_loss + l2_loss
        return combined_loss


class HausdorffDistance:
    def __call__(self, y_pred, y_true):
        y_pred = y_pred.squeeze().detach().cpu().numpy()
        y_true = y_true.squeeze().detach().cpu().numpy()
        hausdorff_distance = []
        for pred, true in zip(y_pred, y_true):
            u_hd = directed_hausdorff(pred, true)[0]
            v_hd = directed_hausdorff(true, pred)[0]
            hausdorff_distance.append(max(u_hd, v_hd))
        return np.mean(hausdorff_distance)
