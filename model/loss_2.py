import os
import sys

sys.path.append(os.path.split(__file__)[0])

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


class SoftAccuracyLoss:
    '''
    软准确率Loss
    '''
    has_bg_cls = False
    net_out_process_type = 'sigmoid'
    name = 'SoftAccuracyLoss'

    def __call__(self, batch_label_pm, batch_pred_pm, pred_class_weight):
        label = batch_label_pm
        pred = batch_pred_pm
        if pred_class_weight is None:
            pred_class_weight = [1]*label.shape[1]
        # shape=[1,C]
        pred_class_weight = torch.tensor(pred_class_weight, dtype=label.dtype, device=label.device).reshape(1, label.shape[1])

        label = torch.clip(label, 0, 1)
        pred = torch.sigmoid(pred)

        label = torch.flatten(label, 2)
        pred = torch.flatten(pred, 2)

        label = (label - 0.5) * 2
        pred = (pred - 0.5) * 2

        soft_acc = (label * pred).mean(2)

        soft_acc_loss = 1 - soft_acc
        soft_acc_loss = soft_acc_loss * pred_class_weight

        soft_acc_loss = soft_acc_loss.mean()
        return soft_acc_loss


class SoftDiceLoss:
    '''
    软DiceLoss
    '''
    has_bg_cls = False
    net_out_process_type = 'sigmoid'
    name = 'SoftDiceLoss'

    def __call__(self, batch_label_pm, batch_pred_pm, pred_class_weight):
        label = batch_label_pm
        pred = batch_pred_pm
        if pred_class_weight is None:
            pred_class_weight = [1]*label.shape[1]
        # shape=[1,C]
        pred_class_weight = torch.tensor(pred_class_weight, dtype=label.dtype, device=label.device).reshape(1, label.shape[1])
        smooth = 1.

        label = torch.clip(label, 0, 1)
        pred = torch.sigmoid(pred)

        label = torch.flatten(label, 2)
        pred = torch.flatten(pred, 2)

        inter = (label * pred).sum(2)
        bg = label.sum(2) + pred.sum(2)

        soft_dice = (2 * inter + smooth) / (bg + smooth)

        soft_dice_loss = 1 - soft_dice
        soft_dice_loss = soft_dice_loss * pred_class_weight

        soft_dice_loss = soft_dice_loss.mean()
        return soft_dice_loss


class GeneralizedDiceLoss:
    '''
    GeneralizedDiceLoss
    '''
    has_bg_cls = False
    net_out_process_type = 'sigmoid'
    name = 'GeneralizedDiceLoss'

    def __call__(self, batch_label_pm, batch_pred_pm, pred_class_weight):
        label = batch_label_pm
        pred = batch_pred_pm
        if pred_class_weight is None:
            pred_class_weight = [1]*label.shape[1]
        # shape=[1,C]
        pred_class_weight = torch.tensor(pred_class_weight, dtype=label.dtype, device=label.device).reshape(1, label.shape[1])

        label = torch.clip(label, 0, 1)
        pred = torch.sigmoid(pred)

        label = torch.flatten(label, 2)
        pred = torch.flatten(pred, 2)

        eps = 1e-6

        w = 1 / (label.sum([0, 2]) ** 2 + eps)
        # ->[1, C]
        w = w[None]

        # [B, C, L]->[B, C]
        inter = (label * pred).sum(2)
        bg = label.sum(2) + pred.sum(2)

        w_inter = w * inter
        w_bg = w * bg

        gen_dice = (2 * w_inter) / (w_bg + eps)

        gen_dice_loss = 1 - gen_dice
        gen_dice_loss = gen_dice_loss * pred_class_weight

        gen_dice_loss = gen_dice_loss.mean()
        return gen_dice_loss


class BoundaryLoss:
    has_bg_cls = False
    net_out_process_type = 'sigmoid'
    name = 'BoundaryLoss'

    def _calc_distmap(self, label_pm):
        label_pos_bm = label_pm.cpu().numpy() > 0.5
        label_neg_bm = np.logical_not(label_pos_bm)

        label_distmap = np.zeros_like(label_pos_bm, dtype=np.float32)

        for B in range(label_pm.shape[0]):
            for C in range(label_pm.shape[1]):
                label_distmap[B, C] = distance_transform_edt(label_neg_bm[B, C]) * label_neg_bm[B, C] - (distance_transform_edt(label_pos_bm[B, C]) - 1) * label_pos_bm[B, C]

            label_distmap = torch.tensor(label_distmap, dtype=label_pm.dtype, device=label_pm.device)
        return label_distmap

    def __call__(self, batch_label_pm, batch_pred_pm, pred_class_weight):
        if pred_class_weight is None:
            pred_class_weight = [1]*batch_label_pm.shape[1]
        # shape=[1,C]
        pred_class_weight = torch.tensor(pred_class_weight, dtype=batch_label_pm.dtype, device=batch_label_pm.device).reshape(1, batch_label_pm.shape[1])

        batch_pred_pm = batch_pred_pm.sigmoid()

        batch_label_distmap = self._calc_distmap(batch_label_pm)
        loss = batch_pred_pm * batch_label_distmap
        loss = loss.mean([2, 3]) * pred_class_weight
        loss = loss.mean()
        return loss



