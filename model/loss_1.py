'''
Loss 函数集合
注意，Loss类状态不会保存，并且只会创建一次

简写
pm 一般概率图
oh onehot类别图
'''

import os
import sys

sys.path.append(os.path.split(__file__)[0])


import torch
import torch.nn as nn
import torch.nn.functional as F
from lovasz_losses import lovasz_hinge, lovasz_softmax


def tr_pm_to_oh(pm: torch.Tensor):
    '''
    转换概率热图到onehot热图
    :param pm:
    :return:
    '''
    bg = 1.-torch.max(pm, 1, keepdim=True)[0]
    oh = torch.cat([bg, pm], 1)
    return oh


class WeightL2LossV1:
    has_bg_cls = False
    net_out_process_type = 'clip'
    name = 'WeightL2LossV1'

    def __call__(self, batch_label_pm, batch_pred_pm, pred_class_weight):
        '''
        加权方法的 距离Loss
        :param batch_label_pm:
        :param batch_pred_pm:
        :param pred_class_weight:
        :return:
        '''
        loss = (3 * (batch_label_pm - batch_pred_pm).abs()).pow(2)
        weight = torch.where((batch_label_pm > 0.5).max(1, keepdim=True)[0], torch.full_like(batch_label_pm, 3),
                             torch.full_like(batch_label_pm, 1))
        loss = loss * weight
        pred_class_weight = torch.reshape(pred_class_weight, [1, -1, 1, 1])
        loss = loss * pred_class_weight
        loss = loss.mean(dim=[2, 3]).sum(dim=1).mean()
        return loss


class CeLoss:
    has_bg_cls = True
    net_out_process_type = 'softmax'
    name = 'CeLoss'

    def __call__(self, batch_label_pm, batch_pred_pm, pred_class_weight):
        '''
        一般交叉熵Loss，使用非one_hot标签，函数内自动计算onehot图
        :param batch_label_pm:
        :param batch_pred_pm:
        :param pred_class_weight:
        :return:
        '''
        oh = tr_pm_to_oh(batch_label_pm)
        batch_label_pm = torch.argmax(oh, 1)
        pred_class_weight = F.pad(pred_class_weight, [1, 0], 'constant', 1)
        loss = F.cross_entropy(batch_pred_pm, batch_label_pm, weight=pred_class_weight)
        return loss


class BceLoss:
    has_bg_cls = False
    net_out_process_type = 'sigmoid'
    name = 'BceLoss'

    def __call__(self, batch_label_pm, batch_pred_pm, pred_class_weight):
        '''
        二分交叉熵Loss，使用非one_hot标签，不需要背景类
        :param batch_label_pm:
        :param batch_pred_pm:
        :param pred_class_weight:
        :return:
        '''
        loss = F.binary_cross_entropy_with_logits(batch_pred_pm, batch_label_pm, weight=pred_class_weight.reshape(1,-1,1,1))
        return loss


class LovaszSoftmaxLoss:
    has_bg_cls = True
    net_out_process_type = 'softmax'
    name = 'LovaszSoftmaxLoss'

    def __call__(self, batch_label_pm, batch_pred_pm, pred_class_weight):
        '''
        lovasz_softmax Loss，使用非one_hot标签，函数内自动计算onehot图
        :param batch_label_pm:
        :param batch_pred_pm:
        :param pred_class_weight:
        :return:
        '''
        # label 值一定大于等于0.5，否则为0，以此为界限，将其转换成softmax可用label
        batch_label_pm = batch_label_pm.clone()
        batch_label_pm[:, 0].fill_(0.1)
        batch_label_pm = torch.argmax(batch_label_pm, 1)
        loss = lovasz_softmax(batch_pred_pm, batch_label_pm)
        return loss


class LovaszHingeLoss:
    has_bg_cls = False
    net_out_process_type = 'clip'
    name = 'LovaszHingeLoss'

    def __call__(self, batch_label_pm, batch_pred_pm, pred_class_weight):
        '''
        lovasz_hinge_loss 其他位置
        :param batch_label_pm:
        :param batch_pred_pm:
        :param pred_class_weight:
        :return:
        '''
        loss = 0
        for c in range(batch_label_pm.shape[1]):
            loss += lovasz_hinge(batch_pred_pm[:, c], batch_label_pm[:, c])
        return loss


if __name__ == '__main__':
    batch_label_pm = torch.rand(3, 6, 16, 16)
    batch_pred_pm = torch.rand(3, 6, 16, 16)
    batch_pred_pm_with_bg = torch.rand(3, 1+6, 16, 16)

    pred_class_weight = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
    pred_class_weight_with_bg = torch.tensor([1., 1, 2, 3, 4, 5, 6], dtype=torch.float32)

    for loss_cls in [LovaszSoftmaxLoss, CeLoss]:
        loss_func = loss_cls()
        loss = loss_func(batch_label_pm, batch_pred_pm_with_bg, pred_class_weight_with_bg)
        print(loss)

    for loss_cls in [WeightL2LossV1, LovaszHingeLoss]:
        loss_func = loss_cls()
        loss = loss_func(batch_label_pm, batch_pred_pm, pred_class_weight)
        print(loss)


