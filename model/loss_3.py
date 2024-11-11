import os
import sys

sys.path.append(os.path.split(__file__)[0])

from loss_1 import BceLoss
from loss_2 import BoundaryLoss, GeneralizedDiceLoss


class BceAndBoundaryLoss:
    '''
    Bce与Boundary组合Loss
    '''
    has_bg_cls = False
    net_out_process_type = 'sigmoid'
    name = 'BceAndBoundaryLoss'

    def __init__(self):
        self.bce = BceLoss()
        self.bud = BoundaryLoss()

        assert self.bce.has_bg_cls == self.bud.has_bg_cls == self.has_bg_cls
        assert self.bce.net_out_process_type == self.bud.net_out_process_type == self.net_out_process_type

    def __call__(self, batch_label_pm, batch_pred_pm, pred_class_weight):
        loss1 = self.bce(batch_label_pm, batch_pred_pm, pred_class_weight)
        loss2 = self.bud(batch_label_pm, batch_pred_pm, pred_class_weight)
        return loss1 + loss2 * 0.01


class BceAndGeneralizedDiceLoss:
    '''
    Bce与GeneralizedDice组合Loss
    '''
    has_bg_cls = False
    net_out_process_type = 'sigmoid'
    name = 'BceAndGeneralizedDiceLoss'

    def __init__(self):
        self.bce = BceLoss()
        self.gd = GeneralizedDiceLoss()

        assert self.bce.has_bg_cls == self.gd.has_bg_cls == self.has_bg_cls
        assert self.bce.net_out_process_type == self.gd.net_out_process_type == self.net_out_process_type

    def __call__(self, batch_label_pm, batch_pred_pm, pred_class_weight):
        loss1 = self.bce(batch_label_pm, batch_pred_pm, pred_class_weight)
        loss2 = self.gd(batch_label_pm, batch_pred_pm, pred_class_weight)
        return loss1 + loss2

