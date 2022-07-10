import torch
from torch import nn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from losses import BaseLoss as Base


class Criterion(Base):
    def __init__(self, config, model):
        super(Criterion, self).__init__(config, model)
        self.name = "CE"
        self.criterion = torch.nn.CrossEntropyLoss()
        if config.AUG.TRAIN.MIX.PROB:
            self.criterion = SoftTargetCrossEntropy()
        elif config.TRAIN.LABEL_SMOOTHING > 0:
            self.criterion = LabelSmoothingCrossEntropy(
                smoothing=config.TRAIN.LABEL_SMOOTHING
            )

    def __call__(self, pred, target):
        if len(pred)==2:
            val = (self.criterion(pred[0], target)+self.criterion(pred[1], target))*0.5
        else:
            val = self.criterion(pred, target)
        self.update(val.item(), batch_size=target.shape[0])
        return val
