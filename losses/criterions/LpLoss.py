import torch
from torch import nn

from losses import BaseLoss as Base


class Criterion(Base):
    def __init__(self, config=None, model=None, p=2.0, reduction="none"):
        super(Criterion, self).__init__(config, model)
        self.name = "LP"
        if config is not None:
            _config = config.TRAIN.LOSS.CRITERION.LPLOSS
            self.p = _config.P
            self.reduction = _config.REDUCTION
        else:
            self.p = p
            self.reduction = reduction

    def __call__(self, pred, target):
        if self.reduction == "none":
            val = (pred - target).abs().pow(self.p).sum(1).mean()
        else:
            val = (pred - target).abs().pow(self.p).mean()
        self.update(val.item(), batch_size=pred.shape[0])
        return val
