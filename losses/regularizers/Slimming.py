import torch
from torch import nn
from losses import BaseLoss as Base


class Regularizer(Base):
    def __init__(self, config, model):
        super(Regularizer, self).__init__(config, model)
        self.name = "Slim"

    def __call__(self, pred, target):
        loss = 0
        for n, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                loss += abs(m.weight).sum()
        self.update(loss.item(), batch_size=pred.shape[0])
        return loss
