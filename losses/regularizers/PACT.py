import torch
from torch import nn

from losses import BaseLoss as Base


class Regularizer(Base):
    def __init__(self, config, model):
        super(Regularizer, self).__init__(config, model)
        self.name = "Pact"

    def __call__(self, pred, target):
        loss = 0
        for n, p in self.model.named_parameters():
            if "alpha" in n:
                loss += (p ** 2).sum()
        self.update(loss.item(), batch_size=pred[0].shape[0] if len(pred)>1 else pred.shape[0])
        return loss
