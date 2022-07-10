import torch
from torch import nn
from losses import BaseLoss as Base


class Regularizer(Base):
    def __init__(self, config, model):
        pass

    def __call__(self, pred, target):
        return 0

    def reset(self):
        pass

    def __repr__(self):
        return ""
