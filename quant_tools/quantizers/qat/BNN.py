import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from quant_tools.quantizers import BaseQuantizer


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.sign()
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


class QWeight(BaseQuantizer):
    def __init__(self, cfg, c_axis, weight):
        super(QWeight, self).__init__(cfg, c_axis=c_axis)
        self.type = "BNN"
        self.weight = weight

    def set_bit(self, bit):
        assert bit in [0, 1], "only support quant(bit=1) or not quant(bit=0)"
        super(QWeight, self).set_bit(bit)

    def _calc_Qscope(self):
        self.Qn = -1
        self.Qp = 1

    def quant(self, x):
        clip_x = torch.clamp(x.data.clone(), -1, +1)
        x.data.copy_(clip_x)
        x_q = STE.apply(x)
        return x_q

    def dequant(self, x):
        return x


class QAct(QWeight):
    def __init__(self, config, c_axis, act_func=None):
        super(__class__.__base__, self).__init__(config, c_axis=c_axis)
        self.type = "BNN"

    def quant(self, x):
        x_q = STE.apply(x)
        return x_q
