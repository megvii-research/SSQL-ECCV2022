import torch
from torch import nn

from quant_tools.quantizers import BaseQuantizer


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_int = x.round()
        return x_int

    @staticmethod
    def backward(ctx, grad):
        return grad

class QWeight(BaseQuantizer):
    def __init__(self, config, c_axis, weight):
        super(QWeight, self).__init__(config, c_axis=c_axis, weight=weight)
        self.type = "uniform"

    def init_quant_params(self):
        if self.bit != 0:
            self.scale, self.zero_point = self.observer.calc_quant_params()
        #print('scale {}, zero point {}'.format(self.scale, self.zero_point))

    def quant(self, x_f):
        if self.is_symmetry:
            x_int = STE.apply(x_f / self.scale) + self.zero_point
            x_q = torch.clamp(x_int, -(2 ** (self.bit - 1)), 2 ** (self.bit - 1) - 1)
        else:
            x_uint = STE.apply(x_f / self.scale) + self.zero_point
            x_q = torch.clamp(x_uint, 0, 2 ** self.bit - 1)
        return x_q

    def dequant(self, x_q):
        x_dq = (x_q - self.zero_point) * self.scale
        return x_dq


class QAct(QWeight):
    def __init__(self, config, c_axis, act_func=None):
        super(__class__.__base__, self).__init__(config, c_axis=c_axis)
        self.type = "uniform"
