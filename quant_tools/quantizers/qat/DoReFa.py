import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import update_config
from quant_tools.quantizers import BaseQuantizer
from quant_tools.quantizers.utils import is_non_negative_act


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Qn, Qp):
        Qrange = Qp - Qn
        x_int = (x * Qrange).round()
        x_int_bounded = x_int.clamp(Qn, Qp)
        x_dq = x_int_bounded / Qrange
        return x_dq

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


class QWeight(BaseQuantizer):
    def __init__(self, config, c_axis, weight=None):
        super(QWeight, self).__init__(config, c_axis=c_axis)
        self.type = "DoReFa"

    def _calc_Qscope(self):
        self.Qn = 0
        self.Qp = 2 ** self.bit - 1

    def forward(self, w):
        if self.bit == 0:
            w_q = w
        else:
            w_tanhed = w.tanh()
            wscale = w_tanhed.abs().reshape(w.shape[self.c_axis], -1).mean(axis=1) * 2
            dst_shape = [1] * len(w.shape)
            dst_shape[self.c_axis] = -1
            wscale = (wscale.reshape(dst_shape)).detach()
            w_normed = w_tanhed / wscale + 0.5
            w_q = (STE.apply(w_normed, self.Qn, self.Qp) - 0.5) * wscale
        return w_q


class QAct(BaseQuantizer):
    def __init__(self, config, c_axis, act_func=None):
        super(QAct, self).__init__(config, c_axis=c_axis)
        self.type = "DoReFa"
        self.offset = 0 if is_non_negative_act(act_func) else 0.5

    def _calc_Qscope(self):
        self.Qn = 0
        self.Qp = 2 ** self.bit - 1

    def forward(self, x):
        if self.bit == 0:
            x_q = x
        else:
            x_bound = (x * 0.1 + self.offset).clamp(0, 1)
            x_q = STE.apply(x_bound, self.Qn, self.Qp) - self.offset
        return x_q
