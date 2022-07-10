import math

import torch
import torch.nn as nn

from quant_tools.quantizers import BaseQuantizer
from quant_tools.quantizers.utils import is_non_negative_act


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, Qn, Qp):
        with_neg = Qn < 0
        min_val = -alpha.item() if with_neg else 0
        max_val = alpha.item()
        Qn_tensor = torch.tensor(Qn)
        ctx.save_for_backward(x, alpha, Qn_tensor)
        y = torch.clamp(x, min=min_val, max=max_val)
        scale = alpha / (Qp - Qn)
        y_q = (y / scale).round() * scale
        return y_q

    @staticmethod
    def backward(ctx, dLdyq):
        x, alpha, Qn = ctx.saved_tensors
        with_neg = Qn.item() < 0
        min_val = -alpha.item() if with_neg else 0
        max_val = alpha.item()
        lower_bound = x < min_val
        upper_bound = x > max_val
        x_mask = ~(lower_bound | upper_bound)
        dLdy = dLdyq * x_mask.float()
        grad_alpha = torch.sum(dLdyq * torch.ge(x, max_val).float()).view(-1)
        if with_neg:
            grad_alpha += torch.sum(dLdyq * torch.le(x, min_val).float()).view(-1)
        return dLdy, grad_alpha, None, None, None


class QAct(BaseQuantizer):
    def __init__(self, config, c_axis, act_func):
        super(QAct, self).__init__(config, c_axis=c_axis, act_func=act_func)
        self.type = "PACT"

    def init_quant_params(self):
        if self.bit != 0:
            self.alpha = nn.Parameter(torch.ones(1).to(self.device) * 10)
        else:
            self.alpha = None

    def _calc_Qscope(self):
        if is_non_negative_act(self.act_func):
            self.Qn = 0
            self.Qp = 2 ** self.bit - 1
        else:
            self.Qn = -(2 ** (self.bit - 1))
            self.Qp = 2 ** (self.bit - 1) - 1

    def forward(self, x):
        if self.bit == 0:
            x_q = x
        else:
            x_q = STE.apply(x, self.alpha, self.Qn, self.Qp)
        return x_q


class QWeight(BaseQuantizer):
    def __init__(self, cfg, c_axis, weight):
        raise NotImplementedError
