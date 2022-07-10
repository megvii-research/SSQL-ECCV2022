import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from quant_tools.observers import BaseObserver
from quant_tools.quantizers import BaseQuantizer
from quant_tools.quantizers.utils import is_non_negative_act
from config import update_config


class Grad_scale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


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
        super(QWeight, self).__init__(config, c_axis=c_axis)
        self.type = "LSQ"
        self.init_weight = weight.detach()
        self.eps = 1e-6
        if self.granularity == "channelwise":  # kernelwise
            oc = self.init_weight.shape[0]
        elif self.granularity == "layerwise":
            oc = 1
        if len(self.init_weight.shape[:]) == 2:
            self.alpha = nn.Parameter(torch.Tensor(oc, 1).to(self.device))
        elif len(self.init_weight.shape[:]) == 4:
            self.alpha = nn.Parameter(torch.Tensor(oc, 1, 1, 1).to(self.device))

    def init_quant_params(self):
        if self.bit > 0:
            init_alpha = (
                2
                * self.init_weight.reshape(self.alpha.shape[0], -1).abs().mean(axis=1)
                / math.sqrt(self.Qp)
            )
            self.alpha.data.copy_(init_alpha.reshape(self.alpha.shape))

    def _calc_Qscope(self):
        self.Qn = -(2 ** (self.bit - 1))
        self.Qp = 2 ** (self.bit - 1) - 1

    def forward(self, w):
        if self.bit == 0:
            w_q = w
        else:
            g = 1.0 / math.sqrt(w.numel() * self.Qp)
            alpha = Grad_scale.apply(self.alpha.clamp(self.eps), g)
            w_q = STE.apply((w / alpha).clamp(self.Qn, self.Qp)) * alpha
        return w_q


class ActObserver(BaseObserver):
    def __init__(self, c_axis, config):
        super(ActObserver, self).__init__(config, c_axis=c_axis)
        self.x = None

    def update(self, x):
        if self.x is None:
            self.x = x.detach().cpu()
        else:
            self.x = torch.cat([self.x, x.detach().cpu()], axis=0)


class QAct(BaseQuantizer):
    def __init__(self, config, c_axis, act_func):
        super(QAct, self).__init__(config, c_axis=c_axis, act_func=act_func)
        self.type = "LSQ"
        self.observer = ActObserver(config, c_axis)
        self.eps = 1e-6
        self.alpha = nn.Parameter(torch.Tensor(1).to(self.device))

    def init_quant_params(self):
        if self.bit > 0:
            init_alpha = 2 * self.observer.x.abs().mean() / math.sqrt(self.Qp)
            self.alpha.data.copy_(init_alpha.reshape(self.alpha.shape))

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
            g = 1.0 / math.sqrt(x.numel() * self.Qp)
            alpha = Grad_scale.apply(self.alpha.clamp(self.eps), g)
            x_q = STE.apply((x / alpha).clamp(self.Qn, self.Qp)) * alpha
        return x_q
