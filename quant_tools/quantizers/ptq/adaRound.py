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
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568
    :param round_mode: controls the forward pass in this quantizer
    """

    def __init__(self, config, c_axis, weight=None):
        super(QWeight, self).__init__(config, c_axis=c_axis, weight=weight)
        self.type = "adaRound"
        self.weight = weight
        self.round_mode = "learned_hard_sigmoid"
        self.alpha = None
        self.soft_targets = False

    def init_quant_params(self):
        if self.bit != 0:
            self.scale, self.zero_point = self.observer.calc_quant_params()
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2 / 3
        self.init_alpha(x=self.weight.clone())

    def set_round_mode(self, mode):
        assert mode in [
            "nearest",
            "nearest_ste",
            "stochastic",
            "learned_hard_sigmoid",
        ], "no support {} mode".format(mode)
        self.round_mode = mode

    def init_alpha(self, x: torch.Tensor):
        if len(x.shape) != len(self.scale.shape):
            self._reshape_quant_params(x)
        x_floor = torch.floor(x / self.scale)
        if self.round_mode == "learned_hard_sigmoid":
            rest = (x / self.scale) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log(
                (self.zeta - self.gamma) / (rest - self.gamma) - 1
            )  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError

    def get_soft_targets(self):
        return torch.clamp(
            torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1
        )

    def quant(self, x_f):
        if self.round_mode == "nearest":
            x_int = torch.round(x_f / self.scale)
        elif self.round_mode == "nearest_ste":
            x_int = STE.apply(x_f / self.scale)
        elif self.round_mode == "stochastic":
            x_floor = torch.floor(x_f / self.scale)
            rest = (x_f / self.scale) - x_floor  # rest of rounding
            x_int = x_floor + torch.bernoulli(rest)
            print("Draw stochastic sample")
        elif self.round_mode == "learned_hard_sigmoid":
            x_floor = torch.floor(x_f / self.scale)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).float()
        else:
            raise ValueError("Wrong rounding mode")
        if self.is_symmetry:
            x_int = x_int + self.zero_point
            x_q = torch.clamp(x_int, -(2 ** (self.bit - 1)), 2 ** (self.bit - 1) - 1)
        else:
            x_uint = x_int + self.zero_point
            x_q = torch.clamp(x_uint, 0, 2 ** self.bit - 1)
        return x_q

    def dequant(self, x_q):
        x_dq = (x_q - self.zero_point) * self.scale
        return x_dq


class QAct(BaseQuantizer):
    def __init__(self, config, c_axis, act_func=None):
        super(QAct, self).__init__(config, c_axis=c_axis)
        self.type = "adaRound"

    def init_quant_params(self):
        if self.bit != 0:
            self.scale, self.zero_point = self.observer.calc_quant_params()

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
