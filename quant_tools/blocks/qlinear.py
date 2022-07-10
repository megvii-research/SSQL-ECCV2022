import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_qblock
from .basic import QuantBasic


__all__ = ["QuantLinear"]


@register_qblock
class QuantLinear(QuantBasic):
    """
    Quantized Module that can perform quantized linear or normal linear.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(self, org_module: nn.Linear, config, w_c_axis=0, a_c_axis=-1):
        super(QuantLinear, self).__init__()
        self.cfg = config
        self.weight = org_module.weight
        self.bias = org_module.bias
        self.w_c_axis = w_c_axis
        self.a_c_axis = a_c_axis

    def forward(self, x_in: torch.Tensor):
        weight = self.weight_quantizer(self.weight) if self.use_wq else self.weight
        out = F.linear(x_in, weight, self.bias)
        out = self.bn_function(out)
        out = self.act_function(out)
        out = self.output_quantizer(out) if self.use_aq else out
        return out
