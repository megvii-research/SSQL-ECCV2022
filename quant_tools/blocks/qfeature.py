import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_tools.quantizers import build_quantizer
from .registry import register_qblock
from .basic import QuantBasic


__all__ = ["QuantFeature"]


@register_qblock
class QuantFeature(QuantBasic):
    """
    Quantized Feature that can output quantized feature or normal feature.
    To activate quantization, please use enable_quant function.
    """

    def __init__(self, config, a_c_axis=1):
        super(QuantFeature, self).__init__()
        self.a_c_axis = a_c_axis

    def forward(self, x_in: torch.Tensor):
        out = self.bn_function(x_in)
        out = self.act_function(out)
        out = self.output_quantizer(out) if self.use_aq else out
        return out

    def build_quantizer(self, cfg):
        self.output_quantizer = build_quantizer(
            cfg, c_axis=self.a_c_axis, act_func=self.act_function
        )

    def set_quant_state(self, w_quant=False, a_quant=False, w_init=False, a_init=False):
        self.use_aq = a_quant
        if self.use_aq and self.output_quantizer and a_init:
            self.output_quantizer.init_quant_params()

    def set_abit(self, bit):
        self.output_quantizer.bit = bit

    def set_wbit(self, bit):
        raise NotImplementedError("QuantFeature wo wbit")
