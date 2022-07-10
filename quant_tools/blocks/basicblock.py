import torch
import torch.nn as nn
from abc import ABC

from quant_tools.quantizers import build_quantizer
from .basic import QuantBasic
from .qconv import QuantConv2d
from .qlinear import QuantLinear

__all__ = ["QuantBasicBlock"]


class QuantBasicBlock(QuantBasic):
    def __init__(self, a_c_axis=1):
        super(QuantBasicBlock, self).__init__()
        self.a_c_axis = a_c_axis

    def build_quantizer(self, cfg):
        self.output_quantizer = build_quantizer(
            cfg, c_axis=self.a_c_axis, act_func=self.act_function
        )

    def set_quant_state(self, w_quant=False, a_quant=False, w_init=False, a_init=False):
        self.use_aq = a_quant
        if self.use_aq and self.output_quantizer and a_init:
            self.output_quantizer.init_quant_params()
        for n, m in self.named_modules():
            if isinstance(m, (QuantConv2d, QuantLinear)):
                m.set_quant_state(w_quant, a_quant)
