import torch
import torch.nn as nn
from abc import ABC

from quant_tools.quantizers import build_quantizer

__all__ = ["QuantBasic"]


class QuantBasic(nn.Module):
    def __init__(self):
        super(QuantBasic, self).__init__()
        self.use_wq = False
        self.use_aq = False
        self.disable_aq = False
        self.weight_quantizer = None
        self.output_quantizer = None
        self.bn_function = nn.Identity()
        self.act_function = nn.Identity()
        self.w_c_axis = None
        self.a_c_axis = None

    def forward(self, x_in: torch.Tensor):
        pass

    def set_quant_state(self, w_quant=False, a_quant=False, w_init=False, a_init=False):
        self.use_wq = w_quant
        if not self.disable_aq:
            self.use_aq = a_quant
        if self.use_wq and self.weight_quantizer and w_init:
            self.weight_quantizer.init_quant_params()
        if self.use_aq and self.output_quantizer and a_init:
            self.output_quantizer.init_quant_params()

    def build_quantizer(self, cfg):
        self.weight_quantizer = build_quantizer(
            cfg, c_axis=self.w_c_axis, weight=self.weight
        )
        self.output_quantizer = build_quantizer(
            cfg, c_axis=self.a_c_axis, act_func=self.act_function
        )

    def set_wbit(self, bit):
        self.weight_quantizer.bit = bit

    def set_abit(self, bit):
        self.output_quantizer.bit = bit

    def weight_vis(self):
        #print(self.weight)
        weight = self.weight_quantizer(self.weight)
        self.weight.data = weight
