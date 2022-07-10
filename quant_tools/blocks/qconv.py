import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_qblock
from .basic import QuantBasic


__all__ = ["QuantConv2d"]


@register_qblock
class QuantConv2d(QuantBasic):
    """
    Quantized Conv2d that can perform quantized convolution or normal convolution.
    To activate quantization, please use enable_quant function.
    """

    def __init__(
        self, org_module: nn.Conv2d, config, w_c_axis=0, a_c_axis=1, disable_aq=False
    ):
        super(QuantConv2d, self).__init__()
        self.cfg = config
        self.fwd_kwargs = dict(
            stride=org_module.stride,
            padding=0,
            dilation=org_module.dilation,
            groups=org_module.groups,
        )
        pad_h, pad_w = org_module.padding
        self.padding = (pad_w, pad_w, pad_h, pad_h)
        self.padding_value = 0
        self.weight = org_module.weight
        self.bias = org_module.bias
        self.w_c_axis = w_c_axis
        self.a_c_axis = a_c_axis
        self.disable_aq = disable_aq

    def forward(self, x_in: torch.Tensor):
        x_in = nn.functional.pad(
            x_in, self.padding, mode="constant", value=self.padding_value
        )
        weight = self.weight_quantizer(self.weight) if self.use_wq else self.weight
        out = F.conv2d(x_in, weight, self.bias, **self.fwd_kwargs)
        out = self.bn_function(out)
        out = self.act_function(out)
        out = self.output_quantizer(out) if self.use_aq else out
        return out
