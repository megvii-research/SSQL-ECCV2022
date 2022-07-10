import torch
import torch.nn as nn

from .basicblock import QuantBasicBlock
from .qconv import QuantConv2d
from .qlinear import QuantLinear
from .registry import register_qblock


__all__ = ["InvertedResidual"]


@register_qblock
class InvertedResidual(QuantBasicBlock):
    def __init__(self, org_block, config):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = org_block.use_res_connect
        self.expand_ratio = org_block.expand_ratio
        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                QuantConv2d(org_block.conv[0], config),
                QuantConv2d(org_block.conv[3], config),
            )
            self.conv[0].bn_function = org_block.conv[1]
            self.conv[0].act_function = nn.ReLU6(inplace=True)
            self.conv[1].bn_function = org_block.conv[4]
        else:
            self.conv = nn.Sequential(
                QuantConv2d(org_block.conv[0], config),
                QuantConv2d(org_block.conv[3], config),
                QuantConv2d(org_block.conv[6], config),
            )
            self.conv[0].bn_function = org_block.conv[1]
            self.conv[0].act_function = nn.ReLU6(inplace=True)
            self.conv[1].bn_function = org_block.conv[4]
            self.conv[1].act_function = nn.ReLU6(inplace=True)
            self.conv[2].bn_function = org_block.conv[7]
        self.act_function = nn.Identity()

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        out = self.act_function(out)
        out = self.output_quantizer(out) if self.use_aq else out
        return out
