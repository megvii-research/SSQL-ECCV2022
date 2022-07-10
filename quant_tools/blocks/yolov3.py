import torch
import torch.nn as nn

from .basicblock import QuantBasicBlock
from .qconv import QuantConv2d
from .registry import register_qblock

__all__ = ["ResLayer"]


@register_qblock
class ResLayer(QuantBasicBlock):
    "Resnet style layer with `ni` inputs."

    def __init__(self, org_block, config):
        super(ResLayer, self).__init__()
        self.layer1 = QuantConv2d(org_block.layer1.conv, config)
        self.layer1.bn_function = org_block.layer1.bn
        self.layer1.act_function = org_block.layer1.relu
        self.layer2 = QuantConv2d(org_block.layer2.conv, config)
        self.layer2.bn_function = org_block.layer2.bn
        self.layer2.act_function = org_block.layer2.relu
        self.act_function = nn.Identity()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out + x
        out = self.act_function(out)
        out = self.output_quantizer(out) if self.use_aq else out
        return out
