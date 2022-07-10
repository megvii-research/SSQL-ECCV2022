import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .basicblock import QuantBasicBlock
from .qconv import QuantConv2d
from .qfeature import QuantFeature
from .registry import register_qblock
from quant_tools.quantizers import build_quantizer


__all__ = ["DenseLayer"]


@register_qblock
class DenseLayer(QuantBasicBlock):
    def __init__(self, org_block, config):
        super(DenseLayer, self).__init__()
        self.drop_rate = org_block.drop_rate
        self.pre_treatment = QuantFeature(config)
        self.pre_treatment.bn_function = org_block.norm1
        self.pre_treatment.act_function = nn.ReLU(inplace=True)
        self.conv1 = QuantConv2d(org_block.conv1, config)
        self.conv1.bn_function = org_block.norm2
        self.conv1.act_function = nn.ReLU(inplace=True)
        self.conv2 = QuantConv2d(org_block.conv2, config)

    def forward(self, input):
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input
        concated_features = torch.cat(prev_features, 1)
        out = self.pre_treatment(concated_features)
        out = self.conv1(out)
        out = self.conv2(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.output_quantizer(out) if self.use_aq else out
        return out


@register_qblock
class Transition(QuantBasicBlock):
    def __init__(self, org_block, config):
        super(Transition, self).__init__()
        self.pre_treatment = QuantFeature(config)
        self.pre_treatment.bn_function = org_block.norm
        self.pre_treatment.act_function = nn.ReLU(inplace=True)
        self.conv = QuantConv2d(org_block.conv, config)
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, input):
        out = self.pre_treatment(input)
        out = self.conv(out)
        out = self.pool(out)
        out = self.output_quantizer(out) if self.use_aq else out
        return out
