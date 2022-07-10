import torch
import torch.nn as nn

from .basicblock import QuantBasicBlock
from .qconv import QuantConv2d
from .registry import register_qblock


__all__ = ["SEModule", "SEBottleNeck"]


@register_qblock
class SEModule(QuantBasicBlock):
    def __init__(self, org_block, config):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = QuantConv2d(org_block.fc1, config)
        self.fc1.bn_function = org_block.bn
        self.fc1.act_function = nn.ReLU(inplace=True)
        self.fc2 = QuantConv2d(org_block.fc2, config)
        self.fc2.act_function = nn.Sigmoid()

    def forward(self, x_in: torch.Tensor):
        identity = x_in
        y = self.avg_pool(x_in)
        y = self.fc1(y)
        y = self.fc2(y)
        y = identity * y.expand_as(x_in)
        y = self.output_quantizer(y) if self.use_aq else y
        return y


@register_qblock
class SEBottleNeck(QuantBasicBlock):
    def __init__(self, org_block, config):
        super(SEBottleNeck, self).__init__()
        self.conv1 = QuantConv2d(org_block.conv1, config)
        self.conv1.bn_function = org_block.bn1
        self.conv1.act_function = nn.ReLU(inplace=True)
        self.conv2 = QuantConv2d(org_block.conv2, config)
        self.conv2.bn_function = org_block.bn2
        self.conv2.act_function = nn.ReLU(inplace=True)
        self.conv3 = QuantConv2d(org_block.conv3, config)
        self.conv3.bn_function = org_block.bn3
        self.se = SEModule(org_block.se, config)
        self.downsample = None
        if org_block.downsample:
            self.downsample = QuantConv2d(org_block.downsample[0], config)
            self.downsample.bn_function = org_block.downsample[1]
        self.act_function = nn.ReLU(inplace=True)

    def forward(self, x_in: torch.Tensor):
        identity = x_in
        out = self.conv1(x_in)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.se(out)
        if self.downsample:
            identity = self.downsample(x_in)
        out = out + identity
        out = self.act_function(out)
        out = self.output_quantizer(out) if self.use_aq else out
        return out
