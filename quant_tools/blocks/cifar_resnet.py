import torch
import torch.nn as nn

from .basicblock import QuantBasicBlock
from .qconv import QuantConv2d
from .qlinear import QuantLinear
from .registry import register_qblock


__all__ = ["CIFAR_BasicBlock", "CIFAR_Bottleneck"]

@register_qblock
class CIFAR_BasicBlock(QuantBasicBlock):
    def __init__(self, org_block, config):
        super(CIFAR_BasicBlock, self).__init__()
        self.conv1 = QuantConv2d(org_block.conv1, config)
        self.conv1.bn_function = org_block.bn1
        self.conv1.act_function = nn.ReLU(inplace=True)
        self.conv2 = QuantConv2d(org_block.conv2, config)
        self.conv2.bn_function = org_block.bn2
        self.downsample = None
        if org_block.downsample:
            self.downsample = QuantConv2d(org_block.downsample[0], config)
            self.downsample.bn_function = org_block.downsample[1]
        self.act_function = nn.ReLU(inplace=True)

    def forward(self, x_in: torch.Tensor):
        identity = x_in
        out = self.conv1(x_in)
        out = self.conv2(out)
        if self.downsample:
            identity = self.downsample(x_in)
        out = out + identity
        out = self.act_function(out)
        out = self.output_quantizer(out) if self.use_aq else out
        return out


@register_qblock
class CIFAR_Bottleneck(QuantBasicBlock):
    def __init__(self, org_block, config):
        super(CIFAR_Bottleneck, self).__init__()
        self.conv1 = QuantConv2d(org_block.conv1, config)
        self.conv1.bn_function = org_block.bn1
        self.conv1.act_function = nn.ReLU(inplace=True)
        self.conv2 = QuantConv2d(org_block.conv2, config)
        self.conv2.bn_function = org_block.bn2
        self.conv2.act_function = nn.ReLU(inplace=True)
        self.conv3 = QuantConv2d(org_block.conv3, config)
        self.conv3.bn_function = org_block.bn3
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
        if self.downsample:
            identity = self.downsample(x_in)
        out = out + identity
        out = self.act_function(out)
        out = self.output_quantizer(out) if self.use_aq else out
        return out
