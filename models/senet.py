#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.layers import make_divisible

from .resnet import ResNet
from .registry import register_block
from .utils import get_state_dict

__all__ = ["senet_r50"]


@register_block
class SEModule(nn.Module):
    """SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """

    def __init__(
        self, channels, reduction=16, rd_channels=None, rd_divisor=8, norm_layer=None
    ):
        super(SEModule, self).__init__()
        if not rd_channels:
            rd_ratio = 1.0 / reduction
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=True)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


@register_block
class SEBottleNeck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        *,
        reduction=16
    ):
        super(SEBottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def senet_r50(ckpt_path, num_classes=1000):
    model = ResNet(
        block=SEBottleNeck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
    )
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if ckpt_path:
        state_dict = get_state_dict(ckpt_path)
        model.load_state_dict(state_dict)
    return model
