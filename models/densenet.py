# !/usr/bin/env python

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict
from .registry import register_block
from .utils import get_state_dict


@register_block
class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.norm1: nn.BatchNorm2d
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.relu1: nn.ReLU
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.conv1: nn.Conv2d
        self.add_module(
            "conv1",
            nn.Conv2d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.norm2: nn.BatchNorm2d
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.conv2: nn.Conv2d
        self.add_module(
            "conv2",
            nn.Conv2d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.drop_rate = float(drop_rate)

    def forward(self, input):
        out = self.norm1(input)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
    ) -> None:
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, features):
        for _, layer in self.items():
            new_features = layer(features)
            features = torch.cat([features, new_features], 1)
        return features


@register_block
class Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate,
        block_config,
        num_init_features,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
    ):
        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(
                            3,
                            num_init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def _load_state_dict(model, ckpt_path):
    # '.'s are no longer allowed in module names, but previous DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = get_state_dict(ckpt_path)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=True)


def _densenet(arch, growth_rate, block_config, num_init_features, ckpt_path, **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if ckpt_path:
        _load_state_dict(model, ckpt_path)
    return model


def densenet121(ckpt_path=None, num_classes=1000):
    return _densenet(
        "densenet121", 32, (6, 12, 24, 16), 64, ckpt_path, num_classes=num_classes
    )


def densenet161(ckpt_path=None, num_classes=1000):
    return _densenet(
        "densenet161", 48, (6, 12, 36, 24), 96, ckpt_path, num_classes=num_classes
    )


def densenet169(ckpt_path=None, num_classes=1000):
    return _densenet(
        "densenet169", 32, (6, 12, 32, 32), 64, ckpt_path, num_classes=num_classes
    )


def densenet201(ckpt_path=None, num_classes=1000):
    return _densenet(
        "densenet201", 32, (6, 12, 48, 32), 64, ckpt_path, num_classes=num_classes
    )
