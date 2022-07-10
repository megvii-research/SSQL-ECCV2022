import re
import math
import collections
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.efficientnet_builder import (
    resolve_bn_args,
    decode_arch_def,
    round_channels,
    resolve_act_layer,
)
from timm.models.efficientnet import _create_effnet
from timm.models.layers import (
    create_conv2d,
    drop_path,
    create_act_layer,
    make_divisible,
)

from .utils import get_state_dict
from .registry import register_block


__all__ = [
    "efficientnet_lite0",
    "efficientnet_lite1",
    "efficientnet_lite2",
    "efficientnet_lite3",
    "efficientnet_lite4",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_b8",
]


def _gen_efficientnet_lite(
    arch,
    channel_multiplier=1.0,
    depth_multiplier=1.0,
    ckpt_path=None,
    num_classes=1000,
    **kwargs
):
    """Creates an EfficientNet-Lite model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
      'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
      'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
      'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
      'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
      'efficientnet-lite4': (1.4, 1.8, 300, 0.3),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """
    arch_def = [
        ["ds_r1_k3_s1_e1_c16"],
        ["ir_r2_k3_s2_e6_c24"],
        ["ir_r2_k5_s2_e6_c40"],
        ["ir_r3_k3_s2_e6_c80"],
        ["ir_r3_k5_s1_e6_c112"],
        ["ir_r4_k5_s2_e6_c192"],
        ["ir_r1_k3_s1_e6_c320"],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, fix_first_last=True),
        num_features=1280,
        stem_size=32,
        fix_stem=True,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        act_layer=resolve_act_layer(kwargs, "relu6"),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet(arch, pretrained=False, **model_kwargs)
    if ckpt_path:
        state_dict = get_state_dict(ckpt_path)
        model.load_state_dict(state_dict)
    return model


def _gen_efficientnet(
    arch,
    channel_multiplier=1.0,
    depth_multiplier=1.0,
    ckpt_path=None,
    num_classes=1000,
    **kwargs
):
    """Creates an EfficientNet model.
    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946
    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """
    arch_def = [
        ["ds_r1_k3_s1_e1_c16_se0.25"],
        ["ir_r2_k3_s2_e6_c24_se0.25"],
        ["ir_r2_k5_s2_e6_c40_se0.25"],
        ["ir_r3_k3_s2_e6_c80_se0.25"],
        ["ir_r3_k5_s1_e6_c112_se0.25"],
        ["ir_r4_k5_s2_e6_c192_se0.25"],
        ["ir_r1_k3_s1_e6_c320_se0.25"],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        act_layer=resolve_act_layer(kwargs, "swish"),
        norm_layer=kwargs.pop("norm_layer", None)
        or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet(arch, pretrained=False, **model_kwargs)
    return model


def efficientnet_lite0(ckpt_path=None, num_classes=1000):
    return _gen_efficientnet_lite(
        "efficientnet_lite0",
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        ckpt_path=ckpt_path,
        num_classes=num_classes,
        bn_eps=1e-3,
        bn_momentum=0.1,
    )


def efficientnet_lite1(ckpt_path=None, num_classes=1000):
    return _gen_efficientnet_lite(
        "efficientnet_lite1",
        channel_multiplier=1.0,
        depth_multiplier=1.1,
        ckpt_path=ckpt_path,
        num_classes=num_classes,
        bn_eps=1e-3,
        bn_momentum=0.1,
    )


def efficientnet_lite2(ckpt_path=None, num_classes=1000):
    return _gen_efficientnet_lite(
        "efficientnet_lite2",
        channel_multiplier=1.1,
        depth_multiplier=1.2,
        ckpt_path=ckpt_path,
        num_classes=num_classes,
        bn_eps=1e-3,
        bn_momentum=0.1,
    )


def efficientnet_lite3(ckpt_path=None, num_classes=1000):
    return _gen_efficientnet_lite(
        "efficientnet_lite3",
        channel_multiplier=1.2,
        depth_multiplier=1.4,
        ckpt_path=ckpt_path,
        num_classes=num_classes,
        bn_eps=1e-3,
        bn_momentum=0.1,
    )


def efficientnet_lite4(ckpt_path=None, num_classes=1000):
    return _gen_efficientnet_lite(
        "efficientnet_lite4",
        channel_multiplier=1.4,
        depth_multiplier=1.8,
        ckpt_path=ckpt_path,
        num_classes=num_classes,
        bn_eps=1e-3,
        bn_momentum=0.1,
    )


def efficientnet_b0(ckpt_path=None, num_classes=1000):
    return _gen_efficientnet(
        "efficientnet_b0",
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        ckpt_path=ckpt_path,
        num_classes=num_classes,
        bn_eps=1e-3,
        bn_momentum=0.1,
    )


def efficientnet_b1(ckpt_path=None, num_classes=1000):
    return _gen_efficientnet(
        "efficientnet_b1",
        channel_multiplier=1.0,
        depth_multiplier=1.1,
        ckpt_path=ckpt_path,
        num_classes=num_classes,
        bn_eps=1e-3,
        bn_momentum=0.1,
    )


def efficientnet_b2(ckpt_path=None, num_classes=1000):
    return _gen_efficientnet(
        "efficientnet_b2",
        channel_multiplier=1.1,
        depth_multiplier=1.2,
        ckpt_path=ckpt_path,
        num_classes=num_classes,
        bn_eps=1e-3,
        bn_momentum=0.1,
    )


def efficientnet_b3(ckpt_path=None, num_classes=1000):
    return _gen_efficientnet(
        "efficientnet_b3",
        channel_multiplier=1.2,
        depth_multiplier=1.4,
        ckpt_path=ckpt_path,
        num_classes=num_classes,
        bn_eps=1e-3,
        bn_momentum=0.1,
    )


def efficientnet_b4(ckpt_path=None, num_classes=1000):
    return _gen_efficientnet(
        "efficientnet_b4",
        channel_multiplier=1.4,
        depth_multiplier=1.8,
        ckpt_path=ckpt_path,
        num_classes=num_classes,
        bn_eps=1e-3,
        bn_momentum=0.1,
    )


def efficientnet_b5(ckpt_path=None, num_classes=1000):
    return _gen_efficientnet(
        "efficientnet_b5",
        channel_multiplier=1.6,
        depth_multiplier=2.2,
        ckpt_path=ckpt_path,
        num_classes=num_classes,
        bn_eps=1e-3,
        bn_momentum=0.1,
    )


def efficientnet_b6(ckpt_path=None, num_classes=1000):
    return _gen_efficientnet(
        "efficientnet_b6",
        channel_multiplier=1.8,
        depth_multiplier=2.6,
        ckpt_path=ckpt_path,
        num_classes=num_classes,
        bn_eps=1e-3,
        bn_momentum=0.1,
    )


def efficientnet_b7(ckpt_path=None, num_classes=1000):
    return _gen_efficientnet(
        "efficientnet_b7",
        channel_multiplier=2.0,
        depth_multiplier=3.1,
        ckpt_path=ckpt_path,
        num_classes=num_classes,
        bn_eps=1e-3,
        bn_momentum=0.1,
    )


def efficientnet_b8(ckpt_path=None, num_classes=1000):
    return _gen_efficientnet(
        "efficientnet_b8",
        channel_multiplier=2.2,
        depth_multiplier=3.6,
        ckpt_path=ckpt_path,
        num_classes=num_classes,
        bn_eps=1e-3,
        bn_momentum=0.1,
    )


@register_block
class DepthwiseSeparableConv(nn.Module):
    """DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """

    def __init__(
        self,
        in_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        dilation=1,
        pad_type="",
        noskip=False,
        pw_kernel_size=1,
        pw_act=False,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        se_layer=None,
        drop_path_rate=0.0,
    ):
        super(DepthwiseSeparableConv, self).__init__()
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.drop_path_rate = drop_path_rate

        self.conv_dw = create_conv2d(
            in_chs,
            in_chs,
            dw_kernel_size,
            stride=stride,
            dilation=dilation,
            padding=pad_type,
            depthwise=True,
        )
        self.bn1 = norm_layer(in_chs)
        self.act1 = act_layer(inplace=True)

        # Squeeze-and-excitation
        self.se = se_layer(in_chs, act_layer=act_layer) if se_layer else nn.Identity()

        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs)
        self.act2 = act_layer(inplace=True) if self.has_pw_act else nn.Identity()

    def feature_info(self, location):
        if location == "expansion":  # after SE, input to PW
            info = dict(
                module="conv_pw",
                hook_type="forward_pre",
                num_chs=self.conv_pw.in_channels,
            )
        else:  # location == 'bottleneck', block output
            info = dict(module="", hook_type="", num_chs=self.conv_pw.out_channels)
        return info

    def forward(self, x):
        shortcut = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


@register_block
class InvertedResidual(nn.Module):
    """Inverted residual block w/ optional SE

    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(
        self,
        in_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        dilation=1,
        pad_type="",
        noskip=False,
        exp_ratio=1.0,
        exp_kernel_size=1,
        pw_kernel_size=1,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        se_layer=None,
        conv_kwargs=None,
        drop_path_rate=0.0,
    ):
        super(InvertedResidual, self).__init__()
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Point-wise expansion
        self.conv_pw = create_conv2d(
            in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs
        )
        self.bn1 = norm_layer(mid_chs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs,
            mid_chs,
            dw_kernel_size,
            stride=stride,
            dilation=dilation,
            padding=pad_type,
            depthwise=True,
            **conv_kwargs,
        )
        self.bn2 = norm_layer(mid_chs)
        self.act2 = act_layer(inplace=True)

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(
            mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs
        )
        self.bn3 = norm_layer(out_chs)

    def feature_info(self, location):
        if location == "expansion":  # after SE, input to PWL
            info = dict(
                module="conv_pwl",
                hook_type="forward_pre",
                num_chs=self.conv_pwl.in_channels,
            )
        else:  # location == 'bottleneck', block output
            info = dict(module="", hook_type="", num_chs=self.conv_pwl.out_channels)
        return info

    def forward(self, x):
        shortcut = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut

        return x


@register_block
class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
        self,
        in_chs,
        rd_ratio=0.25,
        rd_channels=None,
        act_layer=nn.ReLU,
        gate_layer=nn.Sigmoid,
        force_act_layer=None,
        rd_round_fn=None,
    ):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = create_act_layer(act_layer, inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)
