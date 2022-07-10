import torch
import torch.nn as nn
from timm.models.layers import drop_path

from .basic import QuantBasic
from .basicblock import QuantBasicBlock
from .qconv import QuantConv2d
from .qlinear import QuantLinear
from .registry import register_qblock
from . import utils


@register_qblock
class SqueezeExcite(QuantBasicBlock):
    """Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(self, org_block, config):
        super(SqueezeExcite, self).__init__()
        self.conv_reduce = QuantConv2d(org_block.conv_reduce, config)
        self.conv_reduce.act_function = org_block.act1
        self.conv_expand = QuantConv2d(org_block.conv_expand, config)
        self.conv_expand.act_function = org_block.gate

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.conv_expand(x_se)
        out = x * x_se
        out = self.output_quantizer(out) if self.use_aq else out
        return out


@register_qblock
class DepthwiseSeparableConv(QuantBasicBlock):
    """DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """

    def __init__(self, org_block, config):
        super(DepthwiseSeparableConv, self).__init__()
        self.has_residual = org_block.has_residual
        self.has_pw_act = org_block.has_pw_act
        self.drop_path_rate = org_block.drop_path_rate
        self.conv_dw = QuantConv2d(org_block.conv_dw, config)
        self.conv_dw.bn_function = org_block.bn1
        self.conv_dw.act_function = org_block.act1
        self.se = SqueezeExcite(org_block.se, config)
        self.conv_pw = QuantConv2d(org_block.conv_pw, config)
        self.conv_pw.bn_function = org_block.bn2
        self.conv_pw.act_function = org_block.act2

    def forward(self, x):
        shortcut = x

        x = self.conv_dw(x)
        x = self.se(x)
        x = self.conv_pw(x)

        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        out = self.act_function(x)
        out = self.output_quantizer(out) if self.use_aq else out
        return x


@register_qblock
class InvertedResidual(QuantBasicBlock):
    """Inverted residual block w/ optional SE

    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(self, org_block, config):
        super(InvertedResidual, self).__init__()
        self.has_residual = org_block.has_residual
        self.drop_path_rate = org_block.drop_path_rate
        self.conv_pw = QuantConv2d(org_block.conv_pw, config)
        self.conv_pw.bn_function = org_block.bn1
        self.conv_pw.act_function = org_block.act1

        self.conv_dw = QuantConv2d(org_block.conv_dw, config)
        self.conv_dw.bn_function = org_block.bn2
        self.conv_dw.act_function = org_block.act2

        self.se = SqueezeExcite(org_block.se, config)
        self.conv_pwl = QuantConv2d(org_block.conv_pwl, config)
        self.conv_pwl.bn_function = org_block.bn3

    def forward(self, x):
        shortcut = x
        x = self.conv_pw(x)  # Point-wise expansion
        x = self.conv_dw(x)  # Depth-wise convolution
        x = self.se(x)  # Squeeze-and-excitation
        x = self.conv_pwl(x)  # Point-wise linear projection
        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        out = self.act_function(x)
        out = self.output_quantizer(out) if self.use_aq else out
        return x
