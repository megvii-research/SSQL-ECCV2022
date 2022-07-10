import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Any, Optional, Tuple, List

from .basicblock import QuantBasicBlock
from .qconv import QuantConv2d
from .qlinear import QuantLinear
from .registry import register_qblock


@register_qblock
class InceptionA(QuantBasicBlock):
    def __init__(self, org_block, config) -> None:
        super(InceptionA, self).__init__()
        self.branch1x1 = QuantConv2d(org_block.branch1x1.conv, config, disable_aq=True)
        self.branch1x1.bn_function = org_block.branch1x1.bn
        self.branch1x1.act_function = nn.ReLU(inplace=True)

        self.branch5x5_1 = QuantConv2d(org_block.branch5x5_1.conv, config)
        self.branch5x5_1.bn_function = org_block.branch5x5_1.bn
        self.branch5x5_1.act_function = nn.ReLU(inplace=True)
        self.branch5x5_2 = QuantConv2d(
            org_block.branch5x5_2.conv, config, disable_aq=True
        )
        self.branch5x5_2.bn_function = org_block.branch5x5_2.bn
        self.branch5x5_2.act_function = nn.ReLU(inplace=True)

        self.branch3x3dbl_1 = QuantConv2d(org_block.branch3x3dbl_1.conv, config)
        self.branch3x3dbl_1.bn_function = org_block.branch3x3dbl_1.bn
        self.branch3x3dbl_1.act_function = nn.ReLU(inplace=True)
        self.branch3x3dbl_2 = QuantConv2d(org_block.branch3x3dbl_2.conv, config)
        self.branch3x3dbl_2.bn_function = org_block.branch3x3dbl_2.bn
        self.branch3x3dbl_2.act_function = nn.ReLU(inplace=True)
        self.branch3x3dbl_3 = QuantConv2d(
            org_block.branch3x3dbl_3.conv, config, disable_aq=True
        )
        self.branch3x3dbl_3.bn_function = org_block.branch3x3dbl_3.bn
        self.branch3x3dbl_3.act_function = nn.ReLU(inplace=True)

        self.branch_pool = QuantConv2d(
            org_block.branch_pool.conv, config, disable_aq=True
        )
        self.branch_pool.bn_function = org_block.branch_pool.bn
        self.branch_pool.act_function = nn.ReLU(inplace=True)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        outputs = torch.cat(outputs, 1)
        outputs = self.output_quantizer(outputs) if self.use_aq else outputs
        return outputs


@register_qblock
class InceptionB(QuantBasicBlock):
    def __init__(self, org_block, config) -> None:
        super(InceptionB, self).__init__()
        self.branch3x3 = QuantConv2d(org_block.branch3x3.conv, config, disable_aq=True)
        self.branch3x3.bn_function = org_block.branch3x3.bn
        self.branch3x3.act_function = nn.ReLU(inplace=True)

        self.branch3x3dbl_1 = QuantConv2d(org_block.branch3x3dbl_1.conv, config)
        self.branch3x3dbl_1.bn_function = org_block.branch3x3dbl_1.bn
        self.branch3x3dbl_1.act_function = nn.ReLU(inplace=True)
        self.branch3x3dbl_2 = QuantConv2d(org_block.branch3x3dbl_2.conv, config)
        self.branch3x3dbl_2.bn_function = org_block.branch3x3dbl_2.bn
        self.branch3x3dbl_2.act_function = nn.ReLU(inplace=True)
        self.branch3x3dbl_3 = QuantConv2d(
            org_block.branch3x3dbl_3.conv, config, disable_aq=True
        )
        self.branch3x3dbl_3.bn_function = org_block.branch3x3dbl_3.bn
        self.branch3x3dbl_3.act_function = nn.ReLU(inplace=True)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        outputs = torch.cat(outputs, 1)
        outputs = self.output_quantizer(outputs) if self.use_aq else outputs
        return outputs


@register_qblock
class InceptionC(QuantBasicBlock):
    def __init__(self, org_block, config) -> None:
        super(InceptionC, self).__init__()
        self.branch1x1 = QuantConv2d(org_block.branch1x1.conv, config, disable_aq=True)
        self.branch1x1.bn_function = org_block.branch1x1.bn
        self.branch1x1.act_function = nn.ReLU(inplace=True)

        self.branch7x7_1 = QuantConv2d(org_block.branch7x7_1.conv, config)
        self.branch7x7_1.bn_function = org_block.branch7x7_1.bn
        self.branch7x7_1.act_function = nn.ReLU(inplace=True)
        self.branch7x7_2 = QuantConv2d(org_block.branch7x7_2.conv, config)
        self.branch7x7_2.bn_function = org_block.branch7x7_2.bn
        self.branch7x7_2.act_function = nn.ReLU(inplace=True)
        self.branch7x7_3 = QuantConv2d(
            org_block.branch7x7_3.conv, config, disable_aq=True
        )
        self.branch7x7_3.bn_function = org_block.branch7x7_3.bn
        self.branch7x7_3.act_function = nn.ReLU(inplace=True)

        self.branch7x7dbl_1 = QuantConv2d(org_block.branch7x7dbl_1.conv, config)
        self.branch7x7dbl_1.bn_function = org_block.branch7x7dbl_1.bn
        self.branch7x7dbl_1.act_function = nn.ReLU(inplace=True)
        self.branch7x7dbl_2 = QuantConv2d(org_block.branch7x7dbl_2.conv, config)
        self.branch7x7dbl_2.bn_function = org_block.branch7x7dbl_2.bn
        self.branch7x7dbl_2.act_function = nn.ReLU(inplace=True)
        self.branch7x7dbl_3 = QuantConv2d(org_block.branch7x7dbl_3.conv, config)
        self.branch7x7dbl_3.bn_function = org_block.branch7x7dbl_3.bn
        self.branch7x7dbl_3.act_function = nn.ReLU(inplace=True)
        self.branch7x7dbl_4 = QuantConv2d(org_block.branch7x7dbl_4.conv, config)
        self.branch7x7dbl_4.bn_function = org_block.branch7x7dbl_4.bn
        self.branch7x7dbl_4.act_function = nn.ReLU(inplace=True)
        self.branch7x7dbl_5 = QuantConv2d(
            org_block.branch7x7dbl_5.conv, config, disable_aq=True
        )
        self.branch7x7dbl_5.bn_function = org_block.branch7x7dbl_5.bn
        self.branch7x7dbl_5.act_function = nn.ReLU(inplace=True)

        self.branch_pool = QuantConv2d(
            org_block.branch_pool.conv, config, disable_aq=True
        )
        self.branch_pool.bn_function = org_block.branch_pool.bn
        self.branch_pool.act_function = nn.ReLU(inplace=True)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        outputs = torch.cat(outputs, 1)
        outputs = self.output_quantizer(outputs) if self.use_aq else outputs
        return outputs


@register_qblock
class InceptionD(QuantBasicBlock):
    def __init__(self, org_block, config) -> None:
        super(InceptionD, self).__init__()
        self.branch3x3_1 = QuantConv2d(org_block.branch3x3_1.conv, config)
        self.branch3x3_1.bn_function = org_block.branch3x3_1.bn
        self.branch3x3_1.act_function = nn.ReLU(inplace=True)
        self.branch3x3_2 = QuantConv2d(
            org_block.branch3x3_2.conv, config, disable_aq=True
        )
        self.branch3x3_2.bn_function = org_block.branch3x3_2.bn
        self.branch3x3_2.act_function = nn.ReLU(inplace=True)

        self.branch7x7x3_1 = QuantConv2d(org_block.branch7x7x3_1.conv, config)
        self.branch7x7x3_1.bn_function = org_block.branch7x7x3_1.bn
        self.branch7x7x3_1.act_function = nn.ReLU(inplace=True)
        self.branch7x7x3_2 = QuantConv2d(org_block.branch7x7x3_2.conv, config)
        self.branch7x7x3_2.bn_function = org_block.branch7x7x3_2.bn
        self.branch7x7x3_2.act_function = nn.ReLU(inplace=True)
        self.branch7x7x3_3 = QuantConv2d(org_block.branch7x7x3_3.conv, config)
        self.branch7x7x3_3.bn_function = org_block.branch7x7x3_3.bn
        self.branch7x7x3_3.act_function = nn.ReLU(inplace=True)
        self.branch7x7x3_4 = QuantConv2d(
            org_block.branch7x7x3_4.conv, config, disable_aq=True
        )
        self.branch7x7x3_4.bn_function = org_block.branch7x7x3_4.bn
        self.branch7x7x3_4.act_function = nn.ReLU(inplace=True)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        outputs = torch.cat(outputs, 1)
        outputs = self.output_quantizer(outputs) if self.use_aq else outputs
        return outputs


@register_qblock
class InceptionE(QuantBasicBlock):
    def __init__(self, org_block, config) -> None:
        super(InceptionE, self).__init__()
        self.branch1x1 = QuantConv2d(org_block.branch1x1.conv, config, disable_aq=True)
        self.branch1x1.bn_function = org_block.branch1x1.bn
        self.branch1x1.act_function = nn.ReLU(inplace=True)

        self.branch3x3_1 = QuantConv2d(org_block.branch3x3_1.conv, config)
        self.branch3x3_1.bn_function = org_block.branch3x3_1.bn
        self.branch3x3_1.act_function = nn.ReLU(inplace=True)
        self.branch3x3_2a = QuantConv2d(
            org_block.branch3x3_2a.conv, config, disable_aq=True
        )
        self.branch3x3_2a.bn_function = org_block.branch3x3_2a.bn
        self.branch3x3_2a.act_function = nn.ReLU(inplace=True)
        self.branch3x3_2b = QuantConv2d(
            org_block.branch3x3_2b.conv, config, disable_aq=True
        )
        self.branch3x3_2b.bn_function = org_block.branch3x3_2b.bn
        self.branch3x3_2b.act_function = nn.ReLU(inplace=True)

        self.branch3x3dbl_1 = QuantConv2d(org_block.branch3x3dbl_1.conv, config)
        self.branch3x3dbl_1.bn_function = org_block.branch3x3dbl_1.bn
        self.branch3x3dbl_1.act_function = nn.ReLU(inplace=True)
        self.branch3x3dbl_2 = QuantConv2d(org_block.branch3x3dbl_2.conv, config)
        self.branch3x3dbl_2.bn_function = org_block.branch3x3dbl_2.bn
        self.branch3x3dbl_2.act_function = nn.ReLU(inplace=True)
        self.branch3x3dbl_3a = QuantConv2d(
            org_block.branch3x3dbl_3a.conv, config, disable_aq=True
        )
        self.branch3x3dbl_3a.bn_function = org_block.branch3x3dbl_3a.bn
        self.branch3x3dbl_3a.act_function = nn.ReLU(inplace=True)
        self.branch3x3dbl_3b = QuantConv2d(
            org_block.branch3x3dbl_3b.conv, config, disable_aq=True
        )
        self.branch3x3dbl_3b.bn_function = org_block.branch3x3dbl_3b.bn
        self.branch3x3dbl_3b.act_function = nn.ReLU(inplace=True)

        self.branch_pool = QuantConv2d(
            org_block.branch_pool.conv, config, disable_aq=True
        )
        self.branch_pool.bn_function = org_block.branch_pool.bn
        self.branch_pool.act_function = nn.ReLU(inplace=True)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        outputs = torch.cat(outputs, 1)
        outputs = self.output_quantizer(outputs) if self.use_aq else outputs
        return outputs
