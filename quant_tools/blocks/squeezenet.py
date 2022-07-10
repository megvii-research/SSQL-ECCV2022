import torch
import torch.nn as nn

from .basicblock import QuantBasicBlock
from .qconv import QuantConv2d
from .qlinear import QuantLinear
from .registry import register_qblock

__all__ = ["Fire"]


@register_qblock
class Fire(QuantBasicBlock):
    def __init__(self, org_block, config):
        super(Fire, self).__init__()
        self.squeeze = QuantConv2d(org_block.squeeze, config)
        self.squeeze.act_function = nn.ReLU(inplace=True)
        self.expand1x1 = QuantConv2d(org_block.expand1x1, config)
        self.expand1x1.act_function = nn.ReLU(inplace=True)
        self.expand3x3 = QuantConv2d(org_block.expand3x3, config)
        self.expand3x3.act_function = nn.ReLU(inplace=True)

    def set_quant_state(self, w_quant=False, a_quant=False, w_init=False, a_init=False):
        super(Fire, self).set_quant_state(w_quant, a_quant, w_init, a_init)
        self.expand1x1.set_quant_state(w_quant=w_quant, a_quant=False)
        self.expand3x3.set_quant_state(w_quant=w_quant, a_quant=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x)
        x = torch.cat([self.expand1x1(x), self.expand3x3(x)], 1)
        x = self.output_quantizer(x) if self.use_aq else x
        return x
