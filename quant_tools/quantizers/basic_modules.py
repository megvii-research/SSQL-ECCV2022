import torch
from torch import nn
import abc
from quant_tools.observers import build_observer


class Quantizer(nn.Module, abc.ABC):
    def __init__(self, config, c_axis, weight=None, act_func=None):
        super(Quantizer, self).__init__()
        self.type = "Basic"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.granularity = config.GRANULARITY
        self.is_symmetry = config.SYMMETRY
        self.c_axis = c_axis
        self.bit = config.BIT
        self.scale = torch.tensor([1.0])
        self.zero_point = torch.tensor([0.0])
        self.observer = build_observer(config, c_axis)
        self.act_func = act_func
        self.weight = weight
        if self.weight is not None:
            self.observer.update(self.weight.clone())
        self._calc_Qscope()
        self.config = config

    def reset_quantizer(self):
        self.scale = torch.tensor([1.0])
        self.zero_point = torch.tensor([0.0])
        self.observer = build_observer(self.config, self.c_axis)

    def init_quant_params(self):
        pass

    def _reshape_quant_params(self, x):
        dst_shape = [1] * len(x.shape)
        dst_shape[self.c_axis] = -1
        if isinstance(self.scale, nn.Parameter):
            self.scale.data = self.scale.data.reshape(dst_shape)
        else:
            self.scale = self.scale.reshape(dst_shape)
        self.zero_point = self.zero_point.reshape(dst_shape)

    def set_bit(self, bit):
        assert bit >= 0, "only support bit is a non-negative number"
        self.bit = bit
        self.observer.set_bit(bit)
        self._calc_Qscope()

    def _calc_Qscope(self):
        pass

    def quant(self, x):
        pass

    def dequant(self, x):
        pass

    def forward(self, x):
        if self.bit == 0:
            return x
        else:
            if torch.min(self.scale)==0:
                #print('scale', self.scale)
                return x
            self._reshape_quant_params(x)
            x_q = self.quant(x)
            x_dq = self.dequant(x_q)
            #print(x_dq)
        return x_dq

    def __repr__(self):
        return "{}, bit={}, granularity={}, scale={}, zeropoint={}".format(
            self.type, self.bit, self.granularity,self.scale,self.zero_point
        )

    def update(self, x):
        self.observer.update(x)
