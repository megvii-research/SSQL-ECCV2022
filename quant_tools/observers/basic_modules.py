import torch
from torch import nn


class Observer(nn.Module):
    def __init__(self, config, c_axis):
        super(Observer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.register_buffer("min_val", None)
        self.register_buffer("max_val", None)
        self.feature_length = None
        self._actual_cmin = None
        self._actual_cmax = None
        self.granularity = config.GRANULARITY
        self.is_symmetry = config.SYMMETRY
        self.bit = config.BIT
        self.c_axis = c_axis

    def reset_minmax(self):
        self.register_buffer("min_val", None)
        self.register_buffer("max_val", None)

    def set_bit(self, bit):
        self.bit = bit

    def update(self, data):
        pass

    def calc_init_params(self, data):
        value = data.transpose(self.c_axis, 0).reshape(data.shape[self.c_axis], -1)
        self.feature_length = (
            len(value.reshape(-1)) + self.feature_length
            if self.feature_length
            else len(value.reshape(-1))
        )
        v_min = value.min(dim=1).values
        v_max = value.max(dim=1).values
        self._actual_cmin = (
            torch.min(self._actual_cmin, v_min)
            if self._actual_cmin is not None
            else v_min
        )
        self._actual_cmax = (
            torch.max(self._actual_cmax, v_max)
            if self._actual_cmax is not None
            else v_max
        )

    def calc_quant_params(self):
        if self.is_symmetry:
            max_abs_value = torch.maximum(
                torch.abs(self.min_val), torch.abs(self.max_val)
            )
            scale = max_abs_value / float((2 ** (self.bit) - 1) / 2)  # 127.5
            
            # fix scale
            #min_abs = torch.abs(self.min_val / float((2 ** (self.bit) - 1) / 2))
            #max_abs = torch.abs(self.max_val  / float((2 ** (self.bit) - 1) / 2))
            #scale = torch.maximum(min_abs, max_abs)
            zero_point = torch.zeros(scale.shape).to(scale.device)
        else:
            scale = (self.max_val - self.min_val) / (2 ** self.bit - 1)
            zero_point = -torch.round(self.min_val / scale)
        return scale, zero_point


    def __repr__(self):
        if isinstance(self.min_val, (float, torch.Tensor)):
            return "min_value={}, max_value={}".format(
                self.max_val.reshape(-1), self.min_val.reshape(-1)
            )
        else:
            return "min_value & max_value is None"