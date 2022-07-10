import torch
from .basic_modules import Observer as BaseObserver


class Observer(BaseObserver):
    def __init__(self, config, c_axis=1):
        super(Observer, self).__init__(config, c_axis)
        self.register_buffer("tmp_min_val", None)
        self.register_buffer("tmp_max_val", None)

    def reset_minmax(self):
        self.register_buffer("tmp_min_val", None)
        self.register_buffer("tmp_max_val", None)

    def normal_update(self, data):
        data_c_first = data.transpose(self.c_axis, 0).reshape(
            data.shape[self.c_axis], -1
        )
        if self.granularity == "layerwise":
            if self.tmp_max_val is None:
                self.tmp_max_val = data.max()
                self.tmp_min_val = data.min()
            else:
                self.tmp_max_val = torch.max(self.tmp_max_val, data.max())
                self.tmp_min_val = torch.min(self.tmp_min_val, data.min())
        elif self.granularity == "channelwise":
            if self.tmp_max_val is None:
                self.tmp_max_val = data_c_first.max(axis=1).values
                self.tmp_min_val = data_c_first.min(axis=1).values
            else:
                self.tmp_max_val = torch.max(self.tmp_max_val, data_c_first.max(axis=1).values)
                self.tmp_min_val = torch.min(self.tmp_min_val, data_c_first.min(axis=1).values)
        else:
            raise NotImplementedError("no support {}".format(self.granularity))

    def update(self, data, momentum=0.99):
        self.normal_update(data)
        if self.max_val is None:
            self.max_val = self.tmp_max_val
            self.min_val = self.tmp_min_val
        else:
            self.max_val = momentum*self.max_val + (1-momentum)*self.tmp_max_val
            self.min_val = momentum*self.min_val + (1-momentum)*self.tmp_min_val
        