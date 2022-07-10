import torch
from .basic_modules import Observer as BaseObserver


class Observer(BaseObserver):
    def __init__(self, config, c_axis=1):
        super(Observer, self).__init__(config, c_axis)

    def update(self, data):
        data_c_first = data.transpose(self.c_axis, 0).reshape(
            data.shape[self.c_axis], -1
        )
        if self.granularity == "layerwise":
            if self.max_val is None:
                self.max_val = data.max()
                self.min_val = data.min()
            else:
                self.max_val = torch.max(self.max_val, data.max())
                self.min_val = torch.min(self.min_val, data.min())
        elif self.granularity == "channelwise":
            if self.max_val is None:
                self.max_val = data_c_first.max(axis=1).values
                self.min_val = data_c_first.min(axis=1).values
            else:
                self.max_val = torch.max(self.max_val, data_c_first.max(axis=1).values)
                self.min_val = torch.min(self.min_val, data_c_first.min(axis=1).values)
        else:
            raise NotImplementedError("no support {}".format(self.granularity))
