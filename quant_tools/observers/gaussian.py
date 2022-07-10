import torch
from .basic_modules import Observer as BaseObserver


class Observer(BaseObserver):
    def __init__(self, config, c_axis=1):
        super(Observer, self).__init__(config, c_axis)

    def update(self, data):
        data_c_first = data.transpose(self.c_axis, 0).reshape(
            data.shape[self.c_axis], -1
        )
        alpha = 2.0
        if self.granularity == "layerwise":
            mean_val = torch.mean(data)
            std_val = torch.std(data)
            self.min_val = mean_val - alpha*std_val
            self.max_val = mean_val + alpha*std_val
        elif self.granularity == "channelwise":
            mean_val = torch.mean(data_c_first, dim=1)
            std_val = torch.std(data_c_first, dim=1)
            self.min_val = mean_val - alpha*std_val
            self.max_val = mean_val + alpha*std_val
        else:
            raise NotImplementedError("no support {}".format(self.granularity))
        
        
