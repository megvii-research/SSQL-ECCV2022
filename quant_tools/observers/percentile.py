import torch
import numpy as np
import importlib

from .basic_modules import Observer as BaseObserver


class Observer(BaseObserver):
    def __init__(self, config, c_axis=1):
        super(Observer, self).__init__(config, c_axis)
        self.alpha = config.OBSERVER_METHOD.ALPHA
        self.channel = None
        self.length = None
        self.alpha_feature = []

    def init_params(self):
        if self.granularity == "channelwise":
            self.length = int((self.feature_length * self.alpha + 0.5) / self.channel)
        elif self.granularity == "layerwise":
            self.length = int(self.feature_length * self.alpha + 0.5)

        self.isnt_cut_percentile = self.length == 0
        self.length = self.length if self.length else 1

    def observe(self, value):
        value = abs(
            value.transpose(self.c_axis, 0)
            .reshape(value.shape[self.c_axis], -1)
            .detach()
            .cpu()
        )
        if not self.channel:
            self.channel = value.shape[0]
        if not self.length:
            self.init_params()
        if self.granularity == "channelwise":
            for i in range(self.channel):
                max_k, max_index = torch.topk(value[i], self.length, largest=True)
                if len(self.alpha_feature) < self.channel:
                    self.alpha_feature.append(max_k)
                else:
                    tmp_ = torch.cat((max_k, self.alpha_feature[i]))
                    max_k, max_index = torch.topk(
                        tmp_.reshape(-1), self.length, largest=True
                    )
                    self.alpha_feature[i] = max_k
                    del tmp_
        elif self.granularity == "layerwise":
            max_k, max_index = torch.topk(value.reshape(-1), self.length, largest=True)
            if len(self.alpha_feature) == 0:
                self.alpha_feature.append(max_k)
            else:
                assert len(max_k) == len(self.alpha_feature[0])
                tmp_ = torch.cat((max_k, self.alpha_feature[0]))
                max_k, max_index = torch.topk(
                    tmp_.reshape(-1), self.length, largest=True
                )
                self.alpha_feature[0] = max_k
                del tmp_
        del max_k, max_index

    def _get_percentile_min_max(self, channel=None):
        if self.granularity == "channelwise":
            min_val = self._actual_cmin[channel]
            max_val = self._actual_cmax[channel]
            alpha_feature = self.alpha_feature[channel]
        elif self.granularity == "layerwise":
            min_val = self._actual_cmin.min()
            max_val = self._actual_cmax.max()
            alpha_feature = self.alpha_feature[0]
        if max_val - min_val < 1e-10 or self.isnt_cut_percentile:
            return min_val, max_val

        if max_val > 0 and max_val > alpha_feature[-1]:
            max_val = alpha_feature[-1]
        if min_val < 0 and min_val < -alpha_feature[-1]:
            min_val = -alpha_feature[-1]
        new_min = min_val
        new_max = max_val

        return new_min, new_max

    def update(self, data):
        self.observe(data)
        if self.granularity == "channelwise":
            _min = torch.empty(self.channel)
            _max = torch.empty(self.channel)
            for i in range(self.channel):
                _min[i], _max[i] = torch.Tensor(self._get_percentile_min_max(i))
            self.max_val = _max.to(self.device)
            self.min_val = _min.to(self.device)
        elif self.granularity == "layerwise":
            _min, _max = self._get_percentile_min_max()
            self.max_val = torch.Tensor([_max]).repeat(self.channel).to(self.device)
            self.min_val = torch.Tensor([_min]).repeat(self.channel).to(self.device)
