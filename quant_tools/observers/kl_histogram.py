import torch
import numpy as np
import importlib
import copy
from scipy import stats

from .basic_modules import Observer as BaseObserver


class Observer(BaseObserver):
    def __init__(self, config, c_axis=1):
        super(Observer, self).__init__(config, c_axis)
        self.channel = None
        self.bins = config.OBSERVER_METHOD.BINS
        self.histogram = None

    def init_params(self):
        if self.granularity == "channelwise":
            self.histogram = np.zeros([self.channel, self.bins])
        elif self.granularity == "layerwise":
            self.histogram = np.zeros([self.bins])

    def observe(self, value):
        self.device = value.device
        value = abs(
            value.transpose(self.c_axis, 0)
            .reshape(value.shape[self.c_axis], -1)
            .detach()
            .cpu()
        )
        if not self.channel:
            self.channel = value.shape[0]
        if self.histogram is None:
            self.init_params()
        if self.granularity == "channelwise":
            for i in range(self.channel):
                th = float(
                    max(abs(self._actual_cmin[i]), abs(self._actual_cmax[i]))
                    .detach()
                    .cpu()
                    .numpy()
                )
                self.histogram[i] += torch.histc(
                    value[i], bins=self.bins, min=-th, max=th
                ).numpy()
        elif self.granularity == "layerwise":
            th = float(
                max(abs(self._actual_cmin.min()), abs(self._actual_cmax.max()))
                .detach()
                .cpu()
                .numpy()
            )
            self.histogram += torch.histc(
                value, bins=self.bins, min=-th, max=th
            ).numpy()

    def calibrate_entropy(self, distribution, bin_width, num_quantized_bins=255):
        zero_bin_idx = self.bins // 2
        num_half_quantized_bins = num_quantized_bins // 2
        # thresholds = np.zeros([self.bins // 2 + 1 - num_quantized_bins // 2])
        divergence = np.zeros([self.bins // 2 + 1 - num_quantized_bins // 2])
        for i in range(num_half_quantized_bins, zero_bin_idx):
            p_bin_idx_start = zero_bin_idx - i
            p_bin_idx_stop = zero_bin_idx + i + 1
            sliced_nd_hist = np.zeros([p_bin_idx_stop - p_bin_idx_start])
            p = copy.deepcopy(distribution[p_bin_idx_start:p_bin_idx_stop])
            p[0] += sum(distribution[:p_bin_idx_start])
            p[p_bin_idx_stop - p_bin_idx_start - 1] = sum(distribution[p_bin_idx_stop:])
            sliced_nd_hist = copy.deepcopy(distribution[p_bin_idx_start:p_bin_idx_stop])
            num_merged_bins = sliced_nd_hist.size // num_quantized_bins
            quantized_bins = np.zeros([num_quantized_bins])
            for j in range(num_quantized_bins):
                start = j * num_merged_bins
                stop = start + num_merged_bins
                quantized_bins[j] = sliced_nd_hist[start:stop].sum()
            quantized_bins[-1] += sliced_nd_hist[
                num_quantized_bins * num_merged_bins :
            ].sum()
            is_nonzeros = (p != 0).astype(np.int64)
            q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
            for j in range(num_quantized_bins):
                start = j * num_merged_bins
                if j == num_quantized_bins - 1:
                    stop = -1
                else:
                    stop = start + num_merged_bins
                norm = is_nonzeros[start:stop].sum()
                if norm != 0:
                    q[start:stop] = float(quantized_bins[j]) / float(norm)
            q[p == 0] = 0
            p[p == 0] = 0.0001
            q[q == 0] = 0.0001

            divergence[i - num_quantized_bins] = stats.entropy(p, q)
        min_kl_divergence = np.argmin(divergence)
        th = bin_width * min_kl_divergence
        return th

    def get_optimal_threshold(self, channel=None):
        target_bin = 2 ** self.bit - 1
        if self.granularity == "channelwise":
            min_val = self._actual_cmin[channel]
            max_val = self._actual_cmax[channel]
            distribution: np.ndarray = self.histogram[channel]
        elif self.granularity == "layerwise":
            min_val = self._actual_cmin.min()
            max_val = self._actual_cmax.max()
            distribution: np.ndarray = self.histogram
        if max_val - min_val < 1e-10:
            return min_val, max_val
        assert distribution.shape[0] == self.bins, "bins mistmatch"
        if min_val >= 0:
            target_bin = target_bin * 2 + 1

        bin_width = 2 * max(abs(min_val), abs(max_val)) / self.bins
        th = self.calibrate_entropy(distribution, bin_width, target_bin)
        if min_val >= 0:
            new_min = 0
            new_max = th
        else:
            new_min = -th
            new_max = th
        return new_min, new_max

    def update(self, data):
        self.observe(data)
        if self.granularity == "channelwise":
            _min = torch.empty(self.channel)
            _max = torch.empty(self.channel)
            for i in range(self.channel):
                _min[i], _max[i] = torch.Tensor(self.get_optimal_threshold(i))
            self.max_val = _max.to(self.device)
            self.min_val = _min.to(self.device)
        elif self.granularity == "layerwise":
            _min, _max = self.get_optimal_threshold()
            self.max_val = torch.Tensor([_max]).repeat(self.channel).to(self.device)
            self.min_val = torch.Tensor([_min]).repeat(self.channel).to(self.device)
