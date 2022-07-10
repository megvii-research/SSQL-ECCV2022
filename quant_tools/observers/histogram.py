import torch
import numpy as np
import importlib

from .basic_modules import Observer as BaseObserver


class Observer(BaseObserver):
    def __init__(self, config, c_axis=1):
        super(Observer, self).__init__(config, c_axis)

    def update(self, data):  # , c_axis=1, n_bits=8):
        from functools import partial
        import multiprocessing as mp

        data_c_first = data.transpose(self.c_axis, 0).reshape(
            data.shape[self.c_axis], -1
        )
        if self.granularity == "layerwise":
            if self.max_val is None:
                self.max_val, self.min_val = get_best_max_min_val(
                    data_c_first, n_bits=self.bit, mode="torch"
                )
            else:
                pass
                # FIXME: only pick a batch of data to do calibration
                # use first batch instead of last batch
        elif self.granularity == "channelwise":
            if self.max_val is None:
                self.max_val = data_c_first.max(axis=1).values
                self.min_val = data_c_first.min(axis=1).values
                func = partial(get_best_max_min_val, n_bits=self.bit, mode="numpy")
                cpu_input = list(data_c_first.detach().cpu().numpy())
                with mp.Pool(processes=4) as p:
                    # use processes=None will degenerate to single process in rlaunch
                    # but processes=None will be faster than processes=4 without rlaunch
                    output = []
                    for c in range(data_c_first.shape[0]):
                        output.append(p.apply_async(func, (cpu_input[c],)))
                    p.close()
                    p.join()
                for c in range(data_c_first.shape[0]):
                    out = output[c].get()
                    self.max_val[c] = float(out[0])
                    self.min_val[c] = float(out[1])
            else:
                pass
                # FIXME: only pick a batch of data to do calibration
                # use first batch instead of last batch
        else:
            raise NotImplementedError("no support {}".format(self.granularity))


def get_best_max_min_val(data, n_bits, mode="numpy"):
    # a copy for multiprocess, and implements numpy version
    def lp_loss(pred, tgt, p=2.0, reduction="none"):
        """
        loss function measured in L_p Norm
        """
        if mode == "numpy":
            pow_res = np.abs(pred - tgt) ** p
        elif mode == "torch":
            pow_res = (pred - tgt).abs().pow(p)
        if reduction == "none":
            return pow_res.sum(1).mean()
        else:
            return pow_res.mean()

    def quantize(data, n_bits, max, min):
        delta = (max - min) / (2 ** n_bits - 1)
        zero_point = (-min / delta + 0.5) // 1
        # we assume quantization is always signed
        x_int = (data / delta + 0.5) // 1
        if mode == "numpy":
            x_quant = np.clip(x_int + zero_point, 0, 2 ** n_bits - 1)
        elif mode == "torch":
            x_quant = (x_int + zero_point).clamp(0, 2 ** n_bits - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    x_max = data.max()
    x_min = data.min()
    best_score = 1e10
    best_max = x_max
    best_min = x_min

    def calc_percentile(p):
        new_max = x_max * (1.0 - (p / 100))
        new_min = x_min * (1.0 - (p / 100))
        data_q = quantize(data, n_bits, new_max, new_min)
        score = lp_loss(data, data_q, p=2.4, reduction="all")
        return score, new_max, new_min

    L = 0
    R = 80

    for i in range(15):
        mid = (L + L + R) / 3
        midmid = (L + R + R) / 3
        mid_score, mid_max, mid_min = calc_percentile(mid)
        midmid_score, midmid_max, midmid_min = calc_percentile(midmid)
        if mid_score < midmid_score:
            R = midmid
            if mid_score < best_score:
                best_score = mid_score
                best_max = mid_max
                best_min = mid_min
        else:
            L = mid
            if midmid_score < best_score:
                best_score = midmid_score
                best_max = midmid_max
                best_min = midmid_min
    return best_max, best_min
