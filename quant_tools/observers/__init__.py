from .basic_modules import Observer as BaseObserver
from .minmax import Observer as MinMaxObserver
from .mse import Observer as MSEObserver
from .histogram import Observer as HistogramObserver
from .percentile import Observer as PercentileObserver
from .kl_histogram import Observer as KLHisogramObserver
from .momentum_minmax import Observer as MomentumMinMaxObserver
from .gaussian import Observer as GaussianObserver


def build_observer(config, c_axis):
    return {
        "BASIC": BaseObserver,
        "MSE": MSEObserver,
        "MINMAX": MinMaxObserver,
        "HISTOGRAM": HistogramObserver,
        "PERCENTILE": PercentileObserver,
        "KL_HISTOGRAM": KLHisogramObserver,
        "MOMENTUM_MINMAX": MomentumMinMaxObserver,
        "GAUSSIAN": GaussianObserver,
    }[config.OBSERVER_METHOD.NAME](config, c_axis)
