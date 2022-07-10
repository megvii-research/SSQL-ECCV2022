import importlib
from meters.average import METER as avg_meter


class BaseLoss(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.avg_meter = avg_meter(config)
        self.name = ""

    def update(self, val, batch_size):
        self.avg_meter.update(val, batch_size)

    def reset(self):
        self.avg_meter.reset()

    def __repr__(self):
        return "{name}: {meter.val:.3f}({meter.avg:.3f})\t".format(
            name=self.name,
            meter=self.avg_meter,
        )


class LOSS(BaseLoss):
    def __init__(self, config, model):
        super(LOSS, self).__init__(config, model)
        cfg_loss = config.TRAIN.LOSS
        self.criterion = importlib.import_module(
            "losses.criterions." + cfg_loss.CRITERION.NAME
        ).Criterion(config, model)
        if cfg_loss.REGULARIZER.NAME:
            self.regularizer = importlib.import_module(
                "losses.regularizers." + cfg_loss.REGULARIZER.NAME
            ).Regularizer(config, model)
            self._lambda = config.TRAIN.LOSS.LAMBDA
        else:
            self.regularizer = importlib.import_module(
                "losses.regularizers.basic"
            ).Regularizer(config, model)
            self._lambda = 0.0

    def __call__(self, pred, target):
        c_val = self.criterion(pred, target)
        r_val = self.regularizer(pred, target)
        val = c_val + self._lambda * r_val

        batch_size = pred[0].shape[0] if type(pred) is list else pred.shape[0]
        self.update(val.item(), batch_size=batch_size)

        return val

    def reset(self):
        self.avg_meter.reset()
        self.criterion.reset()
        self.regularizer.reset()

    def __repr__(self):
        return (
            "Loss: {meter.val:.3f}({meter.avg:.3f})\t".format(meter=self.avg_meter)
            + str(self.criterion)
            + str(self.regularizer)
        )
