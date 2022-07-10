import torch
from yacs.config import CfgNode as CN

_C = CN()
_C.BASE_CONFIG = ""  # will use this file as base config and cover self's unique changes
_C.DEVICE = None
_C.RANK = 0
_C.WORLD_SIZE = 1
_C.LOCAL_RANK = 0
_C.SEED = None
_C.OUTPUT = ""

_C.EVALUATOR = ""
_C.EVALUATION_DOMAIN = None  # support 'float'/'bn_merged'/'quant'

_C.MODEL = CN()
_C.MODEL.ARCH = ""
_C.MODEL.CHECKPOINT = ""
_C.MODEL.PRETRAINED = ""
_C.MODEL.NUM_CLASSES = 0
_C.MODEL.INPUTSHAPE = [-1, -1]  # h, w

_C.TRAIN = CN()
_C.TRAIN.USE_DDP = False
_C.TRAIN.SYNC_BN = False
_C.TRAIN.LINEAR_EVAL = False
_C.TRAIN.WARMUP_FC = False # load from a linear evaluation model
_C.TRAIN.RESUME = ""
_C.TRAIN.EPOCHS = 0
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.PRETRAIN = ""
_C.TRAIN.DATASET = ""
_C.TRAIN.LABEL_SMOOTHING = 0.0
_C.TRAIN.BATCH_SIZE = 1  # per-gpu
_C.TRAIN.NUM_WORKERS = 0
_C.TRAIN.PRINT_FREQ = 1

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.TYPE = None  # support cosine / step / multiStep
_C.TRAIN.LR_SCHEDULER.WARMUP_EPOCHS = 0
_C.TRAIN.LR_SCHEDULER.WARMUP_LR = 0.0
_C.TRAIN.LR_SCHEDULER.BASE_LR = 0.0
_C.TRAIN.LR_SCHEDULER.FC_LR = 0.0 # specific learning rate for final fc layer
_C.TRAIN.LR_SCHEDULER.MIN_LR = 0.0
_C.TRAIN.LR_SCHEDULER.SPECIFIC_LRS = []
_C.TRAIN.LR_SCHEDULER.DECAY_MILESTONES = []
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCH = 0
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = ""
_C.TRAIN.OPTIMIZER.EPS = None
_C.TRAIN.OPTIMIZER.BETAS = None  # (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.0
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 0.0

_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.CRITERION = CN()
_C.TRAIN.LOSS.REGULARIZER = CN()
_C.TRAIN.LOSS.LAMBDA = 0.0  # a ratio controls the importance of regularizer
_C.TRAIN.LOSS.CRITERION.NAME = ""  # support CrossEntropy / LpLoss
_C.TRAIN.LOSS.CRITERION.LPLOSS = CN()
_C.TRAIN.LOSS.CRITERION.LPLOSS.P = 2.0
_C.TRAIN.LOSS.CRITERION.LPLOSS.REDUCTION = "none"
_C.TRAIN.LOSS.REGULARIZER.NAME = ""  # support PACT/SLIMMING

_C.TRAIN.RUNNER = CN()
_C.TRAIN.RUNNER.NAME = ""  # support default

_C.TRAIN.METER = CN()
_C.TRAIN.METER.NAME = ""  # support ACC / MAP / MIOU
_C.TRAIN.METER.ACC = CN()
_C.TRAIN.METER.ACC.TOPK = []
_C.TRAIN.METER.MAP = CN()
_C.TRAIN.METER.MIOU = CN()

_C.AUG = CN()
_C.AUG.TRAIN = CN()
_C.AUG.TRAIN.RANDOMRESIZEDCROP = CN()
_C.AUG.TRAIN.RANDOMRESIZEDCROP.ENABLE = False
# _C.AUG.TRAIN.RANDOMRESIZEDCROP.SIZE = MODEL.INPUT_SHAPE
_C.AUG.TRAIN.RANDOMRESIZEDCROP.SCALE = (0.08, 1.0)
_C.AUG.TRAIN.RANDOMRESIZEDCROP.INTERPOLATION = "bilinear"
_C.AUG.TRAIN.RESIZE = CN()
_C.AUG.TRAIN.RESIZE.ENABLE = False
_C.AUG.TRAIN.RESIZE.SIZE = (-1, -1)  # h, w
_C.AUG.TRAIN.RESIZE.KEEP_RATIO = True
_C.AUG.TRAIN.RESIZE.INTERPOLATION = "bilinear"
_C.AUG.TRAIN.HORIZONTAL_FLIP = CN()
_C.AUG.TRAIN.HORIZONTAL_FLIP.PROB = 0.0
_C.AUG.TRAIN.VERTICAL_FLIP = CN()
_C.AUG.TRAIN.VERTICAL_FLIP.PROB = 0.0
_C.AUG.TRAIN.RANDOMCROP = CN()
_C.AUG.TRAIN.RANDOMCROP.ENABLE = False
# _C.AUG.TRAIN.RANDOMCROP.SIZE = MODEL.INPUT_SHAPE
_C.AUG.TRAIN.RANDOMCROP.PADDING = 0
_C.AUG.TRAIN.CENTERCROP = CN()
_C.AUG.TRAIN.CENTERCROP.ENABLE = False
_C.AUG.TRAIN.COLOR_JITTER = CN()
_C.AUG.TRAIN.COLOR_JITTER.PROB = 0.0
_C.AUG.TRAIN.COLOR_JITTER.BRIGHTNESS = 0.4
_C.AUG.TRAIN.COLOR_JITTER.CONTRAST = 0.4
_C.AUG.TRAIN.COLOR_JITTER.SATURATION = 0.2
_C.AUG.TRAIN.COLOR_JITTER.HUE = 0.1
_C.AUG.TRAIN.AUTO_AUGMENT = CN()
_C.AUG.TRAIN.AUTO_AUGMENT.ENABLE = False
_C.AUG.TRAIN.AUTO_AUGMENT.POLICY = 0.0
_C.AUG.TRAIN.RANDOMERASE = CN()
_C.AUG.TRAIN.RANDOMERASE.PROB = 0.0
_C.AUG.TRAIN.RANDOMERASE.MODE = "const"
_C.AUG.TRAIN.RANDOMERASE.MAX_COUNT = None
_C.AUG.TRAIN.MIX = CN()  # mixup & cutmix
_C.AUG.TRAIN.MIX.PROB = 0.0
_C.AUG.TRAIN.MIX.MODE = "batch"
_C.AUG.TRAIN.MIX.SWITCH_MIXUP_CUTMIX_PROB = 0.0
_C.AUG.TRAIN.MIX.MIXUP_ALPHA = 0.0
_C.AUG.TRAIN.MIX.CUTMIX_ALPHA = 0.0
_C.AUG.TRAIN.MIX.CUTMIX_MIXMAX = None
_C.AUG.TRAIN.NORMLIZATION = CN()
_C.AUG.TRAIN.NORMLIZATION.MEAN = []
_C.AUG.TRAIN.NORMLIZATION.STD = []
_C.AUG.EVALUATION = CN()
_C.AUG.EVALUATION.RESIZE = CN()
_C.AUG.EVALUATION.RESIZE.ENABLE = False
_C.AUG.EVALUATION.RESIZE.SIZE = (-1, -1)  # h, w
_C.AUG.EVALUATION.RESIZE.KEEP_RATIO = True
_C.AUG.EVALUATION.RESIZE.INTERPOLATION = "bilinear"
_C.AUG.EVALUATION.CENTERCROP = CN()
_C.AUG.EVALUATION.CENTERCROP.ENABLE = False
_C.AUG.EVALUATION.NORMLIZATION = CN()
_C.AUG.EVALUATION.NORMLIZATION.MEAN = []
_C.AUG.EVALUATION.NORMLIZATION.STD = []

_C.QUANT = CN()
_C.QUANT.TYPE = ""  # support 'qat' / ptq
_C.QUANT.BIT_ASSIGNER = CN()
_C.QUANT.BIT_ASSIGNER.NAME = None  # support 'HAWQ'
_C.QUANT.BIT_ASSIGNER.W_BIT_CHOICES = [2, 4, 8]
_C.QUANT.BIT_ASSIGNER.A_BIT_CHOICES = [2, 4, 8, 16]

_C.QUANT.BIT_ASSIGNER.HAWQ = CN()
_C.QUANT.BIT_ASSIGNER.HAWQ.EIGEN_TYPE = "avg"  # support 'max' / 'avg'
_C.QUANT.BIT_ASSIGNER.HAWQ.SENSITIVITY_CALC_ITER_NUM = 50
_C.QUANT.BIT_ASSIGNER.HAWQ.LIMITATION = CN()
_C.QUANT.BIT_ASSIGNER.HAWQ.LIMITATION.BIT_ASCEND_SORT = False
_C.QUANT.BIT_ASSIGNER.HAWQ.LIMITATION.BIT_WIDTH_COEFF = 1e10
_C.QUANT.BIT_ASSIGNER.HAWQ.LIMITATION.BOPS_COEFF = 1e10
_C.QUANT.BIT_CONFIG = (
    []
)  # a mapping, key is layer_name, value is {"w":w_bit, "a":a_bit}
_C.QUANT.FOLD_BN = False
_C.QUANT.W = CN()
_C.QUANT.W.QUANTIZER = None  # support "LSQ" / "DOREFA"
_C.QUANT.W.BIT = 8
_C.QUANT.W.BIT_RANGE = [2,9] # left include, right exclude, default 2~8
_C.QUANT.W.SYMMETRY = True
_C.QUANT.W.GRANULARITY = (
    "channelwise"  # support "layerwise"/"channelwise" currently, default is channelwise
)
_C.QUANT.W.OBSERVER_METHOD = CN()
_C.QUANT.W.OBSERVER_METHOD.NAME = (
    "MINMAX"  # support "MINMAX"/"MSE" currently, default is MINMAX
)
_C.QUANT.W.OBSERVER_METHOD.ALPHA = 0.0001  # support percentile
_C.QUANT.W.OBSERVER_METHOD.BINS = 2049  # support kl_histogram
_C.QUANT.A = CN()
_C.QUANT.A.BIT = 8
_C.QUANT.A.BIT_RANGE = [4,9] # left include, right exclude, default 4~8
_C.QUANT.A.QUANTIZER = None  # support "LSQ" / "DOREFA"
_C.QUANT.A.SYMMETRY = False
_C.QUANT.A.GRANULARITY = (
    "layerwise"  # support "layerwise"/"channelwise" currently, default is layerwise
)
_C.QUANT.A.OBSERVER_METHOD = CN()
_C.QUANT.A.OBSERVER_METHOD.NAME = (
    "MINMAX"  # support "MINMAX"/"MSE" currently, default is MINMAX
)
_C.QUANT.A.OBSERVER_METHOD.ALPHA = 0.0001  # support percentile
_C.QUANT.A.OBSERVER_METHOD.BINS = 2049  # support kl_histogram
_C.QUANT.CALIBRATION = CN()
_C.QUANT.CALIBRATION.PATH = ""
_C.QUANT.CALIBRATION.TYPE = ""  # support tarfile / python_module
_C.QUANT.CALIBRATION.MODULE_PATH = ""  # the import path of calibration dataset
_C.QUANT.CALIBRATION.SIZE = 0
_C.QUANT.CALIBRATION.BATCHSIZE = 1
_C.QUANT.CALIBRATION.NUM_WORKERS = 0

_C.QUANT.FINETUNE = CN()
_C.QUANT.FINETUNE.ENABLE = False
_C.QUANT.FINETUNE.METHOD = ""
_C.QUANT.FINETUNE.BATCHSIZE = 32
_C.QUANT.FINETUNE.ITERS_W = 0
_C.QUANT.FINETUNE.ITERS_A = 0

_C.QUANT.FINETUNE.BRECQ = CN()
_C.QUANT.FINETUNE.BRECQ.KEEP_GPU = True


_C.SSL = CN()
_C.SSL.TYPE = None # support mocov2
_C.SSL.SETTING = CN()
_C.SSL.SETTING.DIM = 128 # output dimension for the MLP head
_C.SSL.SETTING.HIDDEN_DIM = 2048 # hidden dimension for the MLP head

_C.SSL.SETTING.T = 0.07 # temperature for InfoNCE loss
_C.SSL.SETTING.MOCO_K = 65536 # size of memory bank for MoCo
_C.SSL.SETTING.MOMENTUM = 0.999 # MoCo momentum of updating key encoder
_C.SSL.SETTING.MLP = True # whether to use MLP head, default True


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()

    config.defrost()
    if hasattr(args, "checkpoint"):
        config.MODEL.CHECKPOINT = args.checkpoint
    if hasattr(args, "pretrained"):
        config.MODEL.PRETRAINED = args.pretrained
    if hasattr(args, "calibration"):
        config.QUANT.CALIBRATION.PATH = args.calibration
    if hasattr(args, "batch_size"):
        config.QUANT.CALIBRATION.BATCHSIZE = args.batch_size
    if hasattr(args, "num_workers"):
        config.QUANT.CALIBRATION.NUM_WORKERS = args.num_workers
        config.TRAIN.NUM_WORKERS = args.num_workers
    if hasattr(args, "eval_domain"):
        config.EVALUATION_DOMAIN = args.eval_domain
    if hasattr(args, "print_freq"):
        config.TRAIN.PRINT_FREQ = args.print_freq
    if hasattr(args, "output"):
        config.OUTPUT = args.output

    config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # get config depend chain files recursively
    config_depends = []

    tmp_config = _C.clone()
    tmp_config.defrost()
    tmp_config.merge_from_file(args.config)
    config_depends.append(args.config)
    while tmp_config.BASE_CONFIG:
        next_config = tmp_config.BASE_CONFIG
        config_depends.append(next_config)
        tmp_config.BASE_CONFIG = ""
        tmp_config.merge_from_file(next_config)
    # tmp_config's merge order is reversed so can't use it directly

    for conf_path in reversed(config_depends):
        config.merge_from_file(conf_path)

    config.freeze()

    return config


def update_config(config, key, value):
    config.defrost()
    keys = key.split(".")

    def _set_config_attr(cfg, keys, value):
        if len(keys) > 1:
            cfg = getattr(cfg, keys[0].upper())
            _set_config_attr(cfg, keys[1:], value)
        else:
            setattr(cfg, keys[0].upper(), value)

    _set_config_attr(config, keys, value)
    config.freeze()
    return config
