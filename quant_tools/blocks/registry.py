BLOCK_NAME_MAPPING = {}
NAME_QBLOCK_MAPPING = {}


def register_block(fn):
    prefix = ".".join(fn.__module__.split(".")[1:])
    BLOCK_NAME_MAPPING[fn] = ".".join([prefix, fn.__name__])
    return fn


def register_qblock(fn):
    prefix = ".".join(fn.__module__.split(".")[2:])
    NAME_QBLOCK_MAPPING[".".join([prefix, fn.__name__])] = fn
    return fn
