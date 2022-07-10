BLOCK_NAME_MAPPING = {}


def register_block(fn):
    prefix = ".".join(fn.__module__.split(".")[1:])
    BLOCK_NAME_MAPPING[fn] = ".".join([prefix, fn.__name__])
    return fn
