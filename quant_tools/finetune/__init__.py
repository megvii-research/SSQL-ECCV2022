from .brecq import finetune as brecq_finetune


FACTORY = {
    "brecq": brecq_finetune,
}
