# from .block_recon import recon_model
from .utils import get_data_samples
from .reconstruct import reconstruct_modules
from quant_tools.blocks import QuantBasic


def finetune(config, model, dataloader, evaluator):
    cali_data = get_data_samples(dataloader, config.QUANT.CALIBRATION.SIZE)

    model.set_quant_state(True, False, w_init=True)
    model.eval()
    print("accuracy after weight quant params init:")
    evaluator(model)

    def recog_modules(module, kwargs):
        for n, m in module.named_children():
            if isinstance(m, QuantBasic):
                print("Reconstruction for layer {}".format(n))
                reconstruct_modules(module=m, **kwargs)
            else:
                recog_modules(m, kwargs)

    if config.QUANT.W.QUANTIZER == "adaRound":
        kwargs = dict(
            model=model,
            cali_data=cali_data,
            batch_size=config.QUANT.FINETUNE.BATCHSIZE,
            iters=config.QUANT.FINETUNE.ITERS_W,
            asym=True,
            act_quant=False,
            opt_mode="mse",
            warmup=0.2,
            keep_gpu=config.QUANT.FINETUNE.BRECQ.KEEP_GPU,
        )
        recog_modules(model, kwargs)
        model.set_quant_state(True, False)
        model.eval()
        print("accuracy after weight recog:")
        evaluator(model)

    model.calibration(dataloader, config.QUANT.CALIBRATION.BATCHSIZE)
    model.set_quant_state(True, True, a_init=True)
    model.eval()
    print("accuracy after feature quant params init:")
    evaluator(model)

    if config.QUANT.A.QUANTIZER == "adaRound":
        kwargs = dict(
            model=model,
            cali_data=cali_data,
            iters=config.QUANT.FINETUNE.ITERS_A,
            act_quant=True,
            opt_mode="mse",
            lr=4e-4,
            p=2.4,
        )
        recog_modules(model, kwargs)
        model.set_quant_state(True, True)
        model.eval()
        print("accuracy after feature recog:")
        evaluator(model)

    return model
