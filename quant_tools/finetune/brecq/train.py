import torch
from .utils import save_grad_data, save_inp_oup_data


def train(
    model,
    block,
    cali_data,
    loss_func,
    optimizer,
    scheduler,
    batch_size,
    opt_mode,
    iters,
    act_quant,
    asym,
    keep_gpu: bool = True,
):
    # Save data before optimizing the rounding
    cached_inps, cached_outs = save_inp_oup_data(
        model, block, cali_data, asym, act_quant, batch_size, keep_gpu=keep_gpu
    )
    if opt_mode != "mse":
        cached_grads = save_grad_data(
            model, block, cali_data, act_quant, batch_size=batch_size, keep_gpu=keep_gpu
        )
    else:
        cached_grads = None
    device = next(model.parameters()).device
    for i in range(iters):
        idx = torch.randperm(cached_inps.size(0))[:batch_size]
        cur_inp = cached_inps[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        cur_grad = cached_grads[idx].to(device) if opt_mode != "mse" else None

        optimizer.zero_grad()
        out_quant = block(cur_inp)

        err = loss_func(out_quant, cur_out, cur_grad)
        err.backward(retain_graph=True)
        optimizer.step()
        if scheduler:
            scheduler.step()

    torch.cuda.empty_cache()
