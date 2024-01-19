
import torch
import torch.nn as nn
from tqdm import tqdm
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import sys
sys.path.append("../../quantization")
from quantizer import *

# from ppq.core import CUDA
def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def get_named_bnb_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, bnb.nn.Linear4bit)}

# core quantization method (simulated quantization)
def pseudo_quantize_tensor(w, n_bit=8,
                           zero_point=True, q_group_size=-1,
                           inplace=False,
                           get_scale_zp=False
                           ):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    elif q_group_size == -1:
        w = w.reshape(-1, w.shape[-1])
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = - 2 ** (n_bit - 1)
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        ((w.div_(scales).round_().add_(zeros)).clamp_(
            min_int, max_int).sub_(zeros)).mul_(scales)
    else:
        w = (torch.clamp(torch.round(w / scales) +
                         zeros, min_int, max_int) - zeros) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w

@torch.no_grad()
def pseudo_quantize_model_weight(
    model, w_bit, q_config, quant_type="int"
):  
    if quant_type == "int":
        layers = model.model.layers
        for i in tqdm(range(len(layers)), desc=f"pseudo {quant_type} weight quantization..."):
            named_linears = get_named_linears(layers[i])
            for n, m in named_linears.items():
                m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, **q_config)
    
    elif quant_type == "nf3":
        quantizer = SteN2F3Quantizer(q_group_size=q_config["q_group_size"])   
        layers = model.model.layers
        for i in tqdm(range(len(layers)), desc=f"pseudo {quant_type} weight quantization..."):
            named_linears = get_named_linears(layers[i])
            for n, m in named_linears.items():
                # m.cuda()
                m.weight.data = quantizer(m.weight.data)