
import torch
import torch.nn as nn
from tqdm import tqdm
# import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import sys
sys.path.append("../../acr_quant")
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
    model, w_bit, q_config, quant_type="int", model_path=None
):  
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda").squeeze(0)
    zero = torch.tensor([0.0], dtype=torch.float32, device="cuda").squeeze(0)
    if quant_type == "int":
        layers = model.model.layers
        for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
            named_linears = get_named_linears(layers[i])
            for n, m in named_linears.items():
                # m.cuda()
                m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, **q_config)
                # m.cpu()
    elif "lsq" in quant_type:
        if quant_type == "lsq-n2f3" or quant_type == "clsq-n2f3" or quant_type == "clsq-n2f2":
            model_bnb = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            load_in_4bit=True,
                            device_map='auto',
                            quantization_config=BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type=quant_type,
                            ),
                            torch_dtype=torch.bfloat16
                        )
            layers_ori = model.model.layers
            layers_bnb = model_bnb.model.layers
            for i in tqdm(range(len(layers_ori)), desc=f"pseudo {quant_type} weight quantization..."):
                named_linears_ori = get_named_linears(layers_ori[i])
                named_linears_bnb = get_named_bnb_linears(layers_bnb[i])
                for ori_linear, bnb_linear in zip(named_linears_ori.items(), named_linears_bnb.items()):
                    module_ori = ori_linear[1]
                    quantizer = bnb_linear[1].weight_quantizer
                    module_ori.weight.data = quantizer(module_ori.weight.data)
            del model_bnb, layers_bnb
    elif "lsq" not in quant_type:
        if quant_type == "n2f4":
            quantizer = SteN2F4Quantizer(q_group_size=q_config["q_group_size"])
        elif quant_type == "nf3":
            quantizer = SteNF3Quantizer(bit=3, weight=None, q_group_size=q_config["q_group_size"])
        if quant_type == "nf4":
            quantizer = SteNF4Quantizer(bit=4, weight=None, q_group_size=q_config["q_group_size"])
        elif quant_type == "n2f3":
            print("quant_type: n2f3")
            quantizer = SteN2F3Quantizer(bit=3, weight=None, q_group_size=q_config["q_group_size"])
        elif quant_type == "n2f2":
            print("quant_type: n2f2")
            quantizer = SteN2F2Quantizer(bit=2, weight=None, q_group_size=32)
        elif quant_type == "int3-sym":
            quantizer = SteInt3SymQuantizer(q_group_size=q_config["q_group_size"])
        elif quant_type == "int4-asym":
            quantizer = SteInt4AsymQuantizer(q_group_size=q_config["q_group_size"])
        elif quant_type == "int3-asym":
            quantizer = SteInt3AsymQuantizer(q_group_size=q_config["q_group_size"])
        elif quant_type == "int2-asym":
            quantizer = SteInt2AsymQuantizer(q_group_size=q_config["q_group_size"])
            
        layers = model.model.layers
        for i in tqdm(range(len(layers)), desc=f"pseudo {quant_type} weight quantization..."):
            named_linears = get_named_linears(layers[i])
            for n, m in named_linears.items():
                # m.cuda()
                m.weight.data = quantizer(m.weight.data)