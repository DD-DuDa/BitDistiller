import torch
import torch.nn as nn
import gc
import argparse
import os
import sys
from clip_utils import *
from quantizer import pseudo_quantize_tensor, pseudo_quantize_n2f3_tensor
from tqdm import tqdm
from collections import defaultdict
import functools

@torch.no_grad()
def auto_2clip_layer(w, input_feat, n_bit, q_config,
                    n_grid=20,
                    max_shrink=0.5,
                    n_sample_token=512):
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]

    group_size = q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]

    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0::input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []
    best_min_val_all = []
    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size: (i_b + 1) * oc_batch_size]

        org_max_val = w.amax(dim=-1, keepdim=True)  # co, 1, n_group, 1
        org_min_val = w.amin(dim=-1, keepdim=True)

        best_max_val = org_max_val.clone()
        best_min_val = org_min_val.clone()

        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

        for i_s_p in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s_p / n_grid)
            for i_s_n in range(int(max_shrink * n_grid)):
                min_val = org_min_val * (1 - i_s_n / n_grid)
                # min_val = - max_val
                cur_w = torch.clamp(w, min_val, max_val)
                if q_config["quant_type"] == "int":
                    q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, zero_point=True, q_group_size=q_config['q_group_size'])
                elif q_config["quant_type"] == "nf3":
                    q_w = pseudo_quantize_n2f3_tensor(cur_w, q_group_size=q_config['q_group_size'])
                else:
                    quant_type = q_config["quant_type"]
                    raise ValueError(f"Has no support {quant_type}. Valid quant_type:[int, nf3]")
                    
                cur_out = (input_feat * q_w).sum(dim=-1)

                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)

                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
                best_min_val[cur_best_idx] = min_val[cur_best_idx]

        best_max_val_all.append(best_max_val)
        best_min_val_all.append(best_min_val)
    print("loss:", err.mean().item())
    best_max_val = torch.cat(best_max_val_all, dim=0)
    best_min_val = torch.cat(best_min_val_all, dim=0)
    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    return best_max_val.squeeze(1), best_min_val.squeeze(1)

    
@torch.no_grad()
def auto_clip_block(module,
                    w_bit, q_config,
                    input_feat):

    named_linears = {name: m for name,
                     m in module.named_modules() if isinstance(m, nn.Linear)}

    clip_list = []
    for name in named_linears:
        # due to qk bmm, it is hard to clip precisely
        if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
            continue
        named_linears[name].cuda()

        max_val, min_val = auto_2clip_layer(
            named_linears[name].weight, input_feat[name], n_bit=w_bit, q_config=q_config)

        clip_list.append((name, max_val, min_val))

        named_linears[name].cpu()
    return clip_list

@torch.no_grad()
def run_clip(
    model, enc,
    w_bit, q_config,
    n_samples=128, seqlen=1024,
    datasets="pile"
):
    print(f"Using {datasets} dataset to do calibation")
    samples = get_calib_dataset(
              datasets=datasets, tokenizer=enc, n_samples=n_samples, block_size=seqlen)
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}
    
    layers = get_blocks(model)

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")
    
    gc.collect()
    torch.cuda.empty_cache()

    clip_results = {
        "clip": [],
    }

    # solve layer by layer
    for i in tqdm(range(len(layers)), desc="Running Asymmetric Clipping..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name,
                                  feat_dict=input_feat)))
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()

        # now solve for clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        clip_list = auto_clip_block(layer,
                                    w_bit=w_bit, q_config=q_config,
                                    input_feat=input_feat)
        
        apply_clip(layer, clip_list)
        # append prefix to make names global
        clip_results["clip"] += append_str_prefix(clip_list, get_op_name(model, layer) + ".")

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()
        
    return clip_results


def main(args, q_config):
    if args.dump_clip and os.path.exists(args.dump_clip):
        print(f"Found existing AWQ results {args.dump_clip}, exit.")
        exit()

    model, enc = build_model_and_enc(args.model_path)

    if args.run_clip:
        assert args.dump_clip, "Please save the awq results with --dump_awq"

        clip_results = run_clip(
            model, enc,
            w_bit=args.w_bit, q_config=q_config,
            n_samples=args.n_samples, seqlen=args.seqlen, datasets=args.calib_dataset
        )

        if args.dump_clip:
            dirpath = os.path.dirname(args.dump_clip)
            os.makedirs(dirpath, exist_ok=True)
            
            torch.save(clip_results, args.dump_clip)
            print("Clipping results saved at", args.dump_clip)
            
        exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path of the hf model')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--w_bit', type=int, default=2, help='bits of weight')
    parser.add_argument('--q_group_size', type=int, default=128)
    parser.add_argument('--quant_type', type=str, default="int", help="quant_type", choices=["int", "nf3"])
    parser.add_argument('--no_zero_point', action='store_true',
                        help="disable zero_point")
    parser.add_argument('--run_clip', action='store_true',
                        help="perform asym-clipping search process")
    parser.add_argument('--dump_clip', type=str, default=None,
                        help="save the asym-clipping search results")
    parser.add_argument('--n_samples', type=int, default=128,
                        help="Number of calibration data samples.")
    parser.add_argument('--seqlen', type=int, default=1024,
                        help="Length usage of each calibration data.")
    parser.add_argument("--calib_dataset", type=str, default="pile",
            choices=["pile", "gsm8k","code"],
            help="Where to extract calibration data from.",
        )

    args = parser.parse_args()

    q_config = {
        "zero_point": not args.no_zero_point,   # by default True
        "q_group_size": args.q_group_size,      # whether to use group quantization
        "quant_type": args.quant_type
    }

    print("Quantization config:", q_config)
    main(args, q_config)