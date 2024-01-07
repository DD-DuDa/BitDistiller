import os
import numpy as np
import torch
import torch.nn as nn
import time
from logging import getLogger
from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM
import sys
sys.path.append("../")
from test_utils import pseudo_quantize_model_weight


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    # traindata = load_dataset('/root/model/datasets/wikitext/wikitext-2-raw-v1', split='train')
    # testdata = load_dataset('/root/model/datasets/wikitext/wikitext-2-raw-v1', split='test')

    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({'input_ids':inp,'attention_mask': attention_mask})

    return traindataset, testenc

@torch.no_grad()
def llama_eval(model, testenc, dev, seqlen = 2048):
    from tqdm import tqdm
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in tqdm(range(len(layers))):
        # print('layer', i)
        layer = layers[i].to(dev)
        layer = layers[i].to(dtype)
        for j in range(nsamples):
            # print("dtype", inps[j].unsqueeze(0).dtype)
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in tqdm(range(nsamples)):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print('ppl: ')
    print(ppl.item())
    print()

    model.config.use_cache = use_cache

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf', type=str) # quant base model
    parser.add_argument('--dev', type=str, default="cuda:0")
    parser.add_argument('--quant_type', type=str, default="int", help='Quantization data type')
    parser.add_argument('--bits', type=int, default=3, help='Quantization bits')
    parser.add_argument('--group_size', type=int, default=128, help='Quantization group size')

    args = parser.parse_args()
    print(args)
    

    print("loading the model...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, use_safetensors=True, low_cpu_mem_usage=True)

    q_config = {
        "zero_point": True,  # by default True
        "q_group_size": args.group_size,  # whether to use group quantization
    }
    model = model.cuda()
    pseudo_quantize_model_weight(
        model, w_bit=args.bits, q_config=q_config, quant_type=args.quant_type
    )

    dev = torch.device(args.dev)

    dataloader, testloader = get_wikitext2(nsamples=128, seed=0, seqlen=2048, model=args.model)

    llama_eval(model, testloader, dev)

if __name__ == "__main__":
    import logging
    main()