import os
import sys

import argparse
import torch
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, List
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
from tqdm import tqdm
import copy
from torch.cuda.amp import autocast
import random
from datasets import load_dataset
from data_utils import get_gen_dataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def sequence_gather(s, world_size, pad_tok_id):
    local_size = torch.tensor(s.size(), device=s.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_length = max(size[1] for size in all_sizes)
    length_diff = max_length.item() - local_size[1].item()
    if length_diff:
        pad_size = (*s.shape[:-1], length_diff)
        padding = torch.ones(pad_size, device=s.device, dtype=s.dtype) * pad_tok_id
        s = torch.concat((s, padding), dim = -1)
    gathered_s = [torch.ones_like(s)*pad_tok_id for _ in range(world_size)]
    dist.all_gather(gathered_s, s)

    return gathered_s

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(strings, tokenizer: transformers.PreTrainedTokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources,
    targets,
    tokenizer: transformers.PreTrainedTokenizer,
):
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = sources_tokenized["input_ids"]
    return dict(input_ids=input_ids, labels=copy.deepcopy(input_ids))

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, dataset_name: str, tokenizer: transformers.PreTrainedTokenizer, max_sample=None):
        super(SupervisedDataset, self).__init__()
        
        sources, targets = get_gen_dataset(dataset_name, max_sample, tokenizer)

        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], id=i)

def padding(inputs, padding_token, cutoff = None):
    num_elems = len(inputs)
    if cutoff is None:
        cutoff = max([len(item) for item in inputs])
    else:
        cutoff = min(max([len(item) for item in inputs]), cutoff)
    
    tokens = torch.ones(num_elems, cutoff).long().to(inputs[0].device) * padding_token
    for i in range(num_elems):
        toks = inputs[i]
        length = min(cutoff, len(toks))
        tokens[i, -length:] = toks[-length:]
    return tokens

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", 'id'))
        input_ids = padding(input_ids, self.tokenizer.pad_token_id, cutoff = 256)
        labels = padding(labels, IGNORE_INDEX, cutoff = 256)

        return dict(
            input_ids=input_ids,
            labels=labels,
            id=torch.tensor(ids).to(input_ids.device),
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, dataset_name, max_sample=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    gen_dataset = SupervisedDataset(tokenizer=tokenizer, dataset_name=dataset_name, max_sample=max_sample)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return gen_dataset, data_collator

def main(rank, args):
    dist.init_process_group("nccl")
    torch.manual_seed(args.seed)
    world_size = torch.cuda.device_count()

    base_model = args.base_model
    batch_size = args.batch_size
    return_seq_num = 1
    
    n_gpus = torch.cuda.device_count()

    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16
        )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    tokenizer.truncation_side = 'left'
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    torch.cuda.set_device(rank)
    model.to(torch.cuda.current_device())
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    model.eval()

    # Get the generation dataset
    gen_dataset, data_collator = make_supervised_data_module(tokenizer, args.dataset_name, args.max_sample)

    sampler = torch.utils.data.distributed.DistributedSampler(gen_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        gen_dataset, 
        shuffle=False, 
        collate_fn=data_collator, 
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True
    )

    generation_config = GenerationConfig(
        temperature=args.temperature,
        do_sample=True,
        num_beams=return_seq_num,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=return_seq_num,
        top_p=1.0
    )

    all_outputs = []
    total_nums = int(len(gen_dataset) / (world_size * batch_size))
    for step, batch in tqdm(enumerate(dataloader), total=total_nums):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        with torch.no_grad():
            generation_output = model.module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                synced_gpus=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        s = generation_output.sequences
        gather_outputs = sequence_gather(s, world_size, tokenizer.pad_token_id)
        gathered_inputs = sequence_gather(input_ids, world_size, tokenizer.pad_token_id)
        gather_outputs = torch.stack(gather_outputs).reshape(world_size, batch_size,return_seq_num,-1)
        gathered_inputs = torch.stack(gathered_inputs)
        gather_outputs = gather_outputs.transpose(0, 1).reshape(batch_size * world_size * return_seq_num, -1)
        gathered_inputs = gathered_inputs.transpose(0, 1).reshape(batch_size * world_size,-1)

        outputs_string = tokenizer.batch_decode(gather_outputs, skip_special_tokens=True)
        inputs_string = tokenizer.batch_decode(gathered_inputs, skip_special_tokens=True)

        for idx in range(len(inputs_string)):
            temp = []
            for i in range(return_seq_num):
                temp.append([inputs_string[idx], outputs_string[return_seq_num*idx+i].replace(inputs_string[idx], '')])
            all_outputs.append(temp)

    if rank == 0:
        with open(args.out_path + f'/{args.dataset_name}_T{args.temperature}_N{args.max_new_tokens}_S{args.seed}_{args.max_sample}.json', 'w') as f:
            for item in all_outputs[:len(gen_dataset)]:
                f.write(json.dumps(item) + '\n')
    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--base_model", default="", type=str, help="model path")
    parser.add_argument("--dataset_name", default="", type=str, help="name of the datasets")
    parser.add_argument("--batch_size", type=int, default=0, help="batch size")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--out_path", default="", type=str, help="output datapath")
    parser.add_argument("--max_sample", type=int, default=None, help="max_sample")
    parser.add_argument("--temperature", type=int, default=0.7, help="generation temperature")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="max new tokens")
    parser.add_argument("--return_seq_num", type=int, default=1, help="return seq num")
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        try:
            os.makedirs(args.out_path)
            print(f"dir {args.out_path} create successfully")
        except:
            pass
    else:
        print(f"dir {args.out_path} has existed")

    local_rank = int(os.environ["LOCAL_RANK"])
    main(local_rank, args)