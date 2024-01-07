from datasets import load_dataset
import random

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "{instruction}\n{input}\n"
    ),
    "prompt_no_input": (
        "{instruction}\n"
    ),
}

ALPACA_PROMPT_DICT_SOLAR = {
    "prompt_input": (
        "### User: \n{instruction}\n{input}\n\n### Assistant:\n"
    ),
    "prompt_no_input": (
        "### User: \n{instruction}\n\n### Assistant:\n"
    ),
}

OPENORCA_PROMPT_DICT_SOLAR = {
    "prompt_input": (
        "### System:\n{system_prompt}\n\n### User: \n{question}\n\n### Assistant:\n"
    ),
    "prompt_no_input": (
        "### User: \n{question}\n\n### Assistant:\n"
    )
}

ULTRA_PROMPT_DICT_SOLAR = {
    "prompt_no_input": (
        "### User: \n{prompt}\n\n### Assistant:\n"
    )
}

CODE_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
}

MATH_PROMPT_DICT = {
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query}\n\n### Response: Let's think step by step."
    )
}




def get_gen_dataset(dataset_name, max_sample=None, tokenizer=None):
    if dataset_name == "wikitext":
        return get_wiki_dataset(max_sample)
    elif dataset_name == "repajama":
        return get_redpajama_dataset(max_sample)
    elif dataset_name == "alpaca":
        return get_alpaca_dataset(max_sample, tokenizer)
    elif dataset_name == "alpaca-solar":
        return get_alpaca_solar_dataset(max_sample, tokenizer)
    elif dataset_name == "openorca-solar":
        return get_openorca_solar_dataset(max_sample, tokenizer)
    elif dataset_name == "ultra-solar":
        return get_ultra_solar_dataset(max_sample, tokenizer)
    elif dataset_name == "code":
        return get_code_dataset(max_sample, tokenizer)
    elif dataset_name == "math":
        return get_math_dataset(max_sample, tokenizer)
    else:
        raise ValueError(f"{dataset_name} not implement yet")

def extract_random_dataset(sources, targets, max_sample=None):
    if max_sample is not None:
        if max_sample <= len(sources):
            print(f"only use {max_sample} samples")
            random_indices = random.sample(range(len(sources)), max_sample)
            sources = [sources[i] for i in random_indices]
            targets = [targets[i] for i in random_indices]
        else:
            print("max_sample exceeds the length of the array. Using the entire array.")
            sources = sources
            targets = targets
    else:
        print(f"using the whole {len(sources)} samples")

    return sources, targets

def get_wiki_dataset(max_sample):
    wiki_dataset = load_dataset("wikitext", 'wikitext-2-raw-v1', split='train')
    # wiki_dataset = load_dataset("/root/model/datasets/wikitext/wikitext", 'wikitext-2-raw-v1', split='train')

    wiki_long = []
    for text in wiki_dataset['text']:
        if len(text) > 512:
            wiki_long.append(text)
    wiki_front = [long[:128] for long in wiki_long]

    targets = sources = wiki_front

    return extract_random_dataset(sources, targets, max_sample)

def get_redpajama_dataset(max_sample):
    # wiki_dataset = load_dataset("wikitext", 'wikitext-2-raw-v1', split='train')
    wiki_dataset = load_dataset('/root/model/datasets/RedPajama-Data-1T-Sample', split='train')

    wiki_long = []
    for text in wiki_dataset['text']:
        if len(text) > 512:
            wiki_long.append(text)
    wiki_front = [long[:128] for long in wiki_long]

    targets = sources = wiki_front

    return extract_random_dataset(sources, targets, max_sample)



def get_alpaca_dataset(max_sample, tokenizer):
    alpaca_dataset = load_dataset("yahma/alpaca-cleaned", split='train')
    # alpaca_dataset = load_dataset("/root/model/datasets/alpaca-clean", split='train')

    prompt_input, prompt_no_input = ALPACA_PROMPT_DICT["prompt_input"], ALPACA_PROMPT_DICT["prompt_no_input"]

    sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in alpaca_dataset
                ]
    targets = [f"{example['output']}{tokenizer.eos_token}" for example in alpaca_dataset]

    return extract_random_dataset(sources, targets, max_sample)

def get_alpaca_solar_dataset(max_sample, tokenizer):
    alpaca_dataset = load_dataset("yahma/alpaca-cleaned", split='train')
    # alpaca_dataset = load_dataset("/root/model/datasets/alpaca-clean", split='train')

    prompt_input, prompt_no_input = ALPACA_PROMPT_DICT_SOLAR["prompt_input"], ALPACA_PROMPT_DICT_SOLAR["prompt_no_input"]

    sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in alpaca_dataset
                ]
    targets = [f"{example['output']}{tokenizer.eos_token}" for example in alpaca_dataset]

    return extract_random_dataset(sources, targets, max_sample)

def get_openorca_solar_dataset(max_sample, tokenizer):
    openorca_dataset = load_dataset("Open-Orca/OpenOrca", split='train')
    #openorca_dataset = load_dataset("/root/model/datasets/OpenOrca", split='train')

    prompt_input, prompt_no_input = OPENORCA_PROMPT_DICT_SOLAR["prompt_input"], OPENORCA_PROMPT_DICT_SOLAR["prompt_no_input"]

    sources = []
    targets = []
    for example in openorca_dataset.select(range(20000)):
        if len(example['question']) > 1024:
            continue
        if example.get("system_prompt", "") != "":
            sources.append(prompt_input.format_map(example)) 
        else:
            sources.append(prompt_no_input.format_map(example))

        targets.append(f"{example['response']}{tokenizer.eos_token}")

    return extract_random_dataset(sources, targets, max_sample)

def get_ultra_solar_dataset(max_sample, tokenizer):
    # alpaca_dataset = load_dataset("yahma/alpaca-cleaned", split='train')
    ultra_dataset = load_dataset("/root/model/datasets/ultrafeedback_binarized_cleaned", split='train_sft')

    prompt_no_input = ULTRA_PROMPT_DICT_SOLAR["prompt_no_input"]

    sources = []
    targets = []
    for example in ultra_dataset:
        if len(example['prompt']) > 1024:
            continue
        sources.append(prompt_no_input.format_map(example))
        targets.append(f" ")

    return extract_random_dataset(sources, targets, max_sample)

def get_code_dataset(max_sample, tokenizer):
    code_dataset = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split='train')
    # code_dataset = load_dataset('json', data_files="/root/model/datasets/code/EvolInstruct-Code-80k.json", split='train')

    prompt_input, prompt_no_input = CODE_PROMPT_DICT["prompt_input"], CODE_PROMPT_DICT["prompt_no_input"]

    sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in code_dataset
                ]

    targets = [f"{example['output']}{tokenizer.eos_token}" for example in code_dataset]

    return extract_random_dataset(sources, targets, max_sample)

def get_math_dataset(max_sample, tokenizer):
    math_dataset = load_dataset("meta-math/MetaMathQA-40K", split='train')
    # math_dataset = load_dataset('json', data_files="/root/model/acr_duda/gsm8k/data/MetaMath-40K.json", split='train')
    prompt_no_input = MATH_PROMPT_DICT["prompt_no_input"]

    sources = [prompt_no_input.format_map(example) for example in math_dataset]

    targets = [f"{example['response']}{tokenizer.eos_token}" for example in math_dataset]

    return extract_random_dataset(sources, targets, max_sample)