# [ACL 2024] BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation [[paper]](http://arxiv.org/abs/2402.10631)

**Implementing efficient sub-4-bit weight quantization (3 / 2 bits) in LLMs through advanced QAT-based Self-Distillation techniques.**

![overview](./imgs/overview.jpg)

## Comparing general language tasks with other methods
![overview](./imgs/result7b.jpg)
<!-- ![overview](./imgs/result2.png)-->
## Comparing reasoning benchmarks with other methods
![overview](./imgs/result.png)

## Example on 2-bit inference of a Domain-specific LLM (MetaMath)
![gif](./imgs/Bitdistiller.gif)

## News
* [2024/05] 🔥 BitDistiller has been accepted to ACL main 2024! 


## Contents
1. [Setup](#1-setup)
2. [Running](#2-running)
3. [Evaluation](#3-evaluation)
4. [Inferencce](#4-inference)

## 1. Setup
* python 3.9, pytorch >= 1.13
* pip install -r requirement.txt 
  
  (You may need to change the version of transformers according to the model config)

## 2. Running

Our results is running by following 3 steps:

### 2.1. Asymmetric Quantization
* Determine the type of quantization: use `nf3` for 3 bits and `int` for 2 bits. Set `w_bit` and `quant_type` accordingly.

* Perform clipping before training and save the clipping values using dump_clip (see `quantization/autoclip.py`).

>This step can match or surpass the low-bit PTQ quantization results of GPTQ and AWQ.

### 2.2. Generating Teacher Data
* For QAT, create data using the Teacher Model (BF16). The data varies depending on the model (see `data/generation`).


### 2.3. KD-base QAT
* Detailed procedure available in `train/`


### Example Srcipts

<details>
  <summary>LLaMA-2</summary>
  
1. Get the Clipping result
    ```bash
    cd BitDistiller/quantization

    CUDA_VISIBLE_DEVICES=0 python autoclip.py --model_path <model_path> --calib_dataset pile --quant_type int --w_bit 2 --q_group_size 128 --run_clip --dump_clip ./clip_cache/hf-llama2-7b/int2-g128.pt
    ```
2. Get the Teacher Generation Data (Using vllm would be much faster)
    ```bash
    # vllm
    python generate_vllm.py --base_model <model_path> --dataset_name wikitext --out_path ./datasets/hf-llama-2-7b/ --max_sample 3000

    python generate_vllm.py --base_model <model_path> --dataset_name alpaca --out_path ./datasets/hf-llama-2-7b/ --max_sample 5000

    # change to path in .py
    python mix_data.py
    ```

    ```bash
    # torchrun
    cd BitDistiller/data/generation

    bash generate.sh <model_path> wikitext ../datasets/hf-llama-2-7b/ 16 3000

    bash generate.sh <model_path> alpaca ../datasets/hf-llama-2-7b/ 16 5000

    # change to path in .py
    python mix_data.py
    ```
3. Run KD-base QAT
    ```bash
    # Specify the pre-trained model path
    # Specify the num_gpus and batch_size according to your GPU devices
    # Specify the clipping cache path to the --clip

    cd train
    
    bash train.sh ../data/datasets/hf-llama-2-7b/mix_wiki_alpaca_8000.json ./ckpts/hf-llama-2-7b/int2-g128/ ./logs/hf-llama-2-7b/int2-g128/ 4
    ```
</details>

<details>
  <summary>WizardCoder</summary>
  
1. Get the Clipping result
    ```bash
    cd BitDistiller/quantization

    CUDA_VISIBLE_DEVICES=0 python autoclip.py --model_path <model_path> --calib_dataset code --quant_type int --w_bit 2 --q_group_size 128 --run_clip --dump_clip ./clip_cache/WizardCoder-7B/int2-g128.pt
    ```
2. Get the Teacher Generation Data
    ```bash
    # vllm
    python generate_vllm.py --base_model <model_path> --dataset_name code --out_path ./datasets/WizardCoder-7b/ --max_sample 3000
    ```

    ```bash
    cd BitDistiller/data/generation

    bash generate.sh /root/WizardCoder-Python-7B/ code ../datasets/WizardCoder-7b/ 16 3000
    ```
3. Run KD-base QAT
    ```bash
    # Specify the pre-trained model path
    # Specify the num_gpus and batch_size according to your GPU devices
    # Specify the clipping cache path to the --clip

    cd train
    
    bash train.sh ../data/datasets/WizardCoder-7b/code_T0.7_N1024_S42_3000.json ./ckpts/WizardCoder-7b/int2-g128/ ./logs/WizardCoder-7b/int2-g128/ 2
    ```
</details>

<details>
  <summary>MetaMath</summary>

1. Get the Clipping result
    ```bash
    cd BitDistiller/quantization

    CUDA_VISIBLE_DEVICES=0 python autoclip.py --model_path <model_path> --calib_dataset gsm8k --quant_type int --w_bit 2 --q_group_size 128 --run_clip --dump_clip ./clip_cache/MetaMath-7B/int2-g128.pt
    ```
2. Get the Teacher Generation Data
    ```bash
    # vllm
    python generate_vllm.py --base_model <model_path> --dataset_name math --out_path ./datasets/MetaMath-7B/ --max_sample 3000
    ```

    ```bash
    cd BitDistiller/data/generation

    bash generate.sh /root/MetaMath-7B-V1.0/ math ../datasets/MetaMath-7B/ 16 3000
    ```
3. Run KD-base QAT
    ```bash
    # Specify the pre-trained model path
    # Specify the num_gpus and batch_size according to your GPU devices
    # Specify the clipping cache path to the --clip

    cd train
    
    bash train.sh ../data/datasets/MetaMath-7B/math_T0.7_N1024_S42_3000.json ./ckpts/MetaMath-7b/int2-g128/ ./logs/MetaMath-7b/int2-g128/ 2
    ```
</details>

## 3. Evaluation
### Example Srcipts
<details>
  <summary>LLaMA-2</summary>



* Test PPL on WikiText-2
  ```bash
  cd test/general

  python wiki_ppl.py --model ../../train/ckpts/hf-llama-2-7b/int2-g128/checkpoint-200/ --quant_type int --bits 2 --group_size 128
  ```
* Test MMLU
  ```bash
  CUDA_VISIBLE_DEVICES=0 python llm_eval.py --model ../../train/ckpts/hf-llama-2-7b/int2-g128/checkpoint-200/ --eval_tasks hendrycksTest-* --test_set --bits 2 --group_size 128 --quant_type int --num_fewshot 5
  ```
* Test Common-sense QA Tasks
  ```bash
  CUDA_VISIBLE_DEVICES=0 python llm_eval.py --model ../../train/ckpts/hf-llama-2-7b/int2-g128/checkpoint-200/ --eval_tasks arc_challenge,winogrande,hellaswag,piqa --test_set --bits 2 --group_size 128 --quant_type int --num_fewshot 0 
  ```

</details>

<details>
  <summary>WizardCoder</summary>

* Install the environment according to the instructions of [HumanEval](https://github.com/openai/human-eval), 

* Example script:
    ```bash
    cd test/humaneval
    bash gen_preds.sh [checkpoint_path] ./preds/7b/int2-g128/
    ```
</details>

<details>
  <summary>MetaMath</summary>
  
* Example script:

    ```bash
    cd test/gsm8k
    bash test.sh ../../train/ckpts/MetaMath-7b/int2-g128/ ./preds/7b/int2-g128/
    ```
</details>


## 4. Inference
Please see `inference/`



## Reference
If you find BitDistiller useful or relevant to your research, please kindly cite our paper:
```
@misc{du2024bitdistiller,
      title={BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation}, 
      author={Dayou Du and Yijia Zhang and Shijie Cao and Jiaqi Guo and Ting Cao and Xiaowen Chu and Ningyi Xu},
      year={2024},
      eprint={2402.10631},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
