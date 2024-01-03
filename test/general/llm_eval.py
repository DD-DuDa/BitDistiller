import sys
from lm_eval import evaluator, tasks, utils
from utils_eval import LMEvalAdaptor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    LlamaTokenizer
)
import torch
import sys

sys.path.append("../")
from test_utils import pseudo_quantize_model_weight

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument('--eval_tasks', type=str, help='evaluation tasks') # hendrycksTest-*; arc_challenge,winogrande,hellaswag,piqa
    parser.add_argument('--test_set', action="store_true", help='evaluation tasks')
    parser.add_argument('--batch_size', type=int, default=2, help='evaluation tasks')
    parser.add_argument('--bits', type=int, default=2, help='evaluation tasks')
    parser.add_argument('--group_size', type=int, default=128, help='evaluation tasks')
    parser.add_argument('--quant_type', type=str, default="int", help='evaluation tasks')
    parser.add_argument('--num_fewshot', type=int, default=0, help='evaluation tasks')
    args = parser.parse_args()
    print(args)
    if "hendrycksTest" not in args.eval_tasks:
        args.test_set = True
    
    
    model = AutoModelForCausalLM.from_pretrained(args.model, 
                                                torch_dtype=torch.bfloat16, 
                                                use_safetensors=True,
                                                device_map='auto'
                                                )
        
    if args.quant_type is not None:
        q_config = {
            "zero_point": True,  # by default True
            "q_group_size": args.group_size,  # whether to use group quantization
        }
        pseudo_quantize_model_weight(
            model, w_bit=args.bits, q_config=q_config, quant_type=args.quant_type
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model.eval()

    task_names = utils.pattern_match(args.eval_tasks.split(","), tasks.ALL_TASKS)

    lm_eval_model = LMEvalAdaptor(args.model, model, tokenizer, args.batch_size)
    
    results = evaluator.simple_evaluate(
                    model=lm_eval_model,
                    tasks=task_names,
                    batch_size=args.batch_size,
                    no_cache=True,
                    num_fewshot=args.num_fewshot,
                    test_set=args.test_set
                )
    print(results)
    # 初始化总 acc 和计数器
    acc_sum = 0
    count = 0
    # 遍历所有 hendrycksTest 相关的数据
    if "hendrycksTest" in args.eval_tasks:
        for key in results['results']:
            if 'hendrycksTest' in key:
                # 累加 acc 值并增加计数器
                acc_sum += results['results'][key]['acc']
                count += 1

        # 计算平均值
        if count > 0:
            avg_acc = acc_sum / count

            # print("mmlu-acc:", avg_acc)
            mmlu_results = {}
            mmlu_results['mmlu-acc'] = avg_acc

            print(mmlu_results)
    else:
        for key in results['results']:
            acc_sum += results['results'][key]['acc']
            count += 1
        # 计算平均值
        if count > 0:
            avg_acc = acc_sum / count
            print("QA Avg:", avg_acc)