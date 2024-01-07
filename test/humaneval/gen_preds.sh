model=$1
output_path=$2

temp=1.0
max_len=1024
pred_num=1 # 10
num_seqs_per_iter=1 # 2

output_path=${output_path}/T${temp}_N${pred_num}

mkdir -p ${output_path}
echo 'Output path: '$output_path
echo 'Model to eval: '$model

# 164 problems, 21 per GPU if GPU=8
index=0
gpu_num=8
problem_num=21
for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * $problem_num))
  end_index=$(((i + 1) * $problem_num))

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
    CUDA_VISIBLE_DEVICES=$gpu python humaneval_gen.py --model ${model} \
      --start_index ${start_index} --end_index ${end_index} --temperature ${temp} --greedy_decode\
      --num_seqs_per_iter ${num_seqs_per_iter} --N ${pred_num} --max_len ${max_len} --output_path ${output_path} \
      --quant_type int --bits 2 --group_size 128
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done

bash eval.sh $output_path