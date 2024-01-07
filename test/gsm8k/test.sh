export MASTER_ADDR=localhost
export MASTER_PORT=2132
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL_DIR=$1
OUT_DIR=$2

torchrun --nproc_per_node 8 --master_port 7832 test.py \
                        --base_model $MODEL_DIR \
                        --data_path "./test_use.jsonl" \
                        --out_path $OUT_DIR \
                        --batch_size 8 \
                        --group_size 128 \
                        --quant_type int \
                        --bits 2

                        
python eval.py --path "${OUT_DIR}/raw_generation_0.2_1.json"