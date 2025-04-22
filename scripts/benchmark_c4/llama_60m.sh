# LLaMA-60M, GaLore-Adam, 1 A100, 1 Node
export CUDA_VISIBLE_DEVICES=4
torchrun --standalone --nproc_per_node 1 --master_port=29504 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 0.02 \
    --galore_scale 0.25 \
    --rank 128 \
    --update_proj_gap 200 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 200 \