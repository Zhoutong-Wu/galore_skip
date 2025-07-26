# LLaMA-1B, GaLore-Adam, 8 A100, 1 Node
lr=0.001
wd=0.1
name="llama1b-base-adamw-lr${lr}-wd${wd}"
torchrun --standalone --nproc_per_node 1 torchrun_main_skip_ratio.py\
    --model_config configs/llama_1b.json \
    --lr $lr \
    --batch_size 16 \
    --total_batch_size 512 \
    --num_training_steps 25000 \
    --warmup_steps 2500 \
    --weight_decay $wd \
    --dtype bfloat16 \
    --eval_every 1000 \
    --name $name \
    --save_dir checkpoints/LLaMA-1B-ratio75 \
    --grad_clipping 1.0 \
    --max_length 1024 \
    --save_every 5000 \
    --skip_head_ratio 0.75 \