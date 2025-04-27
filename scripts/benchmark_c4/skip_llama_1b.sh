# LLaMA-1B, GaLore-Adam, 8 A100, 1 Node
lr=0.0005
wd=0.1
name="llama1b-skip-adamw-lr${lr}-wd${wd}"
torchrun --standalone --nproc_per_node 8 torchrun_main_skip.py \
    --model_config configs/llama_1b.json \
    --lr $lr \
    --rank 1024 \
    --update_proj_gap 200 \
    --batch_size 64 \
    --total_batch_size 256 \
    --num_training_steps 45000 \
    --warmup_steps 4500 \
    --weight_decay $wd \
    --dtype bfloat16 \
    --eval_every 1000 \
    --name $name \
    --save_dir checkpoints/Skip-LLaMA-1B \
    --grad_clipping 1.0 \