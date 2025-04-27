# LLaMA-3B, GaLore-Adam, 8 A100, 1 Node
lr=0.00025
wd=0.1
name="llama3b-skip-adamw-lr${lr}-wd${wd}"
torchrun --standalone --nproc_per_node 8 torchrun_main_skip.py \
    --model_config configs/llama_3b.json \
    --lr $lr \
    --rank 1024 \
    --update_proj_gap 200 \
    --batch_size 32 \
    --total_batch_size 256 \
    --num_training_steps 120000 \
    --warmup_steps 12000 \
    --weight_decay $wd \
    --dtype bfloat16 \
    --eval_every 1000 \
    --name $name \
    --save_dir checkpoints/Skip-LLaMA-3B \
    --grad_clipping 1.0 \
