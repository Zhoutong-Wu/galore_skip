# LLaMA-130M, GaLore-Adam, 1 A100, 1 Node
lr=0.002
wd=0.1
name="llama130m-base-adamw-lr${lr}-wd${wd}"
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_130m.json \
    --lr $lr \
    --galore_scale 0.25 \
    --rank 128 \
    --update_proj_gap 200 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 200 \
    --save_dir checkpoints/LLaMA-130M \
    --name $name