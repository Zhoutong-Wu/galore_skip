# LLaMA-350M, GaLore-Adam, 4 A100, 1 Node
lr=0.001
wd=0.1
name="llama350m-skip-adamw-lr${lr}-wd${wd}"
torchrun --standalone --nproc_per_node 4 torchrun_main_skip.py \
    --model_config configs/llama_350m.json \
    --lr $lr \
    --galore_scale 0.25 \
    --rank 128 \
    --update_proj_gap 200 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 30000 \
    --warmup_steps 3000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 500 \
    --save_dir checkpoints/Skip-LLaMA-350M \
    --name $name