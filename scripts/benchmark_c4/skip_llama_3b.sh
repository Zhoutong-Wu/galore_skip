# LLaMA-3B, GaLore-Adam, 8 A100, 1 Node
lr=0.0005
wd=0.1
name="llama3b-skip-adamw-lr${lr}-wd${wd}"
bash launch.sh torchrun_main_skip.py \
    --model_config configs/llama_3b.json \
    --lr $lr \
    --rank 1024 \
    --update_proj_gap 200 \
    --batch_size 8 \
    --total_batch_size 512 \
    --num_training_steps 30000 \
    --warmup_steps 3000 \
    --weight_decay $wd \
    --dtype bfloat16 \
    --eval_every 1000 \
    --name $name \
    --save_dir checkpoints/Skip-LLaMA-3B \
    --grad_clipping 1.0 \
    --max_length 1024
