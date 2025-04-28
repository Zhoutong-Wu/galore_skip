# LLaMA-1B, GaLore-Adam, 8 A100, 1 Node
lr=0.0005
wd=0.1
name="llama1b-skip-adamw-lr${lr}-wd${wd}"
bash lunch.sh torchrun_main_skip.py\
    --model_config configs/llama_1b.json \
    --lr $lr \
    --rank 1024 \
    --update_proj_gap 200 \
    --batch_size 16 \
    --total_batch_size 512 \
    --num_training_steps 25000 \
    --warmup_steps 2500 \
    --weight_decay $wd \
    --dtype bfloat16 \
    --eval_every 1000 \
    --name $name \
    --save_dir checkpoints/Skip-LLaMA-1B \
    --grad_clipping 1.0 \
    --max-length 1024