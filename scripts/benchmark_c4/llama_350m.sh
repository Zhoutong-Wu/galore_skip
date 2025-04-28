# LLaMA-350M, GaLore-Adam, 4 A100, 1 Node
lr=0.002
wd=0.1
name="llama350m-base-adamw-lr${lr}-wd${wd}"
bash launch.sh torchrun_main.py \
    --model_config configs/llama_350m.json \
    --lr $lr \
    --galore_scale 0.25 \
    --rank 128 \
    --update_proj_gap 200 \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 15000 \
    --warmup_steps 1500 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_dir checkpoints/LLaMA-350M \
    --name $name \
    --max_length 1024