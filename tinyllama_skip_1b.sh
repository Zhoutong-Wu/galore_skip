lr=0.001
wd=0.1
name="tinyllama1b-base-adamw-lr${lr}-wd${wd}"
torchrun --standalone --nproc_per_node 2 torchrun_main.py\
    --model_config configs/tinyllama_1b.json \
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
    --save_every 2000 \
    --name $name \
    --save_dir checkpoints/TinyLLaMA-1B \
    --grad_clipping 1.0 \
    --max_length 1024 \
    --use_hf_model 