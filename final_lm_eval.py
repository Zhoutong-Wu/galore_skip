import sys
sys.path.append('/home/ztwu/lm-evaluation-harness')
import os
import torch
import json
import argparse
from lm_eval.models.huggingface import HFLM
import transformers
from transformers import AutoTokenizer
from peft_pretraining.modeling_llama_skip import LlamaForCausalLM
from peft_pretraining import training_utils, args_utils
from transformers import GenerationConfig

import lm_eval

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None, help="Path to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the evaluation on")
    
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default="checkpoints/Skip-LLaMA-130M/model_5001")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)  
    
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95) 
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)
    
    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")
    
    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args
    
def main(args):
    # 1) Load tokenizer & model exactly as in CLI:
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=2048)
    model = LlamaForCausalLM.from_pretrained(
        args.continue_from,
        trust_remote_code=True,
        local_files_only=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if not hasattr(model, 'generation_config'):
        model.generation_config = GenerationConfig(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    # 4. Move to device
    model.to(args.device).eval()
    

    # 2) **Replace your custom LlamaLM** with the official HFLM:
    lm_obj = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal", 
        device=args.device,
        batch_size=args.batch_size,
        max_length=2048,
        truncation=False,
        logits_cache=True,
        trust_remote_code=True,    # to use your custom architecture
        peft=None,  
    )

    task_manager = lm_eval.tasks.TaskManager()
    # 3) Then call simple_evaluate exactly as before:
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=["hellaswag", "winogrande", "arc_challenge", "arc_easy", "piqa", "openbookqa", "sciq", "boolq", "rte"],
        num_fewshot=0,
        task_manager=task_manager,
    )
    print(results["results"])
    if args.output_path:  # 如果指定了输出路径
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)  # 确保目录存在
        with open(args.output_path, "w") as f:
            json.dump(results["results"], f, indent=4)  # 保存为 JSON 格式
        print(f"Results saved to {args.output_path}")
    
if __name__ == "__main__":
    args = parse_args(None)
    main(args)