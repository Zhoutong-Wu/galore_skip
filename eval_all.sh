#!/usr/bin/env bash
set -euo pipefail
export http_proxy="http://sys-proxy-rd-relay.byted.org:3128"
export https_proxy="http://sys-proxy-rd-relay.byted.org:3128"
export HF_DATASETS_CACHE="$(pwd)/hf_datasets"
# 全局设置
TASKS="boolq,piqa,mathqa,openbookqa,winogrande,rte,hellaswag,arc_challenge,arc_easy,sciq"
DEVICE="cuda:1"
BATCH_SIZE=16
TOKENIZER="t5-base"
BASE_OUTPUT_DIR="/mnt/bn/ymdong-opensource/lm-evaluation-harness/results"
MODEL_PATHS=(
  #---------------------------LLaMA-1B------------------------------
  #1B baseline
  "/mnt/bn/ymdong-opensource/FRP/llama2_pretraining/checkpoints/LLaMA-1B/model_25001"
  "/mnt/bn/ymdong-opensource/FRP/llama2_pretraining/checkpoints/Skip-LLaMA-1B/model_25001"
  "/mnt/bn/ymdong-opensource/FRP/llama2_pretraining/checkpoints/LLaMA-3B/model_30001"
  "/mnt/bn/ymdong-opensource/FRP/llama2_pretraining/checkpoints/LLaMA-3B/model_30001"
)
for MODEL_DIR in "${MODEL_PATHS[@]}"; do
  # 从路径中提取一个简短名字，用于输出目录区分
  parent=$(dirname "$MODEL_DIR")
  MODEL_NAME=$(basename "${parent}")
  OUTPUT_PATH="${BASE_OUTPUT_DIR}/${MODEL_NAME}"

  echo "=== Evaluating ${MODEL_NAME} ==="
  lm_eval \
    --model hf \
    --trust_remote_code \
    --model_args "pretrained=${MODEL_DIR},tokenizer=${TOKENIZER}" \
    --tasks "${TASKS}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_PATH}"
  echo "=> Results saved to ${OUTPUT_PATH}"
  echo
done