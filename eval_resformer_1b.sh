set -euo pipefail
export http_proxy="http://sys-proxy-rd-relay.byted.org:3128"
export https_proxy="http://sys-proxy-rd-relay.byted.org:3128"
export HF_DATASETS_CACHE="$(pwd)/hf_datasets"
export TOKENIZERS_PARALLELISM="false"

# 全局设置
DEVICE="cuda:2"
BATCH_SIZE=16
TOKENIZER="t5-base"
BASE_OUTPUT_DIR="/mnt/bn/ymdong-opensource/lm-evaluation-harness/results"
MODEL_PATHS=(
  "/mnt/bn/ymdong-opensource/FRP/llama2_pretraining/checkpoints/Skip-LLaMA-1B/model_25001"
)
for MODEL_DIR in "${MODEL_PATHS[@]}"; do
  # 从路径中提取一个简短名字，用于输出目录区分
  parent=$(dirname "$MODEL_DIR")
  MODEL_NAME=$(basename "${parent}")
  OUTPUT_PATH="${BASE_OUTPUT_DIR}/${MODEL_NAME}"

python final_lm_eval.py \
    --model_config configs/llama_1b.json \
    --continue_from "${MODEL_DIR}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_PATH}" 
done