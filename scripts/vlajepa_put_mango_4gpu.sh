#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FFMPEG_THREADS="${FFMPEG_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"

CONFIG_YAML="${CONFIG_YAML:-scripts/config/vlajepa_put_mango_ft.yaml}"
RUN_ID="${RUN_ID:-put_mango_v21_ft_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${REPO_ROOT}/outputs}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-30000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
GRADIENT_CLIPPING="${GRADIENT_CLIPPING:-1.0}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_LATEST_CHECKPOINT_ONLY="${KEEP_LATEST_CHECKPOINT_ONLY:-true}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_FINAL_MODEL="${SAVE_FINAL_MODEL:-true}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
QWEN_LR="${QWEN_LR:-1e-5}"
VJ_PREDICTOR_LR="${VJ_PREDICTOR_LR:-3e-5}"
ACTION_LR="${ACTION_LR:-1e-4}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
WANDB_PROJECT="${WANDB_PROJECT:-vla-jepa}"
WANDB_GROUP="${WANDB_GROUP:-put_mango_v21}"
WANDB_ENTITY="${WANDB_ENTITY:-livion}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-${REPO_ROOT}/datasets}"
QWEN_PATH="${QWEN_PATH:-${REPO_ROOT}/checkpoints/Qwen3-VL-2B-Instruct}"
VJEPA_PATH="${VJEPA_PATH:-${REPO_ROOT}/checkpoints/vjepa2-vitl-fpc64-256}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${REPO_ROOT}/checkpoints/vla-jepa-pretrain/Pretrain/checkpoints/VLA-JEPA-pretrain.pt}"

MODALITY_FILE="${DATA_ROOT_DIR}/put_mango_v21/meta/modality.json"
if [[ ! -f "${MODALITY_FILE}" ]]; then
  echo "Missing ${MODALITY_FILE}. Copy examples/put_mango_v21/modality.json there first." >&2
  exit 1
fi

TRAIN_ARGS=(
  --config_yaml "${CONFIG_YAML}"
  --run_id "${RUN_ID}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_group "${WANDB_GROUP}"
  --framework.qwenvl.base_vlm "${QWEN_PATH}"
  --framework.qwenvl.attn_implementation "${ATTN_IMPLEMENTATION}"
  --framework.vj2_model.base_encoder "${VJEPA_PATH}"
  --datasets.vla_data.data_root_dir "${DATA_ROOT_DIR}"
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --datasets.vla_data.num_workers "${NUM_WORKERS}"
  --trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
  --trainer.gradient_clipping "${GRADIENT_CLIPPING}"
  --trainer.num_warmup_steps "${WARMUP_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
  --trainer.keep_latest_checkpoint_only "${KEEP_LATEST_CHECKPOINT_ONLY}"
  --trainer.save_final_model "${SAVE_FINAL_MODEL}"
  --trainer.eval_interval "${EVAL_INTERVAL}"
  --trainer.logging_frequency "${LOGGING_FREQUENCY}"
  --trainer.learning_rate.qwen_vl_interface "${QWEN_LR}"
  --trainer.learning_rate.vj_predictor "${VJ_PREDICTOR_LR}"
  --trainer.learning_rate.action_model "${ACTION_LR}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  TRAIN_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  if [[ "${RESUME_FROM_CHECKPOINT}" != "latest" && ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
    echo "Resume checkpoint directory does not exist: ${RESUME_FROM_CHECKPOINT}" >&2
    exit 1
  fi
  TRAIN_ARGS+=(--trainer.resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  "${TRAIN_ARGS[@]}"
