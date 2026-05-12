#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# Train cross-encoder from distilled dataset (safe defaults)
# ==========================================================
# Usage:
#   bash cross_encoder/train.sh
#
# Optional overrides (example):
#   TRAIN_BATCH_SIZE=1 MINI_LIST_SIZE=6 LOSS_MSE_WEIGHT=0.35 \
#   bash cross_encoder/train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ===========================
# Config (env overrideable)
# ===========================
PYTHON_BIN="${PYTHON_BIN:-python}"

# Inputs / outputs
RAW_INPUT="${RAW_INPUT:-cross_encoder/dataset/distill/llm_distill2_raw_scores.jsonl}"
PAIRWISE_INPUT="${PAIRWISE_INPUT:-cross_encoder/dataset/distill/llm_distill2_pairwise.jsonl}"
SPLIT_DIR="${SPLIT_DIR:-cross_encoder/dataset/splits}"
OUTPUT_DIR="${OUTPUT_DIR:-cross_encoder/models/bge_reranker_distill}"
MODEL_ID="${MODEL_ID:-dleemiller/ModernCE-base-sts}"

# Train schedule
SEED="${SEED:-42}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-3}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-1}"
EVAL_TEST_AT_END="${EVAL_TEST_AT_END:-false}"                 # true | false
STAGE1_EARLY_STOP="${STAGE1_EARLY_STOP:-false}"                # true | false
STAGE1_EARLY_STOP_PATIENCE="${STAGE1_EARLY_STOP_PATIENCE:-1}"  # epochs
STAGE2_START_FROM_BEST_STAGE1="${STAGE2_START_FROM_BEST_STAGE1:-true}"  # true | false

# Memory-safe knobs
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-32}"
MINI_LIST_SIZE="${MINI_LIST_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"

# Loss calibration knobs (to improve absolute score behavior)
LOSS_KL_WEIGHT="${LOSS_KL_WEIGHT:-1.0}"
LOSS_PAIR_WEIGHT="${LOSS_PAIR_WEIGHT:-0.25}"
LOSS_MSE_WEIGHT="${LOSS_MSE_WEIGHT:-0.40}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.0}"

# Misc
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-100}"
MARGIN_MIN="${MARGIN_MIN:-0.25}"
MARGIN_MAX="${MARGIN_MAX:-1.00}"
PAIR_TYPE_MAX_SHARE="${PAIR_TYPE_MAX_SHARE:-0.35}"
PAIR_TYPE_WEIGHT_MAP="${PAIR_TYPE_WEIGHT_MAP:-strong_vs_weak=1.2,strong_vs_hard=1.1,strong_vs_boundary=1.0,llm_disagreement=0.8,default=1.0}"

USE_PREPARED_SPLITS="${USE_PREPARED_SPLITS:-true}"       # true | false
REGENERATE_SPLITS="${REGENERATE_SPLITS:-false}"          # true | false
APPEND_ARGS_TO_OUTPUT_DIR="${APPEND_ARGS_TO_OUTPUT_DIR:-true}"  # true | false
STRICT_PREPARED_SPLITS="${STRICT_PREPARED_SPLITS:-true}" # true | false
BF16="${BF16:-true}"                                     # true | false
FP16="${FP16:-false}"                                    # true | false
NO_WANDB="${NO_WANDB:-false}"                            # true | false
NO_TQDM="${NO_TQDM:-false}"                              # true | false

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

BOOL_TRUE() {
  [[ "$1" == "true" ]]
}

log "Starting training run"
log "raw_input=${RAW_INPUT}"
log "pairwise_input=${PAIRWISE_INPUT}"
log "split_dir=${SPLIT_DIR}"
log "output_dir=${OUTPUT_DIR}"
log "model_id=${MODEL_ID}"
log "stage1_epochs=${STAGE1_EPOCHS} stage2_epochs=${STAGE2_EPOCHS}"
log "eval_test_at_end=${EVAL_TEST_AT_END}"
log "stage1_early_stop=${STAGE1_EARLY_STOP} patience=${STAGE1_EARLY_STOP_PATIENCE} stage2_start_from_best_stage1=${STAGE2_START_FROM_BEST_STAGE1}"
log "use_prepared_splits=${USE_PREPARED_SPLITS} strict_prepared_splits=${STRICT_PREPARED_SPLITS}"
log "train_batch_size=${TRAIN_BATCH_SIZE} eval_batch_size=${EVAL_BATCH_SIZE} grad_accum_steps=${GRAD_ACCUM_STEPS}"
log "max_length=${MAX_LENGTH} candidate_pool_size=${CANDIDATE_POOL_SIZE} mini_list_size=${MINI_LIST_SIZE}"
log "loss_kl=${LOSS_KL_WEIGHT} loss_pair=${LOSS_PAIR_WEIGHT} loss_mse=${LOSS_MSE_WEIGHT} teacher_temperature=${TEACHER_TEMPERATURE}"
log "learning_rate=${LEARNING_RATE} margin_min=${MARGIN_MIN} margin_max=${MARGIN_MAX}"
log "pair_type_max_share=${PAIR_TYPE_MAX_SHARE}"
log "pair_type_weight_map=${PAIR_TYPE_WEIGHT_MAP}"

CMD=(
  "${PYTHON_BIN}" cross_encoder/train_bge_distill.py
  --raw-input "${RAW_INPUT}"
  --pairwise-input "${PAIRWISE_INPUT}"
  --split-dir "${SPLIT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --model-id "${MODEL_ID}"
  --seed "${SEED}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --stage1-early-stop-patience "${STAGE1_EARLY_STOP_PATIENCE}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --max-length "${MAX_LENGTH}"
  --candidate-pool-size "${CANDIDATE_POOL_SIZE}"
  --mini-list-size "${MINI_LIST_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --margin-min "${MARGIN_MIN}"
  --margin-max "${MARGIN_MAX}"
  --pair-type-max-share "${PAIR_TYPE_MAX_SHARE}"
  --pair-type-weight-map "${PAIR_TYPE_WEIGHT_MAP}"
  --loss-kl-weight "${LOSS_KL_WEIGHT}"
  --loss-pair-weight "${LOSS_PAIR_WEIGHT}"
  --loss-mse-weight "${LOSS_MSE_WEIGHT}"
  --teacher-temperature "${TEACHER_TEMPERATURE}"
  --log-every-steps "${LOG_EVERY_STEPS}"
  --eval-every-steps "${EVAL_EVERY_STEPS}"
)

if BOOL_TRUE "${USE_PREPARED_SPLITS}"; then
  CMD+=(--use-prepared-splits)
else
  CMD+=(--no-use-prepared-splits)
fi

if BOOL_TRUE "${STRICT_PREPARED_SPLITS}"; then
  if ! BOOL_TRUE "${USE_PREPARED_SPLITS}"; then
    echo "Error: STRICT_PREPARED_SPLITS=true requires USE_PREPARED_SPLITS=true" >&2
    exit 1
  fi
  REQUIRED_FILES=(
    "${SPLIT_DIR}/llm_distill_raw_train.jsonl"
    "${SPLIT_DIR}/llm_distill_raw_val.jsonl"
    "${SPLIT_DIR}/llm_distill_pairwise_train.jsonl"
    "${SPLIT_DIR}/llm_distill_pairwise_val.jsonl"
  )
  if BOOL_TRUE "${EVAL_TEST_AT_END}"; then
    REQUIRED_FILES+=(
      "${SPLIT_DIR}/llm_distill_raw_test.jsonl"
      "${SPLIT_DIR}/llm_distill_pairwise_test.jsonl"
    )
  fi
  for req in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${req}" ]]; then
      echo "Error: required split file not found: ${req}" >&2
      exit 1
    fi
  done
fi

if BOOL_TRUE "${EVAL_TEST_AT_END}"; then
  CMD+=(--eval-test-at-end)
else
  CMD+=(--no-eval-test-at-end)
fi

if BOOL_TRUE "${STAGE1_EARLY_STOP}"; then
  CMD+=(--stage1-early-stop)
else
  CMD+=(--no-stage1-early-stop)
fi

if BOOL_TRUE "${STAGE2_START_FROM_BEST_STAGE1}"; then
  CMD+=(--stage2-start-from-best-stage1)
else
  CMD+=(--no-stage2-start-from-best-stage1)
fi

if BOOL_TRUE "${REGENERATE_SPLITS}"; then
  CMD+=(--regenerate-splits)
else
  CMD+=(--no-regenerate-splits)
fi

if BOOL_TRUE "${APPEND_ARGS_TO_OUTPUT_DIR}"; then
  CMD+=(--append-args-to-output-dir)
else
  CMD+=(--no-append-args-to-output-dir)
fi

if BOOL_TRUE "${BF16}"; then
  CMD+=(--bf16)
elif BOOL_TRUE "${FP16}"; then
  CMD+=(--fp16)
fi

if BOOL_TRUE "${NO_WANDB}"; then
  CMD+=(--no-wandb)
fi

if BOOL_TRUE "${NO_TQDM}"; then
  CMD+=(--no-tqdm)
fi

"${CMD[@]}"

log "Done: training completed."
