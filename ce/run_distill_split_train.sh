#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# CE pipeline: distill(+augment) for domain+method -> split -> train2
# ==========================================================
# Usage:
#   bash ce/run_distill_split_train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }
bool_true() { [[ "$1" == "true" ]]; }

PYTHON_BIN="${PYTHON_BIN:-python}"

# ---------------------------
# Stage 1) Distill + Augment (both aspects)
# ---------------------------
TARGET_HIGH="${TARGET_HIGH:-4}"
TARGET_MID="${TARGET_MID:-8}"
TARGET_LOW="${TARGET_LOW:-4}"
PREFILTER_MULTIPLIER="${PREFILTER_MULTIPLIER:-30}"
MAX_SPECS="${MAX_SPECS:-0}"   # 0 = all

DOMAIN_ASPECT="${DOMAIN_ASPECT:-domain}"
METHOD_ASPECT="${METHOD_ASPECT:-method}"

# Distill outputs (inputs for split/train2)
RAW_INPUT="${RAW_INPUT:-ce/dataset/distill/llm_distill_domain_listwise.jsonl}"
METHOD_RAW_INPUT="${METHOD_RAW_INPUT:-ce/dataset/distill/llm_distill_method_listwise.jsonl}"
PAIRWISE_INPUT="${PAIRWISE_INPUT:-ce/dataset/distill/llm_distill_domain_pairwise.jsonl}"
METHOD_PAIRWISE_INPUT="${METHOD_PAIRWISE_INPUT:-ce/dataset/distill/llm_distill_method_pairwise.jsonl}"

# ---------------------------
# Stage 2) Shared split (listwise + pairwise)
# ---------------------------
SPLIT_DIR="${SPLIT_DIR:-ce/dataset/splits}"
SPLIT_SEED="${SPLIT_SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.10}"
TEST_RATIO="${TEST_RATIO:-0.10}"
SPLIT_OVERWRITE="${SPLIT_OVERWRITE:-true}"   # true | false

RAW_TRAIN_INPUT="${RAW_TRAIN_INPUT:-${SPLIT_DIR}/llm_distill_domain_listwise_train.jsonl}"
RAW_VAL_INPUT="${RAW_VAL_INPUT:-${SPLIT_DIR}/llm_distill_domain_listwise_val.jsonl}"
RAW_TEST_INPUT="${RAW_TEST_INPUT:-${SPLIT_DIR}/llm_distill_domain_listwise_test.jsonl}"
METHOD_RAW_TRAIN_INPUT="${METHOD_RAW_TRAIN_INPUT:-${SPLIT_DIR}/llm_distill_method_listwise_train.jsonl}"
METHOD_RAW_VAL_INPUT="${METHOD_RAW_VAL_INPUT:-${SPLIT_DIR}/llm_distill_method_listwise_val.jsonl}"
METHOD_RAW_TEST_INPUT="${METHOD_RAW_TEST_INPUT:-${SPLIT_DIR}/llm_distill_method_listwise_test.jsonl}"

PAIRWISE_TRAIN_INPUT="${PAIRWISE_TRAIN_INPUT:-${SPLIT_DIR}/llm_distill_domain_pairwise_train.jsonl}"
PAIRWISE_VAL_INPUT="${PAIRWISE_VAL_INPUT:-${SPLIT_DIR}/llm_distill_domain_pairwise_val.jsonl}"
PAIRWISE_TEST_INPUT="${PAIRWISE_TEST_INPUT:-${SPLIT_DIR}/llm_distill_domain_pairwise_test.jsonl}"
METHOD_PAIRWISE_TRAIN_INPUT="${METHOD_PAIRWISE_TRAIN_INPUT:-${SPLIT_DIR}/llm_distill_method_pairwise_train.jsonl}"
METHOD_PAIRWISE_VAL_INPUT="${METHOD_PAIRWISE_VAL_INPUT:-${SPLIT_DIR}/llm_distill_method_pairwise_val.jsonl}"
METHOD_PAIRWISE_TEST_INPUT="${METHOD_PAIRWISE_TEST_INPUT:-${SPLIT_DIR}/llm_distill_method_pairwise_test.jsonl}"

# ---------------------------
# Stage 3) Train2 (same knobs as ce/train2.sh)
# ---------------------------
OUTPUT_DIR="${OUTPUT_DIR:-ce/models/bge_reranker_distill}"
MODEL_ID="${MODEL_ID:-dleemiller/ModernCE-base-sts}"

SEED="${SEED:-42}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-4}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-4}"
STAGE1_EARLY_STOP="${STAGE1_EARLY_STOP:-true}"
STAGE1_EARLY_STOP_PATIENCE="${STAGE1_EARLY_STOP_PATIENCE:-2}"
STAGE2_START_FROM_BEST_STAGE1="${STAGE2_START_FROM_BEST_STAGE1:-true}"
STAGE2_EARLY_STOP="${STAGE2_EARLY_STOP:-true}"
STAGE2_EARLY_STOP_PATIENCE="${STAGE2_EARLY_STOP_PATIENCE:-2}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-32}"
MINI_LIST_SIZE="${MINI_LIST_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-100}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
STAGE1_LEARNING_RATE="${STAGE1_LEARNING_RATE:-1e-6}"
STAGE2_LEARNING_RATE="${STAGE2_LEARNING_RATE:-5e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MARGIN_MIN="${MARGIN_MIN:-0.32}"
MARGIN_MAX="${MARGIN_MAX:-1.0}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.2}"

LISTWISE_SCORE_MODE="${LISTWISE_SCORE_MODE:-raw}"  # raw | normalized
LOSS_KL_WEIGHT="${LOSS_KL_WEIGHT:-0.50}"
LOSS_PAIR_WEIGHT="${LOSS_PAIR_WEIGHT:-0.20}"
LOSS_MSE_WEIGHT="${LOSS_MSE_WEIGHT:-0.15}"
LOSS_CLUSTER_MARGIN_WEIGHT="${LOSS_CLUSTER_MARGIN_WEIGHT:-1.0}"
LOSS_CALIBRATION_BAND_WEIGHT="${LOSS_CALIBRATION_BAND_WEIGHT:-0.30}"
STAGE2_CLUSTER_SOURCE="${STAGE2_CLUSTER_SOURCE:-teacher_raw}"  # teacher_raw | teacher_normalized | target_cluster
STAGE2_CLUSTER_HIGH_THRESHOLD="${STAGE2_CLUSTER_HIGH_THRESHOLD:-0.70}"
STAGE2_CLUSTER_MID_THRESHOLD="${STAGE2_CLUSTER_MID_THRESHOLD:-0.30}"
CALIB_BAND_MODE="${CALIB_BAND_MODE:-fixed}"  # fixed | data_driven
CALIB_ANCHOR_STAT="${CALIB_ANCHOR_STAT:-mean}"  # mean | median
CALIB_HIGH_FLOOR="${CALIB_HIGH_FLOOR:-0.75}"
CALIB_MID_CENTER="${CALIB_MID_CENTER:-0.42}"
CALIB_MID_BANDWIDTH="${CALIB_MID_BANDWIDTH:-0.15}"
CALIB_LOW_CEIL="${CALIB_LOW_CEIL:-0.12}"

USE_PREPARED_SPLITS="${USE_PREPARED_SPLITS:-true}"  # true | false
REGENERATE_SPLITS="${REGENERATE_SPLITS:-false}"     # true | false
APPEND_ARGS_TO_OUTPUT_DIR="${APPEND_ARGS_TO_OUTPUT_DIR:-true}"  # true | false
BF16="${BF16:-true}"                                 # true | false
FP16="${FP16:-false}"                                # true | false
NO_WANDB="${NO_WANDB:-false}"                        # true | false
NO_TQDM="${NO_TQDM:-false}"                          # true | false
WANDB_PROJECT="${WANDB_PROJECT:-ce_distill}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_TAGS="${WANDB_TAGS:-ce,domain,method}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_DIR="${WANDB_DIR:-}"

log "Stage 1/3: Distill+augment (${DOMAIN_ASPECT}) with target ${TARGET_HIGH}/${TARGET_MID}/${TARGET_LOW}"
"${PYTHON_BIN}" ce/data_preparation/llm_distillation/llm_distillation.py \
  --run-mode full \
  --judge-aspect "${DOMAIN_ASPECT}" \
  --target-high "${TARGET_HIGH}" \
  --target-mid "${TARGET_MID}" \
  --target-low "${TARGET_LOW}" \
  --prefilter-multiplier "${PREFILTER_MULTIPLIER}" \
  --max-specs "${MAX_SPECS}"

log "Stage 1/3: Distill+augment (${METHOD_ASPECT}) with target ${TARGET_HIGH}/${TARGET_MID}/${TARGET_LOW}"
"${PYTHON_BIN}" ce/data_preparation/llm_distillation/llm_distillation.py \
  --run-mode full \
  --judge-aspect "${METHOD_ASPECT}" \
  --target-high "${TARGET_HIGH}" \
  --target-mid "${TARGET_MID}" \
  --target-low "${TARGET_LOW}" \
  --prefilter-multiplier "${PREFILTER_MULTIPLIER}" \
  --max-specs "${MAX_SPECS}"

log "Stage 2/3: Shared query split for domain+method listwise+pairwise"
SPLIT_ARGS=(
  --domain-input "${RAW_INPUT}"
  --method-input "${METHOD_RAW_INPUT}"
  --domain-pairwise-input "${PAIRWISE_INPUT}"
  --method-pairwise-input "${METHOD_PAIRWISE_INPUT}"
  --output-dir "${SPLIT_DIR}"
  --seed "${SPLIT_SEED}"
  --val-ratio "${VAL_RATIO}"
  --test-ratio "${TEST_RATIO}"
)
if bool_true "${SPLIT_OVERWRITE}"; then
  SPLIT_ARGS+=(--overwrite)
fi
"${PYTHON_BIN}" ce/data_preparation/split_distill_listwise_domain_method.py "${SPLIT_ARGS[@]}"

log "Stage 3/3: Train2 CE model"
CMD=(
  "${PYTHON_BIN}" ce/train2.py
  --raw-input "${RAW_INPUT}"
  --method-raw-input "${METHOD_RAW_INPUT}"
  --pairwise-input "${PAIRWISE_INPUT}"
  --method-pairwise-input "${METHOD_PAIRWISE_INPUT}"
  --split-dir "${SPLIT_DIR}"
  --raw-train-input "${RAW_TRAIN_INPUT}"
  --raw-val-input "${RAW_VAL_INPUT}"
  --raw-test-input "${RAW_TEST_INPUT}"
  --method-raw-train-input "${METHOD_RAW_TRAIN_INPUT}"
  --method-raw-val-input "${METHOD_RAW_VAL_INPUT}"
  --method-raw-test-input "${METHOD_RAW_TEST_INPUT}"
  --pairwise-train-input "${PAIRWISE_TRAIN_INPUT}"
  --pairwise-val-input "${PAIRWISE_VAL_INPUT}"
  --pairwise-test-input "${PAIRWISE_TEST_INPUT}"
  --method-pairwise-train-input "${METHOD_PAIRWISE_TRAIN_INPUT}"
  --method-pairwise-val-input "${METHOD_PAIRWISE_VAL_INPUT}"
  --method-pairwise-test-input "${METHOD_PAIRWISE_TEST_INPUT}"
  --output-dir "${OUTPUT_DIR}"
  --model-id "${MODEL_ID}"
  --seed "${SEED}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --stage1-early-stop-patience "${STAGE1_EARLY_STOP_PATIENCE}"
  --stage2-early-stop-patience "${STAGE2_EARLY_STOP_PATIENCE}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --max-length "${MAX_LENGTH}"
  --candidate-pool-size "${CANDIDATE_POOL_SIZE}"
  --mini-list-size "${MINI_LIST_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --log-every-steps "${LOG_EVERY_STEPS}"
  --eval-every-steps "${EVAL_EVERY_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --stage1-learning-rate "${STAGE1_LEARNING_RATE}"
  --stage2-learning-rate "${STAGE2_LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --margin-min "${MARGIN_MIN}"
  --margin-max "${MARGIN_MAX}"
  --teacher-temperature "${TEACHER_TEMPERATURE}"
  --listwise-score-mode "${LISTWISE_SCORE_MODE}"
  --loss-kl-weight "${LOSS_KL_WEIGHT}"
  --loss-pair-weight "${LOSS_PAIR_WEIGHT}"
  --loss-mse-weight "${LOSS_MSE_WEIGHT}"
  --loss-cluster-margin-weight "${LOSS_CLUSTER_MARGIN_WEIGHT}"
  --loss-calibration-band-weight "${LOSS_CALIBRATION_BAND_WEIGHT}"
  --stage2-cluster-source "${STAGE2_CLUSTER_SOURCE}"
  --stage2-cluster-high-threshold "${STAGE2_CLUSTER_HIGH_THRESHOLD}"
  --stage2-cluster-mid-threshold "${STAGE2_CLUSTER_MID_THRESHOLD}"
  --calib-band-mode "${CALIB_BAND_MODE}"
  --calib-anchor-stat "${CALIB_ANCHOR_STAT}"
  --calib-high-floor "${CALIB_HIGH_FLOOR}"
  --calib-mid-center "${CALIB_MID_CENTER}"
  --calib-mid-bandwidth "${CALIB_MID_BANDWIDTH}"
  --calib-low-ceil "${CALIB_LOW_CEIL}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-run-name "${WANDB_RUN_NAME}"
  --wandb-tags "${WANDB_TAGS}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-dir "${WANDB_DIR}"
)

if bool_true "${USE_PREPARED_SPLITS}"; then
  CMD+=(--use-prepared-splits)
else
  CMD+=(--no-use-prepared-splits)
fi
if bool_true "${REGENERATE_SPLITS}"; then
  CMD+=(--regenerate-splits)
else
  CMD+=(--no-regenerate-splits)
fi
if bool_true "${APPEND_ARGS_TO_OUTPUT_DIR}"; then
  CMD+=(--append-args-to-output-dir)
else
  CMD+=(--no-append-args-to-output-dir)
fi
if bool_true "${STAGE1_EARLY_STOP}"; then
  CMD+=(--stage1-early-stop)
else
  CMD+=(--no-stage1-early-stop)
fi
if bool_true "${STAGE2_START_FROM_BEST_STAGE1}"; then
  CMD+=(--stage2-start-from-best-stage1)
else
  CMD+=(--no-stage2-start-from-best-stage1)
fi
if bool_true "${STAGE2_EARLY_STOP}"; then
  CMD+=(--stage2-early-stop)
else
  CMD+=(--no-stage2-early-stop)
fi
if bool_true "${BF16}"; then
  CMD+=(--bf16)
elif bool_true "${FP16}"; then
  CMD+=(--fp16)
fi
if bool_true "${NO_WANDB}"; then
  CMD+=(--no-wandb)
fi
if bool_true "${NO_TQDM}"; then
  CMD+=(--no-tqdm)
fi

log "Running: ${CMD[*]}"
"${CMD[@]}"

log "Done: domain+method distill/augment -> split -> train2 completed."
