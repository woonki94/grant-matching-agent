#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# Train cross-encoder Stage2 only (standalone)
# ==========================================================
# Usage:
#   1) Edit MODEL_ID below to your Stage1 checkpoint path, then:
#      bash cross_encoder/train_stage2_only.sh
#   2) Or override at runtime:
#      MODEL_ID=/path/to/stage1_checkpoint bash cross_encoder/train_stage2_only.sh
#
# This script is intentionally standalone (independent from train.sh).
# Defaults mirror train.sh, except stage controls are Stage2-only.

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

# Set this to your Stage1 checkpoint directory.
MODEL_ID="${MODEL_ID:-/nfs/hpc/share/kimwoon/grant-matching-agent/cross_encoder/models/bge_reranker_distill__sd42_s15_s25_bs2_ga16_cp32_ml8_lr5em07_t1p2_kl0p8_pw0p2_mse0p45_cm1_cb0p45/stage1_epoch_5}"

# Train schedule (stage2 only)
SEED="${SEED:-42}"
STAGE1_EPOCHS=0
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
STAGE1_EARLY_STOP="${STAGE1_EARLY_STOP:-true}"                # true | false
STAGE1_EARLY_STOP_PATIENCE="${STAGE1_EARLY_STOP_PATIENCE:-1}"  # epochs
STAGE2_START_FROM_BEST_STAGE1=false

# Memory-safe knobs
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-32}"
MINI_LIST_SIZE="${MINI_LIST_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"

# Loss calibration knobs (stage2-only target-shaping defaults)
LOSS_KL_WEIGHT="${LOSS_KL_WEIGHT:-0.35}"
LOSS_PAIR_WEIGHT="${LOSS_PAIR_WEIGHT:-0.15}"
LOSS_MSE_WEIGHT="${LOSS_MSE_WEIGHT:-0.20}"
LISTWISE_SCORE_MODE="${LISTWISE_SCORE_MODE:-raw}"        # raw | normalized
LOSS_CLUSTER_MARGIN_WEIGHT="${LOSS_CLUSTER_MARGIN_WEIGHT:-1.60}"
LOSS_CALIBRATION_BAND_WEIGHT="${LOSS_CALIBRATION_BAND_WEIGHT:-1.40}"
CLUSTER_MARGIN_HM="${CLUSTER_MARGIN_HM:-0.34}"
CLUSTER_MARGIN_ML="${CLUSTER_MARGIN_ML:-0.34}"
CLUSTER_MARGIN_HL="${CLUSTER_MARGIN_HL:-0.68}"
CALIB_BAND_MODE="${CALIB_BAND_MODE:-fixed}"        # fixed | data_driven
CALIB_ANCHOR_STAT="${CALIB_ANCHOR_STAT:-mean}"           # mean | median
CALIB_HIGH_FLOOR="${CALIB_HIGH_FLOOR:-0.88}"
CALIB_MID_CENTER="${CALIB_MID_CENTER:-0.34}"
CALIB_MID_BANDWIDTH="${CALIB_MID_BANDWIDTH:-0.05}"
CALIB_LOW_CEIL="${CALIB_LOW_CEIL:-0.07}"
CALIB_DATA_HIGH_SLACK="${CALIB_DATA_HIGH_SLACK:-0.10}"
CALIB_DATA_LOW_SLACK="${CALIB_DATA_LOW_SLACK:-0.03}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.2}"
MARGIN_MIN="${MARGIN_MIN:-0.25}"
MARGIN_MAX="${MARGIN_MAX:-1.0}"
PAIR_ADD_MID_LOWER_MID="${PAIR_ADD_MID_LOWER_MID:-true}"   # true | false
PAIR_MID_ADD_EASY_CONTRAST="${PAIR_MID_ADD_EASY_CONTRAST:-false}"  # true | false
PAIR_MID_POS_SCORE_MIN="${PAIR_MID_POS_SCORE_MIN:-0.4}"
PAIR_MID_POS_SCORE_MAX="${PAIR_MID_POS_SCORE_MAX:-0.7}"
PAIR_MID_NEG_SCORE_MIN="${PAIR_MID_NEG_SCORE_MIN:-0.2}"
PAIR_MID_NEG_SCORE_MAX="${PAIR_MID_NEG_SCORE_MAX:-0.5}"
PAIR_MID_MARGIN_MIN="${PAIR_MID_MARGIN_MIN:-0.05}"
PAIR_MID_MARGIN_MAX="${PAIR_MID_MARGIN_MAX:-0.4}"
PAIR_MID_MIN_CANDIDATES="${PAIR_MID_MIN_CANDIDATES:-6}"
PAIR_MID_START_RATIO="${PAIR_MID_START_RATIO:-0.30}"
PAIR_MID_END_RATIO="${PAIR_MID_END_RATIO:-0.55}"
PAIR_LOWER_MID_START_RATIO="${PAIR_LOWER_MID_START_RATIO:-0.55}"
PAIR_LOWER_MID_END_RATIO="${PAIR_LOWER_MID_END_RATIO:-0.80}"

# Misc
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-50}"

USE_PREPARED_SPLITS="${USE_PREPARED_SPLITS:-true}"       # true | false
REGENERATE_SPLITS="${REGENERATE_SPLITS:-false}"          # true | false
APPEND_ARGS_TO_OUTPUT_DIR="${APPEND_ARGS_TO_OUTPUT_DIR:-true}"  # true | false
BF16="${BF16:-true}"                                     # true | false
FP16="${FP16:-false}"                                    # true | false
NO_WANDB="${NO_WANDB:-false}"                            # true | false
NO_TQDM="${NO_TQDM:-false}"                              # true | false

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

BOOL_TRUE() {
  [[ "$1" == "true" ]]
}

if [[ "${MODEL_ID}" == "/path/to/stage1_checkpoint" ]]; then
  echo "ERROR: Set MODEL_ID to your Stage1 checkpoint path in this script or via env override."
  exit 1
fi

log "Starting stage2-only training run"
log "raw_input=${RAW_INPUT}"
log "pairwise_input=${PAIRWISE_INPUT}"
log "split_dir=${SPLIT_DIR}"
log "output_dir=${OUTPUT_DIR}"
log "model_id=${MODEL_ID}"
log "stage1_epochs=${STAGE1_EPOCHS} stage2_epochs=${STAGE2_EPOCHS}"
log "stage1_early_stop=${STAGE1_EARLY_STOP} patience=${STAGE1_EARLY_STOP_PATIENCE} stage2_start_from_best_stage1=${STAGE2_START_FROM_BEST_STAGE1}"
log "train_batch_size=${TRAIN_BATCH_SIZE} eval_batch_size=${EVAL_BATCH_SIZE} grad_accum_steps=${GRAD_ACCUM_STEPS}"
log "max_length=${MAX_LENGTH} candidate_pool_size=${CANDIDATE_POOL_SIZE} mini_list_size=${MINI_LIST_SIZE}"
log "loss_kl=${LOSS_KL_WEIGHT} loss_pair=${LOSS_PAIR_WEIGHT} loss_mse=${LOSS_MSE_WEIGHT} teacher_temperature=${TEACHER_TEMPERATURE}"
log "listwise_score_mode=${LISTWISE_SCORE_MODE}"
log "loss_cluster_margin=${LOSS_CLUSTER_MARGIN_WEIGHT} loss_calibration_band=${LOSS_CALIBRATION_BAND_WEIGHT}"
log "cluster_margin_targets hm/ml/hl=${CLUSTER_MARGIN_HM}/${CLUSTER_MARGIN_ML}/${CLUSTER_MARGIN_HL}"
log "calibration mode=${CALIB_BAND_MODE} anchor_stat=${CALIB_ANCHOR_STAT} fixed=${CALIB_HIGH_FLOOR}/${CALIB_MID_CENTER}+/-${CALIB_MID_BANDWIDTH}/${CALIB_LOW_CEIL} data_slack=${CALIB_DATA_HIGH_SLACK}/${CALIB_DATA_LOW_SLACK}"
log "margin_min=${MARGIN_MIN} margin_max=${MARGIN_MAX}"
log "pair_add_mid_lower_mid=${PAIR_ADD_MID_LOWER_MID} pair_mid_add_easy_contrast=${PAIR_MID_ADD_EASY_CONTRAST}"
log "pair_mid_score_range pos=[${PAIR_MID_POS_SCORE_MIN},${PAIR_MID_POS_SCORE_MAX}] neg=[${PAIR_MID_NEG_SCORE_MIN},${PAIR_MID_NEG_SCORE_MAX}] pair_mid_margin=[${PAIR_MID_MARGIN_MIN},${PAIR_MID_MARGIN_MAX}]"
log "pair_mid_min_candidates=${PAIR_MID_MIN_CANDIDATES} mid_ratio=[${PAIR_MID_START_RATIO},${PAIR_MID_END_RATIO}] lower_mid_ratio=[${PAIR_LOWER_MID_START_RATIO},${PAIR_LOWER_MID_END_RATIO}]"

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
  --loss-kl-weight "${LOSS_KL_WEIGHT}"
  --loss-pair-weight "${LOSS_PAIR_WEIGHT}"
  --loss-mse-weight "${LOSS_MSE_WEIGHT}"
  --loss-cluster-margin-weight "${LOSS_CLUSTER_MARGIN_WEIGHT}"
  --loss-calibration-band-weight "${LOSS_CALIBRATION_BAND_WEIGHT}"
  --cluster-margin-hm "${CLUSTER_MARGIN_HM}"
  --cluster-margin-ml "${CLUSTER_MARGIN_ML}"
  --cluster-margin-hl "${CLUSTER_MARGIN_HL}"
  --listwise-score-mode "${LISTWISE_SCORE_MODE}"
  --calib-band-mode "${CALIB_BAND_MODE}"
  --calib-anchor-stat "${CALIB_ANCHOR_STAT}"
  --calib-high-floor "${CALIB_HIGH_FLOOR}"
  --calib-mid-center "${CALIB_MID_CENTER}"
  --calib-mid-bandwidth "${CALIB_MID_BANDWIDTH}"
  --calib-low-ceil "${CALIB_LOW_CEIL}"
  --calib-data-high-slack "${CALIB_DATA_HIGH_SLACK}"
  --calib-data-low-slack "${CALIB_DATA_LOW_SLACK}"
  --teacher-temperature "${TEACHER_TEMPERATURE}"
  --pair-mid-pos-score-min "${PAIR_MID_POS_SCORE_MIN}"
  --pair-mid-pos-score-max "${PAIR_MID_POS_SCORE_MAX}"
  --pair-mid-neg-score-min "${PAIR_MID_NEG_SCORE_MIN}"
  --pair-mid-neg-score-max "${PAIR_MID_NEG_SCORE_MAX}"
  --pair-mid-margin-min "${PAIR_MID_MARGIN_MIN}"
  --pair-mid-margin-max "${PAIR_MID_MARGIN_MAX}"
  --pair-mid-min-candidates "${PAIR_MID_MIN_CANDIDATES}"
  --pair-mid-start-ratio "${PAIR_MID_START_RATIO}"
  --pair-mid-end-ratio "${PAIR_MID_END_RATIO}"
  --pair-lower-mid-start-ratio "${PAIR_LOWER_MID_START_RATIO}"
  --pair-lower-mid-end-ratio "${PAIR_LOWER_MID_END_RATIO}"
  --log-every-steps "${LOG_EVERY_STEPS}"
  --eval-every-steps "${EVAL_EVERY_STEPS}"
)

if BOOL_TRUE "${USE_PREPARED_SPLITS}"; then
  CMD+=(--use-prepared-splits)
else
  CMD+=(--no-use-prepared-splits)
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

if BOOL_TRUE "${PAIR_ADD_MID_LOWER_MID}"; then
  CMD+=(--pair-add-mid-lower-mid)
fi

if BOOL_TRUE "${PAIR_MID_ADD_EASY_CONTRAST}"; then
  CMD+=(--pair-mid-add-easy-contrast)
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

log "Done: stage2-only training completed."
