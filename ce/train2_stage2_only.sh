#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# CE Stage2-only runner (train2.py) from an existing checkpoint
# ==========================================================
# Usage:
#   MODEL_ID=/path/to/stage1_epoch_1 \
#   bash ce/train2_stage2_only.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }
bool_true() { [[ "$1" == "true" ]]; }

PYTHON_BIN="${PYTHON_BIN:-python}"

# Inputs (same surface as run_distill_split_train.sh train stage)
RAW_INPUT="${RAW_INPUT:-ce/dataset/distill/llm_distill_domain_listwise.jsonl}"
METHOD_RAW_INPUT="${METHOD_RAW_INPUT:-ce/dataset/distill/llm_distill_method_listwise.jsonl}"
PAIRWISE_INPUT="${PAIRWISE_INPUT:-ce/dataset/distill/llm_distill_domain_pairwise.jsonl}"
METHOD_PAIRWISE_INPUT="${METHOD_PAIRWISE_INPUT:-ce/dataset/distill/llm_distill_method_pairwise.jsonl}"

SPLIT_DIR="${SPLIT_DIR:-ce/dataset/splits}"
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

# Outputs / model
OUTPUT_DIR="${OUTPUT_DIR:-ce/models/bge_reranker_distill_stage2_only}"
MODEL_ID="${MODEL_ID:-/nfs/stak/users/kimwoon/hpc-share/grant-matching-agent/ce/models/bge_reranker_distill__sd42_s15_s25_bs2_ga16_cp48_ml12_lr5em07_lr11p1em06_lr24p5em07_t1p2_kl0p5_pw0p24_mse0p22_cm0p85_cb0p65_dpw1_mpw1p2_dlw1_mlw0p9/stage1_epoch_3}"
BASE_MODEL="${BASE_MODEL:-dleemiller/ModernCE-base-sts}"

# Training schedule (same knobs; Stage1 fixed to 0)
SEED="${SEED:-42}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-0}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-14}"
STAGE1_EARLY_STOP="${STAGE1_EARLY_STOP:-true}"
STAGE1_EARLY_STOP_PATIENCE="${STAGE1_EARLY_STOP_PATIENCE:-2}"
STAGE2_START_FROM_BEST_STAGE1="${STAGE2_START_FROM_BEST_STAGE1:-true}"
STAGE2_EARLY_STOP="${STAGE2_EARLY_STOP:-true}"
STAGE2_EARLY_STOP_PATIENCE="${STAGE2_EARLY_STOP_PATIENCE:-6}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-48}"
MINI_LIST_SIZE="${MINI_LIST_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-100}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
STAGE1_LEARNING_RATE="${STAGE1_LEARNING_RATE:-1e-6}"
STAGE2_LEARNING_RATE="${STAGE2_LEARNING_RATE:-3.2e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MARGIN_MIN="${MARGIN_MIN:-0.33}"
MARGIN_MAX="${MARGIN_MAX:-1.0}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.2}"

LISTWISE_SCORE_MODE="${LISTWISE_SCORE_MODE:-raw}"  # raw | normalized
LOSS_KL_WEIGHT="${LOSS_KL_WEIGHT:-0.64}"
LOSS_PAIR_WEIGHT="${LOSS_PAIR_WEIGHT:-0.12}"
LOSS_MSE_WEIGHT="${LOSS_MSE_WEIGHT:-0.36}"
LOSS_CLUSTER_MARGIN_WEIGHT="${LOSS_CLUSTER_MARGIN_WEIGHT:-0.55}"
LOSS_CALIBRATION_BAND_WEIGHT="${LOSS_CALIBRATION_BAND_WEIGHT:-0.95}"
DOMAIN_PAIR_LOSS_SCALE="${DOMAIN_PAIR_LOSS_SCALE:-1.0}"
METHOD_PAIR_LOSS_SCALE="${METHOD_PAIR_LOSS_SCALE:-0.95}"
DOMAIN_LIST_LOSS_SCALE="${DOMAIN_LIST_LOSS_SCALE:-1.22}"
METHOD_LIST_LOSS_SCALE="${METHOD_LIST_LOSS_SCALE:-1.05}"
DOMAIN_CALIBRATION_HIGH_SCALE="${DOMAIN_CALIBRATION_HIGH_SCALE:-1.75}"
DOMAIN_CALIBRATION_MID_SCALE="${DOMAIN_CALIBRATION_MID_SCALE:-1.65}"
DOMAIN_CALIBRATION_LOW_SCALE="${DOMAIN_CALIBRATION_LOW_SCALE:-1.05}"
METHOD_CALIBRATION_HIGH_SCALE="${METHOD_CALIBRATION_HIGH_SCALE:-1.55}"
METHOD_CALIBRATION_MID_SCALE="${METHOD_CALIBRATION_MID_SCALE:-0.85}"
METHOD_CALIBRATION_LOW_SCALE="${METHOD_CALIBRATION_LOW_SCALE:-1.65}"
PAIR_ADD_MID_LOWER_MID="${PAIR_ADD_MID_LOWER_MID:-true}"   # true | false
PAIR_MID_POS_SCORE_MIN="${PAIR_MID_POS_SCORE_MIN:-0.40}"
PAIR_MID_POS_SCORE_MAX="${PAIR_MID_POS_SCORE_MAX:-0.72}"
PAIR_MID_NEG_SCORE_MIN="${PAIR_MID_NEG_SCORE_MIN:-0.18}"
PAIR_MID_NEG_SCORE_MAX="${PAIR_MID_NEG_SCORE_MAX:-0.52}"
PAIR_MID_MARGIN_MIN="${PAIR_MID_MARGIN_MIN:-0.05}"
PAIR_MID_MARGIN_MAX="${PAIR_MID_MARGIN_MAX:-0.36}"
PAIR_MID_ADD_EASY_CONTRAST="${PAIR_MID_ADD_EASY_CONTRAST:-true}"  # true | false
PAIR_TYPE_WEIGHT_MAP="${PAIR_TYPE_WEIGHT_MAP:-default=1.0,llm_disagreement=1.18,strong_vs_boundary=1.10,strong_vs_weak=0.98,strong_vs_hard=1.0}"
PAIR_TYPE_MAX_SHARE="${PAIR_TYPE_MAX_SHARE:-1.0}"
CLUSTER_MARGIN_HM="${CLUSTER_MARGIN_HM:-0.22}"
CLUSTER_MARGIN_ML="${CLUSTER_MARGIN_ML:-0.22}"
CLUSTER_MARGIN_HL="${CLUSTER_MARGIN_HL:-0.50}"
STAGE2_CLUSTER_SOURCE="${STAGE2_CLUSTER_SOURCE:-teacher_raw}"  # teacher_raw | teacher_normalized | target_cluster
STAGE2_CLUSTER_HIGH_THRESHOLD="${STAGE2_CLUSTER_HIGH_THRESHOLD:-0.70}"
STAGE2_CLUSTER_MID_THRESHOLD="${STAGE2_CLUSTER_MID_THRESHOLD:-0.30}"
CALIB_BAND_MODE="${CALIB_BAND_MODE:-fixed}"  # fixed | data_driven
CALIB_ANCHOR_STAT="${CALIB_ANCHOR_STAT:-mean}"  # mean | median
CALIB_HIGH_FLOOR="${CALIB_HIGH_FLOOR:-0.80}"
CALIB_MID_CENTER="${CALIB_MID_CENTER:-0.50}"
CALIB_MID_BANDWIDTH="${CALIB_MID_BANDWIDTH:-0.20}"
CALIB_LOW_CEIL="${CALIB_LOW_CEIL:-0.060}"
LOSS_CALIBRATION_HIGH_WEIGHT="${LOSS_CALIBRATION_HIGH_WEIGHT:-2.2}"
LOSS_CALIBRATION_MID_WEIGHT="${LOSS_CALIBRATION_MID_WEIGHT:-2.4}"
LOSS_CALIBRATION_LOW_WEIGHT="${LOSS_CALIBRATION_LOW_WEIGHT:-2.6}"
STAGE2_OOB_SELECTION_SPLIT="${STAGE2_OOB_SELECTION_SPLIT:-val}"  # val | test
STAGE2_OOB_HIGH_WEIGHT="${STAGE2_OOB_HIGH_WEIGHT:-4.0}"
STAGE2_OOB_MID_WEIGHT="${STAGE2_OOB_MID_WEIGHT:-4.5}"
STAGE2_OOB_LOW_WEIGHT="${STAGE2_OOB_LOW_WEIGHT:-3.8}"
STAGE2_GATED_SELECTION="${STAGE2_GATED_SELECTION:-true}"  # true | false
STAGE2_GATED_RANKING_METRIC="${STAGE2_GATED_RANKING_METRIC:-ndcg@10}" # ndcg@10 | mrr@10 | recall@50
STAGE2_GATED_NDCG_MIN="${STAGE2_GATED_NDCG_MIN:-0.955}"
STAGE2_GATED_MRR_MIN="${STAGE2_GATED_MRR_MIN:-0.70}"
STAGE2_GATED_RECALL_MIN="${STAGE2_GATED_RECALL_MIN:-0.70}"
STAGE2_POSTHOC_CALIBRATION="${STAGE2_POSTHOC_CALIBRATION:-true}"  # true | false
STAGE2_POSTHOC_CALIBRATION_FIT_SPLIT="${STAGE2_POSTHOC_CALIBRATION_FIT_SPLIT:-val}" # val | test

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

# Optional eval after training
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"  # true | false
EVAL_BASE_MODEL="${EVAL_BASE_MODEL:-${BASE_MODEL}}"
EVAL_DOMAIN_INPUT="${EVAL_DOMAIN_INPUT:-${RAW_TEST_INPUT}}"
EVAL_METHOD_INPUT="${EVAL_METHOD_INPUT:-${METHOD_RAW_TEST_INPUT}}"
EVAL_SCORE_FIELD="${EVAL_SCORE_FIELD:-teacher_score_raw}"
EVAL_ONLY_SELECTED="${EVAL_ONLY_SELECTED:-false}"  # true | false
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-512}"
EVAL_HIGH_THRESHOLD="${EVAL_HIGH_THRESHOLD:-0.70}"
EVAL_MID_THRESHOLD="${EVAL_MID_THRESHOLD:-0.30}"
EVAL_OOB_MARGIN="${EVAL_OOB_MARGIN:-0.0}"
EVAL_ORDER_TOP_K="${EVAL_ORDER_TOP_K:-5}"
EVAL_PAIR_EPS="${EVAL_PAIR_EPS:-0.01}"
EVAL_HARD_GAP_MAX="${EVAL_HARD_GAP_MAX:-0.15}"
EVAL_MEDIUM_GAP_MAX="${EVAL_MEDIUM_GAP_MAX:-0.40}"
EVAL_SAVE="${EVAL_SAVE:-true}"    # true | false
EVAL_PRINT="${EVAL_PRINT:-true}"  # true | false
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-ce/eval/results}"
EVAL_SAVE_PREFIX="${EVAL_SAVE_PREFIX:-ce_distill_margin_compare_stage2_only}"
EVAL_MODEL_PICK="${EVAL_MODEL_PICK:-best_stage2_gated}"  # best_stage2_gated | best_stage2_ranking | best_val_oob | final_stage2 | auto

log "Stage2-only train2 run"
log "model_id=${MODEL_ID} stage1_epochs=${STAGE1_EPOCHS} stage2_epochs=${STAGE2_EPOCHS}"
log "split_dir=${SPLIT_DIR} output_dir=${OUTPUT_DIR}"

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
  --domain-pair-loss-scale "${DOMAIN_PAIR_LOSS_SCALE}"
  --method-pair-loss-scale "${METHOD_PAIR_LOSS_SCALE}"
  --domain-list-loss-scale "${DOMAIN_LIST_LOSS_SCALE}"
  --method-list-loss-scale "${METHOD_LIST_LOSS_SCALE}"
  --domain-calibration-high-scale "${DOMAIN_CALIBRATION_HIGH_SCALE}"
  --domain-calibration-mid-scale "${DOMAIN_CALIBRATION_MID_SCALE}"
  --domain-calibration-low-scale "${DOMAIN_CALIBRATION_LOW_SCALE}"
  --method-calibration-high-scale "${METHOD_CALIBRATION_HIGH_SCALE}"
  --method-calibration-mid-scale "${METHOD_CALIBRATION_MID_SCALE}"
  --method-calibration-low-scale "${METHOD_CALIBRATION_LOW_SCALE}"
  --pair-mid-pos-score-min "${PAIR_MID_POS_SCORE_MIN}"
  --pair-mid-pos-score-max "${PAIR_MID_POS_SCORE_MAX}"
  --pair-mid-neg-score-min "${PAIR_MID_NEG_SCORE_MIN}"
  --pair-mid-neg-score-max "${PAIR_MID_NEG_SCORE_MAX}"
  --pair-mid-margin-min "${PAIR_MID_MARGIN_MIN}"
  --pair-mid-margin-max "${PAIR_MID_MARGIN_MAX}"
  --pair-type-weight-map "${PAIR_TYPE_WEIGHT_MAP}"
  --pair-type-max-share "${PAIR_TYPE_MAX_SHARE}"
  --cluster-margin-hm "${CLUSTER_MARGIN_HM}"
  --cluster-margin-ml "${CLUSTER_MARGIN_ML}"
  --cluster-margin-hl "${CLUSTER_MARGIN_HL}"
  --loss-calibration-band-weight "${LOSS_CALIBRATION_BAND_WEIGHT}"
  --loss-calibration-high-weight "${LOSS_CALIBRATION_HIGH_WEIGHT}"
  --loss-calibration-mid-weight "${LOSS_CALIBRATION_MID_WEIGHT}"
  --loss-calibration-low-weight "${LOSS_CALIBRATION_LOW_WEIGHT}"
  --stage2-cluster-source "${STAGE2_CLUSTER_SOURCE}"
  --stage2-cluster-high-threshold "${STAGE2_CLUSTER_HIGH_THRESHOLD}"
  --stage2-cluster-mid-threshold "${STAGE2_CLUSTER_MID_THRESHOLD}"
  --calib-band-mode "${CALIB_BAND_MODE}"
  --calib-anchor-stat "${CALIB_ANCHOR_STAT}"
  --calib-high-floor "${CALIB_HIGH_FLOOR}"
  --calib-mid-center "${CALIB_MID_CENTER}"
  --calib-mid-bandwidth "${CALIB_MID_BANDWIDTH}"
  --calib-low-ceil "${CALIB_LOW_CEIL}"
  --stage2-oob-selection-split "${STAGE2_OOB_SELECTION_SPLIT}"
  --stage2-oob-high-weight "${STAGE2_OOB_HIGH_WEIGHT}"
  --stage2-oob-mid-weight "${STAGE2_OOB_MID_WEIGHT}"
  --stage2-oob-low-weight "${STAGE2_OOB_LOW_WEIGHT}"
  --stage2-gated-ranking-metric "${STAGE2_GATED_RANKING_METRIC}"
  --stage2-gated-ndcg-min "${STAGE2_GATED_NDCG_MIN}"
  --stage2-gated-mrr-min "${STAGE2_GATED_MRR_MIN}"
  --stage2-gated-recall-min "${STAGE2_GATED_RECALL_MIN}"
  --stage2-posthoc-calibration-fit-split "${STAGE2_POSTHOC_CALIBRATION_FIT_SPLIT}"
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
if bool_true "${PAIR_ADD_MID_LOWER_MID}"; then
  CMD+=(--pair-add-mid-lower-mid)
fi
if bool_true "${PAIR_MID_ADD_EASY_CONTRAST}"; then
  CMD+=(--pair-mid-add-easy-contrast)
fi
if bool_true "${STAGE2_GATED_SELECTION}"; then
  CMD+=(--stage2-gated-selection)
else
  CMD+=(--no-stage2-gated-selection)
fi
if bool_true "${STAGE2_POSTHOC_CALIBRATION}"; then
  CMD+=(--stage2-posthoc-calibration)
else
  CMD+=(--no-stage2-posthoc-calibration)
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

if bool_true "${EVAL_AFTER_TRAIN}"; then
  log "Post-train eval with ce/eval/eval_finetuned_model.py"

  TRAIN_RUN_DIR="${OUTPUT_DIR}"
  if bool_true "${APPEND_ARGS_TO_OUTPUT_DIR}"; then
    TRAIN_PARENT_DIR="$(dirname "${OUTPUT_DIR}")"
    TRAIN_BASE_NAME="$(basename "${OUTPUT_DIR}")"
    LATEST_APPENDED_RUN="$(ls -td "${TRAIN_PARENT_DIR}/${TRAIN_BASE_NAME}"__* 2>/dev/null | head -n 1 || true)"
    if [[ -n "${LATEST_APPENDED_RUN}" && -d "${LATEST_APPENDED_RUN}" ]]; then
      TRAIN_RUN_DIR="${LATEST_APPENDED_RUN}"
    fi
  fi

  case "${EVAL_MODEL_PICK}" in
    best_stage2_gated)
      EVAL_FINETUNED_MODEL="${TRAIN_RUN_DIR}/best_stage2_gated"
      ;;
    best_stage2_ranking)
      EVAL_FINETUNED_MODEL="${TRAIN_RUN_DIR}/best_stage2_ranking"
      ;;
    best_val_oob)
      EVAL_FINETUNED_MODEL="${TRAIN_RUN_DIR}/best_val_oob"
      ;;
    final_stage2)
      EVAL_FINETUNED_MODEL="${TRAIN_RUN_DIR}/stage2_epoch_${STAGE2_EPOCHS}"
      ;;
    auto|*)
      EVAL_FINETUNED_MODEL="${TRAIN_RUN_DIR}/best_stage2_gated"
      ;;
  esac
  if [[ ! -d "${EVAL_FINETUNED_MODEL}" ]]; then
    EVAL_FINETUNED_MODEL="${TRAIN_RUN_DIR}/best_stage2_ranking"
  fi
  if [[ ! -d "${EVAL_FINETUNED_MODEL}" ]]; then
    EVAL_FINETUNED_MODEL="${TRAIN_RUN_DIR}/best_val_oob"
  fi
  if [[ ! -d "${EVAL_FINETUNED_MODEL}" ]]; then
    EVAL_FINETUNED_MODEL="${TRAIN_RUN_DIR}/stage2_epoch_${STAGE2_EPOCHS}"
  fi
  if [[ ! -d "${EVAL_FINETUNED_MODEL}" ]]; then
    EVAL_FINETUNED_MODEL="$(ls -d "${TRAIN_RUN_DIR}"/stage2_epoch_* 2>/dev/null | sort -V | tail -n 1 || true)"
  fi
  if [[ -z "${EVAL_FINETUNED_MODEL}" || ! -d "${EVAL_FINETUNED_MODEL}" ]]; then
    EVAL_FINETUNED_MODEL="$(ls -d "${TRAIN_RUN_DIR}"/stage1_epoch_* 2>/dev/null | sort -V | tail -n 1 || true)"
  fi
  if [[ -z "${EVAL_FINETUNED_MODEL}" || ! -d "${EVAL_FINETUNED_MODEL}" ]]; then
    EVAL_FINETUNED_MODEL="${TRAIN_RUN_DIR}"
  fi

  EVAL_CMD=(
    "${PYTHON_BIN}" ce/eval/eval_finetuned_model.py
    --no-auto-resolve-finetuned
    --finetuned-model "${EVAL_FINETUNED_MODEL}"
    --base-model "${EVAL_BASE_MODEL}"
    --domain-input "${EVAL_DOMAIN_INPUT}"
    --method-input "${EVAL_METHOD_INPUT}"
    --score-field "${EVAL_SCORE_FIELD}"
    --batch-size "${EVAL_BATCH_SIZE}"
    --max-length "${EVAL_MAX_LENGTH}"
    --high-threshold "${EVAL_HIGH_THRESHOLD}"
    --mid-threshold "${EVAL_MID_THRESHOLD}"
    --oob-margin "${EVAL_OOB_MARGIN}"
    --order-top-k "${EVAL_ORDER_TOP_K}"
    --pair-eps "${EVAL_PAIR_EPS}"
    --hard-gap-max "${EVAL_HARD_GAP_MAX}"
    --medium-gap-max "${EVAL_MEDIUM_GAP_MAX}"
    --output-dir "${EVAL_OUTPUT_DIR}"
    --save-prefix "${EVAL_SAVE_PREFIX}"
  )
  if bool_true "${EVAL_ONLY_SELECTED}"; then
    EVAL_CMD+=(--only-selected)
  else
    EVAL_CMD+=(--no-only-selected)
  fi
  if bool_true "${EVAL_SAVE}"; then
    EVAL_CMD+=(--save)
  else
    EVAL_CMD+=(--no-save)
  fi
  if bool_true "${EVAL_PRINT}"; then
    EVAL_CMD+=(--print)
  else
    EVAL_CMD+=(--no-print)
  fi

  log "eval_train_run_dir=${TRAIN_RUN_DIR}"
  log "eval_finetuned_model=${EVAL_FINETUNED_MODEL}"
  log "Running: ${EVAL_CMD[*]}"
  "${EVAL_CMD[@]}"
fi

log "Done: stage2-only train2 completed."
