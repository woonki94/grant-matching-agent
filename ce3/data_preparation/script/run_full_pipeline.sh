#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

LLM_MODEL_ID="${LLM_MODEL_ID:-Qwen/Qwen3-14B}"
PREFILTER_MODEL_ID="${PREFILTER_MODEL_ID:-dleemiller/ModernCE-base-sts}"

GRANT_DB="${GRANT_DB:-ce3/dataset/source/grant_keywords_spec_keywords_db.json}"
FAC_DB="${FAC_DB:-ce3/dataset/source/fac_specs_db.json}"

ORIGINAL_DECOMPOSITION_OUTPUT="${ORIGINAL_DECOMPOSITION_OUTPUT:-ce3/dataset/decomposed/spec_decompositions_topic_approach_objective.jsonl}"
AUGMENTATION_OUTPUT="${AUGMENTATION_OUTPUT:-ce3/dataset/augmented/spec_augmentations_high.jsonl}"
AUGMENTED_DECOMPOSITION_OUTPUT="${AUGMENTED_DECOMPOSITION_OUTPUT:-ce3/dataset/decomposed/spec_decompositions_augmented.jsonl}"
COMBINED_DECOMPOSITION_OUTPUT="${COMBINED_DECOMPOSITION_OUTPUT:-ce3/dataset/decomposed/spec_decompositions_combined.jsonl}"
PREFILTER_CACHE_BASE="${PREFILTER_CACHE_BASE:-ce3/dataset/source/prefilter_cache.jsonl}"
DISTILLATION_OUTPUT="${DISTILLATION_OUTPUT:-ce3/dataset/distill/llm_distillation.jsonl}"

SEED="${SEED:-42}"
MAX_GRANT_SPECS="${MAX_GRANT_SPECS:-0}"
MAX_FAC_SPECS="${MAX_FAC_SPECS:-0}"

DECOMPOSE_BATCH_SIZE="${DECOMPOSE_BATCH_SIZE:-512}"
DECOMPOSE_MAX_NEW_TOKENS="${DECOMPOSE_MAX_NEW_TOKENS:-256}"
AUGMENT_BATCH_SIZE="${AUGMENT_BATCH_SIZE:-512}"
AUGMENT_MAX_NEW_TOKENS="${AUGMENT_MAX_NEW_TOKENS:-384}"
PREFILTER_BATCH_SIZE="${PREFILTER_BATCH_SIZE:-256}"
PREFILTER_MAX_LENGTH="${PREFILTER_MAX_LENGTH:-256}"
DISTILL_BATCH_SIZE="${DISTILL_BATCH_SIZE:-512}"
DISTILL_MAX_NEW_TOKENS="${DISTILL_MAX_NEW_TOKENS:-32}"

AUGMENTATIONS_PER_ASPECT="${AUGMENTATIONS_PER_ASPECT:-3}"
TARGET_HIGH_PER_GRANT_ASPECT="${TARGET_HIGH_PER_GRANT_ASPECT:-3}"
TARGET_MID_PER_GRANT_ASPECT="${TARGET_MID_PER_GRANT_ASPECT:-6}"
TARGET_LOW_PER_GRANT_ASPECT="${TARGET_LOW_PER_GRANT_ASPECT:-3}"
PREFILTER_HIGH_MULTIPLIER="${PREFILTER_HIGH_MULTIPLIER:-3.0}"
PREFILTER_MID_MULTIPLIER="${PREFILTER_MID_MULTIPLIER:-2.0}"
PREFILTER_LOW_MULTIPLIER="${PREFILTER_LOW_MULTIPLIER:-1.0}"

DECOMPOSE_TEMPERATURE="${DECOMPOSE_TEMPERATURE:-0.0}"
AUGMENT_TEMPERATURE="${AUGMENT_TEMPERATURE:-0.7}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

OVERWRITE_ORIGINAL_DECOMPOSITION="${OVERWRITE_ORIGINAL_DECOMPOSITION:-false}"
OVERWRITE_AUGMENTATION="${OVERWRITE_AUGMENTATION:-false}"
OVERWRITE_AUGMENTED_DECOMPOSITION="${OVERWRITE_AUGMENTED_DECOMPOSITION:-false}"
OVERWRITE_DISTILLATION="${OVERWRITE_DISTILLATION:-false}"
REFRESH_FAILED_DECOMPOSITIONS="${REFRESH_FAILED_DECOMPOSITIONS:-true}"
REFRESH_ALL_DECOMPOSITIONS="${REFRESH_ALL_DECOMPOSITIONS:-false}"
INCLUDE_AUG_AUG="${INCLUDE_AUG_AUG:-false}"

bool_true() { [[ "${1:-}" == "true" ]]; }

run_stage() {
  local name="$1"
  shift
  echo
  echo "========== CE3 full stage: ${name} =========="
  echo "Running: $*"
  "$@"
}

ORIGINAL_DECOMPOSE_CMD=(
  "${PYTHON_BIN}" ce3/data_preparation/decompose_specializations.py
  --model-id "${LLM_MODEL_ID}"
  --grant-db "${GRANT_DB}"
  --fac-db "${FAC_DB}"
  --output-dir "ce3/dataset/decomposed"
  --decomposition-output "${ORIGINAL_DECOMPOSITION_OUTPUT}"
  --seed "${SEED}"
  --max-grant-specs "${MAX_GRANT_SPECS}"
  --max-fac-specs "${MAX_FAC_SPECS}"
  --decompose-batch-size "${DECOMPOSE_BATCH_SIZE}"
  --decompose-max-new-tokens "${DECOMPOSE_MAX_NEW_TOKENS}"
  --temperature "${DECOMPOSE_TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-attempts "${MAX_ATTEMPTS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
)
if bool_true "${OVERWRITE_ORIGINAL_DECOMPOSITION}"; then
  ORIGINAL_DECOMPOSE_CMD+=(--overwrite)
fi
if bool_true "${REFRESH_FAILED_DECOMPOSITIONS}"; then
  ORIGINAL_DECOMPOSE_CMD+=(--refresh-failed-decompositions)
else
  ORIGINAL_DECOMPOSE_CMD+=(--no-refresh-failed-decompositions)
fi
if bool_true "${REFRESH_ALL_DECOMPOSITIONS}"; then
  ORIGINAL_DECOMPOSE_CMD+=(--refresh-all-decompositions)
else
  ORIGINAL_DECOMPOSE_CMD+=(--no-refresh-all-decompositions)
fi

AUGMENT_CMD=(
  "${PYTHON_BIN}" ce3/data_preparation/augment_specializations.py
  --model-id "${LLM_MODEL_ID}"
  --decomposition-output "${ORIGINAL_DECOMPOSITION_OUTPUT}"
  --output-dir "ce3/dataset/augmented"
  --augmentation-output "${AUGMENTATION_OUTPUT}"
  --augmentations-per-aspect "${AUGMENTATIONS_PER_ASPECT}"
  --augment-batch-size "${AUGMENT_BATCH_SIZE}"
  --augment-max-new-tokens "${AUGMENT_MAX_NEW_TOKENS}"
  --temperature "${AUGMENT_TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-attempts "${MAX_ATTEMPTS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
)
if bool_true "${OVERWRITE_AUGMENTATION}"; then
  AUGMENT_CMD+=(--overwrite)
fi

AUGMENTED_DECOMPOSE_CMD=(
  "${PYTHON_BIN}" ce3/data_preparation/decompose_augmented_specializations.py
  --model-id "${LLM_MODEL_ID}"
  --augmentation-input "${AUGMENTATION_OUTPUT}"
  --output-dir "ce3/dataset/decomposed"
  --augmented-decomposition-output "${AUGMENTED_DECOMPOSITION_OUTPUT}"
  --decompose-batch-size "${DECOMPOSE_BATCH_SIZE}"
  --decompose-max-new-tokens "${DECOMPOSE_MAX_NEW_TOKENS}"
  --temperature "${DECOMPOSE_TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-attempts "${MAX_ATTEMPTS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
)
if bool_true "${OVERWRITE_AUGMENTED_DECOMPOSITION}"; then
  AUGMENTED_DECOMPOSE_CMD+=(--overwrite)
fi
if bool_true "${REFRESH_FAILED_DECOMPOSITIONS}"; then
  AUGMENTED_DECOMPOSE_CMD+=(--refresh-failed-decompositions)
else
  AUGMENTED_DECOMPOSE_CMD+=(--no-refresh-failed-decompositions)
fi
if bool_true "${REFRESH_ALL_DECOMPOSITIONS}"; then
  AUGMENTED_DECOMPOSE_CMD+=(--refresh-all-decompositions)
else
  AUGMENTED_DECOMPOSE_CMD+=(--no-refresh-all-decompositions)
fi

COMBINE_CMD=(
  "${PYTHON_BIN}" ce3/data_preparation/combine_decomposition_files.py
  --original-decomposition "${ORIGINAL_DECOMPOSITION_OUTPUT}"
  --augmented-decomposition "${AUGMENTED_DECOMPOSITION_OUTPUT}"
  --combined-output "${COMBINED_DECOMPOSITION_OUTPUT}"
)

PREFILTER_CMD=(
  "${PYTHON_BIN}" ce3/data_preparation/build_prefilter_cache.py
  --model-id "${PREFILTER_MODEL_ID}"
  --decomposition-output "${COMBINED_DECOMPOSITION_OUTPUT}"
  --output-base "${PREFILTER_CACHE_BASE}"
  --batch-size "${PREFILTER_BATCH_SIZE}"
  --max-length "${PREFILTER_MAX_LENGTH}"
)
if bool_true "${INCLUDE_AUG_AUG}"; then
  PREFILTER_CMD+=(--include-aug-aug)
fi

DISTILL_CMD=(
  "${PYTHON_BIN}" ce3/data_preparation/llm_distillation.py
  --model-id "${LLM_MODEL_ID}"
  --decomposition-output "${COMBINED_DECOMPOSITION_OUTPUT}"
  --prefilter-cache-base "${PREFILTER_CACHE_BASE}"
  --output-dir "ce3/dataset/distill"
  --distillation-output "${DISTILLATION_OUTPUT}"
  --target-high-per-grant-aspect "${TARGET_HIGH_PER_GRANT_ASPECT}"
  --target-mid-per-grant-aspect "${TARGET_MID_PER_GRANT_ASPECT}"
  --target-low-per-grant-aspect "${TARGET_LOW_PER_GRANT_ASPECT}"
  --prefilter-high-multiplier "${PREFILTER_HIGH_MULTIPLIER}"
  --prefilter-mid-multiplier "${PREFILTER_MID_MULTIPLIER}"
  --prefilter-low-multiplier "${PREFILTER_LOW_MULTIPLIER}"
  --distill-batch-size "${DISTILL_BATCH_SIZE}"
  --distill-max-new-tokens "${DISTILL_MAX_NEW_TOKENS}"
  --temperature "${DISTILL_TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-attempts "${MAX_ATTEMPTS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
)
if bool_true "${OVERWRITE_DISTILLATION}"; then
  DISTILL_CMD+=(--overwrite)
fi

run_stage "decompose originals" "${ORIGINAL_DECOMPOSE_CMD[@]}"
run_stage "augment high-intent candidates" "${AUGMENT_CMD[@]}"
run_stage "decompose augmented candidates" "${AUGMENTED_DECOMPOSE_CMD[@]}"
run_stage "combine decompositions" "${COMBINE_CMD[@]}"
run_stage "build prefilter cache" "${PREFILTER_CMD[@]}"
run_stage "distill selected pairs" "${DISTILL_CMD[@]}"

echo
echo "CE3 full pipeline complete."
echo "Combined decomposition: ${COMBINED_DECOMPOSITION_OUTPUT}"
echo "Prefilter cache base: ${PREFILTER_CACHE_BASE}"
echo "Distillation output: ${DISTILLATION_OUTPUT}"
