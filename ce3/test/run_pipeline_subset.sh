#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

run_stage() {
  local name="$1"
  shift
  echo
  echo "========== CE3 subset stage: ${name} =========="
  "$@"
}

run_stage "decompose originals" bash ce3/test/component/run_decomposition_subset.sh
run_stage "augment high-intent candidates" bash ce3/test/component/run_augmentation_subset.sh
run_stage "decompose augmented candidates" bash ce3/test/component/run_decompose_augmented_subset.sh
run_stage "combine decompositions" bash ce3/test/component/run_combine_decompositions_subset.sh
run_stage "build prefilter cache" bash ce3/test/component/run_prefilter_cache_subset.sh
run_stage "distill selected pairs" bash ce3/test/component/run_distillation_subset.sh

echo
echo "CE3 subset pipeline complete."
echo "Output directory: ce3/test/output"
