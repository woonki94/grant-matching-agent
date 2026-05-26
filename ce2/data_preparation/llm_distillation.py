from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce2").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce2.data_preparation.utils import (  # noqa: E402
    DECOMPOSITION_OUTPUT_DEFAULT,
    DISTILL_TARGET_HIGH_PER_ASPECT_DEFAULT,
    DISTILL_TARGET_LOW_PER_ASPECT_DEFAULT,
    DISTILL_TARGET_MID_PER_ASPECT_DEFAULT,
    DISTILL_DIR_DEFAULT,
    FAC_DB_DEFAULT,
    GRANT_DB_DEFAULT,
    MAX_ATTEMPTS_DEFAULT,
    DISTILL_BATCH_SIZE_DEFAULT,
    DISTILL_MAX_NEW_TOKENS_DEFAULT,
    MAX_FAC_SPECS_DEFAULT,
    MAX_GRANT_SPECS_DEFAULT,
    MAX_MODEL_LEN_DEFAULT,
    MODEL_ID_DEFAULT,
    PREFILTER_HIGH_THRESHOLD_DEFAULT,
    PREFILTER_LOW_THRESHOLD_DEFAULT,
    PREFILTER_MULTIPLIER_HIGH_DEFAULT,
    PREFILTER_MULTIPLIER_LOW_DEFAULT,
    PREFILTER_MULTIPLIER_MID_DEFAULT,
    PREFILTER_CACHE_OUTPUT_DEFAULT,
    DISTILLATION_OUTPUT_DEFAULT,
    SEED_DEFAULT,
    STS_PREFILTER_MODEL_ID_DEFAULT,
    SUMMARY_OUTPUT_DEFAULT,
    TEMPERATURE_DEFAULT,
    TOP_P_DEFAULT,
    build_summary,
    load_fac_specs,
    load_grant_specs,
    load_jsonl_by_key,
    load_distilled_pair_keys,
    resolve_path,
    distill_pairs,
    select_pairs_from_prefilter_cache,
    select_pairs_with_sts_prefilter,
    write_json,
)
from ce2.data_preparation.llm_runtime import load_llm, unload_llm  # noqa: E402


def _unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CE2 LLM distillation runner over precomputed decompositions.")
    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    p.add_argument("--grant-db", type=str, default=GRANT_DB_DEFAULT)
    p.add_argument("--fac-db", type=str, default=FAC_DB_DEFAULT)
    p.add_argument("--output-dir", type=str, default=DISTILL_DIR_DEFAULT)
    p.add_argument("--decomposition-output", type=str, default=DECOMPOSITION_OUTPUT_DEFAULT)
    p.add_argument("--prefilter-source", type=str, choices=["ce-cache", "sts"], default="ce-cache")
    p.add_argument("--prefilter-cache", type=str, default=PREFILTER_CACHE_OUTPUT_DEFAULT)
    p.add_argument("--distillation-output", type=str, default=DISTILLATION_OUTPUT_DEFAULT)
    p.add_argument("--summary-output", type=str, default=SUMMARY_OUTPUT_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--max-grant-specs", type=int, default=MAX_GRANT_SPECS_DEFAULT)
    p.add_argument("--max-fac-specs", type=int, default=MAX_FAC_SPECS_DEFAULT)
    p.add_argument("--target-high-per-aspect", type=int, default=DISTILL_TARGET_HIGH_PER_ASPECT_DEFAULT)
    p.add_argument("--target-mid-per-aspect", type=int, default=DISTILL_TARGET_MID_PER_ASPECT_DEFAULT)
    p.add_argument("--target-low-per-aspect", type=int, default=DISTILL_TARGET_LOW_PER_ASPECT_DEFAULT)
    p.add_argument("--prefilter-multiplier-high", type=float, default=PREFILTER_MULTIPLIER_HIGH_DEFAULT)
    p.add_argument("--prefilter-multiplier-mid", type=float, default=PREFILTER_MULTIPLIER_MID_DEFAULT)
    p.add_argument("--prefilter-multiplier-low", type=float, default=PREFILTER_MULTIPLIER_LOW_DEFAULT)
    p.add_argument("--prefilter-high-threshold", type=float, default=PREFILTER_HIGH_THRESHOLD_DEFAULT)
    p.add_argument("--prefilter-low-threshold", type=float, default=PREFILTER_LOW_THRESHOLD_DEFAULT)
    p.add_argument("--distill-batch-size", type=int, default=DISTILL_BATCH_SIZE_DEFAULT)
    p.add_argument("--distill-max-new-tokens", type=int, default=DISTILL_MAX_NEW_TOKENS_DEFAULT)
    p.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    p.add_argument("--top-p", type=float, default=TOP_P_DEFAULT)
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN_DEFAULT)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--overwrite", action="store_true", help="Delete existing distillation/summary outputs before running.")
    return p.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    output_dir = resolve_path(PROJECT_ROOT, args.output_dir)
    decomposition_path = resolve_path(PROJECT_ROOT, args.decomposition_output)
    prefilter_cache_path = resolve_path(PROJECT_ROOT, args.prefilter_cache)
    distillation_path = resolve_path(PROJECT_ROOT, args.distillation_output)
    summary_path = resolve_path(PROJECT_ROOT, args.summary_output)
    if not decomposition_path.exists():
        raise FileNotFoundError(
            f"Missing decomposition file: {decomposition_path}\n"
            "Run ce2/data_preparation/decompose_aspect_specs.py first."
        )
    if args.overwrite:
        _unlink_if_exists(distillation_path)
        _unlink_if_exists(summary_path)

    grant_specs = load_grant_specs(resolve_path(PROJECT_ROOT, args.grant_db), max_items=args.max_grant_specs, seed=args.seed)
    fac_specs = load_fac_specs(resolve_path(PROJECT_ROOT, args.fac_db), max_items=args.max_fac_specs, seed=args.seed)
    decompositions = load_jsonl_by_key(decomposition_path, "item_id")
    target_high = max(0, int(args.target_high_per_aspect))
    target_mid = max(0, int(args.target_mid_per_aspect))
    target_low = max(0, int(args.target_low_per_aspect))
    prefilter_high = int(math.ceil(target_high * max(0.0, float(args.prefilter_multiplier_high))))
    prefilter_mid = int(math.ceil(target_mid * max(0.0, float(args.prefilter_multiplier_mid))))
    prefilter_low = int(math.ceil(target_low * max(0.0, float(args.prefilter_multiplier_low))))
    if args.prefilter_source == "ce-cache":
        pairs = select_pairs_from_prefilter_cache(
            grant_specs,
            fac_specs,
            cache_base_path=prefilter_cache_path,
            seed=args.seed,
            high_per_aspect=prefilter_high,
            mid_per_aspect=prefilter_mid,
            low_per_aspect=prefilter_low,
            high_threshold=float(args.prefilter_high_threshold),
            low_threshold=float(args.prefilter_low_threshold),
        )
        pair_selection = "prefilter_cache_threshold_windows"
    else:
        pairs = select_pairs_with_sts_prefilter(
            grant_specs,
            fac_specs,
            decompositions=decompositions,
            seed=args.seed,
            high_per_aspect=prefilter_high,
            mid_per_aspect=prefilter_mid,
            low_per_aspect=prefilter_low,
        )
        pair_selection = "sts_prefilter_balanced"
    output_dir.mkdir(parents=True, exist_ok=True)
    config: Dict[str, Any] = {
        "stage": "distill_setup",
        "model_id": args.model_id,
        "grant_db": str(resolve_path(PROJECT_ROOT, args.grant_db)),
        "fac_db": str(resolve_path(PROJECT_ROOT, args.fac_db)),
        "decomposition_output": str(decomposition_path),
        "prefilter_source": args.prefilter_source,
        "prefilter_cache": str(prefilter_cache_path),
        "distillation_output": str(distillation_path),
        "summary_output": str(summary_path),
        "seed": int(args.seed),
        "max_grant_specs": int(args.max_grant_specs),
        "max_fac_specs": int(args.max_fac_specs),
        "grant_specs_loaded": len(grant_specs),
        "fac_specs_loaded": len(fac_specs),
        "decompositions_loaded": len(decompositions),
        "candidate_pairs": len(pairs),
        "pair_selection": pair_selection,
        "target_high_per_aspect": int(target_high),
        "target_mid_per_aspect": int(target_mid),
        "target_low_per_aspect": int(target_low),
        "prefilter_high_per_aspect": int(prefilter_high),
        "prefilter_mid_per_aspect": int(prefilter_mid),
        "prefilter_low_per_aspect": int(prefilter_low),
        "prefilter_multiplier_high": float(args.prefilter_multiplier_high),
        "prefilter_multiplier_mid": float(args.prefilter_multiplier_mid),
        "prefilter_multiplier_low": float(args.prefilter_multiplier_low),
        "prefilter_high_threshold": float(args.prefilter_high_threshold),
        "prefilter_low_threshold": float(args.prefilter_low_threshold),
        "prefilter_sts_model_id": STS_PREFILTER_MODEL_ID_DEFAULT,
        "use_tqdm": True,
    }
    print(json.dumps(config, ensure_ascii=False))

    bundle: Optional[Dict[str, Any]] = None
    try:
        bundle = load_llm(
            args.model_id,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        existing_pair_ids = load_distilled_pair_keys(distillation_path)
        distill_pairs(
            llm_bundle=bundle,
            model_id=args.model_id,
            pairs=pairs,
            decompositions=decompositions,
            existing_pair_ids=existing_pair_ids,
            output_path=distillation_path,
            batch_size=args.distill_batch_size,
            max_new_tokens=args.distill_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_attempts=args.max_attempts,
            target_high_per_aspect=target_high,
            target_mid_per_aspect=target_mid,
            target_low_per_aspect=target_low,
        )
    finally:
        unload_llm(bundle)

    summary = build_summary(
        distillation_path=distillation_path,
        decomposition_path=decomposition_path,
        started_at=started,
        config=config,
    )
    write_json(summary_path, summary)
    print(f"elapsed_sec={summary.get('elapsed_sec', 0.0):.2f}")
    print(f"distillation_jsonl={distillation_path}")
    print(f"summary_json={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
