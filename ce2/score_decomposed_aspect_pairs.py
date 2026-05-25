from __future__ import annotations

import argparse
import json
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

from ce2.aspect_common import (
    FAC_DB_DEFAULT,
    GRANT_DB_DEFAULT,
    STS_CACHE_PATH_DEFAULT,
    ASPECT_PREFILTER_HIGH_PER_ASPECT_DEFAULT,
    ASPECT_PREFILTER_MID_PER_ASPECT_DEFAULT,
    ASPECT_PREFILTER_LOW_PER_ASPECT_DEFAULT,
    ASPECT_PREFILTER_HIGH_POOL_SIZE_DEFAULT,
    ASPECT_PREFILTER_MID_RANK_START_DEFAULT,
    ASPECT_PREFILTER_MID_RANK_END_DEFAULT,
    load_sts_cache,
    MAX_ATTEMPTS_DEFAULT,
    MAX_FAC_SPECS_DEFAULT,
    MAX_GRANT_SPECS_DEFAULT,
    MAX_MODEL_LEN_DEFAULT,
    MODEL_ID_DEFAULT,
    OUTPUT_DIR_DEFAULT,
    SCORES_OUTPUT_DEFAULT,
    SCORE_BATCH_SIZE_DEFAULT,
    SCORE_MAX_NEW_TOKENS_DEFAULT,
    SEED_DEFAULT,
    SUMMARY_OUTPUT_DEFAULT,
    TEMPERATURE_DEFAULT,
    TOP_P_DEFAULT,
    DECOMPOSITION_OUTPUT_DEFAULT,
    build_summary,
    load_fac_specs,
    load_grant_specs,
    load_jsonl_by_key,
    load_scored_pair_keys,
    resolve_path,
    select_pairs_aspect_prefilter,
    write_json,
)
from ce2.aspect_scoring_utils import score_pairs
from ce2.llm_runtime import load_llm, unload_llm


def _unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CE2 scoring-only runner over precomputed decompositions.")
    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    p.add_argument("--grant-db", type=str, default=GRANT_DB_DEFAULT)
    p.add_argument("--fac-db", type=str, default=FAC_DB_DEFAULT)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--decomposition-output", type=str, default=DECOMPOSITION_OUTPUT_DEFAULT)
    p.add_argument("--sts-cache", type=str, default=STS_CACHE_PATH_DEFAULT)
    p.add_argument("--scores-output", type=str, default=SCORES_OUTPUT_DEFAULT)
    p.add_argument("--summary-output", type=str, default=SUMMARY_OUTPUT_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--max-grant-specs", type=int, default=MAX_GRANT_SPECS_DEFAULT)
    p.add_argument("--max-fac-specs", type=int, default=MAX_FAC_SPECS_DEFAULT)
    p.add_argument("--score-batch-size", type=int, default=SCORE_BATCH_SIZE_DEFAULT)
    p.add_argument("--score-max-new-tokens", type=int, default=SCORE_MAX_NEW_TOKENS_DEFAULT)
    p.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    p.add_argument("--top-p", type=float, default=TOP_P_DEFAULT)
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN_DEFAULT)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--overwrite", action="store_true", help="Delete existing scores/summary outputs before running.")
    return p.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    output_dir = resolve_path(PROJECT_ROOT, args.output_dir)
    decomposition_path = resolve_path(PROJECT_ROOT, args.decomposition_output)
    sts_cache_path = resolve_path(PROJECT_ROOT, args.sts_cache)
    scores_path = resolve_path(PROJECT_ROOT, args.scores_output)
    summary_path = resolve_path(PROJECT_ROOT, args.summary_output)
    if not decomposition_path.exists():
        raise FileNotFoundError(
            f"Missing decomposition file: {decomposition_path}\n"
            "Run ce2/decompose_aspect_specs.py first."
        )
    if args.overwrite:
        _unlink_if_exists(scores_path)
        _unlink_if_exists(summary_path)

    grant_specs = load_grant_specs(resolve_path(PROJECT_ROOT, args.grant_db), max_items=args.max_grant_specs, seed=args.seed)
    fac_specs = load_fac_specs(resolve_path(PROJECT_ROOT, args.fac_db), max_items=args.max_fac_specs, seed=args.seed)
    decompositions = load_jsonl_by_key(decomposition_path, "item_id")
    sts_cache = load_sts_cache(sts_cache_path)
    pairs = select_pairs_aspect_prefilter(
        grant_specs,
        fac_specs,
        sts_cache=sts_cache,
        seed=args.seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config: Dict[str, Any] = {
        "stage": "score_setup",
        "model_id": args.model_id,
        "grant_db": str(resolve_path(PROJECT_ROOT, args.grant_db)),
        "fac_db": str(resolve_path(PROJECT_ROOT, args.fac_db)),
        "decomposition_output": str(decomposition_path),
        "sts_cache": str(sts_cache_path),
        "scores_output": str(scores_path),
        "summary_output": str(summary_path),
        "seed": int(args.seed),
        "max_grant_specs": int(args.max_grant_specs),
        "max_fac_specs": int(args.max_fac_specs),
        "grant_specs_loaded": len(grant_specs),
        "fac_specs_loaded": len(fac_specs),
        "decompositions_loaded": len(decompositions),
        "sts_cache_grants": len(sts_cache),
        "candidate_pairs": len(pairs),
        "pair_selection": "aspect_prefilter_balanced",
        "prefilter_high_per_aspect": int(ASPECT_PREFILTER_HIGH_PER_ASPECT_DEFAULT),
        "prefilter_mid_per_aspect": int(ASPECT_PREFILTER_MID_PER_ASPECT_DEFAULT),
        "prefilter_low_per_aspect": int(ASPECT_PREFILTER_LOW_PER_ASPECT_DEFAULT),
        "prefilter_high_pool_size": int(ASPECT_PREFILTER_HIGH_POOL_SIZE_DEFAULT),
        "prefilter_mid_rank_start": int(ASPECT_PREFILTER_MID_RANK_START_DEFAULT),
        "prefilter_mid_rank_end": int(ASPECT_PREFILTER_MID_RANK_END_DEFAULT),
        "prefilter_method": "sts_cache",
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
        existing_pair_ids = load_scored_pair_keys(scores_path)
        score_pairs(
            llm_bundle=bundle,
            model_id=args.model_id,
            pairs=pairs,
            decompositions=decompositions,
            existing_pair_ids=existing_pair_ids,
            output_path=scores_path,
            batch_size=args.score_batch_size,
            max_new_tokens=args.score_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_attempts=args.max_attempts,
        )
    finally:
        unload_llm(bundle)

    summary = build_summary(
        scores_path=scores_path,
        decomposition_path=decomposition_path,
        started_at=started,
        config=config,
    )
    write_json(summary_path, summary)
    print(f"elapsed_sec={summary.get('elapsed_sec', 0.0):.2f}")
    print(f"scores_jsonl={scores_path}")
    print(f"summary_json={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
