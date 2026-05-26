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

from ce2.data_preparation.utils import (  # noqa: E402
    DECOMPOSE_BATCH_SIZE_DEFAULT,
    DECOMPOSE_MAX_NEW_TOKENS_DEFAULT,
    DECOMPOSITION_DIR_DEFAULT,
    DECOMPOSITION_OUTPUT_DEFAULT,
    FAC_DB_DEFAULT,
    GRANT_DB_DEFAULT,
    MAX_ATTEMPTS_DEFAULT,
    MAX_FAC_SPECS_DEFAULT,
    MAX_GRANT_SPECS_DEFAULT,
    MAX_MODEL_LEN_DEFAULT,
    MODEL_ID_DEFAULT,
    SEED_DEFAULT,
    TEMPERATURE_DEFAULT,
    TOP_P_DEFAULT,
    decompose_specs,
    load_fac_specs,
    load_grant_specs,
    load_jsonl_by_key,
    refresh_decomposition_cache,
    resolve_path,
)
from ce2.data_preparation.llm_runtime import load_llm, unload_llm  # noqa: E402


def _unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CE2 decomposition-only runner (domain/method/target).")
    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    p.add_argument("--grant-db", type=str, default=GRANT_DB_DEFAULT)
    p.add_argument("--fac-db", type=str, default=FAC_DB_DEFAULT)
    p.add_argument("--output-dir", type=str, default=DECOMPOSITION_DIR_DEFAULT)
    p.add_argument("--decomposition-output", type=str, default=DECOMPOSITION_OUTPUT_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--max-grant-specs", type=int, default=MAX_GRANT_SPECS_DEFAULT)
    p.add_argument("--max-fac-specs", type=int, default=MAX_FAC_SPECS_DEFAULT)
    p.add_argument("--decompose-batch-size", type=int, default=DECOMPOSE_BATCH_SIZE_DEFAULT)
    p.add_argument("--decompose-max-new-tokens", type=int, default=DECOMPOSE_MAX_NEW_TOKENS_DEFAULT)
    p.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    p.add_argument("--top-p", type=float, default=TOP_P_DEFAULT)
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN_DEFAULT)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--overwrite", action="store_true", help="Delete existing decomposition output before running.")
    p.add_argument(
        "--refresh-failed-decompositions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-run decomposition for cached rows that are parse-failed or all-empty.",
    )
    p.add_argument(
        "--refresh-all-decompositions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore all cached decomposition rows and regenerate every item.",
    )
    return p.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    output_dir = resolve_path(PROJECT_ROOT, args.output_dir)
    decomposition_path = resolve_path(PROJECT_ROOT, args.decomposition_output)
    if args.overwrite:
        _unlink_if_exists(decomposition_path)

    grant_specs = load_grant_specs(resolve_path(PROJECT_ROOT, args.grant_db), max_items=args.max_grant_specs, seed=args.seed)
    fac_specs = load_fac_specs(resolve_path(PROJECT_ROOT, args.fac_db), max_items=args.max_fac_specs, seed=args.seed)
    items = list(grant_specs) + list(fac_specs)
    output_dir.mkdir(parents=True, exist_ok=True)

    config: Dict[str, Any] = {
        "stage": "decompose_setup",
        "model_id": args.model_id,
        "grant_db": str(resolve_path(PROJECT_ROOT, args.grant_db)),
        "fac_db": str(resolve_path(PROJECT_ROOT, args.fac_db)),
        "decomposition_output": str(decomposition_path),
        "seed": int(args.seed),
        "grant_specs_loaded": len(grant_specs),
        "fac_specs_loaded": len(fac_specs),
        "items_total": len(items),
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
        decompositions = load_jsonl_by_key(decomposition_path, "item_id")
        before = len(decompositions)
        decompositions = refresh_decomposition_cache(
            decompositions,
            refresh_failed_only=bool(args.refresh_failed_decompositions),
            refresh_all=bool(args.refresh_all_decompositions),
        )
        refreshed = before - len(decompositions)
        if refreshed > 0:
            mode = "all" if args.refresh_all_decompositions else "failed_or_empty"
            print(
                f"decompose_refresh_{mode}={refreshed} "
                f"decompose_cached_kept={len(decompositions)}"
            )

        decompose_specs(
            llm_bundle=bundle,
            model_id=args.model_id,
            items=items,
            existing=decompositions,
            output_path=decomposition_path,
            batch_size=args.decompose_batch_size,
            max_new_tokens=args.decompose_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_attempts=args.max_attempts,
        )
    finally:
        unload_llm(bundle)

    elapsed = time.time() - started
    print(f"elapsed_sec={elapsed:.2f}")
    print(f"decomposition_jsonl={decomposition_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
