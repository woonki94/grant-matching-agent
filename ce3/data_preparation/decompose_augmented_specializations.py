from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce3").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce3.data_preparation.decompose_specializations import (  # noqa: E402
    ASPECTS,
    decompose_specializations,
)
from ce3.data_preparation.llm_runtime import clean_text, load_llm, normalize_ws, unload_llm  # noqa: E402
from ce3.data_preparation.utils import (  # noqa: E402
    SpecItem,
    load_jsonl_by_key,
    refresh_decomposition_cache,
    resolve_path,
)


MODEL_ID_DEFAULT = "Qwen/Qwen3-14B"
AUGMENTATION_INPUT_DEFAULT = "ce3/dataset/augmented/spec_augmentations_high.jsonl"
OUTPUT_DIR_DEFAULT = "ce3/dataset/decomposed"
AUGMENTED_DECOMPOSITION_OUTPUT_DEFAULT = "ce3/dataset/decomposed/spec_decompositions_augmented.jsonl"
DECOMPOSE_BATCH_SIZE_DEFAULT = 16
DECOMPOSE_MAX_NEW_TOKENS_DEFAULT = 256
MAX_ATTEMPTS_DEFAULT = 2
MAX_MODEL_LEN_DEFAULT = 4096
TEMPERATURE_DEFAULT = 0.0
TOP_P_DEFAULT = 0.9


def _unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def load_augmented_items(path: Path) -> List[SpecItem]:
    items: List[SpecItem] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            try:
                row: Dict[str, Any] = json.loads(raw)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            item_id = clean_text(row.get("item_id"))
            kind = clean_text(row.get("kind"))
            text = normalize_ws(row.get("text"))
            if not item_id or not kind or not text or item_id in seen:
                continue
            if kind not in {"grant_aug", "fac_aug"}:
                continue
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            items.append(SpecItem(item_id=item_id, kind=kind, text=text, meta=dict(meta)))
            seen.add(item_id)
    return items


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CE3 decomposition runner for augmented specialization rows.")
    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    p.add_argument("--augmentation-input", type=str, default=AUGMENTATION_INPUT_DEFAULT)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--augmented-decomposition-output", type=str, default=AUGMENTED_DECOMPOSITION_OUTPUT_DEFAULT)
    p.add_argument("--decompose-batch-size", type=int, default=DECOMPOSE_BATCH_SIZE_DEFAULT)
    p.add_argument("--decompose-max-new-tokens", type=int, default=DECOMPOSE_MAX_NEW_TOKENS_DEFAULT)
    p.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    p.add_argument("--top-p", type=float, default=TOP_P_DEFAULT)
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN_DEFAULT)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--overwrite", action="store_true", help="Delete existing augmented decomposition output before running.")
    p.add_argument(
        "--refresh-failed-decompositions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-run parse-failed or all-empty cached rows.",
    )
    p.add_argument(
        "--refresh-all-decompositions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore all cached augmented decomposition rows and regenerate every item.",
    )
    return p.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    augmentation_input = resolve_path(args.augmentation_input)
    output_dir = resolve_path(args.output_dir)
    output_path = resolve_path(args.augmented_decomposition_output)
    if not augmentation_input.exists():
        raise FileNotFoundError(f"Missing augmentation input: {augmentation_input}")
    if args.overwrite:
        _unlink_if_exists(output_path)

    items = load_augmented_items(augmentation_input)
    if not items:
        raise RuntimeError(f"No augmented specialization rows found in {augmentation_input}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "stage": "ce3_decompose_augmented_setup",
                "model_id": args.model_id,
                "augmentation_input": str(augmentation_input),
                "augmented_decomposition_output": str(output_path),
                "aspects": list(ASPECTS),
                "items_total": int(len(items)),
            },
            ensure_ascii=False,
        )
    )

    bundle = None
    try:
        bundle = load_llm(
            args.model_id,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        existing = load_jsonl_by_key(output_path, "item_id")
        existing = refresh_decomposition_cache(
            existing,
            refresh_failed_only=bool(args.refresh_failed_decompositions),
            refresh_all=bool(args.refresh_all_decompositions),
            aspects=ASPECTS,
        )
        decompose_specializations(
            llm_bundle=bundle,
            model_id=args.model_id,
            items=items,
            existing=existing,
            output_path=output_path,
            batch_size=args.decompose_batch_size,
            max_new_tokens=args.decompose_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_attempts=args.max_attempts,
        )
    finally:
        unload_llm(bundle)

    print(f"elapsed_sec={time.time() - started:.2f}")
    print(f"augmented_decomposition_jsonl={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
