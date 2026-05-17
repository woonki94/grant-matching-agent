from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()

DOMAIN_INPUT_DEFAULT = "ce/dataset/distill/llm_distill_domain_listwise.jsonl"
METHOD_INPUT_DEFAULT = "ce/dataset/distill/llm_distill_method_listwise.jsonl"
OUTPUT_DIR_DEFAULT = "ce/dataset/splits"

DOMAIN_TRAIN_BASENAME = "llm_distill_domain_listwise_train.jsonl"
DOMAIN_VAL_BASENAME = "llm_distill_domain_listwise_val.jsonl"
DOMAIN_TEST_BASENAME = "llm_distill_domain_listwise_test.jsonl"
METHOD_TRAIN_BASENAME = "llm_distill_method_listwise_train.jsonl"
METHOD_VAL_BASENAME = "llm_distill_method_listwise_val.jsonl"
METHOD_TEST_BASENAME = "llm_distill_method_listwise_test.jsonl"
MANIFEST_BASENAME = "llm_distill_domain_method_listwise_split_manifest.json"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_path(value: str) -> Path:
    p = Path(_clean_text(value)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _split_sizes(n: int, *, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    vr = max(0.0, min(0.49, float(val_ratio)))
    tr = max(0.0, min(0.49, float(test_ratio)))
    if vr + tr >= 0.99:
        tr = max(0.0, 0.99 - vr)

    n_test = int(round(float(n) * tr))
    n_val = int(round(float(n) * vr))
    n_train = int(n - n_test - n_val)
    if n_train <= 0:
        if n_val > 0:
            n_val -= 1
            n_train += 1
        elif n_test > 0:
            n_test -= 1
            n_train += 1
    return n_train, n_val, n_test


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = _clean_text(raw)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no} ({type(exc).__name__}: {exc})") from exc
            if isinstance(obj, dict):
                yield obj


def _query_key(obj: Dict[str, Any]) -> str:
    grant_id = _clean_text(obj.get("grant_id"))
    spec_idx = _clean_text(obj.get("spec_idx"))
    if not grant_id:
        return ""
    return f"{grant_id}::{spec_idx}"


def _collect_query_keys(path: Path) -> List[str]:
    keys = set()
    for row in _iter_jsonl(path):
        key = _query_key(row)
        if key:
            keys.add(key)
    return sorted(keys)


def _build_split_map(
    *,
    keys: Sequence[str],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    keys_list = sorted(set(str(k) for k in keys if _clean_text(k)))
    rng = random.Random(int(seed))
    rng.shuffle(keys_list)
    n_train, n_val, n_test = _split_sizes(len(keys_list), val_ratio=val_ratio, test_ratio=test_ratio)

    split_map: Dict[str, str] = {}
    idx = 0
    for k in keys_list[idx : idx + n_test]:
        split_map[k] = "test"
    idx += n_test
    for k in keys_list[idx : idx + n_val]:
        split_map[k] = "val"
    idx += n_val
    for k in keys_list[idx : idx + n_train]:
        split_map[k] = "train"

    counts = {"train": n_train, "val": n_val, "test": n_test, "total": len(keys_list)}
    return split_map, counts


def _write_split_files(
    *,
    input_path: Path,
    split_map: Dict[str, str],
    output_train: Path,
    output_val: Path,
    output_test: Path,
) -> Dict[str, int]:
    output_train.parent.mkdir(parents=True, exist_ok=True)
    output_val.parent.mkdir(parents=True, exist_ok=True)
    output_test.parent.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0, "skipped": 0, "total_rows": 0}
    with (
        output_train.open("w", encoding="utf-8") as f_train,
        output_val.open("w", encoding="utf-8") as f_val,
        output_test.open("w", encoding="utf-8") as f_test,
    ):
        for row in _iter_jsonl(input_path):
            counts["total_rows"] += 1
            key = _query_key(row)
            split = split_map.get(key, "")
            line = json.dumps(row, ensure_ascii=False)
            if split == "train":
                f_train.write(line + "\n")
                counts["train"] += 1
            elif split == "val":
                f_val.write(line + "\n")
                counts["val"] += 1
            elif split == "test":
                f_test.write(line + "\n")
                counts["test"] += 1
            else:
                counts["skipped"] += 1
    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create shared query-level train/val/test split for domain+method listwise files.")
    p.add_argument("--domain-input", type=str, default=DOMAIN_INPUT_DEFAULT)
    p.add_argument("--method-input", type=str, default=METHOD_INPUT_DEFAULT)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--test-ratio", type=float, default=0.05)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    domain_input = _resolve_path(args.domain_input)
    method_input = _resolve_path(args.method_input)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not domain_input.exists():
        raise RuntimeError(f"Domain input not found: {domain_input}")
    if not method_input.exists():
        raise RuntimeError(f"Method input not found: {method_input}")

    domain_train = output_dir / DOMAIN_TRAIN_BASENAME
    domain_val = output_dir / DOMAIN_VAL_BASENAME
    domain_test = output_dir / DOMAIN_TEST_BASENAME
    method_train = output_dir / METHOD_TRAIN_BASENAME
    method_val = output_dir / METHOD_VAL_BASENAME
    method_test = output_dir / METHOD_TEST_BASENAME
    manifest_path = output_dir / MANIFEST_BASENAME

    outputs = [domain_train, domain_val, domain_test, method_train, method_val, method_test, manifest_path]
    if (not bool(args.overwrite)) and any(p.exists() for p in outputs):
        raise RuntimeError(
            "Output files already exist. Use --overwrite to replace them."
        )

    domain_keys = set(_collect_query_keys(domain_input))
    method_keys = set(_collect_query_keys(method_input))
    all_keys = sorted(domain_keys | method_keys)
    split_map, split_query_counts = _build_split_map(
        keys=all_keys,
        seed=int(args.seed),
        val_ratio=_safe_float(args.val_ratio, default=0.05),
        test_ratio=_safe_float(args.test_ratio, default=0.05),
    )

    domain_row_counts = _write_split_files(
        input_path=domain_input,
        split_map=split_map,
        output_train=domain_train,
        output_val=domain_val,
        output_test=domain_test,
    )
    method_row_counts = _write_split_files(
        input_path=method_input,
        split_map=split_map,
        output_train=method_train,
        output_val=method_val,
        output_test=method_test,
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "domain_input": str(domain_input),
        "method_input": str(method_input),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "val_ratio": float(_safe_float(args.val_ratio, default=0.05)),
        "test_ratio": float(_safe_float(args.test_ratio, default=0.05)),
        "query_key_counts": {
            "domain": int(len(domain_keys)),
            "method": int(len(method_keys)),
            "union": int(len(all_keys)),
            "intersection": int(len(domain_keys & method_keys)),
            "domain_only": int(len(domain_keys - method_keys)),
            "method_only": int(len(method_keys - domain_keys)),
        },
        "split_query_counts": split_query_counts,
        "domain_row_counts": domain_row_counts,
        "method_row_counts": method_row_counts,
        "outputs": {
            "domain_train": str(domain_train),
            "domain_val": str(domain_val),
            "domain_test": str(domain_test),
            "method_train": str(method_train),
            "method_val": str(method_val),
            "method_test": str(method_test),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"query_union={len(all_keys)}")
    print(f"query_intersection={len(domain_keys & method_keys)}")
    print(
        "split_queries="
        f"train:{split_query_counts['train']},"
        f"val:{split_query_counts['val']},"
        f"test:{split_query_counts['test']}"
    )
    print(
        "domain_rows="
        f"train:{domain_row_counts['train']},"
        f"val:{domain_row_counts['val']},"
        f"test:{domain_row_counts['test']},"
        f"skipped:{domain_row_counts['skipped']}"
    )
    print(
        "method_rows="
        f"train:{method_row_counts['train']},"
        f"val:{method_row_counts['val']},"
        f"test:{method_row_counts['test']},"
        f"skipped:{method_row_counts['skipped']}"
    )
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
