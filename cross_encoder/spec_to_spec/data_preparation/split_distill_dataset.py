from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "cross_encoder").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()

RAW_INPUT_DEFAULT = "cross_encoder/spec_to_spec/dataset/llm_distill_raw_scores.jsonl"
PAIRWISE_INPUT_DEFAULT = "cross_encoder/spec_to_spec/dataset/llm_distill_pairwise.jsonl"
DEFAULT_SPLIT_DIR = "cross_encoder/spec_to_spec/dataset/splits"

RAW_TRAIN_BASENAME = "llm_distill_raw_train.jsonl"
RAW_VAL_BASENAME = "llm_distill_raw_val.jsonl"
RAW_TEST_BASENAME = "llm_distill_raw_test.jsonl"
PAIR_TRAIN_BASENAME = "llm_distill_pairwise_train.jsonl"
PAIR_VAL_BASENAME = "llm_distill_pairwise_val.jsonl"
PAIR_TEST_BASENAME = "llm_distill_pairwise_test.jsonl"
MANIFEST_BASENAME = "llm_distill_split_manifest.json"


@dataclass
class SplitPaths:
    split_dir: Path
    raw_train: Path
    raw_val: Path
    raw_test: Path
    pair_train: Path
    pair_val: Path
    pair_test: Path
    manifest: Path


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_path(value: Any) -> Path:
    p = Path(_clean_text(value)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _hash_to_unit_interval(text: str) -> float:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return int(h, 16) / float(0xFFFFFFFF)


def _split_grants(
    grant_ids: Sequence[str],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Set[str], Set[str], Set[str]]:
    uniq = sorted({_clean_text(x) for x in grant_ids if _clean_text(x)})
    if not uniq:
        return set(), set(), set()

    vr = max(0.0, min(0.49, float(val_ratio)))
    tr = max(0.0, min(0.49, float(test_ratio)))
    if vr + tr >= 0.99:
        tr = max(0.0, 0.99 - vr)

    tagged: List[Tuple[float, str]] = []
    for gid in uniq:
        roll = _hash_to_unit_interval(f"{int(seed)}::{gid}")
        tagged.append((roll, gid))
    tagged.sort(key=lambda x: (x[0], x[1]))

    n = len(tagged)
    n_test = int(round(float(n) * tr))
    n_val = int(round(float(n) * vr))

    if n >= 3:
        if tr > 0.0:
            n_test = max(1, n_test)
        if vr > 0.0:
            n_val = max(1, n_val)

    if n_test + n_val >= n:
        overflow = n_test + n_val - (n - 1)
        if overflow > 0:
            reduce_test = min(overflow, n_test)
            n_test -= reduce_test
            overflow -= reduce_test
            if overflow > 0:
                n_val = max(0, n_val - overflow)

    test_ids = [gid for _, gid in tagged[:n_test]]
    val_ids = [gid for _, gid in tagged[n_test : n_test + n_val]]
    train_ids = [gid for _, gid in tagged[n_test + n_val :]]

    if not train_ids:
        if val_ids:
            train_ids.append(val_ids.pop())
        elif test_ids:
            train_ids.append(test_ids.pop())

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    val_set.difference_update(train_set)
    test_set.difference_update(train_set)
    test_set.difference_update(val_set)
    return train_set, val_set, test_set


def _build_split_paths(split_dir: Path) -> SplitPaths:
    d = split_dir.resolve()
    return SplitPaths(
        split_dir=d,
        raw_train=d / RAW_TRAIN_BASENAME,
        raw_val=d / RAW_VAL_BASENAME,
        raw_test=d / RAW_TEST_BASENAME,
        pair_train=d / PAIR_TRAIN_BASENAME,
        pair_val=d / PAIR_VAL_BASENAME,
        pair_test=d / PAIR_TEST_BASENAME,
        manifest=d / MANIFEST_BASENAME,
    )


def _extract_grant_id(line: str) -> str:
    s = _clean_text(line)
    if not s:
        return ""
    try:
        obj = json.loads(s)
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""
    return _clean_text(obj.get("grant_id"))


def _collect_unique_grants(raw_input: Path) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    with raw_input.open("r", encoding="utf-8") as f:
        for line in f:
            gid = _extract_grant_id(line)
            if not gid or gid in seen:
                continue
            seen.add(gid)
            out.append(gid)
    return sorted(out)


def _write_split_jsonl(
    *,
    input_path: Path,
    split_lookup: Dict[str, str],
    output_map: Dict[str, Path],
) -> Dict[str, int]:
    output_map = {k: v.resolve() for k, v in output_map.items()}
    tmp_map = {k: v.with_suffix(v.suffix + ".tmp") for k, v in output_map.items()}
    for p in output_map.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    counts = {"total": 0, "written_train": 0, "written_val": 0, "written_test": 0, "missing_grant": 0, "parse_fail": 0}

    handles: Dict[str, Any] = {
        "train": tmp_map["train"].open("w", encoding="utf-8"),
        "val": tmp_map["val"].open("w", encoding="utf-8"),
        "test": tmp_map["test"].open("w", encoding="utf-8"),
    }
    try:
        with input_path.open("r", encoding="utf-8") as src:
            for raw_line in src:
                counts["total"] += 1
                gid = _extract_grant_id(raw_line)
                if not gid:
                    counts["parse_fail"] += 1
                    continue
                split = split_lookup.get(gid)
                if split not in {"train", "val", "test"}:
                    counts["missing_grant"] += 1
                    continue
                handles[split].write(raw_line.rstrip("\n") + "\n")
                counts[f"written_{split}"] += 1
    finally:
        for h in handles.values():
            h.close()

    for key in ("train", "val", "test"):
        tmp_map[key].replace(output_map[key])
    return counts


def _required_exist(paths: SplitPaths, need_pairwise: bool) -> bool:
    req = [paths.raw_train, paths.raw_val, paths.raw_test]
    if need_pairwise:
        req.extend([paths.pair_train, paths.pair_val, paths.pair_test])
    return all(p.exists() for p in req)


def ensure_split_files(
    *,
    raw_input: Path,
    pairwise_input: Optional[Path],
    split_dir: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    overwrite: bool = False,
) -> Dict[str, Any]:
    raw_input = raw_input.resolve()
    pairwise_input = pairwise_input.resolve() if pairwise_input is not None else None
    paths = _build_split_paths(split_dir.resolve())

    if not raw_input.exists():
        raise RuntimeError(f"raw_input not found: {raw_input}")

    need_pairwise = bool(pairwise_input is not None and pairwise_input.exists())
    if (not overwrite) and _required_exist(paths, need_pairwise):
        return {
            "generated": False,
            "paths": paths,
            "need_pairwise": bool(need_pairwise),
        }

    uniq_grants = _collect_unique_grants(raw_input)
    train_set, val_set, test_set = _split_grants(
        uniq_grants,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    split_lookup: Dict[str, str] = {}
    for gid in train_set:
        split_lookup[gid] = "train"
    for gid in val_set:
        split_lookup[gid] = "val"
    for gid in test_set:
        split_lookup[gid] = "test"

    raw_counts = _write_split_jsonl(
        input_path=raw_input,
        split_lookup=split_lookup,
        output_map={
            "train": paths.raw_train,
            "val": paths.raw_val,
            "test": paths.raw_test,
        },
    )

    pair_counts: Dict[str, int] = {}
    if need_pairwise and pairwise_input is not None:
        pair_counts = _write_split_jsonl(
            input_path=pairwise_input,
            split_lookup=split_lookup,
            output_map={
                "train": paths.pair_train,
                "val": paths.pair_val,
                "test": paths.pair_test,
            },
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_input": str(raw_input),
        "pairwise_input": str(pairwise_input) if pairwise_input is not None else "",
        "split_dir": str(paths.split_dir),
        "split_policy": "deterministic_hash_by_grant_id",
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "grant_counts": {
            "total": int(len(uniq_grants)),
            "train": int(len(train_set)),
            "val": int(len(val_set)),
            "test": int(len(test_set)),
        },
        "raw_counts": raw_counts,
        "pairwise_counts": pair_counts,
        "outputs": {
            "raw_train": str(paths.raw_train),
            "raw_val": str(paths.raw_val),
            "raw_test": str(paths.raw_test),
            "pair_train": str(paths.pair_train),
            "pair_val": str(paths.pair_val),
            "pair_test": str(paths.pair_test),
        },
    }
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "generated": True,
        "paths": paths,
        "need_pairwise": bool(need_pairwise),
        "manifest": manifest,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create deterministic train/val/test split JSONL files for distillation datasets.")
    p.add_argument("--raw-input", type=str, default=RAW_INPUT_DEFAULT)
    p.add_argument("--pairwise-input", type=str, default=PAIRWISE_INPUT_DEFAULT)
    p.add_argument("--split-dir", type=str, default=DEFAULT_SPLIT_DIR)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return p


def main() -> int:
    args = _build_parser().parse_args()

    raw_input = _resolve_path(args.raw_input)
    pairwise_input = _resolve_path(args.pairwise_input) if _clean_text(args.pairwise_input) else None
    split_dir = _resolve_path(args.split_dir)

    result = ensure_split_files(
        raw_input=raw_input,
        pairwise_input=pairwise_input,
        split_dir=split_dir,
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
        overwrite=bool(args.overwrite),
    )

    paths: SplitPaths = result["paths"]
    print(f"generated={bool(result.get('generated'))}")
    print(f"split_dir={paths.split_dir}")
    print(f"raw_train={paths.raw_train}")
    print(f"raw_val={paths.raw_val}")
    print(f"raw_test={paths.raw_test}")
    if bool(result.get("need_pairwise")):
        print(f"pair_train={paths.pair_train}")
        print(f"pair_val={paths.pair_val}")
        print(f"pair_test={paths.pair_test}")
    print(f"manifest={paths.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
