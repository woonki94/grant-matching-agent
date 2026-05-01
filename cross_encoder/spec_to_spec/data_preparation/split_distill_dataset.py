from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


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

HIGH_THRESHOLD_DEFAULT = 0.67
MID_THRESHOLD_DEFAULT = 0.34

SPLITS: Tuple[str, str, str] = ("train", "val", "test")
BANDS: Tuple[str, str, str] = ("high", "mid", "low")


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


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp_01(value: float) -> float:
    x = float(value)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _hash_to_unit_interval(text: str) -> float:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return int(h, 16) / float(0xFFFFFFFF)


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


def _parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    text = _clean_text(line)
    if not text:
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _extract_query_key(obj: Dict[str, Any]) -> str:
    grant_id = _clean_text(obj.get("grant_id"))
    if not grant_id:
        return ""
    spec_idx = _safe_int(obj.get("spec_idx"), default=0)
    if spec_idx < 0:
        spec_idx = 0
    return f"{grant_id}::spec#{spec_idx}"


def _score_from_raw_obj(obj: Dict[str, Any]) -> Optional[float]:
    best: Optional[float] = None
    candidates = obj.get("candidates")
    if isinstance(candidates, list):
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            for key in ("score_raw", "score", "teacher_score_raw", "teacher_score"):
                if key in cand:
                    val = _clamp_01(_safe_float(cand.get(key), default=-1.0))
                    if best is None or val > best:
                        best = val
                    break
    if best is not None:
        return float(best)

    for key in ("teacher_score_used", "teacher_score_raw", "teacher_score", "score_raw", "score"):
        if key in obj:
            return float(_clamp_01(_safe_float(obj.get(key), default=0.0)))
    return None


def _score_from_pair_obj(obj: Dict[str, Any]) -> Optional[float]:
    for key in ("teacher_pos_score_raw", "teacher_pos_score", "teacher_score_used", "teacher_score_raw", "teacher_score", "score"):
        if key in obj:
            return float(_clamp_01(_safe_float(obj.get(key), default=0.0)))
    return None


def _band_from_score(score: float, *, high_threshold: float, mid_threshold: float) -> str:
    s = _clamp_01(score)
    if s >= high_threshold:
        return "high"
    if s >= mid_threshold:
        return "mid"
    return "low"


def _empty_band_counts() -> Dict[str, int]:
    return {"high": 0, "mid": 0, "low": 0, "total": 0}


def _split_sizes(n: int, *, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    vr = max(0.0, min(0.49, float(val_ratio)))
    tr = max(0.0, min(0.49, float(test_ratio)))
    if vr + tr >= 0.99:
        tr = max(0.0, 0.99 - vr)

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

    n_train = n - n_val - n_test
    if n_train <= 0:
        if n_val > 0:
            n_val -= 1
            n_train += 1
        elif n_test > 0:
            n_test -= 1
            n_train += 1
    return int(n_train), int(n_val), int(n_test)


def _collect_query_score_bands(
    raw_input: Path,
    *,
    high_threshold: float,
    mid_threshold: float,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    query_best_score: Dict[str, float] = {}
    stats: Dict[str, Any] = {
        "raw_rows_total": 0,
        "raw_parse_fail": 0,
        "raw_missing_query_key": 0,
        "raw_missing_score": 0,
        "raw_duplicate_query_key": 0,
    }

    with raw_input.open("r", encoding="utf-8") as f:
        for line in f:
            stats["raw_rows_total"] += 1
            obj = _parse_json_line(line)
            if obj is None:
                stats["raw_parse_fail"] += 1
                continue

            query_key = _extract_query_key(obj)
            if not query_key:
                stats["raw_missing_query_key"] += 1
                continue

            score = _score_from_raw_obj(obj)
            if score is None:
                stats["raw_missing_score"] += 1
                continue

            prev = query_best_score.get(query_key)
            if prev is not None:
                stats["raw_duplicate_query_key"] += 1
                if score <= prev:
                    continue
            query_best_score[query_key] = float(score)

    query_band_map: Dict[str, str] = {}
    overall_counts = _empty_band_counts()
    for key, score in query_best_score.items():
        band = _band_from_score(score, high_threshold=high_threshold, mid_threshold=mid_threshold)
        query_band_map[key] = band
        overall_counts[band] += 1
        overall_counts["total"] += 1

    stats["query_count"] = int(len(query_band_map))
    stats["query_band_counts"] = overall_counts
    return query_band_map, stats


def _split_query_keys_stratified(
    query_band_map: Dict[str, str],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    keys_by_band: Dict[str, List[str]] = {b: [] for b in BANDS}
    for key, band in query_band_map.items():
        if band not in keys_by_band:
            band = "low"
        keys_by_band[band].append(key)

    split_lookup: Dict[str, str] = {}
    split_band_counts: Dict[str, Dict[str, int]] = {s: _empty_band_counts() for s in SPLITS}

    for band in BANDS:
        keys = list(keys_by_band[band])
        keys.sort(key=lambda x: (_hash_to_unit_interval(f"{int(seed)}::{band}::{x}"), x))

        _, n_val, n_test = _split_sizes(len(keys), val_ratio=val_ratio, test_ratio=test_ratio)
        test_keys = keys[:n_test]
        val_keys = keys[n_test : n_test + n_val]
        train_keys = keys[n_test + n_val :]

        for k in train_keys:
            split_lookup[k] = "train"
        for k in val_keys:
            split_lookup[k] = "val"
        for k in test_keys:
            split_lookup[k] = "test"

        split_band_counts["train"][band] += int(len(train_keys))
        split_band_counts["val"][band] += int(len(val_keys))
        split_band_counts["test"][band] += int(len(test_keys))

    for split in SPLITS:
        split_band_counts[split]["total"] = (
            int(split_band_counts[split]["high"])
            + int(split_band_counts[split]["mid"])
            + int(split_band_counts[split]["low"])
        )
    return split_lookup, split_band_counts


def _write_split_jsonl_by_query_key(
    *,
    input_path: Path,
    split_lookup: Dict[str, str],
    output_map: Dict[str, Path],
) -> Dict[str, int]:
    output_map = {k: v.resolve() for k, v in output_map.items()}
    tmp_map = {k: v.with_suffix(v.suffix + ".tmp") for k, v in output_map.items()}
    for p in output_map.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    counts = {
        "total": 0,
        "written_train": 0,
        "written_val": 0,
        "written_test": 0,
        "missing_query_key": 0,
        "parse_fail": 0,
        "missing_split_assignment": 0,
    }

    handles: Dict[str, Any] = {
        "train": tmp_map["train"].open("w", encoding="utf-8"),
        "val": tmp_map["val"].open("w", encoding="utf-8"),
        "test": tmp_map["test"].open("w", encoding="utf-8"),
    }
    try:
        with input_path.open("r", encoding="utf-8") as src:
            for raw_line in src:
                counts["total"] += 1
                obj = _parse_json_line(raw_line)
                if obj is None:
                    counts["parse_fail"] += 1
                    continue
                query_key = _extract_query_key(obj)
                if not query_key:
                    counts["missing_query_key"] += 1
                    continue
                split = split_lookup.get(query_key)
                if split not in {"train", "val", "test"}:
                    counts["missing_split_assignment"] += 1
                    continue
                handles[split].write(raw_line.rstrip("\n") + "\n")
                counts[f"written_{split}"] += 1
    finally:
        for h in handles.values():
            h.close()

    for key in SPLITS:
        tmp_map[key].replace(output_map[key])
    return counts


def _count_score_bands_in_file(
    path: Path,
    *,
    score_getter: Callable[[Dict[str, Any]], Optional[float]],
    high_threshold: float,
    mid_threshold: float,
) -> Dict[str, int]:
    out = _empty_band_counts()
    out["parse_fail"] = 0
    out["missing_score"] = 0
    if not path.exists():
        return out

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = _parse_json_line(line)
            if obj is None:
                out["parse_fail"] += 1
                continue
            score = score_getter(obj)
            if score is None:
                out["missing_score"] += 1
                continue
            band = _band_from_score(score, high_threshold=high_threshold, mid_threshold=mid_threshold)
            out[band] += 1
            out["total"] += 1
    return out


def _collect_file_band_counts(
    paths: SplitPaths,
    *,
    need_pairwise: bool,
    high_threshold: float,
    mid_threshold: float,
) -> Dict[str, Dict[str, Dict[str, int]]]:
    raw_counts = {
        "train": _count_score_bands_in_file(
            paths.raw_train,
            score_getter=_score_from_raw_obj,
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
        ),
        "val": _count_score_bands_in_file(
            paths.raw_val,
            score_getter=_score_from_raw_obj,
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
        ),
        "test": _count_score_bands_in_file(
            paths.raw_test,
            score_getter=_score_from_raw_obj,
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
        ),
    }

    pair_counts: Dict[str, Dict[str, int]] = {}
    if need_pairwise:
        pair_counts = {
            "train": _count_score_bands_in_file(
                paths.pair_train,
                score_getter=_score_from_pair_obj,
                high_threshold=high_threshold,
                mid_threshold=mid_threshold,
            ),
            "val": _count_score_bands_in_file(
                paths.pair_val,
                score_getter=_score_from_pair_obj,
                high_threshold=high_threshold,
                mid_threshold=mid_threshold,
            ),
            "test": _count_score_bands_in_file(
                paths.pair_test,
                score_getter=_score_from_pair_obj,
                high_threshold=high_threshold,
                mid_threshold=mid_threshold,
            ),
        }

    return {
        "raw_file_band_counts": raw_counts,
        "pair_file_band_counts": pair_counts,
    }


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
    high_threshold: float = HIGH_THRESHOLD_DEFAULT,
    mid_threshold: float = MID_THRESHOLD_DEFAULT,
) -> Dict[str, Any]:
    raw_input = raw_input.resolve()
    pairwise_input = pairwise_input.resolve() if pairwise_input is not None else None
    paths = _build_split_paths(split_dir.resolve())

    if not raw_input.exists():
        raise RuntimeError(f"raw_input not found: {raw_input}")

    need_pairwise = bool(pairwise_input is not None and pairwise_input.exists())
    high_threshold = _clamp_01(_safe_float(high_threshold, default=HIGH_THRESHOLD_DEFAULT))
    mid_threshold = _clamp_01(_safe_float(mid_threshold, default=MID_THRESHOLD_DEFAULT))
    if mid_threshold > high_threshold:
        mid_threshold = high_threshold

    if (not overwrite) and _required_exist(paths, need_pairwise):
        existing_manifest: Dict[str, Any] = {}
        if paths.manifest.exists():
            try:
                existing_manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
            except Exception:
                existing_manifest = {}
        band_stats = _collect_file_band_counts(
            paths,
            need_pairwise=need_pairwise,
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
        )
        return {
            "generated": False,
            "paths": paths,
            "need_pairwise": bool(need_pairwise),
            "manifest": existing_manifest,
            **band_stats,
        }

    query_band_map, query_stats = _collect_query_score_bands(
        raw_input,
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
    )
    if not query_band_map:
        raise RuntimeError("No valid query keys were extracted from raw_input.")

    split_lookup, split_query_band_counts = _split_query_keys_stratified(
        query_band_map,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    raw_counts = _write_split_jsonl_by_query_key(
        input_path=raw_input,
        split_lookup=split_lookup,
        output_map={
            "train": paths.raw_train,
            "val": paths.raw_val,
            "test": paths.raw_test,
        },
    )

    pairwise_counts: Dict[str, int] = {}
    if need_pairwise and pairwise_input is not None:
        pairwise_counts = _write_split_jsonl_by_query_key(
            input_path=pairwise_input,
            split_lookup=split_lookup,
            output_map={
                "train": paths.pair_train,
                "val": paths.pair_val,
                "test": paths.pair_test,
            },
        )

    file_band_counts = _collect_file_band_counts(
        paths,
        need_pairwise=need_pairwise,
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_input": str(raw_input),
        "pairwise_input": str(pairwise_input) if pairwise_input is not None else "",
        "split_dir": str(paths.split_dir),
        "split_policy": "stratified_by_query_score_band",
        "split_unit": "grant_id + spec_idx",
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "score_band_thresholds": {
            "high_min": float(high_threshold),
            "mid_min": float(mid_threshold),
        },
        "query_stats": query_stats,
        "query_band_counts_by_split": split_query_band_counts,
        "raw_counts": raw_counts,
        "pairwise_counts": pairwise_counts,
        **file_band_counts,
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
        **file_band_counts,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create deterministic train/val/test split JSONL files for distillation datasets.")
    p.add_argument("--raw-input", type=str, default=RAW_INPUT_DEFAULT)
    p.add_argument("--pairwise-input", type=str, default=PAIRWISE_INPUT_DEFAULT)
    p.add_argument("--split-dir", type=str, default=DEFAULT_SPLIT_DIR)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--high-threshold", type=float, default=HIGH_THRESHOLD_DEFAULT)
    p.add_argument("--mid-threshold", type=float, default=MID_THRESHOLD_DEFAULT)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return p


def _print_band_counts(title: str, data: Dict[str, Dict[str, int]]) -> None:
    print(title)
    for split in SPLITS:
        c = data.get(split) or {}
        print(
            f"  {split}: total={int(c.get('total', 0))} "
            f"high={int(c.get('high', 0))} mid={int(c.get('mid', 0))} low={int(c.get('low', 0))}"
        )


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
        high_threshold=float(args.high_threshold),
        mid_threshold=float(args.mid_threshold),
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

    raw_file_band_counts = result.get("raw_file_band_counts") or {}
    if isinstance(raw_file_band_counts, dict):
        _print_band_counts("raw_file_band_counts", raw_file_band_counts)

    pair_file_band_counts = result.get("pair_file_band_counts") or {}
    if bool(result.get("need_pairwise")) and isinstance(pair_file_band_counts, dict):
        _print_band_counts("pair_file_band_counts", pair_file_band_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
