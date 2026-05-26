from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce2").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce2.data_preparation.utils import ASPECTS, DISTILLATION_OUTPUT_DEFAULT, resolve_path  # noqa: E402


AUGMENTATION_INPUT_DEFAULT = "ce2/dataset/distill/augmentation.jsonl"
OUTPUT_DIR_DEFAULT = "ce2/dataset/splits"
MANIFEST_BASENAME = "split_manifest.json"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_ws(value: Any) -> str:
    return " ".join(_clean_text(value).split())


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = _clean_text(raw)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {type(exc).__name__}: {exc}") from exc
            if isinstance(obj, dict):
                yield obj


def _split_sizes(n: int, *, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if int(n) <= 0:
        return 0, 0, 0
    val = min(max(0.0, float(val_ratio)), 0.49)
    test = min(max(0.0, float(test_ratio)), 0.49)
    if val + test >= 0.99:
        test = max(0.0, 0.99 - val)
    n_val = int(round(float(n) * val))
    n_test = int(round(float(n) * test))
    n_train = int(n) - n_val - n_test
    if n_train <= 0:
        if n_val > 0:
            n_val -= 1
            n_train += 1
        elif n_test > 0:
            n_test -= 1
            n_train += 1
    return n_train, n_val, n_test


def _build_split_map(
    *,
    query_ids: Sequence[str],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    ids = sorted({_normalize_ws(x) for x in query_ids if _normalize_ws(x)})
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    n_train, n_val, n_test = _split_sizes(len(ids), val_ratio=val_ratio, test_ratio=test_ratio)

    out: Dict[str, str] = {}
    idx = 0
    for query_id in ids[idx : idx + n_test]:
        out[query_id] = "test"
    idx += n_test
    for query_id in ids[idx : idx + n_val]:
        out[query_id] = "val"
    idx += n_val
    for query_id in ids[idx : idx + n_train]:
        out[query_id] = "train"
    return out, {"train": n_train, "val": n_val, "test": n_test, "total": len(ids)}


def _query_id(row: Dict[str, Any]) -> str:
    grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
    return _normalize_ws(grant.get("item_id"))


def _selected_aspects(row: Dict[str, Any]) -> List[str]:
    clusters = row.get("distill_selected_clusters")
    if isinstance(clusters, list):
        aspects = []
        for cluster in clusters:
            if not isinstance(cluster, dict):
                continue
            aspect = _normalize_ws(cluster.get("aspect"))
            if aspect in ASPECTS:
                aspects.append(aspect)
        if aspects:
            return sorted(set(aspects), key=aspects.index)

    augment_aspect = _normalize_ws(row.get("augment_target_aspect"))
    if augment_aspect in ASPECTS:
        return [augment_aspect]

    return list(ASPECTS)


def _to_aspect_rows(row: Dict[str, Any], *, source_file: str) -> List[Dict[str, Any]]:
    grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
    faculty = row.get("faculty") if isinstance(row.get("faculty"), dict) else {}
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    bands = row.get("bands") if isinstance(row.get("bands"), dict) else {}
    query_id = _normalize_ws(grant.get("item_id"))
    doc_id = _normalize_ws(faculty.get("item_id"))
    pair_id = _normalize_ws(row.get("pair_id")) or f"{query_id}::{doc_id}"
    out: List[Dict[str, Any]] = []

    for aspect in _selected_aspects(row):
        if aspect not in scores:
            continue
        score = float(scores.get(aspect) or 0.0)
        band = _normalize_ws(bands.get(aspect)).lower()
        out.append(
            {
                "aspect": aspect,
                "pair_id": pair_id,
                "query_id": query_id,
                "doc_id": doc_id,
                "query_text": _normalize_ws(grant.get("text")),
                "doc_text": _normalize_ws(faculty.get("text")),
                "score": score,
                "band": band,
                "source": "augmentation" if bool(row.get("is_augmented")) else "distillation",
                "source_file": source_file,
                "pair_source": _normalize_ws(row.get("pair_source")),
                "model_id": _normalize_ws(row.get("model_id")),
                "grant_meta": grant.get("meta") if isinstance(grant.get("meta"), dict) else {},
                "faculty_meta": faculty.get("meta") if isinstance(faculty.get("meta"), dict) else {},
            }
        )
    return out


def _load_aspect_rows(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for row in _iter_jsonl(path):
            rows.extend(_to_aspect_rows(row, source_file=str(path)))
    return rows


def _write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export CE2 distillation rows into aspect-specific train/val/test JSONL files.")
    p.add_argument("--distillation-input", type=str, default=DISTILLATION_OUTPUT_DEFAULT)
    p.add_argument("--augmentation-input", type=str, default=AUGMENTATION_INPUT_DEFAULT)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--include-augmentation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    distillation_path = resolve_path(PROJECT_ROOT, args.distillation_input)
    augmentation_path = resolve_path(PROJECT_ROOT, args.augmentation_input)
    output_dir = resolve_path(PROJECT_ROOT, args.output_dir)
    manifest_path = output_dir / MANIFEST_BASENAME

    if not distillation_path.exists():
        raise FileNotFoundError(f"Distillation input not found: {distillation_path}")

    inputs = [distillation_path]
    if bool(args.include_augmentation) and augmentation_path.exists():
        inputs.append(augmentation_path)

    outputs = [
        output_dir / f"{aspect}_{split}.jsonl"
        for aspect in ASPECTS
        for split in ("train", "val", "test")
    ]
    outputs.append(manifest_path)
    if not bool(args.overwrite):
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError("Split outputs already exist. Use --overwrite.\n" + "\n".join(existing))

    aspect_rows = _load_aspect_rows(inputs)
    split_map, split_counts = _build_split_map(
        query_ids=[_normalize_ws(row.get("query_id")) for row in aspect_rows],
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )

    counts: Dict[str, Dict[str, int]] = {aspect: {"train": 0, "val": 0, "test": 0} for aspect in ASPECTS}
    for aspect in ASPECTS:
        for split in ("train", "val", "test"):
            rows = [
                row
                for row in aspect_rows
                if row.get("aspect") == aspect and split_map.get(_normalize_ws(row.get("query_id"))) == split
            ]
            for row in rows:
                row["split"] = split
            counts[aspect][split] = _write_rows(output_dir / f"{aspect}_{split}.jsonl", rows)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "distillation_input": str(distillation_path),
        "augmentation_input": str(augmentation_path) if augmentation_path in inputs else "",
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "query_split_counts": split_counts,
        "row_counts": counts,
        "total_aspect_rows": int(len(aspect_rows)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"output_dir={output_dir}")
    print(f"manifest={manifest_path}")
    print(f"query_split_counts={json.dumps(split_counts, ensure_ascii=False)}")
    print(f"row_counts={json.dumps(counts, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
