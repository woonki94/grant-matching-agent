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
PAIR_STYLE_DEFAULT = "ce"
PAIR_MAX_PER_QUERY_DEFAULT = 80
PAIR_MAX_DISAGREE_PER_QUERY_DEFAULT = 6
PAIR_MAX_BOUNDARY_PER_QUERY_DEFAULT = 6
PAIR_WEAK_MIN_PER_QUERY_DEFAULT = 10
PAIR_DISAGREE_LOW_SCORE_MAX_DEFAULT = 0.30
PAIR_DISAGREE_PREFILTER_MIN_DEFAULT = 0.70
PAIR_BOUNDARY_MIN_MARGIN_DEFAULT = 0.05


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_ws(value: Any) -> str:
    return " ".join(_clean_text(value).split())


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


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


def _grant_id_and_spec_idx(row: Dict[str, Any], query_id: str) -> Tuple[str, int]:
    grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
    meta = grant.get("meta") if isinstance(grant.get("meta"), dict) else {}
    grant_id = _normalize_ws(row.get("grant_id") or meta.get("grant_id") or query_id)
    spec_idx = _safe_int(row.get("spec_idx") if row.get("spec_idx") is not None else meta.get("spec_idx"), 0)
    return grant_id, spec_idx


def _selected_aspects(row: Dict[str, Any]) -> List[str]:
    row_aspect = _normalize_ws(row.get("aspect"))
    if row_aspect in ASPECTS:
        return [row_aspect]

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
    grant_id, spec_idx = _grant_id_and_spec_idx(row, query_id)
    pair_id = _normalize_ws(row.get("pair_id")) or f"{query_id}::{doc_id}"
    out: List[Dict[str, Any]] = []

    for aspect in _selected_aspects(row):
        if row.get("aspect") == aspect and row.get("score") is not None:
            score = float(row.get("score") or 0.0)
            band = _normalize_ws(row.get("band")).lower()
        elif aspect in scores:
            score = float(scores.get(aspect) or 0.0)
            band = _normalize_ws(bands.get(aspect)).lower()
        else:
            continue
        out.append(
            {
                "aspect": aspect,
                "pair_id": pair_id,
                "grant_id": grant_id,
                "spec_idx": int(spec_idx),
                "query_id": query_id,
                "doc_id": doc_id,
                "query_text": _normalize_ws(grant.get("text")),
                "doc_text": _normalize_ws(faculty.get("text")),
                "score": score,
                "band": band,
                "lexical_prefilter_score": float(row.get("lexical_prefilter_score") or 0.0),
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


def _derive_pairwise_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    style: str,
    pos_k: int,
    hard_k: int,
    weak_k: int,
    cap: int,
    min_margin: float,
    max_disagreement: int,
    max_boundary: int,
    weak_min: int,
    disagreement_prefilter_min: float,
    disagreement_low_score_max: float,
    boundary_min_margin: float,
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        query_id = _normalize_ws(row.get("query_id"))
        doc_id = _normalize_ws(row.get("doc_id"))
        if not query_id or not doc_id:
            continue
        groups.setdefault(query_id, []).append(row)

    out: List[Dict[str, Any]] = []
    pair_style = _normalize_ws(style).lower() or PAIR_STYLE_DEFAULT
    max_pairs = max(1, int(cap))
    margin_floor = max(0.0, float(min_margin))
    p_k = max(1, int(pos_k))
    h_k = max(0, int(hard_k))
    w_k = max(1, int(weak_k))
    weak_minimum = max(0, int(weak_min))
    disagree_cap_base = max(0, int(max_disagreement))
    boundary_cap_base = max(0, int(max_boundary))
    disagree_prefilter_min = float(disagreement_prefilter_min)
    disagree_low_score_max = float(disagreement_low_score_max)
    boundary_margin_floor = max(0.0, float(boundary_min_margin))

    def doc_sort_id(row: Dict[str, Any]) -> str:
        return _normalize_ws(row.get("doc_id"))

    def is_disagreement_candidate(row: Dict[str, Any]) -> bool:
        source = _normalize_ws(row.get("pair_source")).lower()
        aspect = _normalize_ws(row.get("aspect")).lower()
        lexical = float(row.get("lexical_prefilter_score") or 0.0)
        score = float(row.get("score") or 0.0)
        source_marked_high = f"_{aspect}_high" in source or source.endswith("_high") or "high" in source
        return bool((source_marked_high or lexical >= disagree_prefilter_min) and score <= disagree_low_score_max)

    for query_id in sorted(groups):
        candidates = sorted(
            groups[query_id],
            key=lambda r: (-float(r.get("score") or 0.0), _normalize_ws(r.get("doc_id"))),
        )
        if len(candidates) < 2:
            continue

        if pair_style == "ce":
            pos = sorted(
                [row for row in candidates if _normalize_ws(row.get("band")).lower() == "high"],
                key=lambda r: (-float(r.get("score") or 0.0), doc_sort_id(r)),
            )
            mid = sorted(
                [row for row in candidates if _normalize_ws(row.get("band")).lower() == "mid"],
                key=lambda r: (float(r.get("score") or 0.0), doc_sort_id(r)),
            )
            low = sorted(
                [row for row in candidates if _normalize_ws(row.get("band")).lower() == "low"],
                key=lambda r: (float(r.get("score") or 0.0), doc_sort_id(r)),
            )
            if not pos:
                continue
            weak_target = min(weak_minimum, max_pairs, len(pos) * len(low))
            disagree_cap = min(max_pairs, disagree_cap_base, int(0.3 * max_pairs), max(0, max_pairs - weak_target))
            boundary_cap = min(max_pairs, boundary_cap_base, int(0.3 * max_pairs), max(0, max_pairs - weak_target - disagree_cap))
            dis_negatives = sorted(
                [row for row in [*low, *mid] if is_disagreement_candidate(row)],
                key=lambda r: (-float(r.get("lexical_prefilter_score") or 0.0), float(r.get("score") or 0.0), doc_sort_id(r)),
            )
        else:
            pos = candidates[: min(p_k, len(candidates))]
            mid = candidates[min(len(candidates), len(pos)) : min(len(candidates), len(pos) + h_k)]
            low = candidates[-min(w_k, len(candidates)) :]
            weak_target = len(pos) * len(low)
            disagree_cap = 0
            boundary_cap = 0
            dis_negatives = []

        seen_pairs: set[Tuple[str, str, str]] = set()
        row_count = 0

        def add_pair(pos_row: Dict[str, Any], neg_row: Dict[str, Any], pair_type: str, min_pair_margin: float = 0.0) -> bool:
            nonlocal row_count
            if row_count >= max_pairs:
                return False
            pos_doc = _normalize_ws(pos_row.get("doc_id"))
            neg_doc = _normalize_ws(neg_row.get("doc_id"))
            if not pos_doc or not neg_doc or pos_doc == neg_doc:
                return False
            key = (pair_type, pos_doc, neg_doc)
            if key in seen_pairs:
                return False
            pos_score = float(pos_row.get("score") or 0.0)
            neg_score = float(neg_row.get("score") or 0.0)
            margin = float(pos_score - neg_score)
            if margin <= 0.0 or margin < max(margin_floor, min_pair_margin):
                return False
            seen_pairs.add(key)
            row_count += 1
            grant_id = _normalize_ws(pos_row.get("grant_id") or query_id)
            spec_idx = _safe_int(pos_row.get("spec_idx"), 0)
            pos_fac_id = _safe_int(pos_doc, 0)
            neg_fac_id = _safe_int(neg_doc, 0)
            out.append(
                {
                    "aspect": _normalize_ws(pos_row.get("aspect")),
                    "split": _normalize_ws(pos_row.get("split")),
                    "grant_id": grant_id,
                    "spec_idx": int(spec_idx),
                    "query_id": query_id,
                    "query_text": _normalize_ws(pos_row.get("query_text")),
                    "pos_doc_id": pos_doc,
                    "neg_doc_id": neg_doc,
                    "pos_pair_id": _normalize_ws(pos_row.get("pair_id")),
                    "neg_pair_id": _normalize_ws(neg_row.get("pair_id")),
                    "pos_text": _normalize_ws(pos_row.get("doc_text")),
                    "neg_text": _normalize_ws(neg_row.get("doc_text")),
                    "teacher_pos_score": pos_score,
                    "teacher_neg_score": neg_score,
                    "teacher_margin": margin,
                    "pos_band": _normalize_ws(pos_row.get("band")).lower(),
                    "neg_band": _normalize_ws(neg_row.get("band")).lower(),
                    "pos_fac_id": int(pos_fac_id),
                    "pos_fac_spec_id": int(pos_fac_id),
                    "pos_fac_spec_idx": 0,
                    "pos_section": _normalize_ws(pos_row.get("source")) or "unknown",
                    "pos_sts_rank": -1,
                    "pos_sts_rank_percentile": 1.0,
                    "neg_fac_id": int(neg_fac_id),
                    "neg_fac_spec_id": int(neg_fac_id),
                    "neg_fac_spec_idx": 0,
                    "neg_section": _normalize_ws(neg_row.get("source")) or "unknown",
                    "neg_sts_rank": -1,
                    "neg_sts_rank_percentile": 1.0,
                    "pair_type": pair_type,
                }
            )
            return True

        if pair_style == "ce":
            added = 0
            for pos_row in pos:
                for neg_row in dis_negatives:
                    if row_count >= max_pairs or added >= disagree_cap:
                        break
                    if add_pair(pos_row, neg_row, "llm_disagreement", min_pair_margin=0.15):
                        added += 1
                if row_count >= max_pairs or added >= disagree_cap:
                    break

            added = 0
            for pos_row in pos:
                for neg_row in mid:
                    if row_count >= max_pairs or added >= boundary_cap:
                        break
                    if add_pair(pos_row, neg_row, "strong_vs_boundary", min_pair_margin=boundary_margin_floor):
                        added += 1
                if row_count >= max_pairs or added >= boundary_cap:
                    break

            added = 0
            for pos_row in pos:
                for neg_row in low:
                    if row_count >= max_pairs or added >= weak_target:
                        break
                    if add_pair(pos_row, neg_row, "strong_vs_weak"):
                        added += 1
                if row_count >= max_pairs or added >= weak_target:
                    break

            hard_pool = sorted([*low, *mid], key=lambda r: (-float(r.get("score") or 0.0), doc_sort_id(r)))
            for pos_row in pos:
                for neg_row in hard_pool:
                    if row_count >= max_pairs:
                        break
                    add_pair(pos_row, neg_row, "strong_vs_hard")
                if row_count >= max_pairs:
                    break
        else:
            for pos_row in pos:
                for neg_row in low:
                    add_pair(pos_row, neg_row, "derived_strong_vs_weak")
                    if row_count >= max_pairs:
                        break
                if row_count >= max_pairs:
                    break
                for neg_row in mid:
                    add_pair(pos_row, neg_row, "derived_strong_vs_hard")
                    if row_count >= max_pairs:
                        break
                if row_count >= max_pairs:
                    break

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export CE2 distillation rows into aspect-specific train/val/test JSONL files.")
    p.add_argument("--distillation-input", type=str, default=DISTILLATION_OUTPUT_DEFAULT)
    p.add_argument("--augmentation-input", type=str, default=AUGMENTATION_INPUT_DEFAULT)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--include-augmentation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--write-pairwise", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pair-style", type=str, choices=["ce", "window"], default=PAIR_STYLE_DEFAULT)
    p.add_argument("--pair-pos-k", type=int, default=4)
    p.add_argument("--pair-hard-k", type=int, default=4)
    p.add_argument("--pair-weak-k", type=int, default=4)
    p.add_argument("--pair-cap-per-query", type=int, default=PAIR_MAX_PER_QUERY_DEFAULT)
    p.add_argument("--pair-min-margin", type=float, default=0.0)
    p.add_argument("--pair-max-disagreement-per-query", type=int, default=PAIR_MAX_DISAGREE_PER_QUERY_DEFAULT)
    p.add_argument("--pair-max-boundary-per-query", type=int, default=PAIR_MAX_BOUNDARY_PER_QUERY_DEFAULT)
    p.add_argument("--pair-weak-min-per-query", type=int, default=PAIR_WEAK_MIN_PER_QUERY_DEFAULT)
    p.add_argument("--pair-disagreement-prefilter-min", type=float, default=PAIR_DISAGREE_PREFILTER_MIN_DEFAULT)
    p.add_argument("--pair-disagreement-low-score-max", type=float, default=PAIR_DISAGREE_LOW_SCORE_MAX_DEFAULT)
    p.add_argument("--pair-boundary-min-margin", type=float, default=PAIR_BOUNDARY_MIN_MARGIN_DEFAULT)
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
    if bool(args.write_pairwise):
        outputs.extend(
            output_dir / f"{aspect}_pairwise_{split}.jsonl"
            for aspect in ASPECTS
            for split in ("train", "val", "test")
        )
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
    pair_counts: Dict[str, Dict[str, int]] = {aspect: {"train": 0, "val": 0, "test": 0} for aspect in ASPECTS}
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
            if bool(args.write_pairwise):
                pair_rows = _derive_pairwise_rows(
                    rows,
                    style=str(args.pair_style),
                    pos_k=int(args.pair_pos_k),
                    hard_k=int(args.pair_hard_k),
                    weak_k=int(args.pair_weak_k),
                    cap=int(args.pair_cap_per_query),
                    min_margin=float(args.pair_min_margin),
                    max_disagreement=int(args.pair_max_disagreement_per_query),
                    max_boundary=int(args.pair_max_boundary_per_query),
                    weak_min=int(args.pair_weak_min_per_query),
                    disagreement_prefilter_min=float(args.pair_disagreement_prefilter_min),
                    disagreement_low_score_max=float(args.pair_disagreement_low_score_max),
                    boundary_min_margin=float(args.pair_boundary_min_margin),
                )
                pair_counts[aspect][split] = _write_rows(output_dir / f"{aspect}_pairwise_{split}.jsonl", pair_rows)

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
        "pairwise_enabled": bool(args.write_pairwise),
        "pairwise_config": {
            "style": str(args.pair_style),
            "pos_k": int(args.pair_pos_k),
            "hard_k": int(args.pair_hard_k),
            "weak_k": int(args.pair_weak_k),
            "cap_per_query": int(args.pair_cap_per_query),
            "min_margin": float(args.pair_min_margin),
            "max_disagreement_per_query": int(args.pair_max_disagreement_per_query),
            "max_boundary_per_query": int(args.pair_max_boundary_per_query),
            "weak_min_per_query": int(args.pair_weak_min_per_query),
            "disagreement_prefilter_min": float(args.pair_disagreement_prefilter_min),
            "disagreement_low_score_max": float(args.pair_disagreement_low_score_max),
            "boundary_min_margin": float(args.pair_boundary_min_margin),
        },
        "pairwise_row_counts": pair_counts,
        "total_aspect_rows": int(len(aspect_rows)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"output_dir={output_dir}")
    print(f"manifest={manifest_path}")
    print(f"query_split_counts={json.dumps(split_counts, ensure_ascii=False)}")
    print(f"row_counts={json.dumps(counts, ensure_ascii=False)}")
    print(f"pairwise_row_counts={json.dumps(pair_counts, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
