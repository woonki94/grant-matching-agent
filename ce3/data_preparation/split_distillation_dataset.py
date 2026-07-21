from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce3").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce3.data_preparation.llm_runtime import clean_text, normalize_ws, score_to_band
from ce3.data_preparation.utils import resolve_path


DISTILLATION_INPUT_DEFAULT = "ce3/dataset/distill/llm_distillation.jsonl"
OUTPUT_DIR_DEFAULT = "ce3/dataset/splits"
ASPECTS = ("topic", "approach", "objective")
PAIR_MAX_PER_QUERY_DEFAULT = 80
PAIR_MAX_DISAGREEMENT_PER_QUERY_DEFAULT = 6
PAIR_MAX_BOUNDARY_PER_QUERY_DEFAULT = 6
PAIR_WEAK_MIN_PER_QUERY_DEFAULT = 10
PAIR_DISAGREE_PREFILTER_MIN_DEFAULT = 0.70
PAIR_DISAGREE_TEACHER_MAX_DEFAULT = 0.30
PAIR_DISAGREE_MIN_MARGIN_DEFAULT = 0.15
PAIR_BOUNDARY_MIN_MARGIN_DEFAULT = 0.05
PREFIX_BY_ASPECT = {
    "topic": "[TOPIC]",
    "approach": "[APPROACH]",
    "objective": "[OBJECTIVE]",
}


def _clamp_01(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


def _stable_unit_interval(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(0xFFFFFFFFFFFF)


def _assign_split_names(keys: Sequence[str], *, seed: int, val_ratio: float, test_ratio: float) -> Dict[str, str]:
    unique = sorted({clean_text(key) for key in keys if clean_text(key)})
    if not unique:
        return {}
    val = max(0.0, min(0.49, float(val_ratio)))
    test = max(0.0, min(0.49, float(test_ratio)))
    if val + test >= 0.99:
        test = max(0.0, 0.99 - val)

    ordered = sorted(unique, key=lambda key: (_stable_unit_interval(f"{int(seed)}::{key}"), key))
    n = len(ordered)
    n_test = int(round(n * test))
    n_val = int(round(n * val))
    if n >= 3:
        if test > 0.0:
            n_test = max(1, n_test)
        if val > 0.0:
            n_val = max(1, n_val)
    if n_test + n_val >= n:
        overflow = n_test + n_val - (n - 1)
        reduce_test = min(max(0, overflow), n_test)
        n_test -= reduce_test
        overflow -= reduce_test
        if overflow > 0:
            n_val = max(0, n_val - overflow)

    out = {key: "train" for key in ordered}
    for key in ordered[:n_test]:
        out[key] = "test"
    for key in ordered[n_test : n_test + n_val]:
        out[key] = "val"
    return out


def _meta_dict(obj: Dict[str, Any], side: str) -> Dict[str, Any]:
    side_obj = obj.get(side)
    if not isinstance(side_obj, dict):
        return {}
    meta = side_obj.get("meta")
    return dict(meta) if isinstance(meta, dict) else {}


def _root_grant_key(row: Dict[str, Any]) -> str:
    grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
    item_id = clean_text(grant.get("item_id")) if isinstance(grant, dict) else ""
    meta = _meta_dict(row, "grant")
    source_meta = meta.get("source_meta") if isinstance(meta.get("source_meta"), dict) else {}

    for source in (source_meta, meta):
        grant_id = clean_text(source.get("grant_id"))
        if grant_id:
            return f"grant:{grant_id}"

    source_item_id = clean_text(meta.get("source_item_id"))
    for candidate in (source_item_id, item_id):
        if candidate.startswith("grant:"):
            parts = candidate.split(":")
            if len(parts) >= 2:
                return f"grant:{parts[1]}"
        if candidate.startswith("aug:grant:grant:"):
            parts = candidate.split(":")
            if len(parts) >= 4:
                return f"grant:{parts[3]}"

    return item_id or clean_text(row.get("pair_id")) or "unknown"


def _spec_idx(row: Dict[str, Any]) -> int:
    meta = _meta_dict(row, "grant")
    source_meta = meta.get("source_meta") if isinstance(meta.get("source_meta"), dict) else {}
    for source in (meta, source_meta):
        for key in ("grant_spec_idx", "spec_idx", "source_spec_idx"):
            try:
                return int(source.get(key))
            except Exception:
                continue
    grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
    item_id = clean_text(grant.get("item_id")) if isinstance(grant, dict) else ""
    if item_id.startswith("grant:"):
        parts = item_id.split(":")
        if len(parts) >= 3:
            try:
                return int(parts[2])
            except Exception:
                return 0
    return 0


def _grant_id_for_output(row: Dict[str, Any]) -> str:
    grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
    item_id = clean_text(grant.get("item_id")) if isinstance(grant, dict) else ""
    return item_id or clean_text(row.get("pair_id"))


def _prefixed(text: str, aspect: str, *, prefix_mode: str) -> str:
    text = normalize_ws(text)
    mode = clean_text(prefix_mode).lower()
    if mode == "none":
        return text
    prefix = PREFIX_BY_ASPECT.get(aspect, f"[{aspect.upper()}]")
    if text.startswith(prefix):
        return text
    return normalize_ws(f"{prefix} {text}")


def _iter_distillation_rows(path: Path, aspects: Sequence[str]) -> Iterable[Dict[str, Any]]:
    aspect_set = set(aspects)
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            if not bool(row.get("parse_ok", True)):
                continue
            aspect = clean_text(row.get("aspect"))
            if aspect not in aspect_set:
                continue
            grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
            faculty = row.get("faculty") if isinstance(row.get("faculty"), dict) else {}
            if not normalize_ws(grant.get("text")) or not normalize_ws(faculty.get("text")):
                continue
            yield row


def _dedupe_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        pair_id = clean_text(row.get("pair_id"))
        if not pair_id:
            grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
            faculty = row.get("faculty") if isinstance(row.get("faculty"), dict) else {}
            pair_id = "::".join(
                [
                    clean_text(grant.get("item_id")),
                    clean_text(faculty.get("item_id")),
                    clean_text(row.get("aspect")),
                ]
            )
        if not pair_id:
            continue
        old = best.get(pair_id)
        if old is None or int(row.get("attempt", 999999)) < int(old.get("attempt", 999999)):
            best[pair_id] = row
    return list(best.values())


def _doc_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    faculty = row.get("faculty") if isinstance(row.get("faculty"), dict) else {}
    score = _clamp_01(row.get("score"))
    band = clean_text(row.get("band")) or score_to_band(score)
    return {
        "text": normalize_ws(faculty.get("text")),
        "teacher_score": float(score),
        "teacher_score_raw": float(score),
        "teacher_score_norm": float(score),
        "score": float(score),
        "score_raw": float(score),
        "score_norm": float(score),
        "target_cluster": band,
        "band": band,
        "pair_id": clean_text(row.get("pair_id")),
        "fac_item_id": clean_text(faculty.get("item_id")),
        "fac_kind": clean_text(faculty.get("kind")),
        "fac_meta": faculty.get("meta") if isinstance(faculty.get("meta"), dict) else {},
        "ce_prefilter_score": _clamp_01(row.get("ce_prefilter_score")),
        "ce_prefilter_rank": int(row.get("ce_prefilter_rank", 0) or 0),
        "pair_source": clean_text(row.get("pair_source")),
        "selected_for_target": True,
        "is_augmented": clean_text(faculty.get("kind")) == "fac_aug",
    }


def _build_listwise_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    aspect: str,
    prefix_mode: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    first_row: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
        grant_id = clean_text(grant.get("item_id"))
        if not grant_id:
            continue
        grouped[grant_id].append(row)
        first_row.setdefault(grant_id, row)

    out: List[Dict[str, Any]] = []
    for grant_item_id, group_rows in sorted(grouped.items()):
        first = first_row[grant_item_id]
        grant = first.get("grant") if isinstance(first.get("grant"), dict) else {}
        query_text_raw = normalize_ws(grant.get("text"))
        docs = [_doc_from_row(row) for row in group_rows]
        docs = [doc for doc in docs if normalize_ws(doc.get("text"))]
        docs.sort(
            key=lambda doc: (
                -float(doc.get("teacher_score", 0.0)),
                int(doc.get("ce_prefilter_rank", 0)),
                clean_text(doc.get("fac_item_id")),
            )
        )
        if not query_text_raw or not docs:
            continue
        out.append(
            {
                "grant_id": _grant_id_for_output(first),
                "spec_idx": _spec_idx(first),
                "aspect": aspect,
                "query_item_id": grant_item_id,
                "query_kind": clean_text(grant.get("kind")),
                "query_text": _prefixed(query_text_raw, aspect, prefix_mode=prefix_mode),
                "spec_text": _prefixed(query_text_raw, aspect, prefix_mode=prefix_mode),
                "raw_query_text": query_text_raw,
                "grant_meta": grant.get("meta") if isinstance(grant.get("meta"), dict) else {},
                "docs": docs,
                "ranked_docs": docs,
                "candidates": docs,
            }
        )
    return out


def _pair_type(pos_band: str, neg_band: str) -> str:
    if pos_band and neg_band and pos_band != neg_band:
        return f"{pos_band}_vs_{neg_band}"
    return "score_ordered"


def _doc_id(doc: Dict[str, Any]) -> str:
    return clean_text(doc.get("fac_item_id")) or normalize_ws(doc.get("text"))


def _is_disagreement_doc(
    doc: Dict[str, Any],
    *,
    prefilter_min: float,
    teacher_max: float,
) -> bool:
    return (
        float(doc.get("ce_prefilter_score", 0.0)) >= float(prefilter_min)
        and float(doc.get("teacher_score", 0.0)) <= float(teacher_max)
    )


def _pair_row_from_docs(
    row: Dict[str, Any],
    *,
    pos: Dict[str, Any],
    neg: Dict[str, Any],
    pair_type: str,
) -> Dict[str, Any]:
    pos_score = _clamp_01(pos.get("teacher_score"))
    neg_score = _clamp_01(neg.get("teacher_score"))
    return {
        "grant_id": clean_text(row.get("grant_id")),
        "spec_idx": int(row.get("spec_idx", 0) or 0),
        "aspect": clean_text(row.get("aspect")),
        "query_item_id": clean_text(row.get("query_item_id")),
        "query_text": clean_text(row.get("query_text")),
        "raw_query_text": clean_text(row.get("raw_query_text")),
        "pos_text": normalize_ws(pos.get("text")),
        "neg_text": normalize_ws(neg.get("text")),
        "teacher_pos_score": float(pos_score),
        "teacher_neg_score": float(neg_score),
        "teacher_margin": float(pos_score - neg_score),
        "pos_band": clean_text(pos.get("band")),
        "neg_band": clean_text(neg.get("band")),
        "pos_pair_id": clean_text(pos.get("pair_id")),
        "neg_pair_id": clean_text(neg.get("pair_id")),
        "pos_fac_item_id": clean_text(pos.get("fac_item_id")),
        "neg_fac_item_id": clean_text(neg.get("fac_item_id")),
        "pos_ce_prefilter_score": float(pos.get("ce_prefilter_score", 0.0)),
        "neg_ce_prefilter_score": float(neg.get("ce_prefilter_score", 0.0)),
        "pos_ce_prefilter_rank": int(pos.get("ce_prefilter_rank", 0) or 0),
        "neg_ce_prefilter_rank": int(neg.get("ce_prefilter_rank", 0) or 0),
        "pair_type": pair_type,
    }


def _append_pair(
    out: List[Dict[str, Any]],
    emitted: set[tuple[str, str, str]],
    row: Dict[str, Any],
    *,
    pos: Dict[str, Any],
    neg: Dict[str, Any],
    pair_type: str,
    min_margin: float,
) -> bool:
    pos_id = _doc_id(pos)
    neg_id = _doc_id(neg)
    if not pos_id or not neg_id or pos_id == neg_id:
        return False
    sig = (pair_type, pos_id, neg_id)
    if sig in emitted:
        return False
    margin = _clamp_01(pos.get("teacher_score")) - _clamp_01(neg.get("teacher_score"))
    if margin < float(min_margin):
        return False
    out.append(_pair_row_from_docs(row, pos=pos, neg=neg, pair_type=pair_type))
    emitted.add(sig)
    return True


def _build_all_ordered_pairwise_rows(
    listwise_rows: Sequence[Dict[str, Any]],
    *,
    min_margin: float,
    per_query_cap: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cap = max(0, int(per_query_cap))
    margin_floor = max(0.0, float(min_margin))
    for row in listwise_rows:
        docs = list(row.get("docs") or [])
        local: List[Dict[str, Any]] = []
        for i, pos in enumerate(docs):
            pos_score = _clamp_01(pos.get("teacher_score"))
            for neg in docs[i + 1 :]:
                neg_score = _clamp_01(neg.get("teacher_score"))
                margin = float(pos_score - neg_score)
                if margin < margin_floor:
                    continue
                local.append(_pair_row_from_docs(
                    row,
                    pos=pos,
                    neg=neg,
                    pair_type=_pair_type(clean_text(pos.get("band")), clean_text(neg.get("band"))),
                ))
        local.sort(
            key=lambda x: (
                -float(x.get("teacher_margin", 0.0)),
                clean_text(x.get("pos_fac_item_id")),
                clean_text(x.get("neg_fac_item_id")),
            )
        )
        if cap > 0:
            local = local[:cap]
        out.extend(local)
    return out


def _build_controlled_pairwise_rows(
    listwise_rows: Sequence[Dict[str, Any]],
    *,
    max_pairs_per_query: int,
    max_disagreement_per_query: int,
    max_boundary_per_query: int,
    weak_min_per_query: int,
    disagree_prefilter_min: float,
    disagree_teacher_max: float,
    disagree_min_margin: float,
    boundary_min_margin: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    max_pairs = max(1, int(max_pairs_per_query))
    disagree_cap_default = max(0, int(max_disagreement_per_query))
    boundary_cap_default = max(0, int(max_boundary_per_query))
    weak_min = max(0, int(weak_min_per_query))

    for row in listwise_rows:
        docs = list(row.get("docs") or [])
        high = sorted(
            [d for d in docs if clean_text(d.get("band")) == "high"],
            key=lambda x: float(x.get("teacher_score", 0.0)),
            reverse=True,
        )
        mid = sorted(
            [d for d in docs if clean_text(d.get("band")) == "mid"],
            key=lambda x: float(x.get("teacher_score", 0.0)),
            reverse=True,
        )
        low = sorted(
            [d for d in docs if clean_text(d.get("band")) == "low"],
            key=lambda x: float(x.get("teacher_score", 0.0)),
        )
        if not high:
            continue

        local: List[Dict[str, Any]] = []
        emitted: set[tuple[str, str, str]] = set()
        weak_target = min(weak_min, max_pairs, len(high) * len(low))
        disagree_cap = min(disagree_cap_default, max(0, max_pairs - weak_target))
        boundary_cap = min(boundary_cap_default, max(0, max_pairs - weak_target - disagree_cap))

        disagreement_negatives = [
            d for d in list(low) + list(mid)
            if _is_disagreement_doc(
                d,
                prefilter_min=disagree_prefilter_min,
                teacher_max=disagree_teacher_max,
            )
        ]
        disagreement_negatives.sort(
            key=lambda x: (
                float(x.get("teacher_score", 0.0)),
                -float(x.get("ce_prefilter_score", 0.0)),
                int(x.get("ce_prefilter_rank", 0) or 0),
            )
        )

        disagreement_added = 0
        for pos in high:
            for neg in disagreement_negatives:
                if len(local) >= max_pairs or disagreement_added >= disagree_cap:
                    break
                if _append_pair(
                    local,
                    emitted,
                    row,
                    pos=pos,
                    neg=neg,
                    pair_type="llm_disagreement",
                    min_margin=disagree_min_margin,
                ):
                    disagreement_added += 1
            if len(local) >= max_pairs or disagreement_added >= disagree_cap:
                break

        boundary_added = 0
        for pos in high:
            for neg in mid:
                if len(local) >= max_pairs or boundary_added >= boundary_cap:
                    break
                if _append_pair(
                    local,
                    emitted,
                    row,
                    pos=pos,
                    neg=neg,
                    pair_type="strong_vs_boundary",
                    min_margin=boundary_min_margin,
                ):
                    boundary_added += 1
            if len(local) >= max_pairs or boundary_added >= boundary_cap:
                break

        weak_added = 0
        for pos in high:
            for neg in low:
                if len(local) >= max_pairs or weak_added >= weak_target:
                    break
                if _append_pair(
                    local,
                    emitted,
                    row,
                    pos=pos,
                    neg=neg,
                    pair_type="strong_vs_weak",
                    min_margin=0.0,
                ):
                    weak_added += 1
            if len(local) >= max_pairs or weak_added >= weak_target:
                break

        hard_pool = sorted(
            list(mid) + list(low),
            key=lambda x: (
                -float(x.get("teacher_score", 0.0)),
                int(x.get("ce_prefilter_rank", 0) or 0),
                clean_text(x.get("fac_item_id")),
            ),
        )
        for pos in high:
            for neg in hard_pool:
                if len(local) >= max_pairs:
                    break
                _append_pair(
                    local,
                    emitted,
                    row,
                    pos=pos,
                    neg=neg,
                    pair_type="strong_vs_hard",
                    min_margin=0.0,
                )
            if len(local) >= max_pairs:
                break

        out.extend(local)
    return out


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _split_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(clean_text(row.get("aspect")) for row in rows))


def split_dataset(
    *,
    distillation_input: Path,
    output_dir: Path,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    prefix_mode: str,
    pair_generation_mode: str,
    pair_min_margin: float,
    pair_per_query_cap: int,
    pair_max_disagreement_per_query: int,
    pair_max_boundary_per_query: int,
    pair_weak_min_per_query: int,
    pair_disagree_prefilter_min: float,
    pair_disagree_teacher_max: float,
    pair_disagree_min_margin: float,
    pair_boundary_min_margin: float,
) -> Dict[str, Any]:
    rows = _dedupe_rows(_iter_distillation_rows(distillation_input, ASPECTS))
    if not rows:
        raise RuntimeError(f"No usable distillation rows found in {distillation_input}")

    split_keys = [_root_grant_key(row) for row in rows]
    split_by_key = _assign_split_names(split_keys, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
    split_rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split = split_by_key.get(_root_grant_key(row), "train")
        split_rows[split].append(row)

    manifest: Dict[str, Any] = {
        "distillation_input": str(distillation_input),
        "output_dir": str(output_dir),
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "prefix_mode": clean_text(prefix_mode).lower() or "bracket",
        "pair_generation_mode": clean_text(pair_generation_mode).lower() or "controlled",
        "pair_min_margin": float(pair_min_margin),
        "pair_per_query_cap": int(pair_per_query_cap),
        "pair_max_disagreement_per_query": int(pair_max_disagreement_per_query),
        "pair_max_boundary_per_query": int(pair_max_boundary_per_query),
        "pair_weak_min_per_query": int(pair_weak_min_per_query),
        "pair_disagree_prefilter_min": float(pair_disagree_prefilter_min),
        "pair_disagree_teacher_max": float(pair_disagree_teacher_max),
        "pair_disagree_min_margin": float(pair_disagree_min_margin),
        "pair_boundary_min_margin": float(pair_boundary_min_margin),
        "aspects": list(ASPECTS),
        "input_rows": int(len(rows)),
        "input_aspect_counts": _split_counts(rows),
        "split_key_count": int(len(split_by_key)),
        "split_key_counts": dict(Counter(split_by_key.values())),
        "splits": {},
    }

    aggregate_listwise: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    aggregate_pairwise: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for split in ("train", "val", "test"):
        manifest["splits"][split] = {
            "distill_rows": int(len(split_rows[split])),
            "distill_aspect_counts": _split_counts(split_rows[split]),
            "aspects": {},
        }
        for aspect in ASPECTS:
            aspect_rows = [row for row in split_rows[split] if clean_text(row.get("aspect")) == aspect]
            listwise = _build_listwise_rows(aspect_rows, aspect=aspect, prefix_mode=prefix_mode)
            if clean_text(pair_generation_mode).lower() == "all":
                pairwise = _build_all_ordered_pairwise_rows(
                    listwise,
                    min_margin=pair_min_margin,
                    per_query_cap=pair_per_query_cap,
                )
            else:
                pairwise = _build_controlled_pairwise_rows(
                    listwise,
                    max_pairs_per_query=(
                        int(pair_per_query_cap)
                        if int(pair_per_query_cap) > 0
                        else PAIR_MAX_PER_QUERY_DEFAULT
                    ),
                    max_disagreement_per_query=pair_max_disagreement_per_query,
                    max_boundary_per_query=pair_max_boundary_per_query,
                    weak_min_per_query=pair_weak_min_per_query,
                    disagree_prefilter_min=pair_disagree_prefilter_min,
                    disagree_teacher_max=pair_disagree_teacher_max,
                    disagree_min_margin=pair_disagree_min_margin,
                    boundary_min_margin=pair_boundary_min_margin,
                )

            listwise_path = output_dir / f"llm_distill_{aspect}_listwise_{split}.jsonl"
            pairwise_path = output_dir / f"llm_distill_{aspect}_pairwise_{split}.jsonl"
            _write_jsonl(listwise_path, listwise)
            _write_jsonl(pairwise_path, pairwise)

            aggregate_listwise[split].extend(listwise)
            aggregate_pairwise[split].extend(pairwise)
            manifest["splits"][split]["aspects"][aspect] = {
                "listwise_path": str(listwise_path),
                "pairwise_path": str(pairwise_path),
                "listwise_rows": int(len(listwise)),
                "pairwise_rows": int(len(pairwise)),
                "pairwise_type_counts": dict(Counter(clean_text(row.get("pair_type")) for row in pairwise)),
                "distill_rows": int(len(aspect_rows)),
            }

        all_listwise_path = output_dir / f"llm_distill_all_listwise_{split}.jsonl"
        all_pairwise_path = output_dir / f"llm_distill_all_pairwise_{split}.jsonl"
        aggregate_listwise[split].sort(key=lambda x: (clean_text(x.get("aspect")), clean_text(x.get("query_item_id"))))
        aggregate_pairwise[split].sort(
            key=lambda x: (
                clean_text(x.get("aspect")),
                clean_text(x.get("query_item_id")),
                -float(x.get("teacher_margin", 0.0)),
            )
        )
        _write_jsonl(all_listwise_path, aggregate_listwise[split])
        _write_jsonl(all_pairwise_path, aggregate_pairwise[split])
        manifest["splits"][split]["all_listwise_path"] = str(all_listwise_path)
        manifest["splits"][split]["all_pairwise_path"] = str(all_pairwise_path)
        manifest["splits"][split]["all_listwise_rows"] = int(len(aggregate_listwise[split]))
        manifest["splits"][split]["all_pairwise_rows"] = int(len(aggregate_pairwise[split]))
        manifest["splits"][split]["all_pairwise_type_counts"] = dict(
            Counter(clean_text(row.get("pair_type")) for row in aggregate_pairwise[split])
        )

    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split CE3 distillation rows into train/val/test listwise and pairwise datasets.")
    p.add_argument("--distillation-input", type=str, default=DISTILLATION_INPUT_DEFAULT)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--prefix-mode", choices=("bracket", "none"), default="bracket")
    p.add_argument("--pair-generation-mode", choices=("controlled", "all"), default="controlled")
    p.add_argument("--pair-min-margin", type=float, default=0.01)
    p.add_argument(
        "--pair-per-query-cap",
        type=int,
        default=0,
        help="For controlled mode, 0 uses the default max. For all mode, 0 keeps all valid ordered pairs.",
    )
    p.add_argument("--pair-max-disagreement-per-query", type=int, default=PAIR_MAX_DISAGREEMENT_PER_QUERY_DEFAULT)
    p.add_argument("--pair-max-boundary-per-query", type=int, default=PAIR_MAX_BOUNDARY_PER_QUERY_DEFAULT)
    p.add_argument("--pair-weak-min-per-query", type=int, default=PAIR_WEAK_MIN_PER_QUERY_DEFAULT)
    p.add_argument("--pair-disagree-prefilter-min", type=float, default=PAIR_DISAGREE_PREFILTER_MIN_DEFAULT)
    p.add_argument("--pair-disagree-teacher-max", type=float, default=PAIR_DISAGREE_TEACHER_MAX_DEFAULT)
    p.add_argument("--pair-disagree-min-margin", type=float, default=PAIR_DISAGREE_MIN_MARGIN_DEFAULT)
    p.add_argument("--pair-boundary-min-margin", type=float, default=PAIR_BOUNDARY_MIN_MARGIN_DEFAULT)
    return p.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    distillation_input = resolve_path(args.distillation_input)
    output_dir = resolve_path(args.output_dir)
    if not distillation_input.exists():
        raise FileNotFoundError(f"Missing distillation input: {distillation_input}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = split_dataset(
        distillation_input=distillation_input,
        output_dir=output_dir,
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        prefix_mode=args.prefix_mode,
        pair_generation_mode=args.pair_generation_mode,
        pair_min_margin=float(args.pair_min_margin),
        pair_per_query_cap=int(args.pair_per_query_cap),
        pair_max_disagreement_per_query=int(args.pair_max_disagreement_per_query),
        pair_max_boundary_per_query=int(args.pair_max_boundary_per_query),
        pair_weak_min_per_query=int(args.pair_weak_min_per_query),
        pair_disagree_prefilter_min=float(args.pair_disagree_prefilter_min),
        pair_disagree_teacher_max=float(args.pair_disagree_teacher_max),
        pair_disagree_min_margin=float(args.pair_disagree_min_margin),
        pair_boundary_min_margin=float(args.pair_boundary_min_margin),
    )
    print(json.dumps({"stage": "ce3_split_dataset", **manifest}, ensure_ascii=False))
    print(f"elapsed_sec={time.time() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
