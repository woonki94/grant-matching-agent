from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from sqlalchemy import bindparam, text
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "cross_encoder").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.db_conn import SessionLocal


FINETUNED_MODEL_DEFAULT = "cross_encoder/spec_to_spec/models/spec_to_spec_finetuned_ce"
PURE_BGE_MODEL_ID = "BAAI/bge-reranker-base"
OUTPUT_DIR_DEFAULT = "cross_encoder/spec_to_spec/eval/results"
DISTILL_INPUT_DEFAULT = "cross_encoder/spec_to_spec/dataset/llm_distill_raw_scores.jsonl"
DISTILL_INPUT_FALLBACK = "cross_encoder/spec_to_spec/dataset/llm_distill_listwise.jsonl"
DISTILL_TEST_INPUT_DEFAULT = "cross_encoder/spec_to_spec/dataset/splits/llm_distill_raw_test.jsonl"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_ws(value: Any) -> str:
    return " ".join(_clean_text(value).split())


def _safe_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _safe_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _resolve_path(value: Any) -> Path:
    p = Path(_clean_text(value)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _resolve_model_ref(value: str) -> str:
    raw = _clean_text(value)
    if not raw:
        return raw
    p = Path(raw).expanduser()
    if p.exists():
        return str(p.resolve())
    p2 = _resolve_path(raw)
    if p2.exists():
        return str(p2)
    return raw


def _shorten(value: Any, max_chars: int) -> str:
    text = _normalize_ws(value)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _assign_cluster_label(score: float, *, strong_min: float, mid_min: float) -> str:
    if score >= strong_min:
        return "strong"
    if score >= mid_min:
        return "mid"
    return "low"


def _score_docs_for_query(
    *,
    model: Any,
    tokenizer: Any,
    query_text: str,
    doc_texts: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> List[float]:
    out: List[float] = []
    step = max(1, int(batch_size))
    with torch.no_grad():
        for i in range(0, len(doc_texts), step):
            docs = list(doc_texts[i : i + step])
            queries = [query_text] * len(docs)
            enc = tokenizer(
                queries,
                docs,
                max_length=int(max_length),
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = _to_device(enc, device)
            logits = model(**enc).logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            out.extend(float(x) for x in probs.detach().cpu().tolist())
    return out


def _load_grants_for_faculty(
    *,
    faculty_id: int,
    domain_threshold: float,
    max_grants: int,
) -> List[Dict[str, Any]]:
    sql_body = """
        SELECT
            mr.grant_id AS grant_id,
            COALESCE(mr.domain_score, 0.0) AS domain_score,
            COALESCE(mr.llm_score, 0.0) AS llm_score
        FROM match_results mr
        WHERE mr.faculty_id = :faculty_id
          AND COALESCE(mr.domain_score, 0.0) >= :domain_threshold
        ORDER BY
            COALESCE(mr.domain_score, 0.0) DESC,
            COALESCE(mr.llm_score, 0.0) DESC,
            mr.grant_id ASC
    """
    params: Dict[str, Any] = {
        "faculty_id": int(faculty_id),
        "domain_threshold": float(domain_threshold),
    }
    if max_grants > 0:
        sql_body += "\nLIMIT :max_grants"
        params["max_grants"] = int(max_grants)

    with SessionLocal() as sess:
        rows = sess.execute(text(sql_body), params).mappings().all()

    out: List[Dict[str, Any]] = []
    for row in rows:
        grant_id = _clean_text(row.get("grant_id"))
        if not grant_id:
            continue
        out.append(
            {
                "grant_id": grant_id,
                "domain_score": float(row.get("domain_score") or 0.0),
                "llm_score": float(row.get("llm_score") or 0.0),
            }
        )
    return out


def _load_faculty_specs(
    *,
    faculty_id: int,
    embedding_model_id: str,
) -> List[Dict[str, Any]]:
    sql_body = """
        SELECT
            fse.id AS fac_spec_id,
            COALESCE(fse.section, '') AS section,
            COALESCE(fse.spec_text, '') AS spec_text,
            COALESCE(fse.spec_weight, 1.0) AS spec_weight
        FROM faculty_specialization_embedding fse
        WHERE fse.faculty_id = :faculty_id
          AND COALESCE(fse.spec_text, '') <> ''
    """
    params: Dict[str, Any] = {"faculty_id": int(faculty_id)}
    if _clean_text(embedding_model_id):
        sql_body += "\n  AND COALESCE(fse.model, '') = :embedding_model_id"
        params["embedding_model_id"] = _clean_text(embedding_model_id)
    sql_body += """
        ORDER BY
            CASE COALESCE(fse.section, '') WHEN 'research' THEN 0 WHEN 'application' THEN 1 ELSE 2 END ASC,
            COALESCE(fse.spec_weight, 1.0) DESC,
            fse.id ASC
    """

    with SessionLocal() as sess:
        rows = sess.execute(text(sql_body), params).mappings().all()

    out: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        spec_id = _safe_int(row.get("fac_spec_id"), default=0, minimum=0, maximum=9_223_372_036_854_775_807)
        spec_text = _normalize_ws(row.get("spec_text"))
        if spec_id <= 0 or not spec_text:
            continue
        key = spec_text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "fac_spec_id": int(spec_id),
                "section": _clean_text(row.get("section")) or "unknown",
                "text": spec_text,
                "weight": float(row.get("spec_weight") or 1.0),
            }
        )
    return out


def _load_grant_specs(
    *,
    grant_ids: Sequence[str],
    embedding_model_id: str,
) -> List[Dict[str, Any]]:
    if not grant_ids:
        return []

    sql_body = """
        SELECT
            ose.opportunity_id AS grant_id,
            ose.id AS grant_spec_id,
            COALESCE(ose.section, '') AS section,
            COALESCE(ose.spec_text, '') AS spec_text,
            COALESCE(ose.spec_weight, 1.0) AS spec_weight
        FROM opportunity_specialization_embedding ose
        WHERE ose.opportunity_id IN :grant_ids
          AND COALESCE(ose.spec_text, '') <> ''
    """
    params: Dict[str, Any] = {"grant_ids": list(grant_ids)}
    if _clean_text(embedding_model_id):
        sql_body += "\n  AND COALESCE(ose.model, '') = :embedding_model_id"
        params["embedding_model_id"] = _clean_text(embedding_model_id)
    sql_body += """
        ORDER BY
            ose.opportunity_id ASC,
            CASE COALESCE(ose.section, '') WHEN 'research' THEN 0 WHEN 'application' THEN 1 ELSE 2 END ASC,
            COALESCE(ose.spec_weight, 1.0) DESC,
            ose.id ASC
    """
    sql = text(sql_body).bindparams(bindparam("grant_ids", expanding=True))

    with SessionLocal() as sess:
        rows = sess.execute(sql, params).mappings().all()

    out: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        grant_id = _clean_text(row.get("grant_id"))
        spec_id = _safe_int(row.get("grant_spec_id"), default=0, minimum=0, maximum=9_223_372_036_854_775_807)
        spec_text = _normalize_ws(row.get("spec_text"))
        if not grant_id or spec_id <= 0 or not spec_text:
            continue
        key = (grant_id, spec_text.casefold())
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "grant_id": grant_id,
                "grant_spec_id": int(spec_id),
                "section": _clean_text(row.get("section")) or "unknown",
                "text": spec_text,
                "weight": float(row.get("spec_weight") or 1.0),
            }
        )
    return out


def _format_table(
    *,
    rows: Sequence[Dict[str, Any]],
    title: str,
    text_col: str,
    score1_col: str,
    score2_col: str = "",
    score3_col: str = "",
    score3_label: str = "",
    width_text: int = 92,
    truncate_text: bool = True,
) -> str:
    out: List[str] = []
    out.append("")
    out.append(f"=== {title} ===")
    if score2_col:
        if score3_col:
            label3 = _clean_text(score3_label) or "SCORE3"
            out.append(
                f"{'GRANT KEY':<28} {'GRANT SPEC':<{width_text}} {'FINETUNED':>10} {'BASE':>10} {label3:>10}"
            )
            out.append("-" * (28 + 1 + width_text + 1 + 10 + 1 + 10 + 1 + 10))
        else:
            out.append(f"{'GRANT KEY':<28} {'GRANT SPEC':<{width_text}} {'FINETUNED':>10} {'BASE':>10}")
            out.append("-" * (28 + 1 + width_text + 1 + 10 + 1 + 10))
        for row in rows:
            text_value = _clean_text(row.get(text_col))
            rendered_text = _shorten(text_value, width_text) if truncate_text else text_value
            if score3_col:
                out.append(
                    f"{_clean_text(row.get('grant_key')):<28} "
                    f"{rendered_text:<{width_text}} "
                    f"{float(row.get(score1_col) or 0.0):>10.4f} "
                    f"{float(row.get(score2_col) or 0.0):>10.4f} "
                    f"{float(row.get(score3_col) or 0.0):>10.4f}"
                )
            else:
                out.append(
                    f"{_clean_text(row.get('grant_key')):<28} "
                    f"{rendered_text:<{width_text}} "
                    f"{float(row.get(score1_col) or 0.0):>10.4f} "
                    f"{float(row.get(score2_col) or 0.0):>10.4f}"
                )
    else:
        out.append(f"{'GRANT KEY':<28} {'GRANT SPEC':<{width_text}} {'SCORE':>10}")
        out.append("-" * (28 + 1 + width_text + 1 + 10))
        for row in rows:
            text_value = _clean_text(row.get(text_col))
            rendered_text = _shorten(text_value, width_text) if truncate_text else text_value
            out.append(
                f"{_clean_text(row.get('grant_key')):<28} "
                f"{rendered_text:<{width_text}} "
                f"{float(row.get(score1_col) or 0.0):>10.4f}"
            )
    return "\n".join(out)


def _iter_distill_pairs_from_obj(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    grant_id = _clean_text(obj.get("grant_id"))
    spec_idx = _safe_int(obj.get("spec_idx"), default=0, minimum=0, maximum=1_000_000_000)
    query_text = _clean_text(obj.get("query_text") or obj.get("spec_text"))
    if not query_text:
        return out

    docs = list(obj.get("docs") or [])
    if docs:
        for d in docs:
            doc_text = _clean_text(d.get("text") or d.get("fac_spec_text"))
            if not doc_text:
                continue
            teacher_norm = _safe_float(
                d.get("teacher_score") if d.get("teacher_score") is not None else d.get("score"),
                default=0.0,
                minimum=0.0,
                maximum=1.0,
            )
            teacher_raw = _safe_float(
                d.get("teacher_score_raw") if d.get("teacher_score_raw") is not None else teacher_norm,
                default=teacher_norm,
                minimum=0.0,
                maximum=1.0,
            )
            out.append(
                {
                    "grant_id": grant_id,
                    "spec_idx": int(spec_idx),
                    "query_text": query_text,
                    "doc_text": doc_text,
                    "fac_id": _safe_int(d.get("fac_id"), default=0, minimum=0, maximum=2_147_483_647),
                    "fac_spec_id": _safe_int(d.get("fac_spec_id"), default=0, minimum=0, maximum=9_223_372_036_854_775_807),
                    "teacher_score": float(teacher_norm),
                    "teacher_score_raw": float(teacher_raw),
                    "source_band": _clean_text(d.get("band")) or "unknown",
                }
            )
        return out

    candidates = list(obj.get("candidates") or [])
    for c in candidates:
        doc_text = _clean_text(c.get("fac_spec_text") or c.get("text"))
        if not doc_text:
            continue
        teacher_norm = _safe_float(
            c.get("score") if c.get("score") is not None else c.get("teacher_score"),
            default=0.0,
            minimum=0.0,
            maximum=1.0,
        )
        teacher_raw = _safe_float(
            c.get("score_raw") if c.get("score_raw") is not None else teacher_norm,
            default=teacher_norm,
            minimum=0.0,
            maximum=1.0,
        )
        out.append(
            {
                "grant_id": grant_id,
                "spec_idx": int(spec_idx),
                "query_text": query_text,
                "doc_text": doc_text,
                "fac_id": _safe_int(c.get("fac_id"), default=0, minimum=0, maximum=2_147_483_647),
                "fac_spec_id": _safe_int(c.get("fac_spec_id"), default=0, minimum=0, maximum=9_223_372_036_854_775_807),
                "teacher_score": float(teacher_norm),
                "teacher_score_raw": float(teacher_raw),
                "source_band": _clean_text(c.get("band")) or "unknown",
            }
        )
    return out


def _score_band_from_teacher(score: float, *, high_threshold: float, mid_threshold: float) -> str:
    if float(score) >= float(high_threshold):
        return "high"
    if float(score) >= float(mid_threshold):
        return "mid"
    return "low"


def _reservoir_sample_distill_pairs(
    *,
    distill_input: Path,
    sample_high: int,
    sample_mid: int,
    sample_low: int,
    high_threshold: float,
    mid_threshold: float,
    ground_truth_mode: str,
    seed: int,
    max_pairs_scan: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mode = _clean_text(ground_truth_mode).lower()
    if mode not in {"normalized", "raw"}:
        mode = "normalized"

    rng = random.Random(int(seed))
    k_map = {
        "high": max(0, int(sample_high)),
        "mid": max(0, int(sample_mid)),
        "low": max(0, int(sample_low)),
    }
    keep_all_map = {band: (int(k_map[band]) == 0) for band in ("high", "mid", "low")}
    reservoirs: Dict[str, List[Dict[str, Any]]] = {"high": [], "mid": [], "low": []}
    seen_by_band = {"high": 0, "mid": 0, "low": 0}
    total_by_band = {"high": 0, "mid": 0, "low": 0}
    parse_errors = 0
    total_scanned_pairs = 0
    total_lines = 0

    def _maybe_add(item: Dict[str, Any], band: str) -> None:
        seen_by_band[band] += 1
        if bool(keep_all_map.get(band)):
            reservoirs[band].append(dict(item))
            return
        k = int(k_map[band])
        if k <= 0:
            return
        bucket = reservoirs[band]
        if len(bucket) < k:
            bucket.append(dict(item))
            return
        pick = rng.randint(1, seen_by_band[band])
        if pick <= k:
            bucket[pick - 1] = dict(item)

    with distill_input.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                parse_errors += 1
                continue
            pairs = _iter_distill_pairs_from_obj(obj)
            for item in pairs:
                total_scanned_pairs += 1
                if max_pairs_scan > 0 and total_scanned_pairs > max_pairs_scan:
                    break
                teacher_val = float(item["teacher_score_raw"] if mode == "raw" else item["teacher_score"])
                band = _score_band_from_teacher(
                    teacher_val,
                    high_threshold=float(high_threshold),
                    mid_threshold=float(mid_threshold),
                )
                total_by_band[band] += 1
                copied = dict(item)
                copied["teacher_score_used"] = float(teacher_val)
                copied["score_band"] = band
                _maybe_add(copied, band)
            if max_pairs_scan > 0 and total_scanned_pairs > max_pairs_scan:
                break

    selected: List[Dict[str, Any]] = []
    for band in ("high", "mid", "low"):
        selected.extend(reservoirs[band])

    meta = {
        "ground_truth_mode": mode,
        "k_map": k_map,
        "selected_by_band": {k: len(v) for k, v in reservoirs.items()},
        "available_by_band": total_by_band,
        "parse_errors": int(parse_errors),
        "total_scanned_pairs": int(total_scanned_pairs),
        "total_lines": int(total_lines),
    }
    return selected, meta


def _score_query_doc_rows(
    *,
    model: Any,
    tokenizer: Any,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> List[float]:
    if not rows:
        return []

    query_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        query_to_indices[_clean_text(row.get("query_text"))].append(idx)

    out = [0.0] * len(rows)
    for query_text, idxs in query_to_indices.items():
        docs = [_clean_text(rows[i].get("doc_text")) for i in idxs]
        scores = _score_docs_for_query(
            model=model,
            tokenizer=tokenizer,
            query_text=query_text,
            doc_texts=docs,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        for row_idx, score in zip(idxs, scores):
            out[row_idx] = float(score)
    return out


def _compute_margin_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, Any]]] = {"overall": list(rows), "high": [], "mid": [], "low": []}
    for row in rows:
        band = _clean_text(row.get("score_band")).lower()
        if band in {"high", "mid", "low"}:
            groups[band].append(row)

    def _avg(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / float(len(values)))

    out: Dict[str, Dict[str, float]] = {}
    for band_name, band_rows in groups.items():
        out[band_name] = {
            "count": float(len(band_rows)),
            "avg_teacher_score": _avg([float(r.get("teacher_score_used") or 0.0) for r in band_rows]),
            "avg_base_score": _avg([float(r.get("base_score") or 0.0) for r in band_rows]),
            "avg_finetuned_score": _avg([float(r.get("finetuned_score") or 0.0) for r in band_rows]),
            "avg_base_margin": _avg([float(r.get("base_margin") or 0.0) for r in band_rows]),
            "avg_finetuned_margin": _avg([float(r.get("finetuned_margin") or 0.0) for r in band_rows]),
            "avg_base_abs_margin": _avg([float(r.get("base_abs_margin") or 0.0) for r in band_rows]),
            "avg_finetuned_abs_margin": _avg([float(r.get("finetuned_abs_margin") or 0.0) for r in band_rows]),
        }
        out[band_name]["avg_abs_margin_gain"] = float(
            out[band_name]["avg_base_abs_margin"] - out[band_name]["avg_finetuned_abs_margin"]
        )
    return out


def _group_rows_by_query(rows: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            _clean_text(row.get("grant_id")),
            _safe_int(row.get("spec_idx"), default=0, minimum=0, maximum=1_000_000_000),
        )
        grouped[key].append(row)
    return list(grouped.values())


def _compute_order_metrics_for_model(
    *,
    rows: Sequence[Dict[str, Any]],
    model_score_key: str,
    top_k: int,
    pair_eps: float,
    hard_gap_max: float,
    medium_gap_max: float,
    mrr_rel_threshold: float,
    recall_rel_threshold: float,
) -> Dict[str, Any]:
    query_groups = _group_rows_by_query(rows)
    top_k = max(1, int(top_k))
    eps = max(0.0, float(pair_eps))
    hard_gap = max(eps, float(hard_gap_max))
    med_gap = max(hard_gap, float(medium_gap_max))

    query_total = 0
    query_with_2plus = 0
    top1_correct = 0
    topk_overlap_sum = 0.0
    topk_overlap_count = 0
    mrr_sum = 0.0
    mrr_count = 0
    recall_sum = 0.0
    recall_count = 0
    ndcg_sum = 0.0
    ndcg_count = 0

    buckets = {
        "overall": {"pairs": 0, "correct": 0, "pred_margin_sum": 0.0},
        "hard": {"pairs": 0, "correct": 0, "pred_margin_sum": 0.0},
        "medium": {"pairs": 0, "correct": 0, "pred_margin_sum": 0.0},
        "easy": {"pairs": 0, "correct": 0, "pred_margin_sum": 0.0},
    }

    for group in query_groups:
        n = len(group)
        query_total += 1
        if n < 2:
            continue
        query_with_2plus += 1

        teacher_vals = [float(r.get("teacher_score_used") or 0.0) for r in group]
        model_vals = [float(r.get(model_score_key) or 0.0) for r in group]

        teacher_sorted = sorted(range(n), key=lambda i: teacher_vals[i], reverse=True)
        model_sorted = sorted(range(n), key=lambda i: model_vals[i], reverse=True)

        teacher_top_score = teacher_vals[teacher_sorted[0]]
        teacher_top_set = {i for i, v in enumerate(teacher_vals) if abs(v - teacher_top_score) <= eps}
        if model_sorted[0] in teacher_top_set:
            top1_correct += 1

        k_eff = min(top_k, n)
        teacher_topk = set(teacher_sorted[:k_eff])
        model_topk = set(model_sorted[:k_eff])
        topk_overlap_sum += float(len(teacher_topk.intersection(model_topk)) / float(max(1, k_eff)))
        topk_overlap_count += 1

        # MRR@k with teacher relevance threshold.
        rr = 0.0
        for rank, idx in enumerate(model_sorted[:k_eff], start=1):
            if teacher_vals[idx] >= float(mrr_rel_threshold):
                rr = 1.0 / float(rank)
                break
        mrr_sum += float(rr)
        mrr_count += 1

        # Recall@k with teacher relevance threshold.
        relevant = {i for i, y in enumerate(teacher_vals) if y >= float(recall_rel_threshold)}
        if relevant:
            pred_topk = set(model_sorted[:k_eff])
            recall_sum += float(len(pred_topk.intersection(relevant)) / float(len(relevant)))
            recall_count += 1

        # NDCG@k with graded teacher scores as relevance.
        def _dcg(sorted_indices: Sequence[int]) -> float:
            s = 0.0
            for rank, idx in enumerate(sorted_indices[:k_eff], start=1):
                rel = float(teacher_vals[idx])
                denom = math.log2(float(rank + 1.0))
                if denom <= 0.0:
                    continue
                s += float((2.0 ** rel - 1.0) / denom)
            return s

        dcg = _dcg(model_sorted)
        idcg = _dcg(teacher_sorted)
        if idcg > 0.0:
            ndcg_sum += float(dcg / idcg)
            ndcg_count += 1

        for i in range(n):
            yi = teacher_vals[i]
            si = model_vals[i]
            for j in range(i + 1, n):
                yj = teacher_vals[j]
                sj = model_vals[j]
                gap = abs(yi - yj)
                if gap <= eps:
                    continue
                if yi > yj:
                    pred_margin = float(si - sj)
                else:
                    pred_margin = float(sj - si)
                is_correct = pred_margin > 0.0

                bucket_names = ["overall"]
                if gap <= hard_gap:
                    bucket_names.append("hard")
                elif gap <= med_gap:
                    bucket_names.append("medium")
                else:
                    bucket_names.append("easy")

                for b in bucket_names:
                    buckets[b]["pairs"] += 1
                    buckets[b]["correct"] += 1 if is_correct else 0
                    buckets[b]["pred_margin_sum"] += float(pred_margin)

    def _to_metrics(obj: Dict[str, Any]) -> Dict[str, float]:
        p = int(obj.get("pairs") or 0)
        c = int(obj.get("correct") or 0)
        s = float(obj.get("pred_margin_sum") or 0.0)
        return {
            "pair_count": float(p),
            "pair_accuracy": float(c / float(max(1, p))),
            "mean_pred_margin": float(s / float(max(1, p))),
        }

    return {
        "query_count_total": int(query_total),
        "query_count_2plus_docs": int(query_with_2plus),
        "top1_accuracy": float(top1_correct / float(max(1, query_with_2plus))),
        "topk_overlap": float(topk_overlap_sum / float(max(1, topk_overlap_count))),
        "mrr_at_k": float(mrr_sum / float(max(1, mrr_count))),
        "ndcg_at_k": float(ndcg_sum / float(max(1, ndcg_count))),
        "recall_at_k": float(recall_sum / float(max(1, recall_count))),
        "pair_metrics": {k: _to_metrics(v) for k, v in buckets.items()},
        "config": {
            "top_k": int(top_k),
            "pair_eps": float(eps),
            "hard_gap_max": float(hard_gap),
            "medium_gap_max": float(med_gap),
            "mrr_rel_threshold": float(mrr_rel_threshold),
            "recall_rel_threshold": float(recall_rel_threshold),
        },
    }


def _compute_raw_score_sanity(
    *,
    rows: Sequence[Dict[str, Any]],
    model_score_key: str,
    high_threshold: float,
    mid_threshold: float,
) -> Dict[str, Any]:
    by_band: Dict[str, List[float]] = {"high": [], "mid": [], "low": []}
    for row in rows:
        band = _clean_text(row.get("score_band")).lower()
        if band in by_band:
            by_band[band].append(float(row.get(model_score_key) or 0.0))

    def _avg(vals: Sequence[float]) -> float:
        if not vals:
            return 0.0
        return float(sum(vals) / float(len(vals)))

    avg_high = _avg(by_band["high"])
    avg_mid = _avg(by_band["mid"])
    avg_low = _avg(by_band["low"])
    monotonic = bool(avg_high >= avg_mid >= avg_low)

    low_vals = by_band["low"]
    high_vals = by_band["high"]
    weak_leak = float(sum(1 for x in low_vals if x >= float(high_threshold)) / float(max(1, len(low_vals))))
    strong_collapse = float(sum(1 for x in high_vals if x < float(mid_threshold)) / float(max(1, len(high_vals))))

    return {
        "avg_pred_high": float(avg_high),
        "avg_pred_mid": float(avg_mid),
        "avg_pred_low": float(avg_low),
        "monotonic_high_mid_low": bool(monotonic),
        "weak_leak_rate": float(weak_leak),
        "strong_collapse_rate": float(strong_collapse),
        "count_high": int(len(by_band["high"])),
        "count_mid": int(len(by_band["mid"])),
        "count_low": int(len(by_band["low"])),
    }


def _format_margin_summary_table(stats: Dict[str, Dict[str, float]]) -> str:
    order = ["overall", "high", "mid", "low"]
    lines: List[str] = []
    lines.append("")
    lines.append("=== Margin Summary (vs Teacher Ground Truth) ===")
    lines.append(
        f"{'BAND':<8} {'COUNT':>8} {'AVG_GT':>10} {'BASE_MAE':>10} {'FINETUNED_MAE':>14} {'GAIN':>10}"
    )
    lines.append("-" * 68)
    for band in order:
        row = stats.get(band) or {}
        lines.append(
            f"{band:<8} "
            f"{int(row.get('count') or 0):>8} "
            f"{float(row.get('avg_teacher_score') or 0.0):>10.4f} "
            f"{float(row.get('avg_base_abs_margin') or 0.0):>10.4f} "
            f"{float(row.get('avg_finetuned_abs_margin') or 0.0):>14.4f} "
            f"{float(row.get('avg_abs_margin_gain') or 0.0):>10.4f}"
        )
    return "\n".join(lines)


def _format_order_summary_table(
    *,
    finetuned: Dict[str, Any],
    base: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append("=== Order Correctness Summary ===")
    lines.append(
        f"queries_total={int(finetuned.get('query_count_total') or 0)} "
        f"queries_with_2plus_docs={int(finetuned.get('query_count_2plus_docs') or 0)} "
        f"top_k={int((finetuned.get('config') or {}).get('top_k') or 0)}"
    )
    lines.append(
        f"{'MODEL':<12} {'TOP1_ACC':>9} {'OVLP@K':>8} {'MRR@K':>8} {'NDCG@K':>8} {'RECALL@K':>9} {'PAIR_ACC':>9} {'HARD_ACC':>9} {'MEAN_M':>9} {'HARD_M':>9}"
    )
    lines.append("-" * 112)

    def _row(label: str, obj: Dict[str, Any]) -> str:
        pm = obj.get("pair_metrics") or {}
        overall = pm.get("overall") or {}
        hard = pm.get("hard") or {}
        return (
            f"{label:<12} "
            f"{float(obj.get('top1_accuracy') or 0.0):>9.4f} "
            f"{float(obj.get('topk_overlap') or 0.0):>8.4f} "
            f"{float(obj.get('mrr_at_k') or 0.0):>8.4f} "
            f"{float(obj.get('ndcg_at_k') or 0.0):>8.4f} "
            f"{float(obj.get('recall_at_k') or 0.0):>9.4f} "
            f"{float(overall.get('pair_accuracy') or 0.0):>9.4f} "
            f"{float(hard.get('pair_accuracy') or 0.0):>9.4f} "
            f"{float(overall.get('mean_pred_margin') or 0.0):>9.4f} "
            f"{float(hard.get('mean_pred_margin') or 0.0):>9.4f}"
        )

    lines.append(_row("finetuned", finetuned))
    lines.append(_row("base", base))
    return "\n".join(lines)


def _format_raw_sanity_table(
    *,
    finetuned: Dict[str, Any],
    base: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append("=== Raw Score Sanity Summary ===")
    lines.append(
        f"{'MODEL':<12} {'AVG_HIGH':>10} {'AVG_MID':>10} {'AVG_LOW':>10} {'MONOTONIC':>10} {'WEAK_LEAK':>11} {'STRONG_COLL':>12}"
    )
    lines.append("-" * 86)

    def _row(label: str, obj: Dict[str, Any]) -> str:
        return (
            f"{label:<12} "
            f"{float(obj.get('avg_pred_high') or 0.0):>10.4f} "
            f"{float(obj.get('avg_pred_mid') or 0.0):>10.4f} "
            f"{float(obj.get('avg_pred_low') or 0.0):>10.4f} "
            f"{str(bool(obj.get('monotonic_high_mid_low'))):>10} "
            f"{float(obj.get('weak_leak_rate') or 0.0):>11.4f} "
            f"{float(obj.get('strong_collapse_rate') or 0.0):>12.4f}"
        )

    lines.append(_row("finetuned", finetuned))
    lines.append(_row("base", base))
    return "\n".join(lines)


def _format_distill_pairs_table(
    *,
    rows: Sequence[Dict[str, Any]],
    title: str,
    max_rows: int,
    text_width: int,
    truncate_text: bool,
) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append(f"=== {title} ===")
    lines.append(
        f"{'PAIR_KEY':<28} {'GT':>7} {'FINETUNED':>10} {'BASE':>10} {'FT_MAE':>8} {'BASE_MAE':>9} {'QUERY | DOC':<{text_width}}"
    )
    lines.append("-" * (28 + 1 + 7 + 1 + 10 + 1 + 10 + 1 + 8 + 1 + 9 + 1 + text_width))
    limited = list(rows[: max(0, int(max_rows))]) if max_rows > 0 else list(rows)
    for row in limited:
        pair_key = f"{_clean_text(row.get('grant_id'))}::spec#{_safe_int(row.get('spec_idx'), default=0, minimum=0, maximum=1_000_000_000)}"
        merged_text = f"Q: {_clean_text(row.get('query_text'))} || D: {_clean_text(row.get('doc_text'))}"
        rendered_text = _shorten(merged_text, text_width) if truncate_text else merged_text
        lines.append(
            f"{pair_key:<28} "
            f"{float(row.get('teacher_score_used') or 0.0):>7.4f} "
            f"{float(row.get('finetuned_score') or 0.0):>10.4f} "
            f"{float(row.get('base_score') or 0.0):>10.4f} "
            f"{float(row.get('finetuned_abs_margin') or 0.0):>8.4f} "
            f"{float(row.get('base_abs_margin') or 0.0):>9.4f} "
            f"{rendered_text:<{text_width}}"
        )
    return "\n".join(lines)


def _run_distill_eval(args: argparse.Namespace) -> int:
    batch_size = _safe_int(args.batch_size, default=32, minimum=1, maximum=4096)
    max_length = _safe_int(args.max_length, default=512, minimum=64, maximum=4096)
    sample_high = _safe_int(args.distill_sample_high, default=0, minimum=0, maximum=1_000_000)
    sample_mid = _safe_int(args.distill_sample_mid, default=0, minimum=0, maximum=1_000_000)
    sample_low = _safe_int(args.distill_sample_low, default=0, minimum=0, maximum=1_000_000)
    high_threshold = _safe_float(args.distill_high_threshold, default=0.67, minimum=0.0, maximum=1.0)
    mid_threshold = _safe_float(args.distill_mid_threshold, default=0.34, minimum=0.0, maximum=1.0)
    if mid_threshold > high_threshold:
        mid_threshold = high_threshold
    max_pairs_scan = _safe_int(args.distill_max_pairs_scan, default=0, minimum=0, maximum=50_000_000)
    seed = _safe_int(args.distill_seed, default=42, minimum=0, maximum=2_147_483_647)
    print_pair_tables = bool(args.distill_print_pair_tables)
    save_pair_tables = bool(args.distill_save_pair_tables)
    print_per_band = _safe_int(args.distill_print_rows_per_band, default=30, minimum=0, maximum=1_000_000)
    save_rows_per_band = _safe_int(args.distill_save_rows_per_band, default=0, minimum=0, maximum=5_000_000)
    order_top_k = _safe_int(args.distill_order_top_k, default=5, minimum=1, maximum=100)
    pair_eps = _safe_float(args.distill_pair_eps, default=0.01, minimum=0.0, maximum=1.0)
    hard_gap_max = _safe_float(args.distill_hard_gap_max, default=0.15, minimum=0.0, maximum=1.0)
    medium_gap_max = _safe_float(args.distill_medium_gap_max, default=0.40, minimum=0.0, maximum=1.0)
    if medium_gap_max < hard_gap_max:
        medium_gap_max = hard_gap_max
    ground_truth_mode = _clean_text(args.distill_ground_truth).lower() or "normalized"
    if ground_truth_mode not in {"normalized", "raw"}:
        ground_truth_mode = "normalized"

    finetuned_ref = _resolve_model_ref(_clean_text(args.finetuned_model) or FINETUNED_MODEL_DEFAULT)
    base_ref = _resolve_model_ref(_clean_text(args.base_model) or PURE_BGE_MODEL_ID)

    distill_input = _resolve_path(args.distill_test_input)
    if not distill_input.exists():
        distill_input = _resolve_path(args.distill_input)
    if not distill_input.exists():
        fallback = _resolve_path(DISTILL_INPUT_FALLBACK)
        if fallback.exists():
            distill_input = fallback
        else:
            raise RuntimeError(
                f"Distill input file not found: {distill_input}. "
                f"Also fallback not found: {fallback}"
            )

    sampled_rows, sample_meta = _reservoir_sample_distill_pairs(
        distill_input=distill_input,
        sample_high=sample_high,
        sample_mid=sample_mid,
        sample_low=sample_low,
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
        ground_truth_mode=ground_truth_mode,
        seed=seed,
        max_pairs_scan=max_pairs_scan,
    )
    if not sampled_rows:
        raise RuntimeError("No pairs were sampled from distill file. Check thresholds and input file.")

    device = _pick_device()
    tok_finetuned = AutoTokenizer.from_pretrained(finetuned_ref)
    model_finetuned = AutoModelForSequenceClassification.from_pretrained(finetuned_ref, num_labels=1).to(device).eval()
    tok_base = AutoTokenizer.from_pretrained(base_ref)
    model_base = AutoModelForSequenceClassification.from_pretrained(base_ref, num_labels=1).to(device).eval()

    finetuned_scores = _score_query_doc_rows(
        model=model_finetuned,
        tokenizer=tok_finetuned,
        rows=sampled_rows,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    base_scores = _score_query_doc_rows(
        model=model_base,
        tokenizer=tok_base,
        rows=sampled_rows,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )

    rows: List[Dict[str, Any]] = []
    for i, src in enumerate(sampled_rows):
        gt = float(src.get("teacher_score_used") or 0.0)
        ft = float(finetuned_scores[i])
        b = float(base_scores[i])
        row = dict(src)
        row["finetuned_score"] = ft
        row["base_score"] = b
        row["finetuned_margin"] = float(ft - gt)
        row["base_margin"] = float(b - gt)
        row["finetuned_abs_margin"] = abs(float(ft - gt))
        row["base_abs_margin"] = abs(float(b - gt))
        row["abs_margin_gain"] = float(row["base_abs_margin"] - row["finetuned_abs_margin"])
        rows.append(row)

    stats = _compute_margin_stats(rows)
    order_metrics_finetuned = _compute_order_metrics_for_model(
        rows=rows,
        model_score_key="finetuned_score",
        top_k=order_top_k,
        pair_eps=pair_eps,
        hard_gap_max=hard_gap_max,
        medium_gap_max=medium_gap_max,
        mrr_rel_threshold=high_threshold,
        recall_rel_threshold=mid_threshold,
    )
    order_metrics_base = _compute_order_metrics_for_model(
        rows=rows,
        model_score_key="base_score",
        top_k=order_top_k,
        pair_eps=pair_eps,
        hard_gap_max=hard_gap_max,
        medium_gap_max=medium_gap_max,
        mrr_rel_threshold=high_threshold,
        recall_rel_threshold=mid_threshold,
    )
    raw_sanity_finetuned = _compute_raw_score_sanity(
        rows=rows,
        model_score_key="finetuned_score",
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
    )
    raw_sanity_base = _compute_raw_score_sanity(
        rows=rows,
        model_score_key="base_score",
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
    )

    by_band: Dict[str, List[Dict[str, Any]]] = {"high": [], "mid": [], "low": []}
    for row in rows:
        band = _clean_text(row.get("score_band")).lower()
        if band in by_band:
            by_band[band].append(row)
    for band in by_band:
        by_band[band] = sorted(by_band[band], key=lambda x: float(x.get("teacher_score_used") or 0.0), reverse=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"distill_margin_compare_{ts}.json"
    output_txt = output_dir / f"distill_margin_compare_{ts}.txt"

    meta_header = (
        f"mode=distill\n"
        f"distill_input={distill_input}\n"
        f"ground_truth_mode={ground_truth_mode}\n"
        f"sample_high={sample_high} sample_mid={sample_mid} sample_low={sample_low}\n"
        f"high_threshold={high_threshold} mid_threshold={mid_threshold}\n"
        f"selected_high={len(by_band['high'])} selected_mid={len(by_band['mid'])} selected_low={len(by_band['low'])}\n"
        f"available_high={int(sample_meta.get('available_by_band', {}).get('high', 0))} "
        f"available_mid={int(sample_meta.get('available_by_band', {}).get('mid', 0))} "
        f"available_low={int(sample_meta.get('available_by_band', {}).get('low', 0))}\n"
        f"order_top_k={order_top_k} pair_eps={pair_eps} hard_gap_max={hard_gap_max} medium_gap_max={medium_gap_max}\n"
        f"finetuned_model={finetuned_ref}\n"
        f"base_model={base_ref}\n"
        f"device={device}\n"
    )

    blocks_print: List[str] = [
        meta_header,
        _format_order_summary_table(finetuned=order_metrics_finetuned, base=order_metrics_base),
        _format_margin_summary_table(stats),
        _format_raw_sanity_table(finetuned=raw_sanity_finetuned, base=raw_sanity_base),
    ]
    blocks_save: List[str] = [
        meta_header,
        _format_order_summary_table(finetuned=order_metrics_finetuned, base=order_metrics_base),
        _format_margin_summary_table(stats),
        _format_raw_sanity_table(finetuned=raw_sanity_finetuned, base=raw_sanity_base),
    ]

    if print_pair_tables:
        blocks_print.append(
            _format_distill_pairs_table(
                rows=by_band["high"],
                title="High Score Pairs",
                max_rows=print_per_band,
                text_width=140,
                truncate_text=True,
            )
        )
        blocks_print.append(
            _format_distill_pairs_table(
                rows=by_band["mid"],
                title="Mid Score Pairs",
                max_rows=print_per_band,
                text_width=140,
                truncate_text=True,
            )
        )
        blocks_print.append(
            _format_distill_pairs_table(
                rows=by_band["low"],
                title="Low Score Pairs",
                max_rows=print_per_band,
                text_width=140,
                truncate_text=True,
            )
        )

    if save_pair_tables:
        blocks_save.append(
            _format_distill_pairs_table(
                rows=by_band["high"],
                title="High Score Pairs",
                max_rows=save_rows_per_band,
                text_width=140,
                truncate_text=False,
            )
        )
        blocks_save.append(
            _format_distill_pairs_table(
                rows=by_band["mid"],
                title="Mid Score Pairs",
                max_rows=save_rows_per_band,
                text_width=140,
                truncate_text=False,
            )
        )
        blocks_save.append(
            _format_distill_pairs_table(
                rows=by_band["low"],
                title="Low Score Pairs",
                max_rows=save_rows_per_band,
                text_width=140,
                truncate_text=False,
            )
        )
    report_text_print = "\n".join(blocks_print).strip() + "\n"
    report_text_save = "\n".join(blocks_save).strip() + "\n"

    if bool(args.print):
        print(report_text_print)
    if bool(args.save):
        payload = {
            "meta": {
                "created_at_local": datetime.now().isoformat(),
                "mode": "distill",
                "distill_input": str(distill_input),
                "ground_truth_mode": ground_truth_mode,
                "sample_meta": sample_meta,
                "sample_high": int(sample_high),
                "sample_mid": int(sample_mid),
                "sample_low": int(sample_low),
                "distill_print_pair_tables": bool(print_pair_tables),
                "distill_save_pair_tables": bool(save_pair_tables),
                "distill_print_rows_per_band": int(print_per_band),
                "distill_save_rows_per_band": int(save_rows_per_band),
                "high_threshold": float(high_threshold),
                "mid_threshold": float(mid_threshold),
                "order_top_k": int(order_top_k),
                "pair_eps": float(pair_eps),
                "hard_gap_max": float(hard_gap_max),
                "medium_gap_max": float(medium_gap_max),
                "finetuned_model": finetuned_ref,
                "base_model": base_ref,
                "device": str(device),
                "batch_size": int(batch_size),
                "max_length": int(max_length),
                "output_txt": str(output_txt),
            },
            "order_metrics": {
                "finetuned": order_metrics_finetuned,
                "base": order_metrics_base,
            },
            "margin_stats": stats,
            "raw_score_sanity": {
                "finetuned": raw_sanity_finetuned,
                "base": raw_sanity_base,
            },
            "rows": rows,
        }
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        output_txt.write_text(report_text_save, encoding="utf-8")
        print(f"saved_json={output_json}")
        print(f"saved_txt={output_txt}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Spec-to-spec evaluation. Supports DB mode (faculty->grant specs) and distill mode "
            "(LLM-distilled pairs with teacher-score margin analysis)."
        )
    )
    p.add_argument("--eval-source", type=str, default="distill", choices=["db", "distill"], help="Evaluation source mode.")
    p.add_argument("--faculty-id", type=int, default=0, help="Target faculty_id (required in db mode).")
    p.add_argument("--domain-threshold", type=float, default=0.3, help="match_results.domain_score threshold.")
    p.add_argument("--max-grants", type=int, default=0, help="Optional cap on grants selected (0 = all over threshold).")
    p.add_argument("--embedding-model-id", type=str, default="", help="Optional model filter for specialization tables.")

    p.add_argument("--finetuned-model", type=str, default=FINETUNED_MODEL_DEFAULT)
    p.add_argument("--base-model", type=str, default=PURE_BGE_MODEL_ID)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)

    p.add_argument("--max-fac-specs", type=int, default=0, help="Optional cap faculty specs (0 = all).")
    p.add_argument("--max-grant-specs", type=int, default=0, help="Optional cap grant specs total (0 = all).")
    p.add_argument("--cluster-strong-min", type=float, default=0.67, help="Finetuned score threshold for strong cluster.")
    p.add_argument("--cluster-mid-min", type=float, default=0.34, help="Finetuned score threshold for mid cluster.")
    p.add_argument("--margin-top-k", type=int, default=10, help="Top-k rows by absolute |finetuned-base| margin.")
    p.add_argument("--distill-test-input", type=str, default=DISTILL_TEST_INPUT_DEFAULT, help="Preferred test JSONL for distill evaluation.")
    p.add_argument("--distill-input", type=str, default=DISTILL_INPUT_DEFAULT, help="Fallback distilled JSONL input (raw or listwise).")
    p.add_argument(
        "--distill-ground-truth",
        type=str,
        default="raw",
        choices=["normalized", "raw"],
        help="Teacher score field used for margin computation.",
    )
    p.add_argument("--distill-high-threshold", type=float, default=0.67, help="High band threshold for teacher score.")
    p.add_argument("--distill-mid-threshold", type=float, default=0.34, help="Mid band lower threshold for teacher score.")
    p.add_argument(
        "--distill-sample-high",
        type=int,
        default=0,
        help="Reservoir sample size from high band (0 = keep all available high-band pairs).",
    )
    p.add_argument(
        "--distill-sample-mid",
        type=int,
        default=0,
        help="Reservoir sample size from mid band (0 = keep all available mid-band pairs).",
    )
    p.add_argument(
        "--distill-sample-low",
        type=int,
        default=0,
        help="Reservoir sample size from low band (0 = keep all available low-band pairs).",
    )
    p.add_argument("--distill-seed", type=int, default=42, help="Sampling seed for distill pair sampling.")
    p.add_argument(
        "--distill-max-pairs-scan",
        type=int,
        default=0,
        help="Optional cap on scanned query-doc pairs from distill file (0 = scan all).",
    )
    p.add_argument(
        "--distill-print-rows-per-band",
        type=int,
        default=30,
        help="Rows printed per high/mid/low band in distill mode when --distill-print-pair-tables is enabled (0 = all).",
    )
    p.add_argument(
        "--distill-save-rows-per-band",
        type=int,
        default=0,
        help="Rows saved per high/mid/low band in distill mode (0 = all).",
    )
    p.add_argument(
        "--distill-print-pair-tables",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print full high/mid/low pair tables to console (default: false; summary only).",
    )
    p.add_argument(
        "--distill-save-pair-tables",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include high/mid/low pair tables in saved txt output (default: true).",
    )
    p.add_argument("--distill-order-top-k", type=int, default=5, help="Top-k overlap metric cutoff for order correctness.")
    p.add_argument(
        "--distill-pair-eps",
        type=float,
        default=0.01,
        help="Minimum teacher score gap to count a pair in pairwise order correctness.",
    )
    p.add_argument(
        "--distill-hard-gap-max",
        type=float,
        default=0.15,
        help="Teacher-gap upper bound for hard/boundary pair bucket.",
    )
    p.add_argument(
        "--distill-medium-gap-max",
        type=float,
        default=0.40,
        help="Teacher-gap upper bound for medium pair bucket (above this = easy bucket).",
    )

    p.add_argument("--save", action=argparse.BooleanOptionalAction, default=True, help="Save results to files.")
    p.add_argument("--print", action=argparse.BooleanOptionalAction, default=True, help="Print tables to console.")
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    eval_source = _clean_text(args.eval_source).lower() or "db"
    if eval_source == "distill":
        return _run_distill_eval(args)

    if int(args.faculty_id or 0) <= 0:
        raise RuntimeError("--faculty-id is required and must be > 0 in db mode.")
    faculty_id = _safe_int(args.faculty_id, default=1, minimum=1, maximum=2_147_483_647)
    domain_threshold = _safe_float(args.domain_threshold, default=0.3, minimum=0.0, maximum=1.0)
    max_grants = _safe_int(args.max_grants, default=0, minimum=0, maximum=1_000_000)
    max_fac_specs = _safe_int(args.max_fac_specs, default=0, minimum=0, maximum=1_000_000)
    max_grant_specs = _safe_int(args.max_grant_specs, default=0, minimum=0, maximum=5_000_000)
    batch_size = _safe_int(args.batch_size, default=32, minimum=1, maximum=4096)
    max_length = _safe_int(args.max_length, default=512, minimum=64, maximum=4096)
    cluster_strong_min = _safe_float(args.cluster_strong_min, default=0.67, minimum=0.0, maximum=1.0)
    cluster_mid_min = _safe_float(args.cluster_mid_min, default=0.34, minimum=0.0, maximum=1.0)
    margin_top_k = _safe_int(args.margin_top_k, default=10, minimum=1, maximum=100_000)
    if cluster_mid_min > cluster_strong_min:
        cluster_mid_min = cluster_strong_min

    finetuned_ref = _resolve_model_ref(_clean_text(args.finetuned_model) or FINETUNED_MODEL_DEFAULT)
    base_ref = _resolve_model_ref(_clean_text(args.base_model) or PURE_BGE_MODEL_ID)
    embedding_model_id = _clean_text(args.embedding_model_id)

    grants = _load_grants_for_faculty(faculty_id=faculty_id, domain_threshold=domain_threshold, max_grants=max_grants)
    if not grants:
        raise RuntimeError(f"No grants found for faculty_id={faculty_id} with domain_score >= {domain_threshold:.3f}")

    grant_ids = [_clean_text(x.get("grant_id")) for x in grants if _clean_text(x.get("grant_id"))]
    grant_domain_map = {str(x["grant_id"]): float(x.get("domain_score") or 0.0) for x in grants}
    grant_specs = _load_grant_specs(grant_ids=grant_ids, embedding_model_id=embedding_model_id)
    faculty_specs = _load_faculty_specs(faculty_id=faculty_id, embedding_model_id=embedding_model_id)

    if max_fac_specs > 0:
        faculty_specs = faculty_specs[:max_fac_specs]
    if max_grant_specs > 0:
        grant_specs = grant_specs[:max_grant_specs]

    if not faculty_specs:
        raise RuntimeError(f"No faculty specs found for faculty_id={faculty_id}.")
    if not grant_specs:
        raise RuntimeError("No grant specialization specs found from selected grants.")

    device = _pick_device()

    tok_finetuned = AutoTokenizer.from_pretrained(finetuned_ref)
    model_finetuned = AutoModelForSequenceClassification.from_pretrained(finetuned_ref, num_labels=1).to(device).eval()

    tok_base = AutoTokenizer.from_pretrained(base_ref)
    model_base = AutoModelForSequenceClassification.from_pretrained(base_ref, num_labels=1).to(device).eval()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"fac_{faculty_id}_spec_compare_{ts}.json"
    output_txt = output_dir / f"fac_{faculty_id}_spec_compare_{ts}.txt"

    all_text_blocks_print: List[str] = []
    all_text_blocks_save: List[str] = []
    all_rows_json: List[Dict[str, Any]] = []

    meta_header = (
        f"faculty_id={faculty_id}\n"
        f"domain_threshold={domain_threshold}\n"
        f"grant_count={len(grants)}\n"
        f"grant_spec_count={len(grant_specs)}\n"
        f"faculty_spec_count={len(faculty_specs)}\n"
        f"finetuned_model={finetuned_ref}\n"
        f"base_model={base_ref}\n"
        f"device={device}\n"
        f"cluster_strong_min={cluster_strong_min}\n"
        f"cluster_mid_min={cluster_mid_min}\n"
        f"margin_top_k={margin_top_k}\n"
    )
    all_text_blocks_print.append(meta_header)
    all_text_blocks_save.append(meta_header)

    doc_texts = [_clean_text(x.get("text")) for x in grant_specs]
    grant_keys = [
        f"{_clean_text(x.get('grant_id'))}::spec#{_safe_int(x.get('grant_spec_id'), default=0, minimum=0, maximum=9_223_372_036_854_775_807)}"
        for x in grant_specs
    ]

    for fac_idx, fac_spec in enumerate(faculty_specs, start=1):
        query_text = _clean_text(fac_spec.get("text"))
        if not query_text:
            continue

        finetuned_scores = _score_docs_for_query(
            model=model_finetuned,
            tokenizer=tok_finetuned,
            query_text=query_text,
            doc_texts=doc_texts,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        base_scores = _score_docs_for_query(
            model=model_base,
            tokenizer=tok_base,
            query_text=query_text,
            doc_texts=doc_texts,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )

        rows: List[Dict[str, Any]] = []
        for i, grant_spec in enumerate(grant_specs):
            grant_id = _clean_text(grant_spec.get("grant_id"))
            row = {
                "fac_spec_id": int(fac_spec["fac_spec_id"]),
                "fac_spec_section": _clean_text(fac_spec.get("section")) or "unknown",
                "fac_spec_text": query_text,
                "grant_id": grant_id,
                "grant_key": grant_keys[i],
                "grant_domain_score": float(grant_domain_map.get(grant_id, 0.0)),
                "grant_spec_id": int(grant_spec["grant_spec_id"]),
                "grant_spec_section": _clean_text(grant_spec.get("section")) or "unknown",
                "grant_spec_text": _clean_text(grant_spec.get("text")),
                "finetuned_score": float(finetuned_scores[i]),
                "base_score": float(base_scores[i]),
            }
            row["score_margin"] = float(row["finetuned_score"]) - float(row["base_score"])
            row["abs_score_margin"] = abs(float(row["score_margin"]))
            row["score_cluster"] = _assign_cluster_label(
                float(row["finetuned_score"]),
                strong_min=cluster_strong_min,
                mid_min=cluster_mid_min,
            )
            rows.append(row)
            all_rows_json.append(dict(row))

        rows_by_finetuned = sorted(rows, key=lambda x: float(x["finetuned_score"]), reverse=True)
        rows_by_base = sorted(rows, key=lambda x: float(x["base_score"]), reverse=True)
        rows_strong = [x for x in rows_by_finetuned if _clean_text(x.get("score_cluster")) == "strong"]
        rows_mid = [x for x in rows_by_finetuned if _clean_text(x.get("score_cluster")) == "mid"]
        rows_low = [x for x in rows_by_finetuned if _clean_text(x.get("score_cluster")) == "low"]
        rows_by_margin = sorted(rows, key=lambda x: float(x.get("abs_score_margin") or 0.0), reverse=True)[
            : min(len(rows), margin_top_k)
        ]

        title_prefix = (
            f"FAC_SPEC #{fac_idx} "
            f"(fac_spec_id={int(fac_spec['fac_spec_id'])}, section={_clean_text(fac_spec.get('section')) or 'unknown'})\n"
            f"FAC_SPEC_TEXT: {_shorten(query_text, 220)}"
        )

        table_combined_print = _format_table(
            rows=rows,
            title=f"{title_prefix}\nTable 1: Combined (grantkey | finetuned | base)",
            text_col="grant_spec_text",
            score1_col="finetuned_score",
            score2_col="base_score",
            truncate_text=True,
        )
        table_finetuned_print = _format_table(
            rows=rows_by_finetuned,
            title=f"{title_prefix}\nTable 2: Sorted by Finetuned Score",
            text_col="grant_spec_text",
            score1_col="finetuned_score",
            truncate_text=True,
        )
        table_base_print = _format_table(
            rows=rows_by_base,
            title=f"{title_prefix}\nTable 3: Sorted by Base Score",
            text_col="grant_spec_text",
            score1_col="base_score",
            truncate_text=True,
        )
        table_strong_print = _format_table(
            rows=rows_strong,
            title=f"{title_prefix}\nCluster Strong (finetuned >= {cluster_strong_min:.2f})",
            text_col="grant_spec_text",
            score1_col="finetuned_score",
            score2_col="base_score",
            truncate_text=True,
        )
        table_mid_print = _format_table(
            rows=rows_mid,
            title=f"{title_prefix}\nCluster Mid ({cluster_mid_min:.2f} <= finetuned < {cluster_strong_min:.2f})",
            text_col="grant_spec_text",
            score1_col="finetuned_score",
            score2_col="base_score",
            truncate_text=True,
        )
        table_low_print = _format_table(
            rows=rows_low,
            title=f"{title_prefix}\nCluster Low (finetuned < {cluster_mid_min:.2f})",
            text_col="grant_spec_text",
            score1_col="finetuned_score",
            score2_col="base_score",
            truncate_text=True,
        )
        table_margin_print = _format_table(
            rows=rows_by_margin,
            title=f"{title_prefix}\nTop-{min(len(rows), margin_top_k)} Largest |Finetuned-Base| Margins",
            text_col="grant_spec_text",
            score1_col="finetuned_score",
            score2_col="base_score",
            score3_col="score_margin",
            score3_label="MARGIN",
            truncate_text=True,
        )

        table_combined_save = _format_table(
            rows=rows,
            title=f"{title_prefix}\nTable 1: Combined (grantkey | finetuned | base)",
            text_col="grant_spec_text",
            score1_col="finetuned_score",
            score2_col="base_score",
            truncate_text=False,
        )
        table_finetuned_save = _format_table(
            rows=rows_by_finetuned,
            title=f"{title_prefix}\nTable 2: Sorted by Finetuned Score",
            text_col="grant_spec_text",
            score1_col="finetuned_score",
            truncate_text=False,
        )
        table_base_save = _format_table(
            rows=rows_by_base,
            title=f"{title_prefix}\nTable 3: Sorted by Base Score",
            text_col="grant_spec_text",
            score1_col="base_score",
            truncate_text=False,
        )
        table_strong_save = _format_table(
            rows=rows_strong,
            title=f"{title_prefix}\nCluster Strong (finetuned >= {cluster_strong_min:.2f})",
            text_col="grant_spec_text",
            score1_col="finetuned_score",
            score2_col="base_score",
            truncate_text=False,
        )
        table_mid_save = _format_table(
            rows=rows_mid,
            title=f"{title_prefix}\nCluster Mid ({cluster_mid_min:.2f} <= finetuned < {cluster_strong_min:.2f})",
            text_col="grant_spec_text",
            score1_col="finetuned_score",
            score2_col="base_score",
            truncate_text=False,
        )
        table_low_save = _format_table(
            rows=rows_low,
            title=f"{title_prefix}\nCluster Low (finetuned < {cluster_mid_min:.2f})",
            text_col="grant_spec_text",
            score1_col="finetuned_score",
            score2_col="base_score",
            truncate_text=False,
        )
        table_margin_save = _format_table(
            rows=rows_by_margin,
            title=f"{title_prefix}\nTop-{min(len(rows), margin_top_k)} Largest |Finetuned-Base| Margins",
            text_col="grant_spec_text",
            score1_col="finetuned_score",
            score2_col="base_score",
            score3_col="score_margin",
            score3_label="MARGIN",
            truncate_text=False,
        )

        all_text_blocks_print.append(table_combined_print)
        all_text_blocks_print.append(table_finetuned_print)
        all_text_blocks_print.append(table_base_print)
        all_text_blocks_print.append(table_strong_print)
        all_text_blocks_print.append(table_mid_print)
        all_text_blocks_print.append(table_low_print)
        all_text_blocks_print.append(table_margin_print)
        all_text_blocks_save.append(table_combined_save)
        all_text_blocks_save.append(table_finetuned_save)
        all_text_blocks_save.append(table_base_save)
        all_text_blocks_save.append(table_strong_save)
        all_text_blocks_save.append(table_mid_save)
        all_text_blocks_save.append(table_low_save)
        all_text_blocks_save.append(table_margin_save)

    full_text_print = "\n".join(all_text_blocks_print).strip() + "\n"
    full_text_save = "\n".join(all_text_blocks_save).strip() + "\n"

    if bool(args.print):
        print(full_text_print)

    if bool(args.save):
        payload = {
            "meta": {
                "created_at_local": datetime.now().isoformat(),
                "faculty_id": int(faculty_id),
                "domain_threshold": float(domain_threshold),
                "grant_count": int(len(grants)),
                "grant_spec_count": int(len(grant_specs)),
                "faculty_spec_count": int(len(faculty_specs)),
                "finetuned_model": finetuned_ref,
                "base_model": base_ref,
                "device": str(device),
                "batch_size": int(batch_size),
                "max_length": int(max_length),
                "embedding_model_id_filter": embedding_model_id,
                "cluster_strong_min": float(cluster_strong_min),
                "cluster_mid_min": float(cluster_mid_min),
                "margin_top_k": int(margin_top_k),
                "output_txt": str(output_txt),
            },
            "rows": all_rows_json,
        }
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        output_txt.write_text(full_text_save, encoding="utf-8")
        print(f"saved_json={output_json}")
        print(f"saved_txt={output_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
