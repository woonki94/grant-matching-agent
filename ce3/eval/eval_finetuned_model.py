from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce3").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce3.aspect_modeling import (  # noqa: E402
    ASPECTS,
    ASPECT_HEADS_FILE,
    ASPECT_PREFIX_BY_NAME,
    aspect_id_from_name,
    format_aspect_pair,
    load_sequence_classifier_model,
    model_logits,
    strip_aspect_prefix,
)


BASE_MODEL_DEFAULT = "dleemiller/ModernCE-base-sts"
SPLIT_DIR_DEFAULT = "ce3/dataset/splits"
MODEL_DIR_DEFAULT = "ce3/models/aspect_reranker"
OUTPUT_DIR_DEFAULT = "ce3/eval/results"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_path(value: Any) -> Path:
    path = Path(_clean_text(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _clamp_01(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


def _parse_float_list(value: Any, *, default: Sequence[float]) -> List[float]:
    raw = _clean_text(value)
    if not raw:
        return [float(x) for x in default]
    out: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(max(0.0, float(token)))
        except Exception:
            continue
    return out or [float(x) for x in default]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _score_band(score: float, *, high_threshold: float, mid_threshold: float) -> str:
    if score >= float(high_threshold):
        return "high"
    if score >= float(mid_threshold):
        return "mid"
    return "low"


def _is_hf_model_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    return any((path / name).exists() for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"))


def _resolve_model_ref(value: Any) -> str:
    raw = _clean_text(value)
    if not raw:
        return raw
    path = Path(raw).expanduser()
    if path.exists():
        return str(path.resolve())
    resolved = _resolve_path(raw)
    if resolved.exists():
        return str(resolved)
    return raw


def _resolve_default_model_ref(model_dir: Any) -> str:
    root = _resolve_path(model_dir)
    candidates = [
        root / "best_stage2_selected",
        root / "final",
        root / "best_stage1_selected",
        root,
    ]
    if root.is_dir():
        candidates.extend(sorted(root.glob("stage2_epoch_*"), key=lambda p: p.stat().st_mtime, reverse=True))
        candidates.extend(sorted(root.glob("stage1_epoch_*"), key=lambda p: p.stat().st_mtime, reverse=True))
    for candidate in candidates:
        if _is_hf_model_dir(candidate):
            return str(candidate.resolve())
    return str((root / "best_stage2_selected").resolve())


def _has_saved_aspect_heads(model_ref: str) -> bool:
    path = Path(_clean_text(model_ref)).expanduser()
    if not path.is_absolute():
        path = _resolve_path(path)
    return path.is_dir() and (path / ASPECT_HEADS_FILE).exists()


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL row at {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                yield obj


def _doc_score(doc: Dict[str, Any], score_field: str) -> float:
    if score_field in doc:
        return _clamp_01(doc.get(score_field))
    for key in ("teacher_score", "score", "teacher_score_raw", "score_raw"):
        if key in doc:
            return _clamp_01(doc.get(key))
    return 0.0


def _load_listwise_rows(
    *,
    path: Path,
    score_field: str,
    high_threshold: float,
    mid_threshold: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for obj in _iter_jsonl(path):
        aspect = _clean_text(obj.get("aspect")).lower()
        if aspect not in ASPECTS:
            continue
        query_text = _clean_text(obj.get("query_text") or obj.get("spec_text") or obj.get("raw_query_text"))
        raw_query_text = _clean_text(obj.get("raw_query_text")) or strip_aspect_prefix(query_text)
        if not query_text or not raw_query_text:
            continue
        query_key = _clean_text(obj.get("query_item_id")) or "::".join(
            [aspect, _clean_text(obj.get("grant_id")), _clean_text(obj.get("spec_idx"))]
        )
        docs = obj.get("docs") or obj.get("ranked_docs") or obj.get("candidates") or []
        if not isinstance(docs, list):
            continue
        for rank, doc in enumerate(docs, start=1):
            if not isinstance(doc, dict):
                continue
            doc_text = _clean_text(doc.get("text"))
            if not doc_text:
                continue
            score = _doc_score(doc, score_field)
            band = _clean_text(doc.get("band") or doc.get("target_cluster")) or _score_band(
                score,
                high_threshold=high_threshold,
                mid_threshold=mid_threshold,
            )
            rows.append(
                {
                    "aspect": aspect,
                    "grant_id": _clean_text(obj.get("grant_id")),
                    "spec_idx": _safe_int(obj.get("spec_idx")),
                    "query_item_id": query_key,
                    "query_text": query_text,
                    "raw_query_text": raw_query_text,
                    "doc_text": doc_text,
                    "teacher_score": float(score),
                    "band": band,
                    "doc_rank_in_file": int(rank),
                    "pair_id": _clean_text(doc.get("pair_id")),
                    "fac_item_id": _clean_text(doc.get("fac_item_id")),
                    "ce_prefilter_score": _clamp_01(doc.get("ce_prefilter_score")),
                    "ce_prefilter_rank": _safe_int(doc.get("ce_prefilter_rank")),
                }
            )
    return rows


def _load_pairwise_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for obj in _iter_jsonl(path):
        aspect = _clean_text(obj.get("aspect")).lower()
        query_text = _clean_text(obj.get("query_text") or obj.get("raw_query_text"))
        pos_text = _clean_text(obj.get("pos_text"))
        neg_text = _clean_text(obj.get("neg_text"))
        if aspect not in ASPECTS or not query_text or not pos_text or not neg_text:
            continue
        rows.append(
            {
                "aspect": aspect,
                "query_text": query_text,
                "raw_query_text": _clean_text(obj.get("raw_query_text")) or strip_aspect_prefix(query_text),
                "pos_text": pos_text,
                "neg_text": neg_text,
                "teacher_pos_score": _clamp_01(obj.get("teacher_pos_score")),
                "teacher_neg_score": _clamp_01(obj.get("teacher_neg_score")),
                "teacher_margin": max(0.0, float(obj.get("teacher_margin", 0.0) or 0.0)),
                "pair_type": _clean_text(obj.get("pair_type")) or "unknown",
                "pos_band": _clean_text(obj.get("pos_band")),
                "neg_band": _clean_text(obj.get("neg_band")),
            }
        )
    return rows


def _model_inputs(query_text: str, doc_text: str, *, use_long_prefix: bool, aspect: str) -> Tuple[str, str]:
    if use_long_prefix:
        prefixed = query_text
        prefix = ASPECT_PREFIX_BY_NAME.get(aspect, f"[{aspect.upper()}]")
        if not prefixed.upper().lstrip().startswith(prefix):
            prefixed = f"{prefix} {prefixed}"
        return format_aspect_pair(prefixed, doc_text)
    return strip_aspect_prefix(query_text), strip_aspect_prefix(doc_text)


@torch.no_grad()
def _score_text_pairs(
    *,
    model: Any,
    tokenizer: Any,
    pairs: Sequence[Tuple[str, str, str]],
    device: torch.device,
    batch_size: int,
    max_length: int,
    use_long_prefix: bool,
) -> List[float]:
    scores: List[float] = []
    step = max(1, int(batch_size))
    model.eval()
    for start in range(0, len(pairs), step):
        chunk = pairs[start : start + step]
        queries: List[str] = []
        docs: List[str] = []
        aspect_ids: List[int] = []
        for aspect, query_text, doc_text in chunk:
            q, d = _model_inputs(query_text, doc_text, use_long_prefix=use_long_prefix, aspect=aspect)
            queries.append(q)
            docs.append(d)
            aspect_ids.append(aspect_id_from_name(aspect))
        enc = tokenizer(
            queries,
            docs,
            max_length=int(max_length),
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = _to_device(enc, device)
        aspect_tensor = torch.tensor(aspect_ids, dtype=torch.long, device=device)
        logits = model_logits(model, enc, aspect_ids=aspect_tensor).view(-1)
        probs = torch.sigmoid(logits)
        scores.extend(float(x) for x in probs.detach().cpu().tolist())
    return scores


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / math.sqrt(vx * vy))


def _group_rows(rows: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(_clean_text(row.get("aspect")), _clean_text(row.get("query_item_id")))].append(row)
    return list(grouped.values())


def _dcg(rels: Sequence[float]) -> float:
    total = 0.0
    for rank, rel in enumerate(rels, start=1):
        total += float((2.0 ** float(rel) - 1.0) / math.log2(rank + 1.0))
    return total


def _ranking_metrics(
    rows: Sequence[Dict[str, Any]],
    *,
    score_key: str,
    top_k: int,
    pair_eps: float,
    hard_gap_max: float,
    medium_gap_max: float,
    high_threshold: float,
    mid_threshold: float,
) -> Dict[str, Any]:
    groups = _group_rows(rows)
    top_k = max(1, int(top_k))
    eps = max(0.0, float(pair_eps))
    hard_gap = max(eps, float(hard_gap_max))
    medium_gap = max(hard_gap, float(medium_gap_max))

    query_count = 0
    query_2plus = 0
    top1 = 0
    topk_overlap_sum = 0.0
    mrr_sum = 0.0
    mrr_count = 0
    recall_sum = 0.0
    recall_count = 0
    ndcg_sum = 0.0
    ndcg_count = 0
    buckets = {
        "overall": {"pairs": 0, "correct": 0, "margin_sum": 0.0},
        "hard": {"pairs": 0, "correct": 0, "margin_sum": 0.0},
        "medium": {"pairs": 0, "correct": 0, "margin_sum": 0.0},
        "easy": {"pairs": 0, "correct": 0, "margin_sum": 0.0},
    }

    for group in groups:
        n = len(group)
        query_count += 1
        if n < 2:
            continue
        query_2plus += 1
        teacher = [float(row.get("teacher_score", 0.0) or 0.0) for row in group]
        pred = [float(row.get(score_key, 0.0) or 0.0) for row in group]
        teacher_order = sorted(range(n), key=lambda idx: teacher[idx], reverse=True)
        pred_order = sorted(range(n), key=lambda idx: pred[idx], reverse=True)

        best_teacher = teacher[teacher_order[0]]
        if abs(teacher[pred_order[0]] - best_teacher) <= eps:
            top1 += 1

        k_eff = min(top_k, n)
        teacher_topk = set(teacher_order[:k_eff])
        pred_topk = set(pred_order[:k_eff])
        topk_overlap_sum += len(teacher_topk.intersection(pred_topk)) / float(k_eff)

        rr = 0.0
        for rank, idx in enumerate(pred_order[:k_eff], start=1):
            if teacher[idx] >= float(high_threshold):
                rr = 1.0 / float(rank)
                break
        mrr_sum += rr
        mrr_count += 1

        relevant = {idx for idx, score in enumerate(teacher) if score >= float(mid_threshold)}
        if relevant:
            recall_sum += len(pred_topk.intersection(relevant)) / float(len(relevant))
            recall_count += 1

        model_rels = [teacher[idx] for idx in pred_order[:k_eff]]
        ideal_rels = [teacher[idx] for idx in teacher_order[:k_eff]]
        ideal = _dcg(ideal_rels)
        if ideal > 0.0:
            ndcg_sum += _dcg(model_rels) / ideal
            ndcg_count += 1

        for i in range(n):
            for j in range(i + 1, n):
                gap = abs(teacher[i] - teacher[j])
                if gap <= eps:
                    continue
                if teacher[i] > teacher[j]:
                    pred_margin = pred[i] - pred[j]
                else:
                    pred_margin = pred[j] - pred[i]
                bucket_names = ["overall"]
                if gap <= hard_gap:
                    bucket_names.append("hard")
                elif gap <= medium_gap:
                    bucket_names.append("medium")
                else:
                    bucket_names.append("easy")
                for bucket in bucket_names:
                    buckets[bucket]["pairs"] += 1
                    buckets[bucket]["correct"] += int(pred_margin > 0.0)
                    buckets[bucket]["margin_sum"] += float(pred_margin)

    pair_metrics: Dict[str, Dict[str, float]] = {}
    for bucket, values in buckets.items():
        count = int(values["pairs"])
        pair_metrics[bucket] = {
            "pair_count": float(count),
            "pair_accuracy": float(values["correct"] / max(1, count)),
            "mean_pred_margin": float(values["margin_sum"] / max(1, count)),
        }
    return {
        "query_count": int(query_count),
        "query_count_2plus": int(query_2plus),
        "top1_accuracy": float(top1 / max(1, query_2plus)),
        "topk_overlap": float(topk_overlap_sum / max(1, query_2plus)),
        "mrr_at_k": float(mrr_sum / max(1, mrr_count)),
        "ndcg_at_k": float(ndcg_sum / max(1, ndcg_count)),
        "recall_at_k": float(recall_sum / max(1, recall_count)),
        "pair_metrics": pair_metrics,
    }


def _regression_metrics(rows: Sequence[Dict[str, Any]], *, score_key: str) -> Dict[str, float]:
    y = [float(row.get("teacher_score", 0.0) or 0.0) for row in rows]
    p = [float(row.get(score_key, 0.0) or 0.0) for row in rows]
    if not y:
        return {"mae": 0.0, "rmse": 0.0, "pearson": 0.0}
    abs_err = [abs(a - b) for a, b in zip(y, p)]
    sq_err = [(a - b) ** 2 for a, b in zip(y, p)]
    return {
        "mae": float(sum(abs_err) / len(abs_err)),
        "rmse": float(math.sqrt(sum(sq_err) / len(sq_err))),
        "pearson": _pearson(y, p),
    }


def _raw_score_sanity(
    rows: Sequence[Dict[str, Any]],
    *,
    score_key: str,
    high_threshold: float,
    mid_threshold: float,
    oob_margin: float,
) -> Dict[str, Any]:
    by_band: Dict[str, List[float]] = {"high": [], "mid": [], "low": []}
    for row in rows:
        band = _clean_text(row.get("band")).lower()
        if band not in by_band:
            band = _score_band(float(row.get("teacher_score", 0.0) or 0.0), high_threshold=high_threshold, mid_threshold=mid_threshold)
        by_band[band].append(float(row.get(score_key, 0.0) or 0.0))

    def avg(vals: Sequence[float]) -> float:
        return float(sum(vals) / max(1, len(vals)))

    margin = max(0.0, float(oob_margin))
    low_upper = min(1.0, float(mid_threshold) + margin)
    mid_lower = max(0.0, float(mid_threshold) - margin)
    mid_upper = min(1.0, float(high_threshold) + margin)
    high_lower = max(0.0, float(high_threshold) - margin)

    low_vals = by_band["low"]
    mid_vals = by_band["mid"]
    high_vals = by_band["high"]
    low_out = sum(1 for x in low_vals if x >= low_upper)
    mid_low = sum(1 for x in mid_vals if x < mid_lower)
    mid_high = sum(1 for x in mid_vals if x >= mid_upper)
    high_out = sum(1 for x in high_vals if x < high_lower)

    low_rate = low_out / max(1, len(low_vals))
    mid_low_rate = mid_low / max(1, len(mid_vals))
    mid_high_rate = mid_high / max(1, len(mid_vals))
    mid_rate = (mid_low + mid_high) / max(1, len(mid_vals))
    high_rate = high_out / max(1, len(high_vals))
    oob_objective = (2.0 * high_rate + mid_rate + low_rate) / 4.0
    avg_high = avg(high_vals)
    avg_mid = avg(mid_vals)
    avg_low = avg(low_vals)
    return {
        "avg_pred_high": float(avg_high),
        "avg_pred_mid": float(avg_mid),
        "avg_pred_low": float(avg_low),
        "monotonic_high_mid_low": bool(avg_high >= avg_mid >= avg_low),
        "count_high": int(len(high_vals)),
        "count_mid": int(len(mid_vals)),
        "count_low": int(len(low_vals)),
        "low_out_count": int(low_out),
        "mid_out_count": int(mid_low + mid_high),
        "mid_low_out_count": int(mid_low),
        "mid_high_out_count": int(mid_high),
        "high_out_count": int(high_out),
        "low_out_rate": float(low_rate),
        "mid_out_rate": float(mid_rate),
        "mid_low_out_rate": float(mid_low_rate),
        "mid_high_out_rate": float(mid_high_rate),
        "high_out_rate": float(high_rate),
        "weak_leak_rate": float(sum(1 for x in low_vals if x >= high_threshold) / max(1, len(low_vals))),
        "strong_collapse_rate": float(sum(1 for x in high_vals if x < mid_threshold) / max(1, len(high_vals))),
        "oob_objective": float(oob_objective),
    }


def _pairwise_metrics(rows: Sequence[Dict[str, Any]], *, score_prefix: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["overall"].append(row)
        groups[_clean_text(row.get("pair_type")) or "unknown"].append(row)

    out: Dict[str, Any] = {}
    for name, group in sorted(groups.items()):
        correct = 0
        margin_sum = 0.0
        teacher_margin_sum = 0.0
        for row in group:
            pos_score = float(row.get(f"{score_prefix}_pos_score", 0.0) or 0.0)
            neg_score = float(row.get(f"{score_prefix}_neg_score", 0.0) or 0.0)
            margin = pos_score - neg_score
            correct += int(margin > 0.0)
            margin_sum += margin
            teacher_margin_sum += float(row.get("teacher_margin", 0.0) or 0.0)
        count = len(group)
        out[name] = {
            "pair_count": int(count),
            "pair_accuracy": float(correct / max(1, count)),
            "mean_pred_margin": float(margin_sum / max(1, count)),
            "mean_teacher_margin": float(teacher_margin_sum / max(1, count)),
        }
    return out


def _bundle(
    rows: Sequence[Dict[str, Any]],
    *,
    score_key: str,
    top_k: int,
    pair_eps: float,
    hard_gap_max: float,
    medium_gap_max: float,
    high_threshold: float,
    mid_threshold: float,
    oob_margin: float,
) -> Dict[str, Any]:
    return {
        "row_count": int(len(rows)),
        "regression": _regression_metrics(rows, score_key=score_key),
        "ranking": _ranking_metrics(
            rows,
            score_key=score_key,
            top_k=top_k,
            pair_eps=pair_eps,
            hard_gap_max=hard_gap_max,
            medium_gap_max=medium_gap_max,
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
        ),
        "raw_score_sanity": _raw_score_sanity(
            rows,
            score_key=score_key,
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
            oob_margin=oob_margin,
        ),
    }


def _by_aspect(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out = {aspect: [] for aspect in ASPECTS}
    for row in rows:
        aspect = _clean_text(row.get("aspect"))
        if aspect in out:
            out[aspect].append(row)
    return out


def _fmt_pct_cell(out: int, total: int, rate: float) -> str:
    return f"{int(out)}/{int(total)} ({100.0 * float(rate):.2f}%)"


def _format_main_table(metrics: Dict[str, Dict[str, Any]], *, model_label: str) -> str:
    lines = ["", f"=== Main Metrics: {model_label} ==="]
    lines.append(
        f"{'GROUP':<12} {'N':>7} {'MAE':>8} {'RMSE':>8} {'PEARSON':>8} "
        f"{'TOP1':>8} {'MRR@K':>8} {'NDCG@K':>8} {'RECALL@K':>9} {'PAIR_ACC':>9}"
    )
    lines.append("-" * 100)
    for group in ("overall", *ASPECTS):
        obj = metrics.get(group) or {}
        reg = obj.get("regression") or {}
        rank = obj.get("ranking") or {}
        pair = ((rank.get("pair_metrics") or {}).get("overall") or {})
        lines.append(
            f"{group:<12} {int(obj.get('row_count') or 0):>7} "
            f"{float(reg.get('mae') or 0.0):>8.4f} "
            f"{float(reg.get('rmse') or 0.0):>8.4f} "
            f"{float(reg.get('pearson') or 0.0):>8.4f} "
            f"{float(rank.get('top1_accuracy') or 0.0):>8.4f} "
            f"{float(rank.get('mrr_at_k') or 0.0):>8.4f} "
            f"{float(rank.get('ndcg_at_k') or 0.0):>8.4f} "
            f"{float(rank.get('recall_at_k') or 0.0):>9.4f} "
            f"{float(pair.get('pair_accuracy') or 0.0):>9.4f}"
        )
    return "\n".join(lines)


def _format_oob_table(metrics: Dict[str, Dict[str, Any]], *, model_label: str, high_threshold: float, mid_threshold: float, oob_margin: float) -> str:
    lines = ["", f"=== Out-Of-Band Summary: {model_label} ==="]
    lines.append(f"thresholds: high>={high_threshold:.2f}, mid>={mid_threshold:.2f}, oob_margin={oob_margin:.2f}")
    lines.append(
        f"{'GROUP':<12} {'LOW_OUT':>21} {'MID_OUT':>21} {'MID_LOW':>21} "
        f"{'MID_HIGH':>21} {'HIGH_OUT':>21} {'OBJ':>8}"
    )
    lines.append("-" * 122)
    for group in ("overall", *ASPECTS):
        raw = (metrics.get(group) or {}).get("raw_score_sanity") or {}
        low = _fmt_pct_cell(raw.get("low_out_count", 0), raw.get("count_low", 0), raw.get("low_out_rate", 0.0))
        mid = _fmt_pct_cell(raw.get("mid_out_count", 0), raw.get("count_mid", 0), raw.get("mid_out_rate", 0.0))
        mid_low = _fmt_pct_cell(raw.get("mid_low_out_count", 0), raw.get("count_mid", 0), raw.get("mid_low_out_rate", 0.0))
        mid_high = _fmt_pct_cell(raw.get("mid_high_out_count", 0), raw.get("count_mid", 0), raw.get("mid_high_out_rate", 0.0))
        high = _fmt_pct_cell(raw.get("high_out_count", 0), raw.get("count_high", 0), raw.get("high_out_rate", 0.0))
        lines.append(
            f"{group:<12} {low:>21} {mid:>21} {mid_low:>21} {mid_high:>21} "
            f"{high:>21} {float(raw.get('oob_objective') or 0.0):>8.4f}"
        )
    return "\n".join(lines)


def _format_raw_table(metrics: Dict[str, Dict[str, Any]], *, model_label: str) -> str:
    lines = ["", f"=== Raw Score Sanity: {model_label} ==="]
    lines.append(f"{'GROUP':<12} {'AVG_HIGH':>10} {'AVG_MID':>10} {'AVG_LOW':>10} {'MONOTONIC':>10} {'WEAK_LEAK':>11} {'STRONG_COLL':>12}")
    lines.append("-" * 90)
    for group in ("overall", *ASPECTS):
        raw = (metrics.get(group) or {}).get("raw_score_sanity") or {}
        lines.append(
            f"{group:<12} "
            f"{float(raw.get('avg_pred_high') or 0.0):>10.4f} "
            f"{float(raw.get('avg_pred_mid') or 0.0):>10.4f} "
            f"{float(raw.get('avg_pred_low') or 0.0):>10.4f} "
            f"{str(bool(raw.get('monotonic_high_mid_low'))):>10} "
            f"{float(raw.get('weak_leak_rate') or 0.0):>11.4f} "
            f"{float(raw.get('strong_collapse_rate') or 0.0):>12.4f}"
        )
    return "\n".join(lines)


def _format_pairwise_table(pair_metrics: Dict[str, Any], *, model_label: str) -> str:
    lines = ["", f"=== Explicit Pairwise Test: {model_label} ==="]
    lines.append(f"{'PAIR_TYPE':<24} {'N':>8} {'PAIR_ACC':>9} {'PRED_M':>9} {'TEACHER_M':>10}")
    lines.append("-" * 68)
    for pair_type, row in sorted(pair_metrics.items(), key=lambda kv: (0 if kv[0] == "overall" else 1, kv[0])):
        lines.append(
            f"{pair_type:<24} {int(row.get('pair_count') or 0):>8} "
            f"{float(row.get('pair_accuracy') or 0.0):>9.4f} "
            f"{float(row.get('mean_pred_margin') or 0.0):>9.4f} "
            f"{float(row.get('mean_teacher_margin') or 0.0):>10.4f}"
        )
    return "\n".join(lines)


def _format_explicit_pairwise_comparison(
    *,
    teacher_pair_metrics: Dict[str, Any],
    finetuned_pair_metrics: Dict[str, Any],
    base_pair_metrics: Optional[Dict[str, Any]],
) -> str:
    lines = ["", "=== Explicit Pairwise Comparison ==="]
    lines.append(
        f"{'PAIR_TYPE':<24} {'N':>8} {'TEACHER':>9} {'FINETUNED':>10} {'BASE':>9} "
        f"{'IMPROVE':>9} {'FT_M':>9} {'BASE_M':>9}"
    )
    lines.append("-" * 96)
    keys = sorted(
        set(teacher_pair_metrics) | set(finetuned_pair_metrics) | set(base_pair_metrics or {}),
        key=lambda x: (0 if x == "overall" else 1, x),
    )
    for pair_type in keys:
        teacher = teacher_pair_metrics.get(pair_type) or {}
        finetuned = finetuned_pair_metrics.get(pair_type) or {}
        base = (base_pair_metrics or {}).get(pair_type) or {}
        base_acc = float(base.get("pair_accuracy")) if base else None
        ft_acc = float(finetuned.get("pair_accuracy") or 0.0)
        improve = ft_acc - base_acc if base_acc is not None else None
        lines.append(
            f"{pair_type:<24} {int(finetuned.get('pair_count') or teacher.get('pair_count') or base.get('pair_count') or 0):>8} "
            f"{float(teacher.get('pair_accuracy') or 0.0):>9.4f} "
            f"{ft_acc:>10.4f} "
            f"{_fmt_metric_value(base_acc):>9} "
            f"{_fmt_metric_value(improve):>9} "
            f"{float(finetuned.get('mean_pred_margin') or 0.0):>9.4f} "
            f"{_fmt_metric_value(float(base.get('mean_pred_margin')) if base else None):>9}"
        )
    lines.append("IMPROVE is finetuned pair accuracy minus base pair accuracy.")
    return "\n".join(lines)


def _oob_metrics_by_margin(
    rows: Sequence[Dict[str, Any]],
    *,
    score_key: str,
    high_threshold: float,
    mid_threshold: float,
    margins: Sequence[float],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    grouped: Dict[str, Sequence[Dict[str, Any]]] = {"overall": rows, **_by_aspect(rows)}
    for margin in margins:
        label = f"{float(margin):.2f}"
        out[label] = {}
        for group, group_rows in grouped.items():
            out[label][group] = _raw_score_sanity(
                group_rows,
                score_key=score_key,
                high_threshold=high_threshold,
                mid_threshold=mid_threshold,
                oob_margin=float(margin),
            )
    return out


def _nested_get(obj: Dict[str, Any], path: Sequence[str], default: float = 0.0) -> float:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return float(default)
        cur = cur.get(key)
    try:
        return float(cur)
    except Exception:
        return float(default)


def _fmt_metric_value(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _comparison_metric_rows(
    *,
    group: str,
    teacher_metrics: Dict[str, Dict[str, Any]],
    finetuned_metrics: Dict[str, Dict[str, Any]],
    base_metrics: Optional[Dict[str, Dict[str, Any]]],
    teacher_oob: Dict[str, Dict[str, Dict[str, Any]]],
    finetuned_oob: Dict[str, Dict[str, Dict[str, Any]]],
    base_oob: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    margins: Sequence[float],
) -> List[Tuple[str, float, float, Optional[float], Optional[float]]]:
    metric_specs: List[Tuple[str, Sequence[str], bool]] = [
        ("MAE", ("regression", "mae"), False),
        ("RMSE", ("regression", "rmse"), False),
        ("PEARSON", ("regression", "pearson"), True),
        ("TOP1", ("ranking", "top1_accuracy"), True),
        ("MRR@K", ("ranking", "mrr_at_k"), True),
        ("NDCG@K", ("ranking", "ndcg_at_k"), True),
        ("RECALL@K", ("ranking", "recall_at_k"), True),
        ("PAIR_ACC", ("ranking", "pair_metrics", "overall", "pair_accuracy"), True),
    ]
    rows: List[Tuple[str, float, float, Optional[float], Optional[float]]] = []
    for name, path, higher_is_better in metric_specs:
        teacher = _nested_get(teacher_metrics.get(group) or {}, path)
        finetuned = _nested_get(finetuned_metrics.get(group) or {}, path)
        base = _nested_get((base_metrics or {}).get(group) or {}, path) if base_metrics is not None else None
        improve = None
        if base is not None:
            improve = finetuned - base if higher_is_better else base - finetuned
        rows.append((name, teacher, finetuned, base, improve))

    for margin in margins:
        label = f"{float(margin):.2f}"
        prefix = "STRICT" if float(margin) == 0.0 else f"MARGIN{float(margin):.2f}"
        for name, key in (
            (f"{prefix}_LOW_OUT", "low_out_rate"),
            (f"{prefix}_MID_OUT", "mid_out_rate"),
            (f"{prefix}_HIGH_OUT", "high_out_rate"),
            (f"{prefix}_OOB_OBJ", "oob_objective"),
        ):
            teacher = _nested_get((teacher_oob.get(label) or {}).get(group) or {}, (key,))
            finetuned = _nested_get((finetuned_oob.get(label) or {}).get(group) or {}, (key,))
            base = _nested_get(((base_oob or {}).get(label) or {}).get(group) or {}, (key,)) if base_oob is not None else None
            improve = (base - finetuned) if base is not None else None
            rows.append((name, teacher, finetuned, base, improve))

    raw_specs = [
        ("AVG_HIGH", "avg_pred_high"),
        ("AVG_MID", "avg_pred_mid"),
        ("AVG_LOW", "avg_pred_low"),
    ]
    strict_label = f"{0.0:.2f}"
    for name, key in raw_specs:
        teacher = _nested_get((teacher_oob.get(strict_label) or {}).get(group) or {}, (key,))
        finetuned = _nested_get((finetuned_oob.get(strict_label) or {}).get(group) or {}, (key,))
        base = _nested_get(((base_oob or {}).get(strict_label) or {}).get(group) or {}, (key,)) if base_oob is not None else None
        improve = None
        rows.append((name, teacher, finetuned, base, improve))
    return rows


def _format_comparison_table(
    *,
    group: str,
    teacher_metrics: Dict[str, Dict[str, Any]],
    finetuned_metrics: Dict[str, Dict[str, Any]],
    base_metrics: Optional[Dict[str, Dict[str, Any]]],
    teacher_oob: Dict[str, Dict[str, Dict[str, Any]]],
    finetuned_oob: Dict[str, Dict[str, Dict[str, Any]]],
    base_oob: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    margins: Sequence[float],
) -> str:
    lines = ["", f"=== Metric Comparison: {group} ==="]
    lines.append(f"{'METRIC':<20} {'TEACHER':>10} {'FINETUNED':>10} {'BASE':>10} {'IMPROVE':>10}")
    lines.append("-" * 66)
    for name, teacher, finetuned, base, improve in _comparison_metric_rows(
        group=group,
        teacher_metrics=teacher_metrics,
        finetuned_metrics=finetuned_metrics,
        base_metrics=base_metrics,
        teacher_oob=teacher_oob,
        finetuned_oob=finetuned_oob,
        base_oob=base_oob,
        margins=margins,
    ):
        lines.append(
            f"{name:<20} {_fmt_metric_value(teacher):>10} {_fmt_metric_value(finetuned):>10} "
            f"{_fmt_metric_value(base):>10} {_fmt_metric_value(improve):>10}"
        )
    lines.append("IMPROVE is positive when finetuned is better than base; raw averages have NA improve.")
    return "\n".join(lines)


def _compute_model_metrics(
    rows: Sequence[Dict[str, Any]],
    *,
    score_key: str,
    top_k: int,
    pair_eps: float,
    hard_gap_max: float,
    medium_gap_max: float,
    high_threshold: float,
    mid_threshold: float,
    oob_margin: float,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {
        "overall": _bundle(
            rows,
            score_key=score_key,
            top_k=top_k,
            pair_eps=pair_eps,
            hard_gap_max=hard_gap_max,
            medium_gap_max=medium_gap_max,
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
            oob_margin=oob_margin,
        )
    }
    for aspect, aspect_rows in _by_aspect(rows).items():
        out[aspect] = _bundle(
            aspect_rows,
            score_key=score_key,
            top_k=top_k,
            pair_eps=pair_eps,
            hard_gap_max=hard_gap_max,
            medium_gap_max=medium_gap_max,
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
            oob_margin=oob_margin,
        )
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a CE3 aspect-conditioned checkpoint on never-used test split files.")
    p.add_argument("--model", type=str, default="", help="Finetuned checkpoint. Defaults to best_stage2_selected under --model-dir.")
    p.add_argument("--model-dir", type=str, default=MODEL_DIR_DEFAULT)
    p.add_argument("--split-dir", type=str, default=SPLIT_DIR_DEFAULT)
    p.add_argument("--test-listwise", type=str, default="")
    p.add_argument("--test-pairwise", type=str, default="")
    p.add_argument("--score-field", type=str, default="teacher_score")
    p.add_argument("--base-model", type=str, default=BASE_MODEL_DEFAULT)
    p.add_argument("--compare-base", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--base-use-long-prefix", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--multihead",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use saved CE3 aspect heads when present. If the checkpoint has no aspect-head file, the evaluator falls back to the shared head.",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--pair-eps", type=float, default=0.01)
    p.add_argument("--hard-gap-max", type=float, default=0.15)
    p.add_argument("--medium-gap-max", type=float, default=0.40)
    p.add_argument("--high-threshold", type=float, default=0.70)
    p.add_argument("--mid-threshold", type=float, default=0.30)
    p.add_argument("--oob-margin", type=float, default=0.0)
    p.add_argument("--oob-margins", type=str, default="0,0.03,0.05", help="Comma-separated OOB margins for comparison tables.")
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--save-prefix", type=str, default="ce3_test_eval")
    p.add_argument("--save-rows", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return p


def main() -> int:
    started = time.time()
    args = _build_parser().parse_args()

    split_dir = _resolve_path(args.split_dir)
    listwise_path = _resolve_path(args.test_listwise) if _clean_text(args.test_listwise) else split_dir / "llm_distill_all_listwise_test.jsonl"
    pairwise_path = _resolve_path(args.test_pairwise) if _clean_text(args.test_pairwise) else split_dir / "llm_distill_all_pairwise_test.jsonl"
    if not listwise_path.exists():
        raise FileNotFoundError(f"Missing test listwise split: {listwise_path}")
    model_ref = _resolve_model_ref(args.model) if _clean_text(args.model) else _resolve_default_model_ref(args.model_dir)
    high_threshold = _clamp_01(args.high_threshold)
    mid_threshold = min(high_threshold, _clamp_01(args.mid_threshold))
    oob_margin = _clamp_01(args.oob_margin)
    oob_margins = sorted({0.0, *_parse_float_list(args.oob_margins, default=(0.0, 0.03, 0.05))})
    batch_size = max(1, int(args.batch_size))
    max_length = max(64, int(args.max_length))
    top_k = max(1, int(args.top_k))

    rows = _load_listwise_rows(
        path=listwise_path,
        score_field=args.score_field,
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
    )
    if not rows:
        raise RuntimeError(f"No usable test rows loaded from {listwise_path}")
    pair_rows = _load_pairwise_rows(pairwise_path)

    device = _pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=bool(args.trust_remote_code))
    use_multihead = bool(args.multihead) and _has_saved_aspect_heads(model_ref)
    model = load_sequence_classifier_model(
        model_ref,
        num_labels=1,
        multi_aspect_heads=use_multihead,
        trust_remote_code=bool(args.trust_remote_code),
    ).to(device)

    eval_pairs = [(row["aspect"], row["query_text"], row["doc_text"]) for row in rows]
    finetuned_scores = _score_text_pairs(
        model=model,
        tokenizer=tokenizer,
        pairs=eval_pairs,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        use_long_prefix=True,
    )
    for row, score in zip(rows, finetuned_scores):
        row["finetuned_score"] = float(score)

    if pair_rows:
        pair_eval_pos = [(row["aspect"], row["query_text"], row["pos_text"]) for row in pair_rows]
        pair_eval_neg = [(row["aspect"], row["query_text"], row["neg_text"]) for row in pair_rows]
        pos_scores = _score_text_pairs(
            model=model,
            tokenizer=tokenizer,
            pairs=pair_eval_pos,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            use_long_prefix=True,
        )
        neg_scores = _score_text_pairs(
            model=model,
            tokenizer=tokenizer,
            pairs=pair_eval_neg,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            use_long_prefix=True,
        )
        for row, pos_score, neg_score in zip(pair_rows, pos_scores, neg_scores):
            row["finetuned_pos_score"] = float(pos_score)
            row["finetuned_neg_score"] = float(neg_score)

    base_metrics: Optional[Dict[str, Any]] = None
    base_pair_metrics: Optional[Dict[str, Any]] = None
    base_ref = ""
    if bool(args.compare_base):
        base_ref = _resolve_model_ref(args.base_model)
        base_tokenizer = AutoTokenizer.from_pretrained(base_ref, trust_remote_code=bool(args.trust_remote_code))
        base_model = load_sequence_classifier_model(
            base_ref,
            num_labels=1,
            multi_aspect_heads=False,
            trust_remote_code=bool(args.trust_remote_code),
        ).to(device)
        base_pairs = [
            (row["aspect"], row["query_text"] if bool(args.base_use_long_prefix) else row["raw_query_text"], row["doc_text"])
            for row in rows
        ]
        base_scores = _score_text_pairs(
            model=base_model,
            tokenizer=base_tokenizer,
            pairs=base_pairs,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            use_long_prefix=bool(args.base_use_long_prefix),
        )
        for row, score in zip(rows, base_scores):
            row["base_score"] = float(score)
        if pair_rows:
            base_pos = [
                (row["aspect"], row["query_text"] if bool(args.base_use_long_prefix) else row["raw_query_text"], row["pos_text"])
                for row in pair_rows
            ]
            base_neg = [
                (row["aspect"], row["query_text"] if bool(args.base_use_long_prefix) else row["raw_query_text"], row["neg_text"])
                for row in pair_rows
            ]
            base_pos_scores = _score_text_pairs(
                model=base_model,
                tokenizer=base_tokenizer,
                pairs=base_pos,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                use_long_prefix=bool(args.base_use_long_prefix),
            )
            base_neg_scores = _score_text_pairs(
                model=base_model,
                tokenizer=base_tokenizer,
                pairs=base_neg,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                use_long_prefix=bool(args.base_use_long_prefix),
            )
            for row, pos_score, neg_score in zip(pair_rows, base_pos_scores, base_neg_scores):
                row["base_pos_score"] = float(pos_score)
                row["base_neg_score"] = float(neg_score)

    teacher_metrics = _compute_model_metrics(
        rows,
        score_key="teacher_score",
        top_k=top_k,
        pair_eps=float(args.pair_eps),
        hard_gap_max=float(args.hard_gap_max),
        medium_gap_max=float(args.medium_gap_max),
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
        oob_margin=0.0,
    )
    teacher_pair_metrics = _pairwise_metrics(pair_rows, score_prefix="teacher") if pair_rows else {}
    finetuned_metrics = _compute_model_metrics(
        rows,
        score_key="finetuned_score",
        top_k=top_k,
        pair_eps=float(args.pair_eps),
        hard_gap_max=float(args.hard_gap_max),
        medium_gap_max=float(args.medium_gap_max),
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
        oob_margin=oob_margin,
    )
    finetuned_pair_metrics = _pairwise_metrics(pair_rows, score_prefix="finetuned") if pair_rows else {}
    if bool(args.compare_base):
        base_metrics = _compute_model_metrics(
            rows,
            score_key="base_score",
            top_k=top_k,
            pair_eps=float(args.pair_eps),
            hard_gap_max=float(args.hard_gap_max),
            medium_gap_max=float(args.medium_gap_max),
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
            oob_margin=oob_margin,
        )
        base_pair_metrics = _pairwise_metrics(pair_rows, score_prefix="base") if pair_rows else {}

    teacher_oob_by_margin = _oob_metrics_by_margin(
        rows,
        score_key="teacher_score",
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
        margins=oob_margins,
    )
    finetuned_oob_by_margin = _oob_metrics_by_margin(
        rows,
        score_key="finetuned_score",
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
        margins=oob_margins,
    )
    base_oob_by_margin = (
        _oob_metrics_by_margin(
            rows,
            score_key="base_score",
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
            margins=oob_margins,
        )
        if bool(args.compare_base)
        else None
    )

    band_counts = Counter(_clean_text(row.get("band")) for row in rows)
    aspect_counts = Counter(_clean_text(row.get("aspect")) for row in rows)
    pair_type_counts = Counter(_clean_text(row.get("pair_type")) for row in pair_rows)
    elapsed = time.time() - started

    blocks = [
        "mode=ce3_test_eval",
        f"listwise_test={listwise_path}",
        f"pairwise_test={pairwise_path} exists={pairwise_path.exists()}",
        f"finetuned_model={model_ref}",
        f"multihead={use_multihead}",
        f"base_model={base_ref if bool(args.compare_base) else ''}",
        f"posthoc_calibration=False",
        f"device={device}",
        f"rows={len(rows)} aspect_counts={dict(aspect_counts)} band_counts={dict(band_counts)}",
        f"pair_rows={len(pair_rows)} pair_type_counts={dict(pair_type_counts)}",
        f"comparison_oob_margins={','.join(f'{x:.2f}' for x in oob_margins)}",
        _format_comparison_table(
            group="overall",
            teacher_metrics=teacher_metrics,
            finetuned_metrics=finetuned_metrics,
            base_metrics=base_metrics,
            teacher_oob=teacher_oob_by_margin,
            finetuned_oob=finetuned_oob_by_margin,
            base_oob=base_oob_by_margin,
            margins=oob_margins,
        ),
        *[
            _format_comparison_table(
                group=aspect,
                teacher_metrics=teacher_metrics,
                finetuned_metrics=finetuned_metrics,
                base_metrics=base_metrics,
                teacher_oob=teacher_oob_by_margin,
                finetuned_oob=finetuned_oob_by_margin,
                base_oob=base_oob_by_margin,
                margins=oob_margins,
            )
            for aspect in ASPECTS
        ],
        _format_main_table(finetuned_metrics, model_label="finetuned"),
        _format_oob_table(
            finetuned_metrics,
            model_label="finetuned",
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
            oob_margin=oob_margin,
        ),
        _format_raw_table(finetuned_metrics, model_label="finetuned"),
    ]
    if finetuned_pair_metrics:
        blocks.append(
            _format_explicit_pairwise_comparison(
                teacher_pair_metrics=teacher_pair_metrics,
                finetuned_pair_metrics=finetuned_pair_metrics,
                base_pair_metrics=base_pair_metrics,
            )
        )
        blocks.append(_format_pairwise_table(finetuned_pair_metrics, model_label="finetuned"))
    if base_metrics is not None:
        blocks.extend(
            [
                _format_main_table(base_metrics, model_label="base"),
                _format_oob_table(
                    base_metrics,
                    model_label="base",
                    high_threshold=high_threshold,
                    mid_threshold=mid_threshold,
                    oob_margin=oob_margin,
                ),
                _format_raw_table(base_metrics, model_label="base"),
            ]
        )
        if base_pair_metrics:
            blocks.append(_format_pairwise_table(base_pair_metrics, model_label="base"))
    blocks.append(f"elapsed_sec={elapsed:.2f}")
    report = "\n".join(blocks).strip() + "\n"
    print(report)

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = _clean_text(args.save_prefix) or "ce3_test_eval"
    output_json = output_dir / f"{prefix}_{timestamp}.json"
    output_txt = output_dir / f"{prefix}_{timestamp}.txt"
    payload = {
        "meta": {
            "created_at_local": datetime.now().isoformat(),
            "listwise_test": str(listwise_path),
            "pairwise_test": str(pairwise_path),
            "pairwise_test_exists": bool(pairwise_path.exists()),
            "finetuned_model": model_ref,
            "multihead": bool(use_multihead),
            "base_model": base_ref,
            "compare_base": bool(args.compare_base),
            "base_use_long_prefix": bool(args.base_use_long_prefix),
            "posthoc_calibration": False,
            "score_field": _clean_text(args.score_field),
            "batch_size": int(batch_size),
            "max_length": int(max_length),
            "top_k": int(top_k),
            "pair_eps": float(args.pair_eps),
            "hard_gap_max": float(args.hard_gap_max),
            "medium_gap_max": float(args.medium_gap_max),
            "high_threshold": float(high_threshold),
            "mid_threshold": float(mid_threshold),
            "oob_margin": float(oob_margin),
            "oob_margins": [float(x) for x in oob_margins],
            "device": str(device),
            "elapsed_sec": float(elapsed),
            "aspect_counts": dict(aspect_counts),
            "band_counts": dict(band_counts),
            "pair_type_counts": dict(pair_type_counts),
        },
        "teacher": {
            "metrics": teacher_metrics,
            "pairwise_metrics": teacher_pair_metrics,
            "oob_by_margin": teacher_oob_by_margin,
        },
        "finetuned": {
            "metrics": finetuned_metrics,
            "pairwise_metrics": finetuned_pair_metrics,
            "oob_by_margin": finetuned_oob_by_margin,
        },
        "base": {
            "metrics": base_metrics,
            "pairwise_metrics": base_pair_metrics,
            "oob_by_margin": base_oob_by_margin,
        },
    }
    if bool(args.save_rows):
        payload["rows"] = rows
        payload["pair_rows"] = pair_rows
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_txt.write_text(report, encoding="utf-8")
    print(f"saved_json={output_json}")
    print(f"saved_txt={output_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
