from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()

FINETUNED_MODEL_ROOT_DEFAULT = "ce/models/mse_domain_method"
FINETUNED_MODEL_DEFAULT = "/nfs/stak/users/kimwoon/hpc-share/grant-matching-agent/ce/models/bge_reranker_distill__sd42_s15_s25_bs2_ga16_cp48_ml12_lr5em07_lr11p1em06_lr24p5em07_t1p2_kl0p5_pw0p24_mse0p22_cm0p85_cb0p65_dpw1_mpw1p2_dlw1_mlw0p9/stage2_epoch_5"
BASE_MODEL_DEFAULT = "dleemiller/ModernCE-base-sts"
DOMAIN_INPUT_DEFAULT = "ce/dataset/splits/llm_distill_domain_listwise_test.jsonl"
METHOD_INPUT_DEFAULT = "ce/dataset/splits/llm_distill_method_listwise_test.jsonl"
OUTPUT_DIR_DEFAULT = "ce/eval/results"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


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


def _resolve_path(value: str) -> Path:
    p = Path(_clean_text(value)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _is_hf_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    return any((path / n).exists() for n in ("pytorch_model.bin", "model.safetensors", "model.safetensors.index.json"))


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


def _resolve_default_finetuned_model_ref() -> str:
    root = _resolve_path(FINETUNED_MODEL_ROOT_DEFAULT)
    candidates: List[Path] = [root / "best", root / "final", root]
    if root.is_dir():
        for p in root.iterdir():
            if p.is_dir() and p.name.startswith("epoch_"):
                candidates.append(p)
    existing = [p for p in candidates if _is_hf_model_dir(p)]
    if existing:
        existing.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(existing[0])
    return _resolve_model_ref(FINETUNED_MODEL_DEFAULT)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = _clean_text(raw)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL row at {path}:{line_no} ({type(exc).__name__}: {exc})") from exc
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _score_band(score: float, *, high_threshold: float, mid_threshold: float) -> str:
    if score >= float(high_threshold):
        return "high"
    if score >= float(mid_threshold):
        return "mid"
    return "low"


def _load_eval_rows(
    *,
    path: Path,
    aspect: str,
    score_field: str,
    only_selected: bool,
    high_threshold: float,
    mid_threshold: float,
) -> List[Dict[str, Any]]:
    aspect_norm = _clean_text(aspect).lower()
    if aspect_norm not in {"domain", "method"}:
        raise RuntimeError(f"Unsupported aspect: {aspect}")

    prefix = "[DOMAIN]" if aspect_norm == "domain" else "[METHOD]"
    rows: List[Dict[str, Any]] = []
    for obj in _iter_jsonl(path):
        grant_id = _clean_text(obj.get("grant_id"))
        spec_idx = _safe_int(obj.get("spec_idx"), default=0, minimum=0, maximum=1_000_000_000)
        query_raw = _clean_text(obj.get("query_text"))
        if not query_raw:
            continue

        for d in list(obj.get("docs") or []):
            if not isinstance(d, dict):
                continue
            if only_selected and (not bool(d.get("selected_for_target", False))):
                continue
            doc_raw = _clean_text(d.get("text"))
            if not doc_raw:
                continue
            gt = _safe_float(d.get(score_field), default=0.0, minimum=0.0, maximum=1.0)
            band = _score_band(gt, high_threshold=high_threshold, mid_threshold=mid_threshold)
            rows.append(
                {
                    "aspect": aspect_norm,
                    "grant_id": grant_id,
                    "spec_idx": int(spec_idx),
                    "query_key": f"{aspect_norm}::{grant_id}::{spec_idx}",
                    "query_text_raw": query_raw,
                    "doc_text_raw": doc_raw,
                    "query_text_prefixed": f"{prefix} {query_raw}",
                    "doc_text_prefixed": f"{prefix} {doc_raw}",
                    "teacher_score_used": float(gt),
                    "score_band": band,
                }
            )
    return rows


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


def _score_query_doc_rows(
    *,
    model: Any,
    tokenizer: Any,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
    max_length: int,
    query_key_name: str,
    doc_key_name: str,
) -> List[float]:
    if not rows:
        return []

    query_to_indices: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        q = _clean_text(row.get(query_key_name))
        query_to_indices.setdefault(q, []).append(idx)

    out = [0.0] * len(rows)
    for q, idxs in query_to_indices.items():
        docs = [_clean_text(rows[i].get(doc_key_name)) for i in idxs]
        scores = _score_docs_for_query(
            model=model,
            tokenizer=tokenizer,
            query_text=q,
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
        b = _clean_text(row.get("score_band")).lower()
        if b in {"high", "mid", "low"}:
            groups[b].append(row)

    def _avg(vals: Sequence[float]) -> float:
        if not vals:
            return 0.0
        return float(sum(vals) / float(len(vals)))

    out: Dict[str, Dict[str, float]] = {}
    for name, band_rows in groups.items():
        out[name] = {
            "count": float(len(band_rows)),
            "avg_teacher_score": _avg([float(r.get("teacher_score_used") or 0.0) for r in band_rows]),
            "avg_base_abs_margin": _avg([float(r.get("base_abs_margin") or 0.0) for r in band_rows]),
            "avg_finetuned_abs_margin": _avg([float(r.get("finetuned_abs_margin") or 0.0) for r in band_rows]),
        }
        out[name]["avg_abs_margin_gain"] = float(
            out[name]["avg_base_abs_margin"] - out[name]["avg_finetuned_abs_margin"]
        )
    return out


def _group_rows_by_query(rows: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            _clean_text(row.get("aspect")).lower(),
            _clean_text(row.get("grant_id")),
            _safe_int(row.get("spec_idx"), default=0, minimum=0, maximum=1_000_000_000),
        )
        grouped.setdefault(key, []).append(row)
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

        rr = 0.0
        for rank, idx in enumerate(model_sorted[:k_eff], start=1):
            if teacher_vals[idx] >= float(mrr_rel_threshold):
                rr = 1.0 / float(rank)
                break
        mrr_sum += float(rr)
        mrr_count += 1

        relevant = {i for i, y in enumerate(teacher_vals) if y >= float(recall_rel_threshold)}
        if relevant:
            pred_topk = set(model_sorted[:k_eff])
            recall_sum += float(len(pred_topk.intersection(relevant)) / float(len(relevant)))
            recall_count += 1

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
                pred_margin = float(si - sj) if yi > yj else float(sj - si)
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
    oob_margin: float,
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

    margin = max(0.0, float(oob_margin))
    low_upper = min(1.0, float(mid_threshold) + margin)
    mid_lower = max(0.0, float(mid_threshold) - margin)
    mid_upper = min(1.0, float(high_threshold) + margin)
    high_lower = max(0.0, float(high_threshold) - margin)

    low_vals = by_band["low"]
    mid_vals = by_band["mid"]
    high_vals = by_band["high"]

    low_out_count = int(sum(1 for x in low_vals if x >= low_upper))
    mid_out_count = int(sum(1 for x in mid_vals if (x < mid_lower or x >= mid_upper)))
    high_out_count = int(sum(1 for x in high_vals if x < high_lower))

    low_out_rate = float(low_out_count / float(max(1, len(low_vals))))
    mid_out_rate = float(mid_out_count / float(max(1, len(mid_vals))))
    high_out_rate = float(high_out_count / float(max(1, len(high_vals))))
    weak_leak = float(sum(1 for x in low_vals if x >= float(high_threshold)) / float(max(1, len(low_vals))))
    strong_collapse = float(sum(1 for x in high_vals if x < float(mid_threshold)) / float(max(1, len(high_vals))))

    return {
        "avg_pred_high": float(avg_high),
        "avg_pred_mid": float(avg_mid),
        "avg_pred_low": float(avg_low),
        "monotonic_high_mid_low": bool(monotonic),
        "low_out_count": int(low_out_count),
        "mid_out_count": int(mid_out_count),
        "high_out_count": int(high_out_count),
        "low_out_rate": float(low_out_rate),
        "mid_out_rate": float(mid_out_rate),
        "high_out_rate": float(high_out_rate),
        "weak_leak_rate": float(weak_leak),
        "strong_collapse_rate": float(strong_collapse),
        "count_high": int(len(high_vals)),
        "count_mid": int(len(mid_vals)),
        "count_low": int(len(low_vals)),
        "oob_margin": float(margin),
    }


def _format_order_summary_table(*, finetuned: Dict[str, Any], plain: Dict[str, Any]) -> str:
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
    lines.append(_row("plain", plain))
    return "\n".join(lines)


def _format_margin_summary_table(stats: Dict[str, Dict[str, float]]) -> str:
    order = ["overall", "high", "mid", "low"]
    lines: List[str] = []
    lines.append("")
    lines.append("=== Margin Summary (vs Teacher Ground Truth) ===")
    lines.append(f"{'BAND':<8} {'COUNT':>8} {'AVG_GT':>10} {'PLAIN_MAE':>10} {'FINETUNED_MAE':>14} {'GAIN':>10}")
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


def _format_raw_sanity_table(
    *, ground_truth: Dict[str, Any], finetuned: Dict[str, Any], plain: Dict[str, Any]
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

    lines.append(_row("ground_truth", ground_truth))
    lines.append(_row("finetuned", finetuned))
    lines.append(_row("plain", plain))
    return "\n".join(lines)


def _format_out_of_band_table(
    *,
    ground_truth: Dict[str, Any],
    finetuned: Dict[str, Any],
    plain: Dict[str, Any],
    high_threshold: float,
    mid_threshold: float,
    oob_margin: float,
) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append("=== Out-Of-Band Summary ===")
    margin = max(0.0, float(oob_margin))
    lines.append(
        f"thresholds: high>={float(high_threshold):.2f}, mid>={float(mid_threshold):.2f}, oob_margin={margin:.2f}"
    )
    lines.append(f"{'MODEL':<12} {'LOW_OUT':>21} {'MID_OUT':>21} {'HIGH_OUT':>21}")
    lines.append("-" * 78)

    def _cell(obj: Dict[str, Any], *, out_key: str, count_key: str, rate_key: str) -> str:
        out_n = int(obj.get(out_key) or 0)
        total_n = int(obj.get(count_key) or 0)
        rate = float(obj.get(rate_key) or 0.0) * 100.0
        return f"{out_n}/{total_n} ({rate:.2f}%)"

    def _row(label: str, obj: Dict[str, Any]) -> str:
        low_cell = _cell(obj, out_key="low_out_count", count_key="count_low", rate_key="low_out_rate")
        mid_cell = _cell(obj, out_key="mid_out_count", count_key="count_mid", rate_key="mid_out_rate")
        high_cell = _cell(obj, out_key="high_out_count", count_key="count_high", rate_key="high_out_rate")
        return f"{label:<12} {low_cell:>21} {mid_cell:>21} {high_cell:>21}"

    lines.append(_row("ground_truth", ground_truth))
    lines.append(_row("finetuned", finetuned))
    lines.append(_row("plain", plain))
    return "\n".join(lines)


def _compute_metric_bundle(
    *,
    rows: Sequence[Dict[str, Any]],
    order_top_k: int,
    pair_eps: float,
    hard_gap_max: float,
    medium_gap_max: float,
    high_threshold: float,
    mid_threshold: float,
    oob_margin: float,
) -> Dict[str, Any]:
    stats = _compute_margin_stats(rows)
    order_finetuned = _compute_order_metrics_for_model(
        rows=rows,
        model_score_key="finetuned_score",
        top_k=order_top_k,
        pair_eps=pair_eps,
        hard_gap_max=hard_gap_max,
        medium_gap_max=medium_gap_max,
        mrr_rel_threshold=high_threshold,
        recall_rel_threshold=mid_threshold,
    )
    order_plain = _compute_order_metrics_for_model(
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
        oob_margin=oob_margin,
    )
    raw_sanity_plain = _compute_raw_score_sanity(
        rows=rows,
        model_score_key="base_score",
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
        oob_margin=oob_margin,
    )
    raw_sanity_ground_truth = _compute_raw_score_sanity(
        rows=rows,
        model_score_key="teacher_score_used",
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
        oob_margin=oob_margin,
    )
    return {
        "row_count": int(len(rows)),
        "order_metrics": {"finetuned": order_finetuned, "plain": order_plain},
        "margin_stats": stats,
        "raw_score_sanity": {
            "ground_truth": raw_sanity_ground_truth,
            "finetuned": raw_sanity_finetuned,
            "plain": raw_sanity_plain,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate CE finetuned vs plain STS base on domain+method listwise files.")
    p.add_argument("--finetuned-model", type=str, default=FINETUNED_MODEL_DEFAULT)
    p.add_argument("--auto-resolve-finetuned", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--base-model", type=str, default=BASE_MODEL_DEFAULT)
    p.add_argument("--domain-input", type=str, default=DOMAIN_INPUT_DEFAULT)
    p.add_argument("--method-input", type=str, default=METHOD_INPUT_DEFAULT)
    p.add_argument("--score-field", type=str, default="teacher_score_raw", choices=["teacher_score_raw", "teacher_score"])
    p.add_argument("--only-selected", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--high-threshold", type=float, default=0.70)
    p.add_argument("--mid-threshold", type=float, default=0.30)
    p.add_argument("--oob-margin", type=float, default=0.0)
    p.add_argument("--order-top-k", type=int, default=5)
    p.add_argument("--pair-eps", type=float, default=0.01)
    p.add_argument("--hard-gap-max", type=float, default=0.15)
    p.add_argument("--medium-gap-max", type=float, default=0.40)
    p.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--print", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--save-prefix", type=str, default="ce_distill_margin_compare")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    batch_size = _safe_int(args.batch_size, default=32, minimum=1, maximum=4096)
    max_length = _safe_int(args.max_length, default=512, minimum=64, maximum=4096)
    high_threshold = _safe_float(args.high_threshold, default=0.70, minimum=0.0, maximum=1.0)
    mid_threshold = _safe_float(args.mid_threshold, default=0.30, minimum=0.0, maximum=1.0)
    oob_margin = _safe_float(args.oob_margin, default=0.0, minimum=0.0, maximum=1.0)
    if mid_threshold > high_threshold:
        mid_threshold = high_threshold
    order_top_k = _safe_int(args.order_top_k, default=5, minimum=1, maximum=100)
    pair_eps = _safe_float(args.pair_eps, default=0.01, minimum=0.0, maximum=1.0)
    hard_gap_max = _safe_float(args.hard_gap_max, default=0.15, minimum=0.0, maximum=1.0)
    medium_gap_max = _safe_float(args.medium_gap_max, default=0.40, minimum=0.0, maximum=1.0)
    if medium_gap_max < hard_gap_max:
        medium_gap_max = hard_gap_max

    if bool(args.auto_resolve_finetuned):
        finetuned_ref = _resolve_default_finetuned_model_ref()
    else:
        finetuned_ref = _resolve_model_ref(_clean_text(args.finetuned_model) or FINETUNED_MODEL_DEFAULT)
    base_ref = _resolve_model_ref(_clean_text(args.base_model) or BASE_MODEL_DEFAULT)
    domain_path = _resolve_path(args.domain_input)
    method_path = _resolve_path(args.method_input)

    if not domain_path.exists():
        raise RuntimeError(f"Domain input not found: {domain_path}")
    if not method_path.exists():
        raise RuntimeError(f"Method input not found: {method_path}")

    rows_domain = _load_eval_rows(
        path=domain_path,
        aspect="domain",
        score_field=args.score_field,
        only_selected=bool(args.only_selected),
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
    )
    rows_method = _load_eval_rows(
        path=method_path,
        aspect="method",
        score_field=args.score_field,
        only_selected=bool(args.only_selected),
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
    )
    rows = rows_domain + rows_method
    if not rows:
        raise RuntimeError("No rows loaded for evaluation.")

    device = _pick_device()
    tok_finetuned = AutoTokenizer.from_pretrained(finetuned_ref, trust_remote_code=True)
    model_finetuned = AutoModelForSequenceClassification.from_pretrained(finetuned_ref, num_labels=1, trust_remote_code=True).to(device).eval()
    tok_base = AutoTokenizer.from_pretrained(base_ref, trust_remote_code=True)
    model_base = AutoModelForSequenceClassification.from_pretrained(base_ref, num_labels=1, trust_remote_code=True).to(device).eval()

    started = time.time()
    finetuned_scores = _score_query_doc_rows(
        model=model_finetuned,
        tokenizer=tok_finetuned,
        rows=rows,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        query_key_name="query_text_prefixed",
        doc_key_name="doc_text_prefixed",
    )
    base_scores = _score_query_doc_rows(
        model=model_base,
        tokenizer=tok_base,
        rows=rows,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        query_key_name="query_text_raw",
        doc_key_name="doc_text_raw",
    )
    elapsed = max(1e-9, time.time() - started)

    for i, row in enumerate(rows):
        gt = float(row.get("teacher_score_used") or 0.0)
        ft = float(finetuned_scores[i])
        b = float(base_scores[i])
        row["finetuned_score"] = float(ft)
        row["base_score"] = float(b)
        row["finetuned_margin"] = float(ft - gt)
        row["base_margin"] = float(b - gt)
        row["finetuned_abs_margin"] = abs(float(ft - gt))
        row["base_abs_margin"] = abs(float(b - gt))
        row["abs_margin_gain"] = float(row["base_abs_margin"] - row["finetuned_abs_margin"])

    overall_bundle = _compute_metric_bundle(
        rows=rows,
        order_top_k=order_top_k,
        pair_eps=pair_eps,
        hard_gap_max=hard_gap_max,
        medium_gap_max=medium_gap_max,
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
        oob_margin=oob_margin,
    )
    domain_bundle = _compute_metric_bundle(
        rows=rows_domain,
        order_top_k=order_top_k,
        pair_eps=pair_eps,
        hard_gap_max=hard_gap_max,
        medium_gap_max=medium_gap_max,
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
        oob_margin=oob_margin,
    )
    method_bundle = _compute_metric_bundle(
        rows=rows_method,
        order_top_k=order_top_k,
        pair_eps=pair_eps,
        hard_gap_max=hard_gap_max,
        medium_gap_max=medium_gap_max,
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
        oob_margin=oob_margin,
    )

    by_band = {
        "high": int(sum(1 for r in rows if _clean_text(r.get("score_band")) == "high")),
        "mid": int(sum(1 for r in rows if _clean_text(r.get("score_band")) == "mid")),
        "low": int(sum(1 for r in rows if _clean_text(r.get("score_band")) == "low")),
    }

    meta_header = (
        f"mode=distill\n"
        f"domain_input={domain_path}\n"
        f"method_input={method_path}\n"
        f"score_field={_clean_text(args.score_field)}\n"
        f"only_selected={bool(args.only_selected)}\n"
        f"high_threshold={high_threshold} mid_threshold={mid_threshold}\n"
        f"oob_margin={oob_margin}\n"
        f"selected_high={by_band['high']} selected_mid={by_band['mid']} selected_low={by_band['low']}\n"
        f"rows_domain={len(rows_domain)} rows_method={len(rows_method)} rows_total={len(rows)}\n"
        f"order_top_k={order_top_k} pair_eps={pair_eps} hard_gap_max={hard_gap_max} medium_gap_max={medium_gap_max}\n"
        f"finetuned_model={finetuned_ref}\n"
        f"base_model={base_ref}\n"
        f"device={device}\n"
    )

    blocks: List[str] = [
        meta_header,
        "=== OVERALL (DOMAIN + METHOD) ===",
        _format_order_summary_table(
            finetuned=overall_bundle["order_metrics"]["finetuned"],
            plain=overall_bundle["order_metrics"]["plain"],
        ),
        _format_margin_summary_table(overall_bundle["margin_stats"]),
        _format_raw_sanity_table(
            ground_truth=overall_bundle["raw_score_sanity"]["ground_truth"],
            finetuned=overall_bundle["raw_score_sanity"]["finetuned"],
            plain=overall_bundle["raw_score_sanity"]["plain"],
        ),
        _format_out_of_band_table(
            ground_truth=overall_bundle["raw_score_sanity"]["ground_truth"],
            finetuned=overall_bundle["raw_score_sanity"]["finetuned"],
            plain=overall_bundle["raw_score_sanity"]["plain"],
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
            oob_margin=oob_margin,
        ),
        "",
        "=== DOMAIN ONLY ===",
        _format_order_summary_table(
            finetuned=domain_bundle["order_metrics"]["finetuned"],
            plain=domain_bundle["order_metrics"]["plain"],
        ),
        _format_margin_summary_table(domain_bundle["margin_stats"]),
        _format_raw_sanity_table(
            ground_truth=domain_bundle["raw_score_sanity"]["ground_truth"],
            finetuned=domain_bundle["raw_score_sanity"]["finetuned"],
            plain=domain_bundle["raw_score_sanity"]["plain"],
        ),
        _format_out_of_band_table(
            ground_truth=domain_bundle["raw_score_sanity"]["ground_truth"],
            finetuned=domain_bundle["raw_score_sanity"]["finetuned"],
            plain=domain_bundle["raw_score_sanity"]["plain"],
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
            oob_margin=oob_margin,
        ),
        "",
        "=== METHOD ONLY ===",
        _format_order_summary_table(
            finetuned=method_bundle["order_metrics"]["finetuned"],
            plain=method_bundle["order_metrics"]["plain"],
        ),
        _format_margin_summary_table(method_bundle["margin_stats"]),
        _format_raw_sanity_table(
            ground_truth=method_bundle["raw_score_sanity"]["ground_truth"],
            finetuned=method_bundle["raw_score_sanity"]["finetuned"],
            plain=method_bundle["raw_score_sanity"]["plain"],
        ),
        _format_out_of_band_table(
            ground_truth=method_bundle["raw_score_sanity"]["ground_truth"],
            finetuned=method_bundle["raw_score_sanity"]["finetuned"],
            plain=method_bundle["raw_score_sanity"]["plain"],
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
            oob_margin=oob_margin,
        ),
        "",
        f"elapsed_sec={elapsed:.2f}",
        f"rows_total={len(rows)}",
    ]
    report_text = "\n".join(blocks).strip() + "\n"
    if bool(args.print):
        print(report_text)

    if bool(args.save):
        out_dir = _resolve_path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = _clean_text(args.save_prefix) or "ce_distill_margin_compare"
        output_json = out_dir / f"{prefix}_{ts}.json"
        output_txt = out_dir / f"{prefix}_{ts}.txt"
        payload = {
            "meta": {
                "created_at_local": datetime.now().isoformat(),
                "mode": "distill",
                "domain_input": str(domain_path),
                "method_input": str(method_path),
                "score_field": _clean_text(args.score_field),
                "only_selected": bool(args.only_selected),
                "high_threshold": float(high_threshold),
                "mid_threshold": float(mid_threshold),
                "oob_margin": float(oob_margin),
                "order_top_k": int(order_top_k),
                "pair_eps": float(pair_eps),
                "hard_gap_max": float(hard_gap_max),
                "medium_gap_max": float(medium_gap_max),
                "finetuned_model": finetuned_ref,
                "base_model": base_ref,
                "device": str(device),
                "batch_size": int(batch_size),
                "max_length": int(max_length),
                "elapsed_sec": float(elapsed),
            },
            "order_metrics": overall_bundle["order_metrics"],
            "margin_stats": overall_bundle["margin_stats"],
            "raw_score_sanity": overall_bundle["raw_score_sanity"],
            "by_aspect": {
                "domain": domain_bundle,
                "method": method_bundle,
            },
            "rows": rows,
        }
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        output_txt.write_text(report_text, encoding="utf-8")
        print(f"saved_json={output_json}")
        print(f"saved_txt={output_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
