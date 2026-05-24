from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoTokenizer


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce.aspect_modeling import (  # noqa: E402
    aspect_id_from_name,
    clean_aspect_condition_mode,
    format_aspect_pair,
    load_sequence_classifier_model,
    model_logits,
)

INPUT_DEFAULT = "ce/dataset/distill/llm_ground_truth_requirement_common_test_listwise.jsonl"
FINETUNED_MODEL_DEFAULT = "/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/ce/models/stage2_epoch_6"
BASE_MODEL_DEFAULT = "dleemiller/ModernCE-base-sts"
OUTPUT_DIR_DEFAULT = "ce/eval/results"
HIGH_THRESHOLD_DEFAULT = 0.70
MID_THRESHOLD_DEFAULT = 0.30


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


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
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
                yield obj


def _score_band(score: float, *, high_threshold: float, mid_threshold: float) -> str:
    s = float(score)
    if s >= float(high_threshold):
        return "high"
    if s >= float(mid_threshold):
        return "mid"
    return "low"


def _load_ground_truth_rows(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row_idx, obj in enumerate(_iter_jsonl(path), start=1):
        query = _clean_text(obj.get("query") or obj.get("query_text"))
        doc = _clean_text(obj.get("doc") or obj.get("text"))
        if not query or not doc:
            continue

        y = _safe_float(
            obj.get("score", obj.get("teacher_score_raw", obj.get("teacher_score", 0.0))),
            default=0.0,
            minimum=0.0,
            maximum=1.0,
        )

        grant_id = _clean_text(obj.get("grant_id"))
        spec_idx = _safe_int(obj.get("spec_idx"), default=-1, minimum=-1, maximum=1_000_000_000)
        if grant_id and spec_idx >= 0:
            query_key = f"{grant_id}::{spec_idx}"
        else:
            # Fallback for compact ground-truth files with only query/doc/score.
            query_key = f"query::{query}"

        out.append(
            {
                "row_idx": int(row_idx),
                "query": query,
                "doc": doc,
                "gt_score": float(y),
                "query_key": query_key,
                "grant_id": grant_id,
                "spec_idx": int(spec_idx),
            }
        )
    return out


def _score_pairs(
    *,
    model: Any,
    tokenizer: Any,
    queries: Sequence[str],
    docs: Sequence[str],
    aspect: str,
    aspect_condition_mode: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> List[float]:
    if len(queries) != len(docs):
        raise RuntimeError("queries/docs length mismatch")
    out: List[float] = []
    step = max(1, int(batch_size))
    with torch.no_grad():
        for i in range(0, len(queries), step):
            formatted = [
                format_aspect_pair(
                    q,
                    d,
                    aspect_condition_mode=aspect_condition_mode,
                )
                for q, d in zip(queries[i : i + step], docs[i : i + step])
            ]
            q_chunk = [p[0] for p in formatted]
            d_chunk = [p[1] for p in formatted]
            enc = tokenizer(
                q_chunk,
                d_chunk,
                max_length=int(max_length),
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = _to_device(enc, device)
            aspect_ids = torch.full(
                (len(q_chunk),),
                int(aspect_id_from_name(aspect)),
                dtype=torch.int64,
                device=device,
            )
            logits = model_logits(model, enc, aspect_ids=aspect_ids).squeeze(-1)
            probs = torch.sigmoid(logits)
            out.extend(float(x) for x in probs.detach().cpu().tolist())
    return out


def _apply_weighted_scores(
    *,
    rows: List[Dict[str, Any]],
    dom_scores: Sequence[float],
    meth_scores: Sequence[float],
    constraint_scores: Sequence[float],
    out_prefix: str,
    domain_weight: float,
    method_weight: float,
    constraint_weight: float,
) -> None:
    for i, row in enumerate(rows):
        d = float(dom_scores[i])
        m = float(meth_scores[i])
        c = float(constraint_scores[i])
        blended = float(domain_weight * d + method_weight * m + constraint_weight * c)
        row[f"{out_prefix}_domain"] = d
        row[f"{out_prefix}_method"] = m
        row[f"{out_prefix}_constraint"] = c
        row[f"{out_prefix}_weighted"] = blended


def _apply_three_way_scores(
    *,
    rows: List[Dict[str, Any]],
    out_prefix: str,
    domain_weight: float,
    method_weight: float,
    requirement_weight: float,
    out_key: str,
) -> None:
    denom = float(domain_weight + method_weight + requirement_weight)
    if denom <= 0.0:
        domain_weight, method_weight, requirement_weight = 0.25, 0.25, 0.50
        denom = 1.0
    d_w = float(domain_weight / denom)
    m_w = float(method_weight / denom)
    r_w = float(requirement_weight / denom)

    for row in rows:
        d = float(row.get(f"{out_prefix}_domain") or 0.0)
        m = float(row.get(f"{out_prefix}_method") or 0.0)
        r = float(row.get("requirement_score") or 0.0)
        row[out_key] = float(d_w * d + m_w * m + r_w * r)


def _compute_regression(rows: Sequence[Dict[str, Any]], pred_key: str) -> Dict[str, float]:
    ys = [float(r.get("gt_score") or 0.0) for r in rows]
    ps = [float(r.get(pred_key) or 0.0) for r in rows]
    n = len(ys)
    if n <= 0:
        return {"count": 0.0, "mae": 0.0, "rmse": 0.0, "pearson": 0.0}

    abs_err = [abs(p - y) for p, y in zip(ps, ys)]
    sq_err = [(p - y) ** 2 for p, y in zip(ps, ys)]
    mae = float(sum(abs_err) / float(n))
    rmse = float(math.sqrt(sum(sq_err) / float(n)))

    y_mean = float(sum(ys) / float(n))
    p_mean = float(sum(ps) / float(n))
    y_var = float(sum((y - y_mean) ** 2 for y in ys) / float(n))
    p_var = float(sum((p - p_mean) ** 2 for p in ps) / float(n))
    if y_var <= 1e-12 or p_var <= 1e-12:
        pearson = 0.0
    else:
        cov = float(sum((y - y_mean) * (p - p_mean) for y, p in zip(ys, ps)) / float(n))
        pearson = float(cov / math.sqrt(y_var * p_var))

    return {
        "count": float(n),
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
    }


def _compute_band_mae(
    *, rows: Sequence[Dict[str, Any]], pred_key: str, high_threshold: float, mid_threshold: float
) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, Any]]] = {"overall": list(rows), "high": [], "mid": [], "low": []}
    for r in rows:
        b = _score_band(float(r.get("gt_score") or 0.0), high_threshold=high_threshold, mid_threshold=mid_threshold)
        groups[b].append(r)

    out: Dict[str, Dict[str, float]] = {}
    for k, g in groups.items():
        if not g:
            out[k] = {"count": 0.0, "avg_gt": 0.0, "avg_pred": 0.0, "mae": 0.0}
            continue
        ys = [float(r.get("gt_score") or 0.0) for r in g]
        ps = [float(r.get(pred_key) or 0.0) for r in g]
        mae = float(sum(abs(p - y) for p, y in zip(ps, ys)) / float(len(g)))
        out[k] = {
            "count": float(len(g)),
            "avg_gt": float(sum(ys) / float(len(g))),
            "avg_pred": float(sum(ps) / float(len(g))),
            "mae": mae,
        }
    return out


def _compute_oob(
    *, rows: Sequence[Dict[str, Any]], pred_key: str, high_threshold: float, mid_threshold: float
) -> Dict[str, float]:
    low_total = 0
    mid_total = 0
    high_total = 0
    low_out = 0
    mid_out = 0
    mid_low_out = 0
    mid_high_out = 0
    high_out = 0

    for r in rows:
        y = float(r.get("gt_score") or 0.0)
        p = float(r.get(pred_key) or 0.0)
        b = _score_band(y, high_threshold=high_threshold, mid_threshold=mid_threshold)
        if b == "low":
            low_total += 1
            if p >= float(mid_threshold):
                low_out += 1
        elif b == "mid":
            mid_total += 1
            if p < float(mid_threshold):
                mid_low_out += 1
            if p >= float(high_threshold):
                mid_high_out += 1
            if p < float(mid_threshold) or p >= float(high_threshold):
                mid_out += 1
        else:
            high_total += 1
            if p < float(high_threshold):
                high_out += 1

    return {
        "low_out": float(low_out),
        "low_total": float(low_total),
        "low_rate": float(low_out / float(max(1, low_total))),
        "mid_out": float(mid_out),
        "mid_total": float(mid_total),
        "mid_rate": float(mid_out / float(max(1, mid_total))),
        "mid_low_out": float(mid_low_out),
        "mid_low_rate": float(mid_low_out / float(max(1, mid_total))),
        "mid_high_out": float(mid_high_out),
        "mid_high_rate": float(mid_high_out / float(max(1, mid_total))),
        "high_out": float(high_out),
        "high_total": float(high_total),
        "high_rate": float(high_out / float(max(1, high_total))),
    }


def _format_oob_counts(
    *,
    rows: Sequence[Dict[str, Any]],
    pred_keys: Sequence[Tuple[str, str]],
    high_threshold: float,
    mid_threshold: float,
) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append("=== Out-Of-Band Summary ===")
    lines.append(f"thresholds: high>={high_threshold:.2f}, mid>={mid_threshold:.2f}")
    lines.append(
        f"{'SCORE':<14} {'LOW_OUT':>20} {'MID_OUT':>20} {'MID_LOW':>20} {'MID_HIGH':>20} {'HIGH_OUT':>20}"
    )
    lines.append("-" * 118)

    def _cell(n: float, d: float) -> str:
        ni = int(n)
        di = int(d)
        pct = float(ni / float(max(1, di))) * 100.0
        return f"{ni}/{di} ({pct:.2f}%)"

    for label, key in pred_keys:
        oob = _compute_oob(rows=rows, pred_key=key, high_threshold=high_threshold, mid_threshold=mid_threshold)
        lines.append(
            f"{label:<14} "
            f"{_cell(oob.get('low_out') or 0.0, oob.get('low_total') or 0.0):>20} "
            f"{_cell(oob.get('mid_out') or 0.0, oob.get('mid_total') or 0.0):>20} "
            f"{_cell(oob.get('mid_low_out') or 0.0, oob.get('mid_total') or 0.0):>20} "
            f"{_cell(oob.get('mid_high_out') or 0.0, oob.get('mid_total') or 0.0):>20} "
            f"{_cell(oob.get('high_out') or 0.0, oob.get('high_total') or 0.0):>20}"
        )
    return "\n".join(lines)


def _compute_order(
    *, rows: Sequence[Dict[str, Any]], pred_key: str, pair_eps: float, hard_gap_max: float
) -> Dict[str, float]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(_clean_text(r.get("query_key")), []).append(r)

    q_total = 0
    q_2plus = 0
    top1_ok = 0
    pair_total = 0
    pair_ok = 0
    pair_margin_sum = 0.0
    hard_total = 0
    hard_ok = 0
    hard_margin_sum = 0.0

    eps = max(0.0, float(pair_eps))
    hard_gap = max(eps, float(hard_gap_max))

    for _, g in grouped.items():
        q_total += 1
        n = len(g)
        if n < 2:
            continue
        q_2plus += 1

        ys = [float(r.get("gt_score") or 0.0) for r in g]
        ps = [float(r.get(pred_key) or 0.0) for r in g]

        t_sorted = sorted(range(n), key=lambda i: ys[i], reverse=True)
        p_sorted = sorted(range(n), key=lambda i: ps[i], reverse=True)

        t_top = ys[t_sorted[0]]
        t_top_set = {i for i, y in enumerate(ys) if abs(y - t_top) <= eps}
        if p_sorted[0] in t_top_set:
            top1_ok += 1

        for i in range(n):
            for j in range(i + 1, n):
                gap = abs(ys[i] - ys[j])
                if gap <= eps:
                    continue
                margin = float(ps[i] - ps[j]) if ys[i] > ys[j] else float(ps[j] - ps[i])
                ok = margin > 0.0

                pair_total += 1
                pair_margin_sum += float(margin)
                if ok:
                    pair_ok += 1

                if gap <= hard_gap:
                    hard_total += 1
                    hard_margin_sum += float(margin)
                    if ok:
                        hard_ok += 1

    return {
        "queries_total": float(q_total),
        "queries_with_2plus": float(q_2plus),
        "top1_acc": float(top1_ok / float(max(1, q_2plus))),
        "pair_acc": float(pair_ok / float(max(1, pair_total))),
        "mean_margin": float(pair_margin_sum / float(max(1, pair_total))),
        "hard_acc": float(hard_ok / float(max(1, hard_total))),
        "hard_margin": float(hard_margin_sum / float(max(1, hard_total))),
        "pair_count": float(pair_total),
        "hard_pair_count": float(hard_total),
    }


def _split_rows_by_query_group(
    *,
    rows: Sequence[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(_clean_text(r.get("query_key")), []).append(r)

    keys = [k for k in groups.keys() if k]
    if not keys:
        return list(rows), [], {"queries_total": 0, "val_queries": 0, "test_queries": 0}

    rng = random.Random(int(seed))
    rng.shuffle(keys)

    if len(keys) <= 1:
        return list(rows), list(rows), {
            "queries_total": int(len(keys)),
            "val_queries": int(len(keys)),
            "test_queries": int(len(keys)),
        }

    ratio = max(0.05, min(0.95, float(val_ratio)))
    n_val = int(round(float(len(keys)) * ratio))
    n_val = max(1, min(len(keys) - 1, n_val))
    val_keys = set(keys[:n_val])

    val_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    for r in rows:
        if _clean_text(r.get("query_key")) in val_keys:
            val_rows.append(r)
        else:
            test_rows.append(r)

    return val_rows, test_rows, {
        "queries_total": int(len(keys)),
        "val_queries": int(len(val_keys)),
        "test_queries": int(len(keys) - len(val_keys)),
        "val_rows": int(len(val_rows)),
        "test_rows": int(len(test_rows)),
    }


def _set_alpha_weighted(rows: Sequence[Dict[str, Any]], *, prefix: str, alpha: float, out_key: str) -> None:
    a = float(max(0.0, min(1.0, alpha)))
    b = float(1.0 - a)
    d_key = f"{prefix}_domain"
    m_key = f"{prefix}_method"
    for r in rows:
        d = float(r.get(d_key) or 0.0)
        m = float(r.get(m_key) or 0.0)
        r[out_key] = float(a * d + b * m)


def _set_three_way_weighted(
    rows: Sequence[Dict[str, Any]],
    *,
    prefix: str,
    domain_weight: float,
    method_weight: float,
    requirement_weight: float,
    out_key: str,
) -> None:
    denom = float(domain_weight + method_weight + requirement_weight)
    if denom <= 0.0:
        domain_weight, method_weight, requirement_weight = 0.25, 0.25, 0.50
        denom = 1.0
    d_w = float(domain_weight / denom)
    m_w = float(method_weight / denom)
    r_w = float(requirement_weight / denom)

    d_key = f"{prefix}_domain"
    m_key = f"{prefix}_method"
    r_key = "requirement_score"
    for r in rows:
        d = float(r.get(d_key) or 0.0)
        m = float(r.get(m_key) or 0.0)
        req = float(r.get(r_key) or 0.0)
        r[out_key] = float(d_w * d + m_w * m + r_w * req)


def _weighted_oob_objective(
    *,
    oob: Dict[str, float],
    w_high: float,
    w_mid: float,
    w_low: float,
) -> float:
    return float(
        float(w_high) * float(oob.get("high_rate") or 0.0)
        + float(w_mid) * float(oob.get("mid_rate") or 0.0)
        + float(w_low) * float(oob.get("low_rate") or 0.0)
    )


def _alpha_objective(
    *,
    rows: Sequence[Dict[str, Any]],
    pred_key: str,
    objective: str,
    high_threshold: float,
    mid_threshold: float,
    oob_high_weight: float,
    oob_mid_weight: float,
    oob_low_weight: float,
    hybrid_oob_scale: float,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    reg = _compute_regression(rows, pred_key)
    oob = _compute_oob(rows=rows, pred_key=pred_key, high_threshold=high_threshold, mid_threshold=mid_threshold)
    mae = float(reg.get("mae") or 0.0)
    oob_obj = _weighted_oob_objective(
        oob=oob,
        w_high=oob_high_weight,
        w_mid=oob_mid_weight,
        w_low=oob_low_weight,
    )

    obj_mode = _clean_text(objective).lower()
    if obj_mode == "oob":
        score = float(oob_obj)
    elif obj_mode == "hybrid":
        score = float(mae + float(hybrid_oob_scale) * oob_obj)
    else:
        score = float(mae)

    return score, reg, oob


def _make_alpha_grid(alpha_min: float, alpha_max: float, alpha_step: float) -> List[float]:
    lo = float(max(0.0, min(1.0, alpha_min)))
    hi = float(max(0.0, min(1.0, alpha_max)))
    if hi < lo:
        lo, hi = hi, lo
    step = float(max(1e-4, alpha_step))

    vals: List[float] = []
    x = lo
    while x <= hi + 1e-9:
        vals.append(float(round(x, 6)))
        x += step
    if not vals:
        vals = [0.5]
    if 0.5 >= lo and 0.5 <= hi and all(abs(v - 0.5) > 1e-9 for v in vals):
        vals.append(0.5)
        vals.sort()
    return vals


def _make_three_way_grid(step: float) -> List[Tuple[float, float, float]]:
    step = float(max(1e-4, min(1.0, step)))
    n = int(round(1.0 / step))
    n = max(1, n)
    vals: List[Tuple[float, float, float]] = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            d = float(i / n)
            m = float(j / n)
            r = float(k / n)
            vals.append((d, m, r))

    default = (0.25, 0.25, 0.50)
    if all(sum(abs(a - b) for a, b in zip(x, default)) > 1e-9 for x in vals):
        vals.append(default)
    return vals


def _search_alpha(
    *,
    rows_val: Sequence[Dict[str, Any]],
    prefix: str,
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    objective: str,
    high_threshold: float,
    mid_threshold: float,
    oob_high_weight: float,
    oob_mid_weight: float,
    oob_low_weight: float,
    hybrid_oob_scale: float,
) -> Dict[str, Any]:
    work = [dict(r) for r in rows_val]
    grid = _make_alpha_grid(alpha_min, alpha_max, alpha_step)
    trial_out: List[Dict[str, Any]] = []

    best_alpha = 0.5
    best_obj = float("inf")
    best_mae = 0.0
    best_oob_obj = 0.0
    best_oob_rates = {
        "high_rate": 0.0,
        "mid_rate": 0.0,
        "mid_low_rate": 0.0,
        "mid_high_rate": 0.0,
        "low_rate": 0.0,
    }

    for a in grid:
        key = "__alpha_eval__"
        _set_alpha_weighted(work, prefix=prefix, alpha=float(a), out_key=key)
        obj, reg, oob = _alpha_objective(
            rows=work,
            pred_key=key,
            objective=objective,
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
            oob_high_weight=oob_high_weight,
            oob_mid_weight=oob_mid_weight,
            oob_low_weight=oob_low_weight,
            hybrid_oob_scale=hybrid_oob_scale,
        )
        oob_obj = _weighted_oob_objective(
            oob=oob,
            w_high=oob_high_weight,
            w_mid=oob_mid_weight,
            w_low=oob_low_weight,
        )
        mae = float(reg.get("mae") or 0.0)
        trial_out.append(
            {
                "alpha": float(a),
                "objective": float(obj),
                "mae": float(mae),
                "oob_objective": float(oob_obj),
                "oob_high_rate": float(oob.get("high_rate") or 0.0),
                "oob_mid_rate": float(oob.get("mid_rate") or 0.0),
                "oob_mid_low_rate": float(oob.get("mid_low_rate") or 0.0),
                "oob_mid_high_rate": float(oob.get("mid_high_rate") or 0.0),
                "oob_low_rate": float(oob.get("low_rate") or 0.0),
            }
        )
        better = False
        if obj < best_obj - 1e-12:
            better = True
        elif abs(obj - best_obj) <= 1e-12:
            # tie-break toward conservative center weight
            if abs(a - 0.5) < abs(best_alpha - 0.5):
                better = True
        if better:
            best_obj = float(obj)
            best_alpha = float(a)
            best_mae = float(mae)
            best_oob_obj = float(oob_obj)
            best_oob_rates = {
                "high_rate": float(oob.get("high_rate") or 0.0),
                "mid_rate": float(oob.get("mid_rate") or 0.0),
                "mid_low_rate": float(oob.get("mid_low_rate") or 0.0),
                "mid_high_rate": float(oob.get("mid_high_rate") or 0.0),
                "low_rate": float(oob.get("low_rate") or 0.0),
            }

    trial_out.sort(key=lambda x: (float(x.get("objective") or 0.0), abs(float(x.get("alpha") or 0.5) - 0.5)))
    return {
        "best_alpha": float(best_alpha),
        "best_objective": float(best_obj),
        "best_mae": float(best_mae),
        "best_oob_objective": float(best_oob_obj),
        "best_oob_rates": best_oob_rates,
        "trials_top10": trial_out[:10],
        "grid_count": int(len(grid)),
    }


def _search_three_way(
    *,
    rows_val: Sequence[Dict[str, Any]],
    prefix: str,
    step: float,
    objective: str,
    high_threshold: float,
    mid_threshold: float,
    oob_high_weight: float,
    oob_mid_weight: float,
    oob_low_weight: float,
    hybrid_oob_scale: float,
) -> Dict[str, Any]:
    work = [dict(r) for r in rows_val]
    grid = _make_three_way_grid(step)
    trial_out: List[Dict[str, Any]] = []

    best_weights = (0.25, 0.25, 0.50)
    best_obj = float("inf")
    best_mae = 0.0
    best_oob_obj = 0.0
    best_oob_rates = {
        "high_rate": 0.0,
        "mid_rate": 0.0,
        "mid_low_rate": 0.0,
        "mid_high_rate": 0.0,
        "low_rate": 0.0,
    }

    for d_w, m_w, r_w in grid:
        key = "__three_way_eval__"
        _set_three_way_weighted(
            work,
            prefix=prefix,
            domain_weight=d_w,
            method_weight=m_w,
            requirement_weight=r_w,
            out_key=key,
        )
        obj, reg, oob = _alpha_objective(
            rows=work,
            pred_key=key,
            objective=objective,
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
            oob_high_weight=oob_high_weight,
            oob_mid_weight=oob_mid_weight,
            oob_low_weight=oob_low_weight,
            hybrid_oob_scale=hybrid_oob_scale,
        )
        oob_obj = _weighted_oob_objective(
            oob=oob,
            w_high=oob_high_weight,
            w_mid=oob_mid_weight,
            w_low=oob_low_weight,
        )
        mae = float(reg.get("mae") or 0.0)
        trial_out.append(
            {
                "domain_weight": float(d_w),
                "method_weight": float(m_w),
                "requirement_weight": float(r_w),
                "objective": float(obj),
                "mae": float(mae),
                "oob_objective": float(oob_obj),
                "oob_high_rate": float(oob.get("high_rate") or 0.0),
                "oob_mid_rate": float(oob.get("mid_rate") or 0.0),
                "oob_mid_low_rate": float(oob.get("mid_low_rate") or 0.0),
                "oob_mid_high_rate": float(oob.get("mid_high_rate") or 0.0),
                "oob_low_rate": float(oob.get("low_rate") or 0.0),
            }
        )
        better = False
        if obj < best_obj - 1e-12:
            better = True
        elif abs(obj - best_obj) <= 1e-12:
            # Prefer keeping some requirement authority when validation objective ties.
            if abs(r_w - 0.5) < abs(best_weights[2] - 0.5):
                better = True
        if better:
            best_obj = float(obj)
            best_weights = (float(d_w), float(m_w), float(r_w))
            best_mae = float(mae)
            best_oob_obj = float(oob_obj)
            best_oob_rates = {
                "high_rate": float(oob.get("high_rate") or 0.0),
                "mid_rate": float(oob.get("mid_rate") or 0.0),
                "mid_low_rate": float(oob.get("mid_low_rate") or 0.0),
                "mid_high_rate": float(oob.get("mid_high_rate") or 0.0),
                "low_rate": float(oob.get("low_rate") or 0.0),
            }

    trial_out.sort(
        key=lambda x: (
            float(x.get("objective") or 0.0),
            abs(float(x.get("requirement_weight") or 0.0) - 0.5),
        )
    )
    return {
        "best_domain_weight": float(best_weights[0]),
        "best_method_weight": float(best_weights[1]),
        "best_requirement_weight": float(best_weights[2]),
        "best_objective": float(best_obj),
        "best_mae": float(best_mae),
        "best_oob_objective": float(best_oob_obj),
        "best_oob_rates": best_oob_rates,
        "trials_top10": trial_out[:10],
        "grid_count": int(len(grid)),
    }


def _format_summary(
    *,
    model_stats: Dict[str, Dict[str, Any]],
    high_threshold: float,
    mid_threshold: float,
    domain_weight: float,
    method_weight: float,
    dmr_domain_weight: float,
    dmr_method_weight: float,
    dmr_requirement_weight: float,
) -> str:
    lines: List[str] = []
    lines.append("=== Weighted Domain+Method(+Requirement) vs Ground Truth ===")
    lines.append(
        f"dm_weights: domain={domain_weight:.4f}, method={method_weight:.4f} | "
        f"dmr_weights: domain={dmr_domain_weight:.4f}, method={dmr_method_weight:.4f}, requirement={dmr_requirement_weight:.4f} | "
        f"thresholds: high>={high_threshold:.2f}, mid>={mid_threshold:.2f}"
    )
    lines.append("")
    lines.append(
        f"{'MODEL':<12} {'MAE':>8} {'RMSE':>8} {'PEARSON':>8} {'TOP1':>8} {'PAIR_ACC':>9} {'HARD_ACC':>9} {'MID_OOB':>9} {'LOW_OOB':>9} {'HIGH_OOB':>9}"
    )
    lines.append("-" * 104)

    for name in model_stats.keys():
        if name not in model_stats:
            continue
        reg = model_stats[name]["regression"]
        order = model_stats[name]["order"]
        oob = model_stats[name]["oob"]
        lines.append(
            f"{name:<12} "
            f"{float(reg.get('mae') or 0.0):>8.4f} "
            f"{float(reg.get('rmse') or 0.0):>8.4f} "
            f"{float(reg.get('pearson') or 0.0):>8.4f} "
            f"{float(order.get('top1_acc') or 0.0):>8.4f} "
            f"{float(order.get('pair_acc') or 0.0):>9.4f} "
            f"{float(order.get('hard_acc') or 0.0):>9.4f} "
            f"{float(oob.get('mid_rate') or 0.0):>9.4f} "
            f"{float(oob.get('low_rate') or 0.0):>9.4f} "
            f"{float(oob.get('high_rate') or 0.0):>9.4f}"
        )

    lines.append("")
    lines.append("=== Out-Of-Band Percentages ===")
    lines.append(f"{'MODEL':<12} {'LOW':>10} {'MID':>10} {'MID_LOW':>10} {'MID_HIGH':>10} {'HIGH':>10}")
    lines.append("-" * 70)
    for name in model_stats.keys():
        if name not in model_stats:
            continue
        oob = model_stats[name]["oob"]

        def _pct(key: str) -> float:
            return float(oob.get(key) or 0.0) * 100.0

        lines.append(
            f"{name:<12} "
            f"{_pct('low_rate'):>9.2f}% "
            f"{_pct('mid_rate'):>9.2f}% "
            f"{_pct('mid_low_rate'):>9.2f}% "
            f"{_pct('mid_high_rate'):>9.2f}% "
            f"{_pct('high_rate'):>9.2f}%"
        )

    for name in model_stats.keys():
        if name not in model_stats:
            continue
        band = model_stats[name]["band"]
        lines.append("")
        lines.append(f"=== {name.upper()} Band MAE ===")
        lines.append(f"{'BAND':<8} {'COUNT':>8} {'AVG_GT':>10} {'AVG_PRED':>10} {'MAE':>10}")
        lines.append("-" * 52)
        for b in ("overall", "high", "mid", "low"):
            row = band.get(b) or {}
            lines.append(
                f"{b:<8} "
                f"{int(row.get('count') or 0):>8} "
                f"{float(row.get('avg_gt') or 0.0):>10.4f} "
                f"{float(row.get('avg_pred') or 0.0):>10.4f} "
                f"{float(row.get('mae') or 0.0):>10.4f}"
            )

    return "\n".join(lines)


def _format_alpha_search_summary(
    *,
    split_meta: Dict[str, int],
    objective: str,
    alpha_results: Dict[str, Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append("=== In-Memory Alpha Search (Query-Group Split) ===")
    lines.append(
        f"queries_total={int(split_meta.get('queries_total') or 0)} "
        f"val_queries={int(split_meta.get('val_queries') or 0)} "
        f"test_queries={int(split_meta.get('test_queries') or 0)} "
        f"val_rows={int(split_meta.get('val_rows') or 0)} "
        f"test_rows={int(split_meta.get('test_rows') or 0)}"
    )
    lines.append(f"alpha_objective={_clean_text(objective) or 'mae'}")
    lines.append(
        f"{'MODEL':<12} {'BEST_A':>8} {'VAL_OBJ':>10} {'VAL_MAE':>9} {'TEST_MAE':>9} {'TEST_PAIR':>10} {'TEST_MID_OOB':>13} {'TEST_LOW_OOB':>13}"
    )
    lines.append("-" * 96)

    for name in ("finetuned", "base"):
        if name not in alpha_results:
            continue
        d = alpha_results[name]
        lines.append(
            f"{name:<12} "
            f"{float(d.get('best_alpha') or 0.5):>8.3f} "
            f"{float(d.get('val_objective') or 0.0):>10.4f} "
            f"{float(d.get('val_mae') or 0.0):>9.4f} "
            f"{float(d.get('test_mae') or 0.0):>9.4f} "
            f"{float(d.get('test_pair_acc') or 0.0):>10.4f} "
            f"{float(d.get('test_mid_oob') or 0.0):>13.4f} "
            f"{float(d.get('test_low_oob') or 0.0):>13.4f}"
        )

    lines.append("")
    lines.append("=== Best-Alpha Out-Of-Band Percentages ===")
    lines.append(f"{'MODEL':<12} {'LOW':>10} {'MID':>10} {'MID_LOW':>10} {'MID_HIGH':>10} {'HIGH':>10}")
    lines.append("-" * 70)
    for name in ("finetuned", "base"):
        if name not in alpha_results:
            continue
        d = alpha_results[name]

        def _pct(key: str) -> float:
            return float(d.get(key) or 0.0) * 100.0

        lines.append(
            f"{name:<12} "
            f"{_pct('test_low_oob'):>9.2f}% "
            f"{_pct('test_mid_oob'):>9.2f}% "
            f"{_pct('test_mid_low_oob'):>9.2f}% "
            f"{_pct('test_mid_high_oob'):>9.2f}% "
            f"{_pct('test_high_oob'):>9.2f}%"
        )

    return "\n".join(lines)


def _format_three_way_search_summary(
    *,
    split_meta: Dict[str, int],
    objective: str,
    three_way_results: Dict[str, Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append("=== In-Memory Three-Way Search (Domain+Method+Requirement) ===")
    lines.append(
        f"queries_total={int(split_meta.get('queries_total') or 0)} "
        f"val_queries={int(split_meta.get('val_queries') or 0)} "
        f"test_queries={int(split_meta.get('test_queries') or 0)} "
        f"val_rows={int(split_meta.get('val_rows') or 0)} "
        f"test_rows={int(split_meta.get('test_rows') or 0)}"
    )
    lines.append(f"objective={_clean_text(objective) or 'mae'}")
    lines.append(
        f"{'MODEL':<12} {'D_W':>7} {'M_W':>7} {'R_W':>7} {'VAL_OBJ':>10} {'VAL_MAE':>9} {'TEST_MAE':>9} {'TEST_PAIR':>10}"
    )
    lines.append("-" * 88)
    for name, d in three_way_results.items():
        lines.append(
            f"{name:<12} "
            f"{float(d.get('best_domain_weight') or 0.0):>7.3f} "
            f"{float(d.get('best_method_weight') or 0.0):>7.3f} "
            f"{float(d.get('best_requirement_weight') or 0.0):>7.3f} "
            f"{float(d.get('val_objective') or 0.0):>10.4f} "
            f"{float(d.get('val_mae') or 0.0):>9.4f} "
            f"{float(d.get('test_mae') or 0.0):>9.4f} "
            f"{float(d.get('test_pair_acc') or 0.0):>10.4f}"
        )

    lines.append("")
    lines.append("=== Best Three-Way Out-Of-Band Percentages ===")
    lines.append(f"{'MODEL':<12} {'LOW':>10} {'MID':>10} {'MID_LOW':>10} {'MID_HIGH':>10} {'HIGH':>10}")
    lines.append("-" * 70)
    for name, d in three_way_results.items():

        def _pct(key: str) -> float:
            return float(d.get(key) or 0.0) * 100.0

        lines.append(
            f"{name:<12} "
            f"{_pct('test_low_oob'):>9.2f}% "
            f"{_pct('test_mid_oob'):>9.2f}% "
            f"{_pct('test_mid_low_oob'):>9.2f}% "
            f"{_pct('test_mid_high_oob'):>9.2f}% "
            f"{_pct('test_high_oob'):>9.2f}%"
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Score ground-truth query/doc pairs with finetuned [DOMAIN], [METHOD], and [CONSTRAINT] heads."
        )
    )
    p.add_argument("--input", type=str, default=INPUT_DEFAULT)
    p.add_argument("--finetuned-model", type=str, default=FINETUNED_MODEL_DEFAULT)
    p.add_argument(
        "--aspect-condition-mode",
        type=str,
        default="legacy",
        choices=("legacy", "long_prefix", "none"),
        help="Condition formatting for the finetuned aspect scorer.",
    )
    p.add_argument("--domain-weight", type=float, default=0.5)
    p.add_argument("--method-weight", type=float, default=0.5)
    p.add_argument("--constraint-weight", type=float, default=0.5)
    p.add_argument("--high-threshold", type=float, default=HIGH_THRESHOLD_DEFAULT)
    p.add_argument("--mid-threshold", type=float, default=MID_THRESHOLD_DEFAULT)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--save-prefix", type=str, default="domain_method_constraint_pair_scores")
    p.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--print", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _run_model(
    *,
    model_ref: str,
    rows: List[Dict[str, Any]],
    batch_size: int,
    max_length: int,
    domain_weight: float,
    method_weight: float,
    constraint_weight: float,
    device: torch.device,
    out_prefix: str,
    aspect_condition_mode: str,
) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    model = load_sequence_classifier_model(model_ref, num_labels=1, multi_aspect_heads=False)
    model.to(device)
    model.eval()

    q_domain = [f"[DOMAIN] {_clean_text(r.get('query'))}" for r in rows]
    d_domain = [f"[DOMAIN] {_clean_text(r.get('doc'))}" for r in rows]
    q_method = [f"[METHOD] {_clean_text(r.get('query'))}" for r in rows]
    d_method = [f"[METHOD] {_clean_text(r.get('doc'))}" for r in rows]
    q_constraint = [f"[CONSTRAINT] {_clean_text(r.get('query'))}" for r in rows]
    d_constraint = [f"[CONSTRAINT] {_clean_text(r.get('doc'))}" for r in rows]

    s_domain = _score_pairs(
        model=model,
        tokenizer=tokenizer,
        queries=q_domain,
        docs=d_domain,
        aspect="domain",
        aspect_condition_mode=aspect_condition_mode,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    s_method = _score_pairs(
        model=model,
        tokenizer=tokenizer,
        queries=q_method,
        docs=d_method,
        aspect="method",
        aspect_condition_mode=aspect_condition_mode,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    s_constraint = _score_pairs(
        model=model,
        tokenizer=tokenizer,
        queries=q_constraint,
        docs=d_constraint,
        aspect="constraint",
        aspect_condition_mode=aspect_condition_mode,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )

    _apply_weighted_scores(
        rows=rows,
        dom_scores=s_domain,
        meth_scores=s_method,
        constraint_scores=s_constraint,
        out_prefix=out_prefix,
        domain_weight=domain_weight,
        method_weight=method_weight,
        constraint_weight=constraint_weight,
    )

    try:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "model_ref": model_ref,
        "rows": len(rows),
    }


def main() -> int:
    args = parse_args()

    in_path = _resolve_path(args.input)
    out_dir = _resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise RuntimeError(f"Input not found: {in_path}")

    finetuned_model = _resolve_model_ref(args.finetuned_model)

    domain_w = max(0.0, float(args.domain_weight))
    method_w = max(0.0, float(args.method_weight))
    constraint_w = max(0.0, float(args.constraint_weight))
    if (domain_w + method_w + constraint_w) <= 0.0:
        domain_w = 1.0
        method_w = 1.0
        constraint_w = 1.0
    denom = domain_w + method_w + constraint_w
    domain_w = 0.55#float(domain_w / denom)
    method_w = 0.35#float(method_w / denom)
    constraint_w = 0.1#float(constraint_w / denom)

    high_threshold = _safe_float(args.high_threshold, default=HIGH_THRESHOLD_DEFAULT, minimum=0.0, maximum=1.0)
    mid_threshold = _safe_float(args.mid_threshold, default=MID_THRESHOLD_DEFAULT, minimum=0.0, maximum=1.0)
    if mid_threshold > high_threshold:
        mid_threshold = high_threshold

    batch_size = _safe_int(args.batch_size, default=64, minimum=1, maximum=2048)
    max_length = _safe_int(args.max_length, default=512, minimum=16, maximum=8192)
    aspect_condition_mode = clean_aspect_condition_mode(args.aspect_condition_mode)

    rows = _load_ground_truth_rows(in_path)
    if not rows:
        raise RuntimeError("No valid rows loaded from input.")

    print(f"input={in_path}")
    print(f"rows_total={len(rows)}")
    print(f"finetuned_model={finetuned_model}")
    print(f"aspect_condition_mode={aspect_condition_mode}")
    print(
        f"weights: domain={domain_w:.4f}, method={method_w:.4f}, constraint={constraint_w:.4f}"
    )

    device = _pick_device()
    print(f"device={device.type}")

    started = time.time()

    _run_model(
        model_ref=finetuned_model,
        rows=rows,
        batch_size=batch_size,
        max_length=max_length,
        domain_weight=domain_w,
        method_weight=method_w,
        constraint_weight=constraint_w,
        device=device,
        out_prefix="finetuned",
        aspect_condition_mode=aspect_condition_mode,
    )

    pair_rows: List[Dict[str, Any]] = []
    for row in rows:
        pair_rows.append(
            {
                "query": _clean_text(row.get("query")),
                "doc": _clean_text(row.get("doc")),
                "gt_score": float(row.get("gt_score") or 0.0),
                "gt_band": _score_band(
                    float(row.get("gt_score") or 0.0),
                    high_threshold=high_threshold,
                    mid_threshold=mid_threshold,
                ),
                "domain_score": float(row.get("finetuned_domain") or 0.0),
                "method_score": float(row.get("finetuned_method") or 0.0),
                "constraint_score": float(row.get("finetuned_constraint") or 0.0),
                "weighted_score": float(row.get("finetuned_weighted") or 0.0),
            }
        )

    elapsed = float(time.time() - started)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_prefix = _clean_text(args.save_prefix) or "domain_method_constraint_pair_scores"
    save_path = out_dir / f"{save_prefix}_{timestamp}.jsonl"

    if bool(args.save):
        with save_path.open("w", encoding="utf-8") as f:
            for row in pair_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if bool(args.print):
        print("=== Domain/Method/Constraint Pair Score Export ===")
        print(f"rows_total={len(pair_rows)}")
        print(f"elapsed_sec={elapsed:.2f}")
        print(
            _format_oob_counts(
                rows=rows,
                pred_keys=[
                    ("weighted", "finetuned_weighted"),
                    ("domain", "finetuned_domain"),
                    ("method", "finetuned_method"),
                    ("constraint", "finetuned_constraint"),
                ],
                high_threshold=high_threshold,
                mid_threshold=mid_threshold,
            )
        )
        if bool(args.save):
            print(f"saved_jsonl={save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
