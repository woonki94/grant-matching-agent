from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore[assignment]


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce2").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce2.data_preparation.utils import ASPECTS, resolve_path  # noqa: E402


MODEL_DIR_DEFAULT = "ce2/models/basic_distill/best"
SPLIT_DIR_DEFAULT = "ce2/dataset/splits"
OUTPUT_JSON_DEFAULT = "ce2/eval/results/evaluation.json"
OUTPUT_PREDICTIONS_DEFAULT = "ce2/eval/results/predictions.jsonl"
ASPECT_PREFIX = {
    "domain": "[DOMAIN]",
    "method": "[METHOD]",
    "target": "[TARGET]",
}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_ws(value: Any) -> str:
    return " ".join(_clean_text(value).split())


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if not math.isfinite(out):
        out = float(default)
    return max(0.0, min(1.0, out))


def _score_band(score: float, *, high_threshold: float, mid_threshold: float) -> str:
    s = float(score)
    if s >= float(high_threshold):
        return "high"
    if s >= float(mid_threshold):
        return "mid"
    return "low"


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


@dataclass
class EvalRow:
    aspect: str
    split: str
    query_id: str
    doc_id: str
    pair_id: str
    query_text: str
    doc_text: str
    score: float
    teacher_band: str
    source: str
    pred_score: float = 0.0
    pred_band: str = "low"


def _load_rows(split_dir: Path, split: str, aspects: Sequence[str], high_threshold: float, mid_threshold: float) -> List[EvalRow]:
    rows: List[EvalRow] = []
    for aspect in aspects:
        path = split_dir / f"{aspect}_{split}.jsonl"
        if not path.exists():
            continue
        for obj in _iter_jsonl(path):
            query_text = _normalize_ws(obj.get("query_text"))
            doc_text = _normalize_ws(obj.get("doc_text"))
            query_id = _normalize_ws(obj.get("query_id"))
            doc_id = _normalize_ws(obj.get("doc_id"))
            if not query_text or not doc_text or not query_id or not doc_id:
                continue
            score = _clamp01(obj.get("score"))
            teacher_band = _normalize_ws(obj.get("band")).lower()
            if teacher_band not in {"high", "mid", "low"}:
                teacher_band = _score_band(score, high_threshold=high_threshold, mid_threshold=mid_threshold)
            rows.append(
                EvalRow(
                    aspect=aspect,
                    split=split,
                    query_id=query_id,
                    doc_id=doc_id,
                    pair_id=_normalize_ws(obj.get("pair_id")) or f"{query_id}::{doc_id}",
                    query_text=query_text,
                    doc_text=doc_text,
                    score=score,
                    teacher_band=teacher_band,
                    source=_normalize_ws(obj.get("source")),
                )
            )
    return rows


def _format_pair(row: EvalRow) -> Tuple[str, str]:
    prefix = ASPECT_PREFIX.get(row.aspect, f"[{row.aspect.upper()}]")
    return f"{prefix} {row.query_text}", f"{prefix} {row.doc_text}"


def _score_rows(
    *,
    rows: Sequence[EvalRow],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    max_length: int,
    use_tqdm: bool,
    high_threshold: float,
    mid_threshold: float,
) -> None:
    model.eval()
    iterator = range(0, len(rows), max(1, int(batch_size)))
    if use_tqdm and tqdm is not None:
        iterator = tqdm(iterator, desc="Scoring", leave=True)  # type: ignore[assignment]
    with torch.no_grad():
        for start in iterator:
            chunk = list(rows[start : start + max(1, int(batch_size))])
            pairs = [_format_pair(row) for row in chunk]
            queries = [p[0] for p in pairs]
            docs = [p[1] for p in pairs]
            enc = tokenizer(
                queries,
                docs,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {key: value.to(device) for key, value in dict(enc).items()}
            logits = model(**enc).logits.view(-1)
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            for row, pred in zip(chunk, probs):
                row.pred_score = _clamp01(pred)
                row.pred_band = _score_band(row.pred_score, high_threshold=high_threshold, mid_threshold=mid_threshold)


def _regression(rows: Sequence[EvalRow]) -> Dict[str, float]:
    n = len(rows)
    if n <= 0:
        return {"count": 0.0, "mae": 0.0, "rmse": 0.0, "pearson": 0.0, "avg_teacher": 0.0, "avg_pred": 0.0}
    ys = [float(r.score) for r in rows]
    ps = [float(r.pred_score) for r in rows]
    mae = sum(abs(p - y) for p, y in zip(ps, ys)) / float(n)
    rmse = math.sqrt(sum((p - y) ** 2 for p, y in zip(ps, ys)) / float(n))
    y_mean = sum(ys) / float(n)
    p_mean = sum(ps) / float(n)
    y_var = sum((y - y_mean) ** 2 for y in ys) / float(n)
    p_var = sum((p - p_mean) ** 2 for p in ps) / float(n)
    pearson = 0.0
    if y_var > 1e-12 and p_var > 1e-12:
        cov = sum((y - y_mean) * (p - p_mean) for y, p in zip(ys, ps)) / float(n)
        pearson = cov / math.sqrt(y_var * p_var)
    return {
        "count": float(n),
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson": float(pearson),
        "avg_teacher": float(y_mean),
        "avg_pred": float(p_mean),
    }


def _band_summary(rows: Sequence[EvalRow]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[EvalRow]] = {"overall": list(rows), "high": [], "mid": [], "low": []}
    for row in rows:
        if row.teacher_band in groups:
            groups[row.teacher_band].append(row)
    out: Dict[str, Dict[str, float]] = {}
    for band, band_rows in groups.items():
        if not band_rows:
            out[band] = {"count": 0.0, "avg_teacher": 0.0, "avg_pred": 0.0, "mae": 0.0}
            continue
        out[band] = {
            "count": float(len(band_rows)),
            "avg_teacher": float(sum(r.score for r in band_rows) / float(len(band_rows))),
            "avg_pred": float(sum(r.pred_score for r in band_rows) / float(len(band_rows))),
            "mae": float(sum(abs(r.pred_score - r.score) for r in band_rows) / float(len(band_rows))),
        }
    return out


def _compute_oob(rows: Sequence[EvalRow], *, high_threshold: float, mid_threshold: float) -> Dict[str, float]:
    low_total = mid_total = high_total = 0
    low_out = mid_out = mid_low_out = mid_high_out = high_out = 0
    for row in rows:
        y_band = row.teacher_band
        p = float(row.pred_score)
        if y_band == "low":
            low_total += 1
            if p >= float(mid_threshold):
                low_out += 1
        elif y_band == "mid":
            mid_total += 1
            if p < float(mid_threshold):
                mid_low_out += 1
            if p >= float(high_threshold):
                mid_high_out += 1
            if p < float(mid_threshold) or p >= float(high_threshold):
                mid_out += 1
        elif y_band == "high":
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


def _weighted_oob_objective(
    oob: Dict[str, float],
    *,
    high_weight: float,
    mid_weight: float,
    low_weight: float,
    mid_low_weight: float,
    mid_high_weight: float,
    split_mid: bool,
) -> float:
    w_high = max(0.0, float(high_weight))
    w_mid = max(0.0, float(mid_weight))
    w_low = max(0.0, float(low_weight))
    if split_mid:
        w_ml = max(0.0, float(mid_low_weight))
        w_mh = max(0.0, float(mid_high_weight))
        denom = max(1e-6, w_high + w_ml + w_mh + w_low)
        return float(
            (
                w_high * float(oob.get("high_rate", 0.0))
                + w_ml * float(oob.get("mid_low_rate", 0.0))
                + w_mh * float(oob.get("mid_high_rate", 0.0))
                + w_low * float(oob.get("low_rate", 0.0))
            )
            / denom
        )
    denom = max(1e-6, w_high + w_mid + w_low)
    return float(
        (
            w_high * float(oob.get("high_rate", 0.0))
            + w_mid * float(oob.get("mid_rate", 0.0))
            + w_low * float(oob.get("low_rate", 0.0))
        )
        / denom
    )


def _dcg(scores: Sequence[float]) -> float:
    total = 0.0
    for idx, rel in enumerate(scores, start=1):
        total += float(rel) / math.log2(float(idx) + 1.0)
    return total


def _ranking(rows: Sequence[EvalRow], *, top_k: int, rel_threshold: float, pair_eps: float) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], List[EvalRow]] = {}
    for row in rows:
        groups.setdefault((row.aspect, row.query_id), []).append(row)

    top1_total = 0
    top1_correct = 0
    mrr_total = 0
    mrr_sum = 0.0
    ndcg_total = 0
    ndcg_sum = 0.0
    recall_total = 0
    recall_sum = 0.0
    pair_total = 0
    pair_correct = 0

    k = max(1, int(top_k))
    eps = max(0.0, float(pair_eps))
    rel_cut = _clamp01(rel_threshold, default=0.70)

    for group in groups.values():
        if len(group) <= 1:
            continue
        by_teacher = sorted(group, key=lambda r: (-r.score, r.doc_id))
        by_pred = sorted(group, key=lambda r: (-r.pred_score, r.doc_id))

        top1_total += 1
        if by_teacher[0].doc_id == by_pred[0].doc_id:
            top1_correct += 1

        teacher_best_doc = by_teacher[0].doc_id
        for rank, row in enumerate(by_pred[:k], start=1):
            if row.doc_id == teacher_best_doc:
                mrr_sum += 1.0 / float(rank)
                break
        mrr_total += 1

        pred_rels = [r.score for r in by_pred[:k]]
        ideal_rels = [r.score for r in by_teacher[:k]]
        ideal = _dcg(ideal_rels)
        ndcg_sum += 0.0 if ideal <= 1e-12 else _dcg(pred_rels) / ideal
        ndcg_total += 1

        relevant_docs = {r.doc_id for r in group if r.score >= rel_cut}
        if relevant_docs:
            retrieved_docs = {r.doc_id for r in by_pred[:k]}
            recall_sum += len(relevant_docs & retrieved_docs) / float(len(relevant_docs))
            recall_total += 1

        for i, left in enumerate(group):
            for right in group[i + 1 :]:
                delta = float(left.score) - float(right.score)
                if abs(delta) < eps:
                    continue
                pair_total += 1
                pred_delta = float(left.pred_score) - float(right.pred_score)
                if delta * pred_delta > 0.0:
                    pair_correct += 1

    return {
        "query_count_total": float(len(groups)),
        "query_count_2plus_docs": float(top1_total),
        "top1_accuracy": float(top1_correct / float(max(1, top1_total))),
        "mrr_at_k": float(mrr_sum / float(max(1, mrr_total))),
        "ndcg_at_k": float(ndcg_sum / float(max(1, ndcg_total))),
        "recall_at_k": float(recall_sum / float(max(1, recall_total))),
        "pair_accuracy": float(pair_correct / float(max(1, pair_total))),
        "pair_total": float(pair_total),
        "config": {"top_k": float(k), "rel_threshold": float(rel_cut), "pair_eps": float(eps)},
    }


def _score_sanity(rows: Sequence[EvalRow], *, high_threshold: float, mid_threshold: float) -> Dict[str, Any]:
    by_band = {"high": [], "mid": [], "low": []}
    for row in rows:
        if row.teacher_band in by_band:
            by_band[row.teacher_band].append(float(row.pred_score))

    def avg(values: Sequence[float]) -> float:
        return float(sum(values) / float(len(values))) if values else 0.0

    high_avg = avg(by_band["high"])
    mid_avg = avg(by_band["mid"])
    low_avg = avg(by_band["low"])
    low_vals = by_band["low"]
    high_vals = by_band["high"]
    return {
        "avg_pred_high": float(high_avg),
        "avg_pred_mid": float(mid_avg),
        "avg_pred_low": float(low_avg),
        "monotonic_high_mid_low": bool(high_avg >= mid_avg >= low_avg),
        "weak_leak_rate": float(sum(1 for x in low_vals if x >= float(high_threshold)) / float(max(1, len(low_vals)))),
        "strong_collapse_rate": float(sum(1 for x in high_vals if x < float(mid_threshold)) / float(max(1, len(high_vals)))),
    }


def _aspect_groups(rows: Sequence[EvalRow]) -> Dict[str, List[EvalRow]]:
    out: Dict[str, List[EvalRow]] = {"overall": list(rows)}
    for aspect in ASPECTS:
        out[aspect] = [row for row in rows if row.aspect == aspect]
    return out


def _metrics_for_groups(
    rows: Sequence[EvalRow],
    *,
    high_threshold: float,
    mid_threshold: float,
    top_k: int,
    rel_threshold: float,
    pair_eps: float,
    oob_high_weight: float,
    oob_mid_weight: float,
    oob_low_weight: float,
    oob_mid_low_weight: float,
    oob_mid_high_weight: float,
    split_mid_oob: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, group_rows in _aspect_groups(rows).items():
        oob = _compute_oob(group_rows, high_threshold=high_threshold, mid_threshold=mid_threshold)
        out[name] = {
            "regression": _regression(group_rows),
            "band": _band_summary(group_rows),
            "oob": oob,
            "oob_objective": _weighted_oob_objective(
                oob,
                high_weight=oob_high_weight,
                mid_weight=oob_mid_weight,
                low_weight=oob_low_weight,
                mid_low_weight=oob_mid_low_weight,
                mid_high_weight=oob_mid_high_weight,
                split_mid=split_mid_oob,
            ),
            "ranking": _ranking(group_rows, top_k=top_k, rel_threshold=rel_threshold, pair_eps=pair_eps),
            "score_sanity": _score_sanity(group_rows, high_threshold=high_threshold, mid_threshold=mid_threshold),
        }
    return out


def _row_to_json(row: EvalRow) -> Dict[str, Any]:
    return {
        "aspect": row.aspect,
        "split": row.split,
        "query_id": row.query_id,
        "doc_id": row.doc_id,
        "pair_id": row.pair_id,
        "score": float(row.score),
        "teacher_band": row.teacher_band,
        "pred_score": float(row.pred_score),
        "pred_band": row.pred_band,
        "source": row.source,
    }


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_predictions(path: Path, rows: Sequence[EvalRow]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_row_to_json(row), ensure_ascii=False) + "\n")
    return len(rows)


def _format_oob_table(metrics: Dict[str, Any], *, high_threshold: float, mid_threshold: float) -> str:
    lines = [
        "",
        "=== Out-Of-Band Summary ===",
        f"thresholds: high>={high_threshold:.2f}, mid>={mid_threshold:.2f}",
        f"{'GROUP':<10} {'LOW_OUT':>18} {'MID_OUT':>18} {'MID_LOW':>18} {'MID_HIGH':>18} {'HIGH_OUT':>18} {'OBJ':>8}",
        "-" * 116,
    ]

    def cell(oob: Dict[str, float], out_key: str, total_key: str, rate_key: str) -> str:
        return f"{int(oob.get(out_key, 0.0))}/{int(oob.get(total_key, 0.0))} ({100.0 * float(oob.get(rate_key, 0.0)):.2f}%)"

    for name in ("overall", *ASPECTS):
        obj = metrics.get(name) or {}
        oob = obj.get("oob") or {}
        lines.append(
            f"{name:<10} "
            f"{cell(oob, 'low_out', 'low_total', 'low_rate'):>18} "
            f"{cell(oob, 'mid_out', 'mid_total', 'mid_rate'):>18} "
            f"{cell(oob, 'mid_low_out', 'mid_total', 'mid_low_rate'):>18} "
            f"{cell(oob, 'mid_high_out', 'mid_total', 'mid_high_rate'):>18} "
            f"{cell(oob, 'high_out', 'high_total', 'high_rate'):>18} "
            f"{float(obj.get('oob_objective') or 0.0):>8.4f}"
        )
    return "\n".join(lines)


def _format_main_table(metrics: Dict[str, Any]) -> str:
    lines = [
        "",
        "=== Main Metrics ===",
        f"{'GROUP':<10} {'N':>7} {'MAE':>8} {'RMSE':>8} {'PEARSON':>8} {'TOP1':>8} {'MRR@K':>8} {'NDCG@K':>8} {'PAIR_ACC':>9}",
        "-" * 90,
    ]
    for name in ("overall", *ASPECTS):
        obj = metrics.get(name) or {}
        reg = obj.get("regression") or {}
        rank = obj.get("ranking") or {}
        lines.append(
            f"{name:<10} "
            f"{int(reg.get('count') or 0):>7} "
            f"{float(reg.get('mae') or 0.0):>8.4f} "
            f"{float(reg.get('rmse') or 0.0):>8.4f} "
            f"{float(reg.get('pearson') or 0.0):>8.4f} "
            f"{float(rank.get('top1_accuracy') or 0.0):>8.4f} "
            f"{float(rank.get('mrr_at_k') or 0.0):>8.4f} "
            f"{float(rank.get('ndcg_at_k') or 0.0):>8.4f} "
            f"{float(rank.get('pair_accuracy') or 0.0):>9.4f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a CE2 cross-encoder on split files, including out-of-band rates.")
    p.add_argument("--model-dir", type=str, default=MODEL_DIR_DEFAULT)
    p.add_argument("--split-dir", type=str, default=SPLIT_DIR_DEFAULT)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--aspects", type=str, default=",".join(ASPECTS))
    p.add_argument("--output-json", type=str, default=OUTPUT_JSON_DEFAULT)
    p.add_argument("--predictions-output", type=str, default=OUTPUT_PREDICTIONS_DEFAULT)
    p.add_argument("--no-predictions", action="store_true")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--high-threshold", type=float, default=0.70)
    p.add_argument("--mid-threshold", type=float, default=0.30)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--rel-threshold", type=float, default=0.70)
    p.add_argument("--pair-eps", type=float, default=0.05)
    p.add_argument("--oob-high-weight", type=float, default=2.0)
    p.add_argument("--oob-mid-weight", type=float, default=1.0)
    p.add_argument("--oob-low-weight", type=float, default=1.0)
    p.add_argument("--oob-mid-low-weight", type=float, default=1.0)
    p.add_argument("--oob-mid-high-weight", type=float, default=1.0)
    p.add_argument("--split-mid-oob", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--no-tqdm", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Evaluation requires a working Transformers install. Check that `transformers` and "
            "`huggingface-hub` versions are compatible."
        ) from exc

    split_dir = resolve_path(PROJECT_ROOT, args.split_dir)
    model_dir = resolve_path(PROJECT_ROOT, args.model_dir)
    output_json = resolve_path(PROJECT_ROOT, args.output_json)
    predictions_output = resolve_path(PROJECT_ROOT, args.predictions_output)
    aspects = [a.strip().lower() for a in str(args.aspects).split(",") if a.strip()]
    aspects = [a for a in aspects if a in ASPECTS]
    if not aspects:
        raise RuntimeError(f"No valid aspects selected. Valid aspects: {', '.join(ASPECTS)}")

    high_threshold = _clamp01(args.high_threshold, default=0.70)
    mid_threshold = _clamp01(args.mid_threshold, default=0.30)
    if mid_threshold > high_threshold:
        mid_threshold = high_threshold

    rows = _load_rows(split_dir, args.split, aspects, high_threshold, mid_threshold)
    if not rows:
        raise RuntimeError(f"No rows loaded from {split_dir} for split={args.split}")

    if _clean_text(args.device):
        device = torch.device(_clean_text(args.device))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), trust_remote_code=True)
    model.to(device)

    use_tqdm = (not bool(args.no_tqdm)) and tqdm is not None
    _score_rows(
        rows=rows,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        use_tqdm=use_tqdm,
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
    )

    metrics = _metrics_for_groups(
        rows,
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
        top_k=int(args.top_k),
        rel_threshold=float(args.rel_threshold),
        pair_eps=float(args.pair_eps),
        oob_high_weight=float(args.oob_high_weight),
        oob_mid_weight=float(args.oob_mid_weight),
        oob_low_weight=float(args.oob_low_weight),
        oob_mid_low_weight=float(args.oob_mid_low_weight),
        oob_mid_high_weight=float(args.oob_mid_high_weight),
        split_mid_oob=bool(args.split_mid_oob),
    )
    summary = {
        "model_dir": str(model_dir),
        "split_dir": str(split_dir),
        "split": str(args.split),
        "aspects": aspects,
        "row_count": int(len(rows)),
        "device": str(device),
        "thresholds": {
            "high": float(high_threshold),
            "mid": float(mid_threshold),
        },
        "oob_weights": {
            "high": float(args.oob_high_weight),
            "mid": float(args.oob_mid_weight),
            "mid_low": float(args.oob_mid_low_weight),
            "mid_high": float(args.oob_mid_high_weight),
            "low": float(args.oob_low_weight),
            "split_mid": bool(args.split_mid_oob),
        },
        "metrics": metrics,
    }
    _write_json(output_json, summary)
    if not bool(args.no_predictions):
        _write_predictions(predictions_output, rows)

    print(_format_main_table(metrics))
    print(_format_oob_table(metrics, high_threshold=high_threshold, mid_threshold=mid_threshold))
    print(f"\nmetrics_json={output_json}")
    if not bool(args.no_predictions):
        print(f"predictions_jsonl={predictions_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
