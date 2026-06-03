from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce3").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ASPECTS = ("topic", "approach", "objective")
ASPECT_HEADS_FILE = "aspect_heads.pt"
ASPECT_PREFIX_BY_NAME = {
    "topic": "[TOPIC]",
    "approach": "[APPROACH]",
    "objective": "[OBJECTIVE]",
}
GROUND_TRUTH_DEFAULT = "ce3/dataset/ground_truth/overall_coverage_test_subset_claude_opus_test.jsonl"
MODEL_DIR_DEFAULT = "ce3/models/aspect_reranker"
OUTPUT_DIR_DEFAULT = "ce3/eval/results"
PLAIN_CE_MODEL_DEFAULT = "dleemiller/ModernCE-base-sts"
HIGH_THRESHOLD = 0.70
MID_THRESHOLD = 0.30


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_ws(value: Any) -> str:
    return " ".join(_clean_text(value).split())


def _resolve_path(value: Any) -> Path:
    path = Path(_clean_text(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _clamp_01(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = 0.0
    return max(0.0, min(1.0, parsed))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


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
    last_file = root / ".last_train_output_dir"
    if last_file.exists():
        last_root = _resolve_path(last_file.read_text(encoding="utf-8").strip())
        for candidate in (last_root / "best_stage2_selected", last_root / "final", last_root / "best_stage1_selected", last_root):
            if _is_hf_model_dir(candidate):
                return str(candidate.resolve())

    candidates = [root / "best_stage2_selected", root / "final", root / "best_stage1_selected", root]
    if root.is_dir():
        candidates.extend(sorted(root.glob("stage2_epoch_*"), key=lambda p: p.stat().st_mtime, reverse=True))
        candidates.extend(sorted(root.glob("stage1_epoch_*"), key=lambda p: p.stat().st_mtime, reverse=True))
        candidates.extend(sorted(root.glob("*/best_stage2_selected"), key=lambda p: p.stat().st_mtime, reverse=True))
    for candidate in candidates:
        if _is_hf_model_dir(candidate):
            return str(candidate.resolve())
    return str((root / "best_stage2_selected").resolve())


def _pick_device(requested: str) -> Any:
    import torch

    raw = _clean_text(requested).lower()
    if raw:
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch: Dict[str, Any], device: Any) -> Dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def _tqdm_iter(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, **kwargs)


def _load_gt_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for obj in _iter_jsonl(path):
        pair = obj.get("pair") if isinstance(obj.get("pair"), dict) else {}
        grant_text = _normalize_ws(pair.get("grant_text"))
        faculty_text = _normalize_ws(pair.get("faculty_text"))
        if not grant_text or not faculty_text:
            continue
        rows.append(
            {
                "pair_id": _clean_text(pair.get("pair_id")),
                "grant_id": _clean_text(pair.get("grant_id")),
                "spec_idx": _safe_int(pair.get("spec_idx")),
                "query_item_id": _clean_text(pair.get("query_item_id")),
                "fac_item_id": _clean_text(pair.get("fac_item_id")),
                "grant_text": grant_text,
                "faculty_text": faculty_text,
                "gt_score": _clamp_01(obj.get("gt_score")),
            }
        )
    return rows


def _format_model_input(aspect: str, grant_text: str, faculty_text: str) -> Tuple[str, str]:
    from ce3.aspect_modeling import format_aspect_pair

    prefix = ASPECT_PREFIX_BY_NAME.get(aspect, f"[{aspect.upper()}]")
    return format_aspect_pair(f"{prefix} {grant_text}", faculty_text)


def _infer_aspect_scores(
    *,
    model: Any,
    tokenizer: Any,
    rows: Sequence[Dict[str, Any]],
    device: Any,
    batch_size: int,
    max_length: int,
) -> List[Dict[str, Any]]:
    import torch
    from ce3.aspect_modeling import aspect_id_from_name, model_logits

    tasks: List[Tuple[int, str, str, str]] = []
    for row_idx, row in enumerate(rows):
        for aspect in ASPECTS:
            tasks.append((row_idx, aspect, row["grant_text"], row["faculty_text"]))

    out = [dict(row, aspect_scores={}) for row in rows]
    model.eval()
    step = max(1, int(batch_size))
    chunks = [tasks[i : i + step] for i in range(0, len(tasks), step)]
    progress = _tqdm_iter(chunks, desc="Finetuned CE GT inference", unit="batch", dynamic_ncols=True)

    with torch.no_grad():
        for chunk in progress:
            queries: List[str] = []
            docs: List[str] = []
            aspect_ids: List[int] = []
            for _, aspect, grant_text, faculty_text in chunk:
                q, d = _format_model_input(aspect, grant_text, faculty_text)
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
            probs = torch.sigmoid(logits).detach().cpu().tolist()

            for (row_idx, aspect, _, _), score in zip(chunk, probs):
                out[row_idx]["aspect_scores"][aspect] = float(_clamp_01(score))

            if hasattr(progress, "set_postfix"):
                done = sum(len(row.get("aspect_scores", {})) for row in out)
                progress.set_postfix(scored=done, total=len(tasks))

    for row in out:
        scores = [float(row["aspect_scores"][aspect]) for aspect in ASPECTS if aspect in row["aspect_scores"]]
        row["ce_avg_score"] = float(sum(scores) / max(1, len(scores)))
        row["ce_min_score"] = float(min(scores)) if scores else 0.0
        row["ce_max_score"] = float(max(scores)) if scores else 0.0
        row["aggregate_scores"] = _aggregate_scores(row["aspect_scores"])
        row["error"] = float(row["aggregate_scores"]["mean"] - row["gt_score"])
        row["abs_error"] = float(abs(row["error"]))
    return out


def _attach_plain_ce_no_prefix_scores(
    *,
    rows: Sequence[Dict[str, Any]],
    model_ref: str,
    device: Any,
    batch_size: int,
    max_length: int,
    trust_remote_code: bool,
) -> List[Dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = [dict(row) for row in rows]
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=bool(trust_remote_code))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ref,
        num_labels=1,
        trust_remote_code=bool(trust_remote_code),
    )
    model.to(device)
    model.eval()

    step = max(1, int(batch_size))
    chunks = [list(range(i, min(i + step, len(out)))) for i in range(0, len(out), step)]
    progress = _tqdm_iter(chunks, desc="Plain CE no-prefix GT inference", unit="batch", dynamic_ncols=True)

    with torch.no_grad():
        for chunk in progress:
            queries = [out[row_idx]["grant_text"] for row_idx in chunk]
            docs = [out[row_idx]["faculty_text"] for row_idx in chunk]
            enc = tokenizer(
                queries,
                docs,
                max_length=int(max_length),
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = _to_device(enc, device)
            logits = model(**enc).logits.view(-1)
            probs = torch.sigmoid(logits).detach().cpu().tolist()

            for row_idx, score in zip(chunk, probs):
                plain_score = float(_clamp_01(score))
                aggregates = out[row_idx].setdefault("aggregate_scores", {})
                aggregates["plain_ce_no_prefix"] = plain_score
                out[row_idx]["plain_ce_no_prefix_score"] = plain_score

            if hasattr(progress, "set_postfix"):
                done = sum(1 for row in out if "plain_ce_no_prefix_score" in row)
                progress.set_postfix(scored=done, total=len(out))

    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _weighted_mean(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    denom = sum(float(weights.get(aspect, 0.0)) for aspect in ASPECTS)
    if denom <= 0.0:
        return _mean([float(scores.get(aspect, 0.0)) for aspect in ASPECTS])
    return float(
        sum(float(scores.get(aspect, 0.0)) * float(weights.get(aspect, 0.0)) for aspect in ASPECTS) / denom
    )


def _geometric_mean(values: Sequence[float]) -> float:
    vals = [max(0.0, float(x)) for x in values]
    if not vals or min(vals) <= 0.0:
        return 0.0
    return float(math.prod(vals) ** (1.0 / len(vals)))


def _harmonic_mean(values: Sequence[float]) -> float:
    vals = [max(0.0, float(x)) for x in values]
    if not vals or min(vals) <= 0.0:
        return 0.0
    return float(len(vals) / sum(1.0 / x for x in vals))


def _aggregate_scores(aspect_scores: Dict[str, Any]) -> Dict[str, float]:
    scores = {aspect: _clamp_01(aspect_scores.get(aspect, 0.0)) for aspect in ASPECTS}
    t = scores["topic"]
    a = scores["approach"]
    o = scores["objective"]
    vals = [t, a, o]
    mean_score = _mean(vals)
    min_score = min(vals)
    sorted_vals = sorted(vals)

    aggregators = {
        "mean": mean_score,
        "min_mean_0.4_0.6": 0.4 * min_score + 0.6 * mean_score,
        "topic_objective_half_approach": (t + o + 0.5 * a) / 2.5,
        "geometric_mean": _geometric_mean(vals),
        "harmonic_mean": _harmonic_mean(vals),
        "no_min_drop_min_mean": _mean(sorted_vals[1:]),  # Average only the two strongest aspects.
        "topic_objective_mean": (t + o) / 2.0,
        "topic_high_0.50": _weighted_mean(scores, {"topic": 0.50, "approach": 0.25, "objective": 0.25}),
        "approach_high_0.50": _weighted_mean(scores, {"topic": 0.25, "approach": 0.50, "objective": 0.25}),
        "objective_high_0.50": _weighted_mean(scores, {"topic": 0.25, "approach": 0.25, "objective": 0.50}),
        "topic_high_0.60": _weighted_mean(scores, {"topic": 0.60, "approach": 0.20, "objective": 0.20}),
        "approach_high_0.60": _weighted_mean(scores, {"topic": 0.20, "approach": 0.60, "objective": 0.20}),
        "objective_high_0.60": _weighted_mean(scores, {"topic": 0.20, "approach": 0.20, "objective": 0.60}),
        "topic_objective_high": _weighted_mean(scores, {"topic": 0.40, "approach": 0.20, "objective": 0.40}),
    }
    return {name: _clamp_01(value) for name, value in aggregators.items()}


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / math.sqrt(vx * vy))


def _rank(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    return _pearson(_rank(xs), _rank(ys))


def _score_band(score: float, *, high_threshold: float, mid_threshold: float) -> str:
    if float(score) >= float(high_threshold):
        return "high"
    if float(score) >= float(mid_threshold):
        return "mid"
    return "low"


def _coverage_metrics(gt: Sequence[float], pred: Sequence[float], *, threshold: float) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for y, p in zip(gt, pred):
        y_pos = float(y) >= float(threshold)
        p_pos = float(p) >= float(threshold)
        if y_pos and p_pos:
            tp += 1
        elif (not y_pos) and p_pos:
            fp += 1
        elif y_pos and (not p_pos):
            fn += 1
        else:
            tn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / max(1, tp + tn + fp + fn)),
        "fp_rate": float(fp / max(1, fp + tn)),
        "fn_rate": float(fn / max(1, fn + tp)),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def _metrics(
    rows: Sequence[Dict[str, Any]],
    *,
    score_name: str,
    high_threshold: float,
    mid_threshold: float,
) -> Dict[str, Any]:
    gt = [float(row["gt_score"]) for row in rows]
    pred = [float((row.get("aggregate_scores") or {}).get(score_name, row.get("ce_avg_score", 0.0))) for row in rows]
    errors = [p - y for y, p in zip(gt, pred)]
    abs_errors = [abs(x) for x in errors]
    sq_errors = [x * x for x in errors]
    band_acc = sum(
        1
        for y, p in zip(gt, pred)
        if _score_band(y, high_threshold=high_threshold, mid_threshold=mid_threshold)
        == _score_band(p, high_threshold=high_threshold, mid_threshold=mid_threshold)
    ) / max(1, len(rows))
    aspect_means = {
        aspect: _mean([float(row["aspect_scores"].get(aspect, 0.0)) for row in rows])
        for aspect in ASPECTS
    }
    return {
        "score_name": score_name,
        "n": int(len(rows)),
        "mae": _mean(abs_errors),
        "rmse": math.sqrt(_mean(sq_errors)),
        "bias_pred_minus_gt": _mean(errors),
        "pearson": _pearson(gt, pred),
        "spearman": _spearman(gt, pred),
        "mean_gt": _mean(gt),
        "mean_pred": _mean(pred),
        "aspect_pred_means": aspect_means,
        "band_accuracy": float(band_acc),
        "gt_band_counts": dict(Counter(_score_band(x, high_threshold=high_threshold, mid_threshold=mid_threshold) for x in gt)),
        "pred_band_counts": dict(Counter(_score_band(x, high_threshold=high_threshold, mid_threshold=mid_threshold) for x in pred)),
        "any_coverage": _coverage_metrics(gt, pred, threshold=mid_threshold),
        "high_coverage": _coverage_metrics(gt, pred, threshold=high_threshold),
    }


def _all_aggregator_metrics(
    rows: Sequence[Dict[str, Any]],
    *,
    high_threshold: float,
    mid_threshold: float,
) -> Dict[str, Dict[str, Any]]:
    if not rows:
        return {}
    first = rows[0].get("aggregate_scores") if isinstance(rows[0].get("aggregate_scores"), dict) else {}
    names = list(first.keys()) or ["mean"]
    return {
        name: _metrics(rows, score_name=name, high_threshold=high_threshold, mid_threshold=mid_threshold)
        for name in names
    }


def _print_single_metrics(metrics: Dict[str, Any]) -> None:
    print(f"\n=== {metrics.get('score_name', 'score')} vs Claude Overall GT ===")
    print(f"{'METRIC':<24} {'VALUE':>12}")
    print("-" * 38)
    for key in ("n", "mae", "rmse", "bias_pred_minus_gt", "pearson", "spearman", "mean_gt", "mean_pred", "band_accuracy"):
        value = metrics.get(key)
        if isinstance(value, int):
            text = str(value)
        elif isinstance(value, float):
            text = f"{value:.4f}"
        else:
            text = str(value)
        print(f"{key:<24} {text:>12}")
    print(f"\naspect_pred_means={metrics.get('aspect_pred_means', {})}")
    print(f"gt_band_counts={metrics.get('gt_band_counts', {})}")
    print(f"pred_band_counts={metrics.get('pred_band_counts', {})}")
    for task in ("any_coverage", "high_coverage"):
        vals = metrics.get(task) if isinstance(metrics.get(task), dict) else {}
        print(
            f"{task}: precision={float(vals.get('precision', 0.0)):.4f} "
            f"recall={float(vals.get('recall', 0.0)):.4f} "
            f"f1={float(vals.get('f1', 0.0)):.4f} "
            f"accuracy={float(vals.get('accuracy', 0.0)):.4f} "
            f"fp_rate={float(vals.get('fp_rate', 0.0)):.4f} "
            f"fn_rate={float(vals.get('fn_rate', 0.0)):.4f}"
        )


def _print_aggregator_table(metric_map: Dict[str, Dict[str, Any]]) -> None:
    print("\n=== Aggregator Comparison vs Claude Overall GT ===")
    header = (
        f"{'AGGREGATOR':<30} {'MAE':>7} {'RMSE':>7} {'BIAS':>7} {'R':>7} {'RHO':>7} "
        f"{'BAND':>7} {'ANY_F1':>7} {'HIGH_P':>7} {'HIGH_R':>7} {'HIGH_F1':>7} {'HIGH_FP':>7}"
    )
    print(header)
    print("-" * len(header))
    ordered = sorted(metric_map.items(), key=lambda x: (float(x[1].get("mae", 999.0)), -float(x[1].get("spearman", 0.0))))
    for name, metrics in ordered:
        any_cov = metrics.get("any_coverage") if isinstance(metrics.get("any_coverage"), dict) else {}
        high_cov = metrics.get("high_coverage") if isinstance(metrics.get("high_coverage"), dict) else {}
        print(
            f"{name:<30} "
            f"{float(metrics.get('mae', 0.0)):>7.4f} "
            f"{float(metrics.get('rmse', 0.0)):>7.4f} "
            f"{float(metrics.get('bias_pred_minus_gt', 0.0)):>7.4f} "
            f"{float(metrics.get('pearson', 0.0)):>7.4f} "
            f"{float(metrics.get('spearman', 0.0)):>7.4f} "
            f"{float(metrics.get('band_accuracy', 0.0)):>7.4f} "
            f"{float(any_cov.get('f1', 0.0)):>7.4f} "
            f"{float(high_cov.get('precision', 0.0)):>7.4f} "
            f"{float(high_cov.get('recall', 0.0)):>7.4f} "
            f"{float(high_cov.get('f1', 0.0)):>7.4f} "
            f"{float(high_cov.get('fp_rate', 0.0)):>7.4f}"
        )


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer CE3 finetuned aspect scores on Claude overall GT pairs and compare averaged CE score to GT.")
    p.add_argument("--ground-truth", type=str, default=GROUND_TRUTH_DEFAULT)
    p.add_argument("--model", type=str, default="")
    p.add_argument("--model-dir", type=str, default=MODEL_DIR_DEFAULT)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--save-prefix", type=str, default="overall_gt_vs_finetuned_ce")
    p.add_argument("--details-output", type=str, default="")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-multihead", action="store_true")
    p.add_argument("--plain-ce-model", type=str, default=PLAIN_CE_MODEL_DEFAULT)
    p.add_argument("--skip-plain-ce", action="store_true")
    p.add_argument("--high-threshold", type=float, default=HIGH_THRESHOLD)
    p.add_argument("--mid-threshold", type=float, default=MID_THRESHOLD)
    return p.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    ground_truth = _resolve_path(args.ground_truth)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not ground_truth.exists():
        raise FileNotFoundError(f"Missing Claude GT JSONL: {ground_truth}")

    model_ref = _resolve_model_ref(args.model) if _clean_text(args.model) else _resolve_default_model_ref(args.model_dir)
    device = _pick_device(args.device)
    high_threshold = _clamp_01(args.high_threshold)
    mid_threshold = min(high_threshold, _clamp_01(args.mid_threshold))

    rows = _load_gt_rows(ground_truth)
    if not rows:
        raise RuntimeError(f"No usable GT rows loaded from {ground_truth}")

    print(f"ground_truth={ground_truth}")
    print(f"model_ref={model_ref}")
    print(f"multihead={not bool(args.no_multihead)}")
    print(f"device={device}")
    print(f"gt_rows={len(rows)}")

    from ce3.aspect_modeling import load_sequence_classifier_model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=bool(args.trust_remote_code))
    model = load_sequence_classifier_model(
        model_ref,
        num_labels=1,
        multi_aspect_heads=not bool(args.no_multihead),
        trust_remote_code=bool(args.trust_remote_code),
    )
    model.to(device)

    scored_rows = _infer_aspect_scores(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    plain_ce_model_ref = _resolve_model_ref(args.plain_ce_model) if _clean_text(args.plain_ce_model) else ""
    if plain_ce_model_ref and not bool(args.skip_plain_ce):
        try:
            model.to("cpu")
        except Exception:
            pass
        del model
        import torch

        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        scored_rows = _attach_plain_ce_no_prefix_scores(
            rows=scored_rows,
            model_ref=plain_ce_model_ref,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            trust_remote_code=bool(args.trust_remote_code),
        )
    aggregator_metrics = _all_aggregator_metrics(scored_rows, high_threshold=high_threshold, mid_threshold=mid_threshold)
    metrics = aggregator_metrics.get("mean") or next(iter(aggregator_metrics.values()))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = _clean_text(args.save_prefix) or "overall_gt_vs_finetuned_ce"
    output_json = output_dir / f"{prefix}_{timestamp}.json"
    details_output = _resolve_path(args.details_output) if _clean_text(args.details_output) else output_dir / f"{prefix}_{timestamp}.details.jsonl"

    summary = {
        "mode": "ce3_overall_gt_vs_finetuned_ce_avg",
        "ground_truth": str(ground_truth),
        "model_ref": model_ref,
        "multihead": not bool(args.no_multihead),
        "plain_ce_model_ref": plain_ce_model_ref,
        "plain_ce_no_prefix": bool(plain_ce_model_ref and not bool(args.skip_plain_ce)),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "high_threshold": float(high_threshold),
        "mid_threshold": float(mid_threshold),
        "metrics": metrics,
        "aggregator_metrics": aggregator_metrics,
        "output_json": str(output_json),
        "details_output": str(details_output),
        "elapsed_sec": float(time.time() - started),
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(details_output, scored_rows)

    _print_aggregator_table(aggregator_metrics)
    _print_single_metrics(metrics)
    print(f"\noutput_json={output_json}")
    print(f"details_output={details_output}")
    print(f"elapsed_sec={time.time() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
