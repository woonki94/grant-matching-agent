from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
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

from ce3.eval.compare_overall_gt_to_finetuned_ce import (  # noqa: E402
    ASPECTS,
    _all_aggregator_metrics,
    _clamp_01,
    _infer_aspect_scores,
    _load_gt_rows,
    _pick_device,
    _print_aggregator_table,
    _resolve_default_model_ref,
    _resolve_model_ref,
    _resolve_path,
    _write_jsonl,
)


GT_BASE_DEFAULT = "ce3/dataset/ground_truth/overall_coverage_test_subset_claude_opus"
GT_TRAIN_DEFAULT = f"{GT_BASE_DEFAULT}_train.jsonl"
GT_VAL_DEFAULT = f"{GT_BASE_DEFAULT}_val.jsonl"
GT_TEST_DEFAULT = f"{GT_BASE_DEFAULT}_test.jsonl"
MODEL_DIR_DEFAULT = "ce3/models/aspect_reranker"
OUTPUT_DIR_DEFAULT = "ce3/models/overall_combiner"
HIGH_THRESHOLD = 0.70
MID_THRESHOLD = 0.30
LINEAR_FEATURES = ("topic", "approach", "objective", "min", "mean", "max")
MLP_FEATURES = (
    "topic",
    "approach",
    "objective",
    "min",
    "mean",
    "max",
    "topic*objective",
    "topic*approach",
    "approach*objective",
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _is_hf_model_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    return any((path / name).exists() for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"))


def _threshold_to_logit(threshold: float) -> float:
    t = min(1.0 - 1e-6, max(1e-6, float(threshold)))
    return float(math.log(t / (1.0 - t)))


def _feature_values(aspect_scores: Dict[str, Any]) -> Dict[str, float]:
    t = _clamp_01(aspect_scores.get("topic", 0.0))
    a = _clamp_01(aspect_scores.get("approach", 0.0))
    o = _clamp_01(aspect_scores.get("objective", 0.0))
    vals = [t, a, o]
    return {
        "topic": t,
        "approach": a,
        "objective": o,
        "min": min(vals),
        "mean": sum(vals) / 3.0,
        "max": max(vals),
        "topic*objective": t * o,
        "topic*approach": t * a,
        "approach*objective": a * o,
    }


def _feature_names_for(kind: str) -> Tuple[str, ...]:
    if kind == "linear":
        return LINEAR_FEATURES
    if kind == "mlp":
        return MLP_FEATURES
    raise ValueError(f"Unknown combiner kind: {kind}")


def _build_feature_matrix(rows: Sequence[Dict[str, Any]], feature_names: Sequence[str]) -> Tuple[List[List[float]], List[float]]:
    features: List[List[float]] = []
    labels: List[float] = []
    for row in rows:
        values = _feature_values(row.get("aspect_scores") or {})
        features.append([float(values[name]) for name in feature_names])
        labels.append(_clamp_01(row.get("gt_score")))
    return features, labels


def _build_model(kind: str, input_dim: int, hidden_dim: int, dropout: float) -> Any:
    import torch
    from torch import nn

    if kind == "linear":
        return nn.Linear(int(input_dim), 1)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
    raise ValueError(f"Unknown combiner kind: {kind}")


def _tensorize(rows: Sequence[Dict[str, Any]], feature_names: Sequence[str], device: Any) -> Tuple[Any, Any]:
    import torch

    x, y = _build_feature_matrix(rows, feature_names)
    return (
        torch.tensor(x, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1),
    )


def _combiner_loss(
    logits: Any,
    targets: Any,
    *,
    any_boundary_weight: float,
    high_boundary_weight: float,
    mid_threshold: float,
    high_threshold: float,
) -> Tuple[Any, Dict[str, float]]:
    import torch
    import torch.nn.functional as F

    probs = torch.sigmoid(logits)
    mse = F.mse_loss(probs, targets)
    any_loss = logits.new_tensor(0.0)
    high_loss = logits.new_tensor(0.0)
    if float(any_boundary_weight) > 0.0:
        any_targets = (targets >= float(mid_threshold)).to(dtype=logits.dtype)
        any_logits = logits - _threshold_to_logit(mid_threshold)
        any_loss = F.binary_cross_entropy_with_logits(any_logits, any_targets)
    if float(high_boundary_weight) > 0.0:
        high_targets = (targets >= float(high_threshold)).to(dtype=logits.dtype)
        high_logits = logits - _threshold_to_logit(high_threshold)
        high_loss = F.binary_cross_entropy_with_logits(high_logits, high_targets)
    total = mse + float(any_boundary_weight) * any_loss + float(high_boundary_weight) * high_loss
    return total, {
        "mse": float(mse.detach().cpu().item()),
        "any_boundary": float(any_loss.detach().cpu().item()),
        "high_boundary": float(high_loss.detach().cpu().item()),
        "total": float(total.detach().cpu().item()),
    }


def _predict_scores(model: Any, rows: Sequence[Dict[str, Any]], feature_names: Sequence[str], device: Any, batch_size: int) -> List[float]:
    import torch

    model.eval()
    x, _ = _tensorize(rows, feature_names, device)
    preds: List[float] = []
    step = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), step):
            logits = model(x[start : start + step]).view(-1)
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            preds.extend(float(_clamp_01(x)) for x in probs)
    return preds


def _attach_predictions(
    rows: Sequence[Dict[str, Any]],
    *,
    model: Any,
    feature_names: Sequence[str],
    score_name: str,
    device: Any,
    batch_size: int,
) -> List[Dict[str, Any]]:
    preds = _predict_scores(model, rows, feature_names, device, batch_size)
    out: List[Dict[str, Any]] = []
    for row, pred in zip(rows, preds):
        item = copy.deepcopy(row)
        item.setdefault("aggregate_scores", {})
        item["aggregate_scores"][score_name] = float(pred)
        out.append(item)
    return out


def _selection_score(metrics: Dict[str, Any], mode: str) -> float:
    mode = _clean_text(mode).lower()
    if mode == "rmse":
        return float(metrics.get("rmse", 999.0))
    if mode == "mae_bias":
        return float(metrics.get("mae", 999.0)) + 0.25 * abs(float(metrics.get("bias_pred_minus_gt", 0.0)))
    return float(metrics.get("mae", 999.0))


def _train_one_combiner(
    *,
    kind: str,
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    device: Any,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    any_boundary_weight: float,
    high_boundary_weight: float,
    mid_threshold: float,
    high_threshold: float,
    patience: int,
    selection_metric: str,
) -> Tuple[Any, Dict[str, Any]]:
    import torch

    feature_names = _feature_names_for(kind)
    x_train, y_train = _tensorize(train_rows, feature_names, device)
    x_val, y_val = _tensorize(val_rows, feature_names, device)
    model = _build_model(kind, len(feature_names), hidden_dim, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))

    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    n = int(x_train.shape[0])
    step = max(1, int(batch_size))

    for epoch in range(1, max(1, int(epochs)) + 1):
        model.train()
        order = torch.randperm(n, generator=generator).to(device)
        train_losses: List[float] = []
        for start in range(0, n, step):
            idx = order[start : start + step]
            logits = model(x_train[idx])
            loss, parts = _combiner_loss(
                logits,
                y_train[idx],
                any_boundary_weight=any_boundary_weight,
                high_boundary_weight=high_boundary_weight,
                mid_threshold=mid_threshold,
                high_threshold=high_threshold,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(float(parts["total"]))

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss, val_parts = _combiner_loss(
                val_logits,
                y_val,
                any_boundary_weight=any_boundary_weight,
                high_boundary_weight=high_boundary_weight,
                mid_threshold=mid_threshold,
                high_threshold=high_threshold,
            )
            val_probs = torch.sigmoid(val_logits).detach().cpu().view(-1).tolist()
        val_eval_rows = []
        score_name = f"learned_{kind}"
        for row, pred in zip(val_rows, val_probs):
            item = copy.deepcopy(row)
            item.setdefault("aggregate_scores", {})
            item["aggregate_scores"][score_name] = float(_clamp_01(pred))
            val_eval_rows.append(item)
        val_metrics = _all_aggregator_metrics(
            val_eval_rows,
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
        ).get(score_name, {})
        score = _selection_score(val_metrics, selection_metric)
        row = {
            "epoch": int(epoch),
            "train_loss": sum(train_losses) / max(1, len(train_losses)),
            "val_loss": float(val_loss.detach().cpu().item()),
            "val_loss_parts": val_parts,
            "val_metrics": val_metrics,
            "selection_score": float(score),
        }
        history.append(row)
        if score < best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if int(patience) > 0 and epochs_without_improvement >= int(patience):
            break

    model.load_state_dict(best_state)
    meta = {
        "kind": kind,
        "feature_names": list(feature_names),
        "best_epoch": int(best_epoch),
        "best_selection_score": float(best_score),
        "epochs_run": int(len(history)),
        "history": history,
    }
    return model, meta


def _load_or_infer_split(
    *,
    split_name: str,
    path: Path,
    ce_model: Any,
    tokenizer: Any,
    device: Any,
    batch_size: int,
    max_length: int,
) -> List[Dict[str, Any]]:
    rows = _load_gt_rows(path)
    if not rows:
        raise RuntimeError(f"No usable rows loaded for {split_name}: {path}")
    return _infer_aspect_scores(
        model=ce_model,
        tokenizer=tokenizer,
        rows=rows,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )


def _print_model_comparison(metric_map: Dict[str, Dict[str, Any]], *, title: str) -> None:
    print(f"\n=== {title} ===")
    _print_aggregator_table(metric_map)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a frozen-CE overall coverage combiner on Claude GT splits.")
    p.add_argument("--gt-train", type=str, default=GT_TRAIN_DEFAULT)
    p.add_argument("--gt-val", type=str, default=GT_VAL_DEFAULT)
    p.add_argument("--gt-test", type=str, default=GT_TEST_DEFAULT)
    p.add_argument("--ce-model", type=str, default="")
    p.add_argument("--ce-model-dir", type=str, default=MODEL_DIR_DEFAULT)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--combiner", choices=("linear", "mlp", "both"), default="both")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--learning-rate", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--infer-batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--loss-any-boundary-weight", type=float, default=0.10)
    p.add_argument("--loss-high-boundary-weight", type=float, default=0.20)
    p.add_argument("--selection-metric", choices=("mae", "rmse", "mae_bias"), default="mae_bias")
    p.add_argument("--high-threshold", type=float, default=HIGH_THRESHOLD)
    p.add_argument("--mid-threshold", type=float, default=MID_THRESHOLD)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-multihead", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    _seed_everything(int(args.seed))

    gt_train = _resolve_path(args.gt_train)
    gt_val = _resolve_path(args.gt_val)
    gt_test = _resolve_path(args.gt_test)
    for label, path in (("train", gt_train), ("val", gt_val), ("test", gt_test)):
        if not path.exists():
            raise FileNotFoundError(f"Missing GT {label} file: {path}")

    ce_model_ref = _resolve_model_ref(args.ce_model) if _clean_text(args.ce_model) else _resolve_default_model_ref(args.ce_model_dir)
    device = _pick_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = _clean_text(args.run_name) or f"overall_combiner_{timestamp}"
    output_root = _resolve_path(args.output_dir)
    output_dir = output_root / run_name
    if output_dir.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Output already exists: {output_dir}. Pass --overwrite or set --run-name.")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"gt_train={gt_train}")
    print(f"gt_val={gt_val}")
    print(f"gt_test={gt_test}")
    print(f"ce_model_ref={ce_model_ref}")
    print(f"output_dir={output_dir}")
    print(f"device={device}")

    from ce3.aspect_modeling import load_sequence_classifier_model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ce_model_ref, trust_remote_code=bool(args.trust_remote_code))
    ce_model = load_sequence_classifier_model(
        ce_model_ref,
        num_labels=1,
        multi_aspect_heads=not bool(args.no_multihead),
        trust_remote_code=bool(args.trust_remote_code),
    )
    ce_model.to(device)
    ce_model.eval()
    for param in ce_model.parameters():
        param.requires_grad = False

    split_rows = {
        "train": _load_or_infer_split(
            split_name="train",
            path=gt_train,
            ce_model=ce_model,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.infer_batch_size,
            max_length=args.max_length,
        ),
        "val": _load_or_infer_split(
            split_name="val",
            path=gt_val,
            ce_model=ce_model,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.infer_batch_size,
            max_length=args.max_length,
        ),
        "test": _load_or_infer_split(
            split_name="test",
            path=gt_test,
            ce_model=ce_model,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.infer_batch_size,
            max_length=args.max_length,
        ),
    }

    kinds = ["linear", "mlp"] if args.combiner == "both" else [args.combiner]
    trained: Dict[str, Dict[str, Any]] = {}
    for kind in kinds:
        print(f"\ntraining_combiner={kind}")
        model, meta = _train_one_combiner(
            kind=kind,
            train_rows=split_rows["train"],
            val_rows=split_rows["val"],
            device=device,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            any_boundary_weight=args.loss_any_boundary_weight,
            high_boundary_weight=args.loss_high_boundary_weight,
            mid_threshold=args.mid_threshold,
            high_threshold=args.high_threshold,
            patience=args.patience,
            selection_metric=args.selection_metric,
        )
        score_name = f"learned_{kind}"
        feature_names = _feature_names_for(kind)
        scored_splits = {
            split: _attach_predictions(
                rows,
                model=model,
                feature_names=feature_names,
                score_name=score_name,
                device=device,
                batch_size=args.train_batch_size,
            )
            for split, rows in split_rows.items()
        }
        metrics = {
            split: _all_aggregator_metrics(
                rows,
                high_threshold=args.high_threshold,
                mid_threshold=args.mid_threshold,
            )
            for split, rows in scored_splits.items()
        }
        trained[kind] = {
            "model": model,
            "meta": meta,
            "score_name": score_name,
            "feature_names": list(feature_names),
            "scored_splits": scored_splits,
            "metrics": metrics,
            "val_selection_score": _selection_score(metrics["val"].get(score_name, {}), args.selection_metric),
        }
        _print_model_comparison(metrics["val"], title=f"Validation Aggregators With learned_{kind}")

    best_kind = min(trained, key=lambda k: float(trained[k]["val_selection_score"]))
    best = trained[best_kind]
    best_score_name = best["score_name"]

    torch = __import__("torch")
    torch.save(
        {
            "state_dict": best["model"].state_dict(),
            "kind": best_kind,
            "feature_names": best["feature_names"],
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "score_name": best_score_name,
        },
        output_dir / "combiner.pt",
    )
    config = {
        "architecture": "OverallCoverageCombiner",
        "best_kind": best_kind,
        "feature_names": best["feature_names"],
        "score_name": best_score_name,
        "ce_model_ref": ce_model_ref,
        "gt_train": str(gt_train),
        "gt_val": str(gt_val),
        "gt_test": str(gt_test),
        "mid_threshold": float(args.mid_threshold),
        "high_threshold": float(args.high_threshold),
        "selection_metric": args.selection_metric,
        "loss_any_boundary_weight": float(args.loss_any_boundary_weight),
        "loss_high_boundary_weight": float(args.loss_high_boundary_weight),
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics_summary = {
        "mode": "ce3_train_overall_combiner",
        "output_dir": str(output_dir),
        "best_kind": best_kind,
        "best_score_name": best_score_name,
        "ce_model_ref": ce_model_ref,
        "split_sizes": {split: len(rows) for split, rows in split_rows.items()},
        "train_meta": {kind: {k: v for k, v in data["meta"].items() if k != "history"} for kind, data in trained.items()},
        "history": {kind: data["meta"].get("history", []) for kind, data in trained.items()},
        "metrics": {kind: data["metrics"] for kind, data in trained.items()},
        "elapsed_sec": float(time.time() - started),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split, rows in best["scored_splits"].items():
        _write_jsonl(output_dir / f"{split}.details.jsonl", rows)

    print(f"\nbest_combiner={best_kind} val_selection_score={best['val_selection_score']:.6f}")
    print("\nValidation:")
    _print_aggregator_table(best["metrics"]["val"])
    print("\nTest:")
    _print_aggregator_table(best["metrics"]["test"])
    print(f"\ncombiner_path={output_dir / 'combiner.pt'}")
    print(f"metrics_path={output_dir / 'metrics.json'}")
    print(f"elapsed_sec={time.time() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
