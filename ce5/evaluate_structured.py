"""Evaluate a structured CE5 requirement matcher on its verified held-out split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CHECKPOINT = (
    REPO_ROOT / "ce5" / "models" / "structured_requirement_matcher_v1" / "best.pt"
)
DEFAULT_JUDGMENTS = (
    REPO_ROOT
    / "ce5"
    / "dataset"
    / "judgments"
    / "structured_teacher_judgments_v1.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "ce5" / "eval" / "results" / "structured_requirement_matcher_v1"
)
DEFAULT_BASELINE_MODEL = "dleemiller/ModernCE-base-sts"


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = REPO_ROOT / expanded
    return expanded.resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pair_id_sha256(examples: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for pair_id in sorted(str(example.pair_id) for example in examples):
        digest.update(pair_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _verify_split(
    examples: Sequence[Any],
    *,
    split_name: str,
    manifest_path: Path,
    allow_unverified: bool,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    entry = manifest.get(split_name)
    expected = (
        _clean_text(entry.get("pair_id_sha256"))
        if isinstance(entry, Mapping)
        else ""
    )
    actual = _pair_id_sha256(examples)
    result = {
        "manifest": str(manifest_path),
        "verified": False,
        "expected_pair_id_sha256": expected or None,
        "actual_pair_id_sha256": actual,
        "examples_before_limit": len(examples),
    }
    if not expected:
        if not allow_unverified:
            raise RuntimeError(
                f"Cannot verify {split_name}: {manifest_path} has no matching entry"
            )
        return result
    if expected != actual:
        raise RuntimeError(
            f"Reconstructed {split_name} does not match the training manifest: "
            f"expected {expected}, got {actual}"
        )
    result["verified"] = True
    return result


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average = (position + 1 + end) / 2.0
        for offset in range(position, end):
            ranks[order[offset]] = average
        position = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else 0.0


def _pairwise_accuracy(
    predictions: Sequence[float],
    targets: Sequence[float],
    query_ids: Sequence[str],
    *,
    min_gap: float,
) -> tuple[float, int]:
    grouped: dict[str, list[int]] = {}
    for index, query_id in enumerate(query_ids):
        grouped.setdefault(query_id, []).append(index)
    correct = 0.0
    comparisons = 0
    for indices in grouped.values():
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                target_difference = targets[left] - targets[right]
                if abs(target_difference) < min_gap:
                    continue
                prediction_difference = predictions[left] - predictions[right]
                comparisons += 1
                if prediction_difference == 0.0:
                    correct += 0.5
                elif prediction_difference * target_difference > 0.0:
                    correct += 1.0
    return (correct / comparisons if comparisons else 0.0), comparisons


def _score_band(score: float) -> str:
    if score < 0.25:
        return "low"
    if score < 0.75:
        return "mid"
    return "high"


def _metrics(
    predictions: Sequence[float],
    targets: Sequence[float],
    query_ids: Sequence[str],
    *,
    min_gap: float,
) -> dict[str, Any]:
    if not predictions or len(predictions) != len(targets):
        raise ValueError("Predictions and targets must be equally sized and nonempty")
    errors = [
        prediction - target
        for prediction, target in zip(predictions, targets, strict=True)
    ]
    mse = _mean([error * error for error in errors])
    pairwise, comparisons = _pairwise_accuracy(
        predictions,
        targets,
        query_ids,
        min_gap=min_gap,
    )
    band_counts = {
        band: {"examples": 0, "out_of_band": 0} for band in ("low", "mid", "high")
    }
    high_placement = {
        "below_0.25": 0,
        "0.25-0.50": 0,
        "0.50-0.65": 0,
        "0.65-0.75": 0,
    }
    for prediction, target in zip(predictions, targets, strict=True):
        teacher_band = _score_band(target)
        predicted_band = _score_band(prediction)
        band_counts[teacher_band]["examples"] += 1
        band_counts[teacher_band]["out_of_band"] += int(
            predicted_band != teacher_band
        )
        if teacher_band == "high" and prediction < 0.75:
            if prediction < 0.25:
                high_placement["below_0.25"] += 1
            elif prediction < 0.50:
                high_placement["0.25-0.50"] += 1
            elif prediction < 0.65:
                high_placement["0.50-0.65"] += 1
            else:
                high_placement["0.65-0.75"] += 1
    by_band = {
        band: {
            **counts,
            "ratio": counts["out_of_band"] / counts["examples"]
            if counts["examples"]
            else None,
        }
        for band, counts in band_counts.items()
    }
    high_examples = band_counts["high"]["examples"]
    return {
        "examples": len(predictions),
        "mae": _mean([abs(error) for error in errors]),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "pearson": _pearson(predictions, targets),
        "spearman": _pearson(_average_ranks(predictions), _average_ranks(targets)),
        "pairwise_accuracy": pairwise,
        "pairwise_comparisons": comparisons,
        "prediction_mean": _mean(predictions),
        "target_mean": _mean(targets),
        "out_of_score_band_by_teacher_band": by_band,
        "high_oob_placement": {
            "teacher_examples": high_examples,
            "counts": high_placement,
            "ratios_of_teacher_high": {
                name: count / high_examples if high_examples else None
                for name, count in high_placement.items()
            },
        },
    }


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _resolve_precision(value: str, device: torch.device) -> torch.dtype:
    if value == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def _autocast(device: torch.device, precision: torch.dtype) -> Any:
    if device.type == "cuda" and precision in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=precision)
    return nullcontext()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _matching_args(training_arguments: Mapping[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        match_coverage_cost=float(training_arguments.get("match_coverage_cost", 1.0)),
        match_grant_span_cost=float(
            training_arguments.get("match_grant_span_cost", 1.0)
        ),
        match_faculty_span_cost=float(
            training_arguments.get("match_faculty_span_cost", 1.0)
        ),
    )


@torch.inference_mode()
def _evaluate_structured(
    model: Any,
    loader: DataLoader[Any],
    device: torch.device,
    precision: torch.dtype,
    *,
    matching_args: argparse.Namespace,
    min_score_gap: float,
    no_progress: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from ce5.training import train_structured_requirement_matcher as trainer

    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    query_ids: list[str] = []
    records: list[dict[str, Any]] = []
    coverage_errors: list[float] = []
    grant_masses: list[float] = []
    faculty_masses: list[float] = []
    matched_active: list[float] = []
    unmatched_active: list[float] = []
    contributions: list[float] = []
    slot_match_counts: Counter[int] = Counter()
    inference_seconds = 0.0
    progress: Iterable[Any] = loader
    if tqdm is not None and not no_progress:
        progress = tqdm(loader, desc="Structured CE5 evaluation", dynamic_ncols=True)
    wall_started = time.perf_counter()
    for raw_batch in progress:
        batch = trainer._move_batch(raw_batch, device)
        _sync(device)
        forward_started = time.perf_counter()
        with _autocast(device, precision):
            output = model(**batch["encoded"])
        _sync(device)
        inference_seconds += time.perf_counter() - forward_started
        matches = trainer._match_slots(output, batch, matching_args)
        matches_by_example: dict[int, list[tuple[int, int]]] = {}
        for batch_index, slot_index, teacher_index in matches:
            matches_by_example.setdefault(batch_index, []).append(
                (slot_index, teacher_index)
            )
        batch_scores = output.scores.detach().float().cpu().tolist()
        for batch_index, pair_id in enumerate(batch["pair_ids"]):
            score = float(batch_scores[batch_index])
            target = float(batch["overall_scores"][batch_index].item())
            predictions.append(score)
            targets.append(target)
            query_ids.append(batch["query_ids"][batch_index])
            contributions.append(
                float(output.structured_contribution[batch_index].float().item())
            )
            requirement_records: list[dict[str, Any]] = []
            matched_slots: set[int] = set()
            for slot_index, teacher_index in sorted(
                matches_by_example.get(batch_index, []),
                key=lambda pair: pair[1],
            ):
                matched_slots.add(slot_index)
                slot_match_counts[slot_index] += 1
                predicted_coverage = float(
                    output.slot_coverage_scores[batch_index, slot_index]
                    .float()
                    .item()
                )
                teacher_coverage = float(
                    batch["teacher_coverage"][batch_index, teacher_index].item()
                )
                coverage_errors.append(predicted_coverage - teacher_coverage)
                grant_mass = float(
                    trainer._attention_mass(
                        output.target_attention_weights[batch_index, slot_index],
                        batch["target_span_masks"][batch_index, teacher_index],
                    )
                    .float()
                    .item()
                )
                grant_masses.append(grant_mass)
                has_support = bool(
                    batch["teacher_has_support"][batch_index, teacher_index]
                )
                faculty_mass: Optional[float] = None
                if has_support:
                    faculty_mass = float(
                        trainer._attention_mass(
                            output.candidate_attention_weights[
                                batch_index,
                                slot_index,
                            ],
                            batch["candidate_span_masks"][
                                batch_index,
                                teacher_index,
                            ],
                        )
                        .float()
                        .item()
                    )
                    faculty_masses.append(faculty_mass)
                active_probability = float(
                    output.slot_active_probabilities[batch_index, slot_index]
                    .float()
                    .item()
                )
                matched_active.append(active_probability)
                requirement_records.append(
                    {
                        "requirement_claim_id": batch["requirement_ids"][
                            batch_index
                        ][teacher_index],
                        "matched_slot": slot_index,
                        "teacher_coverage": teacher_coverage,
                        "predicted_coverage": predicted_coverage,
                        "absolute_error": abs(predicted_coverage - teacher_coverage),
                        "predicted_active_probability": active_probability,
                        "grant_span_attention_mass": grant_mass,
                        "faculty_span_attention_mass": faculty_mass,
                        "teacher_has_faculty_support": has_support,
                    }
                )
            for slot_index in range(model.architecture_config.num_requirement_slots):
                if slot_index not in matched_slots:
                    unmatched_active.append(
                        float(
                            output.slot_active_probabilities[batch_index, slot_index]
                            .float()
                            .item()
                        )
                    )
            records.append(
                {
                    "pair_id": pair_id,
                    "grant_item_id": batch["query_ids"][batch_index],
                    "teacher_score": target,
                    "structured_ce5_score": score,
                    "absolute_error": abs(score - target),
                    "structured_contribution": contributions[-1],
                    "predicted_overall_confidence": float(
                        output.overall_confidence_score[batch_index].float().item()
                    ),
                    "requirements": requirement_records,
                }
            )
    wall_seconds = time.perf_counter() - wall_started
    coverage_mse = _mean([error * error for error in coverage_errors])
    result = {
        "metrics": _metrics(
            predictions,
            targets,
            query_ids,
            min_gap=min_score_gap,
        ),
        "structured_diagnostics": {
            "matched_requirements": len(coverage_errors),
            "requirement_coverage_mae": _mean(
                [abs(error) for error in coverage_errors]
            ),
            "requirement_coverage_rmse": math.sqrt(coverage_mse),
            "grant_span_attention_mass": _mean(grant_masses),
            "faculty_span_attention_mass": _mean(faculty_masses),
            "matched_slot_active_mean": _mean(matched_active),
            "unmatched_slot_active_mean": _mean(unmatched_active),
            "structured_contribution_mean": _mean(contributions),
            "slot_match_counts": {
                f"slot_{index}": slot_match_counts[index]
                for index in range(model.architecture_config.num_requirement_slots)
            },
        },
        "runtime": {
            "wall_seconds": wall_seconds,
            "model_inference_seconds": inference_seconds,
            "examples_per_second": len(predictions) / max(inference_seconds, 1e-12),
            "batches": len(loader),
        },
    }
    return result, records


@torch.inference_mode()
def _evaluate_baseline(
    model: Any,
    loader: DataLoader[Any],
    device: torch.device,
    precision: torch.dtype,
    *,
    min_score_gap: float,
    no_progress: bool,
) -> tuple[dict[str, Any], dict[str, float]]:
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    query_ids: list[str] = []
    predictions_by_pair: dict[str, float] = {}
    inference_seconds = 0.0
    progress: Iterable[Any] = loader
    if tqdm is not None and not no_progress:
        progress = tqdm(loader, desc="Untouched ModernCE baseline", dynamic_ncols=True)
    wall_started = time.perf_counter()
    for batch in progress:
        encoded = {
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in batch["encoded"].items()
            if key not in {"target_mask", "candidate_mask"}
        }
        _sync(device)
        forward_started = time.perf_counter()
        with _autocast(device, precision):
            output = model(**encoded)
        _sync(device)
        inference_seconds += time.perf_counter() - forward_started
        logits = output.logits.detach().float().cpu()
        if logits.ndim == 2 and logits.shape[-1] == 1:
            scores = torch.sigmoid(logits[:, 0])
        elif logits.ndim == 2 and logits.shape[-1] == 2:
            scores = torch.softmax(logits, dim=-1)[:, 1]
        else:
            raise RuntimeError(f"Unexpected baseline logits shape: {tuple(logits.shape)}")
        for index, score in enumerate(scores.tolist()):
            numeric = float(score)
            predictions.append(numeric)
            targets.append(float(batch["overall_scores"][index]))
            query_ids.append(batch["query_ids"][index])
            predictions_by_pair[batch["pair_ids"][index]] = numeric
    wall_seconds = time.perf_counter() - wall_started
    return {
        "metrics": _metrics(
            predictions,
            targets,
            query_ids,
            min_gap=min_score_gap,
        ),
        "runtime": {
            "wall_seconds": wall_seconds,
            "model_inference_seconds": inference_seconds,
            "examples_per_second": len(predictions) / max(inference_seconds, 1e-12),
            "batches": len(loader),
        },
    }, predictions_by_pair


def _ratio(metrics: Mapping[str, Any], band: str) -> float:
    value = metrics["out_of_score_band_by_teacher_band"][band]["ratio"]
    return float(value) if value is not None else 0.0


def _print_summary(
    *,
    split_label: str,
    verified: bool,
    structured: Mapping[str, Any],
    baseline: Optional[Mapping[str, Any]],
    summary_path: Path,
) -> None:
    rows = []
    if baseline is not None:
        rows.append(("Untouched STS", baseline))
    rows.append(("CE5 structured", structured))
    print("\n=== CE5 structured evaluation ===")
    print(
        f"Split: {split_label} | examples: "
        f"{structured['metrics']['examples']:,} | manifest: "
        f"{'VERIFIED' if verified else 'UNVERIFIED'}"
    )
    print(
        "Method             MAE       RMSE      Pearson    Spearman   "
        "PairAcc    Low OOB    Mid OOB    High OOB   Ex/s"
    )
    print(
        "-----------------  --------  --------  ---------  ---------  "
        "---------  ---------  ---------  ---------  ---------"
    )
    for name, result in rows:
        metrics = result["metrics"]
        print(
            f"{name:<17}  {metrics['mae']:8.4f}  {metrics['rmse']:8.4f}  "
            f"{metrics['pearson']:9.4f}  {metrics['spearman']:9.4f}  "
            f"{100 * metrics['pairwise_accuracy']:8.2f}%  "
            f"{100 * _ratio(metrics, 'low'):8.2f}%  "
            f"{100 * _ratio(metrics, 'mid'):8.2f}%  "
            f"{100 * _ratio(metrics, 'high'):8.2f}%  "
            f"{result['runtime']['examples_per_second']:9,.0f}"
        )
    diagnostics = structured["structured_diagnostics"]
    print("\n=== Structured diagnostics ===")
    print(
        f"Requirement coverage: MAE {diagnostics['requirement_coverage_mae']:.4f} | "
        f"RMSE {diagnostics['requirement_coverage_rmse']:.4f}"
    )
    print(
        f"Attention mass: grant {diagnostics['grant_span_attention_mass']:.3f} | "
        f"faculty support {diagnostics['faculty_span_attention_mass']:.3f}"
    )
    print(
        f"Slot activity: matched {diagnostics['matched_slot_active_mean']:.3f} | "
        f"unused {diagnostics['unmatched_slot_active_mean']:.3f}"
    )
    print(
        f"Structured contribution: "
        f"{diagnostics['structured_contribution_mean']:.3f}"
    )
    if baseline is not None:
        base_metrics = baseline["metrics"]
        current = structured["metrics"]
        print(
            "CE5 vs untouched STS: "
            f"MAE {(current['mae'] / base_metrics['mae'] - 1) * 100:+.2f}%, "
            f"RMSE {(current['rmse'] / base_metrics['rmse'] - 1) * 100:+.2f}%, "
            f"Pearson {current['pearson'] - base_metrics['pearson']:+.4f}, "
            f"Spearman {current['spearman'] - base_metrics['spearman']:+.4f}, "
            f"PairAcc {(current['pairwise_accuracy'] - base_metrics['pairwise_accuracy']) * 100:+.2f} pp"
        )
    print(f"Full report: {summary_path}\n")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a structured CE5 checkpoint on its verified split."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--evaluation-split",
        choices=("test", "validation", "train", "all"),
        default="test",
    )
    parser.add_argument("--limit-evaluation", type=_nonnegative_int, default=0)
    parser.add_argument("--skip-split-verification", action="store_true")
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--max-length", type=_positive_int)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--baseline-model-id", default=DEFAULT_BASELINE_MODEL)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_path = _resolve_path(args.checkpoint)
    judgment_path = _resolve_path(args.judgments)
    output_dir = _resolve_path(args.output_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not judgment_path.exists():
        raise FileNotFoundError(f"Judgments not found: {judgment_path}")
    training_config_path = (
        _resolve_path(args.training_config)
        if args.training_config
        else checkpoint_path.parent / "run_config.json"
    )
    training_config = _load_json(training_config_path)
    training_arguments = training_config.get("arguments")
    if not isinstance(training_arguments, Mapping):
        raise RuntimeError(f"Missing training arguments in {training_config_path}")

    from ce5.modeling.structured_requirement_matcher import (
        ARCHITECTURE_TYPE,
        ModernCEStructuredRequirementMatcher,
    )
    from ce5.training import train_structured_requirement_matcher as trainer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    examples, loading_stats = trainer.load_structured_judgments(
        judgment_path,
        min_confidence=float(training_arguments.get("min_confidence", 0.0)),
        duplicate_policy=str(
            training_arguments.get("duplicate_policy", "highest-confidence")
        ),
        max_examples=int(training_arguments.get("max_examples", 0)),
    )
    train_examples, validation_examples, test_examples = trainer.split_examples_three_way(
        examples,
        validation_ratio=float(training_arguments.get("validation_ratio", 0.1)),
        test_ratio=float(training_arguments.get("test_ratio", 0.1)),
        split_group=str(training_arguments.get("split_group", "owner")),
        seed=int(training_arguments.get("seed", 42)),
    )
    split_map = {
        "train": train_examples,
        "validation": validation_examples,
        "test": test_examples,
        "all": examples,
    }
    evaluation_examples = split_map[args.evaluation_split]
    manifest_path = checkpoint_path.parent / "split_manifest.json"
    verification = {
        "manifest": str(manifest_path),
        "verified": False,
        "examples_before_limit": len(evaluation_examples),
    }
    if args.evaluation_split != "all":
        verification = _verify_split(
            evaluation_examples,
            split_name=args.evaluation_split,
            manifest_path=manifest_path,
            allow_unverified=args.skip_split_verification,
        )
    if args.limit_evaluation > 0:
        evaluation_examples = evaluation_examples[: args.limit_evaluation]
    if not evaluation_examples:
        raise RuntimeError("Selected evaluation split is empty")

    device = _resolve_device(args.device)
    precision = _resolve_precision(args.precision, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    model = ModernCEStructuredRequirementMatcher.from_checkpoint(
        checkpoint_path,
        torch_dtype=precision,
        trust_remote_code=args.trust_remote_code,
    )
    if model.architecture_type != ARCHITECTURE_TYPE:
        raise RuntimeError("Checkpoint is not a structured requirement matcher")
    tokenizer_path = checkpoint_path.parent / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path)
        if tokenizer_path.exists()
        else model.architecture_config.backbone_model_id,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    max_length = int(
        args.max_length
        or training_arguments.get("max_length", 128)
    )
    collator = trainer.StructuredJudgmentCollator(
        tokenizer,
        max_length=max_length,
        num_requirement_slots=model.architecture_config.num_requirement_slots,
    )
    loader = DataLoader(
        trainer.StructuredJudgmentDataset(evaluation_examples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model.to(device)
    structured_result, prediction_records = _evaluate_structured(
        model,
        loader,
        device,
        precision,
        matching_args=_matching_args(training_arguments),
        min_score_gap=float(training_arguments.get("ranking_min_score_gap", 0.1)),
        no_progress=args.no_progress,
    )

    baseline_result: Optional[dict[str, Any]] = None
    if not args.skip_baseline:
        baseline_id = _clean_text(args.baseline_model_id)
        baseline_model = AutoModelForSequenceClassification.from_pretrained(
            baseline_id,
            torch_dtype=precision,
            trust_remote_code=args.trust_remote_code,
        ).to(device)
        baseline_result, baseline_predictions = _evaluate_baseline(
            baseline_model,
            loader,
            device,
            precision,
            min_score_gap=float(
                training_arguments.get("ranking_min_score_gap", 0.1)
            ),
            no_progress=args.no_progress,
        )
        for record in prediction_records:
            record["untouched_modernce_score"] = baseline_predictions[
                record["pair_id"]
            ]
        del baseline_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    scope = args.evaluation_split
    predictions_path = output_dir / f"{scope}_predictions.jsonl"
    summary_path = output_dir / "evaluation_summary.json"
    summary = {
        "schema_version": "ce5.structured-evaluation.v1",
        "created_at_utc": _utc_now(),
        "checkpoint": str(checkpoint_path),
        "judgments": str(judgment_path),
        "training_config": str(training_config_path),
        "evaluation_split": args.evaluation_split,
        "split_verification": verification,
        "loading": loading_stats,
        "evaluated_examples": len(evaluation_examples),
        "ce5": structured_result,
        "untouched_modernce": baseline_result,
        "artifacts": {"predictions": str(predictions_path)},
    }
    _write_json(summary_path, summary)
    _write_jsonl(predictions_path, prediction_records)
    _print_summary(
        split_label=args.evaluation_split,
        verified=bool(verification.get("verified")),
        structured=structured_result,
        baseline=baseline_result,
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
