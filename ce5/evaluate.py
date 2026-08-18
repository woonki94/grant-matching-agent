"""Evaluate a CE5 checkpoint and inspect what its latent experts learned.

The evaluator reproduces the three-way grouped split used by
``ce5/training/train_independent_latent_heads.py``. It
reports ordinary teacher-imitation metrics, but also measures expert variance,
expert correlations, gate behavior, weighted logit contributions, and latent
attention overlap.  The untouched ModernCE checkpoint can be evaluated on the
same examples as a speed and quality baseline.

Generated files are JSON/JSONL so later analysis is not coupled to W&B:

* ``evaluation_summary.json`` -- aggregate metrics and latent diagnostics
* ``<split>_predictions.jsonl`` -- one record per evaluated pair
* ``attention_examples.jsonl`` -- high-gate and high-error explanations
* ``human_audit_sample.jsonl`` -- score-stratified examples for manual review
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import math
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_JUDGMENTS = (
    REPO_ROOT / "ce5" / "dataset" / "judgments" / "teacher_judgments_v2.jsonl"
)
DEFAULT_CHECKPOINT = REPO_ROOT / "ce5" / "models" / "latent_head_distilled_v2" / "best.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ce5" / "eval" / "results" / "latent_head_distilled_v2"
DEFAULT_BASELINE_MODEL = "dleemiller/ModernCE-base-sts"
SCORE_BINS = (
    ("0.00-0.25", 0.00, 0.25, False),
    ("0.25-0.50", 0.25, 0.50, False),
    ("0.50-0.75", 0.50, 0.75, False),
    ("0.75-1.00", 0.75, 1.00, True),
)


class ExampleDataset(Dataset[Any]):
    def __init__(self, examples: Sequence[Any]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Any:
        return self.examples[index]


class EvaluationCollator:
    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        include_pair_masks: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.include_pair_masks = bool(include_pair_masks)

    def __call__(self, examples: Sequence[Any]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [example.grant_text for example in examples],
            [example.faculty_text for example in examples],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if self.include_pair_masks:
            from ce5.training.pair_tokenization import add_pair_sequence_masks

            add_pair_sequence_masks(encoded, self.tokenizer)
        return {"encoded": encoded, "examples": list(examples)}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _resolve_path(value: Path) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_id_sha256(examples: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for pair_id in sorted(str(example.pair_id) for example in examples):
        digest.update(pair_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _verify_split_manifest(
    examples: Sequence[Any],
    *,
    split_name: str,
    manifest_path: Path,
    allow_unverified: bool,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    manifest_entry = manifest.get(split_name)
    expected_hash = (
        _clean_text(manifest_entry.get("pair_id_sha256"))
        if isinstance(manifest_entry, Mapping)
        else ""
    )
    actual_hash = _pair_id_sha256(examples)
    result: dict[str, Any] = {
        "manifest": str(manifest_path),
        "required": not allow_unverified,
        "verified": False,
        "expected_pair_id_sha256": expected_hash or None,
        "actual_pair_id_sha256": actual_hash,
        "examples_before_limit": len(examples),
    }
    if not expected_hash:
        if not allow_unverified:
            raise RuntimeError(
                f"Cannot verify the {split_name} split: missing {manifest_path}. "
                "Re-run training or use --skip-split-verification only for a "
                "legacy checkpoint."
            )
        return result
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"The reconstructed {split_name} split does not match {manifest_path}: "
            f"expected {expected_hash}, got {actual_hash}. Check the judgments "
            "file and training split arguments."
        )
    result["verified"] = True
    return result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0,1]")
    return parsed


def _safe_label(value: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()
    return label or "model"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _summarize_values(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": _mean(values),
        "std": _std(values),
        "min": min(values) if values else 0.0,
        "p05": _quantile(values, 0.05),
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p95": _quantile(values, 0.95),
        "max": max(values) if values else 0.0,
    }


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
    if len(left) != len(right) or not left:
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    denominator = left_scale * right_scale
    return numerator / denominator if denominator > 0.0 else 0.0


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _pairwise_accuracy(
    predictions: Sequence[float],
    targets: Sequence[float],
    query_ids: Sequence[str],
    *,
    min_score_gap: float,
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
                if abs(target_difference) < min_score_gap:
                    continue
                prediction_difference = predictions[left] - predictions[right]
                comparisons += 1
                if prediction_difference == 0.0:
                    correct += 0.5
                elif prediction_difference * target_difference > 0.0:
                    correct += 1.0
    return (correct / comparisons if comparisons else 0.0), comparisons


def _metrics(
    predictions: Sequence[float],
    targets: Sequence[float],
    query_ids: Sequence[str],
    *,
    min_score_gap: float,
) -> dict[str, Any]:
    if not predictions or len(predictions) != len(targets):
        raise ValueError("Predictions and targets must be non-empty and equally sized")
    errors = [prediction - target for prediction, target in zip(predictions, targets, strict=True)]
    band_names = ("low", "mid", "high")
    band_counts = {
        name: {"examples": 0, "out_of_band": 0} for name in band_names
    }
    confusion_counts = {
        teacher_band: {
            predicted_band: 0
            for predicted_band in (*band_names, "outside_[0,1]")
        }
        for teacher_band in band_names
    }
    missed_predictions: dict[str, list[float]] = {name: [] for name in band_names}
    missed_boundary_gaps: dict[str, list[float]] = {
        name: [] for name in band_names
    }
    missed_destinations = {
        teacher_band: {
            predicted_band: 0
            for predicted_band in (*band_names, "outside_[0,1]")
        }
        for teacher_band in band_names
    }
    high_miss_bins = {
        "below_0.25": 0,
        "0.25-0.50": 0,
        "0.50-0.65": 0,
        "0.65-0.75": 0,
        "outside_[0,1]": 0,
    }
    out_of_score_band = 0
    for prediction, target in zip(predictions, targets, strict=True):
        teacher_band = _oob_score_band(target)
        outside_unit_interval = prediction < 0.0 or prediction > 1.0
        predicted_band = (
            "outside_[0,1]"
            if outside_unit_interval
            else _oob_score_band(prediction)
        )
        is_out_of_band = outside_unit_interval or predicted_band != teacher_band
        band_counts[teacher_band]["examples"] += 1
        band_counts[teacher_band]["out_of_band"] += int(is_out_of_band)
        confusion_counts[teacher_band][predicted_band] += 1
        out_of_score_band += int(is_out_of_band)
        if not is_out_of_band:
            continue
        missed_predictions[teacher_band].append(float(prediction))
        missed_destinations[teacher_band][predicted_band] += 1
        missed_boundary_gaps[teacher_band].append(
            _distance_to_score_band(prediction, teacher_band)
        )
        if teacher_band == "high":
            if outside_unit_interval:
                high_miss_bins["outside_[0,1]"] += 1
            elif prediction < 0.25:
                high_miss_bins["below_0.25"] += 1
            elif prediction < 0.50:
                high_miss_bins["0.25-0.50"] += 1
            elif prediction < 0.65:
                high_miss_bins["0.50-0.65"] += 1
            else:
                high_miss_bins["0.65-0.75"] += 1
    out_of_band_by_teacher_band = {
        name: {
            **counts,
            "ratio": counts["out_of_band"] / counts["examples"]
            if counts["examples"]
            else None,
        }
        for name, counts in band_counts.items()
    }
    score_band_confusion = {
        teacher_band: {
            "teacher_examples": band_counts[teacher_band]["examples"],
            "counts": confusion_counts[teacher_band],
            "ratios": {
                predicted_band: count / band_counts[teacher_band]["examples"]
                if band_counts[teacher_band]["examples"]
                else None
                for predicted_band, count in confusion_counts[teacher_band].items()
            },
        }
        for teacher_band in band_names
    }
    oob_placement = {
        teacher_band: {
            "teacher_examples": band_counts[teacher_band]["examples"],
            "out_of_band_examples": band_counts[teacher_band]["out_of_band"],
            "destination_counts": missed_destinations[teacher_band],
            "destination_ratios_of_teacher_band": {
                predicted_band: count / band_counts[teacher_band]["examples"]
                if band_counts[teacher_band]["examples"]
                else None
                for predicted_band, count in missed_destinations[teacher_band].items()
            },
            "missed_prediction_summary": _summarize_values(
                missed_predictions[teacher_band]
            ),
            "distance_to_nearest_correct_band": _summarize_values(
                missed_boundary_gaps[teacher_band]
            ),
        }
        for teacher_band in band_names
    }
    high_examples = band_counts["high"]["examples"]
    high_oob_placement = {
        "teacher_examples": high_examples,
        "counts": high_miss_bins,
        "ratios_of_teacher_high": {
            name: count / high_examples if high_examples else None
            for name, count in high_miss_bins.items()
        },
    }
    outside_unit_interval = sum(
        prediction < 0.0 or prediction > 1.0 for prediction in predictions
    )
    mse = _mean([error * error for error in errors])
    pairwise, pair_count = _pairwise_accuracy(
        predictions,
        targets,
        query_ids,
        min_score_gap=min_score_gap,
    )
    return {
        "examples": len(predictions),
        "mae": _mean([abs(error) for error in errors]),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "pearson": _pearson(predictions, targets),
        "spearman": _spearman(predictions, targets),
        "pairwise_accuracy": pairwise,
        "pairwise_comparisons": pair_count,
        "prediction_mean": _mean(predictions),
        "prediction_std": _std(predictions),
        "target_mean": _mean(targets),
        "target_std": _std(targets),
        "mean_error": _mean(errors),
        "out_of_score_band_ratio": out_of_score_band / len(predictions),
        "out_of_score_band_by_teacher_band": out_of_band_by_teacher_band,
        "score_band_confusion": score_band_confusion,
        "oob_placement_by_teacher_band": oob_placement,
        "high_oob_placement": high_oob_placement,
        "outside_unit_interval_ratio": outside_unit_interval / len(predictions),
    }


def _score_bin(score: float) -> str:
    for name, lower, upper, include_upper in SCORE_BINS:
        if score >= lower and (score <= upper if include_upper else score < upper):
            return name
    return SCORE_BINS[-1][0]


def _oob_score_band(score: float) -> str:
    if score < 0.25:
        return "low"
    if score < 0.75:
        return "mid"
    return "high"


def _distance_to_score_band(score: float, band: str) -> float:
    if band == "low":
        if score < 0.0:
            return -score
        return max(0.0, score - 0.25)
    if band == "mid":
        if score < 0.25:
            return 0.25 - score
        return max(0.0, score - 0.75)
    if band == "high":
        if score > 1.0:
            return score - 1.0
        return max(0.0, 0.75 - score)
    raise ValueError(f"Unsupported score band: {band}")


def _correlation_matrix(
    values_by_name: Mapping[str, Sequence[float]],
    *,
    rank: bool = False,
) -> dict[str, dict[str, float]]:
    names = list(values_by_name)
    output: dict[str, dict[str, float]] = {}
    for left_name in names:
        output[left_name] = {}
        for right_name in names:
            correlation = (
                _spearman(values_by_name[left_name], values_by_name[right_name])
                if rank
                else _pearson(values_by_name[left_name], values_by_name[right_name])
            )
            output[left_name][right_name] = correlation
    return output


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _resolve_precision(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[requested]


def _autocast(device: torch.device, precision: torch.dtype) -> Any:
    if device.type == "cuda" and precision in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=precision)
    return torch.autocast(device_type="cpu", enabled=False)


def _move_encoded(encoded: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in encoded.items()
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_start(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _memory_summary(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    divisor = 1024.0**3
    return {
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / divisor,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / divisor,
    }


def _make_loader(
    examples: Sequence[Any],
    tokenizer: Any,
    *,
    batch_size: int,
    max_length: int,
    num_workers: int,
    device: torch.device,
    include_pair_masks: bool = False,
) -> DataLoader[Any]:
    return DataLoader(
        ExampleDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=EvaluationCollator(
            tokenizer,
            max_length=max_length,
            include_pair_masks=include_pair_masks,
        ),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def _top_attention_tokens(
    tokenizer: Any,
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
    weights: Sequence[float],
    token_type_ids: Optional[Sequence[int]],
    *,
    maximum: int,
) -> list[dict[str, Any]]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    sep_id = getattr(tokenizer, "sep_token_id", None)
    first_separator: Optional[int] = None
    if sep_id is not None:
        first_separator = next(
            (index for index, token_id in enumerate(input_ids) if token_id == sep_id),
            None,
        )
    valid = [
        index
        for index, (token_id, mask) in enumerate(zip(input_ids, attention_mask, strict=True))
        if mask and token_id not in special_ids
    ]
    valid.sort(key=lambda index: weights[index], reverse=True)
    tokens = tokenizer.convert_ids_to_tokens(list(input_ids))
    output: list[dict[str, Any]] = []
    for position in valid[:maximum]:
        token = str(tokens[position])
        try:
            readable = _clean_text(tokenizer.convert_tokens_to_string([token])) or token
        except Exception:
            readable = token
        if token_type_ids is not None:
            segment = "faculty" if int(token_type_ids[position]) > 0 else "grant"
        elif first_separator is not None and position > first_separator:
            segment = "faculty"
        else:
            segment = "grant"
        output.append(
            {
                "position": position,
                "segment": segment,
                "token": token,
                "text": readable,
                "attention": float(weights[position]),
            }
        )
    return output


def _push_top(
    heap: list[tuple[float, int, dict[str, Any]]],
    *,
    key: float,
    sequence: int,
    record: dict[str, Any],
    limit: int,
) -> None:
    if limit <= 0:
        return
    item = (float(key), sequence, record)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif key > heap[0][0]:
        heapq.heapreplace(heap, item)


def _base_prediction_record(
    example: Any,
    *,
    prediction: float,
    expert_names: Sequence[str],
    expert_scores: Sequence[float],
    expert_logits: Sequence[float],
    gates: Sequence[float],
    contributions: Sequence[float],
) -> dict[str, Any]:
    return {
        "pair_id": example.pair_id,
        "grant_id": example.grant_id,
        "grant_item_id": example.grant_item_id,
        "faculty_id": example.faculty_id,
        "faculty_item_id": example.faculty_item_id,
        "grant_text": example.grant_text,
        "faculty_text": example.faculty_text,
        "teacher_score": float(example.score),
        "teacher_confidence": float(example.confidence),
        "prefilter_band": example.prefilter_band or "unknown",
        "teacher_score_bin": _score_bin(float(example.score)),
        "ce5_score": prediction,
        "ce5_error": prediction - float(example.score),
        "ce5_absolute_error": abs(prediction - float(example.score)),
        "gate_winner": expert_names[max(range(len(gates)), key=gates.__getitem__)],
        "gate_weights": dict(zip(expert_names, gates, strict=True)),
        "expert_scores": dict(zip(expert_names, expert_scores, strict=True)),
        "expert_logits": dict(zip(expert_names, expert_logits, strict=True)),
        "weighted_logit_contributions": dict(
            zip(expert_names, contributions, strict=True)
        ),
    }


def _conditioned_expert_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    group_field: str,
    expert_names: Sequence[str],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get(group_field) or "unknown"), []).append(record)
    output: dict[str, Any] = {}
    for group_name, rows in sorted(grouped.items()):
        output[group_name] = {
            "examples": len(rows),
            "mean_gate": {
                expert: _mean(
                    [float(row["gate_weights"][expert]) for row in rows]
                )
                for expert in expert_names
            },
            "gate_winner_fraction": {
                expert: sum(row["gate_winner"] == expert for row in rows) / len(rows)
                for expert in expert_names
            },
            "mean_expert_score": {
                expert: _mean(
                    [float(row["expert_scores"][expert]) for row in rows]
                )
                for expert in expert_names
            },
        }
    return output


def _metrics_by_group(
    records: Sequence[Mapping[str, Any]],
    *,
    group_field: str,
    prediction_field: str,
    min_score_gap: float,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get(group_field) or "unknown"), []).append(record)
    output: dict[str, Any] = {}
    for group_name, rows in sorted(grouped.items()):
        output[group_name] = _metrics(
            [float(row[prediction_field]) for row in rows],
            [float(row["teacher_score"]) for row in rows],
            [str(row["grant_item_id"]) for row in rows],
            min_score_gap=min_score_gap,
        )
    return output


@torch.inference_mode()
def _evaluate_latent_model(
    model: Any,
    loader: DataLoader[Any],
    tokenizer: Any,
    device: torch.device,
    precision: torch.dtype,
    *,
    min_score_gap: float,
    attention_examples_per_head: int,
    largest_error_examples: int,
    top_attention_tokens: int,
    no_progress: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    expert_names = tuple(model.expert_names)
    uses_base_expert = bool(model.architecture_config.use_base_sts_expert)
    latent_names = tuple(
        f"latent_{index}"
        for index in range(model.architecture_config.num_latent_heads)
    )
    records: list[dict[str, Any]] = []
    head_scores: dict[str, list[float]] = {name: [] for name in expert_names}
    gates: dict[str, list[float]] = {name: [] for name in expert_names}
    contributions: dict[str, list[float]] = {name: [] for name in expert_names}
    predicted_expert_errors: dict[str, list[float]] = {
        name: [] for name in expert_names
    }
    actual_expert_errors: dict[str, list[float]] = {
        name: [] for name in expert_names
    }
    reliability_top1_correct = 0
    reliability_examples = 0
    latent_routing: dict[str, list[float]] = {name: [] for name in latent_names}
    latent_contributions: list[float] = []
    gate_entropies: list[float] = []
    normalized_gate_entropies: list[float] = []
    attention_similarity_sum = torch.zeros(
        (len(latent_names), len(latent_names)), dtype=torch.float64
    )
    attention_similarity_examples = 0
    gate_heaps: dict[str, list[tuple[float, int, dict[str, Any]]]] = {
        name: [] for name in latent_names
    }
    error_heap: list[tuple[float, int, dict[str, Any]]] = []
    sequence = 0
    inference_seconds = 0.0

    progress: Iterable[Any] = loader
    if tqdm is not None and not no_progress:
        progress = tqdm(loader, desc="CE5 checkpoint evaluation", dynamic_ncols=True)
    _memory_start(device)
    wall_started = time.perf_counter()
    for batch in progress:
        encoded_cpu = batch["encoded"]
        encoded = _move_encoded(encoded_cpu, device)
        _sync(device)
        forward_started = time.perf_counter()
        with _autocast(device, precision):
            output = model(**encoded)
        _sync(device)
        inference_seconds += time.perf_counter() - forward_started

        batch_scores = output.scores.detach().float().cpu()
        batch_head_scores = output.head_scores.detach().float().cpu()
        batch_head_logits = output.head_logits.detach().float().cpu()
        batch_gates = output.gate_weights.detach().float().cpu()
        batch_contributions = batch_gates * batch_head_logits
        output_predicted_errors = getattr(
            output,
            "predicted_expert_errors",
            None,
        )
        batch_predicted_errors = (
            output_predicted_errors.detach().float().cpu()
            if output_predicted_errors is not None
            else None
        )
        batch_attention = output.attention_weights.detach().float().cpu()
        output_target_attention = getattr(output, "target_attention_weights", None)
        batch_target_attention = (
            output_target_attention.detach().float().cpu()
            if output_target_attention is not None
            else None
        )
        output_candidate_attention = getattr(
            output,
            "candidate_attention_weights",
            None,
        )
        batch_candidate_attention = (
            output_candidate_attention.detach().float().cpu()
            if output_candidate_attention is not None
            else None
        )
        output_routing = getattr(output, "routing_weights", None)
        batch_routing = (
            output_routing.detach().float().cpu()
            if output_routing is not None
            else None
        )
        output_latent_contribution = getattr(output, "latent_contribution", None)
        batch_latent_contribution = (
            output_latent_contribution.detach().float().cpu()
            if output_latent_contribution is not None
            else None
        )
        attention_mask = encoded_cpu["attention_mask"].cpu()
        input_ids = encoded_cpu["input_ids"].cpu()
        token_types = encoded_cpu.get("token_type_ids")
        token_types = token_types.cpu() if token_types is not None else None

        normalized_attention = torch.nn.functional.normalize(
            batch_attention, p=2, dim=-1, eps=1e-8
        )
        attention_similarity_sum += torch.bmm(
            normalized_attention,
            normalized_attention.transpose(1, 2),
        ).sum(dim=0).double()
        attention_similarity_examples += int(batch_attention.shape[0])

        for row_index, example in enumerate(batch["examples"]):
            prediction = float(batch_scores[row_index].item())
            expert_score_values = batch_head_scores[row_index].tolist()
            expert_logit_values = batch_head_logits[row_index].tolist()
            gate_values = batch_gates[row_index].tolist()
            contribution_values = batch_contributions[row_index].tolist()
            record = _base_prediction_record(
                example,
                prediction=prediction,
                expert_names=expert_names,
                expert_scores=expert_score_values,
                expert_logits=expert_logit_values,
                gates=gate_values,
                contributions=contribution_values,
            )
            if batch_predicted_errors is not None:
                predicted_error_values = batch_predicted_errors[
                    row_index
                ].tolist()
                actual_error_values = [
                    abs(float(score) - float(example.score))
                    for score in expert_score_values
                ]
                record["predicted_expert_errors"] = dict(
                    zip(expert_names, predicted_error_values, strict=True)
                )
                record["actual_expert_errors"] = dict(
                    zip(expert_names, actual_error_values, strict=True)
                )
                predicted_winner = min(
                    range(len(expert_names)),
                    key=predicted_error_values.__getitem__,
                )
                oracle_winner = min(
                    range(len(expert_names)),
                    key=actual_error_values.__getitem__,
                )
                reliability_top1_correct += int(predicted_winner == oracle_winner)
                reliability_examples += 1
                for expert_index, expert_name in enumerate(expert_names):
                    predicted_expert_errors[expert_name].append(
                        float(predicted_error_values[expert_index])
                    )
                    actual_expert_errors[expert_name].append(
                        float(actual_error_values[expert_index])
                    )
            if batch_routing is not None:
                routing_values = batch_routing[row_index].tolist()
                record["latent_routing_weights"] = dict(
                    zip(latent_names, routing_values, strict=True)
                )
                for latent_index, latent_name in enumerate(latent_names):
                    latent_routing[latent_name].append(
                        float(routing_values[latent_index])
                    )
            if batch_latent_contribution is not None:
                contribution = float(batch_latent_contribution[row_index].item())
                record["latent_contribution"] = contribution
                latent_contributions.append(contribution)
            records.append(record)
            for expert_index, expert_name in enumerate(expert_names):
                head_scores[expert_name].append(float(expert_score_values[expert_index]))
                gates[expert_name].append(float(gate_values[expert_index]))
                contributions[expert_name].append(
                    float(contribution_values[expert_index])
                )
            entropy = -sum(
                gate * math.log(max(gate, 1e-12)) for gate in gate_values
            )
            gate_entropies.append(entropy)
            normalized_gate_entropies.append(
                entropy / math.log(len(expert_names)) if len(expert_names) > 1 else 0.0
            )

            ids = [int(value) for value in input_ids[row_index].tolist()]
            mask = [int(value) for value in attention_mask[row_index].tolist()]
            types = (
                [int(value) for value in token_types[row_index].tolist()]
                if token_types is not None
                else None
            )
            for latent_index, latent_name in enumerate(latent_names):
                expert_index = latent_index + int(uses_base_expert)
                gate_value = float(gate_values[expert_index])
                heap = gate_heaps[latent_name]
                if attention_examples_per_head > 0 and (
                    len(heap) < attention_examples_per_head
                    or gate_value > heap[0][0]
                ):
                    attention_record = {
                        **record,
                        "selection_type": "highest_gate_for_expert",
                        "selected_expert": latent_name,
                        "selected_gate_weight": gate_value,
                        "top_attention_tokens": _top_attention_tokens(
                            tokenizer,
                            ids,
                            mask,
                            batch_attention[row_index, latent_index].tolist(),
                            types,
                            maximum=top_attention_tokens,
                        ),
                    }
                    if (
                        batch_target_attention is not None
                        and batch_candidate_attention is not None
                    ):
                        attention_record["top_target_attention_tokens"] = (
                            _top_attention_tokens(
                                tokenizer,
                                ids,
                                mask,
                                batch_target_attention[
                                    row_index,
                                    latent_index,
                                ].tolist(),
                                types,
                                maximum=top_attention_tokens,
                            )
                        )
                        attention_record["top_candidate_attention_tokens"] = (
                            _top_attention_tokens(
                                tokenizer,
                                ids,
                                mask,
                                batch_candidate_attention[
                                    row_index,
                                    latent_index,
                                ].tolist(),
                                types,
                                maximum=top_attention_tokens,
                            )
                        )
                    _push_top(
                        heap,
                        key=gate_value,
                        sequence=sequence,
                        record=attention_record,
                        limit=attention_examples_per_head,
                    )
                    sequence += 1

            absolute_error = float(record["ce5_absolute_error"])
            if largest_error_examples > 0 and (
                len(error_heap) < largest_error_examples
                or absolute_error > error_heap[0][0]
            ):
                error_record = {
                    **record,
                    "selection_type": "largest_absolute_error",
                    "attention_by_latent": {
                        latent_name: _top_attention_tokens(
                            tokenizer,
                            ids,
                            mask,
                            batch_attention[row_index, latent_index].tolist(),
                            types,
                            maximum=top_attention_tokens,
                        )
                        for latent_index, latent_name in enumerate(latent_names)
                    },
                }
                if (
                    batch_target_attention is not None
                    and batch_candidate_attention is not None
                ):
                    error_record["target_attention_by_latent"] = {
                        latent_name: _top_attention_tokens(
                            tokenizer,
                            ids,
                            mask,
                            batch_target_attention[
                                row_index,
                                latent_index,
                            ].tolist(),
                            types,
                            maximum=top_attention_tokens,
                        )
                        for latent_index, latent_name in enumerate(latent_names)
                    }
                    error_record["candidate_attention_by_latent"] = {
                        latent_name: _top_attention_tokens(
                            tokenizer,
                            ids,
                            mask,
                            batch_candidate_attention[
                                row_index,
                                latent_index,
                            ].tolist(),
                            types,
                            maximum=top_attention_tokens,
                        )
                        for latent_index, latent_name in enumerate(latent_names)
                    }
                _push_top(
                    error_heap,
                    key=absolute_error,
                    sequence=sequence,
                    record=error_record,
                    limit=largest_error_examples,
                )
                sequence += 1

    wall_seconds = time.perf_counter() - wall_started
    predictions = [float(record["ce5_score"]) for record in records]
    targets = [float(record["teacher_score"]) for record in records]
    query_ids = [str(record["grant_item_id"]) for record in records]
    expert_diagnostics: dict[str, Any] = {}
    for expert_name in expert_names:
        absolute_contribution = [abs(value) for value in contributions[expert_name]]
        expert_diagnostics[expert_name] = {
            "scores": _summarize_values(head_scores[expert_name]),
            "score_pearson_with_teacher": _pearson(
                head_scores[expert_name], targets
            ),
            "score_spearman_with_teacher": _spearman(
                head_scores[expert_name], targets
            ),
            "gate_weights": _summarize_values(gates[expert_name]),
            "weighted_logit_contribution": _summarize_values(
                contributions[expert_name]
            ),
            "mean_absolute_logit_contribution": _mean(absolute_contribution),
            "gate_winner_fraction": sum(
                record["gate_winner"] == expert_name for record in records
            )
            / len(records),
        }
        if predicted_expert_errors[expert_name]:
            reliability_residuals = [
                predicted - actual
                for predicted, actual in zip(
                    predicted_expert_errors[expert_name],
                    actual_expert_errors[expert_name],
                    strict=True,
                )
            ]
            expert_diagnostics[expert_name]["reliability"] = {
                "predicted_error": _summarize_values(
                    predicted_expert_errors[expert_name]
                ),
                "actual_error": _summarize_values(
                    actual_expert_errors[expert_name]
                ),
                "mae": _mean([abs(value) for value in reliability_residuals]),
                "bias": _mean(reliability_residuals),
                "pearson": _pearson(
                    predicted_expert_errors[expert_name],
                    actual_expert_errors[expert_name],
                ),
                "spearman": _spearman(
                    predicted_expert_errors[expert_name],
                    actual_expert_errors[expert_name],
                ),
            }

    absolute_sums = [
        sum(abs(float(value)) for value in record["weighted_logit_contributions"].values())
        for record in records
    ]
    for expert_name in expert_names:
        shares = [
            abs(float(record["weighted_logit_contributions"][expert_name]))
            / max(absolute_sums[index], 1e-12)
            for index, record in enumerate(records)
        ]
        expert_diagnostics[expert_name]["mean_absolute_contribution_share"] = _mean(
            shares
        )

    similarity = (
        attention_similarity_sum / max(1, attention_similarity_examples)
    ).tolist()
    attention_similarity = {
        left_name: {
            right_name: float(similarity[left_index][right_index])
            for right_index, right_name in enumerate(latent_names)
        }
        for left_index, left_name in enumerate(latent_names)
    }
    attention_examples = [
        item[2]
        for latent_name in latent_names
        for item in sorted(gate_heaps[latent_name], reverse=True)
    ]
    attention_examples.extend(
        item[2] for item in sorted(error_heap, reverse=True)
    )
    diagnostics = {
        "metrics": _metrics(
            predictions,
            targets,
            query_ids,
            min_score_gap=min_score_gap,
        ),
        "metrics_by_teacher_score_bin": _metrics_by_group(
            records,
            group_field="teacher_score_bin",
            prediction_field="ce5_score",
            min_score_gap=min_score_gap,
        ),
        "metrics_by_prefilter_band": _metrics_by_group(
            records,
            group_field="prefilter_band",
            prediction_field="ce5_score",
            min_score_gap=min_score_gap,
        ),
        "expert_diagnostics": expert_diagnostics,
        "expert_score_pearson_matrix": _correlation_matrix(head_scores),
        "expert_score_spearman_matrix": _correlation_matrix(
            head_scores, rank=True
        ),
        "latent_attention_cosine_similarity_matrix": attention_similarity,
        "gate_entropy": _summarize_values(gate_entropies),
        "normalized_gate_entropy": _summarize_values(normalized_gate_entropies),
        "conditioned_by_teacher_score_bin": _conditioned_expert_summary(
            records,
            group_field="teacher_score_bin",
            expert_names=expert_names,
        ),
        "conditioned_by_prefilter_band": _conditioned_expert_summary(
            records,
            group_field="prefilter_band",
            expert_names=expert_names,
        ),
        "possible_score_anchor_order": sorted(
            expert_names,
            key=lambda name: expert_diagnostics[name]["scores"]["mean"],
        ),
        "runtime": {
            "wall_seconds": wall_seconds,
            "model_inference_seconds": inference_seconds,
            "examples_per_second_wall": len(records) / max(wall_seconds, 1e-12),
            "examples_per_second_model": len(records)
            / max(inference_seconds, 1e-12),
            "batches": len(loader),
            **_memory_summary(device),
        },
    }
    if latent_contributions:
        diagnostics["latent_contribution"] = _summarize_values(
            latent_contributions
        )
    if any(latent_routing.values()):
        diagnostics["latent_routing_weights"] = {
            name: _summarize_values(values)
            for name, values in latent_routing.items()
        }
    if reliability_examples:
        diagnostics["reliability_routing"] = {
            "examples": reliability_examples,
            "predicted_lowest_error_matches_oracle_fraction": (
                reliability_top1_correct / reliability_examples
            ),
        }
    return diagnostics, records, attention_examples


@torch.inference_mode()
def _mean_gate_weights(
    model: Any,
    loader: DataLoader[Any],
    device: torch.device,
    precision: torch.dtype,
    *,
    no_progress: bool,
) -> dict[str, float]:
    """Estimate fixed deployment weights without reading teacher labels."""

    model.eval()
    gate_sum: Optional[Tensor] = None
    example_count = 0
    progress: Iterable[Any] = loader
    if tqdm is not None and not no_progress:
        progress = tqdm(
            loader,
            desc="Validation gates for static-router ablation",
            dynamic_ncols=True,
        )
    for batch in progress:
        encoded = _move_encoded(batch["encoded"], device)
        with _autocast(device, precision):
            output = model(**encoded)
        batch_gates = output.gate_weights.detach().float().sum(dim=0).cpu()
        gate_sum = batch_gates if gate_sum is None else gate_sum + batch_gates
        example_count += int(output.gate_weights.shape[0])
    if gate_sum is None or example_count == 0:
        raise RuntimeError("Static-router calibration split is empty")
    weights = gate_sum / example_count
    return {
        name: float(weights[index].item())
        for index, name in enumerate(model.expert_names)
    }


def _sigmoid_scalar(logit: float) -> float:
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _router_ablation_diagnostics(
    records: Sequence[dict[str, Any]],
    *,
    expert_names: Sequence[str],
    static_gate_weights: Mapping[str, float],
    min_score_gap: float,
) -> dict[str, Any]:
    """Score counterfactual routers from one set of cached expert logits."""

    if not records:
        raise ValueError("Router ablations require prediction records")
    if set(static_gate_weights) != set(expert_names):
        raise ValueError("Static gate weights do not match checkpoint experts")
    static_total = sum(float(static_gate_weights[name]) for name in expert_names)
    if static_total <= 0.0:
        raise ValueError("Static gate weights must have positive total mass")
    normalized_static = {
        name: float(static_gate_weights[name]) / static_total
        for name in expert_names
    }
    uniform_weight = 1.0 / len(expert_names)
    prediction_lists: dict[str, list[float]] = {
        "dynamic_router": [],
        "static_validation_mean": [],
        "uniform": [],
    }
    if "base_sts" in expert_names:
        prediction_lists["base_sts_only"] = []
    for name in expert_names:
        if name != "base_sts":
            prediction_lists[f"expert_{name}"] = []
    prediction_lists["oracle_best_expert"] = []
    oracle_selections = {name: 0 for name in expert_names}

    targets: list[float] = []
    query_ids: list[str] = []
    for record in records:
        raw_logits = record.get("expert_logits")
        if not isinstance(raw_logits, Mapping):
            raise RuntimeError("Prediction record is missing expert logits")
        logits = {name: float(raw_logits[name]) for name in expert_names}
        target = float(record["teacher_score"])
        dynamic_score = float(record["ce5_score"])
        static_logit = sum(
            normalized_static[name] * logits[name] for name in expert_names
        )
        uniform_logit = uniform_weight * sum(logits.values())
        expert_scores = {
            name: _sigmoid_scalar(logits[name]) for name in expert_names
        }
        oracle_name = min(
            expert_names,
            key=lambda name: abs(expert_scores[name] - target),
        )
        oracle_selections[oracle_name] += 1
        ablation_scores: dict[str, float] = {
            "dynamic_router": dynamic_score,
            "static_validation_mean": _sigmoid_scalar(static_logit),
            "uniform": _sigmoid_scalar(uniform_logit),
            "oracle_best_expert": expert_scores[oracle_name],
        }
        if "base_sts" in expert_names:
            ablation_scores["base_sts_only"] = expert_scores["base_sts"]
        for name in expert_names:
            if name != "base_sts":
                ablation_scores[f"expert_{name}"] = expert_scores[name]
        record["router_ablation_scores"] = ablation_scores
        record["oracle_best_expert"] = oracle_name
        for method_name, score in ablation_scores.items():
            prediction_lists[method_name].append(score)
        targets.append(target)
        query_ids.append(str(record["grant_item_id"]))

    methods = {
        method_name: {
            "metrics": _metrics(
                predictions,
                targets,
                query_ids,
                min_score_gap=min_score_gap,
            )
        }
        for method_name, predictions in prediction_lists.items()
    }
    dynamic_metrics = methods["dynamic_router"]["metrics"]
    for method_name, result in methods.items():
        if method_name == "dynamic_router":
            continue
        method_metrics = result["metrics"]
        result["delta_vs_dynamic"] = {
            metric: float(method_metrics[metric]) - float(dynamic_metrics[metric])
            for metric in (
                "mae",
                "rmse",
                "pearson",
                "spearman",
                "pairwise_accuracy",
            )
        }
    return {
        "mixture_space": "logit",
        "static_gate_weights": normalized_static,
        "methods": methods,
        "oracle": {
            "teacher_aware": True,
            "deployable": False,
            "selection_counts": oracle_selections,
            "selection_fractions": {
                name: count / len(records)
                for name, count in oracle_selections.items()
            },
        },
    }


@torch.inference_mode()
def _evaluate_sequence_classifier(
    model: Any,
    loader: DataLoader[Any],
    device: torch.device,
    precision: torch.dtype,
    *,
    min_score_gap: float,
    description: str,
    no_progress: bool,
) -> tuple[dict[str, Any], dict[str, float]]:
    model.eval()
    predictions_by_pair: dict[str, float] = {}
    targets: list[float] = []
    predictions: list[float] = []
    query_ids: list[str] = []
    inference_seconds = 0.0
    progress: Iterable[Any] = loader
    if tqdm is not None and not no_progress:
        progress = tqdm(loader, desc=description, dynamic_ncols=True)
    _memory_start(device)
    wall_started = time.perf_counter()
    for batch in progress:
        encoded = _move_encoded(batch["encoded"], device)
        _sync(device)
        forward_started = time.perf_counter()
        with _autocast(device, precision):
            output = model(**encoded)
        _sync(device)
        inference_seconds += time.perf_counter() - forward_started
        logits = output.logits.detach().float().cpu()
        if logits.ndim == 2 and logits.shape[-1] == 1:
            batch_scores = torch.sigmoid(logits[:, 0])
        elif logits.ndim == 2 and logits.shape[-1] == 2:
            batch_scores = torch.softmax(logits, dim=-1)[:, 1]
        else:
            raise RuntimeError(
                f"Expected one or two classifier logits, received {tuple(logits.shape)}"
            )
        for example, score in zip(batch["examples"], batch_scores.tolist(), strict=True):
            numeric_score = float(score)
            predictions_by_pair[example.pair_id] = numeric_score
            predictions.append(numeric_score)
            targets.append(float(example.score))
            query_ids.append(example.grant_item_id)
    wall_seconds = time.perf_counter() - wall_started
    result = {
        "metrics": _metrics(
            predictions,
            targets,
            query_ids,
            min_score_gap=min_score_gap,
        ),
        "runtime": {
            "wall_seconds": wall_seconds,
            "model_inference_seconds": inference_seconds,
            "examples_per_second_wall": len(predictions) / max(wall_seconds, 1e-12),
            "examples_per_second_model": len(predictions)
            / max(inference_seconds, 1e-12),
            "batches": len(loader),
            **_memory_summary(device),
        },
    }
    return result, predictions_by_pair


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_teacher_metadata(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            pair_id = _clean_text(row.get("pair_id"))
            if pair_id:
                output[pair_id] = {
                    "teacher_model": _clean_text(row.get("teacher_model")),
                    "teacher_rationale": _clean_text(row.get("rationale")),
                    "judgment_id": _clean_text(row.get("judgment_id")),
                }
    return output


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _audit_sample(
    records: Sequence[Mapping[str, Any]],
    *,
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if size <= 0:
        return []
    grouped: dict[str, list[Mapping[str, Any]]] = {
        name: [] for name, *_ in SCORE_BINS
    }
    for record in records:
        grouped[str(record["teacher_score_bin"])].append(record)
    rng = random.Random(seed)
    selected: list[Mapping[str, Any]] = []
    per_bin = max(1, math.ceil(min(size, len(records)) / len(SCORE_BINS)))
    for name, *_ in SCORE_BINS:
        candidates = list(grouped[name])
        rng.shuffle(candidates)
        selected.extend(candidates[:per_bin])
    selected_ids = {str(record["pair_id"]) for record in selected}
    if len(selected) < min(size, len(records)):
        remaining = [
            record for record in records if str(record["pair_id"]) not in selected_ids
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: size - len(selected)])
    selected = selected[:size]
    output: list[dict[str, Any]] = []
    for record in selected:
        output.append(
            {
                **dict(record),
                "review_status": "pending",
                "human_score": None,
                "human_confidence": None,
                "human_notes": "",
            }
        )
    return output


def _comparison_delta(
    ce5_metrics: Mapping[str, Any], baseline_metrics: Mapping[str, Any]
) -> dict[str, float]:
    output: dict[str, float] = {}
    for metric in ("mae", "rmse", "pearson", "spearman", "pairwise_accuracy"):
        if metric in ce5_metrics and metric in baseline_metrics:
            output[f"ce5_minus_baseline_{metric}"] = float(ce5_metrics[metric]) - float(
                baseline_metrics[metric]
            )
    return output


def _print_console_summary(
    *,
    split_name: str,
    examples: int,
    split_verified: bool,
    ce5_result: Mapping[str, Any],
    ce5_method_name: str = "CE5 latent-head",
    comparisons: Mapping[str, Any],
    summary_path: Path,
) -> None:
    methods: list[tuple[str, Mapping[str, Any]]] = []
    if "untouched_modernce" in comparisons:
        methods.append(("Untouched STS", comparisons["untouched_modernce"]))
    if "fine_tuned_single_head" in comparisons:
        methods.append(("Fine-tuned CE", comparisons["fine_tuned_single_head"]))
    methods.append((ce5_method_name, ce5_result))

    columns = (
        ("Method", 17),
        ("MAE", 8),
        ("RMSE", 8),
        ("Pearson", 9),
        ("Spearman", 9),
        ("PairAcc", 9),
        ("Low OOB", 9),
        ("Mid OOB", 9),
        ("High OOB", 9),
        ("Ex/s", 9),
    )
    header = "  ".join(label.ljust(width) for label, width in columns)
    print("\n=== CE5 evaluation ===")
    verification = "VERIFIED" if split_verified else "NOT VERIFIED"
    print(f"Split: {split_name} | examples: {examples:,} | manifest: {verification}")
    print(header)
    print("  ".join("-" * width for _, width in columns))
    for method_name, result in methods:
        metrics = result["metrics"]
        runtime = result.get("runtime", {})
        oob_by_band = metrics["out_of_score_band_by_teacher_band"]

        def format_oob(name: str, width: int) -> str:
            ratio = oob_by_band[name]["ratio"]
            return (
                "n/a" if ratio is None else f"{100.0 * float(ratio):.2f}%"
            ).rjust(width)

        values = (
            method_name.ljust(columns[0][1]),
            f"{float(metrics['mae']):.4f}".rjust(columns[1][1]),
            f"{float(metrics['rmse']):.4f}".rjust(columns[2][1]),
            f"{float(metrics['pearson']):.4f}".rjust(columns[3][1]),
            f"{float(metrics['spearman']):.4f}".rjust(columns[4][1]),
            f"{100.0 * float(metrics['pairwise_accuracy']):.2f}%".rjust(columns[5][1]),
            format_oob("low", columns[6][1]),
            format_oob("mid", columns[7][1]),
            format_oob("high", columns[8][1]),
            f"{float(runtime.get('examples_per_second_model', 0.0)):,.0f}".rjust(columns[9][1]),
        )
        print("  ".join(values))

    print("\nHigh-OOB placement (% of all teacher-high examples)")
    print(
        "Method             <0.25     0.25-.50  0.50-.65  0.65-.75  "
        "Miss p50  Gap p50"
    )
    print(
        "-----------------  --------  --------  --------  --------  "
        "--------  -------"
    )
    for method_name, result in methods:
        metrics = result["metrics"]
        placement = metrics["high_oob_placement"]
        ratios = placement["ratios_of_teacher_high"]
        high_oob = metrics["oob_placement_by_teacher_band"]["high"]
        missed_count = int(high_oob["out_of_band_examples"])

        def placement_percent(name: str) -> str:
            ratio = ratios[name]
            return "n/a" if ratio is None else f"{100.0 * float(ratio):.2f}%"

        missed_median = (
            f"{float(high_oob['missed_prediction_summary']['median']):.3f}"
            if missed_count
            else "n/a"
        )
        gap_median = (
            f"{float(high_oob['distance_to_nearest_correct_band']['median']):.3f}"
            if missed_count
            else "n/a"
        )
        print(
            f"{method_name:<17}  "
            f"{placement_percent('below_0.25'):>8}  "
            f"{placement_percent('0.25-0.50'):>8}  "
            f"{placement_percent('0.50-0.65'):>8}  "
            f"{placement_percent('0.65-0.75'):>8}  "
            f"{missed_median:>8}  {gap_median:>7}"
        )

    for comparison_name, label in (
        ("untouched_modernce", "untouched STS"),
        ("fine_tuned_single_head", "fine-tuned CE"),
    ):
        comparison = comparisons.get(comparison_name)
        if not isinstance(comparison, Mapping):
            continue
        baseline = comparison["metrics"]
        ce5_metrics = ce5_result["metrics"]
        mae_change = 100.0 * (
            float(ce5_metrics["mae"]) / max(float(baseline["mae"]), 1e-12) - 1.0
        )
        rmse_change = 100.0 * (
            float(ce5_metrics["rmse"]) / max(float(baseline["rmse"]), 1e-12) - 1.0
        )
        pearson_change = float(ce5_metrics["pearson"]) - float(baseline["pearson"])
        spearman_change = float(ce5_metrics["spearman"]) - float(baseline["spearman"])
        pairwise_change = 100.0 * (
            float(ce5_metrics["pairwise_accuracy"])
            - float(baseline["pairwise_accuracy"])
        )
        print(
            f"CE5 vs {label}: MAE {mae_change:+.2f}%, "
            f"RMSE {rmse_change:+.2f}%, Pearson {pearson_change:+.4f}, "
            f"Spearman {spearman_change:+.4f}, PairAcc {pairwise_change:+.2f} pp"
        )
        ce5_oob = ce5_metrics["out_of_score_band_by_teacher_band"]
        baseline_oob = baseline["out_of_score_band_by_teacher_band"]
        band_changes: list[str] = []
        for band_name in ("low", "mid", "high"):
            ce5_ratio = ce5_oob[band_name]["ratio"]
            baseline_ratio = baseline_oob[band_name]["ratio"]
            if ce5_ratio is None or baseline_ratio is None:
                band_changes.append(f"{band_name} n/a")
            else:
                change = 100.0 * (float(ce5_ratio) - float(baseline_ratio))
                band_changes.append(f"{band_name} {change:+.2f} pp")
        print(f"OOB change vs {label}: " + ", ".join(band_changes))
    print(
        "OOB bands by teacher score: low [0,.25), mid [.25,.75), high [.75,1]."
    )
    print(
        "Miss p50 is the median prediction among missed highs; Gap p50 is its "
        "median distance below 0.75."
    )
    print(f"Full report: {summary_path}\n")


def _print_router_ablation_summary(diagnostics: Mapping[str, Any]) -> None:
    methods = diagnostics.get("methods")
    if not isinstance(methods, Mapping):
        return
    method_labels = {
        "dynamic_router": "Dynamic router",
        "static_validation_mean": "Static validation mean",
        "uniform": "Uniform weights",
        "base_sts_only": "Base STS only",
        "oracle_best_expert": "Oracle best expert*",
    }
    ordered_names = [
        name
        for name in (
            "dynamic_router",
            "static_validation_mean",
            "uniform",
            "base_sts_only",
        )
        if name in methods
    ]
    ordered_names.extend(
        sorted(name for name in methods if name.startswith("expert_latent_"))
    )
    if "oracle_best_expert" in methods:
        ordered_names.append("oracle_best_expert")

    columns = (
        ("Router/expert", 24),
        ("MAE", 8),
        ("RMSE", 8),
        ("Pearson", 9),
        ("Spearman", 9),
        ("PairAcc", 9),
        ("Low OOB", 9),
        ("Mid OOB", 9),
        ("High OOB", 9),
    )
    print("\n=== Router counterfactuals ===")
    print("  ".join(label.ljust(width) for label, width in columns))
    print("  ".join("-" * width for _, width in columns))
    for method_name in ordered_names:
        result = methods[method_name]
        metrics = result["metrics"]
        oob = metrics["out_of_score_band_by_teacher_band"]

        def format_oob(name: str, width: int) -> str:
            ratio = oob[name]["ratio"]
            return (
                "n/a" if ratio is None else f"{100.0 * float(ratio):.2f}%"
            ).rjust(width)

        label = method_labels.get(
            method_name,
            method_name.removeprefix("expert_") + " only",
        )
        values = (
            label.ljust(columns[0][1]),
            f"{float(metrics['mae']):.4f}".rjust(columns[1][1]),
            f"{float(metrics['rmse']):.4f}".rjust(columns[2][1]),
            f"{float(metrics['pearson']):.4f}".rjust(columns[3][1]),
            f"{float(metrics['spearman']):.4f}".rjust(columns[4][1]),
            f"{100.0 * float(metrics['pairwise_accuracy']):.2f}%".rjust(
                columns[5][1]
            ),
            format_oob("low", columns[6][1]),
            format_oob("mid", columns[7][1]),
            format_oob("high", columns[8][1]),
        )
        print("  ".join(values))

    static_weights = diagnostics.get("static_gate_weights", {})
    if isinstance(static_weights, Mapping):
        rendered_weights = ", ".join(
            f"{name}={float(weight):.3f}"
            for name, weight in static_weights.items()
        )
        print(f"Static weights from validation: {rendered_weights}")
    if "base_sts_only" in methods:
        print(
            "Base STS only is the checkpoint's trained base expert, not the "
            "untouched external STS baseline."
        )
    print(
        "* Oracle chooses the closest individual expert using each test teacher "
        "label. It is an upper bound, not a deployable method."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate CE5 and diagnose latent expert behavior."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument(
        "--evaluation-split",
        choices=("test", "validation", "train", "all"),
        default="test",
        help="Dataset partition to evaluate (default: held-out test).",
    )
    parser.add_argument(
        "--pair-type",
        choices=("all", "grant_faculty", "grant_grant", "faculty_faculty"),
        default="all",
        help="Evaluate only one pair type after verifying the complete split.",
    )
    parser.add_argument("--validation-ratio", type=_unit_float)
    parser.add_argument("--test-ratio", type=_unit_float)
    parser.add_argument(
        "--split-group",
        choices=(
            "owner",
            "target_owner_id",
            "target_item_id",
            "grant_id",
            "grant_item_id",
            "faculty_id",
            "faculty_item_id",
            "pair_id",
        ),
    )
    parser.add_argument(
        "--skip-split-verification",
        action="store_true",
        help="Allow evaluation without matching the training split manifest (legacy only).",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--min-confidence", type=_unit_float)
    parser.add_argument(
        "--duplicate-policy",
        choices=("highest-confidence", "latest", "error"),
    )
    parser.add_argument("--max-examples", type=_nonnegative_int)
    parser.add_argument("--limit-evaluation", type=_nonnegative_int, default=0)
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--max-length", type=_positive_int)
    parser.add_argument("--ranking-min-score-gap", type=_unit_float)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--router-ablations",
        action="store_true",
        help=(
            "On the held-out test split, compare the dynamic router with "
            "validation-derived static weights, uniform weights, individual "
            "experts, and a teacher-aware oracle."
        ),
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--baseline-model-id", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument(
        "--single-head-model",
        default="",
        help="Optional fine-tuned Hugging Face sequence-classifier path.",
    )
    parser.add_argument("--attention-examples-per-head", type=_nonnegative_int, default=3)
    parser.add_argument("--largest-error-examples", type=_nonnegative_int, default=10)
    parser.add_argument("--top-attention-tokens", type=_positive_int, default=8)
    parser.add_argument("--audit-sample-size", type=_nonnegative_int, default=300)
    return parser


def _setting(
    cli_value: Any,
    training_arguments: Mapping[str, Any],
    name: str,
    fallback: Any,
) -> Any:
    if cli_value is not None:
        return cli_value
    value = training_arguments.get(name)
    return fallback if value is None else value


def main() -> int:
    args = build_parser().parse_args()
    if args.router_ablations and args.evaluation_split != "test":
        raise ValueError(
            "--router-ablations requires --evaluation-split test so static "
            "weights can be estimated on validation and applied to held-out data"
        )
    started = time.time()
    checkpoint_path = _resolve_path(args.checkpoint)
    judgment_path = _resolve_path(args.judgments)
    output_dir = _resolve_path(args.output_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CE5 checkpoint not found: {checkpoint_path}")
    if not judgment_path.exists():
        raise FileNotFoundError(f"Teacher judgments not found: {judgment_path}")

    training_config_path = (
        _resolve_path(args.training_config)
        if args.training_config
        else checkpoint_path.parent / "run_config.json"
    )
    training_config = _load_json(training_config_path)
    training_arguments = training_config.get("arguments")
    if not isinstance(training_arguments, Mapping):
        training_arguments = {}
    validation_ratio = float(
        _setting(args.validation_ratio, training_arguments, "validation_ratio", 0.1)
    )
    test_ratio = float(
        _setting(args.test_ratio, training_arguments, "test_ratio", 0.1)
    )
    split_group = str(_setting(args.split_group, training_arguments, "split_group", "owner"))
    seed = int(_setting(args.seed, training_arguments, "seed", 42))
    min_confidence = float(
        _setting(args.min_confidence, training_arguments, "min_confidence", 0.0)
    )
    duplicate_policy = str(
        _setting(
            args.duplicate_policy,
            training_arguments,
            "duplicate_policy",
            "highest-confidence",
        )
    )
    max_examples = int(
        _setting(args.max_examples, training_arguments, "max_examples", 0)
    )
    max_length = int(_setting(args.max_length, training_arguments, "max_length", 128))
    ranking_min_score_gap = float(
        _setting(
            args.ranking_min_score_gap,
            training_arguments,
            "ranking_min_score_gap",
            0.1,
        )
    )

    from ce5.training.train_independent_latent_heads import (
        _mix_training_examples,
        load_judgments,
        split_examples_three_way,
    )
    from ce5.modeling.registry import load_model_from_checkpoint
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    examples, loading_stats = load_judgments(
        judgment_path,
        min_confidence=min_confidence,
        duplicate_policy=duplicate_policy,
        max_examples=max_examples,
    )
    train_pool, validation_examples, test_examples = split_examples_three_way(
        examples,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        split_group=split_group,
        seed=seed,
    )
    use_all_training_pairs = bool(training_arguments.get("use_all_training_pairs", False))
    training_pair_mix = training_arguments.get(
        "training_pair_mix",
        {"grant_faculty": 0.8, "grant_grant": 0.1, "faculty_faculty": 0.1},
    )
    if use_all_training_pairs:
        train_examples = list(train_pool)
    elif isinstance(training_pair_mix, Mapping):
        train_examples = _mix_training_examples(
            train_pool,
            ratios={str(key): float(value) for key, value in training_pair_mix.items()},
            seed=seed,
        )
    else:
        raise RuntimeError("training_pair_mix in run_config.json must be an object")

    if args.evaluation_split == "test":
        evaluation_examples = test_examples
    elif args.evaluation_split == "validation":
        evaluation_examples = validation_examples
    elif args.evaluation_split == "train":
        evaluation_examples = train_examples
    else:
        evaluation_examples = examples

    split_manifest_path = checkpoint_path.parent / "split_manifest.json"
    split_verification: dict[str, Any] = {
        "manifest": str(split_manifest_path),
        "required": False,
        "verified": False,
    }
    if args.evaluation_split != "all":
        split_verification = _verify_split_manifest(
            evaluation_examples,
            split_name=args.evaluation_split,
            manifest_path=split_manifest_path,
            allow_unverified=args.skip_split_verification,
        )
    verified_split_examples = len(evaluation_examples)
    if args.pair_type != "all":
        evaluation_examples = [
            example
            for example in evaluation_examples
            if example.pair_type == args.pair_type
        ]
    if args.limit_evaluation > 0:
        evaluation_examples = evaluation_examples[: args.limit_evaluation]
    if not evaluation_examples:
        raise RuntimeError("The selected evaluation split is empty")

    device = _resolve_device(args.device)
    precision = _resolve_precision(args.precision, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    model = load_model_from_checkpoint(
        checkpoint_path,
        torch_dtype=precision,
        trust_remote_code=args.trust_remote_code,
    )
    is_directional_matcher = bool(getattr(model, "requires_pair_masks", False))
    evaluated_architecture_type = str(getattr(model, "architecture_type", ""))
    tokenizer_reference = _clean_text(args.tokenizer)
    if not tokenizer_reference:
        saved_tokenizer = checkpoint_path.parent / "tokenizer"
        tokenizer_reference = (
            str(saved_tokenizer)
            if saved_tokenizer.exists()
            else model.architecture_config.backbone_model_id
        )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_reference,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    loader = _make_loader(
        evaluation_examples,
        tokenizer,
        batch_size=args.batch_size,
        max_length=max_length,
        num_workers=args.num_workers,
        device=device,
        include_pair_masks=is_directional_matcher,
    )
    ce5_diagnostics, prediction_records, attention_examples = _evaluate_latent_model(
        model,
        loader,
        tokenizer,
        device,
        precision,
        min_score_gap=ranking_min_score_gap,
        attention_examples_per_head=args.attention_examples_per_head,
        largest_error_examples=args.largest_error_examples,
        top_attention_tokens=args.top_attention_tokens,
        no_progress=args.no_progress,
    )
    if args.router_ablations:
        validation_verification = _verify_split_manifest(
            validation_examples,
            split_name="validation",
            manifest_path=split_manifest_path,
            allow_unverified=args.skip_split_verification,
        )
        static_gate_examples = list(validation_examples)
        if args.pair_type != "all":
            static_gate_examples = [
                example
                for example in static_gate_examples
                if example.pair_type == args.pair_type
            ]
        if not static_gate_examples:
            raise RuntimeError(
                "The validation split has no examples for static-router calibration"
            )
        static_gate_loader = _make_loader(
            static_gate_examples,
            tokenizer,
            batch_size=args.batch_size,
            max_length=max_length,
            num_workers=args.num_workers,
            device=device,
            include_pair_masks=is_directional_matcher,
        )
        static_gate_weights = _mean_gate_weights(
            model,
            static_gate_loader,
            device,
            precision,
            no_progress=args.no_progress,
        )
        router_ablations = _router_ablation_diagnostics(
            prediction_records,
            expert_names=model.expert_names,
            static_gate_weights=static_gate_weights,
            min_score_gap=ranking_min_score_gap,
        )
        router_ablations["static_gate_source"] = {
            "split": "validation",
            "pair_type": args.pair_type,
            "examples": len(static_gate_examples),
            "split_verification": validation_verification,
        }
        ce5_diagnostics["router_ablations"] = router_ablations
    architecture = model.architecture_dict()
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    comparisons: dict[str, Any] = {}
    if not args.skip_baseline:
        baseline_id = _clean_text(args.baseline_model_id)
        baseline_tokenizer = AutoTokenizer.from_pretrained(
            baseline_id,
            trust_remote_code=args.trust_remote_code,
        )
        baseline_loader = _make_loader(
            evaluation_examples,
            baseline_tokenizer,
            batch_size=args.batch_size,
            max_length=max_length,
            num_workers=args.num_workers,
            device=device,
        )
        baseline_model = AutoModelForSequenceClassification.from_pretrained(
            baseline_id,
            torch_dtype=precision,
            trust_remote_code=args.trust_remote_code,
        ).to(device)
        baseline_result, baseline_predictions = _evaluate_sequence_classifier(
            baseline_model,
            baseline_loader,
            device,
            precision,
            min_score_gap=ranking_min_score_gap,
            description="Untouched ModernCE baseline",
            no_progress=args.no_progress,
        )
        comparisons["untouched_modernce"] = {
            "model_id": baseline_id,
            **baseline_result,
            "delta": _comparison_delta(
                ce5_diagnostics["metrics"], baseline_result["metrics"]
            ),
        }
        for record in prediction_records:
            record["untouched_modernce_score"] = baseline_predictions[record["pair_id"]]
            record["ce5_minus_untouched_modernce"] = (
                float(record["ce5_score"])
                - float(record["untouched_modernce_score"])
            )
        del baseline_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if _clean_text(args.single_head_model):
        single_head_reference = _clean_text(args.single_head_model)
        single_tokenizer = AutoTokenizer.from_pretrained(
            single_head_reference,
            trust_remote_code=args.trust_remote_code,
        )
        single_loader = _make_loader(
            evaluation_examples,
            single_tokenizer,
            batch_size=args.batch_size,
            max_length=max_length,
            num_workers=args.num_workers,
            device=device,
        )
        single_model = AutoModelForSequenceClassification.from_pretrained(
            single_head_reference,
            torch_dtype=precision,
            trust_remote_code=args.trust_remote_code,
        ).to(device)
        single_result, single_predictions = _evaluate_sequence_classifier(
            single_model,
            single_loader,
            device,
            precision,
            min_score_gap=ranking_min_score_gap,
            description="Fine-tuned single-head comparison",
            no_progress=args.no_progress,
        )
        comparisons["fine_tuned_single_head"] = {
            "model_id": single_head_reference,
            **single_result,
            "delta": _comparison_delta(
                ce5_diagnostics["metrics"], single_result["metrics"]
            ),
        }
        for record in prediction_records:
            record["fine_tuned_single_head_score"] = single_predictions[
                record["pair_id"]
            ]
        del single_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    teacher_metadata = _load_teacher_metadata(judgment_path)
    for record in prediction_records:
        record.update(teacher_metadata.get(str(record["pair_id"]), {}))
    attention_by_pair = {
        str(record["pair_id"]): teacher_metadata.get(str(record["pair_id"]), {})
        for record in attention_examples
    }
    for record in attention_examples:
        record.update(attention_by_pair[str(record["pair_id"])])

    audit_records = _audit_sample(
        prediction_records,
        size=args.audit_sample_size,
        seed=seed + 101,
    )
    prediction_scope = (
        args.evaluation_split
        if args.pair_type == "all"
        else f"{args.evaluation_split}_{args.pair_type}"
    )
    predictions_path = output_dir / f"{prediction_scope}_predictions.jsonl"
    summary = {
        "schema_version": "ce5.evaluation.v2",
        "created_at_utc": _utc_now(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "judgments": str(judgment_path),
        "judgments_sha256": _file_sha256(judgment_path),
        "training_config": str(training_config_path)
        if training_config_path.exists()
        else None,
        "architecture": architecture,
        "configuration": {
            "evaluation_split": args.evaluation_split,
            "pair_type": args.pair_type,
            "validation_ratio": validation_ratio,
            "test_ratio": test_ratio,
            "split_group": split_group,
            "seed": seed,
            "min_confidence": min_confidence,
            "duplicate_policy": duplicate_policy,
            "max_examples": max_examples,
            "limit_evaluation": args.limit_evaluation,
            "batch_size": args.batch_size,
            "max_length": max_length,
            "ranking_min_score_gap": ranking_min_score_gap,
            "router_ablations": bool(args.router_ablations),
            "device": str(device),
            "precision": str(precision).replace("torch.", ""),
        },
        "data": {
            "loading": loading_stats,
            "all_examples": len(examples),
            "train_pool_examples": len(train_pool),
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
            "test_examples": len(test_examples),
            "verified_split_examples_before_pair_type_filter": verified_split_examples,
            "evaluated_examples": len(evaluation_examples),
            "split_verification": split_verification,
        },
        "ce5": ce5_diagnostics,
        "comparisons": comparisons,
        "outputs": {
            "predictions": str(predictions_path),
            "attention_examples": str(output_dir / "attention_examples.jsonl"),
            "human_audit_sample": str(output_dir / "human_audit_sample.jsonl"),
        },
        "elapsed_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "evaluation_summary.json"
    _write_json(summary_path, summary)
    _write_jsonl(predictions_path, prediction_records)
    _write_jsonl(output_dir / "attention_examples.jsonl", attention_examples)
    _write_jsonl(output_dir / "human_audit_sample.jsonl", audit_records)

    _print_console_summary(
        split_name=prediction_scope,
        examples=len(evaluation_examples),
        split_verified=bool(split_verification["verified"]),
        ce5_result=ce5_diagnostics,
        ce5_method_name=(
            "CE5 dir-private"
            if evaluated_architecture_type == "directional_private_experts"
            else "CE5 independent-v2"
            if evaluated_architecture_type == "independent_pair_aware_heads"
            else "CE5 logit-router"
            if evaluated_architecture_type == "independent_logit_aware_router"
            else "CE5 reliability-router"
            if evaluated_architecture_type
            == "independent_reliability_aware_router"
            else "CE5 directional"
            if is_directional_matcher
            else "CE5 latent-head"
        ),
        comparisons=comparisons,
        summary_path=summary_path,
    )
    router_diagnostics = ce5_diagnostics.get("router_ablations")
    if isinstance(router_diagnostics, Mapping):
        _print_router_ablation_summary(router_diagnostics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
