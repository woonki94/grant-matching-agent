"""Evaluate a CE5 latent-head checkpoint and inspect what its experts learned.

The evaluator reproduces the grouped split used by ``ce5/train.py``.  It
reports ordinary teacher-imitation metrics, but also measures expert variance,
expert correlations, gate behavior, weighted logit contributions, and latent
attention overlap.  The untouched ModernCE checkpoint can be evaluated on the
same examples as a speed and quality baseline.

Generated files are JSON/JSONL so later analysis is not coupled to W&B:

* ``evaluation_summary.json`` -- aggregate metrics and latent diagnostics
* ``validation_predictions.jsonl`` -- one record per evaluated pair
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
    REPO_ROOT / "ce5" / "dataset" / "judgments" / "teacher_judgments.jsonl"
)
DEFAULT_CHECKPOINT = REPO_ROOT / "ce5" / "models" / "latent_head_distilled" / "best.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ce5" / "eval" / "results" / "latent_head_distilled"
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
    def __init__(self, tokenizer: Any, *, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, examples: Sequence[Any]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [example.grant_text for example in examples],
            [example.faculty_text for example in examples],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
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
) -> dict[str, float | int]:
    if not predictions or len(predictions) != len(targets):
        raise ValueError("Predictions and targets must be non-empty and equally sized")
    errors = [prediction - target for prediction, target in zip(predictions, targets, strict=True)]
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
    }


def _score_bin(score: float) -> str:
    for name, lower, upper, include_upper in SCORE_BINS:
        if score >= lower and (score <= upper if include_upper else score < upper):
            return name
    return SCORE_BINS[-1][0]


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
) -> DataLoader[Any]:
    return DataLoader(
        ExampleDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=EvaluationCollator(tokenizer, max_length=max_length),
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
        batch_attention = output.attention_weights.detach().float().cpu()
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
            gate_values = batch_gates[row_index].tolist()
            contribution_values = batch_contributions[row_index].tolist()
            record = _base_prediction_record(
                example,
                prediction=prediction,
                expert_names=expert_names,
                expert_scores=expert_score_values,
                gates=gate_values,
                contributions=contribution_values,
            )
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
    return diagnostics, records, attention_examples


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate CE5 and diagnose latent expert behavior."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--evaluation-split", choices=("validation", "train", "all"), default="validation")
    parser.add_argument("--validation-ratio", type=_unit_float)
    parser.add_argument(
        "--split-group",
        choices=("grant_id", "grant_item_id", "faculty_id", "faculty_item_id", "pair_id"),
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
    split_group = str(_setting(args.split_group, training_arguments, "split_group", "grant_id"))
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

    from ce5.model import ModernCELatentHeadModel
    from ce5.train import load_judgments, split_examples
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    examples, loading_stats = load_judgments(
        judgment_path,
        min_confidence=min_confidence,
        duplicate_policy=duplicate_policy,
        max_examples=max_examples,
    )
    train_examples, validation_examples = split_examples(
        examples,
        validation_ratio=validation_ratio,
        split_group=split_group,
        seed=seed,
    )
    if args.evaluation_split == "validation":
        evaluation_examples = validation_examples
    elif args.evaluation_split == "train":
        evaluation_examples = train_examples
    else:
        evaluation_examples = examples
    if args.limit_evaluation > 0:
        evaluation_examples = evaluation_examples[: args.limit_evaluation]
    if not evaluation_examples:
        raise RuntimeError("The selected evaluation split is empty")

    device = _resolve_device(args.device)
    precision = _resolve_precision(args.precision, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    model = ModernCELatentHeadModel.from_checkpoint(
        checkpoint_path,
        torch_dtype=precision,
        trust_remote_code=args.trust_remote_code,
    )
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
    summary = {
        "schema_version": "ce5.evaluation.v1",
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
            "validation_ratio": validation_ratio,
            "split_group": split_group,
            "seed": seed,
            "min_confidence": min_confidence,
            "duplicate_policy": duplicate_policy,
            "max_examples": max_examples,
            "limit_evaluation": args.limit_evaluation,
            "batch_size": args.batch_size,
            "max_length": max_length,
            "ranking_min_score_gap": ranking_min_score_gap,
            "device": str(device),
            "precision": str(precision).replace("torch.", ""),
        },
        "data": {
            "loading": loading_stats,
            "all_examples": len(examples),
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
            "evaluated_examples": len(evaluation_examples),
        },
        "ce5": ce5_diagnostics,
        "comparisons": comparisons,
        "outputs": {
            "predictions": str(output_dir / "validation_predictions.jsonl"),
            "attention_examples": str(output_dir / "attention_examples.jsonl"),
            "human_audit_sample": str(output_dir / "human_audit_sample.jsonl"),
        },
        "elapsed_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "evaluation_summary.json"
    _write_json(summary_path, summary)
    _write_jsonl(output_dir / "validation_predictions.jsonl", prediction_records)
    _write_jsonl(output_dir / "attention_examples.jsonl", attention_examples)
    _write_jsonl(output_dir / "human_audit_sample.jsonl", audit_records)

    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "ce5_metrics": ce5_diagnostics["metrics"],
                "comparison_metrics": {
                    name: result["metrics"] for name, result in comparisons.items()
                },
                "evaluated_examples": len(evaluation_examples),
                "attention_examples": len(attention_examples),
                "audit_examples": len(audit_records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
