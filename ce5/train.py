"""Train the CE5 multi-latent-head cross encoder from LLM judgments.

The trainer consumes ``ce5.judgment.v2`` JSONL records written by
``ce5/data_preparation/llm_distillation.py``.  It optimizes continuous
capability-coverage scores, optionally weighted by teacher confidence, and adds
an in-batch ranking objective for candidates that share a target capability.

The default train/validation/test split keeps each source owner together and
approximately preserves every pair-type/teacher-score stratum.  The held-out
test set is recorded but never evaluated during training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

JUDGMENT_SCHEMA_VERSION = "ce5.judgment.v2"
LEGACY_JUDGMENT_SCHEMA_VERSION = "ce5.judgment.v1"
DEFAULT_MODEL_ID = "dleemiller/ModernCE-base-sts"
DEFAULT_JUDGMENTS = REPO_ROOT / "ce5" / "dataset" / "judgments" / "teacher_judgments_v2.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ce5" / "models" / "latent_head_distilled_v2"
PAIR_TYPES = ("grant_faculty", "grant_grant", "faculty_faculty")
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class JudgmentExample:
    pair_id: str
    pair_type: str
    direction: str
    target_item_id: str
    target_owner_id: str
    target_text: str
    candidate_item_id: str
    candidate_owner_id: str
    candidate_text: str
    score: float
    confidence: float
    prefilter_band: str
    judgment_id: str

    # Compatibility aliases keep the existing evaluator usable for legacy G-F
    # checkpoints while the internal representation is source-neutral.
    @property
    def grant_item_id(self) -> str:
        return self.target_item_id

    @property
    def grant_id(self) -> str:
        return self.target_owner_id

    @property
    def grant_text(self) -> str:
        return self.target_text

    @property
    def faculty_item_id(self) -> str:
        return self.candidate_item_id

    @property
    def faculty_id(self) -> str:
        return self.candidate_owner_id

    @property
    def faculty_text(self) -> str:
        return self.candidate_text

    def split_value(self, field: str) -> str:
        if field == "owner":
            owner_kind = (
                "faculty" if self.pair_type == "faculty_faculty" else "grant"
            )
            return f"{owner_kind}:{self.target_owner_id or self.target_item_id}"
        value = getattr(self, field, "")
        if value:
            return str(value)
        return self.pair_id


class JudgmentDataset(Dataset[JudgmentExample]):
    def __init__(self, examples: Sequence[JudgmentExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> JudgmentExample:
        return self.examples[index]


class JudgmentCollator:
    def __init__(self, tokenizer: Any, *, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, examples: Sequence[JudgmentExample]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [example.target_text for example in examples],
            [example.candidate_text for example in examples],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "encoded": encoded,
            "labels": torch.tensor(
                [example.score for example in examples], dtype=torch.float32
            ),
            "confidences": torch.tensor(
                [example.confidence for example in examples], dtype=torch.float32
            ),
            "query_ids": [example.target_item_id for example in examples],
            "pair_ids": [example.pair_id for example in examples],
            "pair_types": [example.pair_type for example in examples],
        }


class GroupedBatchSampler(Sampler[list[int]]):
    """Keep examples from the same query adjacent while retaining full batches."""

    def __init__(
        self,
        examples: Sequence[JudgmentExample],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        grouped: dict[str, list[int]] = {}
        for index, example in enumerate(examples):
            grouped.setdefault(example.target_item_id, []).append(index)
        self.groups = list(grouped.values())
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.total_examples = len(examples)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        groups = [list(group) for group in self.groups]
        if self.shuffle:
            rng.shuffle(groups)
            for group in groups:
                rng.shuffle(group)
        self.epoch += 1
        ordered = [index for group in groups for index in group]
        for start in range(0, len(ordered), self.batch_size):
            yield ordered[start : start + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(self.total_examples / self.batch_size)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = REPO_ROOT / expanded
    return expanded.resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0,1]")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _parse_training_pair_mix(value: str) -> dict[str, float]:
    parsed = {pair_type: 0.0 for pair_type in PAIR_TYPES}
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                "pair mix must use pair_type=ratio entries"
            )
        raw_name, raw_ratio = part.split("=", 1)
        name = raw_name.strip().lower().replace("-", "_")
        if name not in parsed:
            raise argparse.ArgumentTypeError(
                f"unsupported pair type {name!r}; use {','.join(PAIR_TYPES)}"
            )
        try:
            ratio = float(raw_ratio)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid ratio for {name}: {raw_ratio!r}"
            ) from exc
        if ratio < 0.0:
            raise argparse.ArgumentTypeError("pair mix ratios cannot be negative")
        parsed[name] = ratio
    total = sum(parsed.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise argparse.ArgumentTypeError(
            f"pair mix ratios must sum to 1.0; received {total:.6f}"
        )
    if parsed["grant_faculty"] <= 0.0:
        raise argparse.ArgumentTypeError(
            "grant_faculty must have a positive training ratio"
        )
    return parsed


def _parse_judgment(row: Mapping[str, Any], *, path: Path, line_number: int) -> JudgmentExample:
    schema_version = row.get("schema_version")
    if schema_version not in {
        JUDGMENT_SCHEMA_VERSION,
        LEGACY_JUDGMENT_SCHEMA_VERSION,
    }:
        raise RuntimeError(
            f"Unsupported schema at {path}:{line_number}: {schema_version!r}"
        )
    if schema_version == LEGACY_JUDGMENT_SCHEMA_VERSION:
        required_text = {
            "pair_id": _clean_text(row.get("pair_id")),
            "target_item_id": _clean_text(row.get("grant_item_id")),
            "target_text": _clean_text(row.get("grant_text")),
            "candidate_item_id": _clean_text(row.get("faculty_item_id")),
            "candidate_text": _clean_text(row.get("faculty_text")),
        }
        pair_type = "grant_faculty"
        direction = "grant_to_faculty"
        target_owner_id = _clean_text(row.get("grant_id"))
        candidate_owner_id = _clean_text(row.get("faculty_id"))
    else:
        required_text = {
            "pair_id": _clean_text(row.get("pair_id")),
            "target_item_id": _clean_text(row.get("target_item_id")),
            "target_text": _clean_text(row.get("target_text")),
            "candidate_item_id": _clean_text(row.get("candidate_item_id")),
            "candidate_text": _clean_text(row.get("candidate_text")),
        }
        pair_type = _clean_text(row.get("pair_type"))
        direction = _clean_text(row.get("direction"))
        target_owner_id = _clean_text(row.get("target_owner_id"))
        candidate_owner_id = _clean_text(row.get("candidate_owner_id"))
    missing = [name for name, value in required_text.items() if not value]
    if missing:
        raise RuntimeError(
            f"Missing required field(s) at {path}:{line_number}: {', '.join(missing)}"
        )
    if pair_type not in PAIR_TYPES:
        raise RuntimeError(
            f"Unsupported pair_type at {path}:{line_number}: "
            f"{pair_type!r}"
        )
    try:
        score = float(row["score"])
        confidence = float(row["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid score or confidence at {path}:{line_number}"
        ) from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError(f"Score outside [0,1] at {path}:{line_number}")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise RuntimeError(f"Confidence outside [0,1] at {path}:{line_number}")
    return JudgmentExample(
        **required_text,
        pair_type=pair_type,
        direction=direction,
        target_owner_id=target_owner_id,
        candidate_owner_id=candidate_owner_id,
        score=score,
        confidence=confidence,
        prefilter_band=_clean_text(row.get("prefilter_band")).lower(),
        judgment_id=_clean_text(row.get("judgment_id")),
    )


def load_judgments(
    path: Path,
    *,
    min_confidence: float,
    duplicate_policy: str,
    max_examples: int,
) -> tuple[list[JudgmentExample], dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Teacher judgments not found: {path}")
    by_pair: dict[str, JudgmentExample] = {}
    rows_read = 0
    below_confidence = 0
    duplicates = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            rows_read += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(raw, Mapping):
                raise RuntimeError(f"Expected an object at {path}:{line_number}")
            example = _parse_judgment(raw, path=path, line_number=line_number)
            if example.confidence < min_confidence:
                below_confidence += 1
                continue
            existing = by_pair.get(example.pair_id)
            if existing is not None:
                duplicates += 1
                if duplicate_policy == "error":
                    raise RuntimeError(
                        f"Duplicate pair_id {example.pair_id!r} at {path}:{line_number}"
                    )
                if duplicate_policy == "highest-confidence" and existing.confidence >= example.confidence:
                    continue
            by_pair[example.pair_id] = example
    examples = list(by_pair.values())
    if max_examples > 0:
        examples = examples[:max_examples]
    if not examples:
        raise RuntimeError("No usable teacher judgments remain after filtering")
    stats = {
        "rows_read": rows_read,
        "below_min_confidence": below_confidence,
        "duplicate_pair_ids": duplicates,
        "examples_loaded": len(examples),
    }
    return examples, stats


def _score_bin(score: float) -> str:
    if score < 0.25:
        return "0.00-0.24"
    if score < 0.50:
        return "0.25-0.49"
    if score < 0.75:
        return "0.50-0.74"
    return "0.75-1.00"


def _stable_tiebreak(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\x1f{value}".encode("utf-8")).digest()


def split_examples_three_way(
    examples: Sequence[JudgmentExample],
    *,
    validation_ratio: float,
    test_ratio: float,
    split_group: str,
    seed: int,
) -> tuple[list[JudgmentExample], list[JudgmentExample], list[JudgmentExample]]:
    if validation_ratio + test_ratio >= 1.0:
        raise ValueError("validation_ratio + test_ratio must be smaller than 1")
    if validation_ratio <= 0.0 and test_ratio <= 0.0:
        return list(examples), [], []
    grouped: dict[str, list[JudgmentExample]] = {}
    for example in examples:
        grouped.setdefault(example.split_value(split_group), []).append(example)
    required_splits = 1 + int(validation_ratio > 0.0) + int(test_ratio > 0.0)
    if len(grouped) < required_splits:
        raise RuntimeError(
            f"Grouped splitting requires at least {required_splits} distinct "
            f"{split_group} values"
        )

    ratios = {"validation": validation_ratio, "test": test_ratio}
    held_out_splits = [name for name in ("validation", "test") if ratios[name] > 0.0]
    feature_totals: Counter[tuple[str, str]] = Counter(
        (example.pair_type, _score_bin(example.score)) for example in examples
    )
    target_rows = {name: len(examples) * ratios[name] for name in held_out_splits}
    target_features = {
        name: {
            feature: count * ratios[name]
            for feature, count in feature_totals.items()
        }
        for name in held_out_splits
    }
    current_rows = {name: 0 for name in held_out_splits}
    current_features = {
        name: Counter() for name in held_out_splits
    }
    assignments: dict[str, str] = {}

    group_features = {
        key: Counter(
            (example.pair_type, _score_bin(example.score)) for example in values
        )
        for key, values in grouped.items()
    }
    ordered_keys = sorted(
        grouped,
        key=lambda key: (
            -len(grouped[key]),
            _stable_tiebreak(seed, key),
        ),
    )

    for key in ordered_keys:
        row_count = len(grouped[key])
        features = group_features[key]
        best_split = "train"
        best_gain = 0.0
        best_tiebreak = b""
        for split_name in held_out_splits:
            row_target = max(1.0, target_rows[split_name])
            before_penalty = (
                (current_rows[split_name] - row_target) / row_target
            ) ** 2
            after_penalty = (
                (current_rows[split_name] + row_count - row_target) / row_target
            ) ** 2
            for feature, feature_count in features.items():
                feature_target = max(1.0, target_features[split_name][feature])
                before_penalty += (
                    (current_features[split_name][feature] - feature_target)
                    / feature_target
                ) ** 2
                after_penalty += (
                    (
                        current_features[split_name][feature]
                        + feature_count
                        - feature_target
                    ) / feature_target
                ) ** 2
            gain = before_penalty - after_penalty
            tiebreak = _stable_tiebreak(seed, f"{key}\x1f{split_name}")
            if gain > best_gain or (
                math.isclose(gain, best_gain) and gain > 0.0 and tiebreak < best_tiebreak
            ):
                best_gain = gain
                best_split = split_name
                best_tiebreak = tiebreak
        assignments[key] = best_split
        if best_split != "train":
            current_rows[best_split] += row_count
            current_features[best_split].update(features)

    split_values: dict[str, list[JudgmentExample]] = {
        name: [] for name in SPLIT_NAMES
    }
    for key, values in grouped.items():
        split_values[assignments[key]].extend(values)
    for split_name in ("train", *held_out_splits):
        if not split_values[split_name]:
            raise RuntimeError(f"The grouped split produced an empty {split_name} set")
    return (
        split_values["train"],
        split_values["validation"],
        split_values["test"],
    )


def split_examples(
    examples: Sequence[JudgmentExample],
    *,
    validation_ratio: float,
    split_group: str,
    seed: int,
) -> tuple[list[JudgmentExample], list[JudgmentExample]]:
    """Reproduce the original V1 two-way split used by legacy evaluations."""
    if validation_ratio <= 0.0:
        return list(examples), []
    grouped: dict[str, list[JudgmentExample]] = {}
    for example in examples:
        grouped.setdefault(example.split_value(split_group), []).append(example)
    if len(grouped) < 2:
        raise RuntimeError(
            f"Grouped validation requires at least two distinct {split_group} values"
        )
    group_keys = sorted(grouped)
    random.Random(seed).shuffle(group_keys)
    validation_target = max(1, round(len(examples) * validation_ratio))
    validation_keys: set[str] = set()
    validation_count = 0
    for key in group_keys[:-1]:
        if validation_count >= validation_target:
            break
        validation_keys.add(key)
        validation_count += len(grouped[key])
    train = [
        example
        for example in examples
        if example.split_value(split_group) not in validation_keys
    ]
    validation = [
        example
        for example in examples
        if example.split_value(split_group) in validation_keys
    ]
    if not train or not validation:
        raise RuntimeError("The grouped split produced an empty train or validation set")
    return train, validation


def _mix_training_examples(
    examples: Sequence[JudgmentExample],
    *,
    ratios: Mapping[str, float],
    seed: int,
) -> list[JudgmentExample]:
    """Keep every G-F row and sample auxiliary pair types to requested ratios."""
    by_type: dict[str, list[JudgmentExample]] = defaultdict(list)
    for example in examples:
        by_type[example.pair_type].append(example)
    grant_faculty_ratio = ratios.get("grant_faculty", 0.0)
    if grant_faculty_ratio <= 0.0:
        raise ValueError("The training mix must assign a positive grant_faculty ratio")
    if not by_type["grant_faculty"]:
        raise RuntimeError("The training split contains no grant_faculty examples")

    target_total = len(by_type["grant_faculty"]) / grant_faculty_ratio
    selected = list(by_type["grant_faculty"])
    for pair_type in ("grant_grant", "faculty_faculty"):
        requested = round(target_total * ratios.get(pair_type, 0.0))
        available = sorted(
            by_type[pair_type],
            key=lambda example: _stable_tiebreak(seed, example.pair_id),
        )
        selected.extend(available[:requested])
    random.Random(seed).shuffle(selected)
    return selected


def _dataset_summary(examples: Sequence[JudgmentExample]) -> dict[str, Any]:
    if not examples:
        return {
            "examples": 0,
            "pair_types": {},
            "score_bins": {},
            "bands": {},
        }
    scores = [example.score for example in examples]
    return {
        "examples": len(examples),
        "target_owners": len(
            {example.target_owner_id for example in examples if example.target_owner_id}
        ),
        "target_items": len({example.target_item_id for example in examples}),
        "candidate_owners": len(
            {
                example.candidate_owner_id
                for example in examples
                if example.candidate_owner_id
            }
        ),
        "pair_types": dict(sorted(Counter(example.pair_type for example in examples).items())),
        "score_bins": dict(sorted(Counter(_score_bin(example.score) for example in examples).items())),
        "score_min": min(scores),
        "score_mean": sum(scores) / len(scores),
        "score_max": max(scores),
        "bands": dict(
            sorted(
                Counter(example.prefilter_band or "unknown" for example in examples).items()
            )
        ),
    }


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
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


def _autocast_context(device: torch.device, dtype: torch.dtype) -> Any:
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _move_encoded(encoded: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in encoded.items()
    }


def _confidence_weights(confidences: Tensor, *, floor: float, power: float) -> Tensor:
    if power == 0.0:
        return torch.ones_like(confidences)
    return confidences.clamp(min=floor, max=1.0).pow(power)


def _pointwise_loss(
    logits: Tensor,
    scores: Tensor,
    targets: Tensor,
    weights: Tensor,
    *,
    loss_name: str,
) -> Tensor:
    if loss_name == "smooth_l1":
        losses = F.smooth_l1_loss(scores, targets, reduction="none")
    elif loss_name == "mse":
        losses = F.mse_loss(scores, targets, reduction="none")
    elif loss_name == "bce":
        losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    else:  # pragma: no cover - argparse and model config validate this
        raise ValueError(f"Unsupported score loss: {loss_name}")
    return torch.sum(losses * weights) / weights.sum().clamp_min(1e-8)


def _attention_diversity_loss(attention_weights: Tensor) -> Tensor:
    """Penalize latent heads that attend to the same token pattern."""

    num_heads = attention_weights.shape[1]
    if num_heads < 2:
        return attention_weights.new_zeros(())
    normalized = F.normalize(attention_weights, p=2, dim=-1, eps=1e-8)
    similarities = torch.bmm(normalized, normalized.transpose(1, 2))
    off_diagonal = ~torch.eye(
        num_heads,
        dtype=torch.bool,
        device=attention_weights.device,
    ).unsqueeze(0)
    return similarities.masked_select(off_diagonal).square().mean()


def _gate_balance_loss(gate_weights: Tensor) -> Tensor:
    """Return the KL divergence between average expert use and uniform use."""

    average_usage = gate_weights.mean(dim=0).clamp_min(1e-8)
    num_experts = gate_weights.shape[-1]
    return torch.sum(average_usage * torch.log(average_usage * num_experts))


def _ranking_loss(
    logits: Tensor,
    targets: Tensor,
    weights: Tensor,
    query_ids: Sequence[str],
    *,
    min_score_gap: float,
    margin: float,
    max_pairs: int,
) -> tuple[Tensor, int]:
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    gaps: list[float] = []
    grouped: dict[str, list[int]] = {}
    for index, query_id in enumerate(query_ids):
        grouped.setdefault(query_id, []).append(index)
    detached_targets = targets.detach().float().cpu().tolist()
    for indices in grouped.values():
        for left_offset, left in enumerate(indices):
            for right in indices[left_offset + 1 :]:
                difference = detached_targets[left] - detached_targets[right]
                if abs(difference) < min_score_gap:
                    continue
                positive, negative = (left, right) if difference > 0.0 else (right, left)
                positive_indices.append(positive)
                negative_indices.append(negative)
                gaps.append(abs(difference))
    if not positive_indices:
        return logits.sum() * 0.0, 0
    if max_pairs > 0 and len(positive_indices) > max_pairs:
        keep = sorted(range(len(gaps)), key=gaps.__getitem__, reverse=True)[:max_pairs]
        positive_indices = [positive_indices[index] for index in keep]
        negative_indices = [negative_indices[index] for index in keep]
    positive_index = torch.tensor(positive_indices, device=logits.device)
    negative_index = torch.tensor(negative_indices, device=logits.device)
    pair_weights = torch.sqrt(weights[positive_index] * weights[negative_index])
    pair_losses = F.softplus(
        -(logits[positive_index] - logits[negative_index] - margin)
    )
    loss = torch.sum(pair_losses * pair_weights) / pair_weights.sum().clamp_min(1e-8)
    return loss, len(positive_indices)


def _batch_losses(
    model: ModernCELatentHeadModel,
    batch: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Tensor], Any, int]:
    encoded = _move_encoded(batch["encoded"], device)
    targets = batch["labels"].to(device)
    confidences = batch["confidences"].to(device)
    weights = _confidence_weights(
        confidences,
        floor=args.confidence_weight_floor,
        power=args.confidence_weight_power,
    )
    output = model(**encoded)
    pointwise = _pointwise_loss(
        output.logits,
        output.scores,
        targets,
        weights,
        loss_name=args.score_loss,
    )
    ranking, rank_pairs = _ranking_loss(
        output.logits,
        targets,
        weights,
        batch["query_ids"],
        min_score_gap=args.ranking_min_score_gap,
        margin=args.ranking_margin,
        max_pairs=args.ranking_max_pairs_per_batch,
    )
    diversity = _attention_diversity_loss(output.attention_weights)
    gate_balance = _gate_balance_loss(output.gate_weights)
    total = (
        pointwise
        + args.ranking_loss_weight * ranking
        + args.diversity_loss_weight * diversity
        + args.gate_balance_loss_weight * gate_balance
    )
    return {
        "total": total,
        "pointwise": pointwise,
        "ranking": ranking,
        "diversity": diversity,
        "gate_balance": gate_balance,
    }, output, rank_pairs


def _parameter_groups(
    model: ModernCELatentHeadModel,
    *,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    groups: dict[tuple[bool, bool], list[Tensor]] = {
        (True, True): [],
        (True, False): [],
        (False, True): [],
        (False, False): [],
    }
    for name, parameter in model.named_parameters():
        is_encoder = name.startswith("encoder.")
        use_decay = parameter.ndim > 1 and not name.endswith("bias")
        groups[(is_encoder, use_decay)].append(parameter)
    output: list[dict[str, Any]] = []
    for (is_encoder, use_decay), parameters in groups.items():
        if parameters:
            output.append(
                {
                    "params": parameters,
                    "lr": encoder_lr if is_encoder else head_lr,
                    "weight_decay": weight_decay if use_decay else 0.0,
                    "group_name": (
                        f"{'encoder' if is_encoder else 'heads'}_"
                        f"{'decay' if use_decay else 'no_decay'}"
                    ),
                }
            )
    return output


def _set_encoder_trainable(model: ModernCELatentHeadModel, trainable: bool) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = trainable


def _parameter_counts(model: ModernCELatentHeadModel) -> dict[str, int]:
    parameters = tuple(model.parameters())
    return {
        "total": sum(parameter.numel() for parameter in parameters),
        "trainable": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
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
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return numerator / (left_scale * right_scale)


def _pairwise_accuracy(
    predictions: Sequence[float],
    targets: Sequence[float],
    query_ids: Sequence[str],
    *,
    min_score_gap: float,
) -> tuple[float, int]:
    correct = 0.0
    count = 0
    grouped: dict[str, list[int]] = {}
    for index, query_id in enumerate(query_ids):
        grouped.setdefault(query_id, []).append(index)
    for indices in grouped.values():
        for left_offset, left in enumerate(indices):
            for right in indices[left_offset + 1 :]:
                teacher_difference = targets[left] - targets[right]
                if abs(teacher_difference) < min_score_gap:
                    continue
                student_difference = predictions[left] - predictions[right]
                if student_difference == 0.0:
                    correct += 0.5
                elif student_difference * teacher_difference > 0.0:
                    correct += 1.0
                count += 1
    return (correct / count if count else 0.0), count


def _prediction_metrics(
    predictions: Sequence[float],
    targets: Sequence[float],
    query_ids: Sequence[str],
    *,
    min_score_gap: float,
) -> dict[str, float | int]:
    if not predictions:
        return {}
    errors = [
        prediction - target
        for prediction, target in zip(predictions, targets, strict=True)
    ]
    mse = sum(error * error for error in errors) / len(errors)
    pair_accuracy, pair_count = _pairwise_accuracy(
        predictions,
        targets,
        query_ids,
        min_score_gap=min_score_gap,
    )
    return {
        "examples": len(predictions),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "pearson": _pearson(predictions, targets),
        "spearman": _pearson(_average_ranks(predictions), _average_ranks(targets)),
        "pairwise_accuracy": pair_accuracy,
        "pairwise_comparisons": pair_count,
        "prediction_mean": sum(predictions) / len(predictions),
        "target_mean": sum(targets) / len(targets),
    }


@torch.no_grad()
def evaluate(
    model: ModernCELatentHeadModel,
    loader: DataLoader[Any],
    device: torch.device,
    precision: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    query_ids: list[str] = []
    pair_types: list[str] = []
    loss_sums = {name: 0.0 for name in ("total", "pointwise", "ranking", "diversity", "gate_balance")}
    gate_sum: Optional[Tensor] = None
    head_score_sum: Optional[Tensor] = None
    example_count = 0
    ranking_pairs = 0
    for batch in loader:
        with _autocast_context(device, precision):
            losses, output, batch_rank_pairs = _batch_losses(model, batch, device, args)
        batch_size = int(batch["labels"].numel())
        for name, value in losses.items():
            loss_sums[name] += float(value.detach().float().item()) * batch_size
        predictions.extend(output.scores.detach().float().cpu().tolist())
        targets.extend(batch["labels"].tolist())
        query_ids.extend(batch["query_ids"])
        pair_types.extend(batch["pair_types"])
        current_gate = output.gate_weights.detach().float().sum(dim=0).cpu()
        current_heads = output.head_scores.detach().float().sum(dim=0).cpu()
        gate_sum = current_gate if gate_sum is None else gate_sum + current_gate
        head_score_sum = current_heads if head_score_sum is None else head_score_sum + current_heads
        example_count += batch_size
        ranking_pairs += batch_rank_pairs
    if example_count == 0:
        return {}
    overall_metrics = _prediction_metrics(
        predictions,
        targets,
        query_ids,
        min_score_gap=args.ranking_min_score_gap,
    )
    by_pair_type: dict[str, dict[str, float | int]] = {}
    for pair_type in PAIR_TYPES:
        indices = [
            index for index, value in enumerate(pair_types) if value == pair_type
        ]
        if not indices:
            continue
        by_pair_type[pair_type] = _prediction_metrics(
            [predictions[index] for index in indices],
            [targets[index] for index in indices],
            [query_ids[index] for index in indices],
            min_score_gap=args.ranking_min_score_gap,
        )
    expert_names = model.expert_names
    return {
        "loss": {name: total / example_count for name, total in loss_sums.items()},
        **overall_metrics,
        "by_pair_type": by_pair_type,
        "ranking_pairs_in_batches": ranking_pairs,
        "gate_usage": {
            name: float(gate_sum[index].item() / example_count)
            for index, name in enumerate(expert_names)
        },
        "expert_score_mean": {
            name: float(head_score_sum[index].item() / example_count)
            for index, name in enumerate(expert_names)
        },
    }


def _train_epoch(
    model: ModernCELatentHeadModel,
    loader: DataLoader[Any],
    optimizer: AdamW,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
    precision: torch.dtype,
    args: argparse.Namespace,
    *,
    epoch: int,
    global_step: int,
    wandb_run: Any = None,
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {name: 0.0 for name in ("total", "pointwise", "ranking", "diversity", "gate_balance")}
    example_count = 0
    ranking_pairs = 0
    step_totals = {name: 0.0 for name in totals}
    step_example_count = 0
    step_ranking_pairs = 0
    progress: Iterable[Any] = loader
    if tqdm is not None and not args.no_progress:
        progress = tqdm(loader, desc=f"train epoch {epoch + 1}", dynamic_ncols=True)
    for batch_index, batch in enumerate(progress):
        with _autocast_context(device, precision):
            losses, _, batch_rank_pairs = _batch_losses(model, batch, device, args)
            backward_loss = losses["total"] / args.gradient_accumulation_steps
        scaler.scale(backward_loss).backward()
        should_step = (
            (batch_index + 1) % args.gradient_accumulation_steps == 0
            or batch_index + 1 == len(loader)
        )
        batch_size = int(batch["labels"].numel())
        for name, value in losses.items():
            detached_loss = float(value.detach().float().item())
            totals[name] += detached_loss * batch_size
            step_totals[name] += detached_loss * batch_size
        example_count += batch_size
        step_example_count += batch_size
        ranking_pairs += batch_rank_pairs
        step_ranking_pairs += batch_rank_pairs
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if (
                wandb_run is not None
                and global_step % args.wandb_log_every_steps == 0
            ):
                log_payload = {
                    f"train/loss_{name}": total / max(1, step_example_count)
                    for name, total in step_totals.items()
                }
                log_payload.update(
                    {
                        "train/epoch": epoch + 1,
                        "train/ranking_pairs": step_ranking_pairs,
                        "train/encoder_frozen": float(
                            epoch < args.frozen_encoder_epochs
                        ),
                    }
                )
                for group in optimizer.param_groups:
                    group_name = str(group.get("group_name", "parameters"))
                    log_payload[f"train/learning_rate/{group_name}"] = float(
                        group["lr"]
                    )
                log_payload["trainer/global_step"] = global_step
                wandb_run.log(log_payload)
            step_totals = {name: 0.0 for name in totals}
            step_example_count = 0
            step_ranking_pairs = 0
        if tqdm is not None and hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{totals['total'] / example_count:.4f}")
    metrics = {name: total / max(1, example_count) for name, total in totals.items()}
    metrics["ranking_pairs"] = float(ranking_pairs)
    return metrics, global_step


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _pair_id_sha256(examples: Sequence[JudgmentExample]) -> str:
    digest = hashlib.sha256()
    for pair_id in sorted(example.pair_id for example in examples):
        digest.update(pair_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _split_manifest_entry(
    examples: Sequence[JudgmentExample],
    *,
    split_group: str,
) -> dict[str, Any]:
    return {
        "examples": len(examples),
        "pair_id_sha256": _pair_id_sha256(examples),
        "groups": sorted({example.split_value(split_group) for example in examples}),
        "summary": _dataset_summary(examples),
    }


def _flatten_numeric_metrics(prefix: str, value: Any) -> dict[str, float]:
    if isinstance(value, Mapping):
        flattened: dict[str, float] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            flattened.update(_flatten_numeric_metrics(child_prefix, child))
        return flattened
    if isinstance(value, bool):
        return {prefix: float(value)}
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return {prefix: float(value)}
    return {}


def _init_wandb(
    args: argparse.Namespace,
    *,
    run_config: Mapping[str, Any],
    output_dir: Path,
) -> Any:
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging was requested but wandb is not installed"
        ) from exc
    tags = [
        tag.strip()
        for tag in str(args.wandb_tags).split(",")
        if tag.strip()
    ]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        tags=tags or None,
        mode=args.wandb_mode,
        dir=str(output_dir),
        config=dict(run_config),
    )
    run.define_metric("trainer/global_step")
    for metric_pattern in ("train/*", "train_epoch/*", "validation/*"):
        run.define_metric(metric_pattern, step_metric="trainer/global_step")
    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CE5 from continuous LLM judgments.")
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--max-examples", type=_nonnegative_int, default=0)
    parser.add_argument("--min-confidence", type=_unit_interval, default=0.0)
    parser.add_argument(
        "--duplicate-policy",
        choices=("highest-confidence", "latest", "error"),
        default="highest-confidence",
    )
    parser.add_argument("--validation-ratio", type=_unit_interval, default=0.1)
    parser.add_argument("--test-ratio", type=_unit_interval, default=0.1)
    parser.add_argument(
        "--split-group",
        choices=("owner", "target_owner_id", "target_item_id", "pair_id"),
        default="owner",
        help=(
            "Grouping unit kept wholly in one split. owner keeps G-F/G-G by "
            "grant owner and F-F by faculty owner (default: owner)."
        ),
    )
    parser.add_argument(
        "--training-pair-mix",
        type=_parse_training_pair_mix,
        default={
            "grant_faculty": 0.8,
            "grant_grant": 0.1,
            "faculty_faculty": 0.1,
        },
        help=(
            "Training-only ratios, e.g. grant_faculty=0.8,grant_grant=0.1,"
            "faculty_faculty=0.1. Every training G-F row is retained and "
            "same-side rows are deterministically sampled."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-latent-heads", type=_positive_int, default=6)
    parser.add_argument("--attention-dim", type=_positive_int, default=128)
    parser.add_argument("--head-dim", type=_positive_int, default=192)
    parser.add_argument("--dropout", type=_unit_interval, default=0.1)
    parser.add_argument("--head-dropout", type=_unit_interval, default=0.0)
    parser.add_argument("--no-base-sts-expert", action="store_true")
    parser.add_argument("--base-sts-gate-bias", type=float, default=4.0)
    parser.add_argument("--score-loss", choices=("smooth_l1", "mse", "bce"), default="smooth_l1")
    parser.add_argument("--diversity-loss-weight", type=_nonnegative_float, default=0.01)
    parser.add_argument("--gate-balance-loss-weight", type=_nonnegative_float, default=0.0)

    parser.add_argument("--epochs", type=_positive_int, default=5)
    parser.add_argument("--frozen-encoder-epochs", type=_nonnegative_int, default=1)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--max-length", type=_positive_int, default=512)
    parser.add_argument("--encoder-lr", type=_positive_float, default=1e-5)
    parser.add_argument("--head-lr", type=_positive_float, default=1e-4)
    parser.add_argument("--weight-decay", type=_nonnegative_float, default=0.01)
    parser.add_argument("--warmup-ratio", type=_unit_interval, default=0.1)
    parser.add_argument("--scheduler", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--gradient-accumulation-steps", type=_positive_int, default=1)
    parser.add_argument("--max-grad-norm", type=_positive_float, default=1.0)
    parser.add_argument("--confidence-weight-floor", type=_unit_interval, default=0.25)
    parser.add_argument("--confidence-weight-power", type=_nonnegative_float, default=1.0)
    parser.add_argument("--ranking-loss-weight", type=_nonnegative_float, default=0.2)
    parser.add_argument("--ranking-min-score-gap", type=_unit_interval, default=0.1)
    parser.add_argument("--ranking-margin", type=_nonnegative_float, default=0.0)
    parser.add_argument("--ranking-max-pairs-per-batch", type=_nonnegative_int, default=512)

    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a device such as cuda:0")
    parser.add_argument(
        "--precision",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="ce5-distillation")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument(
        "--wandb-tags",
        default="ce5,distillation,latent-heads",
        help="Comma-separated W&B tags.",
    )
    parser.add_argument("--wandb-log-every-steps", type=_positive_int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize judgments without loading the tokenizer or model.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    _set_seed(args.seed)
    judgment_path = _resolve_path(args.judgments)
    output_dir = _resolve_path(args.output_dir)
    examples, loading_stats = load_judgments(
        judgment_path,
        min_confidence=args.min_confidence,
        duplicate_policy=args.duplicate_policy,
        max_examples=args.max_examples,
    )
    train_pool, validation_examples, test_examples = split_examples_three_way(
        examples,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        split_group=args.split_group,
        seed=args.seed,
    )
    train_examples = _mix_training_examples(
        train_pool,
        ratios=args.training_pair_mix,
        seed=args.seed,
    )
    data_summary = {
        "loading": loading_stats,
        "all": _dataset_summary(examples),
        "train_before_pair_mix": _dataset_summary(train_pool),
        "train": _dataset_summary(train_examples),
        "validation": _dataset_summary(validation_examples) if validation_examples else None,
        "test": _dataset_summary(test_examples) if test_examples else None,
        "split_group": args.split_group,
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
        "training_pair_mix_requested": args.training_pair_mix,
    }
    if args.dry_run:
        print(json.dumps(data_summary, indent=2, ensure_ascii=False))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "split_manifest.json",
        {
            "schema_version": "ce5.split-manifest.v1",
            "created_at_utc": _utc_now(),
            "judgments": str(judgment_path),
            "seed": args.seed,
            "split_group": args.split_group,
            "validation_ratio": args.validation_ratio,
            "test_ratio": args.test_ratio,
            "training_pair_mix_requested": args.training_pair_mix,
            "train_before_pair_mix": _split_manifest_entry(
                train_pool,
                split_group=args.split_group,
            ),
            "train": _split_manifest_entry(
                train_examples,
                split_group=args.split_group,
            ),
            "validation": _split_manifest_entry(
                validation_examples,
                split_group=args.split_group,
            ),
            "test": _split_manifest_entry(
                test_examples,
                split_group=args.split_group,
            ),
        },
    )

    from transformers import (
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )

    from ce5.model import ModernCELatentHeadModel

    device = _resolve_device(args.device)
    precision = _resolve_precision(args.precision, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    resume_path = _resolve_path(args.resume_checkpoint) if args.resume_checkpoint else None
    if resume_path is not None:
        model = ModernCELatentHeadModel.from_checkpoint(
            resume_path,
            torch_dtype=precision,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer_model_id = model.architecture_config.backbone_model_id
    else:
        model = ModernCELatentHeadModel.from_pretrained(
            args.model_id,
            num_latent_heads=args.num_latent_heads,
            attention_dim=args.attention_dim,
            head_dim=args.head_dim,
            dropout=args.dropout,
            head_dropout=args.head_dropout,
            use_base_sts_expert=not args.no_base_sts_expert,
            base_sts_gate_bias=args.base_sts_gate_bias,
            torch_dtype=precision,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer_model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_id,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    if args.gradient_checkpointing:
        enable_checkpointing = getattr(model.encoder, "gradient_checkpointing_enable", None)
        if not callable(enable_checkpointing):
            raise RuntimeError("The selected encoder does not support gradient checkpointing")
        enable_checkpointing()

    collator = JudgmentCollator(tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        JudgmentDataset(train_examples),
        batch_sampler=GroupedBatchSampler(
            train_examples,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
        ),
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader: Optional[DataLoader[Any]] = None
    if validation_examples:
        validation_loader = DataLoader(
            JudgmentDataset(validation_examples),
            batch_sampler=GroupedBatchSampler(
                validation_examples,
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.seed,
            ),
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    if args.frozen_encoder_epochs > 0:
        _set_encoder_trainable(model, False)
    optimizer = AdamW(
        _parameter_groups(
            model,
            encoder_lr=args.encoder_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_steps = optimizer_steps_per_epoch * args.epochs
    warmup_steps = round(total_steps * args.warmup_ratio)
    scheduler_factory = (
        get_cosine_schedule_with_warmup
        if args.scheduler == "cosine"
        else get_linear_schedule_with_warmup
    )
    scheduler = scheduler_factory(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == torch.float16)

    tokenizer.save_pretrained(output_dir / "tokenizer")
    run_config = {
        "schema_version": "ce5.training-run.v2",
        "created_at_utc": _utc_now(),
        "judgments": str(judgment_path),
        "output_dir": str(output_dir),
        "arguments": vars(args) | {
            "judgments": str(args.judgments),
            "output_dir": str(args.output_dir),
            "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        },
        "data": data_summary,
        "model_architecture": model.architecture_dict(),
        "training_objective": {
            "score_loss": args.score_loss,
            "confidence_weight_floor": args.confidence_weight_floor,
            "confidence_weight_power": args.confidence_weight_power,
            "ranking_loss_weight": args.ranking_loss_weight,
            "ranking_min_score_gap": args.ranking_min_score_gap,
            "ranking_margin": args.ranking_margin,
            "ranking_max_pairs_per_batch": args.ranking_max_pairs_per_batch,
            "diversity_loss_weight": args.diversity_loss_weight,
            "gate_balance_loss_weight": args.gate_balance_loss_weight,
            "training_pair_mix_requested": args.training_pair_mix,
        },
        "parameter_counts_at_start": _parameter_counts(model),
        "device": str(device),
        "precision": str(precision).replace("torch.", ""),
    }
    _write_json(output_dir / "run_config.json", run_config)
    wandb_run = _init_wandb(args, run_config=run_config, output_dir=output_dir)

    history: list[dict[str, Any]] = []
    best_rmse = math.inf
    global_step = 0
    for epoch in range(args.epochs):
        if epoch == args.frozen_encoder_epochs:
            _set_encoder_trainable(model, True)
        train_metrics, global_step = _train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            precision,
            args,
            epoch=epoch,
            global_step=global_step,
            wandb_run=wandb_run,
        )
        validation_metrics = (
            evaluate(model, validation_loader, device, precision, args)
            if validation_loader is not None
            else {}
        )
        epoch_record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "encoder_frozen": epoch < args.frozen_encoder_epochs,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        _write_json(output_dir / "history.json", {"epochs": history})
        model.save_checkpoint(output_dir / "last.pt")
        grant_faculty_validation = (
            validation_metrics.get("by_pair_type", {}).get("grant_faculty", {})
            if validation_metrics
            else {}
        )
        current_rmse = float(
            grant_faculty_validation.get(
                "rmse",
                validation_metrics.get("rmse", train_metrics["pointwise"]),
            )
        )
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            model.save_checkpoint(output_dir / "best.pt")
            _write_json(output_dir / "best_metrics.json", epoch_record)
        if wandb_run is not None:
            epoch_log = {
                "trainer/global_step": global_step,
                "train_epoch/epoch": epoch + 1,
                "train_epoch/encoder_frozen": float(
                    epoch < args.frozen_encoder_epochs
                ),
            }
            epoch_log.update(_flatten_numeric_metrics("train_epoch", train_metrics))
            epoch_log.update(
                _flatten_numeric_metrics("validation", validation_metrics)
            )
            wandb_run.log(epoch_log)
        print(json.dumps(epoch_record, ensure_ascii=False))

    model.save_checkpoint(output_dir / "last.pt")
    final_summary = {
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": time.time() - started,
        "epochs": args.epochs,
        "global_step": global_step,
        "best_validation_rmse": best_rmse,
        "best_metric": "validation/by_pair_type/grant_faculty/rmse",
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
    }
    if wandb_run is not None:
        final_summary["wandb_run_id"] = str(getattr(wandb_run, "id", ""))
        final_summary["wandb_run_url"] = str(getattr(wandb_run, "url", ""))
    _write_json(output_dir / "training_summary.json", final_summary)
    if wandb_run is not None:
        wandb_run.summary.update(final_summary)
        wandb_run.finish()
    print(json.dumps(final_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
