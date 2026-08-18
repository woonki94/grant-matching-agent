"""Train CE5's structured requirement-slot matcher.

The trainer consumes ``ce5.structured-judgment.v1`` records. The student sees
only raw grant/faculty text, while decomposed requirements, supporting faculty
claims, and their source spans provide privileged training supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
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
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = "ce5.structured-judgment.v1"
ARCHITECTURE_TYPE = "structured_requirement_matcher"
DEFAULT_MODEL_ID = "dleemiller/ModernCE-base-sts"
DEFAULT_JUDGMENTS = (
    REPO_ROOT
    / "ce5"
    / "dataset"
    / "judgments"
    / "structured_teacher_judgments_v1.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ce5" / "models" / "structured_requirement_matcher_v1"
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class ClaimSpan:
    claim_id: str
    claim: str
    source_span: str
    source_span_start: int
    source_span_end: int


@dataclass(frozen=True)
class RequirementSupervision:
    requirement_claim_id: str
    source_span_start: int
    source_span_end: int
    coverage_score: float
    confidence: float
    supporting_capability_claim_ids: tuple[str, ...]
    supporting_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class StructuredJudgmentExample:
    judgment_id: str
    pair_id: str
    grant_item_id: str
    grant_owner_id: str
    grant_text: str
    faculty_item_id: str
    faculty_owner_id: str
    faculty_text: str
    overall_score: float
    overall_confidence: float
    prefilter_band: str
    requirements: tuple[RequirementSupervision, ...]

    def split_value(self, field: str) -> str:
        if field == "owner":
            return f"grant:{self.grant_owner_id or self.grant_item_id}"
        value = getattr(self, field, "")
        return str(value or self.pair_id)


class StructuredJudgmentDataset(Dataset[StructuredJudgmentExample]):
    def __init__(self, examples: Sequence[StructuredJudgmentExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> StructuredJudgmentExample:
        return self.examples[index]


class GroupedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        examples: Sequence[StructuredJudgmentExample],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        grouped: dict[str, list[int]] = {}
        for index, example in enumerate(examples):
            grouped.setdefault(example.grant_item_id, []).append(index)
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


def _finite_unit(value: Any, *, field: str, location: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {field} at {location}") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise RuntimeError(f"{field} outside [0,1] at {location}")
    return parsed


def _parse_claims(
    raw_claims: Any,
    *,
    source_text: str,
    field: str,
    location: str,
) -> dict[str, ClaimSpan]:
    if not isinstance(raw_claims, list) or not raw_claims:
        raise RuntimeError(f"Missing {field} at {location}")
    output: dict[str, ClaimSpan] = {}
    for index, raw_claim in enumerate(raw_claims):
        if not isinstance(raw_claim, Mapping):
            raise RuntimeError(f"Invalid {field}[{index}] at {location}")
        claim_id = _clean_text(raw_claim.get("claim_id"))
        claim = _clean_text(raw_claim.get("claim"))
        source_span = _clean_text(raw_claim.get("source_span"))
        try:
            start = int(raw_claim.get("source_span_start"))
            end = int(raw_claim.get("source_span_end"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid {field} offsets at {location}") from exc
        if not claim_id or not claim or claim_id in output:
            raise RuntimeError(f"Missing or duplicate {field} ID at {location}")
        if start < 0 or end <= start or source_text[start:end] != source_span:
            raise RuntimeError(f"Ungrounded {field} span for {claim_id} at {location}")
        output[claim_id] = ClaimSpan(claim_id, claim, source_span, start, end)
    return output


def _parse_example(
    row: Mapping[str, Any],
    *,
    path: Path,
    line_number: int,
) -> StructuredJudgmentExample:
    location = f"{path}:{line_number}"
    if row.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported structured judgment schema at {location}: "
            f"{row.get('schema_version')!r}"
        )
    if row.get("pair_type") != "grant_faculty":
        raise RuntimeError(f"Only grant_faculty rows are supported at {location}")
    required_text = {
        "judgment_id": _clean_text(row.get("judgment_id")),
        "pair_id": _clean_text(row.get("pair_id")),
        "grant_item_id": _clean_text(row.get("grant_item_id")),
        "grant_text": _clean_text(row.get("grant_text")),
        "faculty_item_id": _clean_text(row.get("faculty_item_id")),
        "faculty_text": _clean_text(row.get("faculty_text")),
    }
    missing = [name for name, value in required_text.items() if not value]
    if missing:
        raise RuntimeError(f"Missing {', '.join(missing)} at {location}")
    grants = _parse_claims(
        row.get("grant_requirements"),
        source_text=required_text["grant_text"],
        field="grant_requirements",
        location=location,
    )
    capabilities = _parse_claims(
        row.get("faculty_capabilities"),
        source_text=required_text["faculty_text"],
        field="faculty_capabilities",
        location=location,
    )
    raw_coverage = row.get("requirement_coverage")
    if not isinstance(raw_coverage, list) or len(raw_coverage) != len(grants):
        raise RuntimeError(f"Incomplete requirement_coverage at {location}")
    requirements: list[RequirementSupervision] = []
    seen_requirements: set[str] = set()
    for index, raw_entry in enumerate(raw_coverage):
        if not isinstance(raw_entry, Mapping):
            raise RuntimeError(f"Invalid requirement_coverage[{index}] at {location}")
        requirement_id = _clean_text(raw_entry.get("requirement_claim_id"))
        if requirement_id not in grants or requirement_id in seen_requirements:
            raise RuntimeError(f"Unknown or duplicate requirement ID at {location}")
        raw_supporting = raw_entry.get("supporting_capability_claim_ids")
        if not isinstance(raw_supporting, list):
            raise RuntimeError(f"Missing faculty support IDs at {location}")
        supporting_ids: list[str] = []
        supporting_spans: list[tuple[int, int]] = []
        for raw_support_id in raw_supporting:
            support_id = _clean_text(raw_support_id)
            if support_id not in capabilities or support_id in supporting_ids:
                raise RuntimeError(f"Unknown or duplicate faculty support ID at {location}")
            supporting_ids.append(support_id)
            support = capabilities[support_id]
            supporting_spans.append(
                (support.source_span_start, support.source_span_end)
            )
        requirement = grants[requirement_id]
        requirements.append(
            RequirementSupervision(
                requirement_claim_id=requirement_id,
                source_span_start=requirement.source_span_start,
                source_span_end=requirement.source_span_end,
                coverage_score=_finite_unit(
                    raw_entry.get("coverage_score"),
                    field="coverage_score",
                    location=location,
                ),
                confidence=_finite_unit(
                    raw_entry.get("confidence"),
                    field="requirement confidence",
                    location=location,
                ),
                supporting_capability_claim_ids=tuple(supporting_ids),
                supporting_spans=tuple(supporting_spans),
            )
        )
        seen_requirements.add(requirement_id)
    if seen_requirements != set(grants):
        raise RuntimeError(f"Missing requirement coverage IDs at {location}")
    return StructuredJudgmentExample(
        **required_text,
        grant_owner_id=_clean_text(row.get("grant_owner_id")),
        faculty_owner_id=_clean_text(row.get("faculty_owner_id")),
        overall_score=_finite_unit(
            row.get("overall_score"),
            field="overall_score",
            location=location,
        ),
        overall_confidence=_finite_unit(
            row.get("overall_confidence"),
            field="overall_confidence",
            location=location,
        ),
        prefilter_band=_clean_text(row.get("prefilter_band")).lower(),
        requirements=tuple(requirements),
    )


def load_structured_judgments(
    path: Path,
    *,
    min_confidence: float,
    duplicate_policy: str,
    max_examples: int,
) -> tuple[list[StructuredJudgmentExample], dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Structured judgments not found: {path}")
    by_pair: dict[str, StructuredJudgmentExample] = {}
    rows_read = below_confidence = duplicates = 0
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
            example = _parse_example(raw, path=path, line_number=line_number)
            if example.overall_confidence < min_confidence:
                below_confidence += 1
                continue
            existing = by_pair.get(example.pair_id)
            if existing is not None:
                duplicates += 1
                if duplicate_policy == "error":
                    raise RuntimeError(f"Duplicate pair_id {example.pair_id!r}")
                if (
                    duplicate_policy == "highest-confidence"
                    and existing.overall_confidence >= example.overall_confidence
                ):
                    continue
            by_pair[example.pair_id] = example
    examples = list(by_pair.values())
    if max_examples > 0:
        examples = sorted(
            examples,
            key=lambda example: hashlib.sha256(
                example.pair_id.encode("utf-8")
            ).digest(),
        )[:max_examples]
    if not examples:
        raise RuntimeError("No structured judgments remain after filtering")
    return examples, {
        "rows_read": rows_read,
        "below_min_confidence": below_confidence,
        "duplicate_pair_ids": duplicates,
        "examples_loaded": len(examples),
    }


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
    examples: Sequence[StructuredJudgmentExample],
    *,
    validation_ratio: float,
    test_ratio: float,
    split_group: str,
    seed: int,
) -> tuple[
    list[StructuredJudgmentExample],
    list[StructuredJudgmentExample],
    list[StructuredJudgmentExample],
]:
    if validation_ratio + test_ratio >= 1.0:
        raise ValueError("validation_ratio + test_ratio must be smaller than 1")
    grouped: dict[str, list[StructuredJudgmentExample]] = {}
    for example in examples:
        grouped.setdefault(example.split_value(split_group), []).append(example)
    held_out = [
        name
        for name, ratio in (("validation", validation_ratio), ("test", test_ratio))
        if ratio > 0.0
    ]
    if len(grouped) < 1 + len(held_out):
        raise RuntimeError("Too few split groups for the requested three-way split")
    if not held_out:
        return list(examples), [], []

    ratios = {"validation": validation_ratio, "test": test_ratio}
    feature_totals = Counter(_score_bin(example.overall_score) for example in examples)
    target_rows = {name: len(examples) * ratios[name] for name in held_out}
    target_features = {
        name: {
            feature: count * ratios[name] for feature, count in feature_totals.items()
        }
        for name in held_out
    }
    current_rows = {name: 0 for name in held_out}
    current_features = {name: Counter() for name in held_out}
    group_features = {
        key: Counter(_score_bin(example.overall_score) for example in values)
        for key, values in grouped.items()
    }
    assignments: dict[str, str] = {}
    ordered = sorted(
        grouped,
        key=lambda key: (-len(grouped[key]), _stable_tiebreak(seed, key)),
    )
    for key in ordered:
        row_count = len(grouped[key])
        features = group_features[key]
        best_split = "train"
        best_gain = 0.0
        best_tiebreak = b""
        for split_name in held_out:
            row_target = max(1.0, target_rows[split_name])
            before = ((current_rows[split_name] - row_target) / row_target) ** 2
            after = (
                (current_rows[split_name] + row_count - row_target) / row_target
            ) ** 2
            for feature, count in features.items():
                target = max(1.0, target_features[split_name][feature])
                before += ((current_features[split_name][feature] - target) / target) ** 2
                after += (
                    (current_features[split_name][feature] + count - target) / target
                ) ** 2
            gain = before - after
            tiebreak = _stable_tiebreak(seed, f"{key}\x1f{split_name}")
            if gain > best_gain or (
                math.isclose(gain, best_gain)
                and gain > 0.0
                and (not best_tiebreak or tiebreak < best_tiebreak)
            ):
                best_split = split_name
                best_gain = gain
                best_tiebreak = tiebreak
        assignments[key] = best_split
        if best_split != "train":
            current_rows[best_split] += row_count
            current_features[best_split].update(features)

    result = {name: [] for name in SPLIT_NAMES}
    for key, values in grouped.items():
        result[assignments[key]].extend(values)
    for name in ("train", *held_out):
        if not result[name]:
            raise RuntimeError(f"Grouped split produced an empty {name} set")
    return result["train"], result["validation"], result["test"]


def _span_token_mask(
    offsets: Sequence[Sequence[int]],
    sequence_ids: Sequence[Optional[int]],
    *,
    sequence_id: int,
    spans: Sequence[tuple[int, int]],
) -> list[bool]:
    mask: list[bool] = []
    for token_offset, token_sequence_id in zip(offsets, sequence_ids, strict=True):
        token_start, token_end = int(token_offset[0]), int(token_offset[1])
        overlaps = any(
            token_start < span_end and token_end > span_start
            for span_start, span_end in spans
        )
        mask.append(token_sequence_id == sequence_id and token_end > token_start and overlaps)
    return mask


class StructuredJudgmentCollator:
    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        num_requirement_slots: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.num_requirement_slots = int(num_requirement_slots)

    def __call__(self, examples: Sequence[StructuredJudgmentExample]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [example.grant_text for example in examples],
            [example.faculty_text for example in examples],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        sequence_length = int(encoded["input_ids"].shape[1])
        target_rows: list[list[bool]] = []
        candidate_rows: list[list[bool]] = []
        teacher_valid = torch.zeros(
            len(examples),
            self.num_requirement_slots,
            dtype=torch.bool,
        )
        teacher_coverage = torch.zeros_like(teacher_valid, dtype=torch.float32)
        teacher_confidence = torch.zeros_like(teacher_valid, dtype=torch.float32)
        target_span_masks = torch.zeros(
            len(examples),
            self.num_requirement_slots,
            sequence_length,
            dtype=torch.bool,
        )
        candidate_span_masks = torch.zeros_like(target_span_masks)
        teacher_has_support = torch.zeros_like(teacher_valid)
        requirement_ids: list[list[str]] = []

        for batch_index, example in enumerate(examples):
            if len(example.requirements) > self.num_requirement_slots:
                raise RuntimeError(
                    f"Pair {example.pair_id} has {len(example.requirements)} "
                    f"requirements but the model has {self.num_requirement_slots} slots"
                )
            sequence_ids = encoded.sequence_ids(batch_index)
            if sequence_ids is None or len(sequence_ids) != sequence_length:
                raise RuntimeError("Fast tokenizer sequence IDs are required")
            target_row = [sequence_id == 0 for sequence_id in sequence_ids]
            candidate_row = [sequence_id == 1 for sequence_id in sequence_ids]
            if not any(target_row) or not any(candidate_row):
                raise RuntimeError("Tokenization removed an entire pair side")
            target_rows.append(target_row)
            candidate_rows.append(candidate_row)
            item_offsets = offsets[batch_index].tolist()
            item_requirement_ids: list[str] = []
            for requirement_index, requirement in enumerate(example.requirements):
                teacher_valid[batch_index, requirement_index] = True
                teacher_coverage[batch_index, requirement_index] = requirement.coverage_score
                teacher_confidence[batch_index, requirement_index] = requirement.confidence
                target_tokens = _span_token_mask(
                    item_offsets,
                    sequence_ids,
                    sequence_id=0,
                    spans=[
                        (
                            requirement.source_span_start,
                            requirement.source_span_end,
                        )
                    ],
                )
                if not any(target_tokens):
                    raise RuntimeError(
                        f"Grant evidence span was truncated for pair {example.pair_id}; "
                        "increase --max-length"
                    )
                target_span_masks[batch_index, requirement_index] = torch.tensor(
                    target_tokens,
                    dtype=torch.bool,
                )
                if requirement.supporting_spans:
                    candidate_tokens = _span_token_mask(
                        item_offsets,
                        sequence_ids,
                        sequence_id=1,
                        spans=requirement.supporting_spans,
                    )
                    if not any(candidate_tokens):
                        raise RuntimeError(
                            f"Faculty evidence span was truncated for pair "
                            f"{example.pair_id}; increase --max-length"
                        )
                    candidate_span_masks[
                        batch_index,
                        requirement_index,
                    ] = torch.tensor(candidate_tokens, dtype=torch.bool)
                    teacher_has_support[batch_index, requirement_index] = True
                item_requirement_ids.append(requirement.requirement_claim_id)
            requirement_ids.append(item_requirement_ids)

        attention_mask = encoded["attention_mask"].bool()
        encoded["target_mask"] = torch.tensor(target_rows, dtype=torch.bool) & attention_mask
        encoded["candidate_mask"] = (
            torch.tensor(candidate_rows, dtype=torch.bool) & attention_mask
        )
        return {
            "encoded": encoded,
            "overall_scores": torch.tensor(
                [example.overall_score for example in examples],
                dtype=torch.float32,
            ),
            "overall_confidences": torch.tensor(
                [example.overall_confidence for example in examples],
                dtype=torch.float32,
            ),
            "teacher_valid": teacher_valid,
            "teacher_coverage": teacher_coverage,
            "teacher_confidence": teacher_confidence,
            "target_span_masks": target_span_masks,
            "candidate_span_masks": candidate_span_masks,
            "teacher_has_support": teacher_has_support,
            "requirement_ids": requirement_ids,
            "query_ids": [example.grant_item_id for example in examples],
            "pair_ids": [example.pair_id for example in examples],
        }


def _minimum_cost_assignment(cost: Tensor) -> list[tuple[int, int]]:
    """Exact rectangular Hungarian assignment as (slot, teacher) pairs."""

    if cost.ndim != 2:
        raise ValueError("cost must have shape [teachers, slots]")
    teacher_count, slot_count = map(int, cost.shape)
    if teacher_count == 0:
        return []
    if teacher_count > slot_count:
        raise ValueError("There must be at least as many slots as teachers")
    values = cost.detach().float().cpu().tolist()
    u = [0.0] * (teacher_count + 1)
    v = [0.0] * (slot_count + 1)
    p = [0] * (slot_count + 1)
    way = [0] * (slot_count + 1)
    for teacher in range(1, teacher_count + 1):
        p[0] = teacher
        column = 0
        min_values = [math.inf] * (slot_count + 1)
        used = [False] * (slot_count + 1)
        while True:
            used[column] = True
            current_teacher = p[column]
            delta = math.inf
            next_column = 0
            for candidate_column in range(1, slot_count + 1):
                if used[candidate_column]:
                    continue
                current = (
                    values[current_teacher - 1][candidate_column - 1]
                    - u[current_teacher]
                    - v[candidate_column]
                )
                if current < min_values[candidate_column]:
                    min_values[candidate_column] = current
                    way[candidate_column] = column
                if min_values[candidate_column] < delta:
                    delta = min_values[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(slot_count + 1):
                if used[candidate_column]:
                    u[p[candidate_column]] += delta
                    v[candidate_column] -= delta
                else:
                    min_values[candidate_column] -= delta
            column = next_column
            if p[column] == 0:
                break
        while True:
            previous = way[column]
            p[column] = p[previous]
            column = previous
            if column == 0:
                break
    assignment = [
        (column - 1, p[column] - 1)
        for column in range(1, slot_count + 1)
        if p[column] != 0
    ]
    if len(assignment) != teacher_count:
        raise RuntimeError("Hungarian matching did not assign every teacher requirement")
    return sorted(assignment, key=lambda pair: pair[1])


def _attention_mass(attention: Tensor, masks: Tensor) -> Tensor:
    return torch.sum(attention * masks.to(attention.dtype), dim=-1).clamp_min(1e-8)


def _match_slots(output: Any, batch: Mapping[str, Any], args: argparse.Namespace) -> list[tuple[int, int, int]]:
    matches: list[tuple[int, int, int]] = []
    with torch.no_grad():
        for batch_index in range(int(batch["teacher_valid"].shape[0])):
            teacher_count = int(batch["teacher_valid"][batch_index].sum().item())
            teacher_coverage = batch["teacher_coverage"][batch_index, :teacher_count]
            coverage_cost = torch.abs(
                teacher_coverage[:, None]
                - output.slot_coverage_scores[batch_index][None, :]
            )
            target_masks = batch["target_span_masks"][batch_index, :teacher_count]
            target_mass = torch.einsum(
                "sl,rl->rs",
                output.target_attention_weights[batch_index],
                target_masks.to(output.target_attention_weights.dtype),
            ).clamp_min(1e-8)
            cost = (
                args.match_coverage_cost * coverage_cost
                - args.match_grant_span_cost * torch.log(target_mass)
            )
            has_support = batch["teacher_has_support"][batch_index, :teacher_count]
            if torch.any(has_support):
                candidate_masks = batch["candidate_span_masks"][
                    batch_index,
                    :teacher_count,
                ]
                candidate_mass = torch.einsum(
                    "sl,rl->rs",
                    output.candidate_attention_weights[batch_index],
                    candidate_masks.to(output.candidate_attention_weights.dtype),
                ).clamp_min(1e-8)
                support_cost = -torch.log(candidate_mass)
                cost = cost + args.match_faculty_span_cost * torch.where(
                    has_support[:, None],
                    support_cost,
                    torch.zeros_like(support_cost),
                )
            for slot_index, teacher_index in _minimum_cost_assignment(cost):
                matches.append((batch_index, slot_index, teacher_index))
    return matches


def _confidence_weights(values: Tensor, *, floor: float, power: float) -> Tensor:
    if power == 0.0:
        return torch.ones_like(values)
    return values.clamp(min=floor, max=1.0).pow(power)


def _weighted_mean(losses: Tensor, weights: Tensor) -> Tensor:
    return torch.sum(losses * weights) / weights.sum().clamp_min(1e-8)


def _ranking_loss(
    logits: Tensor,
    targets: Tensor,
    query_ids: Sequence[str],
    *,
    min_score_gap: float,
    max_pairs: int,
) -> tuple[Tensor, int]:
    positive: list[int] = []
    negative: list[int] = []
    gaps: list[float] = []
    grouped: dict[str, list[int]] = {}
    for index, query_id in enumerate(query_ids):
        grouped.setdefault(query_id, []).append(index)
    target_values = targets.detach().float().cpu().tolist()
    for indices in grouped.values():
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                difference = target_values[left] - target_values[right]
                if abs(difference) < min_score_gap:
                    continue
                high, low = (left, right) if difference > 0 else (right, left)
                positive.append(high)
                negative.append(low)
                gaps.append(abs(difference))
    if not positive:
        return logits.sum() * 0.0, 0
    if max_pairs > 0 and len(positive) > max_pairs:
        keep = sorted(range(len(gaps)), key=gaps.__getitem__, reverse=True)[:max_pairs]
        positive = [positive[index] for index in keep]
        negative = [negative[index] for index in keep]
    positive_index = torch.tensor(positive, device=logits.device)
    negative_index = torch.tensor(negative, device=logits.device)
    return F.softplus(
        -(logits[positive_index] - logits[negative_index])
    ).mean(), len(positive)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    moved["encoded"] = {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch["encoded"].items()
    }
    for key in (
        "overall_scores",
        "overall_confidences",
        "teacher_valid",
        "teacher_coverage",
        "teacher_confidence",
        "target_span_masks",
        "candidate_span_masks",
        "teacher_has_support",
    ):
        moved[key] = batch[key].to(device, non_blocking=device.type == "cuda")
    return moved


def _batch_losses(
    model: Any,
    raw_batch: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Tensor], Any, list[tuple[int, int, int]]]:
    batch = _move_batch(raw_batch, device)
    output = model(**batch["encoded"])
    matches = _match_slots(output, batch, args)
    batch_indices = torch.tensor([match[0] for match in matches], device=device)
    slot_indices = torch.tensor([match[1] for match in matches], device=device)
    teacher_indices = torch.tensor([match[2] for match in matches], device=device)

    overall_weights = _confidence_weights(
        batch["overall_confidences"],
        floor=args.confidence_weight_floor,
        power=args.confidence_weight_power,
    )
    high_weights = torch.where(
        batch["overall_scores"] >= args.high_score_threshold,
        torch.full_like(batch["overall_scores"], args.high_score_weight),
        torch.ones_like(batch["overall_scores"]),
    )
    overall = _weighted_mean(
        F.smooth_l1_loss(
            output.scores,
            batch["overall_scores"],
            reduction="none",
        ),
        overall_weights * high_weights,
    )
    overall_confidence = F.smooth_l1_loss(
        output.overall_confidence_score,
        batch["overall_confidences"],
    )

    teacher_coverage = batch["teacher_coverage"][batch_indices, teacher_indices]
    teacher_confidence = batch["teacher_confidence"][batch_indices, teacher_indices]
    requirement_weights = _confidence_weights(
        teacher_confidence,
        floor=args.confidence_weight_floor,
        power=args.confidence_weight_power,
    )
    slot_coverage = _weighted_mean(
        F.smooth_l1_loss(
            output.slot_coverage_scores[batch_indices, slot_indices],
            teacher_coverage,
            reduction="none",
        ),
        requirement_weights,
    )
    slot_confidence = F.smooth_l1_loss(
        output.slot_confidence_scores[batch_indices, slot_indices],
        teacher_confidence,
    )

    activity_targets = torch.zeros_like(output.slot_active_logits)
    activity_targets[batch_indices, slot_indices] = 1.0
    activity = F.binary_cross_entropy_with_logits(
        output.slot_active_logits,
        activity_targets,
    )
    matched_target_attention = output.target_attention_weights[
        batch_indices,
        slot_indices,
    ]
    matched_target_masks = batch["target_span_masks"][
        batch_indices,
        teacher_indices,
    ]
    grant_span = -torch.log(
        _attention_mass(matched_target_attention, matched_target_masks)
    ).mean()

    support_flags = batch["teacher_has_support"][batch_indices, teacher_indices]
    if torch.any(support_flags):
        faculty_attention = output.candidate_attention_weights[
            batch_indices[support_flags],
            slot_indices[support_flags],
        ]
        faculty_masks = batch["candidate_span_masks"][
            batch_indices[support_flags],
            teacher_indices[support_flags],
        ]
        faculty_span = -torch.log(
            _attention_mass(faculty_attention, faculty_masks)
        ).mean()
    else:
        faculty_span = output.logits.sum() * 0.0
    ranking, ranking_pairs = _ranking_loss(
        output.logits,
        batch["overall_scores"],
        batch["query_ids"],
        min_score_gap=args.ranking_min_score_gap,
        max_pairs=args.ranking_max_pairs_per_batch,
    )
    total = (
        args.overall_loss_weight * overall
        + args.coverage_loss_weight * slot_coverage
        + args.activity_loss_weight * activity
        + args.grant_span_loss_weight * grant_span
        + args.faculty_span_loss_weight * faculty_span
        + args.confidence_loss_weight * (slot_confidence + overall_confidence)
        + args.ranking_loss_weight * ranking
    )
    losses = {
        "total": total,
        "overall": overall,
        "coverage": slot_coverage,
        "activity": activity,
        "grant_span": grant_span,
        "faculty_span": faculty_span,
        "slot_confidence": slot_confidence,
        "overall_confidence": overall_confidence,
        "ranking": ranking,
        "ranking_pairs": output.logits.new_tensor(float(ranking_pairs)),
    }
    return losses, output, matches


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
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return numerator / (left_scale * right_scale)


def _prediction_metrics(
    predictions: Sequence[float],
    targets: Sequence[float],
) -> dict[str, float | int]:
    errors = [
        prediction - target
        for prediction, target in zip(predictions, targets, strict=True)
    ]
    mse = sum(error * error for error in errors) / max(1, len(errors))
    return {
        "examples": len(predictions),
        "mae": sum(abs(error) for error in errors) / max(1, len(errors)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "pearson": _pearson(predictions, targets),
        "spearman": _pearson(_average_ranks(predictions), _average_ranks(targets)),
        "prediction_mean": sum(predictions) / max(1, len(predictions)),
        "target_mean": sum(targets) / max(1, len(targets)),
    }


def _autocast_context(device: torch.device, dtype: torch.dtype) -> Any:
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


@torch.no_grad()
def evaluate(
    model: Any,
    loader: DataLoader[Any],
    device: torch.device,
    precision: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.eval()
    loss_totals: Counter[str] = Counter()
    predictions: list[float] = []
    targets: list[float] = []
    coverage_errors: list[float] = []
    target_masses: list[float] = []
    faculty_masses: list[float] = []
    matched_active: list[float] = []
    unmatched_active: list[float] = []
    contributions: list[float] = []
    example_count = 0
    for raw_batch in loader:
        with _autocast_context(device, precision):
            losses, output, matches = _batch_losses(model, raw_batch, device, args)
        batch = _move_batch(raw_batch, device)
        batch_size = int(batch["overall_scores"].numel())
        example_count += batch_size
        for name, value in losses.items():
            if name != "ranking_pairs":
                loss_totals[name] += float(value.detach().float().item()) * batch_size
        predictions.extend(output.scores.detach().float().cpu().tolist())
        targets.extend(batch["overall_scores"].detach().float().cpu().tolist())
        contributions.extend(
            output.structured_contribution.detach().float().cpu().tolist()
        )
        matched_set: set[tuple[int, int]] = set()
        for batch_index, slot_index, teacher_index in matches:
            matched_set.add((batch_index, slot_index))
            predicted = float(
                output.slot_coverage_scores[batch_index, slot_index]
                .detach()
                .float()
                .item()
            )
            target = float(batch["teacher_coverage"][batch_index, teacher_index].item())
            coverage_errors.append(predicted - target)
            target_masses.append(
                float(
                    _attention_mass(
                        output.target_attention_weights[batch_index, slot_index],
                        batch["target_span_masks"][batch_index, teacher_index],
                    )
                    .detach()
                    .float()
                    .item()
                )
            )
            if bool(batch["teacher_has_support"][batch_index, teacher_index]):
                faculty_masses.append(
                    float(
                        _attention_mass(
                            output.candidate_attention_weights[batch_index, slot_index],
                            batch["candidate_span_masks"][batch_index, teacher_index],
                        )
                        .detach()
                        .float()
                        .item()
                    )
                )
            matched_active.append(
                float(
                    output.slot_active_probabilities[batch_index, slot_index]
                    .detach()
                    .float()
                    .item()
                )
            )
        for batch_index in range(batch_size):
            for slot_index in range(
                model.architecture_config.num_requirement_slots
            ):
                if (batch_index, slot_index) not in matched_set:
                    unmatched_active.append(
                        float(
                            output.slot_active_probabilities[batch_index, slot_index]
                            .detach()
                            .float()
                            .item()
                        )
                    )
    metrics = _prediction_metrics(predictions, targets)
    coverage_mse = (
        sum(error * error for error in coverage_errors) / max(1, len(coverage_errors))
    )
    return {
        "loss": {
            name: total / max(1, example_count)
            for name, total in sorted(loss_totals.items())
        },
        **metrics,
        "requirement_coverage_mae": sum(abs(error) for error in coverage_errors)
        / max(1, len(coverage_errors)),
        "requirement_coverage_rmse": math.sqrt(coverage_mse),
        "grant_span_attention_mass": sum(target_masses) / max(1, len(target_masses)),
        "faculty_span_attention_mass": sum(faculty_masses)
        / max(1, len(faculty_masses)),
        "matched_slot_active_mean": sum(matched_active) / max(1, len(matched_active)),
        "unmatched_slot_active_mean": sum(unmatched_active)
        / max(1, len(unmatched_active)),
        "structured_contribution_mean": sum(contributions)
        / max(1, len(contributions)),
        "matched_requirements": len(coverage_errors),
        "supported_requirements": len(faculty_masses),
    }


def _set_encoder_trainable(model: Any, trainable: bool) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = trainable


def _parameter_groups(
    model: Any,
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
    output = []
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


def _train_epoch(
    model: Any,
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
    wandb_run: Any,
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals: Counter[str] = Counter()
    example_count = 0
    progress: Iterable[Any] = loader
    if tqdm is not None and not args.no_progress:
        progress = tqdm(loader, desc=f"train epoch {epoch + 1}", dynamic_ncols=True)
    for batch_index, raw_batch in enumerate(progress):
        with _autocast_context(device, precision):
            losses, _, _ = _batch_losses(model, raw_batch, device, args)
            backward_loss = losses["total"] / args.gradient_accumulation_steps
        scaler.scale(backward_loss).backward()
        batch_size = int(raw_batch["overall_scores"].numel())
        example_count += batch_size
        for name, value in losses.items():
            totals[name] += float(value.detach().float().item()) * batch_size
        should_step = (
            (batch_index + 1) % args.gradient_accumulation_steps == 0
            or batch_index + 1 == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if wandb_run is not None and global_step % args.wandb_log_every_steps == 0:
                payload = {
                    f"train/loss_{name}": total / max(1, example_count)
                    for name, total in totals.items()
                }
                payload["train/epoch"] = epoch + 1
                payload["train/encoder_frozen"] = float(
                    not any(parameter.requires_grad for parameter in model.encoder.parameters())
                )
                payload["trainer/global_step"] = global_step
                for group in optimizer.param_groups:
                    payload[
                        f"train/learning_rate/{group.get('group_name', 'parameters')}"
                    ] = float(group["lr"])
                wandb_run.log(payload)
        if tqdm is not None and hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{totals['total'] / example_count:.4f}")
    return {
        name: total / max(1, example_count) for name, total in sorted(totals.items())
    }, global_step


def _dataset_summary(examples: Sequence[StructuredJudgmentExample]) -> dict[str, Any]:
    if not examples:
        return {"examples": 0}
    scores = [example.overall_score for example in examples]
    requirement_counts = [len(example.requirements) for example in examples]
    return {
        "examples": len(examples),
        "grant_owners": len({example.grant_owner_id for example in examples}),
        "grant_items": len({example.grant_item_id for example in examples}),
        "faculty_items": len({example.faculty_item_id for example in examples}),
        "score_bins": dict(
            sorted(Counter(_score_bin(score) for score in scores).items())
        ),
        "score_mean": sum(scores) / len(scores),
        "requirements": sum(requirement_counts),
        "requirements_per_pair": dict(sorted(Counter(requirement_counts).items())),
        "max_requirements_per_pair": max(requirement_counts),
        "supported_requirements": sum(
            bool(requirement.supporting_spans)
            for example in examples
            for requirement in example.requirements
        ),
        "prefilter_bands": dict(
            sorted(Counter(example.prefilter_band for example in examples).items())
        ),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _pair_id_sha256(examples: Sequence[StructuredJudgmentExample]) -> str:
    digest = hashlib.sha256()
    for pair_id in sorted(example.pair_id for example in examples):
        digest.update(pair_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _split_manifest_entry(
    examples: Sequence[StructuredJudgmentExample],
    *,
    split_group: str,
) -> dict[str, Any]:
    return {
        "examples": len(examples),
        "pair_id_sha256": _pair_id_sha256(examples),
        "groups": sorted({example.split_value(split_group) for example in examples}),
        "summary": _dataset_summary(examples),
    }


def _flatten_numeric(prefix: str, value: Any) -> dict[str, float]:
    if isinstance(value, Mapping):
        output: dict[str, float] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            output.update(_flatten_numeric(child_prefix, child))
        return output
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return {prefix: float(value)}
    return {}


def _init_wandb(args: argparse.Namespace, *, config: Mapping[str, Any], output_dir: Path) -> Any:
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging was requested but wandb is not installed") from exc
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()] or None,
        mode=args.wandb_mode,
        dir=str(output_dir),
        config=dict(config),
    )
    run.define_metric("trainer/global_step")
    for pattern in ("train/*", "train_epoch/*", "validation/*"):
        run.define_metric(pattern, step_metric="trainer/global_step")
    return run


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0,1]")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the CE5 structured grant-requirement matcher."
    )
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
        choices=("owner", "grant_owner_id", "grant_item_id", "pair_id"),
        default="owner",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--num-requirement-slots",
        type=_nonnegative_int,
        default=0,
        help="0 derives the slot count from the maximum requirements in the dataset.",
    )
    parser.add_argument("--latent-dim", type=_positive_int, default=256)
    parser.add_argument("--num-refinement-blocks", type=_positive_int, default=2)
    parser.add_argument("--cross-attention-heads", type=_positive_int, default=8)
    parser.add_argument("--self-attention-heads", type=_positive_int, default=4)
    parser.add_argument("--ffn-dim", type=_positive_int, default=768)
    parser.add_argument("--dropout", type=_unit_interval, default=0.1)
    parser.add_argument("--no-base-sts-expert", action="store_true")
    parser.add_argument(
        "--initial-structured-contribution",
        type=_positive_float,
        default=0.1,
    )

    parser.add_argument("--match-coverage-cost", type=_nonnegative_float, default=1.0)
    parser.add_argument("--match-grant-span-cost", type=_nonnegative_float, default=1.0)
    parser.add_argument("--match-faculty-span-cost", type=_nonnegative_float, default=1.0)
    parser.add_argument("--overall-loss-weight", type=_nonnegative_float, default=1.0)
    parser.add_argument("--coverage-loss-weight", type=_nonnegative_float, default=1.0)
    parser.add_argument("--activity-loss-weight", type=_nonnegative_float, default=0.25)
    parser.add_argument("--grant-span-loss-weight", type=_nonnegative_float, default=0.25)
    parser.add_argument("--faculty-span-loss-weight", type=_nonnegative_float, default=0.25)
    parser.add_argument("--confidence-loss-weight", type=_nonnegative_float, default=0.05)
    parser.add_argument("--ranking-loss-weight", type=_nonnegative_float, default=0.1)
    parser.add_argument("--ranking-min-score-gap", type=_unit_interval, default=0.1)
    parser.add_argument(
        "--ranking-max-pairs-per-batch",
        type=_nonnegative_int,
        default=512,
    )
    parser.add_argument("--confidence-weight-floor", type=_unit_interval, default=0.25)
    parser.add_argument("--confidence-weight-power", type=_nonnegative_float, default=1.0)
    parser.add_argument("--high-score-threshold", type=_unit_interval, default=0.75)
    parser.add_argument("--high-score-weight", type=_positive_float, default=1.0)

    parser.add_argument("--epochs", type=_positive_int, default=5)
    parser.add_argument("--frozen-encoder-epochs", type=_nonnegative_int, default=1)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--max-length", type=_positive_int, default=128)
    parser.add_argument("--encoder-lr", type=_positive_float, default=1e-5)
    parser.add_argument("--head-lr", type=_positive_float, default=1e-4)
    parser.add_argument("--weight-decay", type=_nonnegative_float, default=0.01)
    parser.add_argument("--warmup-ratio", type=_unit_interval, default=0.1)
    parser.add_argument("--scheduler", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--gradient-accumulation-steps", type=_positive_int, default=1)
    parser.add_argument("--max-grad-norm", type=_positive_float, default=1.0)
    parser.add_argument("--device", default="auto")
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
        default="ce5,structured-distillation,requirement-slots",
    )
    parser.add_argument("--wandb-log-every-steps", type=_positive_int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser


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


def main() -> int:
    args = build_parser().parse_args()
    if args.validation_ratio <= 0.0:
        raise ValueError(
            "--validation-ratio must be greater than zero because checkpoint "
            "selection uses held-out validation RMSE"
        )
    started = time.time()
    _set_seed(args.seed)
    judgment_path = _resolve_path(args.judgments)
    output_dir = _resolve_path(args.output_dir)
    examples, loading_stats = load_structured_judgments(
        judgment_path,
        min_confidence=args.min_confidence,
        duplicate_policy=args.duplicate_policy,
        max_examples=args.max_examples,
    )
    dataset_max_requirements = max(len(example.requirements) for example in examples)
    num_requirement_slots = args.num_requirement_slots or dataset_max_requirements
    if num_requirement_slots < dataset_max_requirements:
        raise RuntimeError(
            f"--num-requirement-slots={num_requirement_slots} is smaller than the "
            f"dataset maximum of {dataset_max_requirements}"
        )
    train_examples, validation_examples, test_examples = split_examples_three_way(
        examples,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        split_group=args.split_group,
        seed=args.seed,
    )
    data_summary = {
        "loading": loading_stats,
        "all": _dataset_summary(examples),
        "train": _dataset_summary(train_examples),
        "validation": _dataset_summary(validation_examples),
        "test": _dataset_summary(test_examples),
        "split_group": args.split_group,
        "num_requirement_slots": num_requirement_slots,
    }
    if args.dry_run:
        print(json.dumps(data_summary, ensure_ascii=False, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "split_manifest.json",
        {
            "schema_version": "ce5.structured-split-manifest.v1",
            "created_at_utc": _utc_now(),
            "judgments": str(judgment_path),
            "seed": args.seed,
            "split_group": args.split_group,
            "validation_ratio": args.validation_ratio,
            "test_ratio": args.test_ratio,
            "train": _split_manifest_entry(train_examples, split_group=args.split_group),
            "validation": _split_manifest_entry(
                validation_examples,
                split_group=args.split_group,
            ),
            "test": _split_manifest_entry(test_examples, split_group=args.split_group),
        },
    )

    from transformers import (
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )

    from ce5.modeling.structured_requirement_matcher import (
        ModernCEStructuredRequirementMatcher,
    )

    device = _resolve_device(args.device)
    precision = _resolve_precision(args.precision, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    resume_path = _resolve_path(args.resume_checkpoint) if args.resume_checkpoint else None
    if resume_path is not None:
        model = ModernCEStructuredRequirementMatcher.from_checkpoint(
            resume_path,
            torch_dtype=precision,
            trust_remote_code=args.trust_remote_code,
        )
        if model.architecture_config.num_requirement_slots != num_requirement_slots:
            raise RuntimeError("Resume checkpoint slot count differs from the dataset")
        tokenizer_model_id = model.architecture_config.backbone_model_id
    else:
        model = ModernCEStructuredRequirementMatcher.from_pretrained(
            args.model_id,
            num_requirement_slots=num_requirement_slots,
            latent_dim=args.latent_dim,
            num_refinement_blocks=args.num_refinement_blocks,
            cross_attention_heads=args.cross_attention_heads,
            self_attention_heads=args.self_attention_heads,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
            use_base_sts_expert=not args.no_base_sts_expert,
            initial_structured_contribution=args.initial_structured_contribution,
            torch_dtype=precision,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer_model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_id,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Structured span supervision requires a fast tokenizer")
    model.to(device)
    if args.gradient_checkpointing:
        enable = getattr(model.encoder, "gradient_checkpointing_enable", None)
        if not callable(enable):
            raise RuntimeError("The encoder does not support gradient checkpointing")
        enable()

    collator = StructuredJudgmentCollator(
        tokenizer,
        max_length=args.max_length,
        num_requirement_slots=num_requirement_slots,
    )
    train_loader = DataLoader(
        StructuredJudgmentDataset(train_examples),
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
    validation_loader = DataLoader(
        StructuredJudgmentDataset(validation_examples),
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
    steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * args.epochs
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
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and precision == torch.float16,
    )
    tokenizer.save_pretrained(output_dir / "tokenizer")
    run_config = {
        "schema_version": "ce5.structured-training-run.v1",
        "created_at_utc": _utc_now(),
        "judgments": str(judgment_path),
        "output_dir": str(output_dir),
        "architecture_type": ARCHITECTURE_TYPE,
        "arguments": vars(args)
        | {
            "judgments": str(args.judgments),
            "output_dir": str(args.output_dir),
            "resume_checkpoint": str(args.resume_checkpoint)
            if args.resume_checkpoint
            else None,
            "num_requirement_slots": num_requirement_slots,
        },
        "data": data_summary,
        "model_architecture": model.architecture_dict(),
        "device": str(device),
        "precision": str(precision).replace("torch.", ""),
        "optimizer_steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
    }
    _write_json(output_dir / "run_config.json", run_config)
    wandb_run = _init_wandb(args, config=run_config, output_dir=output_dir)

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
        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            precision,
            args,
        )
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "encoder_frozen": not any(
                parameter.requires_grad for parameter in model.encoder.parameters()
            ),
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        _write_json(output_dir / "history.json", {"epochs": history})
        model.save_checkpoint(output_dir / "last.pt")
        if float(validation_metrics["rmse"]) < best_rmse:
            best_rmse = float(validation_metrics["rmse"])
            model.save_checkpoint(output_dir / "best.pt")
            _write_json(output_dir / "best_metrics.json", record)
        if wandb_run is not None:
            payload = {"trainer/global_step": global_step, "train_epoch/epoch": epoch + 1}
            payload.update(_flatten_numeric("train_epoch", train_metrics))
            payload.update(_flatten_numeric("validation", validation_metrics))
            wandb_run.log(payload)
        print(json.dumps(record, ensure_ascii=False))

    summary = {
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": time.time() - started,
        "epochs": args.epochs,
        "global_step": global_step,
        "best_validation_rmse": best_rmse,
        "best_metric": "validation/rmse",
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
    }
    _write_json(output_dir / "training_summary.json", summary)
    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
