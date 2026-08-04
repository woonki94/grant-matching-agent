"""Score CE5 prefilter candidates with a local instruction-tuned LLM.

This is the offline teacher stage.  It reads grant-faculty, grant-grant, and
faculty-faculty high/mid/low candidate sets created by ``build_prefilter.py``,
builds one independent prompt per directed pair, and uses vLLM to produce
continuous capability-coverage judgments.

CE-STS scores, ranks, and selection bands are deliberately excluded from the
teacher prompt.  Pair provenance and same-owner status are also hidden so the
teacher must judge only the two capability statements.  Output is append-only
and resumable.  Invalid generations are retried once and then written to a
separate error file; they are never silently converted to zero-score labels.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASET_DIR = REPO_ROOT / "ce5" / "dataset"
SOURCE_DIR = DATASET_DIR / "source"
DEFAULT_GRANT_FACULTY_PREFILTER = SOURCE_DIR / "prefilter_candidates.jsonl"
DEFAULT_GRANT_GRANT_PREFILTER = SOURCE_DIR / "grant_pair_candidates.jsonl"
DEFAULT_FACULTY_FACULTY_PREFILTER = SOURCE_DIR / "faculty_pair_candidates.jsonl"
DEFAULT_OUTPUT = DATASET_DIR / "judgments" / "teacher_judgments_v2.jsonl"
DEFAULT_ERRORS = DATASET_DIR / "judgments" / "teacher_judgment_errors_v2.jsonl"
DEFAULT_MANIFEST = DATASET_DIR / "judgments" / "teacher_judgments_v2.manifest.json"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3-14B"
GRANT_FACULTY_PREFILTER_SCHEMA_VERSION = 2
GRANT_GRANT_PREFILTER_SCHEMA_VERSION = "ce5.grant-pair-prefilter.v1"
FACULTY_FACULTY_PREFILTER_SCHEMA_VERSION = "ce5.faculty-pair-prefilter.v1"
JUDGMENT_SCHEMA_VERSION = "ce5.judgment.v2"
ERROR_SCHEMA_VERSION = "ce5.judgment-error.v2"
PROMPT_VERSION = "directed-capability-coverage-v2"
BANDS = ("high", "mid", "low")
PAIR_TYPES = ("grant_faculty", "grant_grant", "faculty_faculty")


SYSTEM_PROMPT = """
You are a strict evaluator of directed coverage between two capability statements.

Your task is directional:
Determine how completely the CANDIDATE CAPABILITY would satisfy or cover the TARGET CAPABILITY.

Treat both statements as normalized descriptions of capabilities. Judge only
their meanings. Do not infer anything from their source, owner, wording style,
or possible relationship outside the text.

This is not ordinary semantic similarity. Do not give a high score merely
because both statements mention the same topic, domain, population,
technology, or broad goal. The candidate must contain a relevant method,
expertise, experience, system, or transferable ability that covers the target.

Do not assume important capabilities that are absent from the candidate.
Closely related or transferable capability may receive partial credit, but
missing essential parts of the target must reduce the score. Coverage may be
asymmetric: the score for candidate B covering target A need not equal the
score for candidate A covering target B.

Use a genuinely continuous score from 0.00 to 1.00. Do not quantize scores to
fixed increments such as 0.25. Choose the value that best reflects the degree
of coverage:
- 0.00: unrelated, contradictory, or no useful capability coverage
- 0.01-0.24: only a weak topical or transferable connection
- 0.25-0.49: limited coverage with major missing capabilities
- 0.50-0.74: meaningful partial coverage with important gaps
- 0.75-0.94: strong coverage with a smaller but real gap
- 0.95-1.00: direct and essentially complete coverage

Confidence describes confidence in your judgment, not match strength:
- 0.00: the texts are too vague or ambiguous to judge reliably
- 1.00: the relationship is explicit and unambiguous

Return exactly one JSON object with this schema:
{
  "score": <number from 0.00 to 1.00>,
  "confidence": <number from 0.00 to 1.00>,
  "rationale": "<one concise sentence grounded only in the two statements>"
}

Do not output markdown, analysis, aspect scores, a categorical band, or any text outside the JSON object.
""".strip()


USER_PROMPT_TEMPLATE = """
TARGET CAPABILITY:
{target_text}

CANDIDATE CAPABILITY:
{candidate_text}
""".strip()


REPAIR_PROMPT = """
Your previous response did not match the required JSON schema. Return only one valid JSON object containing numeric score and confidence values in [0,1] and one concise rationale string. Do not include markdown or any other text.
""".strip()


@dataclass(frozen=True)
class CandidatePair:
    pair_id: str
    source_pair_id: str
    pair_type: str
    direction: str
    target_item_id: str
    target_owner_id: str | int
    target_keyword_index: int
    target_text: str
    candidate_item_id: str
    candidate_owner_id: str | int
    candidate_keyword_index: int
    candidate_text: str
    prefilter_band: str
    prefilter_score_direction: str
    ce_sts_score: float
    ce_sts_logit: float
    ce_sts_rank: int
    ce_sts_rank_percentile: float


@dataclass(frozen=True)
class ParsedJudgment:
    score: float
    confidence: float
    rationale: str


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _resolve_path(value: Path) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(*parts: Any, prefix: str) -> str:
    raw = "\x1f".join(_clean_text(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:24]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_bands(value: str) -> tuple[str, ...]:
    requested = tuple(
        token.strip().lower() for token in value.split(",") if token.strip()
    )
    invalid = sorted(set(requested) - set(BANDS))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported band(s): {', '.join(invalid)}; use high,mid,low"
        )
    if not requested:
        raise argparse.ArgumentTypeError("At least one band is required")
    return tuple(dict.fromkeys(requested))


def _parse_pair_types(value: str) -> tuple[str, ...]:
    requested = tuple(
        token.strip().lower().replace("-", "_")
        for token in value.split(",")
        if token.strip()
    )
    invalid = sorted(set(requested) - set(PAIR_TYPES))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported pair type(s): {', '.join(invalid)}; "
            "use grant_faculty,grant_grant,faculty_faculty"
        )
    if not requested:
        raise argparse.ArgumentTypeError("At least one pair type is required")
    return tuple(dict.fromkeys(requested))


def _directed_pair(
    *,
    source_pair_id: str,
    pair_type: str,
    direction: str,
    target_item_id: str,
    target_owner_id: str | int,
    target_keyword_index: int,
    target_text: str,
    candidate_item_id: str,
    candidate_owner_id: str | int,
    candidate_keyword_index: int,
    candidate_text: str,
    prefilter_band: str,
    prefilter_score_direction: str,
    ce_sts_score: float,
    ce_sts_logit: float,
    ce_sts_rank: int,
    ce_sts_rank_percentile: float,
) -> CandidatePair:
    pair_id = _stable_id(
        pair_type,
        target_item_id,
        candidate_item_id,
        prefix="directed_pair",
    )
    return CandidatePair(
        pair_id=pair_id,
        source_pair_id=source_pair_id,
        pair_type=pair_type,
        direction=direction,
        target_item_id=target_item_id,
        target_owner_id=target_owner_id,
        target_keyword_index=target_keyword_index,
        target_text=target_text,
        candidate_item_id=candidate_item_id,
        candidate_owner_id=candidate_owner_id,
        candidate_keyword_index=candidate_keyword_index,
        candidate_text=candidate_text,
        prefilter_band=prefilter_band,
        prefilter_score_direction=prefilter_score_direction,
        ce_sts_score=ce_sts_score,
        ce_sts_logit=ce_sts_logit,
        ce_sts_rank=ce_sts_rank,
        ce_sts_rank_percentile=ce_sts_rank_percentile,
    )


def _load_grant_faculty_pairs(
    path: Path,
    *,
    bands: Sequence[str],
    per_query_counts: Mapping[str, int],
    seed: int,
) -> list[CandidatePair]:
    if not path.exists():
        raise FileNotFoundError(f"Prefilter output not found: {path}")

    allowed_bands = set(bands)
    output: list[CandidatePair] = []
    seen_pair_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, Mapping):
                raise RuntimeError(f"Expected an object at {path}:{line_number}")
            if row.get("schema_version") != GRANT_FACULTY_PREFILTER_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported prefilter schema at {path}:{line_number}: "
                    f"{row.get('schema_version')}"
                )

            grant_item_id = _clean_text(row.get("grant_item_id"))
            grant_text = _clean_text(row.get("grant_text"))
            relevance_sets = row.get("relevance_sets")
            if not grant_item_id or not grant_text or not isinstance(relevance_sets, Mapping):
                raise RuntimeError(
                    f"Incomplete prefilter grant record at {path}:{line_number}"
                )

            for band in BANDS:
                if band not in allowed_bands:
                    continue
                candidates = relevance_sets.get(band)
                if not isinstance(candidates, list):
                    continue
                requested_count = per_query_counts.get(band, 0)
                if requested_count <= 0:
                    continue

                valid_candidates = [
                    candidate
                    for candidate in candidates
                    if isinstance(candidate, Mapping)
                    and _clean_text(candidate.get("faculty_item_id"))
                    and _clean_text(candidate.get("faculty_text"))
                ]
                if band == "high":
                    # Mine the strongest CE-prefiltered candidates first.  The
                    # teacher still receives no CE score or band information.
                    valid_candidates.sort(
                        key=lambda candidate: (
                            -float(candidate.get("ce_sts_score", 0.0)),
                            int(candidate.get("ce_sts_rank", -1)),
                            _clean_text(candidate.get("faculty_item_id")),
                        )
                    )
                else:
                    # Mid/low candidates were already sampled by the prefilter.
                    # Hash ordering avoids repeatedly choosing just a band edge
                    # while remaining reproducible across machines and resumes.
                    valid_candidates.sort(
                        key=lambda candidate: hashlib.sha256(
                            (
                                f"{seed}\x1f{grant_item_id}\x1f{band}\x1f"
                                f"{_clean_text(candidate.get('faculty_item_id'))}"
                            ).encode("utf-8")
                        ).digest()
                    )

                for candidate in valid_candidates[:requested_count]:
                    if not isinstance(candidate, Mapping):
                        continue
                    faculty_item_id = _clean_text(candidate.get("faculty_item_id"))
                    faculty_text = _clean_text(candidate.get("faculty_text"))
                    if not faculty_item_id or not faculty_text:
                        continue
                    source_pair_id = _stable_id(
                        grant_item_id,
                        faculty_item_id,
                        prefix="pair",
                    )
                    pair_id = _stable_id(
                        "grant_faculty",
                        grant_item_id,
                        faculty_item_id,
                        prefix="directed_pair",
                    )
                    if pair_id in seen_pair_ids:
                        continue
                    seen_pair_ids.add(pair_id)
                    output.append(
                        _directed_pair(
                            source_pair_id=source_pair_id,
                            pair_type="grant_faculty",
                            direction="grant_to_faculty",
                            target_item_id=grant_item_id,
                            target_owner_id=row.get("grant_id", ""),
                            target_keyword_index=int(
                                row.get("grant_keyword_index", -1) or 0
                            ),
                            target_text=grant_text,
                            candidate_item_id=faculty_item_id,
                            candidate_owner_id=candidate.get("faculty_id", ""),
                            candidate_keyword_index=int(
                                candidate.get("faculty_keyword_index", -1) or 0
                            ),
                            candidate_text=faculty_text,
                            prefilter_band=band,
                            prefilter_score_direction="grant_to_faculty",
                            ce_sts_score=float(candidate.get("ce_sts_score", 0.0)),
                            ce_sts_logit=float(candidate.get("ce_sts_logit", 0.0)),
                            ce_sts_rank=int(candidate.get("ce_sts_rank", -1)),
                            ce_sts_rank_percentile=float(
                                candidate.get("ce_sts_rank_percentile", 0.0)
                            ),
                        )
                    )
    return output


def _load_same_side_pairs(
    path: Path,
    *,
    pair_type: str,
    schema_version: str,
    owner_id_field: str,
    bands: Sequence[str],
    both_directions: bool,
) -> list[CandidatePair]:
    if not path.exists():
        raise FileNotFoundError(f"Prefilter output not found: {path}")
    allowed_bands = set(bands)
    output: list[CandidatePair] = []
    seen_pair_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, Mapping):
                raise RuntimeError(f"Expected an object at {path}:{line_number}")
            if row.get("schema_version") != schema_version:
                raise RuntimeError(
                    f"Unsupported prefilter schema at {path}:{line_number}: "
                    f"{row.get('schema_version')}"
                )

            source_pair_id = _clean_text(row.get("pair_id"))
            owner_id = row.get(owner_id_field, "")
            left_item_id = _clean_text(row.get("left_item_id"))
            left_text = _clean_text(row.get("left_text"))
            right_item_id = _clean_text(row.get("right_item_id"))
            right_text = _clean_text(row.get("right_text"))
            band = _clean_text(row.get("prefilter_band")).lower()
            if (
                not source_pair_id
                or not left_item_id
                or not left_text
                or not right_item_id
                or not right_text
            ):
                raise RuntimeError(
                    f"Incomplete {pair_type} prefilter record at {path}:{line_number}"
                )
            if band not in allowed_bands:
                continue

            common = {
                "source_pair_id": source_pair_id,
                "pair_type": pair_type,
                "target_owner_id": owner_id,
                "candidate_owner_id": owner_id,
                "prefilter_band": band,
                # build_prefilter.py scored only the stored left-to-right order.
                # Reverse teacher judgments retain that discovery score solely
                # as provenance; it is never shown to the teacher.
                "prefilter_score_direction": "left_to_right",
                "ce_sts_score": float(row.get("ce_sts_score", 0.0)),
                "ce_sts_logit": float(row.get("ce_sts_logit", 0.0)),
                "ce_sts_rank": int(row.get("ce_sts_global_rank", -1)),
                "ce_sts_rank_percentile": float(
                    row.get("ce_sts_rank_percentile", 0.0)
                ),
            }
            directions = [
                _directed_pair(
                    **common,
                    direction="left_to_right",
                    target_item_id=left_item_id,
                    target_keyword_index=int(row.get("left_keyword_index", -1)),
                    target_text=left_text,
                    candidate_item_id=right_item_id,
                    candidate_keyword_index=int(row.get("right_keyword_index", -1)),
                    candidate_text=right_text,
                )
            ]
            if both_directions:
                directions.append(
                    _directed_pair(
                        **common,
                        direction="right_to_left",
                        target_item_id=right_item_id,
                        target_keyword_index=int(row.get("right_keyword_index", -1)),
                        target_text=right_text,
                        candidate_item_id=left_item_id,
                        candidate_keyword_index=int(row.get("left_keyword_index", -1)),
                        candidate_text=left_text,
                    )
                )
            for candidate in directions:
                if candidate.pair_id in seen_pair_ids:
                    continue
                seen_pair_ids.add(candidate.pair_id)
                output.append(candidate)
    return output


def _candidate_counts(candidates: Sequence[CandidatePair]) -> dict[str, int]:
    return dict(sorted(Counter(candidate.pair_type for candidate in candidates).items()))


def _extract_json_object(raw_text: str) -> Optional[dict[str, Any]]:
    text = raw_text.strip()
    if not text:
        return None
    text = re.sub(r"(?is)<think>[\s\S]*?</think>", "", text).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    candidates = [text]
    candidates.extend(
        match.strip()
        for match in re.findall(
            r"```(?:json)?\s*([\s\S]*?)```",
            text,
            flags=re.IGNORECASE,
        )
    )
    depth = 0
    start: Optional[int] = None
    for index, character in enumerate(text):
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : index + 1])
                start = None

    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_judgment(raw_text: str) -> tuple[Optional[ParsedJudgment], str]:
    payload = _extract_json_object(raw_text)
    if payload is None:
        return None, "no_valid_json_object"
    try:
        score = float(payload["score"])
        confidence = float(payload["confidence"])
    except (KeyError, TypeError, ValueError):
        return None, "missing_or_non_numeric_score_or_confidence"
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None, "score_out_of_range"
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None, "confidence_out_of_range"
    rationale = _clean_text(payload.get("rationale") or payload.get("reason"))
    if not rationale:
        return None, "missing_rationale"
    return ParsedJudgment(score, confidence, rationale), ""


def _apply_chat_template(
    tokenizer: Any,
    *,
    candidate: CandidatePair,
    enable_thinking: bool,
    invalid_response: str = "",
) -> str:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        target_text=candidate.target_text,
        candidate_text=candidate.candidate_text,
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if invalid_response:
        messages.extend(
            (
                {"role": "assistant", "content": invalid_response},
                {"role": "user", "content": REPAIR_PROMPT},
            )
        )

    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("The selected tokenizer does not support chat templates")
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _load_completed_judgment_ids(
    path: Path,
    *,
    generation_fingerprint: str,
) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid existing judgment JSONL at {path}:{line_number}"
                ) from exc
            if row.get("schema_version") != JUDGMENT_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Existing output at {path}:{line_number} uses schema "
                    f"{row.get('schema_version')!r}; use --overwrite or choose a "
                    "different --output for the v2 source-neutral judgments."
                )
            if row.get("generation_fingerprint") != generation_fingerprint:
                raise RuntimeError(
                    f"Existing output at {path}:{line_number} was generated with "
                    "different teacher settings; use --overwrite or choose a "
                    "different --output."
                )
            judgment_id = _clean_text(row.get("judgment_id"))
            if judgment_id:
                completed.add(judgment_id)
    return completed


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _generated_texts(
    llm: Any,
    prompts: Sequence[str],
    sampling_params: Any,
    *,
    use_tqdm: bool,
) -> list[str]:
    outputs = llm.generate(
        list(prompts),
        sampling_params,
        use_tqdm=use_tqdm,
    )
    texts: list[str] = []
    for output in outputs:
        choices = getattr(output, "outputs", None)
        if not choices:
            texts.append("")
            continue
        texts.append(_clean_text(getattr(choices[0], "text", "")))
    return texts


def _try_progress(total: int, *, disabled: bool) -> Any:
    if disabled:
        return None
    try:
        from tqdm.auto import tqdm

        return tqdm(
            total=total,
            desc="CE5 LLM distillation",
            unit="pair",
            dynamic_ncols=True,
        )
    except Exception:
        return None


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


def _positive_unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in (0,1]")
    return parsed


def _interleave_pair_types(
    candidates_by_type: Mapping[str, Sequence[CandidatePair]],
) -> list[CandidatePair]:
    """Combine pair types without placing one entire source before the others."""
    positions = {pair_type: 0 for pair_type in PAIR_TYPES}
    output: list[CandidatePair] = []
    while True:
        added = False
        for pair_type in PAIR_TYPES:
            candidates = candidates_by_type.get(pair_type, ())
            position = positions[pair_type]
            if position >= len(candidates):
                continue
            output.append(candidates[position])
            positions[pair_type] += 1
            added = True
        if not added:
            return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score CE5 prefilter candidates with a local vLLM teacher."
    )
    parser.add_argument(
        "--grant-faculty-prefilter",
        "--prefilter",
        dest="grant_faculty_prefilter",
        type=Path,
        default=DEFAULT_GRANT_FACULTY_PREFILTER,
        help=(
            "Grant-faculty prefilter JSONL. --prefilter remains as a backward-"
            "compatible alias."
        ),
    )
    parser.add_argument(
        "--grant-grant-prefilter",
        type=Path,
        default=DEFAULT_GRANT_GRANT_PREFILTER,
    )
    parser.add_argument(
        "--faculty-faculty-prefilter",
        type=Path,
        default=DEFAULT_FACULTY_FACULTY_PREFILTER,
    )
    parser.add_argument(
        "--pair-types",
        type=_parse_pair_types,
        default=PAIR_TYPES,
        help=(
            "Comma-separated inputs to distill (default: "
            "grant_faculty,grant_grant,faculty_faculty)."
        ),
    )
    parser.add_argument(
        "--same-side-directions",
        choices=("both", "forward"),
        default="both",
        help=(
            "Distill both directed orders of G-G/F-F pairs, or only the "
            "prefilter's left-to-right order (default: both)."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--model-id",
        default=DEFAULT_TEACHER_MODEL,
        help=(
            "Hugging Face model ID or local teacher path "
            f"(default: {DEFAULT_TEACHER_MODEL})."
        ),
    )
    parser.add_argument("--bands", type=_parse_bands, default=BANDS)
    parser.add_argument(
        "--high-per-query",
        type=_nonnegative_int,
        default=8,
        help="Maximum CE-high candidates selected per grant keyword (default: 8).",
    )
    parser.add_argument(
        "--mid-per-query",
        type=_nonnegative_int,
        default=2,
        help="Maximum CE-mid candidates selected per grant keyword (default: 2).",
    )
    parser.add_argument(
        "--low-per-query",
        type=_nonnegative_int,
        default=1,
        help="Maximum CE-low candidates selected per grant keyword (default: 1).",
    )
    parser.add_argument(
        "--max-pairs",
        type=_nonnegative_int,
        default=0,
        help=(
            "Optional final cap after loading and interleaving all selected pair "
            "types; 0 scores every selected pair (default: 0)."
        ),
    )
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=128)
    parser.add_argument("--temperature", type=_unit_float, default=0.0)
    parser.add_argument("--top-p", type=_positive_unit_float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=_positive_int, default=1)
    parser.add_argument("--max-model-len", type=_positive_int, default=2048)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=_positive_unit_float,
        default=0.90,
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--download-dir", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--vllm-tqdm", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing judgment and error files instead of resuming.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print example prompts without loading an LLM.",
    )
    parser.add_argument("--dry-run-limit", type=_positive_int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    pair_types = tuple(args.pair_types)
    prefilter_paths = {
        "grant_faculty": _resolve_path(args.grant_faculty_prefilter),
        "grant_grant": _resolve_path(args.grant_grant_prefilter),
        "faculty_faculty": _resolve_path(args.faculty_faculty_prefilter),
    }
    output_path = _resolve_path(args.output)
    error_path = _resolve_path(args.errors)
    manifest_path = _resolve_path(args.manifest)
    bands = tuple(args.bands)
    per_query_counts = {
        "high": args.high_per_query,
        "mid": args.mid_per_query,
        "low": args.low_per_query,
    }
    if (
        "grant_faculty" in pair_types
        and not any(per_query_counts[band] > 0 for band in bands)
    ):
        raise ValueError(
            "At least one selected band must have a positive per-query count"
        )

    candidates_by_type: dict[str, list[CandidatePair]] = {}
    if "grant_faculty" in pair_types:
        candidates_by_type["grant_faculty"] = _load_grant_faculty_pairs(
            prefilter_paths["grant_faculty"],
            bands=bands,
            per_query_counts=per_query_counts,
            seed=args.seed,
        )
    if "grant_grant" in pair_types:
        candidates_by_type["grant_grant"] = _load_same_side_pairs(
            prefilter_paths["grant_grant"],
            pair_type="grant_grant",
            schema_version=GRANT_GRANT_PREFILTER_SCHEMA_VERSION,
            owner_id_field="grant_id",
            bands=bands,
            both_directions=args.same_side_directions == "both",
        )
    if "faculty_faculty" in pair_types:
        candidates_by_type["faculty_faculty"] = _load_same_side_pairs(
            prefilter_paths["faculty_faculty"],
            pair_type="faculty_faculty",
            schema_version=FACULTY_FACULTY_PREFILTER_SCHEMA_VERSION,
            owner_id_field="faculty_id",
            bands=bands,
            both_directions=args.same_side_directions == "both",
        )
    loaded_candidate_counts = {
        pair_type: len(values) for pair_type, values in candidates_by_type.items()
    }
    candidates = _interleave_pair_types(candidates_by_type)
    if args.max_pairs > 0:
        candidates = candidates[: args.max_pairs]
    if not candidates:
        raise RuntimeError("No candidate pairs were loaded from the selected bands")

    model_id = _clean_text(args.model_id)
    if not model_id:
        raise ValueError("--model-id cannot be empty")
    generation_identity = {
        "teacher_model": model_id,
        "prompt_version": PROMPT_VERSION,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "max_new_tokens": args.max_new_tokens,
    }
    generation_fingerprint = hashlib.sha256(
        json.dumps(generation_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()

    judgment_ids = {
        candidate.pair_id: _stable_id(
            candidate.pair_id,
            generation_fingerprint,
            prefix="judgment",
        )
        for candidate in candidates
    }

    if args.dry_run:
        for pair_type in pair_types:
            print(f"{pair_type}_prefilter={prefilter_paths[pair_type]}")
        print(f"candidate_pairs={len(candidates)}")
        print(f"candidate_counts={_candidate_counts(candidates)}")
        print(f"bands={','.join(bands)}")
        print(
            "per_query_counts="
            + ",".join(f"{band}:{per_query_counts[band]}" for band in BANDS)
        )
        print(f"model_id={model_id}")
        examples_printed = 0
        for pair_type in pair_types:
            type_candidates = [
                candidate
                for candidate in candidates
                if candidate.pair_type == pair_type
            ]
            for candidate in type_candidates[: args.dry_run_limit]:
                examples_printed += 1
                print(
                    f"--- pair {examples_printed}: {candidate.pair_id} "
                    f"(stored type={candidate.pair_type}, "
                    f"direction={candidate.direction}) ---"
                )
                print(
                    USER_PROMPT_TEMPLATE.format(
                        target_text=candidate.target_text,
                        candidate_text=candidate.candidate_text,
                    )
                )
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output_mode = "w"
        error_mode = "w"
        completed_ids: set[str] = set()
    else:
        output_mode = "a"
        error_mode = "a"
        completed_ids = _load_completed_judgment_ids(
            output_path,
            generation_fingerprint=generation_fingerprint,
        )
    pending = [
        candidate
        for candidate in candidates
        if judgment_ids[candidate.pair_id] not in completed_ids
    ]
    if not pending:
        print(f"output={output_path}")
        print("All selected candidate pairs already have this teacher judgment.")
        return 0

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is required on the HPC runtime. Install it before running "
            "CE5 local-LLM distillation."
        ) from exc

    llm_kwargs: dict[str, Any] = {
        "model": model_id,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
    }
    if _clean_text(args.download_dir):
        llm_kwargs["download_dir"] = args.download_dir
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    sampling_kwargs: dict[str, Any] = {
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    try:
        if "seed" in inspect.signature(SamplingParams).parameters:
            sampling_kwargs["seed"] = args.seed
    except (TypeError, ValueError):
        pass
    sampling_params = SamplingParams(**sampling_kwargs)

    progress = _try_progress(len(pending), disabled=args.no_progress)
    accepted_this_run = 0
    invalid_this_run = 0
    retried_this_run = 0
    run_id = f"llm-distill-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    with output_path.open(output_mode, encoding="utf-8") as output_handle, error_path.open(
        error_mode, encoding="utf-8"
    ) as error_handle:
        for batch in _chunks(pending, args.batch_size):
            prompts = [
                _apply_chat_template(
                    tokenizer,
                    candidate=candidate,
                    enable_thinking=args.enable_thinking,
                )
                for candidate in batch
            ]
            responses = _generated_texts(
                llm,
                prompts,
                sampling_params,
                use_tqdm=args.vllm_tqdm,
            )
            parsed_results = [_parse_judgment(response) for response in responses]

            invalid_indices = [
                index
                for index, (parsed, _) in enumerate(parsed_results)
                if parsed is None
            ]
            if invalid_indices:
                retried_this_run += len(invalid_indices)
                retry_prompts = [
                    _apply_chat_template(
                        tokenizer,
                        candidate=batch[index],
                        enable_thinking=False,
                        invalid_response=responses[index],
                    )
                    for index in invalid_indices
                ]
                retry_responses = _generated_texts(
                    llm,
                    retry_prompts,
                    sampling_params,
                    use_tqdm=args.vllm_tqdm,
                )
                for index, retry_response in zip(
                    invalid_indices,
                    retry_responses,
                    strict=True,
                ):
                    responses[index] = retry_response
                    parsed_results[index] = _parse_judgment(retry_response)

            created_at = _utc_now()
            for candidate, response, (parsed, parse_error) in zip(
                batch,
                responses,
                parsed_results,
                strict=True,
            ):
                judgment_id = judgment_ids[candidate.pair_id]
                if parsed is None:
                    error_handle.write(
                        json.dumps(
                            {
                                "schema_version": ERROR_SCHEMA_VERSION,
                                "judgment_id": judgment_id,
                                "pair_id": candidate.pair_id,
                                "source_pair_id": candidate.source_pair_id,
                                "pair_type": candidate.pair_type,
                                "direction": candidate.direction,
                                "teacher_model": model_id,
                                "prompt_version": PROMPT_VERSION,
                                "parse_error": parse_error,
                                "raw_response": response,
                                "run_id": run_id,
                                "created_at": created_at,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    invalid_this_run += 1
                    continue

                record = {
                    "schema_version": JUDGMENT_SCHEMA_VERSION,
                    "judgment_id": judgment_id,
                    **asdict(candidate),
                    "teacher_model": model_id,
                    "prompt_version": PROMPT_VERSION,
                    "generation_fingerprint": generation_fingerprint,
                    "score": parsed.score,
                    "confidence": parsed.confidence,
                    "rationale": parsed.rationale,
                    "raw_response": response,
                    "temperature": args.temperature,
                    "run_id": run_id,
                    "created_at": created_at,
                }
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted_this_run += 1

            output_handle.flush()
            error_handle.flush()
            if progress is not None:
                progress.update(len(batch))

    if progress is not None:
        progress.close()

    elapsed = time.time() - started
    selected_prefilters = {
        pair_type: {
            "path": str(prefilter_paths[pair_type]),
            "sha256": _file_sha256(prefilter_paths[pair_type]),
        }
        for pair_type in pair_types
    }
    manifest = {
        "schema_version": 2,
        "created_at_utc": _utc_now(),
        "run_id": run_id,
        "prefilters": selected_prefilters,
        "output": str(output_path),
        "errors": str(error_path),
        "teacher_model": model_id,
        "prompt_version": PROMPT_VERSION,
        "generation_fingerprint": generation_fingerprint,
        "configuration": {
            **generation_identity,
            "pair_types": list(pair_types),
            "same_side_directions": args.same_side_directions,
            "bands": list(bands),
            "grant_faculty_per_query_counts": per_query_counts,
            "max_pairs": args.max_pairs,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": args.dtype,
        },
        "candidate_pairs_loaded": len(candidates),
        "candidate_pairs_loaded_before_global_cap": sum(
            loaded_candidate_counts.values()
        ),
        "candidate_counts_before_global_cap": loaded_candidate_counts,
        "candidate_counts_selected": _candidate_counts(candidates),
        "already_completed": len(candidates) - len(pending),
        "attempted_this_run": len(pending),
        "accepted_this_run": accepted_this_run,
        "invalid_after_retry_this_run": invalid_this_run,
        "retried_this_run": retried_this_run,
        "elapsed_seconds_this_run": elapsed,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"output={output_path}")
    print(f"errors={error_path}")
    print(f"manifest={manifest_path}")
    print(f"accepted_this_run={accepted_this_run}")
    print(f"invalid_after_retry_this_run={invalid_this_run}")
    print(f"elapsed_seconds_this_run={elapsed:.2f}")

    del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return 0 if invalid_this_run == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
