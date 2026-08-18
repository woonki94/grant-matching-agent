"""Build structured grant-faculty teacher judgments from cached source claims.

This is the second offline CE5 teacher stage. It joins the grant requirement
and faculty capability decompositions produced by ``build_keyword_claims.py``
to the existing grant-faculty candidates from ``build_prefilter.py``. The LLM
then judges every grant requirement separately and identifies the faculty
claims that support it, in addition to producing one holistic score.

Only grant-faculty pairs are accepted. CE-STS scores, ranks, and prefilter
bands are retained as sampling/audit metadata but never included in the
teacher prompt. Output is append-only and resumable. Invalid responses are
retried once and then written to a separate error file.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
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

BASE_DISTILLATION_PATH = Path(__file__).with_name("llm_distillation.py")
BASE_DISTILLATION_SPEC = importlib.util.spec_from_file_location(
    "ce5_llm_distillation_for_structured_data",
    BASE_DISTILLATION_PATH,
)
if BASE_DISTILLATION_SPEC is None or BASE_DISTILLATION_SPEC.loader is None:
    raise RuntimeError(
        f"Unable to load grant-faculty candidate utilities from "
        f"{BASE_DISTILLATION_PATH}"
    )
base_distillation = importlib.util.module_from_spec(BASE_DISTILLATION_SPEC)
sys.modules[BASE_DISTILLATION_SPEC.name] = base_distillation
BASE_DISTILLATION_SPEC.loader.exec_module(base_distillation)


DATASET_DIR = REPO_ROOT / "ce5" / "dataset"
SOURCE_DIR = DATASET_DIR / "source"
JUDGMENT_DIR = DATASET_DIR / "judgments"
DEFAULT_PREFILTER = SOURCE_DIR / "prefilter_candidates.jsonl"
DEFAULT_GRANT_CLAIMS = SOURCE_DIR / "grant_requirement_claims.jsonl"
DEFAULT_FACULTY_CLAIMS = SOURCE_DIR / "faculty_capability_claims.jsonl"
DEFAULT_OUTPUT = JUDGMENT_DIR / "structured_teacher_judgments_v1.jsonl"
DEFAULT_ERRORS = JUDGMENT_DIR / "structured_teacher_judgment_errors_v1.jsonl"
DEFAULT_MANIFEST = JUDGMENT_DIR / "structured_teacher_judgments_v1.manifest.json"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3-14B"

SOURCE_CLAIM_SCHEMA_VERSION = "ce5.source-claims.v1"
JUDGMENT_SCHEMA_VERSION = "ce5.structured-judgment.v1"
ERROR_SCHEMA_VERSION = "ce5.structured-judgment-error.v1"
MANIFEST_SCHEMA_VERSION = "ce5.structured-judgment-manifest.v1"
PROMPT_VERSION = "grant-faculty-structured-coverage-v1"
BANDS = ("high", "mid", "low")


SYSTEM_PROMPT = """
You are a strict evaluator of whether a faculty capability satisfies the
requirements expressed by a grant specialization keyword.

You receive the original GRANT KEYWORD and FACULTY KEYWORD plus independently
extracted requirement and capability claims. Treat the decompositions as
grounded views of the original texts. Judge only evidence present in those
texts. Do not infer credentials, methods, experience, applications, or domain
knowledge that the faculty text does not state.

This is directed capability coverage, not ordinary semantic similarity. Shared
topics or goals are insufficient unless the faculty claims provide a relevant
method, expertise, experience, system, application, or transferable ability.

For every grant requirement claim:
- Return exactly one requirement_coverage entry using its supplied ID.
- Give a genuinely continuous coverage_score from 0.00 to 1.00.
- List only supplied faculty capability IDs that materially support it.
- Use an empty supporting_capability_claim_ids list when none support it.
- Confidence measures confidence in the coverage judgment, not match strength.

Coverage score guidance:
- 0.00: no evidenced coverage
- 0.01-0.24: weak topical or highly indirect connection
- 0.25-0.49: limited coverage with major missing capabilities
- 0.50-0.74: meaningful partial coverage with important gaps
- 0.75-0.94: strong coverage with a smaller but real gap
- 0.95-1.00: direct and essentially complete coverage

Then return one overall_score for the complete grant-faculty pair. Judge the
faculty's collective coverage of the complete grant requirement, accounting
for both covered and materially missing requirements. The overall score need
not be a mechanical mean of requirement scores. Use the same continuous scale.

Return exactly one JSON object:
{
  "requirement_coverage": [
    {
      "requirement_claim_id": "<one supplied grant requirement ID>",
      "coverage_score": <number from 0.00 to 1.00>,
      "supporting_capability_claim_ids": ["<supplied faculty claim ID>"],
      "confidence": <number from 0.00 to 1.00>
    }
  ],
  "overall_score": <number from 0.00 to 1.00>,
  "overall_confidence": <number from 0.00 to 1.00>,
  "rationale": "<one concise sentence describing the decisive coverage and gap>"
}

Do not output markdown, hidden analysis, selection metadata, a categorical
band, invented IDs, or any text outside the JSON object.
""".strip()


REPAIR_PROMPT = """
Your previous response was invalid. Return only one JSON object matching the
required schema. Include exactly one entry for every supplied grant requirement
ID and no others. Supporting IDs must come only from the supplied faculty
capability IDs. All scores and confidence values must be finite numbers in
[0,1]. Include one nonempty concise rationale and no markdown.
""".strip()


@dataclass(frozen=True)
class SourceClaim:
    claim_id: str
    claim: str
    source_span: str
    source_span_start: int
    source_span_end: int


@dataclass(frozen=True)
class SourceClaimSet:
    item_type: str
    item_id: str
    owner_id: str | int
    keyword_index: int
    source_text: str
    source_text_sha256: str
    record_id: str
    generation_fingerprint: str
    claims: tuple[SourceClaim, ...]


@dataclass(frozen=True)
class StructuredPair:
    candidate: base_distillation.CandidatePair
    grant: SourceClaimSet
    faculty: SourceClaimSet


@dataclass(frozen=True)
class RequirementCoverage:
    requirement_claim_id: str
    coverage_score: float
    supporting_capability_claim_ids: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class ParsedStructuredJudgment:
    requirement_coverage: tuple[RequirementCoverage, ...]
    overall_score: float
    overall_confidence: float
    rationale: str


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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_id(*parts: Any, prefix: str) -> str:
    raw = "\x1f".join(_clean_text(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:24]}"


def _finite_unit_float(value: Any, *, field: str) -> tuple[Optional[float], str]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, f"{field}_is_not_numeric"
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        return None, f"{field}_out_of_range"
    return parsed, ""


def _load_claim_cache(
    path: Path,
    *,
    expected_item_type: str,
    expected_prefix: str,
) -> dict[str, SourceClaimSet]:
    if not path.exists():
        raise FileNotFoundError(f"Source claim cache not found: {path}")
    output: dict[str, SourceClaimSet] = {}
    generation_fingerprints: set[str] = set()
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
            if row.get("schema_version") != SOURCE_CLAIM_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported claim schema at {path}:{line_number}: "
                    f"{row.get('schema_version')!r}"
                )
            if row.get("item_type") != expected_item_type:
                raise RuntimeError(
                    f"Unexpected item_type at {path}:{line_number}: "
                    f"{row.get('item_type')!r}"
                )

            item_id = _clean_text(row.get("item_id"))
            source_text = _clean_text(row.get("source_text"))
            source_hash = _clean_text(row.get("source_text_sha256"))
            record_id = _clean_text(row.get("record_id"))
            generation_fingerprint = _clean_text(
                row.get("generation_fingerprint")
            )
            if not item_id.startswith(f"{expected_prefix}:"):
                raise RuntimeError(
                    f"Invalid item_id at {path}:{line_number}: {item_id!r}"
                )
            if item_id in output:
                raise RuntimeError(f"Duplicate item_id at {path}:{line_number}: {item_id}")
            if not all(
                (source_text, source_hash, record_id, generation_fingerprint)
            ):
                raise RuntimeError(
                    f"Incomplete claim record at {path}:{line_number}"
                )
            if source_hash != _sha256_text(source_text):
                raise RuntimeError(
                    f"Incorrect source text hash at {path}:{line_number}"
                )

            raw_claims = row.get("claims")
            if not isinstance(raw_claims, list) or not raw_claims:
                raise RuntimeError(
                    f"Missing claims at {path}:{line_number}"
                )
            parsed_claims: list[SourceClaim] = []
            seen_claim_ids: set[str] = set()
            for claim_index, raw_claim in enumerate(raw_claims):
                if not isinstance(raw_claim, Mapping):
                    raise RuntimeError(
                        f"Invalid claim {claim_index} at {path}:{line_number}"
                    )
                claim_id = _clean_text(raw_claim.get("claim_id"))
                claim = _clean_text(raw_claim.get("claim"))
                source_span = _clean_text(raw_claim.get("source_span"))
                try:
                    start = int(raw_claim.get("source_span_start"))
                    end = int(raw_claim.get("source_span_end"))
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Invalid source span offsets for claim {claim_index} at "
                        f"{path}:{line_number}"
                    ) from exc
                if not claim_id or claim_id in seen_claim_ids or not claim:
                    raise RuntimeError(
                        f"Missing or duplicate claim ID at {path}:{line_number}"
                    )
                if not claim_id.startswith(f"{item_id}:"):
                    raise RuntimeError(
                        f"Claim ID {claim_id!r} does not belong to {item_id!r} "
                        f"at {path}:{line_number}"
                    )
                if start < 0 or end <= start or source_text[start:end] != source_span:
                    raise RuntimeError(
                        f"Ungrounded source span for {claim_id} at "
                        f"{path}:{line_number}"
                    )
                seen_claim_ids.add(claim_id)
                parsed_claims.append(
                    SourceClaim(
                        claim_id=claim_id,
                        claim=claim,
                        source_span=source_span,
                        source_span_start=start,
                        source_span_end=end,
                    )
                )

            generation_fingerprints.add(generation_fingerprint)
            output[item_id] = SourceClaimSet(
                item_type=expected_item_type,
                item_id=item_id,
                owner_id=row.get("owner_id", ""),
                keyword_index=int(row.get("keyword_index", -1)),
                source_text=source_text,
                source_text_sha256=source_hash,
                record_id=record_id,
                generation_fingerprint=generation_fingerprint,
                claims=tuple(parsed_claims),
            )
    if not output:
        raise RuntimeError(f"No source claims were loaded from {path}")
    if len(generation_fingerprints) != 1:
        raise RuntimeError(
            f"Claim cache {path} mixes generation fingerprints; rebuild it "
            "with one decomposition configuration"
        )
    return output


def _join_candidates_to_claims(
    candidates: Sequence[base_distillation.CandidatePair],
    *,
    grant_claims: Mapping[str, SourceClaimSet],
    faculty_claims: Mapping[str, SourceClaimSet],
) -> list[StructuredPair]:
    output: list[StructuredPair] = []
    missing_grants: set[str] = set()
    missing_faculty: set[str] = set()
    text_mismatches: list[str] = []
    for candidate in candidates:
        if candidate.pair_type != "grant_faculty":
            raise RuntimeError(
                "Structured distillation accepts only grant_faculty candidates"
            )
        grant = grant_claims.get(candidate.target_item_id)
        faculty = faculty_claims.get(candidate.candidate_item_id)
        if grant is None:
            missing_grants.add(candidate.target_item_id)
            continue
        if faculty is None:
            missing_faculty.add(candidate.candidate_item_id)
            continue
        if _clean_text(candidate.target_text) != grant.source_text:
            text_mismatches.append(candidate.target_item_id)
            continue
        if _clean_text(candidate.candidate_text) != faculty.source_text:
            text_mismatches.append(candidate.candidate_item_id)
            continue
        output.append(StructuredPair(candidate, grant, faculty))

    if missing_grants or missing_faculty or text_mismatches:
        details = []
        if missing_grants:
            details.append(
                f"missing grant decompositions={len(missing_grants)} "
                f"sample={sorted(missing_grants)[:3]}"
            )
        if missing_faculty:
            details.append(
                f"missing faculty decompositions={len(missing_faculty)} "
                f"sample={sorted(missing_faculty)[:3]}"
            )
        if text_mismatches:
            details.append(
                f"prefilter/claim text mismatches={len(text_mismatches)} "
                f"sample={sorted(set(text_mismatches))[:3]}"
            )
        raise RuntimeError(
            "Cannot build a complete structured dataset: " + "; ".join(details)
        )
    return output


def _prompt_claims(claims: Sequence[SourceClaim]) -> list[dict[str, str]]:
    return [
        {
            "id": claim.claim_id,
            "claim": claim.claim,
            "source_span": claim.source_span,
        }
        for claim in claims
    ]


def _user_prompt(pair: StructuredPair) -> str:
    prompt_payload = {
        "grant_keyword": pair.grant.source_text,
        "grant_requirements": _prompt_claims(pair.grant.claims),
        "faculty_keyword": pair.faculty.source_text,
        "faculty_capabilities": _prompt_claims(pair.faculty.claims),
    }
    return (
        "Evaluate this grant-faculty pair.\n\nINPUT:\n"
        + json.dumps(prompt_payload, ensure_ascii=False, indent=2)
    )


def _apply_chat_template(
    tokenizer: Any,
    *,
    pair: StructuredPair,
    enable_thinking: bool,
    invalid_response: str = "",
) -> str:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(pair)},
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


def _extract_json_object(raw_text: str) -> Optional[dict[str, Any]]:
    text = str(raw_text or "").strip()
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


def _parse_structured_judgment(
    raw_text: str,
    *,
    requirement_claim_ids: Sequence[str],
    capability_claim_ids: Sequence[str],
) -> tuple[Optional[ParsedStructuredJudgment], str]:
    payload = _extract_json_object(raw_text)
    if payload is None:
        return None, "no_valid_json_object"

    expected_requirements = tuple(requirement_claim_ids)
    expected_requirement_set = set(expected_requirements)
    allowed_capabilities = set(capability_claim_ids)
    raw_coverages = payload.get("requirement_coverage")
    if not isinstance(raw_coverages, list):
        return None, "missing_requirement_coverage"
    if len(raw_coverages) != len(expected_requirements):
        return None, "requirement_coverage_count_mismatch"

    parsed_by_id: dict[str, RequirementCoverage] = {}
    for index, raw_coverage in enumerate(raw_coverages):
        if not isinstance(raw_coverage, Mapping):
            return None, f"requirement_coverage_{index}_is_not_an_object"
        requirement_id = _clean_text(raw_coverage.get("requirement_claim_id"))
        if requirement_id not in expected_requirement_set:
            return None, f"requirement_coverage_{index}_has_unknown_id"
        if requirement_id in parsed_by_id:
            return None, f"requirement_coverage_{index}_duplicates_id"

        coverage_score, error = _finite_unit_float(
            raw_coverage.get("coverage_score"),
            field=f"requirement_coverage_{index}_coverage_score",
        )
        if error:
            return None, error
        confidence, error = _finite_unit_float(
            raw_coverage.get("confidence"),
            field=f"requirement_coverage_{index}_confidence",
        )
        if error:
            return None, error

        raw_supporting = raw_coverage.get("supporting_capability_claim_ids")
        if not isinstance(raw_supporting, list):
            return None, f"requirement_coverage_{index}_missing_supporting_ids"
        supporting: list[str] = []
        seen_supporting: set[str] = set()
        for raw_id in raw_supporting:
            capability_id = _clean_text(raw_id)
            if capability_id not in allowed_capabilities:
                return None, f"requirement_coverage_{index}_has_unknown_support_id"
            if capability_id in seen_supporting:
                return None, f"requirement_coverage_{index}_duplicates_support_id"
            seen_supporting.add(capability_id)
            supporting.append(capability_id)

        assert coverage_score is not None
        assert confidence is not None
        parsed_by_id[requirement_id] = RequirementCoverage(
            requirement_claim_id=requirement_id,
            coverage_score=coverage_score,
            supporting_capability_claim_ids=tuple(supporting),
            confidence=confidence,
        )

    if set(parsed_by_id) != expected_requirement_set:
        return None, "missing_requirement_claim_id"
    overall_score, error = _finite_unit_float(
        payload.get("overall_score"),
        field="overall_score",
    )
    if error:
        return None, error
    overall_confidence, error = _finite_unit_float(
        payload.get("overall_confidence"),
        field="overall_confidence",
    )
    if error:
        return None, error
    rationale = _clean_text(payload.get("rationale"))
    if not rationale:
        return None, "missing_rationale"

    assert overall_score is not None
    assert overall_confidence is not None
    return (
        ParsedStructuredJudgment(
            requirement_coverage=tuple(
                parsed_by_id[claim_id] for claim_id in expected_requirements
            ),
            overall_score=overall_score,
            overall_confidence=overall_confidence,
            rationale=rationale,
        ),
        "",
    )


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
                    f"{row.get('schema_version')!r}; use --overwrite or a new path"
                )
            if row.get("generation_fingerprint") != generation_fingerprint:
                raise RuntimeError(
                    f"Existing output at {path}:{line_number} uses different "
                    "teacher or input settings; use --overwrite or a new path"
                )
            judgment_id = _clean_text(row.get("judgment_id"))
            if judgment_id:
                completed.add(judgment_id)
    return completed


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _deterministic_pair_cap(
    candidates: Sequence[base_distillation.CandidatePair],
    *,
    max_pairs: int,
    seed: int,
) -> list[base_distillation.CandidatePair]:
    if max_pairs <= 0 or len(candidates) <= max_pairs:
        return list(candidates)
    return sorted(
        candidates,
        key=lambda candidate: hashlib.sha256(
            f"{seed}\x1f{candidate.pair_id}".encode("utf-8")
        ).digest(),
    )[:max_pairs]


def _generated_texts(
    llm: Any,
    prompts: Sequence[str],
    sampling_params: Any,
    *,
    use_tqdm: bool,
) -> list[str]:
    outputs = llm.generate(list(prompts), sampling_params, use_tqdm=use_tqdm)
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
            desc="CE5 structured G-F distillation",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create structured per-requirement G-F teacher judgments from "
            "prefilter candidates and independent source-claim caches."
        )
    )
    parser.add_argument("--prefilter", type=Path, default=DEFAULT_PREFILTER)
    parser.add_argument(
        "--grant-claims",
        type=Path,
        default=DEFAULT_GRANT_CLAIMS,
    )
    parser.add_argument(
        "--faculty-claims",
        type=Path,
        default=DEFAULT_FACULTY_CLAIMS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-id", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--bands", type=_parse_bands, default=BANDS)
    parser.add_argument("--high-per-query", type=_nonnegative_int, default=8)
    parser.add_argument("--mid-per-query", type=_nonnegative_int, default=2)
    parser.add_argument("--low-per-query", type=_nonnegative_int, default=1)
    parser.add_argument(
        "--all-prefilter-candidates",
        action="store_true",
        help="Use every available candidate in each selected prefilter band.",
    )
    parser.add_argument(
        "--max-pairs",
        type=_nonnegative_int,
        default=0,
        help="Optional cap after candidate selection; 0 means no global cap.",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=1024)
    parser.add_argument("--temperature", type=_unit_float, default=0.0)
    parser.add_argument("--top-p", type=_positive_unit_float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=_positive_int, default=1)
    parser.add_argument("--max-model-len", type=_positive_int, default=4096)
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
        help="Replace output and error files instead of resuming.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all joins and print prompts without loading vLLM.",
    )
    parser.add_argument("--dry-run-limit", type=_positive_int, default=2)
    return parser


def _write_manifest(
    path: Path,
    *,
    started: float,
    run_id: str,
    source_paths: Mapping[str, Path],
    source_hashes: Mapping[str, str],
    generation_identity: Mapping[str, Any],
    generation_fingerprint: str,
    output_path: Path,
    error_path: Path,
    grant_claims: Mapping[str, SourceClaimSet],
    faculty_claims: Mapping[str, SourceClaimSet],
    pairs: Sequence[StructuredPair],
    pending_count: int,
    already_completed: int,
    accepted_this_run: int,
    invalid_this_run: int,
    retried_this_run: int,
) -> None:
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "run_id": run_id,
        "elapsed_seconds_this_run": time.time() - started,
        "inputs": {
            name: {"path": str(source_paths[name]), "sha256": source_hashes[name]}
            for name in ("prefilter", "grant_claims", "faculty_claims")
        },
        "output": str(output_path),
        "errors": str(error_path),
        "generation_identity": dict(generation_identity),
        "generation_fingerprint": generation_fingerprint,
        "source_claim_counts": {
            "grant_items": len(grant_claims),
            "faculty_items": len(faculty_claims),
            "grant_claims": sum(len(item.claims) for item in grant_claims.values()),
            "faculty_claims": sum(
                len(item.claims) for item in faculty_claims.values()
            ),
        },
        "selected_pairs": len(pairs),
        "selected_pairs_by_prefilter_band": dict(
            sorted(Counter(pair.candidate.prefilter_band for pair in pairs).items())
        ),
        "grant_requirements_per_selected_pair": dict(
            sorted(Counter(len(pair.grant.claims) for pair in pairs).items())
        ),
        "already_completed": already_completed,
        "attempted_this_run": pending_count,
        "accepted_this_run": accepted_this_run,
        "invalid_after_retry_this_run": invalid_this_run,
        "retried_this_run": retried_this_run,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    source_paths = {
        "prefilter": _resolve_path(args.prefilter),
        "grant_claims": _resolve_path(args.grant_claims),
        "faculty_claims": _resolve_path(args.faculty_claims),
    }
    output_path = _resolve_path(args.output)
    error_path = _resolve_path(args.errors)
    manifest_path = _resolve_path(args.manifest)
    for path in source_paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
    configured_paths = [*source_paths.values(), output_path, error_path, manifest_path]
    if len(set(configured_paths)) != len(configured_paths):
        raise ValueError("Input, output, error, and manifest paths must be distinct")

    grant_claims = _load_claim_cache(
        source_paths["grant_claims"],
        expected_item_type="grant_requirement",
        expected_prefix="grant",
    )
    faculty_claims = _load_claim_cache(
        source_paths["faculty_claims"],
        expected_item_type="faculty_capability",
        expected_prefix="faculty",
    )
    bands = tuple(args.bands)
    if args.all_prefilter_candidates:
        per_query_counts = {band: sys.maxsize for band in BANDS}
    else:
        per_query_counts = {
            "high": args.high_per_query,
            "mid": args.mid_per_query,
            "low": args.low_per_query,
        }
    candidates = base_distillation._load_grant_faculty_pairs(
        source_paths["prefilter"],
        bands=bands,
        per_query_counts=per_query_counts,
        seed=args.seed,
    )
    candidates = _deterministic_pair_cap(
        candidates,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    if not candidates:
        raise RuntimeError("No grant-faculty candidates were selected")
    pairs = _join_candidates_to_claims(
        candidates,
        grant_claims=grant_claims,
        faculty_claims=faculty_claims,
    )

    model_id = _clean_text(args.model_id)
    if not model_id:
        raise ValueError("--model-id cannot be empty")
    source_hashes = {name: _file_sha256(path) for name, path in source_paths.items()}
    generation_identity = {
        "teacher_model": model_id,
        "prompt_version": PROMPT_VERSION,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "max_new_tokens": args.max_new_tokens,
        "prefilter_sha256": source_hashes["prefilter"],
        "grant_claims_sha256": source_hashes["grant_claims"],
        "faculty_claims_sha256": source_hashes["faculty_claims"],
    }
    generation_fingerprint = hashlib.sha256(
        json.dumps(generation_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    judgment_ids = {
        pair.candidate.pair_id: _stable_id(
            pair.candidate.pair_id,
            pair.grant.record_id,
            pair.faculty.record_id,
            generation_fingerprint,
            prefix="structured_judgment",
        )
        for pair in pairs
    }
    if len(judgment_ids) != len(pairs):
        raise RuntimeError("Duplicate pair IDs were selected")

    completed = set()
    if not args.overwrite:
        completed = _load_completed_judgment_ids(
            output_path,
            generation_fingerprint=generation_fingerprint,
        )
    pending = [
        pair
        for pair in pairs
        if judgment_ids[pair.candidate.pair_id] not in completed
    ]

    if args.dry_run:
        print(f"prefilter={source_paths['prefilter']}")
        print(f"grant_claims={source_paths['grant_claims']}")
        print(f"faculty_claims={source_paths['faculty_claims']}")
        print(f"grant_claim_items={len(grant_claims)}")
        print(f"faculty_claim_items={len(faculty_claims)}")
        print(f"selected_pairs={len(pairs)}")
        print(f"pending_pairs={len(pending)}")
        for pair in pairs[: args.dry_run_limit]:
            print(f"--- pair {pair.candidate.pair_id} ---")
            print(SYSTEM_PROMPT)
            print(_user_prompt(pair))
        print(f"model_id={model_id}")
        print(f"generation_fingerprint={generation_fingerprint}")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    accepted_this_run = 0
    invalid_this_run = 0
    retried_this_run = 0
    run_id = "structured-distillation-" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    llm: Any = None
    progress: Any = None
    with output_path.open(mode, encoding="utf-8") as output_handle, error_path.open(
        mode,
        encoding="utf-8",
    ) as error_handle:
        if pending:
            try:
                from vllm import LLM, SamplingParams
            except ImportError as exc:
                raise RuntimeError(
                    "vLLM is required on the HPC runtime. Install it before "
                    "running structured distillation."
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

            for batch in _chunks(pending, args.batch_size):
                prompts = [
                    _apply_chat_template(
                        tokenizer,
                        pair=pair,
                        enable_thinking=args.enable_thinking,
                    )
                    for pair in batch
                ]
                responses = _generated_texts(
                    llm,
                    prompts,
                    sampling_params,
                    use_tqdm=args.vllm_tqdm,
                )
                parsed_results = [
                    _parse_structured_judgment(
                        response,
                        requirement_claim_ids=[
                            claim.claim_id for claim in pair.grant.claims
                        ],
                        capability_claim_ids=[
                            claim.claim_id for claim in pair.faculty.claims
                        ],
                    )
                    for pair, response in zip(batch, responses, strict=True)
                ]
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
                            pair=batch[index],
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
                        pair = batch[index]
                        parsed_results[index] = _parse_structured_judgment(
                            retry_response,
                            requirement_claim_ids=[
                                claim.claim_id for claim in pair.grant.claims
                            ],
                            capability_claim_ids=[
                                claim.claim_id for claim in pair.faculty.claims
                            ],
                        )

                created_at = _utc_now()
                for pair, response, (parsed, parse_error) in zip(
                    batch,
                    responses,
                    parsed_results,
                    strict=True,
                ):
                    candidate = pair.candidate
                    judgment_id = judgment_ids[candidate.pair_id]
                    if parsed is None:
                        error_handle.write(
                            json.dumps(
                                {
                                    "schema_version": ERROR_SCHEMA_VERSION,
                                    "judgment_id": judgment_id,
                                    "pair_id": candidate.pair_id,
                                    "grant_item_id": pair.grant.item_id,
                                    "faculty_item_id": pair.faculty.item_id,
                                    "teacher_model": model_id,
                                    "prompt_version": PROMPT_VERSION,
                                    "generation_fingerprint": generation_fingerprint,
                                    "parse_error": parse_error,
                                    "raw_response": response,
                                    "run_id": run_id,
                                    "created_at_utc": created_at,
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
                        "pair_id": candidate.pair_id,
                        "source_pair_id": candidate.source_pair_id,
                        "pair_type": "grant_faculty",
                        "direction": "grant_to_faculty",
                        "grant_item_id": pair.grant.item_id,
                        "grant_owner_id": candidate.target_owner_id,
                        "grant_keyword_index": candidate.target_keyword_index,
                        "grant_text": pair.grant.source_text,
                        "faculty_item_id": pair.faculty.item_id,
                        "faculty_owner_id": candidate.candidate_owner_id,
                        "faculty_keyword_index": candidate.candidate_keyword_index,
                        "faculty_text": pair.faculty.source_text,
                        "grant_decomposition_record_id": pair.grant.record_id,
                        "faculty_decomposition_record_id": pair.faculty.record_id,
                        "grant_requirements": [
                            asdict(claim) for claim in pair.grant.claims
                        ],
                        "faculty_capabilities": [
                            asdict(claim) for claim in pair.faculty.claims
                        ],
                        "requirement_coverage": [
                            asdict(coverage)
                            for coverage in parsed.requirement_coverage
                        ],
                        "overall_score": parsed.overall_score,
                        "overall_confidence": parsed.overall_confidence,
                        "rationale": parsed.rationale,
                        "prefilter_band": candidate.prefilter_band,
                        "ce_sts_score": candidate.ce_sts_score,
                        "ce_sts_logit": candidate.ce_sts_logit,
                        "ce_sts_rank": candidate.ce_sts_rank,
                        "ce_sts_rank_percentile": candidate.ce_sts_rank_percentile,
                        "teacher_model": model_id,
                        "prompt_version": PROMPT_VERSION,
                        "generation_fingerprint": generation_fingerprint,
                        "temperature": args.temperature,
                        "raw_response": response,
                        "run_id": run_id,
                        "created_at_utc": created_at,
                    }
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    accepted_this_run += 1

                output_handle.flush()
                error_handle.flush()
                if progress is not None:
                    progress.update(len(batch))
        else:
            print("All selected G-F pairs already have structured judgments.")

    if progress is not None:
        progress.close()
    _write_manifest(
        manifest_path,
        started=started,
        run_id=run_id,
        source_paths=source_paths,
        source_hashes=source_hashes,
        generation_identity=generation_identity,
        generation_fingerprint=generation_fingerprint,
        output_path=output_path,
        error_path=error_path,
        grant_claims=grant_claims,
        faculty_claims=faculty_claims,
        pairs=pairs,
        pending_count=len(pending),
        already_completed=len(pairs) - len(pending),
        accepted_this_run=accepted_this_run,
        invalid_this_run=invalid_this_run,
        retried_this_run=retried_this_run,
    )
    print(f"output={output_path}")
    print(f"errors={error_path}")
    print(f"manifest={manifest_path}")
    print(f"accepted_this_run={accepted_this_run}")
    print(f"invalid_after_retry_this_run={invalid_this_run}")
    print(f"elapsed_seconds_this_run={time.time() - started:.2f}")

    if llm is not None:
        del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
