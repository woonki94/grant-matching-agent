"""Decompose CE5 grant and faculty keywords into cached atomic claims.

This is an offline, source-level preprocessing stage.  Grant requirements and
faculty capabilities are decomposed independently so a paired judge cannot
redefine either source to fit its counterpart.  The resulting JSONL files use
the exact item IDs produced by ``build_prefilter.py`` and can therefore be
joined to existing prefilter candidates without rebuilding that cache.

The process is append-only and resumable by ``(item_id, source_text_sha256)``.
Invalid generations are retried once, then written to role-specific error
files.  Every accepted source span must be an exact substring of the normalized
source keyword; fuzzy or hallucinated evidence is rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, TextIO


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SOURCE_DIR = REPO_ROOT / "ce5" / "dataset" / "source"
DEFAULT_GRANT_DB = SOURCE_DIR / "grant_specialization_keyword_db.json"
DEFAULT_FACULTY_DB = SOURCE_DIR / "faculty_specialization_keywords_db.json"
DEFAULT_GRANT_OUTPUT = SOURCE_DIR / "grant_requirement_claims.jsonl"
DEFAULT_FACULTY_OUTPUT = SOURCE_DIR / "faculty_capability_claims.jsonl"
DEFAULT_GRANT_ERRORS = SOURCE_DIR / "grant_requirement_claim_errors.jsonl"
DEFAULT_FACULTY_ERRORS = SOURCE_DIR / "faculty_capability_claim_errors.jsonl"
DEFAULT_MANIFEST = SOURCE_DIR / "source_claims.manifest.json"
DEFAULT_MODEL_ID = "Qwen/Qwen3-14B"

SCHEMA_VERSION = "ce5.source-claims.v1"
ERROR_SCHEMA_VERSION = "ce5.source-claim-error.v1"
PROMPT_VERSION = "ce5-source-claims-v1"
ROLES = ("grant", "faculty")


COMMON_RULES = """
Return the smallest nonredundant set of independently assessable claims in the
source keyword. If the keyword expresses one capability, return exactly one
claim. Do not split method, domain, population, or application merely because
they are separate noun phrases; keep details together when they jointly define
one capability.

Every claim must be grounded in one exact contiguous substring copied from the
SOURCE KEYWORD. The source_span must preserve the source wording. Do not add
external knowledge, implied credentials, generic background abilities, or
claims not supported by the text.

Confidence is confidence that the decomposition faithfully represents the
source, not the strength or quality of the capability.

Return exactly one JSON object:
{
  "claims": [
    {
      "claim": "<one independently assessable normalized claim>",
      "source_span": "<exact contiguous substring from SOURCE KEYWORD>"
    }
  ],
  "confidence": <number from 0.00 to 1.00>
}

Do not output markdown, commentary, scores, pair judgments, fixed aspect names,
or any text outside the JSON object.
""".strip()


GRANT_SYSTEM_PROMPT = (
    """
You extract atomic capability requirements from a grant specialization keyword.

Describe only what an applicant, investigator, team, or institution would need
to be able to do. Preserve meaningful constraints such as the required method,
application, population, setting, objective, or deliverable when they are part
of the same assessable requirement. Do not rewrite the grant as a faculty
biography and do not infer requirements from any possible matching candidate.
""".strip()
    + "\n\n"
    + COMMON_RULES
)


FACULTY_SYSTEM_PROMPT = (
    """
You extract atomic demonstrated capabilities from a faculty specialization keyword.

Describe only methods, expertise, experience, systems, applications, or
transferable abilities evidenced by the text. Preserve meaningful constraints
when they jointly define the capability. Do not convert the text into grant
requirements and do not infer abilities that are merely plausible for someone
in the same broad field.
""".strip()
    + "\n\n"
    + COMMON_RULES
)


USER_PROMPT_TEMPLATE = """
SOURCE KEYWORD:
{source_text}
""".strip()


REPAIR_PROMPT = """
Your previous response was invalid. Return only one JSON object with a nonempty
claims array and numeric confidence in [0,1]. Every claim needs a nonempty claim
and a source_span copied exactly as one contiguous substring of SOURCE KEYWORD.

Do not manufacture a source_span by deleting a conjunction or intervening
words. In coordinated phrases such as "A and B of C", neither "A of C" nor
"B of C" is a contiguous source span. When separately grounded claims would
require that rewrite, merge the coordinated elements into one composite claim
and use the complete relevant source phrase—up to the entire SOURCE KEYWORD—as
its exact source_span.

Return fewer claims when necessary to preserve exact grounding. Do not include
markdown or additional text.
""".strip()


@dataclass(frozen=True)
class SourceKeyword:
    role: str
    item_type: str
    item_id: str
    owner_id: str | int
    keyword_index: int
    source_text: str
    source_text_sha256: str


@dataclass(frozen=True)
class ParsedClaim:
    claim: str
    source_span: str
    source_span_start: int
    source_span_end: int


@dataclass(frozen=True)
class ParsedDecomposition:
    claims: tuple[ParsedClaim, ...]
    confidence: float


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _resolve_path(value: Path) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(*parts: Any, prefix: str) -> str:
    raw = "\x1f".join(_clean_text(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:24]}"


def _load_source_keywords(
    path: Path,
    *,
    role: str,
) -> list[SourceKeyword]:
    if role not in ROLES:
        raise ValueError(f"Unsupported role: {role!r}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected an object in {path}")
    collection_name = "grants" if role == "grant" else "faculty"
    owner_key = "grant_id" if role == "grant" else "faculty_id"
    item_type = "grant_requirement" if role == "grant" else "faculty_capability"
    rows = payload.get(collection_name)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a '{collection_name}' list in {path}")

    output: list[SourceKeyword] = []
    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        owner_id = row.get(owner_key)
        if owner_id is None or (isinstance(owner_id, str) and not owner_id.strip()):
            continue
        keywords = row.get("specialization_keywords")
        if not isinstance(keywords, list):
            continue
        for keyword_index, raw_text in enumerate(keywords):
            source_text = _clean_text(raw_text)
            if not source_text:
                continue
            item_id = f"{role}:{owner_id}:{keyword_index}"
            if item_id in seen_ids:
                raise RuntimeError(f"Duplicate source item ID in {path}: {item_id}")
            seen_ids.add(item_id)
            output.append(
                SourceKeyword(
                    role=role,
                    item_type=item_type,
                    item_id=item_id,
                    owner_id=owner_id,
                    keyword_index=keyword_index,
                    source_text=source_text,
                    source_text_sha256=_sha256_text(source_text),
                )
            )
    return output


def _extract_json_object(raw_text: str) -> Optional[dict[str, Any]]:
    text = str(raw_text or "").strip()
    candidates: list[str] = []
    fenced = re.findall(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidates.extend(fenced)
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


def _canonical_source_span(source_text: str, proposed_span: Any) -> Optional[tuple[str, int, int]]:
    span = _clean_text(proposed_span)
    if not span:
        return None
    start = source_text.find(span)
    if start < 0:
        match = re.search(re.escape(span), source_text, flags=re.IGNORECASE)
        if match is None:
            return None
        start, end = match.span()
        span = source_text[start:end]
        return span, start, end
    end = start + len(span)
    return span, start, end


def _parse_decomposition(
    raw_text: str,
    *,
    source_text: str,
    max_claims: int,
) -> tuple[Optional[ParsedDecomposition], str]:
    payload = _extract_json_object(raw_text)
    if payload is None:
        return None, "no_valid_json_object"
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        return None, "missing_or_empty_claims"
    if len(raw_claims) > max_claims:
        return None, f"too_many_claims:{len(raw_claims)}>{max_claims}"
    try:
        confidence = float(payload["confidence"])
    except (KeyError, TypeError, ValueError):
        return None, "missing_or_non_numeric_confidence"
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None, "confidence_out_of_range"

    claims: list[ParsedClaim] = []
    seen_claims: set[str] = set()
    for index, raw_claim in enumerate(raw_claims):
        if not isinstance(raw_claim, Mapping):
            return None, f"claim_{index}_is_not_an_object"
        claim = _clean_text(raw_claim.get("claim"))
        if not claim:
            return None, f"claim_{index}_missing_claim"
        claim_key = claim.casefold()
        if claim_key in seen_claims:
            return None, f"claim_{index}_duplicates_an_earlier_claim"
        seen_claims.add(claim_key)
        span = _canonical_source_span(source_text, raw_claim.get("source_span"))
        if span is None:
            return None, f"claim_{index}_source_span_not_in_source"
        canonical_span, start, end = span
        claims.append(
            ParsedClaim(
                claim=claim,
                source_span=canonical_span,
                source_span_start=start,
                source_span_end=end,
            )
        )
    return ParsedDecomposition(tuple(claims), confidence), ""


def _system_prompt(role: str) -> str:
    if role == "grant":
        return GRANT_SYSTEM_PROMPT
    if role == "faculty":
        return FACULTY_SYSTEM_PROMPT
    raise ValueError(f"Unsupported role: {role!r}")


def _apply_chat_template(
    tokenizer: Any,
    *,
    item: SourceKeyword,
    enable_thinking: bool,
    invalid_response: str = "",
) -> str:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _system_prompt(item.role)},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(source_text=item.source_text),
        },
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


def _build_record(
    item: SourceKeyword,
    parsed: ParsedDecomposition,
    *,
    model_id: str,
    generation_fingerprint: str,
    run_id: str,
    created_at_utc: str,
) -> dict[str, Any]:
    claim_prefix = "r" if item.role == "grant" else "c"
    claims = [
        {
            "claim_id": f"{item.item_id}:{claim_prefix}{index}",
            "claim": claim.claim,
            "source_span": claim.source_span,
            "source_span_start": claim.source_span_start,
            "source_span_end": claim.source_span_end,
        }
        for index, claim in enumerate(parsed.claims)
    ]
    record_id = _stable_id(
        item.item_id,
        item.source_text_sha256,
        generation_fingerprint,
        prefix="source_claims",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "generation_fingerprint": generation_fingerprint,
        "item_type": item.item_type,
        "item_id": item.item_id,
        "owner_id": item.owner_id,
        "keyword_index": item.keyword_index,
        "source_text": item.source_text,
        "source_text_sha256": item.source_text_sha256,
        "claims": claims,
        "decomposition_confidence": parsed.confidence,
        "teacher_model": model_id,
        "prompt_version": PROMPT_VERSION,
        "run_id": run_id,
        "created_at_utc": created_at_utc,
    }


def _load_completed_keys(
    path: Path,
    *,
    generation_fingerprint: str,
    expected_item_type: str,
) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
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
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            if row.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Existing output at {path}:{line_number} uses schema "
                    f"{row.get('schema_version')!r}; use --overwrite or a new path"
                )
            if row.get("generation_fingerprint") != generation_fingerprint:
                raise RuntimeError(
                    f"Existing output at {path}:{line_number} uses different "
                    "generation settings; use --overwrite or a new output path"
                )
            if row.get("item_type") != expected_item_type:
                raise RuntimeError(
                    f"Unexpected item_type at {path}:{line_number}: "
                    f"{row.get('item_type')!r}"
                )
            item_id = _clean_text(row.get("item_id"))
            source_hash = _clean_text(row.get("source_text_sha256"))
            if item_id and source_hash:
                completed.add((item_id, source_hash))
    return completed


def _current_claim_counts(
    path: Path,
    *,
    selected_keys: set[tuple[str, str]],
    generation_fingerprint: str,
) -> list[int]:
    by_key: dict[tuple[str, str], int] = {}
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("generation_fingerprint") != generation_fingerprint:
                continue
            key = (
                _clean_text(row.get("item_id")),
                _clean_text(row.get("source_text_sha256")),
            )
            if key not in selected_keys:
                continue
            claims = row.get("claims")
            if isinstance(claims, list):
                by_key[key] = len(claims)
    return list(by_key.values())


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _interleave_roles(
    grants: Sequence[SourceKeyword],
    faculty: Sequence[SourceKeyword],
) -> list[SourceKeyword]:
    output: list[SourceKeyword] = []
    for grant, faculty_item in zip_longest(grants, faculty):
        if grant is not None:
            output.append(grant)
        if faculty_item is not None:
            output.append(faculty_item)
    return output


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
            desc="CE5 source claim extraction",
            unit="keyword",
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


def _parse_roles(value: str) -> tuple[str, ...]:
    roles = tuple(
        token.strip().lower() for token in value.split(",") if token.strip()
    )
    invalid = sorted(set(roles) - set(ROLES))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported role(s): {', '.join(invalid)}; use grant,faculty"
        )
    if not roles:
        raise argparse.ArgumentTypeError("At least one role is required")
    return tuple(dict.fromkeys(roles))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose exported grant and faculty specialization keywords into "
            "independent atomic-claim JSONL caches with a local vLLM teacher."
        )
    )
    parser.add_argument("--grant-db", type=Path, default=DEFAULT_GRANT_DB)
    parser.add_argument("--faculty-db", type=Path, default=DEFAULT_FACULTY_DB)
    parser.add_argument("--grant-output", type=Path, default=DEFAULT_GRANT_OUTPUT)
    parser.add_argument("--faculty-output", type=Path, default=DEFAULT_FACULTY_OUTPUT)
    parser.add_argument("--grant-errors", type=Path, default=DEFAULT_GRANT_ERRORS)
    parser.add_argument("--faculty-errors", type=Path, default=DEFAULT_FACULTY_ERRORS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--roles", type=_parse_roles, default=ROLES)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-grant-keywords", type=_nonnegative_int, default=0)
    parser.add_argument("--max-faculty-keywords", type=_nonnegative_int, default=0)
    parser.add_argument("--max-claims-per-keyword", type=_positive_int, default=6)
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=512)
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
        help="Replace selected role outputs and error files instead of resuming.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source files and print prompts without loading vLLM.",
    )
    parser.add_argument("--dry-run-limit", type=_positive_int, default=2)
    return parser


def _write_manifest(
    path: Path,
    *,
    started: float,
    roles: Sequence[str],
    generation_identity: Mapping[str, Any],
    generation_fingerprint: str,
    source_paths: Mapping[str, Path],
    source_items: Mapping[str, Sequence[SourceKeyword]],
    output_paths: Mapping[str, Path],
    completed_keys: Mapping[str, set[tuple[str, str]]],
    accepted_this_run: Mapping[str, int],
    invalid_this_run: Mapping[str, int],
    retried_this_run: Mapping[str, int],
) -> None:
    role_stats: dict[str, Any] = {}
    for role in roles:
        selected_keys = {
            (item.item_id, item.source_text_sha256) for item in source_items[role]
        }
        current_completed = selected_keys & completed_keys[role]
        claim_counts = _current_claim_counts(
            output_paths[role],
            selected_keys=selected_keys,
            generation_fingerprint=generation_fingerprint,
        )
        role_stats[role] = {
            "source_path": str(source_paths[role]),
            "source_sha256": _file_sha256(source_paths[role]),
            "output_path": str(output_paths[role]),
            "selected_keywords": len(selected_keys),
            "completed_keywords": len(current_completed),
            "missing_keywords": len(selected_keys - current_completed),
            "accepted_this_run": accepted_this_run[role],
            "invalid_this_run": invalid_this_run[role],
            "retried_this_run": retried_this_run[role],
            "claims_per_keyword": dict(sorted(Counter(claim_counts).items())),
            "mean_claims_per_keyword": (
                sum(claim_counts) / len(claim_counts) if claim_counts else 0.0
            ),
        }
    payload = {
        "schema_version": "ce5.source-claims-manifest.v1",
        "created_at_utc": _utc_now(),
        "elapsed_seconds": time.time() - started,
        "roles": list(roles),
        "generation_identity": dict(generation_identity),
        "generation_fingerprint": generation_fingerprint,
        "role_stats": role_stats,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    roles = tuple(args.roles)
    source_paths = {
        "grant": _resolve_path(args.grant_db),
        "faculty": _resolve_path(args.faculty_db),
    }
    output_paths = {
        "grant": _resolve_path(args.grant_output),
        "faculty": _resolve_path(args.faculty_output),
    }
    error_paths = {
        "grant": _resolve_path(args.grant_errors),
        "faculty": _resolve_path(args.faculty_errors),
    }
    manifest_path = _resolve_path(args.manifest)
    for role in roles:
        if not source_paths[role].exists():
            raise FileNotFoundError(f"Input file not found: {source_paths[role]}")
    selected_paths = [
        path
        for role in roles
        for path in (output_paths[role], error_paths[role])
    ]
    if len(set(selected_paths)) != len(selected_paths):
        raise ValueError("Selected output and error paths must all be distinct")

    source_items: dict[str, list[SourceKeyword]] = {role: [] for role in ROLES}
    if "grant" in roles:
        source_items["grant"] = _load_source_keywords(
            source_paths["grant"],
            role="grant",
        )
        if args.max_grant_keywords > 0:
            source_items["grant"] = source_items["grant"][
                : args.max_grant_keywords
            ]
    if "faculty" in roles:
        source_items["faculty"] = _load_source_keywords(
            source_paths["faculty"],
            role="faculty",
        )
        if args.max_faculty_keywords > 0:
            source_items["faculty"] = source_items["faculty"][
                : args.max_faculty_keywords
            ]
    for role in roles:
        if not source_items[role]:
            raise RuntimeError(f"No {role} specialization keywords were loaded")

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
        "max_claims_per_keyword": args.max_claims_per_keyword,
    }
    generation_fingerprint = hashlib.sha256(
        json.dumps(generation_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()

    completed_keys: dict[str, set[tuple[str, str]]] = {
        role: set() for role in ROLES
    }
    if not args.overwrite:
        for role in roles:
            completed_keys[role] = _load_completed_keys(
                output_paths[role],
                generation_fingerprint=generation_fingerprint,
                expected_item_type=(
                    "grant_requirement" if role == "grant" else "faculty_capability"
                ),
            )
    pending_by_role = {
        role: [
            item
            for item in source_items[role]
            if (item.item_id, item.source_text_sha256) not in completed_keys[role]
        ]
        for role in roles
    }
    pending = _interleave_roles(
        pending_by_role.get("grant", []),
        pending_by_role.get("faculty", []),
    )

    if args.dry_run:
        print(f"roles={','.join(roles)}")
        for role in roles:
            print(f"{role}_db={source_paths[role]}")
            print(f"{role}_keywords={len(source_items[role])}")
            print(f"{role}_pending={len(pending_by_role[role])}")
            for item in source_items[role][: args.dry_run_limit]:
                print(f"--- {role} item {item.item_id} ---")
                print(_system_prompt(role))
                print(USER_PROMPT_TEMPLATE.format(source_text=item.source_text))
        print(f"model_id={model_id}")
        print(f"generation_fingerprint={generation_fingerprint}")
        return 0

    for role in roles:
        output_paths[role].parent.mkdir(parents=True, exist_ok=True)
        error_paths[role].parent.mkdir(parents=True, exist_ok=True)
    accepted_this_run = Counter({role: 0 for role in ROLES})
    invalid_this_run = Counter({role: 0 for role in ROLES})
    retried_this_run = Counter({role: 0 for role in ROLES})

    output_handles: dict[str, TextIO] = {}
    error_handles: dict[str, TextIO] = {}
    mode = "w" if args.overwrite else "a"
    try:
        for role in roles:
            output_handles[role] = output_paths[role].open(mode, encoding="utf-8")
            error_handles[role] = error_paths[role].open(mode, encoding="utf-8")

        if pending:
            try:
                from vllm import LLM, SamplingParams
            except ImportError as exc:
                raise RuntimeError(
                    "vLLM is required on the HPC runtime. Install it before "
                    "running CE5 source claim extraction."
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
            run_id = (
                "source-claims-"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            )

            for batch in _chunks(pending, args.batch_size):
                prompts = [
                    _apply_chat_template(
                        tokenizer,
                        item=item,
                        enable_thinking=args.enable_thinking,
                    )
                    for item in batch
                ]
                responses = _generated_texts(
                    llm,
                    prompts,
                    sampling_params,
                    use_tqdm=args.vllm_tqdm,
                )
                parsed_results = [
                    _parse_decomposition(
                        response,
                        source_text=item.source_text,
                        max_claims=args.max_claims_per_keyword,
                    )
                    for item, response in zip(batch, responses, strict=True)
                ]
                invalid_indices = [
                    index
                    for index, (parsed, _) in enumerate(parsed_results)
                    if parsed is None
                ]
                if invalid_indices:
                    for index in invalid_indices:
                        retried_this_run[batch[index].role] += 1
                    retry_prompts = [
                        _apply_chat_template(
                            tokenizer,
                            item=batch[index],
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
                        parsed_results[index] = _parse_decomposition(
                            retry_response,
                            source_text=batch[index].source_text,
                            max_claims=args.max_claims_per_keyword,
                        )

                created_at = _utc_now()
                for item, response, (parsed, parse_error) in zip(
                    batch,
                    responses,
                    parsed_results,
                    strict=True,
                ):
                    if parsed is None:
                        invalid_this_run[item.role] += 1
                        error_handles[item.role].write(
                            json.dumps(
                                {
                                    "schema_version": ERROR_SCHEMA_VERSION,
                                    "item_type": item.item_type,
                                    "item_id": item.item_id,
                                    "owner_id": item.owner_id,
                                    "keyword_index": item.keyword_index,
                                    "source_text": item.source_text,
                                    "source_text_sha256": item.source_text_sha256,
                                    "generation_fingerprint": generation_fingerprint,
                                    "error": parse_error,
                                    "raw_response": response,
                                    "run_id": run_id,
                                    "created_at_utc": created_at,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue
                    record = _build_record(
                        item,
                        parsed,
                        model_id=model_id,
                        generation_fingerprint=generation_fingerprint,
                        run_id=run_id,
                        created_at_utc=created_at,
                    )
                    output_handles[item.role].write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                    completed_keys[item.role].add(
                        (item.item_id, item.source_text_sha256)
                    )
                    accepted_this_run[item.role] += 1
                for role in roles:
                    output_handles[role].flush()
                    error_handles[role].flush()
                if progress is not None:
                    progress.update(len(batch))
            if progress is not None:
                progress.close()
        else:
            print("All selected source keywords already have claim decompositions.")
    finally:
        for handle in output_handles.values():
            handle.close()
        for handle in error_handles.values():
            handle.close()

    _write_manifest(
        manifest_path,
        started=started,
        roles=roles,
        generation_identity=generation_identity,
        generation_fingerprint=generation_fingerprint,
        source_paths=source_paths,
        source_items=source_items,
        output_paths=output_paths,
        completed_keys=completed_keys,
        accepted_this_run=accepted_this_run,
        invalid_this_run=invalid_this_run,
        retried_this_run=retried_this_run,
    )
    for role in roles:
        print(f"{role}_output={output_paths[role]}")
        print(f"{role}_errors={error_paths[role]}")
        print(f"{role}_accepted_this_run={accepted_this_run[role]}")
        print(f"{role}_invalid_this_run={invalid_this_run[role]}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
