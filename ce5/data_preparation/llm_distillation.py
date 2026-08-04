"""Score CE5 prefilter candidates with a local instruction-tuned LLM.

This is the offline teacher stage.  It reads the high/mid/low candidate sets
created by ``build_prefilter.py``, builds one independent prompt per pair, and
uses vLLM to produce continuous capability-coverage judgments.

CE-STS scores, ranks, and selection bands are deliberately excluded from the
teacher prompt so they cannot anchor the LLM's judgment.  Output is append-only
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASET_DIR = REPO_ROOT / "ce5" / "dataset"
DEFAULT_PREFILTER = DATASET_DIR / "source" / "prefilter_candidates.jsonl"
DEFAULT_OUTPUT = DATASET_DIR / "judgments" / "teacher_judgments.jsonl"
DEFAULT_ERRORS = DATASET_DIR / "judgments" / "teacher_judgment_errors.jsonl"
DEFAULT_MANIFEST = DATASET_DIR / "judgments" / "teacher_judgments.manifest.json"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3-14B"
PREFILTER_SCHEMA_VERSION = 2
JUDGMENT_SCHEMA_VERSION = "ce5.judgment.v1"
ERROR_SCHEMA_VERSION = "ce5.judgment-error.v1"
PROMPT_VERSION = "capability-coverage-v1"
BANDS = ("high", "mid", "low")


SYSTEM_PROMPT = """
You are a strict evaluator for matching grant requirements to faculty capabilities.

Your task is directional:
Determine how strongly the FACULTY CAPABILITY provides evidence that the faculty can satisfy the GRANT REQUIREMENT.

This is not ordinary semantic similarity. Do not give a high score merely because both statements mention the same topic, domain, population, technology, or broad goal. The faculty statement must demonstrate a relevant capability, method, expertise, experience, or transferable ability that covers the requirement.

Do not assume important capabilities that are not supported by the faculty statement. Closely related or transferable capability may receive partial credit, but missing essential requirements must reduce the score.

Use a continuous score from 0.00 to 1.00:
- 0.00: unrelated, contradictory, or no evidence of the required capability
- 0.25: topical connection but little evidence that the requirement can be satisfied
- 0.50: partial or plausibly transferable capability with substantial gaps
- 0.75: substantial capability coverage with a meaningful remaining gap
- 1.00: direct, specific, and strong evidence that the requirement can be satisfied

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
GRANT REQUIREMENT:
{grant_text}

FACULTY CAPABILITY:
{faculty_text}
""".strip()


REPAIR_PROMPT = """
Your previous response did not match the required JSON schema. Return only one valid JSON object containing numeric score and confidence values in [0,1] and one concise rationale string. Do not include markdown or any other text.
""".strip()


@dataclass(frozen=True)
class CandidatePair:
    pair_id: str
    grant_item_id: str
    grant_id: str | int
    grant_keyword_index: int
    grant_text: str
    faculty_item_id: str
    faculty_id: str | int
    faculty_keyword_index: int
    faculty_text: str
    prefilter_band: str
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


def _load_candidate_pairs(
    path: Path,
    *,
    bands: Sequence[str],
    maximum: int,
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
            if row.get("schema_version") != PREFILTER_SCHEMA_VERSION:
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
                for candidate in candidates:
                    if not isinstance(candidate, Mapping):
                        continue
                    faculty_item_id = _clean_text(candidate.get("faculty_item_id"))
                    faculty_text = _clean_text(candidate.get("faculty_text"))
                    if not faculty_item_id or not faculty_text:
                        continue
                    pair_id = _stable_id(
                        grant_item_id,
                        faculty_item_id,
                        prefix="pair",
                    )
                    if pair_id in seen_pair_ids:
                        continue
                    seen_pair_ids.add(pair_id)
                    output.append(
                        CandidatePair(
                            pair_id=pair_id,
                            grant_item_id=grant_item_id,
                            grant_id=row.get("grant_id", ""),
                            grant_keyword_index=int(
                                row.get("grant_keyword_index", -1) or 0
                            ),
                            grant_text=grant_text,
                            faculty_item_id=faculty_item_id,
                            faculty_id=candidate.get("faculty_id", ""),
                            faculty_keyword_index=int(
                                candidate.get("faculty_keyword_index", -1) or 0
                            ),
                            faculty_text=faculty_text,
                            prefilter_band=band,
                            ce_sts_score=float(candidate.get("ce_sts_score", 0.0)),
                            ce_sts_logit=float(candidate.get("ce_sts_logit", 0.0)),
                            ce_sts_rank=int(candidate.get("ce_sts_rank", -1)),
                            ce_sts_rank_percentile=float(
                                candidate.get("ce_sts_rank_percentile", 0.0)
                            ),
                        )
                    )
                    if maximum > 0 and len(output) >= maximum:
                        return output
    return output


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
        grant_text=candidate.grant_text,
        faculty_text=candidate.faculty_text,
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


def _load_completed_judgment_ids(path: Path) -> set[str]:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score CE5 prefilter candidates with a local vLLM teacher."
    )
    parser.add_argument("--prefilter", type=Path, default=DEFAULT_PREFILTER)
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
    parser.add_argument("--max-pairs", type=_nonnegative_int, default=0)
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
    prefilter_path = _resolve_path(args.prefilter)
    output_path = _resolve_path(args.output)
    error_path = _resolve_path(args.errors)
    manifest_path = _resolve_path(args.manifest)
    bands = tuple(args.bands)

    candidates = _load_candidate_pairs(
        prefilter_path,
        bands=bands,
        maximum=args.max_pairs,
    )
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
        print(f"prefilter={prefilter_path}")
        print(f"candidate_pairs={len(candidates)}")
        print(f"bands={','.join(bands)}")
        print(f"model_id={model_id}")
        for index, candidate in enumerate(candidates[: args.dry_run_limit], start=1):
            print(f"--- pair {index}: {candidate.pair_id} ---")
            print(USER_PROMPT_TEMPLATE.format(
                grant_text=candidate.grant_text,
                faculty_text=candidate.faculty_text,
            ))
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
        completed_ids = _load_completed_judgment_ids(output_path)
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
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "run_id": run_id,
        "prefilter": str(prefilter_path),
        "prefilter_sha256": _file_sha256(prefilter_path),
        "output": str(output_path),
        "errors": str(error_path),
        "teacher_model": model_id,
        "prompt_version": PROMPT_VERSION,
        "generation_fingerprint": generation_fingerprint,
        "configuration": {
            **generation_identity,
            "bands": list(bands),
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": args.dtype,
        },
        "candidate_pairs_loaded": len(candidates),
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
