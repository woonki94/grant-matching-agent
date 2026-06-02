from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce3").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_llm_client, settings  # noqa: E402


TEST_INPUT_DEFAULT = "ce3/dataset/splits/llm_distill_all_listwise_test.jsonl"
OUTPUT_DEFAULT = "ce3/dataset/ground_truth/overall_coverage_test_subset_claude_opus.jsonl"
ASPECTS = ("topic", "approach", "objective")
PREFIXES = ("[TOPIC]", "[APPROACH]", "[OBJECTIVE]")
DEFAULT_MAX_PAIRS = 10
DEFAULT_BATCH_SIZE = 24
DEFAULT_MAX_RETRIES = 2
DEFAULT_PER_QUERY_CAP = 3
HIGH_THRESHOLD = 0.70
MID_THRESHOLD = 0.30
MAX_TEXT_CHARS = 4000


class CoverageScoreItem(BaseModel):
    q: int = Field(..., description="1-based task index in this request")
    score: float = Field(..., description="Overall coverage score from 0.0 to 1.0")


class CoverageScoreBatch(BaseModel):
    items: List[CoverageScoreItem] = Field(default_factory=list)


SYSTEM_PROMPT = """
You are a strict grant-to-faculty specialization coverage judge.

You will receive multiple scoring tasks. Each task has:
- q: task index
- grant_specialization: a specialization keyword/requirement from a grant
- faculty_specialization: a specialization keyword/capability from a faculty profile

Your job:
Return one OVERALL COVERAGE SCORE in [0,1] for how well the faculty specialization covers the grant specialization.

Judge the full meaning, not just word overlap. Consider:
- Topic/context match: field, problem area, application area, population, system, or domain.
- Approach/capability match: method, technique, workflow, expertise, action, analysis, design, intervention, or work performed.
- Objective/target match: target object, beneficiary, material, system, outcome, setting, use case, purpose, or required condition.

Scoring guide:
- 0.90-1.00: near-exact or strongly equivalent coverage of topic, approach, and objective.
- 0.70-0.89: strong coverage; most central intent is covered, with only minor missing specificity.
- 0.50-0.69: useful partial coverage; important overlap exists, but one central element is missing, weaker, or only implied.
- 0.30-0.49: weak-to-moderate adjacent coverage; related but not enough for confident coverage.
- 0.00-0.29: low coverage; overlap is generic, broad, incidental, or mostly absent.

Important calibration rules:
- Do not give a high score for broad domain similarity alone.
- Do not give a high score for method similarity if the target/objective is different.
- Do not give a high score when a central population, system, setting, material, or purpose is missing.
- Prefer conservative scores when the faculty specialization is plausible but not explicit.
- Use the full 0..1 range. Near-boundary scores are allowed when the evidence is genuinely borderline.

Output MUST be exactly one JSON object:
{{
  "items": [
    {{"q": <int>, "score": <float 0..1>}}
  ]
}}

Rules:
- Include exactly one item for each input q.
- Do not include explanations.
- Do not include markdown.
- Do not include text outside JSON.
""".strip()


USER_PROMPT_TEMPLATE = """
Tasks JSON:
{tasks_json}
""".strip()


def _build_chain(model_id: str) -> Any:
    try:
        from langchain_core.prompts import ChatPromptTemplate
    except Exception as exc:
        raise RuntimeError("Missing dependency: langchain-core") from exc

    llm = get_llm_client(model_id).build()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT_TEMPLATE),
        ]
    )
    return prompt | llm.with_structured_output(CoverageScoreBatch)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_ws(value: Any) -> str:
    return " ".join(_clean_text(value).split())


def _resolve_path(value: Any) -> Path:
    path = Path(_clean_text(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _clamp_01(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    return max(0.0, min(1.0, parsed))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _strip_aspect_prefix(text: Any) -> str:
    raw = _normalize_ws(text)
    upper = raw.upper()
    for prefix in PREFIXES:
        if upper.startswith(prefix):
            return _normalize_ws(raw[len(prefix) :])
    return raw


def _score_band(score: float, *, high_threshold: float = HIGH_THRESHOLD, mid_threshold: float = MID_THRESHOLD) -> str:
    if float(score) >= float(high_threshold):
        return "high"
    if float(score) >= float(mid_threshold):
        return "mid"
    return "low"


def _band_letter(score: Optional[float], *, high_threshold: float, mid_threshold: float) -> str:
    if score is None:
        return "?"
    band = _score_band(score, high_threshold=high_threshold, mid_threshold=mid_threshold)
    return {"high": "H", "mid": "M", "low": "L"}.get(band, "?")


def _root_pair_id(pair_id: str, aspect: str) -> str:
    raw = _clean_text(pair_id)
    if raw.endswith(f"::{aspect}"):
        return raw[: -len(f"::{aspect}")]
    return raw


def _text_sig(text: str) -> str:
    return hashlib.sha1(_normalize_ws(text).lower().encode("utf-8")).hexdigest()[:16]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = _clean_text(raw)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL row at {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                yield obj


def _pair_key(row: Dict[str, Any], doc: Dict[str, Any], aspect: str) -> str:
    pair_id = _root_pair_id(_clean_text(doc.get("pair_id")), aspect)
    if pair_id:
        return pair_id
    query_item_id = _clean_text(row.get("query_item_id")) or _clean_text(row.get("grant_id"))
    fac_item_id = _clean_text(doc.get("fac_item_id"))
    if query_item_id and fac_item_id:
        return f"{query_item_id}::{fac_item_id}"
    return f"{query_item_id or _text_sig(row.get('raw_query_text') or row.get('query_text'))}::text::{_text_sig(doc.get('text'))}"


def _load_candidate_pairs(
    path: Path,
    *,
    high_threshold: float,
    mid_threshold: float,
) -> List[Dict[str, Any]]:
    pairs: Dict[str, Dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        aspect = _clean_text(row.get("aspect")).lower()
        if aspect not in ASPECTS:
            continue
        grant_text = _strip_aspect_prefix(row.get("raw_query_text") or row.get("query_text") or row.get("spec_text"))
        if not grant_text:
            continue
        docs = row.get("docs") or row.get("ranked_docs") or row.get("candidates") or []
        if not isinstance(docs, list):
            continue
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            faculty_text = _normalize_ws(doc.get("text"))
            if not faculty_text:
                continue
            key = _pair_key(row, doc, aspect)
            existing = pairs.setdefault(
                key,
                {
                    "pair_id": key,
                    "grant_id": _clean_text(row.get("grant_id")),
                    "spec_idx": _safe_int(row.get("spec_idx")),
                    "query_item_id": _clean_text(row.get("query_item_id")),
                    "grant_text": grant_text,
                    "fac_item_id": _clean_text(doc.get("fac_item_id")),
                    "faculty_text": faculty_text,
                    "aspect_scores": {},
                },
            )
            if not existing.get("grant_text"):
                existing["grant_text"] = grant_text
            if not existing.get("faculty_text"):
                existing["faculty_text"] = faculty_text
            score = _clamp_01(doc.get("teacher_score", doc.get("score")))
            existing["aspect_scores"][aspect] = score

    out: List[Dict[str, Any]] = []
    for item in pairs.values():
        scores = dict(item.get("aspect_scores") or {})
        if not scores:
            continue
        proxy = sum(float(v) for v in scores.values()) / max(1, len(scores))
        pattern = "".join(_band_letter(scores.get(aspect), high_threshold=high_threshold, mid_threshold=mid_threshold) for aspect in ASPECTS)
        item["proxy_score"] = float(proxy)
        item["proxy_band"] = _score_band(proxy, high_threshold=high_threshold, mid_threshold=mid_threshold)
        item["aspect_pattern"] = pattern
        out.append(item)
    return out


def _candidate_sort_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
    # Prefer complete aspect evidence, then stable IDs. This is deterministic, not random.
    completeness = sum(1 for aspect in ASPECTS if aspect in (item.get("aspect_scores") or {}))
    return (
        -int(completeness),
        _clean_text(item.get("query_item_id")),
        _clean_text(item.get("fac_item_id")),
        _clean_text(item.get("pair_id")),
        _normalize_ws(item.get("faculty_text")).lower(),
    )


def _select_varied_subset(
    pairs: Sequence[Dict[str, Any]],
    *,
    max_pairs: int,
    per_query_cap: int,
) -> List[Dict[str, Any]]:
    if int(max_pairs) <= 0 or len(pairs) <= int(max_pairs):
        return sorted(list(pairs), key=_candidate_sort_key)

    buckets: Dict[Tuple[str, str], Deque[Dict[str, Any]]] = defaultdict(deque)
    for item in sorted(pairs, key=_candidate_sort_key):
        buckets[(_clean_text(item.get("proxy_band")), _clean_text(item.get("aspect_pattern")))].append(item)

    band_rank = {"high": 0, "mid": 1, "low": 2}

    def interleaved_bucket_keys(keys: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
        by_band: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for key in sorted(keys, key=lambda x: (band_rank.get(x[0], 99), x[1])):
            by_band[key[0]].append(key)
        out: List[Tuple[str, str]] = []
        max_len = max((len(v) for v in by_band.values()), default=0)
        for i in range(max_len):
            for band in ("high", "mid", "low"):
                if i < len(by_band.get(band, [])):
                    out.append(by_band[band][i])
        return out

    bucket_keys = interleaved_bucket_keys(buckets.keys())
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    per_query_counts: Counter[str] = Counter()
    exhausted_rounds = 0
    cap = max(1, int(per_query_cap))

    while len(selected) < int(max_pairs) and exhausted_rounds < 2:
        made_progress = False
        for key in bucket_keys:
            bucket = buckets[key]
            while bucket:
                item = bucket.popleft()
                pair_id = _clean_text(item.get("pair_id"))
                query_key = _clean_text(item.get("query_item_id")) or _clean_text(item.get("grant_id"))
                if pair_id in selected_ids:
                    continue
                if per_query_counts[query_key] >= cap and exhausted_rounds == 0:
                    # First pass keeps query diversity. Second pass relaxes this if needed.
                    continue
                selected.append(item)
                selected_ids.add(pair_id)
                per_query_counts[query_key] += 1
                made_progress = True
                break
            if len(selected) >= int(max_pairs):
                break
        if not made_progress:
            exhausted_rounds += 1
            if exhausted_rounds == 1:
                # Rebuild remaining candidates for a relaxed pass without the per-query cap.
                remaining = [item for bucket in buckets.values() for item in bucket if _clean_text(item.get("pair_id")) not in selected_ids]
                buckets = defaultdict(deque)
                for item in sorted(remaining, key=_candidate_sort_key):
                    buckets[(_clean_text(item.get("proxy_band")), _clean_text(item.get("aspect_pattern")))].append(item)
                bucket_keys = interleaved_bucket_keys(buckets.keys())
    return selected


def _truncate_text(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    clean = _normalize_ws(text)
    if len(clean) <= int(limit):
        return clean
    return clean[: max(0, int(limit) - 3)].rstrip() + "..."


def _chunks(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    step = max(1, int(size))
    for i in range(0, len(seq), step):
        yield seq[i : i + step]


def _tqdm_iter(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, **kwargs)


def _select_model_id(model_id_arg: str, *, allow_fallback: bool) -> str:
    explicit = _clean_text(model_id_arg)
    if explicit:
        return explicit
    opus = _clean_text(getattr(settings, "opus", "")) or _clean_text(getattr(settings, "bedrock_claude_opus", ""))
    if opus:
        return opus
    if not bool(allow_fallback):
        raise RuntimeError("Claude Opus is not configured. Set BEDROCK_CLAUDE_OPUS, pass --model-id, or use --allow-model-fallback.")
    for candidate in (
        _clean_text(getattr(settings, "sonnet", "")),
        _clean_text(getattr(settings, "bedrock_claude_sonnet", "")),
        _clean_text(getattr(settings, "haiku", "")),
        _clean_text(getattr(settings, "bedrock_claude_haiku", "")),
    ):
        if candidate:
            return candidate
    raise RuntimeError("No Claude model id configured.")


def _score_with_llm(
    *,
    chain: Any,
    pairs: Sequence[Dict[str, Any]],
    batch_size: int,
    max_retries: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    stats = {
        "llm_calls": 0,
        "retries_used": 0,
        "failed_batches": 0,
        "skipped_pairs": 0,
    }
    batches = list(_chunks(list(pairs), batch_size))
    print(f"scoring_pairs={len(pairs)} scoring_batches={len(batches)} batch_size={max(1, int(batch_size))}")
    progress = _tqdm_iter(
        enumerate(batches, start=1),
        total=len(batches),
        desc="Claude GT batches",
        unit="batch",
        dynamic_ncols=True,
    )
    for batch_idx, batch in progress:
        tasks = [
            {
                "q": i + 1,
                "grant_specialization": _truncate_text(item.get("grant_text", "")),
                "faculty_specialization": _truncate_text(item.get("faculty_text", "")),
            }
            for i, item in enumerate(batch)
        ]
        parsed: Dict[int, Dict[str, Any]] = {}
        last_error: Optional[Exception] = None
        for attempt in range(max(1, int(max_retries) + 1)):
            stats["llm_calls"] += 1
            if attempt > 0:
                stats["retries_used"] += 1
            try:
                result = chain.invoke({"tasks_json": json.dumps(tasks, ensure_ascii=False, separators=(",", ":"))})
                for item in list(getattr(result, "items", []) or []):
                    payload = item.model_dump() if hasattr(item, "model_dump") else dict(item or {})
                    q = _safe_int(payload.get("q"))
                    if q > 0:
                        parsed[q] = payload
                if len(parsed) >= len(batch):
                    break
            except Exception as exc:
                last_error = exc
        if len(parsed) < len(batch):
            stats["failed_batches"] += 1
            reason = f"{type(last_error).__name__}: {last_error}" if last_error else "missing_items"
            print(f"warn=batch_partial_or_failed batch={batch_idx} parsed={len(parsed)}/{len(batch)} reason={reason}")

        for i, item in enumerate(batch, start=1):
            payload = parsed.get(i)
            if not payload:
                stats["skipped_pairs"] += 1
                continue
            score = _clamp_01(payload.get("score"))
            scored.append(
                {
                    "pair": {
                        "pair_id": _clean_text(item.get("pair_id")),
                        "grant_id": _clean_text(item.get("grant_id")),
                        "spec_idx": _safe_int(item.get("spec_idx")),
                        "query_item_id": _clean_text(item.get("query_item_id")),
                        "grant_text": _normalize_ws(item.get("grant_text")),
                        "fac_item_id": _clean_text(item.get("fac_item_id")),
                        "faculty_text": _normalize_ws(item.get("faculty_text")),
                    },
                    "gt_score": float(score),
                }
            )
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                scored=len(scored),
                skipped=stats["skipped_pairs"],
                retries=stats["retries_used"],
                failed=stats["failed_batches"],
            )
    return scored, stats


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CE3 overall coverage ground truth on a deterministic varied test subset with Claude Opus.")
    p.add_argument("--test-input", type=str, default=TEST_INPUT_DEFAULT)
    p.add_argument("--output", type=str, default=OUTPUT_DEFAULT)
    p.add_argument("--model-id", type=str, default="")
    p.add_argument("--allow-model-fallback", action="store_true")
    p.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    p.add_argument("--per-query-cap", type=int, default=DEFAULT_PER_QUERY_CAP)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument("--high-threshold", type=float, default=HIGH_THRESHOLD)
    p.add_argument("--mid-threshold", type=float, default=MID_THRESHOLD)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    test_input = _resolve_path(args.test_input)
    output = _resolve_path(args.output)
    if not test_input.exists():
        raise FileNotFoundError(f"Missing test listwise input: {test_input}")
    if output.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)

    high_threshold = _clamp_01(args.high_threshold, HIGH_THRESHOLD)
    mid_threshold = min(high_threshold, _clamp_01(args.mid_threshold, MID_THRESHOLD))
    max_pairs = max(1, int(args.max_pairs))
    per_query_cap = max(1, int(args.per_query_cap))
    batch_size = max(1, int(args.batch_size))
    max_retries = max(0, int(args.max_retries))
    model_id = _select_model_id(args.model_id, allow_fallback=bool(args.allow_model_fallback))

    candidates = _load_candidate_pairs(test_input, high_threshold=high_threshold, mid_threshold=mid_threshold)
    if not candidates:
        raise RuntimeError(f"No usable candidate pairs loaded from {test_input}")
    selected = _select_varied_subset(candidates, max_pairs=max_pairs, per_query_cap=per_query_cap)

    print(f"test_input={test_input}")
    print(f"output={output}")
    print(f"model_id={model_id}")
    print(f"candidate_pairs={len(candidates)} selected_pairs={len(selected)}")
    print(f"selected_proxy_bands={dict(Counter(_clean_text(x.get('proxy_band')) for x in selected))}")
    print(f"selected_aspect_patterns={dict(Counter(_clean_text(x.get('aspect_pattern')) for x in selected))}")

    chain = _build_chain(model_id)
    scored, llm_stats = _score_with_llm(
        chain=chain,
        pairs=selected,
        batch_size=batch_size,
        max_retries=max_retries,
    )
    _write_jsonl(output, scored)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_input": str(test_input),
        "output": str(output),
        "model_id": model_id,
        "candidate_pairs": int(len(candidates)),
        "selected_pairs": int(len(selected)),
        "written_pairs": int(len(scored)),
        "max_pairs": int(max_pairs),
        "per_query_cap": int(per_query_cap),
        "batch_size": int(batch_size),
        "max_retries": int(max_retries),
        "selected_proxy_bands": dict(Counter(_clean_text(x.get("proxy_band")) for x in selected)),
        "selected_aspect_patterns": dict(Counter(_clean_text(x.get("aspect_pattern")) for x in selected)),
        "llm": llm_stats,
        "elapsed_sec": float(time.time() - started),
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "ce3_overall_ground_truth_done", **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
