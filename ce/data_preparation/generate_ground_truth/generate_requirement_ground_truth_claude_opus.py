from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field


# ==========================================================
# Path bootstrap
# ==========================================================
def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_llm_client, settings

# ==========================================================
# Defaults
# ==========================================================
DOMAIN_TEST_INPUT_DEFAULT = "ce/dataset/splits/llm_distill_domain_listwise_test.jsonl"
METHOD_TEST_INPUT_DEFAULT = "ce/dataset/splits/llm_distill_method_listwise_test.jsonl"
OUTPUT_DEFAULT = "ce/dataset/distill/llm_ground_truth_requirement_common_test_listwise.jsonl"

LLM_BATCH_SIZE_DEFAULT = 12
LLM_MAX_RETRIES_DEFAULT = 2
HIGH_THRESHOLD_DEFAULT = 0.70
MID_THRESHOLD_DEFAULT = 0.30
MAX_QUERY_CHARS = 900
MAX_DOC_CHARS = 700


# ==========================================================
# Prompt + schema
# ==========================================================
class RequirementScoreItem(BaseModel):
    q: int = Field(..., description="1-based task index in this call")
    score: float = Field(..., description="Requirement coverage score in [0,1]")


class RequirementScoreBatch(BaseModel):
    items: List[RequirementScoreItem] = Field(default_factory=list)


REQ_SYSTEM_PROMPT = """
You are a strict requirement-coverage judge for grant requirement matching.

You will receive multiple tasks. Each task has:
- q: task index
- query: requirement text
- candidate: candidate specialization text

Scoring objective:
Return how well the candidate covers the requirement as written.
Coverage includes whether the candidate matches the requirement's intent, constraints, and specificity.

Scoring guide:
- high (>= 0.70): strong and specific coverage of the requirement
- mid (>= 0.30 and < 0.70): partial/adjacent coverage, but incomplete or diluted
- low (< 0.30): weak or no meaningful coverage

Output MUST be exactly one JSON object with this schema:
{
  "items": [
    {"q": <int>, "score": <float 0..1>}
  ]
}

Rules:
- Include exactly one item for each input q.
- No markdown.
- No text outside JSON.
""".strip()

REQ_USER_PROMPT_TEMPLATE = """
Tasks JSON:
{tasks_json}
""".strip()


def _build_chain(model_id: str):
    try:
        from langchain_core.prompts import ChatPromptTemplate
    except Exception as e:
        raise RuntimeError("Missing dependency: langchain-core") from e

    llm = get_llm_client(model_id).build()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REQ_SYSTEM_PROMPT),
            ("human", REQ_USER_PROMPT_TEMPLATE),
        ]
    )
    return prompt | llm.with_structured_output(RequirementScoreBatch)


# ==========================================================
# Helpers
# ==========================================================
def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_ws(value: Any) -> str:
    return " ".join(_clean_text(value).split())


def _resolve_path(value: Any) -> Path:
    p = Path(_clean_text(value)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _safe_int(value: Any, default: int = 0, minimum: int = 0, maximum: int = 2_147_483_647) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _safe_float(value: Any, default: float = 0.0, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _truncate_text(value: str, limit: int) -> str:
    text = _normalize_ws(value)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = _clean_text(raw)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_no} ({type(exc).__name__}: {exc})"
                ) from exc
            if isinstance(obj, dict):
                yield obj


def _band_from_score(score: float, *, high_threshold: float, mid_threshold: float) -> str:
    s = float(score)
    if s >= float(high_threshold):
        return "high"
    if s >= float(mid_threshold):
        return "mid"
    return "low"


def _pair_key(
    *,
    grant_id: str,
    spec_idx: int,
    fac_id: int,
    fac_spec_id: int,
    fac_spec_idx: int,
    doc_text: str,
) -> str:
    if fac_id != 0 or fac_spec_id != 0 or fac_spec_idx != 0:
        return f"{grant_id}::{spec_idx}::{fac_id}::{fac_spec_id}::{fac_spec_idx}"
    # fallback if ids are missing
    return f"{grant_id}::{spec_idx}::text::{_normalize_ws(doc_text).lower()}"


def _load_pairs(path: Path) -> Dict[str, Dict[str, Any]]:
    pair_map: Dict[str, Dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        grant_id = _clean_text(row.get("grant_id"))
        spec_idx = _safe_int(row.get("spec_idx"), default=0, minimum=0, maximum=50_000_000)
        query_text = _normalize_ws(row.get("query_text"))
        if not grant_id or not query_text:
            continue

        for doc in list(row.get("docs") or []):
            if not isinstance(doc, dict):
                continue
            text = _normalize_ws(doc.get("text"))
            if not text:
                continue
            fac_id = _safe_int(doc.get("fac_id"), default=0, minimum=-2_147_483_648, maximum=2_147_483_647)
            fac_spec_id = _safe_int(doc.get("fac_spec_id"), default=0, minimum=-9_223_372_036_854_775_808, maximum=9_223_372_036_854_775_807)
            fac_spec_idx = _safe_int(doc.get("fac_spec_idx"), default=0, minimum=-2_147_483_648, maximum=2_147_483_647)
            key = _pair_key(
                grant_id=grant_id,
                spec_idx=spec_idx,
                fac_id=fac_id,
                fac_spec_id=fac_spec_id,
                fac_spec_idx=fac_spec_idx,
                doc_text=text,
            )
            pair_map[key] = {
                "grant_id": grant_id,
                "spec_idx": int(spec_idx),
                "query_text": query_text,
                "text": text,
                "fac_id": int(fac_id),
                "fac_spec_id": int(fac_spec_id),
                "fac_spec_idx": int(fac_spec_idx),
                "section": _clean_text(doc.get("section")) or "unknown",
                "domain_or_method_teacher_score": _safe_float(
                    doc.get("teacher_score_raw", doc.get("teacher_score", 0.0)),
                    default=0.0,
                    minimum=0.0,
                    maximum=1.0,
                ),
            }
    return pair_map


def _chunks(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    step = max(1, int(size))
    for i in range(0, len(seq), step):
        yield seq[i : i + step]


def _select_model_id(arg_model_id: str) -> str:
    configured = _clean_text(arg_model_id)
    if configured:
        return configured
    # Prefer Opus as requested.
    for candidate in (
        _clean_text(getattr(settings, "opus", "")),
        _clean_text(getattr(settings, "bedrock_claude_opus", "")),
        _clean_text(getattr(settings, "sonnet", "")),
        _clean_text(getattr(settings, "haiku", "")),
    ):
        if candidate:
            return candidate
    raise RuntimeError("No Claude model id found. Set BEDROCK_CLAUDE_OPUS (or pass --model-id).")


def _score_common_pairs(
    *,
    chain: Any,
    pairs: Sequence[Dict[str, Any]],
    batch_size: int,
    max_retries: int,
    high_threshold: float,
    mid_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    total_calls = 0
    retries_used = 0
    fallback_count = 0
    failed_batches = 0

    for batch_idx, batch in enumerate(_chunks(list(pairs), batch_size), start=1):
        tasks = [
            {
                "q": int(i + 1),
                "query": _truncate_text(item.get("query_text", ""), MAX_QUERY_CHARS),
                "candidate": _truncate_text(item.get("text", ""), MAX_DOC_CHARS),
            }
            for i, item in enumerate(batch)
        ]

        parsed_items: Dict[int, Dict[str, Any]] = {}
        last_error: Optional[Exception] = None
        for attempt in range(max(1, int(max_retries))):
            total_calls += 1
            if attempt > 0:
                retries_used += 1
            try:
                out = chain.invoke(
                    {"tasks_json": json.dumps(tasks, ensure_ascii=False, separators=(",", ":"))}
                )
                items = list(getattr(out, "items", []) or [])
                for it in items:
                    payload = it.model_dump() if hasattr(it, "model_dump") else dict(it or {})
                    q = _safe_int(payload.get("q"), default=0, minimum=0, maximum=1_000_000)
                    if q <= 0:
                        continue
                    parsed_items[q] = payload
                break
            except Exception as exc:
                last_error = exc
                continue

        if not parsed_items:
            failed_batches += 1
            err_text = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown"
            print(f"warn=batch_failed idx={batch_idx} reason={err_text}")

        for i, item in enumerate(batch, start=1):
            raw_payload = parsed_items.get(i) or {}
            score = _safe_float(raw_payload.get("score"), default=0.0, minimum=0.0, maximum=1.0)
            band = _band_from_score(score, high_threshold=high_threshold, mid_threshold=mid_threshold)
            if not parsed_items:
                fallback_count += 1

            scored.append(
                {
                    **item,
                    "teacher_score": float(score),
                    "teacher_score_raw": float(score),
                    "target_cluster": str(band),
                }
            )

    stats = {
        "llm_calls": int(total_calls),
        "retries_used": int(retries_used),
        "fallback_pairs": int(fallback_count),
        "failed_batches": int(failed_batches),
    }
    return scored, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate requirement-coverage teacher ground-truth scores with Claude Opus "
            "from common test pairs between domain/method listwise splits."
        )
    )
    p.add_argument("--domain-test-input", type=str, default=DOMAIN_TEST_INPUT_DEFAULT)
    p.add_argument("--method-test-input", type=str, default=METHOD_TEST_INPUT_DEFAULT)
    p.add_argument("--output", type=str, default=OUTPUT_DEFAULT)
    p.add_argument("--model-id", type=str, default="")
    p.add_argument("--batch-size", type=int, default=LLM_BATCH_SIZE_DEFAULT)
    p.add_argument("--max-retries", type=int, default=LLM_MAX_RETRIES_DEFAULT)
    p.add_argument("--max-pairs", type=int, default=0, help="Optional cap for common pairs (0 = all).")
    p.add_argument("--high-threshold", type=float, default=HIGH_THRESHOLD_DEFAULT)
    p.add_argument("--mid-threshold", type=float, default=MID_THRESHOLD_DEFAULT)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    domain_path = _resolve_path(args.domain_test_input)
    method_path = _resolve_path(args.method_test_input)
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not domain_path.exists():
        raise RuntimeError(f"Domain test split not found: {domain_path}")
    if not method_path.exists():
        raise RuntimeError(f"Method test split not found: {method_path}")

    high_threshold = _safe_float(args.high_threshold, default=HIGH_THRESHOLD_DEFAULT, minimum=0.0, maximum=1.0)
    mid_threshold = _safe_float(args.mid_threshold, default=MID_THRESHOLD_DEFAULT, minimum=0.0, maximum=1.0)
    if mid_threshold > high_threshold:
        mid_threshold = high_threshold

    model_id = _select_model_id(_clean_text(args.model_id))
    batch_size = _safe_int(args.batch_size, default=LLM_BATCH_SIZE_DEFAULT, minimum=1, maximum=128)
    max_retries = _safe_int(args.max_retries, default=LLM_MAX_RETRIES_DEFAULT, minimum=1, maximum=8)
    max_pairs = _safe_int(args.max_pairs, default=0, minimum=0, maximum=50_000_000)

    print(f"domain_test_input={domain_path}")
    print(f"method_test_input={method_path}")
    print(f"output={output_path}")
    print(f"model_id={model_id}")

    domain_pairs = _load_pairs(domain_path)
    method_pairs = _load_pairs(method_path)

    common_keys = sorted(set(domain_pairs.keys()) & set(method_pairs.keys()))
    if max_pairs > 0:
        common_keys = common_keys[:max_pairs]

    common_rows: List[Dict[str, Any]] = []
    for key in common_keys:
        d = domain_pairs[key]
        m = method_pairs[key]
        common_rows.append(
            {
                "grant_id": d["grant_id"],
                "spec_idx": int(d["spec_idx"]),
                "query_text": d["query_text"],
                "text": d["text"],
                "fac_id": int(d["fac_id"]),
                "fac_spec_id": int(d["fac_spec_id"]),
                "fac_spec_idx": int(d["fac_spec_idx"]),
                "section": d["section"],
                "domain_teacher_score_raw": float(d["domain_or_method_teacher_score"]),
                "method_teacher_score_raw": float(m["domain_or_method_teacher_score"]),
            }
        )

    if not common_rows:
        raise RuntimeError(
            "No common test pairs found between domain/method listwise test splits. "
            "Check split files and pair ids/text alignment."
        )

    chain = _build_chain(model_id)
    scored_rows, llm_stats = _score_common_pairs(
        chain=chain,
        pairs=common_rows,
        batch_size=batch_size,
        max_retries=max_retries,
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
    )

    grouped: Dict[Tuple[str, int], Dict[str, Any]] = {}
    by_query_docs: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        qk = (_clean_text(row.get("grant_id")), _safe_int(row.get("spec_idx"), default=0, minimum=0, maximum=50_000_000))
        by_query_docs[qk].append(row)
        if qk not in grouped:
            grouped[qk] = {
                "grant_id": qk[0],
                "spec_idx": int(qk[1]),
                "query_text": _normalize_ws(row.get("query_text")),
            }

    rows_out: List[Dict[str, Any]] = []
    for qk in sorted(grouped.keys(), key=lambda x: (x[0], int(x[1]))):
        base = grouped[qk]
        docs_sorted = sorted(
            by_query_docs[qk],
            key=lambda x: float(x.get("teacher_score_raw", 0.0)),
            reverse=True,
        )
        docs: List[Dict[str, Any]] = []
        for rank_idx, doc in enumerate(docs_sorted, start=1):
            docs.append(
                {
                    "rank": int(rank_idx),
                    "teacher_score": float(doc.get("teacher_score", 0.0)),
                    "teacher_score_raw": float(doc.get("teacher_score_raw", 0.0)),
                    "target_cluster": _clean_text(doc.get("target_cluster")) or "low",
                    "fac_id": int(doc.get("fac_id") or 0),
                    "fac_spec_id": int(doc.get("fac_spec_id") or 0),
                    "fac_spec_idx": int(doc.get("fac_spec_idx") or 0),
                }
            )

        rows_out.append(
            {
                "grant_id": base["grant_id"],
                "spec_idx": int(base["spec_idx"]),
                "source": "common_test_pairs_domain_method",
                "score_model_id": model_id,
                "docs": docs,
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_docs = int(sum(len(list(r.get("docs") or [])) for r in rows_out))
    elapsed_meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "domain_test_input": str(domain_path),
        "method_test_input": str(method_path),
        "output": str(output_path),
        "model_id": model_id,
        "queries": int(len(rows_out)),
        "common_pairs": int(len(common_rows)),
        "docs_written": int(total_docs),
        "high_threshold": float(high_threshold),
        "mid_threshold": float(mid_threshold),
        "batch_size": int(batch_size),
        "max_retries": int(max_retries),
        **llm_stats,
    }
    print(json.dumps(elapsed_meta, ensure_ascii=False, indent=2))
    print("done=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
