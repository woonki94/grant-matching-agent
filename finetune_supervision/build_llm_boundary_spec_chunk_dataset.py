from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import get_llm_client, settings
from utils.thread_pool import build_thread_local_getter, parallel_map, resolve_pool_size


class BoundaryQueryRow(BaseModel):
    q: int = Field(..., description="1-based query index within the batch")
    first_negative_rank: int = Field(
        ...,
        description="1-based rank position where negatives begin. Use 0 if no clear negatives exist.",
    )


class BoundaryQueryOut(BaseModel):
    items: List[BoundaryQueryRow] = Field(default_factory=list)


BOUNDARY_SYSTEM_PROMPT = (
    "You receive a query and ranked candidate chunks from best to worst.\n"
    "Return the first rank position where candidates should be treated as negatives.\n"
    "\n"
    "Output must be JSON only with key 'items'.\n"
    "Each item must contain:\n"
    "- q: query id from input\n"
    "- first_negative_rank: integer boundary\n"
    "\n"
    "Interpretation:\n"
    "- positives are ranks 1..(first_negative_rank-2)\n"
    "- rank (first_negative_rank-1) is uncertain and should be excluded\n"
    "- negatives are ranks first_negative_rank..end\n"
    "\n"
    "True-positive definition (strict):\n"
    "- Directly supports the exact query capability/objective.\n"
    "- Evidence should be specific to evaluation/measurement/program capability in the same intent.\n"
    "- Broadly related domain overlap alone is NOT sufficient.\n"
    "\n"
    "True-negative definition:\n"
    "- Not directly supporting the query objective, even if in a nearby domain.\n"
    "- CV/profile/contact/publication-list snippets without clear query support are negatives.\n"
    "\n"
    "Rules:\n"
    "- first_negative_rank is 1-based.\n"
    "- Use 0 only if the list has no clear negatives.\n"
    "- Be conservative: if uncertain whether items are true positives, start negatives earlier.\n"
    "- Do not treat generic bios, addresses, citation dumps, or directory text as positives.\n"
    "- No explanation and no extra keys."
)

BOUNDARY_HUMAN_PROMPT = "Tasks JSON:\n{tasks_json}"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_limit(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out != out or out in (float("inf"), float("-inf")):
        return float(default)
    return float(out)


def _trim_text(value: Any, *, max_chars: int, max_words: int) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    if int(max_words) > 0:
        text = " ".join(text.split()[: int(max_words)])
    if int(max_chars) > 0 and len(text) > int(max_chars):
        text = text[: int(max_chars)].rstrip()
    return text


def _clip_for_log(value: Any, max_chars: int) -> str:
    text = _clean_text(value)
    cap = max(80, int(max_chars))
    if len(text) <= cap:
        return text
    return text[:cap].rstrip() + " ...<truncated>"


_JUNK_PATTERNS = [
    r"the system can't perform the operation now",
    r"\bcited by\b",
    r"\bverified email at\b",
    r"\bgoogle scholar\b",
    r"\boffice location\b",
    r"\bshipping address\b",
    r"\bfax:\b",
    r"\bphone:\b",
    r"\ball since \d{4} citations\b",
    r"\bcurriculum vitae\b",
]


def _is_junk_candidate_text(text: str) -> bool:
    t = _clean_text(text).lower()
    if not t:
        return True
    if len(t) < 24:
        return True
    hit = 0
    for pat in _JUNK_PATTERNS:
        if re.search(pat, t):
            hit += 1
    # Two+ strong junk signals -> drop.
    if hit >= 2:
        return True
    # Extremely list-like profile blobs with many separators are noisy.
    if t.count("|") >= 10:
        return True
    return False


def _build_boundary_chain(model_id: str):
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", BOUNDARY_SYSTEM_PROMPT),
            ("human", BOUNDARY_HUMAN_PROMPT),
        ]
    )
    llm = get_llm_client(model_id).build()
    return prompt | llm.with_structured_output(BoundaryQueryOut)


def _resolve_latest_listwise(dataset_dir: Path) -> Path:
    cands = sorted(dataset_dir.glob("*_listwise_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No listwise dataset found under {dataset_dir}")
    return cands[0].resolve()


def _load_listwise_rows(
    path: Path,
    *,
    max_queries: int,
    min_candidates: int,
    candidate_min_words: int,
    candidate_max_words: int,
    drop_junk_candidates: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    safe_max = _safe_limit(max_queries, default=0, minimum=0, maximum=10_000_000)
    safe_min_candidates = _safe_limit(min_candidates, default=4, minimum=2, maximum=128)
    safe_min_words = _safe_limit(candidate_min_words, default=4, minimum=1, maximum=100)
    safe_max_words = _safe_limit(candidate_max_words, default=260, minimum=16, maximum=3000)
    if safe_max_words < safe_min_words:
        safe_max_words = safe_min_words

    out: List[Dict[str, Any]] = []
    skipped = Counter()

    with path.open("r", encoding="utf-8") as f:
        for row_idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            if safe_max > 0 and len(out) >= safe_max:
                break

            try:
                item = dict(json.loads(line) or {})
            except Exception:
                skipped["parse_error"] += 1
                continue

            query = _clean_text(item.get("query"))
            query_group = _clean_text(item.get("query_group")) or f"query::{len(out)}"
            candidates = list(item.get("candidates") or [])
            ranking = [
                int(x)
                for x in list(item.get("ranking") or [])
                if int(x) > 0
            ]

            if not query or not candidates or not ranking:
                skipped["bad_shape"] += 1
                continue

            by_i: Dict[int, str] = {}
            for c in candidates:
                idx = int(c.get("i") or 0)
                text = _clean_text(c.get("t") or c.get("text"))
                if idx <= 0 or not text:
                    continue
                wc = len(text.split())
                if wc < safe_min_words:
                    skipped["candidate_too_short"] += 1
                    continue
                if wc > safe_max_words:
                    skipped["candidate_too_long"] += 1
                    continue
                if bool(drop_junk_candidates) and _is_junk_candidate_text(text):
                    skipped["candidate_junk"] += 1
                    continue
                by_i[idx] = text

            ordered_indices = [i for i in ranking if i in by_i]
            if len(ordered_indices) < safe_min_candidates:
                skipped["too_few_candidates"] += 1
                continue

            out.append(
                {
                    "row_idx": int(row_idx),
                    "query": query,
                    "query_group": query_group,
                    "candidates": [{"i": int(i), "t": by_i[int(i)]} for i in ordered_indices],
                    "ranking": [int(i) for i in ordered_indices],
                    "label_source": _clean_text(item.get("label_source")) or "unknown",
                }
            )

    meta = {
        "input_path": str(path),
        "queries_loaded": int(len(out)),
        "rows_skipped": int(sum(skipped.values())),
        "skip_counts": dict(skipped),
        "candidate_filter": {
            "min_words": int(safe_min_words),
            "max_words": int(safe_max_words),
            "drop_junk_candidates": bool(drop_junk_candidates),
        },
    }
    return out, meta


def _label_boundaries_with_llm(
    rows: Sequence[Dict[str, Any]],
    *,
    llm_model: str,
    llm_batch_size: int,
    llm_max_workers: int,
    llm_max_retries: int,
    llm_query_max_chars: int,
    llm_query_max_words: int,
    llm_candidate_max_chars: int,
    llm_candidate_max_words: int,
    max_ranked_candidates_for_prompt: int,
    allow_llm_fallback: bool,
    fallback_boundary_rank: int,
    log_llm_io: bool,
    llm_log_max_chars: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    query_rows = list(rows or [])
    total_queries = len(query_rows)
    if total_queries <= 0:
        return [], {
            "queries_total": 0,
            "queries_labeled": 0,
            "batches": 0,
            "llm_batch_size": int(llm_batch_size),
            "llm_max_workers": int(llm_max_workers),
            "llm_model": str(llm_model),
            "llm_calls": 0,
            "retries_used": 0,
            "fallback_query_count": 0,
            "fallback_query_ratio": 0.0,
            "failed_query_count": 0,
            "failed_batches": 0,
        }

    batch_size = max(1, int(llm_batch_size))
    max_attempts = max(1, int(llm_max_retries))
    prompt_rank_cap = max(2, int(max_ranked_candidates_for_prompt))
    total_batches = (total_queries + batch_size - 1) // batch_size
    safe_workers = resolve_pool_size(
        max_workers=max(1, int(llm_max_workers)),
        task_count=max(1, int(total_batches)),
    )

    payloads: List[Dict[str, Any]] = []
    batch_num = 0
    for start in range(0, total_queries, batch_size):
        batch_num += 1
        batch = query_rows[start : start + batch_size]

        tasks: List[Dict[str, Any]] = []
        max_rank_map: Dict[int, int] = {}
        for q_local, row in enumerate(batch, start=1):
            ranked = list(row.get("ranking") or [])
            ranked_cands = list(row.get("candidates") or [])
            rank_to_text: Dict[int, str] = {}
            for rank_pos, idx in enumerate(ranked, start=1):
                by_idx = next((c for c in ranked_cands if int(c.get("i") or 0) == int(idx)), None)
                if by_idx is None:
                    continue
                rank_to_text[int(rank_pos)] = _clean_text(by_idx.get("t"))

            n = len(rank_to_text)
            max_rank_map[q_local] = int(n)
            cap_n = min(n, prompt_rank_cap)
            task_ranked = [
                {
                    "r": int(r),
                    "t": _trim_text(
                        rank_to_text[r],
                        max_chars=int(llm_candidate_max_chars),
                        max_words=int(llm_candidate_max_words),
                    ),
                }
                for r in range(1, cap_n + 1)
                if _clean_text(rank_to_text.get(r))
            ]
            tasks.append(
                {
                    "q": int(q_local),
                    "query": _trim_text(
                        row.get("query"),
                        max_chars=int(llm_query_max_chars),
                        max_words=int(llm_query_max_words),
                    ),
                    "ranked_candidates": task_ranked,
                }
            )

        payloads.append(
            {
                "batch_number": int(batch_num),
                "rows": batch,
                "max_rank_map": max_rank_map,
                "tasks_json": json.dumps(tasks, ensure_ascii=False),
            }
        )

    get_chain = build_thread_local_getter(lambda: _build_boundary_chain(llm_model))
    lock = threading.Lock()
    call_counter = {"value": 0}
    retry_counter = {"value": 0}

    def _fallback_rows(batch_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out_local: List[Dict[str, Any]] = []
        for row in list(batch_rows or []):
            ranked = list(row.get("ranking") or [])
            n = len(ranked)
            if n <= 0:
                continue
            b = min(max(1, int(fallback_boundary_rank)), n)
            row_out = dict(row)
            row_out["first_negative_rank"] = int(b)
            row_out["boundary_label_source"] = "fallback_boundary_rank"
            out_local.append(row_out)
        return out_local

    def _run_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        batch_number = int(payload.get("batch_number") or 0)
        batch_rows = list(payload.get("rows") or [])
        max_rank_map = dict(payload.get("max_rank_map") or {})
        tasks_json = _clean_text(payload.get("tasks_json"))
        last_error: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            with lock:
                call_counter["value"] += 1
                call_id = int(call_counter["value"])

            if bool(log_llm_io):
                print(
                    f"[llm_boundary_call] count={call_id} batch={batch_number}/{total_batches} "
                    f"attempt={attempt}/{max_attempts} payload="
                    f"{_clip_for_log(tasks_json, llm_log_max_chars)}",
                    file=sys.stderr,
                )

            try:
                result = get_chain().invoke({"tasks_json": tasks_json})
                items = list(getattr(result, "items", []) or [])
                by_q: Dict[int, int] = {}
                for item in items:
                    q = int(getattr(item, "q", 0) or 0)
                    boundary = int(getattr(item, "first_negative_rank", 0) or 0)
                    if q <= 0:
                        continue
                    by_q[q] = int(boundary)

                out_rows: List[Dict[str, Any]] = []
                for q_local, row in enumerate(batch_rows, start=1):
                    n = int(max_rank_map.get(q_local) or 0)
                    if n <= 0:
                        continue
                    if q_local not in by_q:
                        raise RuntimeError(f"missing boundary for query_local={q_local}")
                    b = int(by_q[q_local])
                    if b < 0 or b > n:
                        raise RuntimeError(
                            f"boundary out of range query_local={q_local}: boundary={b}, n={n}"
                        )
                    row_out = dict(row)
                    row_out["first_negative_rank"] = int(b)
                    row_out["boundary_label_source"] = "llm_boundary"
                    out_rows.append(row_out)
                return {
                    "rows": out_rows,
                    "fallback_count": 0,
                    "failed_count": 0,
                    "failed_batch": False,
                }
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                print(
                    f"[llm_boundary_error] batch={batch_number}/{total_batches} "
                    f"attempt={attempt}/{max_attempts} error={last_error}",
                    file=sys.stderr,
                )
                if attempt < max_attempts:
                    with lock:
                        retry_counter["value"] += 1
                    continue
                break

        if bool(allow_llm_fallback):
            fb = _fallback_rows(batch_rows)
            return {
                "rows": fb,
                "fallback_count": int(len(fb)),
                "failed_count": 0,
                "failed_batch": True,
            }
        return {
            "rows": [],
            "fallback_count": 0,
            "failed_count": int(len(batch_rows)),
            "failed_batch": True,
        }

    results = parallel_map(
        payloads,
        max_workers=safe_workers,
        run_item=_run_payload,
    )

    labeled: List[Dict[str, Any]] = []
    fallback_count = 0
    failed_count = 0
    failed_batches = 0
    for result in list(results or []):
        labeled.extend(list(result.get("rows") or []))
        fallback_count += int(result.get("fallback_count") or 0)
        failed_count += int(result.get("failed_count") or 0)
        if bool(result.get("failed_batch")):
            failed_batches += 1

    meta = {
        "queries_total": int(total_queries),
        "queries_labeled": int(len(labeled)),
        "batches": int(total_batches),
        "llm_batch_size": int(batch_size),
        "llm_max_workers": int(safe_workers),
        "llm_model": str(llm_model),
        "llm_calls": int(call_counter["value"]),
        "retries_used": int(retry_counter["value"]),
        "fallback_query_count": int(fallback_count),
        "fallback_query_ratio": (0.0 if total_queries <= 0 else float(fallback_count) / float(total_queries)),
        "failed_query_count": int(failed_count),
        "failed_batches": int(failed_batches),
    }
    return labeled, meta


def _derive_boundary_buckets(
    row: Dict[str, Any],
    *,
    max_positive_count: int,
) -> Dict[str, Any]:
    ranked = [int(x) for x in list(row.get("ranking") or []) if int(x) > 0]
    boundary = int(row.get("first_negative_rank") or 0)
    n = len(ranked)
    safe_max_pos = _safe_limit(max_positive_count, default=2, minimum=1, maximum=32)

    positive_ranks: List[int] = []
    uncertain_ranks: List[int] = []
    negative_ranks: List[int] = []

    if n > 0 and boundary > 0:
        pos_end = min(boundary - 2, safe_max_pos)
        if pos_end >= 1:
            positive_ranks = list(range(1, pos_end + 1))
        u = boundary - 1
        if 1 <= u <= n:
            uncertain_ranks = [int(u)]
        if boundary <= n:
            negative_ranks = list(range(boundary, n + 1))

    return {
        "positive_indices": [int(ranked[r - 1]) for r in positive_ranks if 1 <= r <= n],
        "uncertain_indices": [int(ranked[r - 1]) for r in uncertain_ranks if 1 <= r <= n],
        "negative_indices": [int(ranked[r - 1]) for r in negative_ranks if 1 <= r <= n],
        "positive_ranks": [int(x) for x in positive_ranks],
        "uncertain_ranks": [int(x) for x in uncertain_ranks],
        "negative_ranks": [int(x) for x in negative_ranks],
    }


def _build_boundary_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    drop_no_positive: bool,
    drop_no_negative: bool,
    boundary_min_rank: int,
    boundary_max_rank: int,
    max_positive_count: int,
    min_negative_count: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    counts = Counter()
    safe_boundary_min = _safe_limit(boundary_min_rank, default=3, minimum=1, maximum=128)
    safe_boundary_max = _safe_limit(boundary_max_rank, default=6, minimum=safe_boundary_min, maximum=256)
    safe_min_negative_count = _safe_limit(min_negative_count, default=3, minimum=1, maximum=128)

    for row in list(rows or []):
        row_out = dict(row)
        boundary = int(row_out.get("first_negative_rank") or 0)
        n = len(list(row_out.get("ranking") or []))
        # Keep only boundaries in a practical confidence range.
        if boundary <= 0 or n <= 0:
            counts["dropped_invalid_boundary"] += 1
            continue
        if boundary < safe_boundary_min or boundary > min(safe_boundary_max, n):
            counts["dropped_boundary_out_of_range"] += 1
            continue

        derived = _derive_boundary_buckets(
            row_out,
            max_positive_count=int(max_positive_count),
        )
        row_out.update(derived)

        positives = list(row_out.get("positive_indices") or [])
        negatives = list(row_out.get("negative_indices") or [])

        if bool(drop_no_positive) and (not positives):
            counts["dropped_no_positive"] += 1
            continue
        if bool(drop_no_negative) and (not negatives):
            counts["dropped_no_negative"] += 1
            continue
        if len(negatives) < safe_min_negative_count:
            counts["dropped_too_few_negatives"] += 1
            continue

        row_out["label_source"] = "llm_boundary_labels"
        out.append(row_out)

    meta = {
        "queries_total": int(len(list(rows or []))),
        "queries_kept": int(len(out)),
        "queries_dropped": int(len(list(rows or [])) - len(out)),
        "dropped_no_positive": int(counts.get("dropped_no_positive", 0)),
        "dropped_no_negative": int(counts.get("dropped_no_negative", 0)),
        "dropped_invalid_boundary": int(counts.get("dropped_invalid_boundary", 0)),
        "dropped_boundary_out_of_range": int(counts.get("dropped_boundary_out_of_range", 0)),
        "dropped_too_few_negatives": int(counts.get("dropped_too_few_negatives", 0)),
        "boundary_min_rank": int(safe_boundary_min),
        "boundary_max_rank": int(safe_boundary_max),
        "max_positive_count": int(max_positive_count),
        "min_negative_count": int(safe_min_negative_count),
    }
    return out, meta


def _to_pair_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    hard_neg_window_size: int,
    hard_neg_weight: float,
    easy_neg_weight: float,
    max_pairs_per_query: int,
    max_pairs_total: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(int(seed))
    safe_hard_window = max(1, int(hard_neg_window_size))
    safe_max_per_query = max(1, int(max_pairs_per_query))
    safe_max_total = max(0, int(max_pairs_total))
    safe_hard_w = float(max(0.0, hard_neg_weight))
    safe_easy_w = float(max(0.0, easy_neg_weight))

    out: List[Dict[str, Any]] = []
    counts = Counter()

    for row in list(rows or []):
        query = _clean_text(row.get("query"))
        query_group = _clean_text(row.get("query_group"))
        ranked = [int(x) for x in list(row.get("ranking") or []) if int(x) > 0]
        candidates = list(row.get("candidates") or [])
        boundary = int(row.get("first_negative_rank") or 0)
        positives = [int(x) for x in list(row.get("positive_indices") or []) if int(x) > 0]
        negatives = [int(x) for x in list(row.get("negative_indices") or []) if int(x) > 0]
        source = _clean_text(row.get("label_source")) or "llm_boundary_labels"

        by_i: Dict[int, str] = {}
        for c in candidates:
            i = int(c.get("i") or 0)
            t = _clean_text(c.get("t"))
            if i > 0 and t:
                by_i[i] = t

        rank_pos = {idx: pos for pos, idx in enumerate(ranked, start=1)}
        pair_rows_for_query: List[Dict[str, Any]] = []
        for p in positives:
            ptxt = _clean_text(by_i.get(int(p)))
            if not ptxt:
                continue
            for n in negatives:
                ntxt = _clean_text(by_i.get(int(n)))
                if not ntxt or ntxt == ptxt:
                    continue
                neg_rank = int(rank_pos.get(int(n)) or 0)
                neg_type = "easy_negative"
                pair_weight = safe_easy_w
                neg_teacher_score = 0.0
                if boundary > 0 and neg_rank >= boundary and neg_rank < (boundary + safe_hard_window):
                    neg_type = "hard_negative"
                    pair_weight = safe_hard_w
                    neg_teacher_score = 0.2
                teacher_margin = max(0.0, 1.0 - float(neg_teacher_score))
                pair_rows_for_query.append(
                    {
                        "query": query,
                        "positive": ptxt,
                        "negative": ntxt,
                        "negative_type": neg_type,
                        "pair_weight": float(pair_weight),
                        "query_group": query_group,
                        "label_source": source,
                        "positive_teacher_score": 1.0,
                        "negative_teacher_score": float(neg_teacher_score),
                        "teacher_margin": float(teacher_margin),
                    }
                )

        if len(pair_rows_for_query) > safe_max_per_query:
            # Keep harder rows first, then random from the remainder.
            hard = [r for r in pair_rows_for_query if _clean_text(r.get("negative_type")) == "hard_negative"]
            easy = [r for r in pair_rows_for_query if _clean_text(r.get("negative_type")) != "hard_negative"]
            picked = list(hard[:safe_max_per_query])
            if len(picked) < safe_max_per_query and easy:
                rng.shuffle(easy)
                picked.extend(easy[: safe_max_per_query - len(picked)])
            pair_rows_for_query = picked[:safe_max_per_query]

        for r in pair_rows_for_query:
            out.append(r)
            if _clean_text(r.get("negative_type")) == "hard_negative":
                counts["hard_pairs"] += 1
            else:
                counts["easy_pairs"] += 1

    if safe_max_total > 0 and len(out) > safe_max_total:
        rng.shuffle(out)
        out = out[:safe_max_total]

    meta = {
        "rows_output": int(len(out)),
        "hard_pairs": int(counts.get("hard_pairs", 0)),
        "easy_pairs": int(counts.get("easy_pairs", 0)),
    }
    return out, meta


def _save_jsonl(
    rows: Sequence[Dict[str, Any]],
    *,
    output_dir: Path,
    stem: str,
    meta: Dict[str, Any],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{stem}.jsonl"
    meta_path = output_dir / f"{stem}.meta.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in list(rows or []):
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    payload = dict(meta or {})
    payload["row_count"] = int(len(list(rows or [])))
    payload["jsonl_path"] = str(jsonl_path)
    payload["meta_path"] = str(meta_path)
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"jsonl_path": str(jsonl_path), "meta_path": str(meta_path)}


def build_dataset(
    *,
    input_listwise_jsonl: str,
    output_dir: str,
    output_prefix: str,
    max_queries: int,
    min_candidates: int,
    candidate_min_words: int,
    candidate_max_words: int,
    drop_junk_candidates: bool,
    llm_model: str,
    llm_batch_size: int,
    llm_max_workers: int,
    llm_max_retries: int,
    llm_query_max_chars: int,
    llm_query_max_words: int,
    llm_candidate_max_chars: int,
    llm_candidate_max_words: int,
    max_ranked_candidates_for_prompt: int,
    allow_llm_fallback: bool,
    fallback_boundary_rank: int,
    drop_no_positive: bool,
    drop_no_negative: bool,
    boundary_min_rank: int,
    boundary_max_rank: int,
    max_positive_count: int,
    min_negative_count: int,
    hard_neg_window_size: int,
    hard_neg_weight: float,
    easy_neg_weight: float,
    max_pairs_per_query: int,
    max_pairs_total: int,
    seed: int,
    log_llm_io: bool,
    llm_log_max_chars: int,
) -> Dict[str, Any]:
    input_path = Path(_clean_text(input_listwise_jsonl)).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    safe_model = _clean_text(llm_model or settings.haiku)
    if not safe_model:
        raise ValueError("No LLM model configured. Set --llm-model or BEDROCK_CLAUDE_HAIKU.")

    loaded_rows, load_meta = _load_listwise_rows(
        input_path,
        max_queries=int(max_queries),
        min_candidates=int(min_candidates),
        candidate_min_words=int(candidate_min_words),
        candidate_max_words=int(candidate_max_words),
        drop_junk_candidates=bool(drop_junk_candidates),
    )
    if not loaded_rows:
        raise RuntimeError("No valid listwise rows loaded from input.")

    labeled_rows, llm_meta = _label_boundaries_with_llm(
        loaded_rows,
        llm_model=safe_model,
        llm_batch_size=int(llm_batch_size),
        llm_max_workers=int(llm_max_workers),
        llm_max_retries=int(llm_max_retries),
        llm_query_max_chars=int(llm_query_max_chars),
        llm_query_max_words=int(llm_query_max_words),
        llm_candidate_max_chars=int(llm_candidate_max_chars),
        llm_candidate_max_words=int(llm_candidate_max_words),
        max_ranked_candidates_for_prompt=int(max_ranked_candidates_for_prompt),
        allow_llm_fallback=bool(allow_llm_fallback),
        fallback_boundary_rank=int(fallback_boundary_rank),
        log_llm_io=bool(log_llm_io),
        llm_log_max_chars=int(llm_log_max_chars),
    )
    if not labeled_rows:
        raise RuntimeError("No rows labeled by boundary LLM step.")

    boundary_rows, boundary_meta = _build_boundary_rows(
        labeled_rows,
        drop_no_positive=bool(drop_no_positive),
        drop_no_negative=bool(drop_no_negative),
        boundary_min_rank=int(boundary_min_rank),
        boundary_max_rank=int(boundary_max_rank),
        max_positive_count=int(max_positive_count),
        min_negative_count=int(min_negative_count),
    )
    if not boundary_rows:
        raise RuntimeError("No boundary rows left after filtering.")

    pair_rows, pair_meta = _to_pair_rows(
        boundary_rows,
        hard_neg_window_size=int(hard_neg_window_size),
        hard_neg_weight=float(hard_neg_weight),
        easy_neg_weight=float(easy_neg_weight),
        max_pairs_per_query=int(max_pairs_per_query),
        max_pairs_total=int(max_pairs_total),
        seed=int(seed),
    )
    if not pair_rows:
        raise RuntimeError("No pair rows derived from boundary labels.")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_prefix = _clean_text(output_prefix) or "spec_chunk_boundary"
    out_dir = Path(_clean_text(output_dir)).expanduser().resolve()

    boundary_paths = _save_jsonl(
        boundary_rows,
        output_dir=out_dir,
        stem=f"{safe_prefix}_listwise_boundary_{ts}",
        meta={
            "kind": "spec_chunk_boundary_listwise",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_listwise_jsonl": str(input_path),
            "load_meta": load_meta,
            "llm": llm_meta,
            "boundary_meta": boundary_meta,
        },
    )
    pair_paths = _save_jsonl(
        pair_rows,
        output_dir=out_dir,
        stem=f"{safe_prefix}_pairs_boundary_{ts}",
        meta={
            "kind": "spec_chunk_boundary_pairs",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_listwise_jsonl": str(input_path),
            "load_meta": load_meta,
            "llm": llm_meta,
            "boundary_meta": boundary_meta,
            "pair_meta": pair_meta,
        },
    )

    return {
        "kind": "spec_chunk_boundary_dataset",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_listwise_jsonl": str(input_path),
        "params": {
            "max_queries": int(max_queries),
            "min_candidates": int(min_candidates),
            "candidate_min_words": int(candidate_min_words),
            "candidate_max_words": int(candidate_max_words),
            "drop_junk_candidates": bool(drop_junk_candidates),
            "llm_model": safe_model,
            "llm_batch_size": int(llm_batch_size),
            "llm_max_workers": int(llm_max_workers),
            "llm_max_retries": int(llm_max_retries),
            "llm_query_max_chars": int(llm_query_max_chars),
            "llm_query_max_words": int(llm_query_max_words),
            "llm_candidate_max_chars": int(llm_candidate_max_chars),
            "llm_candidate_max_words": int(llm_candidate_max_words),
            "max_ranked_candidates_for_prompt": int(max_ranked_candidates_for_prompt),
            "allow_llm_fallback": bool(allow_llm_fallback),
            "fallback_boundary_rank": int(fallback_boundary_rank),
            "drop_no_positive": bool(drop_no_positive),
            "drop_no_negative": bool(drop_no_negative),
            "boundary_min_rank": int(boundary_min_rank),
            "boundary_max_rank": int(boundary_max_rank),
            "max_positive_count": int(max_positive_count),
            "min_negative_count": int(min_negative_count),
            "hard_neg_window_size": int(hard_neg_window_size),
            "hard_neg_weight": float(hard_neg_weight),
            "easy_neg_weight": float(easy_neg_weight),
            "max_pairs_per_query": int(max_pairs_per_query),
            "max_pairs_total": int(max_pairs_total),
            "seed": int(seed),
            "log_llm_io": bool(log_llm_io),
            "llm_log_max_chars": int(llm_log_max_chars),
        },
        "load_meta": load_meta,
        "llm": llm_meta,
        "boundary_meta": boundary_meta,
        "pair_meta": pair_meta,
        "output": {
            "boundary_listwise_jsonl_path": str(boundary_paths.get("jsonl_path", "")),
            "boundary_listwise_meta_path": str(boundary_paths.get("meta_path", "")),
            "boundary_pairs_jsonl_path": str(pair_paths.get("jsonl_path", "")),
            "boundary_pairs_meta_path": str(pair_paths.get("meta_path", "")),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build spec->chunk supervision dataset by asking LLM for the first-negative boundary "
            "on existing ranked listwise rows."
        )
    )
    parser.add_argument("--input-listwise-jsonl", type=str, default="")
    parser.add_argument("--dataset-dir", type=str, default="finetune_supervision/dataset")
    parser.add_argument("--output-dir", type=str, default="finetune_supervision/dataset")
    parser.add_argument("--output-prefix", type=str, default="spec_chunk_boundary")
    parser.add_argument("--max-queries", type=int, default=0, help="0 means all")
    parser.add_argument("--min-candidates", type=int, default=4)
    parser.add_argument("--candidate-min-words", type=int, default=4)
    parser.add_argument("--candidate-max-words", type=int, default=260)
    parser.add_argument("--drop-junk-candidates", dest="drop_junk_candidates", action="store_true")
    parser.add_argument("--keep-junk-candidates", dest="drop_junk_candidates", action="store_false")
    parser.set_defaults(drop_junk_candidates=True)

    parser.add_argument("--llm-model", type=str, default=(settings.haiku or "").strip())
    parser.add_argument("--llm-batch-size", type=int, default=16)
    parser.add_argument("--llm-max-workers", type=int, default=8)
    parser.add_argument("--llm-max-retries", type=int, default=2)
    parser.add_argument("--llm-query-max-chars", type=int, default=220)
    parser.add_argument("--llm-query-max-words", type=int, default=40)
    parser.add_argument("--llm-candidate-max-chars", type=int, default=220)
    parser.add_argument("--llm-candidate-max-words", type=int, default=40)
    parser.add_argument("--max-ranked-candidates-for-prompt", type=int, default=16)
    parser.add_argument("--allow-llm-fallback", action="store_true")
    parser.add_argument("--fallback-boundary-rank", type=int, default=3)
    parser.add_argument("--log-llm-io", action="store_true")
    parser.add_argument("--llm-log-max-chars", type=int, default=1200)

    parser.add_argument("--drop-no-positive", dest="drop_no_positive", action="store_true")
    parser.add_argument("--keep-no-positive", dest="drop_no_positive", action="store_false")
    parser.set_defaults(drop_no_positive=True)

    parser.add_argument("--drop-no-negative", dest="drop_no_negative", action="store_true")
    parser.add_argument("--keep-no-negative", dest="drop_no_negative", action="store_false")
    parser.set_defaults(drop_no_negative=True)

    parser.add_argument("--boundary-min-rank", type=int, default=3)
    parser.add_argument("--boundary-max-rank", type=int, default=6)
    parser.add_argument("--max-positive-count", type=int, default=2)
    parser.add_argument("--min-negative-count", type=int, default=3)

    parser.add_argument("--hard-neg-window-size", type=int, default=3)
    parser.add_argument("--hard-neg-weight", type=float, default=3.0)
    parser.add_argument("--easy-neg-weight", type=float, default=1.0)
    parser.add_argument("--max-pairs-per-query", type=int, default=24)
    parser.add_argument("--max-pairs-total", type=int, default=0, help="0 means all")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    dataset_dir = Path(_clean_text(args.dataset_dir)).expanduser().resolve()
    input_listwise = _clean_text(args.input_listwise_jsonl)
    if not input_listwise:
        input_path = _resolve_latest_listwise(dataset_dir)
    else:
        input_path = Path(input_listwise).expanduser().resolve()

    payload = build_dataset(
        input_listwise_jsonl=str(input_path),
        output_dir=_clean_text(args.output_dir),
        output_prefix=_clean_text(args.output_prefix),
        max_queries=int(args.max_queries),
        min_candidates=int(args.min_candidates),
        candidate_min_words=int(args.candidate_min_words),
        candidate_max_words=int(args.candidate_max_words),
        drop_junk_candidates=bool(args.drop_junk_candidates),
        llm_model=_clean_text(args.llm_model),
        llm_batch_size=int(args.llm_batch_size),
        llm_max_workers=int(args.llm_max_workers),
        llm_max_retries=int(args.llm_max_retries),
        llm_query_max_chars=int(args.llm_query_max_chars),
        llm_query_max_words=int(args.llm_query_max_words),
        llm_candidate_max_chars=int(args.llm_candidate_max_chars),
        llm_candidate_max_words=int(args.llm_candidate_max_words),
        max_ranked_candidates_for_prompt=int(args.max_ranked_candidates_for_prompt),
        allow_llm_fallback=bool(args.allow_llm_fallback),
        fallback_boundary_rank=int(args.fallback_boundary_rank),
        drop_no_positive=bool(args.drop_no_positive),
        drop_no_negative=bool(args.drop_no_negative),
        boundary_min_rank=int(args.boundary_min_rank),
        boundary_max_rank=int(args.boundary_max_rank),
        max_positive_count=int(args.max_positive_count),
        min_negative_count=int(args.min_negative_count),
        hard_neg_window_size=int(args.hard_neg_window_size),
        hard_neg_weight=float(args.hard_neg_weight),
        easy_neg_weight=float(args.easy_neg_weight),
        max_pairs_per_query=int(args.max_pairs_per_query),
        max_pairs_total=int(args.max_pairs_total),
        seed=int(args.seed),
        log_llm_io=bool(args.log_llm_io),
        llm_log_max_chars=int(args.llm_log_max_chars),
    )

    print("Boundary supervision dataset build complete.")
    print(f"  input listwise jsonl   : {payload.get('input_listwise_jsonl', '')}")
    print(f"  loaded queries         : {payload.get('load_meta', {}).get('queries_loaded', 0)}")
    print(f"  llm calls              : {payload.get('llm', {}).get('llm_calls', 0)}")
    print(f"  llm retries            : {payload.get('llm', {}).get('retries_used', 0)}")
    print(f"  llm fallback queries   : {payload.get('llm', {}).get('fallback_query_count', 0)}")
    print(f"  kept boundary queries  : {payload.get('boundary_meta', {}).get('queries_kept', 0)}")
    print(f"  dropped (boundary rng) : {payload.get('boundary_meta', {}).get('dropped_boundary_out_of_range', 0)}")
    print(f"  dropped (few negatives): {payload.get('boundary_meta', {}).get('dropped_too_few_negatives', 0)}")
    print(f"  output pairs rows      : {payload.get('pair_meta', {}).get('rows_output', 0)}")
    print(f"  boundary listwise jsonl: {payload.get('output', {}).get('boundary_listwise_jsonl_path', '')}")
    print(f"  boundary pairs jsonl   : {payload.get('output', {}).get('boundary_pairs_jsonl_path', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
