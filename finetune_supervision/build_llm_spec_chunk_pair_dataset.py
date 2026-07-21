from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy import text

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import get_embedding_client, get_llm_client, settings
from db.db_conn import SessionLocal
from utils.content_extractor import load_extracted_content
from utils.embedder import embed_texts
from utils.thread_pool import build_thread_local_getter, parallel_map, resolve_pool_size


class RankedQueryRow(BaseModel):
    q: int = Field(..., description="1-based query index within batch input")
    ranked: List[int] = Field(default_factory=list, description="Candidate indices in best->worst order")


class RankedQueryOut(BaseModel):
    items: List[RankedQueryRow] = Field(default_factory=list)


RANK_SYSTEM_PROMPT = (
    "You rank faculty context chunks for grant specialization queries.\\n"
    "\\n"
    "Objective:\\n"
    "- Produce the most semantically correct ordering of candidate chunks for each query.\\n"
    "- Prefer chunks that directly support the exact specialization scope.\\n"
    "- Penalize generic, adjacent, or only loosely related chunks.\\n"
    "\\n"
    "Return JSON only.\\n"
    "Output schema:\\n"
    "{{\\n"
    "  \"items\": [\\n"
    "    {{\"q\": 1, \"ranked\": [3,1,2]}}\\n"
    "  ]\\n"
    "}}\\n"
    "Rules:\\n"
    "- ranked must contain every candidate index exactly once.\\n"
    "- Candidate indices are 1-based integers from input.\\n"
    "- No explanations, no extra keys.\\n"
)

RANK_HUMAN_PROMPT = "Tasks JSON:\\n{tasks_json}"


@dataclass
class SpecRow:
    grant_id: str
    spec_id: str
    section: str
    text: str
    weight: float
    embedding: List[float]


@dataclass
class ChunkRow:
    chunk_id: str
    chunk_type: str
    text: str
    embedding: List[float]


@dataclass
class AdditionalInfoExtractRow:
    id: int
    additional_info_url: str
    content_path: str
    extract_status: str
    extract_error: str
    chunk_index: int


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


def _safe_unit_float(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _coerce_vector(value: Any) -> List[float]:
    if isinstance(value, np.ndarray):
        try:
            return [float(x) for x in value.astype(float).tolist()]
        except Exception:
            return []
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for x in value:
            try:
                out.append(float(x))
            except Exception:
                return []
        return out
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        try:
            arr = np.fromstring(raw, sep=",", dtype=np.float32)
        except Exception:
            return []
        if arr.size <= 0:
            return []
        return [float(x) for x in arr.tolist()]
    return []


def _trim_text(value: Any, *, max_chars: int, max_words: int) -> str:
    t = _clean_text(value)
    if not t:
        return ""
    mw = max(1, int(max_words))
    mc = max(8, int(max_chars))
    toks = t.split()
    if len(toks) > mw:
        t = " ".join(toks[:mw])
    if len(t) > mc:
        t = t[:mc].rstrip()
    return t


def _clip_for_log(value: Any, max_chars: int) -> str:
    text = _clean_text(value)
    limit = max(80, int(max_chars))
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ...<truncated>"


def _extract_weighted_specs(keywords: Any) -> List[Tuple[str, float, str, int]]:
    if not isinstance(keywords, dict):
        return []
    out: List[Tuple[str, float, str, int]] = []
    for section in ("research", "application"):
        sec = keywords.get(section)
        if not isinstance(sec, dict):
            continue
        specs = list(sec.get("specialization") or [])
        for idx, item in enumerate(specs):
            text = ""
            weight = 1.0
            if isinstance(item, dict):
                text = _clean_text(item.get("t") or item.get("text") or item.get("value"))
                weight = _safe_unit_float(item.get("w"), default=1.0)
            else:
                text = _clean_text(item)
            if not text:
                continue
            out.append((text, float(weight), section, int(idx)))
    return out


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms <= 0.0, 1.0, norms)
    return vectors / norms


def _fetch_grant_specs_from_embedding_table(
    *,
    grant_limit: int,
    min_spec_weight: float,
    embedding_model: str,
) -> List[SpecRow]:
    sql = text(
        """
        WITH limited_grants AS (
            SELECT ok.opportunity_id
            FROM opportunity_keywords ok
            WHERE ok.keywords IS NOT NULL
            ORDER BY ok.opportunity_id ASC
            LIMIT :grant_limit
        )
        SELECT
            ose.opportunity_id::text AS grant_id,
            ose.section AS section,
            ose.spec_text AS spec_text,
            ose.spec_norm AS spec_norm,
            ose.spec_weight AS spec_weight,
            ose.spec_vec AS spec_vec
        FROM opportunity_specialization_embedding ose
        JOIN limited_grants lg ON lg.opportunity_id = ose.opportunity_id
        WHERE ose.model = :embedding_model
          AND COALESCE(ose.spec_weight, 1.0) >= :min_spec_weight
        ORDER BY ose.opportunity_id ASC, ose.id ASC
        """
    )
    with SessionLocal() as sess:
        rows = sess.execute(
            sql,
            {
                "grant_limit": int(max(1, grant_limit)),
                "embedding_model": _clean_text(embedding_model),
                "min_spec_weight": float(max(0.0, min_spec_weight)),
            },
        ).mappings().all()

    out: List[SpecRow] = []
    for row in rows:
        item = dict(row or {})
        grant_id = _clean_text(item.get("grant_id"))
        section = _clean_text(item.get("section"))
        spec_text = _clean_text(item.get("spec_text"))
        spec_norm = _clean_text(item.get("spec_norm"))
        spec_weight = float(item.get("spec_weight") if item.get("spec_weight") is not None else 1.0)
        spec_vec = _coerce_vector(item.get("spec_vec"))
        if not grant_id or not spec_text or not spec_vec:
            continue
        spec_id = f"grant:{grant_id}:{section}:{spec_norm or len(out)}"
        out.append(
            SpecRow(
                grant_id=grant_id,
                spec_id=spec_id,
                section=section,
                text=spec_text,
                weight=float(spec_weight),
                embedding=spec_vec,
            )
        )
    return out


def _fetch_grant_specs_and_embed(
    *,
    grant_limit: int,
    min_spec_weight: float,
    embed_batch_size: int,
    embed_max_workers: int,
) -> List[SpecRow]:
    sql = text(
        """
        SELECT
            o.opportunity_id::text AS grant_id,
            ok.keywords AS keywords
        FROM opportunity_keywords ok
        JOIN opportunity o ON o.opportunity_id = ok.opportunity_id
        WHERE ok.keywords IS NOT NULL
        ORDER BY o.opportunity_id ASC
        LIMIT :grant_limit
        """
    )
    with SessionLocal() as sess:
        rows = sess.execute(sql, {"grant_limit": int(max(1, grant_limit))}).mappings().all()

    specs: List[SpecRow] = []
    for row in rows:
        item = dict(row or {})
        grant_id = _clean_text(item.get("grant_id"))
        keywords = item.get("keywords") or {}
        for spec_text, weight, section, idx in _extract_weighted_specs(keywords):
            if float(weight) < float(max(0.0, min_spec_weight)):
                continue
            specs.append(
                SpecRow(
                    grant_id=grant_id,
                    spec_id=f"grant:{grant_id}:{section}:{idx}",
                    section=section,
                    text=spec_text,
                    weight=float(weight),
                    embedding=[],
                )
            )

    if not specs:
        return []

    safe_batch = _safe_limit(embed_batch_size, default=64, minimum=1, maximum=512)
    safe_workers = resolve_pool_size(
        max_workers=max(1, int(embed_max_workers)),
        task_count=max(1, int((len(specs) + safe_batch - 1) // safe_batch)),
    )

    unique_texts: List[str] = []
    seen = set()
    for row in specs:
        t = _clean_text(row.text)
        if not t or t in seen:
            continue
        seen.add(t)
        unique_texts.append(t)

    payloads = [unique_texts[i : i + safe_batch] for i in range(0, len(unique_texts), safe_batch)]
    get_embedder = build_thread_local_getter(lambda: get_embedding_client().build())

    def _run_payload(texts: List[str]) -> List[Tuple[str, List[float]]]:
        vecs = embed_texts(texts, embedding_client=get_embedder())
        if vecs.ndim != 2 or vecs.shape[0] != len(texts):
            raise RuntimeError("Embedding batch returned invalid shape for grant specs.")
        return [(t, [float(x) for x in vecs[i].tolist()]) for i, t in enumerate(texts)]

    mapping: Dict[str, List[float]] = {}
    for item in parallel_map(payloads, max_workers=safe_workers, run_item=_run_payload):
        for t, vec in list(item or []):
            mapping[t] = list(vec or [])

    out: List[SpecRow] = []
    for row in specs:
        vec = list(mapping.get(_clean_text(row.text)) or [])
        if not vec:
            continue
        out.append(replace(row, embedding=vec))
    return out


def _load_chunks(
    *,
    faculty_limit: int,
    publication_limit_per_faculty: int,
    additional_info_limit_per_faculty: int,
    min_chunk_chars: int,
    chunk_max_chars: int,
    chunk_max_words: int,
) -> List[ChunkRow]:
    safe_faculty_limit = _safe_limit(faculty_limit, default=200000, minimum=1, maximum=2_000_000)
    safe_pub_limit = _safe_limit(publication_limit_per_faculty, default=5, minimum=0, maximum=100)
    safe_info_limit = _safe_limit(additional_info_limit_per_faculty, default=5, minimum=0, maximum=100)
    safe_min_chars = _safe_limit(min_chunk_chars, default=24, minimum=1, maximum=4000)
    safe_chunk_chars = _safe_limit(chunk_max_chars, default=1200, minimum=64, maximum=30000)
    safe_chunk_words = _safe_limit(chunk_max_words, default=200, minimum=8, maximum=4000)

    publication_sql = text(
        """
        WITH limited_faculty AS (
            SELECT f.faculty_id
            FROM faculty f
            ORDER BY f.faculty_id ASC
            LIMIT :faculty_limit
        ),
        ranked_pub AS (
            SELECT
                fp.*,
                row_number() OVER (
                    PARTITION BY fp.faculty_id
                    ORDER BY fp.year DESC NULLS LAST, fp.id ASC
                ) AS rn
            FROM faculty_publication fp
            JOIN limited_faculty lf ON lf.faculty_id = fp.faculty_id
        )
        SELECT
            rp.id AS publication_id,
            coalesce(rp.title, '') AS title,
            coalesce(rp.abstract, '') AS abstract
        FROM ranked_pub rp
        WHERE rp.rn <= :publication_limit
        ORDER BY rp.id ASC
        """
    )

    additional_info_sql = text(
        """
        WITH limited_faculty AS (
            SELECT f.faculty_id
            FROM faculty f
            ORDER BY f.faculty_id ASC
            LIMIT :faculty_limit
        ),
        ranked_info AS (
            SELECT
                fa.*,
                row_number() OVER (
                    PARTITION BY fa.faculty_id
                    ORDER BY coalesce(fa.chunk_index, 0) ASC, fa.id ASC
                ) AS rn
            FROM faculty_additional_info fa
            JOIN limited_faculty lf ON lf.faculty_id = fa.faculty_id
        )
        SELECT
            ri.id AS additional_info_id,
            coalesce(ri.additional_info_url, '') AS additional_info_url,
            coalesce(ri.content_path, '') AS content_path,
            coalesce(ri.extract_status, '') AS extract_status,
            coalesce(ri.extract_error, '') AS extract_error,
            coalesce(ri.chunk_index, 0) AS chunk_index
        FROM ranked_info ri
        WHERE ri.rn <= :additional_info_limit
        ORDER BY ri.id ASC
        """
    )

    with SessionLocal() as sess:
        pub_rows = sess.execute(
            publication_sql,
            {"faculty_limit": int(safe_faculty_limit), "publication_limit": int(safe_pub_limit)},
        ).mappings().all()
        info_rows = sess.execute(
            additional_info_sql,
            {"faculty_limit": int(safe_faculty_limit), "additional_info_limit": int(safe_info_limit)},
        ).mappings().all()

    chunks: List[ChunkRow] = []
    seen = set()

    for row in pub_rows:
        item = dict(row or {})
        pub_id = _clean_text(item.get("publication_id"))
        if not pub_id:
            continue
        title = _clean_text(item.get("title"))
        abstract = _clean_text(item.get("abstract"))
        body = title if not abstract else f"{title}. {abstract}"
        body = _trim_text(body, max_chars=safe_chunk_chars, max_words=safe_chunk_words)
        if len(body) < safe_min_chars:
            continue
        cid = f"pub:{pub_id}"
        if cid in seen:
            continue
        seen.add(cid)
        chunks.append(ChunkRow(chunk_id=cid, chunk_type="publication", text=body, embedding=[]))

    info_load_rows: List[AdditionalInfoExtractRow] = []
    for row in info_rows:
        item = dict(row or {})
        row_id = int(item.get("additional_info_id") or 0)
        if row_id <= 0:
            continue
        content_path = _clean_text(item.get("content_path"))
        if not content_path:
            continue
        info_load_rows.append(
            AdditionalInfoExtractRow(
                id=row_id,
                additional_info_url=_clean_text(item.get("additional_info_url")),
                content_path=content_path,
                extract_status=_clean_text(item.get("extract_status")),
                extract_error=_clean_text(item.get("extract_error")),
                chunk_index=int(item.get("chunk_index") or 0),
            )
        )

    info_blocks: List[Dict[str, Any]] = []
    if info_load_rows:
        try:
            info_blocks = load_extracted_content(
                info_load_rows,
                url_attr="additional_info_url",
                group_chunks=False,
                include_row_meta=True,
            )
        except Exception as e:
            print(f"[warn] additional-info extraction load failed: {type(e).__name__}: {e}")
            info_blocks = []

    for block in info_blocks:
        row_id = int(block.get("row_id") or 0)
        if row_id <= 0:
            continue
        body = _trim_text(block.get("content"), max_chars=safe_chunk_chars, max_words=safe_chunk_words)
        if len(body) < safe_min_chars:
            continue
        cid = f"addinfo:{row_id}"
        if cid in seen:
            continue
        seen.add(cid)
        chunks.append(ChunkRow(chunk_id=cid, chunk_type="additional_info", text=body, embedding=[]))

    return chunks


def _embed_chunks(rows: Sequence[ChunkRow], *, batch_size: int, max_workers: int) -> List[ChunkRow]:
    chunk_rows = list(rows or [])
    if not chunk_rows:
        return []

    safe_batch = _safe_limit(batch_size, default=64, minimum=1, maximum=512)
    safe_workers = resolve_pool_size(
        max_workers=max(1, int(max_workers)),
        task_count=max(1, int((len(chunk_rows) + safe_batch - 1) // safe_batch)),
    )

    unique_texts: List[str] = []
    seen = set()
    for row in chunk_rows:
        t = _clean_text(row.text)
        if not t or t in seen:
            continue
        seen.add(t)
        unique_texts.append(t)

    payloads = [unique_texts[i : i + safe_batch] for i in range(0, len(unique_texts), safe_batch)]
    get_embedder = build_thread_local_getter(lambda: get_embedding_client().build())

    def _run_payload(texts: List[str]) -> List[Tuple[str, List[float]]]:
        vecs = embed_texts(texts, embedding_client=get_embedder())
        if vecs.ndim != 2 or vecs.shape[0] != len(texts):
            raise RuntimeError("Embedding batch returned invalid shape for chunks.")
        return [(t, [float(x) for x in vecs[i].tolist()]) for i, t in enumerate(texts)]

    mapping: Dict[str, List[float]] = {}
    for item in parallel_map(payloads, max_workers=safe_workers, run_item=_run_payload):
        for t, vec in list(item or []):
            mapping[t] = list(vec or [])

    out: List[ChunkRow] = []
    for row in chunk_rows:
        vec = list(mapping.get(_clean_text(row.text)) or [])
        if not vec:
            continue
        out.append(replace(row, embedding=vec))
    return out


def _select_common_dim(specs: Sequence[SpecRow], chunks: Sequence[ChunkRow]) -> Tuple[int, List[SpecRow], List[ChunkRow]]:
    spec_dims = Counter(len(x.embedding) for x in specs if x.embedding)
    chunk_dims = Counter(len(x.embedding) for x in chunks if x.embedding)
    common = [d for d in spec_dims if d in chunk_dims]
    if not common:
        return 0, [], []
    best = max(common, key=lambda d: (spec_dims[d] * chunk_dims[d], d))
    return int(best), [x for x in specs if len(x.embedding) == best], [x for x in chunks if len(x.embedding) == best]


def _build_candidates_for_query(
    *,
    query_vec: np.ndarray,
    chunks: Sequence[ChunkRow],
    chunk_mat_norm: np.ndarray,
    candidate_pool_size: int,
    top_keep: int,
    hard_window: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    sims = np.dot(chunk_mat_norm, query_vec)
    order = np.argsort(-sims).tolist()
    n = len(order)
    if n <= 1:
        return []

    safe_pool = max(2, min(int(candidate_pool_size), n))
    safe_top_keep = max(1, min(int(top_keep), safe_pool))
    safe_hard_window = max(0, int(hard_window))

    chosen: List[int] = []
    chosen_set = set()

    for idx in order[:safe_top_keep]:
        if idx in chosen_set:
            continue
        chosen_set.add(idx)
        chosen.append(idx)

    hard_start = safe_top_keep
    hard_end = min(n, hard_start + max(safe_hard_window, safe_pool * 3))
    hard_region = [int(i) for i in order[hard_start:hard_end] if int(i) not in chosen_set]
    rng.shuffle(hard_region)
    for idx in hard_region:
        if len(chosen) >= safe_pool:
            break
        chosen_set.add(idx)
        chosen.append(idx)

    if len(chosen) < safe_pool:
        tail = [int(i) for i in order[hard_end:] if int(i) not in chosen_set]
        rng.shuffle(tail)
        for idx in tail:
            if len(chosen) >= safe_pool:
                break
            chosen_set.add(idx)
            chosen.append(idx)

    chosen.sort(key=lambda i: float(sims[i]), reverse=True)
    out: List[Dict[str, Any]] = []
    for i, ridx in enumerate(chosen, start=1):
        row = chunks[int(ridx)]
        out.append(
            {
                "i": int(i),
                "t": str(row.text),
                "id": str(row.chunk_id),
                "type": str(row.chunk_type),
                "cos": float(sims[int(ridx)]),
            }
        )
    return out


def _build_rank_chain(model_id: str):
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RANK_SYSTEM_PROMPT),
            ("human", RANK_HUMAN_PROMPT),
        ]
    )
    llm = get_llm_client(model_id).build()
    return prompt | llm.with_structured_output(RankedQueryOut)


def _sanitize_ranking(
    ranked: Sequence[Any],
    *,
    valid_indices: Sequence[int],
    default_rank: Sequence[int],
) -> Tuple[List[int], bool]:
    valid_set = set(int(x) for x in valid_indices)
    out: List[int] = []
    seen = set()
    had_any = False

    raw_list: List[int] = []
    for raw in list(ranked or []):
        try:
            idx = int(raw)
        except Exception:
            continue
        raw_list.append(idx)
        if idx in valid_set and idx not in seen:
            seen.add(idx)
            out.append(idx)
            had_any = True

    # Handle accidental 0-based outputs.
    if (not had_any) and raw_list and valid_indices:
        valid_sorted = sorted(valid_set)
        if valid_sorted and valid_sorted[0] == 1 and valid_sorted[-1] == len(valid_sorted):
            if min(raw_list) >= 0 and max(raw_list) <= (len(valid_sorted) - 1):
                for idx0 in raw_list:
                    idx1 = int(idx0) + 1
                    if idx1 in valid_set and idx1 not in seen:
                        seen.add(idx1)
                        out.append(idx1)
                had_any = bool(out)

    for idx in list(default_rank or []):
        i = int(idx)
        if i in valid_set and i not in seen:
            seen.add(i)
            out.append(i)

    return out, had_any


def _label_rankings_with_llm(
    *,
    query_sets: Sequence[Dict[str, Any]],
    llm_model: str,
    llm_batch_size: int,
    llm_max_workers: int,
    llm_max_retries: int,
    llm_query_max_chars: int,
    llm_query_max_words: int,
    llm_candidate_max_chars: int,
    llm_candidate_max_words: int,
    allow_llm_fallback: bool,
    log_llm_io: bool,
    llm_log_max_chars: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = list(query_sets or [])
    total_queries = len(rows)
    if total_queries <= 0:
        return [], {
            "queries_total": 0,
            "queries_labeled": 0,
            "batches": 0,
            "llm_calls": 0,
            "retries_used": 0,
            "fallback_query_count": 0,
            "partial_fallback_query_count": 0,
        }

    safe_batch = max(1, int(llm_batch_size))
    total_batches = (total_queries + safe_batch - 1) // safe_batch
    safe_workers = resolve_pool_size(max_workers=max(1, int(llm_max_workers)), task_count=max(1, total_batches))
    safe_retries = max(1, int(llm_max_retries))

    payloads: List[Dict[str, Any]] = []
    for bidx, start in enumerate(range(0, total_queries, safe_batch), start=1):
        end = min(total_queries, start + safe_batch)
        batch_rows = rows[start:end]
        valid_index_map: Dict[int, List[int]] = {}
        default_rank_map: Dict[int, List[int]] = {}
        tasks: List[Dict[str, Any]] = []

        for q_local, row in enumerate(batch_rows, start=1):
            cands = list(row.get("candidates") or [])
            valid = [int(c.get("i") or 0) for c in cands if int(c.get("i") or 0) > 0]
            default = [int(c.get("i")) for c in sorted(cands, key=lambda x: float(x.get("cos") or -1e9), reverse=True)]
            valid_index_map[q_local] = valid
            default_rank_map[q_local] = default
            tasks.append(
                {
                    "q": int(q_local),
                    "query": _trim_text(row.get("query"), max_chars=int(llm_query_max_chars), max_words=int(llm_query_max_words)),
                    "c": [
                        {
                            "i": int(c.get("i") or 0),
                            "t": _trim_text(c.get("t"), max_chars=int(llm_candidate_max_chars), max_words=int(llm_candidate_max_words)),
                        }
                        for c in cands
                    ],
                }
            )

        payloads.append(
            {
                "batch_number": int(bidx),
                "rows": batch_rows,
                "valid_index_map": valid_index_map,
                "default_rank_map": default_rank_map,
                "tasks_json": json.dumps(tasks, ensure_ascii=False),
            }
        )

    get_chain = build_thread_local_getter(lambda: _build_rank_chain(llm_model))
    call_lock = threading.Lock()
    call_counter = {"value": 0}

    def _run_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        batch_number = int(payload.get("batch_number") or 0)
        batch_rows = list(payload.get("rows") or [])
        valid_index_map = dict(payload.get("valid_index_map") or {})
        default_rank_map = dict(payload.get("default_rank_map") or {})
        tasks_json = _clean_text(payload.get("tasks_json"))

        ranked_map: Dict[int, List[int]] = {}
        batch_ok = False
        retries_local = 0
        fallback_local = 0
        partial_local = 0
        skipped_query_local = 0

        def _build_single_query_tasks_json(*, q_local: int, row: Dict[str, Any]) -> str:
            cands = list(row.get("candidates") or [])
            one = [
                {
                    "q": 1,
                    "query": _trim_text(
                        row.get("query"),
                        max_chars=int(llm_query_max_chars),
                        max_words=int(llm_query_max_words),
                    ),
                    "c": [
                        {
                            "i": int(c.get("i") or 0),
                            "t": _trim_text(
                                c.get("t"),
                                max_chars=int(llm_candidate_max_chars),
                                max_words=int(llm_candidate_max_words),
                            ),
                        }
                        for c in cands
                    ],
                }
            ]
            return json.dumps(one, ensure_ascii=False)

        def _invoke_single_query_retry(*, q_local: int, row: Dict[str, Any]) -> List[int]:
            single_tasks_json = _build_single_query_tasks_json(q_local=q_local, row=row)
            for sub_attempt in range(0, safe_retries):
                if sub_attempt > 0:
                    retries_local_nonlocal[0] += 1
                try:
                    with call_lock:
                        call_counter["value"] += 1
                        call_id = int(call_counter["value"])
                    print(
                        f"[llm_call_repair] count={call_id} batch={batch_number}/{total_batches} "
                        f"query_local={q_local} attempt={sub_attempt + 1}/{safe_retries}"
                    )
                    if bool(log_llm_io):
                        print(
                            "[llm_input_repair] "
                            f"batch={batch_number}/{total_batches} query_local={q_local} "
                            f"payload={_clip_for_log(single_tasks_json, int(llm_log_max_chars))}"
                        )
                    result = get_chain().invoke({"tasks_json": single_tasks_json})
                    if bool(log_llm_io):
                        try:
                            raw_out = result.model_dump() if hasattr(result, "model_dump") else dict(result or {})
                            print(
                                "[llm_output_repair] "
                                f"batch={batch_number}/{total_batches} query_local={q_local} "
                                f"payload={_clip_for_log(json.dumps(raw_out, ensure_ascii=False), int(llm_log_max_chars))}"
                            )
                        except Exception:
                            print(
                                f"[llm_output_repair] batch={batch_number}/{total_batches} "
                                f"query_local={q_local} (unable to serialize)"
                            )

                    repaired_ranked: List[int] = []
                    for item in list(getattr(result, "items", []) or []):
                        row_obj = item.model_dump() if hasattr(item, "model_dump") else dict(item or {})
                        ranked_raw = row_obj.get("ranked")
                        if not isinstance(ranked_raw, list):
                            continue
                        tmp: List[int] = []
                        for x in ranked_raw:
                            try:
                                tmp.append(int(x))
                            except Exception:
                                continue
                        if tmp:
                            repaired_ranked = tmp
                            break
                    if repaired_ranked:
                        return repaired_ranked
                except Exception as e:
                    print(
                        f"[llm_call_repair_error] batch={batch_number}/{total_batches} "
                        f"query_local={q_local} attempt={sub_attempt + 1}/{safe_retries} "
                        f"error={type(e).__name__}: {e}"
                    )
            return []

        retries_local_nonlocal = [0]

        for attempt in range(0, safe_retries):
            if attempt > 0:
                retries_local += 1
            try:
                with call_lock:
                    call_counter["value"] += 1
                    call_id = int(call_counter["value"])
                print(
                    f"[llm_call] count={call_id} batch={batch_number}/{total_batches} "
                    f"attempt={attempt + 1}/{safe_retries} batch_queries={len(batch_rows)}"
                )
                if bool(log_llm_io):
                    print(
                        "[llm_input] "
                        f"batch={batch_number}/{total_batches} "
                        f"payload={_clip_for_log(tasks_json, int(llm_log_max_chars))}"
                    )
                result = get_chain().invoke({"tasks_json": tasks_json})
                if bool(log_llm_io):
                    try:
                        raw_out = result.model_dump() if hasattr(result, "model_dump") else dict(result or {})
                        print(
                            "[llm_output] "
                            f"batch={batch_number}/{total_batches} "
                            f"payload={_clip_for_log(json.dumps(raw_out, ensure_ascii=False), int(llm_log_max_chars))}"
                        )
                    except Exception:
                        print(f"[llm_output] batch={batch_number}/{total_batches} (unable to serialize)")
                for item in list(getattr(result, "items", []) or []):
                    row = item.model_dump() if hasattr(item, "model_dump") else dict(item or {})
                    q_local = int(row.get("q") or 0)
                    ranked_raw = row.get("ranked")
                    if q_local <= 0 or q_local not in valid_index_map or not isinstance(ranked_raw, list):
                        continue
                    tmp_ranked: List[int] = []
                    for x in ranked_raw:
                        try:
                            tmp_ranked.append(int(x))
                        except Exception:
                            continue
                    ranked_map[q_local] = tmp_ranked
                batch_ok = True
                break
            except Exception as e:
                print(
                    f"[llm_call_error] batch={batch_number}/{total_batches} "
                    f"attempt={attempt + 1}/{safe_retries} error={type(e).__name__}: {e}"
                )

        # If full batch call failed and fallback is off, skip this batch instead of crashing the run.
        if (not batch_ok) and (not allow_llm_fallback):
            print(f"[llm_batch_skipped] batch={batch_number}/{total_batches} reason=llm_batch_failure")
            return {
                "failed": False,
                "error": "llm_batch_failure",
                "rows": [],
                "batch_number": int(batch_number),
                "retries_used": int(retries_local + retries_local_nonlocal[0]),
                "fallback_query_count": int(fallback_local),
                "partial_fallback_query_count": int(partial_local),
                "skipped_batch": True,
                "skipped_query_count": int(len(batch_rows)),
            }

        out_rows: List[Dict[str, Any]] = []
        skip_entire_batch = False
        skip_reason = ""
        for q_local, row in enumerate(batch_rows, start=1):
            ranking, had_any_valid = _sanitize_ranking(
                ranked_map.get(q_local, []),
                valid_indices=valid_index_map.get(q_local, []),
                default_rank=default_rank_map.get(q_local, []),
            )
            if len(ranking) < 2:
                continue

            # Output checker + single-query repair before assigning label.
            if batch_ok and (not had_any_valid or len(ranking) != len(valid_index_map.get(q_local, []))):
                repaired_ranked = _invoke_single_query_retry(q_local=q_local, row=row)
                if repaired_ranked:
                    ranking, had_any_valid = _sanitize_ranking(
                        repaired_ranked,
                        valid_indices=valid_index_map.get(q_local, []),
                        default_rank=default_rank_map.get(q_local, []),
                    )
                    retries_local = int(retries_local + retries_local_nonlocal[0])
                    retries_local_nonlocal[0] = 0

            out = dict(row or {})
            out["ranking"] = ranking
            if not batch_ok:
                if not allow_llm_fallback:
                    skip_entire_batch = True
                    skip_reason = f"LLM failed batch={batch_number}"
                    break
                out["label_source"] = "fallback_cosine_batch"
                fallback_local += 1
            elif not had_any_valid:
                if not allow_llm_fallback:
                    skip_entire_batch = True
                    skip_reason = f"LLM invalid ranking batch={batch_number} query_local={q_local}"
                    skipped_query_local = int(len(batch_rows))
                    break
                out["label_source"] = "fallback_cosine_query"
                fallback_local += 1
            elif len(ranking) != len(valid_index_map.get(q_local, [])):
                if not allow_llm_fallback:
                    skip_entire_batch = True
                    skip_reason = f"LLM partial ranking batch={batch_number} query_local={q_local}"
                    skipped_query_local = int(len(batch_rows))
                    break
                out["label_source"] = "llm_partial_ranking"
                partial_local += 1
            else:
                out["label_source"] = "llm_ranking"

            out_rows.append(out)

        if skip_entire_batch:
            print(
                f"[llm_batch_skipped] batch={batch_number}/{total_batches} reason={skip_reason}"
            )
            return {
                "failed": False,
                "error": str(skip_reason or "skipped_batch"),
                "rows": [],
                "batch_number": int(batch_number),
                "retries_used": int(retries_local + retries_local_nonlocal[0]),
                "fallback_query_count": int(fallback_local),
                "partial_fallback_query_count": int(partial_local),
                "skipped_batch": True,
                "skipped_query_count": int(max(0, skipped_query_local)),
            }

        return {
            "failed": False,
            "error": "",
            "rows": out_rows,
            "batch_number": int(batch_number),
            "retries_used": int(retries_local + retries_local_nonlocal[0]),
            "fallback_query_count": int(fallback_local),
            "partial_fallback_query_count": int(partial_local),
            "skipped_batch": False,
            "skipped_query_count": int(skipped_query_local),
        }

    run_results = parallel_map(payloads, max_workers=safe_workers, run_item=_run_payload)
    run_results = sorted(list(run_results or []), key=lambda x: int((x or {}).get("batch_number") or 0))

    labeled: List[Dict[str, Any]] = []
    retries = 0
    fallback = 0
    partial = 0
    skipped_batches = 0
    skipped_queries = 0
    for item in run_results:
        row = dict(item or {})
        labeled.extend(list(row.get("rows") or []))
        retries += int(row.get("retries_used") or 0)
        fallback += int(row.get("fallback_query_count") or 0)
        partial += int(row.get("partial_fallback_query_count") or 0)
        if bool(row.get("skipped_batch")):
            skipped_batches += 1
        skipped_queries += int(row.get("skipped_query_count") or 0)

    llm_calls = int(call_counter["value"])
    return labeled, {
        "queries_total": int(total_queries),
        "queries_labeled": int(len(labeled)),
        "batches": int(total_batches),
        "llm_batch_size": int(safe_batch),
        "llm_max_workers": int(safe_workers),
        "llm_model": str(llm_model or ""),
        "llm_calls": int(llm_calls),
        "retries_used": int(retries),
        "fallback_query_count": int(fallback),
        "fallback_query_ratio": (0.0 if total_queries <= 0 else float(fallback) / float(total_queries)),
        "partial_fallback_query_count": int(partial),
        "skipped_batch_count": int(skipped_batches),
        "skipped_query_count": int(skipped_queries),
    }


def _to_pair_rows(listwise_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in list(listwise_rows or []):
        query = _clean_text(row.get("query"))
        query_group = _clean_text(row.get("query_group"))
        ranking = [int(x) for x in list(row.get("ranking") or [])]
        cands = list(row.get("candidates") or [])
        by_idx = {int(c.get("i") or 0): dict(c) for c in cands if int(c.get("i") or 0) > 0}
        n = max(1, len(ranking))
        denom = max(1, n - 1)

        for pos, idx in enumerate(ranking):
            c = by_idx.get(int(idx)) or {}
            doc = _clean_text(c.get("t"))
            if not query or not doc:
                continue
            label_score = 1.0 - (float(pos) / float(denom))
            out.append(
                {
                    "query": query,
                    "doc": doc,
                    "query_group": query_group,
                    "label_score": float(max(0.0, min(1.0, label_score))),
                    "label_source": _clean_text(row.get("label_source")) or "unknown",
                }
            )
    return out


def _compact_listwise_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for row in list(rows or []):
        cands = list(row.get("candidates") or [])
        compact.append(
            {
                "query": _clean_text(row.get("query")),
                "query_group": _clean_text(row.get("query_group")),
                "candidates": [
                    {
                        "i": int(c.get("i") or 0),
                        "t": _clean_text(c.get("t")),
                    }
                    for c in cands
                    if int(c.get("i") or 0) > 0 and _clean_text(c.get("t"))
                ],
                "ranking": [int(x) for x in list(row.get("ranking") or []) if int(x) > 0],
                "label_source": _clean_text(row.get("label_source")) or "unknown",
            }
        )
    return compact


def _save_jsonl(rows: Sequence[Dict[str, Any]], *, output_dir: Path, output_prefix: str, meta: Dict[str, Any]) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = f"{_clean_text(output_prefix) or 'dataset'}_{ts}"
    jsonl_path = output_dir / f"{stem}.jsonl"
    meta_path = output_dir / f"{stem}.meta.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in list(rows or []):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload = dict(meta or {})
    payload["row_count"] = int(len(list(rows or [])))
    payload["jsonl_path"] = str(jsonl_path)
    payload["meta_path"] = str(meta_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {"jsonl_path": str(jsonl_path), "meta_path": str(meta_path)}


def _to_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_dataset(
    *,
    max_queries: int,
    max_pair_rows: int,
    grant_limit: int,
    faculty_limit: int,
    grant_min_spec_weight: float,
    use_stored_grant_embeddings: bool,
    spec_embedding_model: str,
    publication_limit_per_faculty: int,
    additional_info_limit_per_faculty: int,
    min_chunk_chars: int,
    chunk_max_chars: int,
    chunk_max_words: int,
    embed_batch_size: int,
    embed_max_workers: int,
    candidate_pool_size: int,
    top_keep: int,
    hard_window_size: int,
    llm_model: str,
    llm_batch_size: int,
    llm_max_workers: int,
    llm_max_retries: int,
    llm_query_max_chars: int,
    llm_query_max_words: int,
    llm_candidate_max_chars: int,
    llm_candidate_max_words: int,
    allow_llm_fallback: bool,
    keep_low_quality_llm_sets: bool,
    log_llm_io: bool,
    llm_log_max_chars: int,
    seed: int,
    output_dir: Path,
    output_prefix: str,
) -> Dict[str, Any]:
    safe_seed = int(seed)
    rng = random.Random(safe_seed)

    safe_max_queries = _safe_limit(max_queries, default=4000, minimum=1, maximum=2_000_000)
    safe_max_rows = _safe_limit(max_pair_rows, default=0, minimum=0, maximum=20_000_000)
    safe_pool = _safe_limit(candidate_pool_size, default=8, minimum=2, maximum=64)
    safe_top_keep = _safe_limit(top_keep, default=3, minimum=1, maximum=64)
    safe_hard_window = _safe_limit(hard_window_size, default=24, minimum=0, maximum=2000)

    safe_embed_batch = _safe_limit(embed_batch_size, default=64, minimum=1, maximum=512)
    safe_embed_workers = _safe_limit(embed_max_workers, default=4, minimum=1, maximum=64)

    safe_llm_batch = _safe_limit(llm_batch_size, default=16, minimum=1, maximum=64)
    safe_llm_workers = _safe_limit(llm_max_workers, default=4, minimum=1, maximum=64)
    safe_llm_retries = _safe_limit(llm_max_retries, default=2, minimum=1, maximum=8)

    safe_query_chars = _safe_limit(llm_query_max_chars, default=220, minimum=32, maximum=4000)
    safe_query_words = _safe_limit(llm_query_max_words, default=40, minimum=8, maximum=512)
    safe_cand_chars = _safe_limit(llm_candidate_max_chars, default=220, minimum=32, maximum=4000)
    safe_cand_words = _safe_limit(llm_candidate_max_words, default=40, minimum=8, maximum=512)

    safe_spec_model = _clean_text(spec_embedding_model) or _clean_text(settings.bedrock_embed_model_id)
    safe_llm_model = _clean_text(llm_model) or _clean_text(settings.haiku)

    if bool(use_stored_grant_embeddings):
        specs_raw = _fetch_grant_specs_from_embedding_table(
            grant_limit=int(max(1, grant_limit)),
            min_spec_weight=float(max(0.0, grant_min_spec_weight)),
            embedding_model=safe_spec_model,
        )
    else:
        specs_raw = _fetch_grant_specs_and_embed(
            grant_limit=int(max(1, grant_limit)),
            min_spec_weight=float(max(0.0, grant_min_spec_weight)),
            embed_batch_size=int(safe_embed_batch),
            embed_max_workers=int(safe_embed_workers),
        )

    chunks_raw = _load_chunks(
        faculty_limit=int(faculty_limit),
        publication_limit_per_faculty=int(publication_limit_per_faculty),
        additional_info_limit_per_faculty=int(additional_info_limit_per_faculty),
        min_chunk_chars=int(min_chunk_chars),
        chunk_max_chars=int(chunk_max_chars),
        chunk_max_words=int(chunk_max_words),
    )
    chunks_emb = _embed_chunks(chunks_raw, batch_size=safe_embed_batch, max_workers=safe_embed_workers)

    common_dim, specs, chunks = _select_common_dim(specs_raw, chunks_emb)
    if common_dim <= 0 or not specs or not chunks:
        raise RuntimeError("No shared embedding dimension between grant specs and chunk embeddings.")

    if len(specs) > safe_max_queries:
        rng.shuffle(specs)
        specs = specs[:safe_max_queries]

    chunk_mat = np.asarray([x.embedding for x in chunks], dtype=np.float32)
    chunk_mat_norm = _normalize_rows(chunk_mat)
    spec_mat = np.asarray([x.embedding for x in specs], dtype=np.float32)
    spec_mat_norm = _normalize_rows(spec_mat)

    query_sets: List[Dict[str, Any]] = []
    for i, spec in enumerate(specs):
        candidates = _build_candidates_for_query(
            query_vec=spec_mat_norm[i],
            chunks=chunks,
            chunk_mat_norm=chunk_mat_norm,
            candidate_pool_size=safe_pool,
            top_keep=safe_top_keep,
            hard_window=safe_hard_window,
            rng=rng,
        )
        if len(candidates) < 2:
            continue
        query_sets.append(
            {
                "query_group": f"grant:{spec.grant_id}:{spec.section}:{_clean_text(spec.text).lower()}",
                "query": str(spec.text),
                "candidates": candidates,
            }
        )

    ranked_rows, llm_meta = _label_rankings_with_llm(
        query_sets=query_sets,
        llm_model=safe_llm_model,
        llm_batch_size=safe_llm_batch,
        llm_max_workers=safe_llm_workers,
        llm_max_retries=safe_llm_retries,
        llm_query_max_chars=safe_query_chars,
        llm_query_max_words=safe_query_words,
        llm_candidate_max_chars=safe_cand_chars,
        llm_candidate_max_words=safe_cand_words,
        allow_llm_fallback=bool(allow_llm_fallback),
        log_llm_io=bool(log_llm_io),
        llm_log_max_chars=int(llm_log_max_chars),
    )

    ranked_before = list(ranked_rows or [])
    if not bool(keep_low_quality_llm_sets):
        ranked_rows = [dict(x) for x in ranked_before if _clean_text(x.get("label_source")) == "llm_ranking"]

    if not ranked_rows:
        raise RuntimeError("No clean llm_ranking rows left. Try --keep-low-quality-llm-sets or improve prompt/model.")

    pair_rows = _to_pair_rows(ranked_rows)
    compact_listwise_rows = _compact_listwise_rows(ranked_rows)
    if safe_max_rows > 0 and len(pair_rows) > safe_max_rows:
        rng.shuffle(pair_rows)
        pair_rows = pair_rows[:safe_max_rows]

    listwise_paths = _save_jsonl(
        compact_listwise_rows,
        output_dir=output_dir,
        output_prefix=f"{output_prefix}_listwise",
        meta={
            "kind": "spec_chunk_listwise_ranking",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "llm": llm_meta,
        },
    )
    pair_paths = _save_jsonl(
        pair_rows,
        output_dir=output_dir,
        output_prefix=f"{output_prefix}_pairs",
        meta={
            "kind": "spec_chunk_pair_ranking",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "llm": llm_meta,
        },
    )

    label_counts_before = Counter(_clean_text(x.get("label_source")) for x in ranked_before)
    label_counts_after = Counter(_clean_text(x.get("label_source")) for x in ranked_rows)

    return {
        "params": {
            "max_queries": int(safe_max_queries),
            "max_pair_rows": int(safe_max_rows),
            "grant_limit": int(grant_limit),
            "faculty_limit": int(faculty_limit),
            "grant_min_spec_weight": float(max(0.0, grant_min_spec_weight)),
            "use_stored_grant_embeddings": bool(use_stored_grant_embeddings),
            "spec_embedding_model": safe_spec_model,
            "publication_limit_per_faculty": int(publication_limit_per_faculty),
            "additional_info_limit_per_faculty": int(additional_info_limit_per_faculty),
            "candidate_pool_size": int(safe_pool),
            "top_keep": int(safe_top_keep),
            "hard_window_size": int(safe_hard_window),
            "llm_model": safe_llm_model,
            "llm_batch_size": int(safe_llm_batch),
            "llm_max_workers": int(safe_llm_workers),
            "llm_max_retries": int(safe_llm_retries),
            "allow_llm_fallback": bool(allow_llm_fallback),
            "keep_low_quality_llm_sets": bool(keep_low_quality_llm_sets),
            "log_llm_io": bool(log_llm_io),
            "llm_log_max_chars": int(llm_log_max_chars),
            "seed": int(safe_seed),
        },
        "counts": {
            "specs_raw": int(len(specs_raw)),
            "chunks_raw": int(len(chunks_raw)),
            "chunks_embedded": int(len(chunks_emb)),
            "common_embedding_dim": int(common_dim),
            "specs_used": int(len(specs)),
            "queries_generated": int(len(query_sets)),
            "queries_ranked_before_quality_filter": int(len(ranked_before)),
            "queries_ranked_after_quality_filter": int(len(ranked_rows)),
            "queries_listwise_saved": int(len(compact_listwise_rows)),
            "pair_rows_saved": int(len(pair_rows)),
            "label_source_counts_before_quality_filter": dict(label_counts_before),
            "label_source_counts_after_quality_filter": dict(label_counts_after),
        },
        "llm": llm_meta,
        "output": {
            "listwise_jsonl_path": str(listwise_paths.get("jsonl_path")),
            "listwise_meta_path": str(listwise_paths.get("meta_path")),
            "pairs_jsonl_path": str(pair_paths.get("jsonl_path")),
            "pairs_meta_path": str(pair_paths.get("meta_path")),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    default_output_dir = Path(__file__).resolve().parent / "dataset"
    p = argparse.ArgumentParser(
        description=(
            "Ranking-only LLM distillation dataset builder for grant-spec -> faculty-chunk pairs."
        )
    )

    p.add_argument("--max-queries", type=int, default=4600)
    p.add_argument("--max-pair-rows", type=int, default=0)
    p.add_argument("--grant-limit", type=int, default=400000)
    p.add_argument("--faculty-limit", type=int, default=400000)
    p.add_argument("--grant-min-spec-weight", type=float, default=0.0)

    p.add_argument("--use-stored-grant-embeddings", dest="use_stored_grant_embeddings", action="store_true")
    p.add_argument("--no-use-stored-grant-embeddings", dest="use_stored_grant_embeddings", action="store_false")
    p.set_defaults(use_stored_grant_embeddings=True)
    p.add_argument("--spec-embedding-model", type=str, default=_clean_text(settings.bedrock_embed_model_id))

    p.add_argument("--publication-limit-per-faculty", type=int, default=5)
    p.add_argument("--additional-info-limit-per-faculty", type=int, default=5)
    p.add_argument("--min-chunk-chars", type=int, default=24)
    p.add_argument("--chunk-max-chars", type=int, default=1200)
    p.add_argument("--chunk-max-words", type=int, default=200)
    p.add_argument("--embed-batch-size", type=int, default=64)
    p.add_argument("--embed-max-workers", type=int, default=4)

    p.add_argument("--candidate-pool-size", type=int, default=8)
    p.add_argument("--top-keep", type=int, default=3)
    p.add_argument("--hard-window-size", type=int, default=24)

    p.add_argument("--llm-model", type=str, default=_clean_text(settings.haiku))
    p.add_argument("--llm-batch-size", type=int, default=16)
    p.add_argument("--llm-max-workers", type=int, default=4)
    p.add_argument("--llm-max-retries", type=int, default=2)

    p.add_argument("--llm-query-max-chars", type=int, default=220)
    p.add_argument("--llm-query-max-words", type=int, default=40)
    p.add_argument("--llm-candidate-max-chars", type=int, default=220)
    p.add_argument("--llm-candidate-max-words", type=int, default=40)

    p.add_argument("--allow-llm-fallback", action="store_true")
    p.add_argument("--keep-low-quality-llm-sets", action="store_true")
    p.add_argument("--log-llm-io", action="store_true")
    p.add_argument("--llm-log-max-chars", type=int, default=2000)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=str(default_output_dir))
    p.add_argument("--output-prefix", type=str, default="spec_chunk_rankdistill")
    p.add_argument("--json-only", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    payload = build_dataset(
        max_queries=int(args.max_queries),
        max_pair_rows=int(args.max_pair_rows),
        grant_limit=int(args.grant_limit),
        faculty_limit=int(args.faculty_limit),
        grant_min_spec_weight=float(args.grant_min_spec_weight),
        use_stored_grant_embeddings=bool(args.use_stored_grant_embeddings),
        spec_embedding_model=_clean_text(args.spec_embedding_model),
        publication_limit_per_faculty=int(args.publication_limit_per_faculty),
        additional_info_limit_per_faculty=int(args.additional_info_limit_per_faculty),
        min_chunk_chars=int(args.min_chunk_chars),
        chunk_max_chars=int(args.chunk_max_chars),
        chunk_max_words=int(args.chunk_max_words),
        embed_batch_size=int(args.embed_batch_size),
        embed_max_workers=int(args.embed_max_workers),
        candidate_pool_size=int(args.candidate_pool_size),
        top_keep=int(args.top_keep),
        hard_window_size=int(args.hard_window_size),
        llm_model=_clean_text(args.llm_model),
        llm_batch_size=int(args.llm_batch_size),
        llm_max_workers=int(args.llm_max_workers),
        llm_max_retries=int(args.llm_max_retries),
        llm_query_max_chars=int(args.llm_query_max_chars),
        llm_query_max_words=int(args.llm_query_max_words),
        llm_candidate_max_chars=int(args.llm_candidate_max_chars),
        llm_candidate_max_words=int(args.llm_candidate_max_words),
        allow_llm_fallback=bool(args.allow_llm_fallback),
        keep_low_quality_llm_sets=bool(args.keep_low_quality_llm_sets),
        log_llm_io=bool(args.log_llm_io),
        llm_log_max_chars=int(args.llm_log_max_chars),
        seed=int(args.seed),
        output_dir=Path(_clean_text(args.output_dir)),
        output_prefix=_clean_text(args.output_prefix) or "spec_chunk_rankdistill",
    )

    if not bool(args.json_only):
        print("Spec->chunk ranking distillation dataset build complete.")
        print(f"  created_at_utc       : {datetime.now(timezone.utc).isoformat()}")
        print(f"  queries ranked       : {payload.get('counts', {}).get('queries_ranked_after_quality_filter', 0)}")
        print(f"  pair rows saved      : {payload.get('counts', {}).get('pair_rows_saved', 0)}")
        print(f"  llm calls            : {payload.get('llm', {}).get('llm_calls', 0)}")
        print(f"  listwise jsonl       : {payload.get('output', {}).get('listwise_jsonl_path', '')}")
        print(f"  pair jsonl           : {payload.get('output', {}).get('pairs_jsonl_path', '')}")
        print()

    print(json.dumps(_to_json_ready(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
