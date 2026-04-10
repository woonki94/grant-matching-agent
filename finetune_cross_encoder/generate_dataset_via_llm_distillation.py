from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy import text

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import get_llm_client, settings
from db.db_conn import SessionLocal
from utils.thread_pool import build_thread_local_getter, parallel_map, resolve_pool_size


DEFAULT_OUTPUT_PREFIX = "spec_pair_llm_distill_v2"
DEFAULT_MAX_QUERIES = 20_000
DEFAULT_MAX_PAIRS = 1_000_000
DEFAULT_CANDIDATE_POOL_SIZE = 24
DEFAULT_LLM_BATCH_SIZE = 8
DEFAULT_LLM_MAX_RETRIES = 2
DEFAULT_LLM_WORKERS = 4
DEFAULT_FACULTY_LIMIT = 200_000
DEFAULT_GRANT_LIMIT = 200_000

POS_TEACHER_SCORE_TOP1 = 0.95
POS_TEACHER_SCORE_DECAY = 0.05
HARD_NEG_TEACHER_SCORE = 0.15
EASY_NEG_TEACHER_SCORE = 0.02
MIN_TEACHER_MARGIN = 0.50
HARD_NEG_MAX_COSINE_GAP = 0.30


class CandidateLabelItem(BaseModel):
    q: int = Field(..., description="1-based index in current batch")
    skip: bool = Field(default=False)
    positive_indices: List[int] = Field(default_factory=list)
    hard_negative_indices: List[int] = Field(default_factory=list)
    easy_negative_indices: List[int] = Field(default_factory=list)


class CandidateLabelOut(BaseModel):
    items: List[CandidateLabelItem] = Field(default_factory=list)


DISTILL_SYSTEM_PROMPT = (
    "You are distilling training labels for a grant-to-faculty cross-encoder reranker.\n"
    "Each task has ONE query and a fixed list of candidate faculty specializations (indexed).\n"
    "\n"
    "Goal:\n"
    "- Select positives, hard negatives, and easy negatives USING ONLY candidate indices.\n"
    "- positives: truly relevant and supportive to the query.\n"
    "- hard negatives: semantically close but incorrect for the query's intent.\n"
    "- easy negatives: clearly irrelevant.\n"
    "\n"
    "Rules:\n"
    "- Do not invent indices.\n"
    "- No overlap across positive/hard/easy lists.\n"
    "- positives: 1 to 3 items when possible.\n"
    "- hard negatives: 4 to 10 items when possible.\n"
    "- easy negatives: 0 to 2 items.\n"
    "- If the query is too ambiguous for safe labeling, set skip=true and leave lists empty.\n"
    "- Return JSON only.\n"
)

DISTILL_HUMAN_PROMPT = "Tasks JSON:\n{tasks_json}"


@dataclass
class SpecRow:
    source: str  # "faculty" | "grant"
    owner_id: str
    owner_email: str
    spec_node_id: str
    text: str
    spec_weight: float
    domains: List[str]
    embedding: List[float]

    @property
    def spec_id(self) -> str:
        return f"{self.source}:{self.owner_id}:{self.spec_node_id}"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_text_key(value: Any) -> str:
    return " ".join(_clean_text(value).lower().split())


def _stable_text_id(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return "empty"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


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


def _coerce_domain_list(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in values:
        token = _clean_text(item).lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _extract_domains(keywords: Any) -> List[str]:
    if not isinstance(keywords, dict):
        return []
    out: List[str] = []
    for section in ("research", "application"):
        sec = keywords.get(section)
        if not isinstance(sec, dict):
            continue
        out.extend(list(sec.get("domain") or []))
    return _coerce_domain_list(out)


def _coerce_vector(value: Any) -> List[float]:
    raw = value
    if raw is None:
        return []
    if hasattr(raw, "tolist"):
        try:
            raw = raw.tolist()
        except Exception:
            raw = value
    if isinstance(raw, str):
        t = _clean_text(raw)
        if not t:
            return []
        try:
            raw = json.loads(t)
        except Exception:
            return []
    if isinstance(raw, (list, tuple)):
        out: List[float] = []
        for x in raw:
            try:
                out.append(float(x))
            except Exception:
                return []
        return out
    return []


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms <= 0.0, 1.0, norms)
    return vectors / norms


def _build_distill_chain(model_id: str):
    try:
        from langchain_core.prompts import ChatPromptTemplate
    except Exception as e:
        raise RuntimeError("Missing dependency for LLM distillation. Install: pip install langchain-core") from e

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DISTILL_SYSTEM_PROMPT),
            ("human", DISTILL_HUMAN_PROMPT),
        ]
    )
    llm = get_llm_client(model_id).build()
    return prompt | llm.with_structured_output(CandidateLabelOut)


def _sanitize_indices(
    raw_indices: Sequence[Any],
    *,
    valid_indices: Sequence[int],
    reserved: Optional[Sequence[int]] = None,
) -> List[int]:
    valid_set = set(int(x) for x in list(valid_indices or []))
    reserved_set = set(int(x) for x in list(reserved or []))
    out: List[int] = []
    seen = set()
    for raw in list(raw_indices or []):
        try:
            idx = int(raw)
        except Exception:
            continue
        if idx not in valid_set or idx in reserved_set or idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def _fetch_faculty_specs_from_embedding_table(
    *,
    min_spec_weight: float,
    limit: int,
    embedding_model: str,
) -> List[SpecRow]:
    sql = text(
        """
        WITH limited_faculty AS (
            SELECT fk.faculty_id
            FROM faculty_keywords fk
            WHERE fk.keywords IS NOT NULL
            ORDER BY fk.faculty_id ASC
            LIMIT :limit
        )
        SELECT
            fse.faculty_id::text AS owner_id,
            lower(coalesce(f.email, '')) AS owner_email,
            fk.keywords AS keywords,
            fse.section AS section,
            fse.spec_text AS spec_text,
            fse.spec_norm AS spec_norm,
            fse.spec_weight AS spec_weight,
            fse.spec_vec AS spec_vec
        FROM faculty_specialization_embedding fse
        JOIN limited_faculty lf ON lf.faculty_id = fse.faculty_id
        JOIN faculty f ON f.faculty_id = fse.faculty_id
        LEFT JOIN faculty_keywords fk ON fk.faculty_id = fse.faculty_id
        WHERE fse.model = :embedding_model
          AND COALESCE(fse.spec_weight, 1.0) >= :min_spec_weight
        ORDER BY fse.faculty_id ASC, fse.id ASC
        """
    )
    out: List[SpecRow] = []
    with SessionLocal() as sess:
        rows = sess.execute(
            sql,
            {
                "limit": int(max(1, limit)),
                "embedding_model": _clean_text(embedding_model),
                "min_spec_weight": float(max(0.0, min_spec_weight)),
            },
        ).mappings().all()

    for row in rows:
        item = dict(row or {})
        owner_id = _clean_text(item.get("owner_id"))
        owner_email = _clean_text(item.get("owner_email")).lower()
        spec_text = _clean_text(item.get("spec_text"))
        spec_norm = _clean_text(item.get("spec_norm"))
        section = _clean_text(item.get("section")) or "research"
        spec_weight = float(item.get("spec_weight") if item.get("spec_weight") is not None else 1.0)
        embedding = _coerce_vector(item.get("spec_vec"))
        if not owner_id or not spec_text or not embedding:
            continue
        spec_node_id = f"{section}:{spec_norm or _stable_text_id(spec_text)}"
        keywords = item.get("keywords") or {}
        domains = _extract_domains(keywords)
        out.append(
            SpecRow(
                source="faculty",
                owner_id=owner_id,
                owner_email=owner_email,
                spec_node_id=spec_node_id,
                text=spec_text,
                spec_weight=float(spec_weight),
                domains=list(domains),
                embedding=embedding,
            )
        )
    return out


def _fetch_grant_specs_from_embedding_table(
    *,
    min_spec_weight: float,
    limit: int,
    embedding_model: str,
) -> List[SpecRow]:
    sql = text(
        """
        WITH limited_grants AS (
            SELECT ok.opportunity_id
            FROM opportunity_keywords ok
            WHERE ok.keywords IS NOT NULL
            ORDER BY ok.opportunity_id ASC
            LIMIT :limit
        )
        SELECT
            ose.opportunity_id::text AS owner_id,
            ok.keywords AS keywords,
            ose.section AS section,
            ose.spec_text AS spec_text,
            ose.spec_norm AS spec_norm,
            ose.spec_weight AS spec_weight,
            ose.spec_vec AS spec_vec
        FROM opportunity_specialization_embedding ose
        JOIN limited_grants lg ON lg.opportunity_id = ose.opportunity_id
        LEFT JOIN opportunity_keywords ok ON ok.opportunity_id = ose.opportunity_id
        WHERE ose.model = :embedding_model
          AND COALESCE(ose.spec_weight, 1.0) >= :min_spec_weight
        ORDER BY ose.opportunity_id ASC, ose.id ASC
        """
    )
    out: List[SpecRow] = []
    with SessionLocal() as sess:
        rows = sess.execute(
            sql,
            {
                "limit": int(max(1, limit)),
                "embedding_model": _clean_text(embedding_model),
                "min_spec_weight": float(max(0.0, min_spec_weight)),
            },
        ).mappings().all()

    for row in rows:
        item = dict(row or {})
        owner_id = _clean_text(item.get("owner_id"))
        spec_text = _clean_text(item.get("spec_text"))
        spec_norm = _clean_text(item.get("spec_norm"))
        section = _clean_text(item.get("section")) or "research"
        spec_weight = float(item.get("spec_weight") if item.get("spec_weight") is not None else 1.0)
        embedding = _coerce_vector(item.get("spec_vec"))
        if not owner_id or not spec_text or not embedding:
            continue
        spec_node_id = f"{section}:{spec_norm or _stable_text_id(spec_text)}"
        keywords = item.get("keywords") or {}
        domains = _extract_domains(keywords)
        out.append(
            SpecRow(
                source="grant",
                owner_id=owner_id,
                owner_email="",
                spec_node_id=spec_node_id,
                text=spec_text,
                spec_weight=float(spec_weight),
                domains=list(domains),
                embedding=embedding,
            )
        )
    return out


def _select_common_embedding_dim(
    faculty_specs: Sequence[SpecRow],
    grant_specs: Sequence[SpecRow],
) -> Tuple[int, List[SpecRow], List[SpecRow]]:
    fac_counter: Dict[int, int] = {}
    grant_counter: Dict[int, int] = {}
    for x in faculty_specs:
        if x.embedding:
            fac_counter[len(x.embedding)] = fac_counter.get(len(x.embedding), 0) + 1
    for x in grant_specs:
        if x.embedding:
            grant_counter[len(x.embedding)] = grant_counter.get(len(x.embedding), 0) + 1
    common_dims = [d for d in fac_counter if d in grant_counter]
    if not common_dims:
        return 0, [], []
    best_dim = max(common_dims, key=lambda d: (fac_counter[d] * grant_counter[d], fac_counter[d] + grant_counter[d], d))
    fac = [x for x in faculty_specs if len(x.embedding) == best_dim]
    grants = [x for x in grant_specs if len(x.embedding) == best_dim]
    return int(best_dim), fac, grants


def _build_query_candidate_sets(
    *,
    grant_specs: Sequence[SpecRow],
    faculty_specs: Sequence[SpecRow],
    candidate_pool_size: int,
    max_queries: int,
    seed: int,
) -> List[Dict[str, Any]]:
    safe_pool = _safe_limit(candidate_pool_size, default=DEFAULT_CANDIDATE_POOL_SIZE, minimum=8, maximum=128)
    safe_max_queries = _safe_limit(max_queries, default=DEFAULT_MAX_QUERIES, minimum=1, maximum=2_000_000)
    rng = random.Random(int(seed))

    fac_mat = np.asarray([x.embedding for x in faculty_specs], dtype=np.float32)
    grant_mat = np.asarray([x.embedding for x in grant_specs], dtype=np.float32)
    fac_mat = _normalize_rows(fac_mat)
    grant_mat = _normalize_rows(grant_mat)

    query_rows: List[Dict[str, Any]] = []
    easy_reserve = 2
    for g_idx, grant in enumerate(grant_specs):
        sims = np.dot(fac_mat, grant_mat[g_idx])
        ordered = np.argsort(-sims).tolist()
        if not ordered:
            continue

        top_keep = max(6, safe_pool - easy_reserve)
        top_idx = [int(x) for x in ordered[: min(len(ordered), top_keep)]]
        tail_pool = [int(x) for x in ordered[max(len(top_idx), safe_pool) :]]
        easy_idx: List[int] = []
        if tail_pool:
            easy_idx = rng.sample(tail_pool, min(easy_reserve, len(tail_pool)))

        candidate_idx = list(dict.fromkeys(top_idx + easy_idx))
        if len(candidate_idx) < 8:
            for idx in ordered:
                i = int(idx)
                if i in candidate_idx:
                    continue
                candidate_idx.append(i)
                if len(candidate_idx) >= 8:
                    break
        candidate_idx = candidate_idx[:safe_pool]
        if len(candidate_idx) < 4:
            continue

        easy_set = set(easy_idx)
        candidates: List[Dict[str, Any]] = []
        for local_i, fac_i in enumerate(candidate_idx, start=1):
            fac = faculty_specs[int(fac_i)]
            prior_bucket = "easy_pool" if int(fac_i) in easy_set else "hard_pool"
            candidates.append(
                {
                    "i": int(local_i),
                    "faculty_idx": int(fac_i),
                    "faculty_spec_id": fac.spec_id,
                    "faculty_id": fac.owner_id,
                    "faculty_email": fac.owner_email,
                    "faculty_spec_weight": float(fac.spec_weight),
                    "faculty_domains": list(fac.domains),
                    "text": fac.text,
                    "cosine_sim": float(sims[int(fac_i)]),
                    "prior_bucket": prior_bucket,
                }
            )

        query_rows.append(
            {
                "query_spec_id": grant.spec_id,
                "grant_id": grant.owner_id,
                "grant_domains": list(grant.domains),
                "grant_spec_weight": float(grant.spec_weight),
                "query": grant.text,
                "candidates": candidates,
            }
        )

    if len(query_rows) <= safe_max_queries:
        return query_rows
    rng.shuffle(query_rows)
    return query_rows[:safe_max_queries]


def _fallback_label_for_query(query_set: Dict[str, Any]) -> Dict[str, List[int]]:
    candidates = list(query_set.get("candidates") or [])
    ordered = sorted(candidates, key=lambda x: float(x.get("cosine_sim") or 0.0), reverse=True)
    idxs = [int(x.get("i") or 0) for x in ordered if int(x.get("i") or 0) > 0]
    if not idxs:
        return {
            "positive_indices": [],
            "hard_negative_indices": [],
            "easy_negative_indices": [],
        }
    pos = idxs[: min(2, len(idxs))]
    rem = [i for i in idxs if i not in set(pos)]
    hard = rem[: min(8, len(rem))]
    rem2 = [i for i in rem if i not in set(hard)]
    easy = rem2[-min(2, len(rem2)) :] if rem2 else []
    return {
        "positive_indices": pos,
        "hard_negative_indices": hard,
        "easy_negative_indices": easy,
    }


def _label_query_sets_with_llm(
    *,
    query_sets: Sequence[Dict[str, Any]],
    llm_model: str,
    llm_batch_size: int,
    llm_max_retries: int,
    llm_workers: int,
    save_partial_on_error: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    safe_batch = _safe_limit(llm_batch_size, default=DEFAULT_LLM_BATCH_SIZE, minimum=1, maximum=64)
    safe_retries = _safe_limit(llm_max_retries, default=DEFAULT_LLM_MAX_RETRIES, minimum=1, maximum=8)
    batches: List[List[Dict[str, Any]]] = []
    rows = list(query_sets or [])
    for i in range(0, len(rows), safe_batch):
        batches.append(rows[i : i + safe_batch])

    lock = threading.Lock()
    stats = {
        "queries_total": int(len(rows)),
        "queries_labeled": 0,
        "llm_calls": 0,
        "llm_failures": 0,
        "fallback_queries": 0,
        "retries_used": 0,
        "batches": int(len(batches)),
        "llm_model": _clean_text(llm_model),
    }

    get_chain = build_thread_local_getter(lambda: _build_distill_chain(llm_model))

    def _run_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tasks_payload = []
        for local_q, row in enumerate(batch, start=1):
            tasks_payload.append(
                {
                    "q": int(local_q),
                    "query": _clean_text(row.get("query")),
                    "grant_domains": list(row.get("grant_domains") or []),
                    "candidates": [
                        {
                            "i": int(c.get("i") or 0),
                            "text": _clean_text(c.get("text")),
                            "cosine_sim": float(c.get("cosine_sim") or 0.0),
                            "domains": list(c.get("faculty_domains") or []),
                        }
                        for c in list(row.get("candidates") or [])
                    ],
                }
            )

        last_error: Optional[Exception] = None
        parsed_items: Dict[int, Dict[str, Any]] = {}
        for attempt in range(safe_retries):
            with lock:
                stats["llm_calls"] += 1
            if attempt > 0:
                with lock:
                    stats["retries_used"] += 1
            try:
                out = get_chain().invoke({"tasks_json": json.dumps(tasks_payload, ensure_ascii=False)})
                items = list(getattr(out, "items", []) or [])
                parsed_items = {}
                for it in items:
                    payload = it.model_dump() if hasattr(it, "model_dump") else dict(it or {})
                    q = int(payload.get("q") or 0)
                    if q <= 0:
                        continue
                    parsed_items[q] = payload
                break
            except Exception as e:
                last_error = e
                continue

        batch_out: List[Dict[str, Any]] = []
        for local_q, row in enumerate(batch, start=1):
            raw = parsed_items.get(int(local_q))
            valid_idx = [int(c.get("i") or 0) for c in list(row.get("candidates") or []) if int(c.get("i") or 0) > 0]
            valid_set = set(valid_idx)

            if raw is None:
                with lock:
                    stats["fallback_queries"] += 1
                if not save_partial_on_error:
                    raise RuntimeError(
                        "LLM labeling failed and save_partial_on_error is disabled. "
                        f"last_error={last_error}"
                    )
                fallback = _fallback_label_for_query(row)
                pos_idx = list(fallback["positive_indices"])
                hard_idx = list(fallback["hard_negative_indices"])
                easy_idx = list(fallback["easy_negative_indices"])
                label_source = "cosine_fallback"
            else:
                skip = bool(raw.get("skip"))
                if skip:
                    pos_idx = []
                    hard_idx = []
                    easy_idx = []
                else:
                    pos_idx = _sanitize_indices(raw.get("positive_indices") or [], valid_indices=valid_idx)
                    hard_idx = _sanitize_indices(
                        raw.get("hard_negative_indices") or [],
                        valid_indices=valid_idx,
                        reserved=pos_idx,
                    )
                    easy_idx = _sanitize_indices(
                        raw.get("easy_negative_indices") or [],
                        valid_indices=valid_idx,
                        reserved=(list(pos_idx) + list(hard_idx)),
                    )
                label_source = "llm_distill"

            if not pos_idx and valid_idx:
                pos_idx = [valid_idx[0]]
                label_source = "cosine_fallback"

            if not hard_idx:
                rem = [i for i in valid_idx if i not in set(pos_idx)]
                hard_idx = rem[: min(8, len(rem))]
                if hard_idx:
                    label_source = "cosine_fallback"

            if not easy_idx:
                rem = [i for i in valid_idx if i not in set(pos_idx) and i not in set(hard_idx)]
                easy_idx = rem[-min(2, len(rem)) :] if rem else []

            candidates = list(row.get("candidates") or [])
            by_i = {int(c.get("i") or 0): c for c in candidates}
            pos_idx = [i for i in pos_idx if i in valid_set and i in by_i]
            hard_idx = [i for i in hard_idx if i in valid_set and i in by_i and i not in set(pos_idx)]
            easy_idx = [
                i
                for i in easy_idx
                if i in valid_set and i in by_i and i not in set(pos_idx) and i not in set(hard_idx)
            ]

            batch_out.append(
                {
                    **row,
                    "positive_indices": pos_idx,
                    "hard_negative_indices": hard_idx,
                    "easy_negative_indices": easy_idx,
                    "label_source": label_source,
                }
            )

        if parsed_items == {}:
            with lock:
                stats["llm_failures"] += 1

        return batch_out

    safe_workers = resolve_pool_size(
        max_workers=_safe_limit(llm_workers, default=DEFAULT_LLM_WORKERS, minimum=1, maximum=64),
        task_count=max(1, len(batches)),
    )
    results = parallel_map(
        batches,
        max_workers=safe_workers,
        run_item=_run_batch,
    )

    labeled: List[Dict[str, Any]] = []
    for part in list(results or []):
        labeled.extend(list(part or []))
    stats["queries_labeled"] = int(len(labeled))
    stats["fallback_query_ratio"] = (
        float(stats["fallback_queries"]) / float(stats["queries_labeled"])
        if stats["queries_labeled"] > 0
        else 0.0
    )
    return labeled, stats


def _expand_to_pair_and_triplet_rows(
    *,
    labeled_queries: Sequence[Dict[str, Any]],
    max_pairs: int,
    seed: int,
    min_teacher_margin: float,
    hard_neg_max_cosine_gap: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(int(seed))
    safe_max_pairs = _safe_limit(max_pairs, default=DEFAULT_MAX_PAIRS, minimum=0, maximum=10_000_000)
    safe_min_margin = float(max(0.0, min_teacher_margin))
    safe_hard_gap = float(max(0.0, hard_neg_max_cosine_gap))

    pair_rows: List[Dict[str, Any]] = []
    triplet_rows: List[Dict[str, Any]] = []
    skipped_missing = 0
    skipped_overlap = 0
    skipped_easy_hard = 0
    skipped_margin = 0
    skipped_hard_gap = 0

    for qrow in list(labeled_queries or []):
        candidates = list(qrow.get("candidates") or [])
        by_i = {int(c.get("i") or 0): c for c in candidates}
        pos_idx = [int(i) for i in list(qrow.get("positive_indices") or []) if int(i) in by_i]
        hard_idx = [int(i) for i in list(qrow.get("hard_negative_indices") or []) if int(i) in by_i]
        easy_idx = [int(i) for i in list(qrow.get("easy_negative_indices") or []) if int(i) in by_i]
        label_source = _clean_text(qrow.get("label_source")) or "unknown"

        if not pos_idx or not hard_idx:
            skipped_missing += 1
            continue

        for pos_rank, pos_i in enumerate(pos_idx, start=1):
            pos = by_i.get(int(pos_i))
            if pos is None:
                skipped_missing += 1
                continue
            pos_text = _clean_text(pos.get("text"))
            if not pos_text:
                skipped_missing += 1
                continue

            pos_score = max(0.70, POS_TEACHER_SCORE_TOP1 - float(pos_rank - 1) * POS_TEACHER_SCORE_DECAY)
            pos_cos = float(pos.get("cosine_sim") or 0.0)

            neg_plan: List[Tuple[int, str, float]] = []
            for i in hard_idx:
                neg_plan.append((int(i), "llm_hard_negative", HARD_NEG_TEACHER_SCORE))
            for i in easy_idx:
                neg_plan.append((int(i), "llm_easy_negative", EASY_NEG_TEACHER_SCORE))

            for neg_i, neg_type, neg_score in neg_plan:
                neg = by_i.get(int(neg_i))
                if neg is None:
                    skipped_missing += 1
                    continue
                neg_text = _clean_text(neg.get("text"))
                if not neg_text:
                    skipped_missing += 1
                    continue
                if _normalize_text_key(pos_text) == _normalize_text_key(neg_text):
                    skipped_overlap += 1
                    continue
                if int(pos_i) == int(neg_i):
                    skipped_overlap += 1
                    continue

                neg_cos = float(neg.get("cosine_sim") or 0.0)
                if neg_type == "llm_hard_negative":
                    if (pos_cos - neg_cos) > safe_hard_gap:
                        skipped_hard_gap += 1
                        continue
                elif neg_type == "llm_easy_negative":
                    if neg_cos > pos_cos:
                        skipped_easy_hard += 1
                        continue

                margin = float(pos_score - neg_score)
                if margin < safe_min_margin:
                    skipped_margin += 1
                    continue

                row = {
                    "distill_method": "llm_index_distill_v2",
                    "label": 1,
                    "query": _clean_text(qrow.get("query")),
                    "positive": pos_text,
                    "negative": neg_text,
                    "query_spec_id": _clean_text(qrow.get("query_spec_id")),
                    "grant_id": _clean_text(qrow.get("grant_id")),
                    "grant_domains": list(qrow.get("grant_domains") or []),
                    "grant_spec_weight": _safe_unit_float(qrow.get("grant_spec_weight"), default=0.0),
                    "positive_spec_id": _clean_text(pos.get("faculty_spec_id")),
                    "negative_spec_id": _clean_text(neg.get("faculty_spec_id")),
                    "positive_faculty_id": _clean_text(pos.get("faculty_id")),
                    "negative_faculty_id": _clean_text(neg.get("faculty_id")),
                    "positive_faculty_email": _clean_text(pos.get("faculty_email")).lower(),
                    "negative_faculty_email": _clean_text(neg.get("faculty_email")).lower(),
                    "positive_domains": list(pos.get("faculty_domains") or []),
                    "negative_domains": list(neg.get("faculty_domains") or []),
                    "positive_spec_weight": _safe_unit_float(pos.get("faculty_spec_weight"), default=0.0),
                    "negative_spec_weight": _safe_unit_float(neg.get("faculty_spec_weight"), default=0.0),
                    "positive_rank": int(pos_rank),
                    "negative_rank": int(neg_i),
                    "candidate_count": int(len(candidates)),
                    "positive_teacher_score": float(pos_score),
                    "negative_teacher_score": float(neg_score),
                    "teacher_margin": float(margin),
                    "preference_strength": float(margin),
                    "positive_cosine_sim": float(pos_cos),
                    "negative_cosine_sim": float(neg_cos),
                    "positive_candidate_type": "llm_positive",
                    "negative_candidate_type": neg_type,
                    "label_source": label_source,
                }
                pair_rows.append(row)
                triplet_rows.append(
                    {
                        "query": row["query"],
                        "positive": row["positive"],
                        "negative": row["negative"],
                        "query_spec_id": row["query_spec_id"],
                        "positive_spec_id": row["positive_spec_id"],
                        "negative_spec_id": row["negative_spec_id"],
                        "negative_candidate_type": row["negative_candidate_type"],
                        "positive_teacher_score": row["positive_teacher_score"],
                        "negative_teacher_score": row["negative_teacher_score"],
                        "teacher_margin": row["teacher_margin"],
                        "label_source": row["label_source"],
                    }
                )

    rng.shuffle(pair_rows)
    rng.shuffle(triplet_rows)
    if safe_max_pairs > 0 and len(pair_rows) > safe_max_pairs:
        pair_rows = pair_rows[:safe_max_pairs]
    if safe_max_pairs > 0 and len(triplet_rows) > safe_max_pairs:
        triplet_rows = triplet_rows[:safe_max_pairs]

    meta = {
        "pairs_generated_before_cap": int(len(pair_rows)),
        "pairs_saved_after_cap": int(len(pair_rows)),
        "triplets_saved_after_cap": int(len(triplet_rows)),
        "skipped_missing": int(skipped_missing),
        "skipped_overlap": int(skipped_overlap),
        "skipped_easy_hard": int(skipped_easy_hard),
        "skipped_margin": int(skipped_margin),
        "skipped_hard_gap": int(skipped_hard_gap),
    }
    return pair_rows, triplet_rows, meta


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_outputs(
    *,
    pair_rows: Sequence[Dict[str, Any]],
    triplet_rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    output_prefix: str,
    meta: Dict[str, Any],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = f"{_clean_text(output_prefix) or DEFAULT_OUTPUT_PREFIX}_{ts}"

    pair_jsonl_path = output_dir / f"{stem}.jsonl"
    triplet_jsonl_path = output_dir / f"{stem}.triplets.jsonl"
    meta_path = output_dir / f"{stem}.meta.json"

    with pair_jsonl_path.open("w", encoding="utf-8") as f:
        for row in list(pair_rows or []):
            f.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")

    with triplet_jsonl_path.open("w", encoding="utf-8") as f:
        for row in list(triplet_rows or []):
            f.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")

    meta_payload = {
        **dict(meta or {}),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "jsonl_path": str(pair_jsonl_path),
        "triplet_jsonl_path": str(triplet_jsonl_path),
        "row_count": int(len(list(pair_rows or []))),
        "triplet_row_count": int(len(list(triplet_rows or []))),
    }
    meta_path.write_text(json.dumps(json_ready(meta_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "jsonl_path": str(pair_jsonl_path),
        "triplet_jsonl_path": str(triplet_jsonl_path),
        "meta_path": str(meta_path),
    }


def build_dataset(
    *,
    llm_model: str,
    llm_batch_size: int,
    llm_max_retries: int,
    llm_workers: int,
    save_partial_on_error: bool,
    embedding_model: str,
    faculty_min_spec_weight: float,
    grant_min_spec_weight: float,
    faculty_limit: int,
    grant_limit: int,
    candidate_pool_size: int,
    max_queries: int,
    max_pairs: int,
    min_teacher_margin: float,
    hard_neg_max_cosine_gap: float,
    seed: int,
    output_dir: Path,
    output_prefix: str,
) -> Dict[str, Any]:
    safe_model = _clean_text(llm_model) or _clean_text(settings.haiku)
    if not safe_model:
        raise RuntimeError("Missing --llm-model and settings.haiku is empty.")
    safe_embedding_model = _clean_text(embedding_model) or _clean_text(settings.bedrock_embed_model_id)
    if not safe_embedding_model:
        raise RuntimeError("Missing --embedding-model and settings.bedrock_embed_model_id is empty.")

    safe_fac_limit = _safe_limit(faculty_limit, default=DEFAULT_FACULTY_LIMIT, minimum=1, maximum=2_000_000)
    safe_grant_limit = _safe_limit(grant_limit, default=DEFAULT_GRANT_LIMIT, minimum=1, maximum=2_000_000)
    safe_seed = int(seed)

    faculty_specs_raw = _fetch_faculty_specs_from_embedding_table(
        min_spec_weight=float(max(0.0, faculty_min_spec_weight)),
        limit=safe_fac_limit,
        embedding_model=safe_embedding_model,
    )
    grant_specs_raw = _fetch_grant_specs_from_embedding_table(
        min_spec_weight=float(max(0.0, grant_min_spec_weight)),
        limit=safe_grant_limit,
        embedding_model=safe_embedding_model,
    )

    common_dim, faculty_specs, grant_specs = _select_common_embedding_dim(faculty_specs_raw, grant_specs_raw)
    if common_dim <= 0 or not faculty_specs or not grant_specs:
        raise RuntimeError(
            "No shared embedding dimension found between faculty and grant specs from embedding tables. "
            "Check embedding model id and table coverage."
        )

    query_sets = _build_query_candidate_sets(
        grant_specs=grant_specs,
        faculty_specs=faculty_specs,
        candidate_pool_size=int(candidate_pool_size),
        max_queries=int(max_queries),
        seed=safe_seed,
    )
    if not query_sets:
        raise RuntimeError("No query candidate sets built. Check DB coverage or limits.")

    labeled_queries, llm_meta = _label_query_sets_with_llm(
        query_sets=query_sets,
        llm_model=safe_model,
        llm_batch_size=int(llm_batch_size),
        llm_max_retries=int(llm_max_retries),
        llm_workers=int(llm_workers),
        save_partial_on_error=bool(save_partial_on_error),
    )
    if not labeled_queries:
        raise RuntimeError("No labeled queries produced.")

    pair_rows, triplet_rows, pair_meta = _expand_to_pair_and_triplet_rows(
        labeled_queries=labeled_queries,
        max_pairs=int(max_pairs),
        seed=safe_seed,
        min_teacher_margin=float(min_teacher_margin),
        hard_neg_max_cosine_gap=float(hard_neg_max_cosine_gap),
    )
    if not pair_rows:
        raise RuntimeError(
            "No pair rows generated after denoising. "
            "Try loosening --hard-neg-max-cosine-gap or lowering --min-teacher-margin."
        )

    label_source_counts: Dict[str, int] = {}
    neg_type_counts: Dict[str, int] = {}
    for row in pair_rows:
        src = _clean_text(row.get("label_source"))
        neg_type = _clean_text(row.get("negative_candidate_type"))
        label_source_counts[src] = label_source_counts.get(src, 0) + 1
        neg_type_counts[neg_type] = neg_type_counts.get(neg_type, 0) + 1

    paths = _save_outputs(
        pair_rows=pair_rows,
        triplet_rows=triplet_rows,
        output_dir=output_dir,
        output_prefix=output_prefix,
        meta={
            "params": {
                "llm_model": safe_model,
                "llm_batch_size": int(llm_batch_size),
                "llm_max_retries": int(llm_max_retries),
                "llm_workers": int(llm_workers),
                "save_partial_on_error": bool(save_partial_on_error),
                "embedding_model": safe_embedding_model,
                "faculty_min_spec_weight": float(max(0.0, faculty_min_spec_weight)),
                "grant_min_spec_weight": float(max(0.0, grant_min_spec_weight)),
                "faculty_limit": int(safe_fac_limit),
                "grant_limit": int(safe_grant_limit),
                "candidate_pool_size": int(candidate_pool_size),
                "max_queries": int(max_queries),
                "max_pairs": int(max_pairs),
                "min_teacher_margin": float(min_teacher_margin),
                "hard_neg_max_cosine_gap": float(hard_neg_max_cosine_gap),
                "seed": int(safe_seed),
            },
            "counts": {
                "faculty_specs_raw": int(len(faculty_specs_raw)),
                "grant_specs_raw": int(len(grant_specs_raw)),
                "embedding_common_dim": int(common_dim),
                "faculty_specs_used": int(len(faculty_specs)),
                "grant_specs_used": int(len(grant_specs)),
                "query_sets_built": int(len(query_sets)),
                "queries_labeled": int(len(labeled_queries)),
                "pairs_saved_after_cap": int(len(pair_rows)),
                "triplets_saved_after_cap": int(len(triplet_rows)),
                "label_source_counts": dict(label_source_counts),
                "negative_candidate_type_counts": dict(neg_type_counts),
                **dict(pair_meta or {}),
            },
            "llm": dict(llm_meta or {}),
        },
    )

    return {
        "params": {
            "llm_model": safe_model,
            "embedding_model": safe_embedding_model,
            "candidate_pool_size": int(candidate_pool_size),
            "max_queries": int(max_queries),
            "max_pairs": int(max_pairs),
            "seed": int(safe_seed),
        },
        "counts": {
            "faculty_specs_raw": int(len(faculty_specs_raw)),
            "grant_specs_raw": int(len(grant_specs_raw)),
            "embedding_common_dim": int(common_dim),
            "faculty_specs_used": int(len(faculty_specs)),
            "grant_specs_used": int(len(grant_specs)),
            "query_sets_built": int(len(query_sets)),
            "queries_labeled": int(len(labeled_queries)),
            "pairs_saved_after_cap": int(len(pair_rows)),
            "triplets_saved_after_cap": int(len(triplet_rows)),
            "label_source_counts": dict(label_source_counts),
            "negative_candidate_type_counts": dict(neg_type_counts),
            **dict(pair_meta or {}),
        },
        "llm": dict(llm_meta or {}),
        "output": paths,
    }


def _build_parser() -> argparse.ArgumentParser:
    default_output_dir = Path(__file__).resolve().parent / "dataset"
    p = argparse.ArgumentParser(
        description=(
            "Build v2 LLM-distilled cross-encoder dataset with denoised hard negatives "
            "using real faculty/grant specialization candidates from embedding tables."
        )
    )
    p.add_argument("--llm-model", type=str, default=(settings.haiku or "").strip(), help="Bedrock model id for distillation.")
    p.add_argument("--llm-batch-size", type=int, default=DEFAULT_LLM_BATCH_SIZE, help="Queries per LLM call.")
    p.add_argument("--llm-max-retries", type=int, default=DEFAULT_LLM_MAX_RETRIES, help="Retries per LLM batch.")
    p.add_argument("--llm-workers", type=int, default=DEFAULT_LLM_WORKERS, help="Parallel LLM batch workers.")
    p.add_argument(
        "--save-partial-on-error",
        dest="save_partial_on_error",
        action="store_true",
        help="Fallback to cosine labels when LLM batch fails (default: enabled).",
    )
    p.add_argument(
        "--no-save-partial-on-error",
        dest="save_partial_on_error",
        action="store_false",
        help="Fail hard when an LLM batch fails.",
    )
    p.set_defaults(save_partial_on_error=True)

    p.add_argument(
        "--embedding-model",
        type=str,
        default=(settings.bedrock_embed_model_id or "").strip(),
        help="Embedding model id filter for specialization embedding tables.",
    )
    p.add_argument("--faculty-min-spec-weight", type=float, default=0.0, help="Minimum faculty specialization weight.")
    p.add_argument("--grant-min-spec-weight", type=float, default=0.0, help="Minimum grant specialization weight.")
    p.add_argument("--faculty-limit", type=int, default=DEFAULT_FACULTY_LIMIT, help="Max faculty rows to read.")
    p.add_argument("--grant-limit", type=int, default=DEFAULT_GRANT_LIMIT, help="Max grant rows to read.")

    p.add_argument("--candidate-pool-size", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE, help="Candidate count per query sent to LLM.")
    p.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES, help="Max query specs to distill.")
    p.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS, help="Cap pair rows saved (0 = all).")
    p.add_argument("--min-teacher-margin", type=float, default=MIN_TEACHER_MARGIN, help="Minimum positive-negative teacher margin.")
    p.add_argument("--hard-neg-max-cosine-gap", type=float, default=HARD_NEG_MAX_COSINE_GAP, help="Max (pos_cos - hard_neg_cos) to keep hard negatives sharp.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    p.add_argument("--output-dir", type=str, default=str(default_output_dir), help="Output directory.")
    p.add_argument("--output-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX, help="Output file prefix.")
    p.add_argument("--json-only", action="store_true", help="Print JSON payload only.")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    payload = build_dataset(
        llm_model=_clean_text(args.llm_model),
        llm_batch_size=int(args.llm_batch_size),
        llm_max_retries=int(args.llm_max_retries),
        llm_workers=int(args.llm_workers),
        save_partial_on_error=bool(args.save_partial_on_error),
        embedding_model=_clean_text(args.embedding_model),
        faculty_min_spec_weight=float(args.faculty_min_spec_weight),
        grant_min_spec_weight=float(args.grant_min_spec_weight),
        faculty_limit=int(args.faculty_limit),
        grant_limit=int(args.grant_limit),
        candidate_pool_size=int(args.candidate_pool_size),
        max_queries=int(args.max_queries),
        max_pairs=int(args.max_pairs),
        min_teacher_margin=float(args.min_teacher_margin),
        hard_neg_max_cosine_gap=float(args.hard_neg_max_cosine_gap),
        seed=int(args.seed),
        output_dir=Path(_clean_text(args.output_dir)),
        output_prefix=_clean_text(args.output_prefix) or DEFAULT_OUTPUT_PREFIX,
    )

    if not bool(args.json_only):
        print("v2 LLM distillation dataset build complete.")
        print(f"  queries labeled         : {payload.get('counts', {}).get('queries_labeled', 0)}")
        print(f"  pairs saved             : {payload.get('counts', {}).get('pairs_saved_after_cap', 0)}")
        print(f"  triplets saved          : {payload.get('counts', {}).get('triplets_saved_after_cap', 0)}")
        print(f"  label sources           : {payload.get('counts', {}).get('label_source_counts', {})}")
        print(f"  output jsonl            : {payload.get('output', {}).get('jsonl_path', '')}")
        print(f"  output triplets         : {payload.get('output', {}).get('triplet_jsonl_path', '')}")
        print()

    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
