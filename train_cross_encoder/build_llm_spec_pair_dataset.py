from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import threading
from collections import Counter
from dataclasses import dataclass, replace
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

from config import get_embedding_client, get_llm_client, settings
from db.db_conn import SessionLocal
from utils.thread_pool import build_thread_local_getter, parallel_map, resolve_pool_size
from utils.embedder import embed_texts


class RankedQueryRow(BaseModel):
    q: int = Field(..., description="1-based query index within batch input")
    ranked: List[int] = Field(default_factory=list, description="Candidate indices in best->worst order")


class RankedQueryOut(BaseModel):
    items: List[RankedQueryRow] = Field(default_factory=list)


RANK_BATCH_SYSTEM_PROMPT = (
    "You are evaluating whether candidate research specializations support a query research specialization.\n"
    "\n"
    "Goal:\n"
    "For each query specialization, rank candidate specializations from MOST supportive to LEAST supportive.\n"
    "\n"
    "A candidate is supportive if it:\n"
    "- directly addresses the research topic\n"
    "- provides methods or techniques used for the topic\n"
    "- represents a closely related research direction\n"
    "\n"
    "A candidate is less relevant if it:\n"
    "- is from a different scientific domain\n"
    "- only shares generic AI/ML terminology\n"
    "- does not meaningfully contribute to the research goal\n"
    "\n"
    "Instructions:\n"
    "- Rank candidates by semantic support for the query specialization.\n"
    "- Do NOT explain your reasoning.\n"
    "- Return JSON only.\n"
    "Output schema:\n"
    "{{\n"
    '  "items": [\n'
    '    {{"q": 1, "ranked": [3,1,2]}}\n'
    "  ]\n"
    "}}\n"
    "Rules:\n"
    "- Each ranked list must contain every candidate index exactly once.\n"
    "- Preserve the candidate index space provided in the task.\n"
    "- Always produce a ranking even if relevance is weak.\n"
)


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


# Hard-negative focused defaults.
DEFAULT_TOP_K_CANDIDATES = 2
DEFAULT_HARD_NEGATIVES_PER_GRANT = 6
DEFAULT_RANDOM_NEGATIVES_PER_GRANT = 2
DEFAULT_CANDIDATES_PER_QUERY = 10
DEFAULT_MAX_PAIRS = 200000

# Teacher label sharpening.
HARD_NEGATIVE_POS_FLOOR = 0.80
HARD_NEGATIVE_NEG_CAP = 0.20
HARD_NEGATIVE_MIN_MARGIN = 0.50
MINED_FALSE_POSITIVE_POS_FLOOR = 0.90
MINED_FALSE_POSITIVE_NEG_CAP = 0.10
MINED_FALSE_POSITIVE_MIN_MARGIN = 0.70
GENERAL_MIN_MARGIN = 0.15

# Probe-mining.
DEFAULT_MINED_MAX_ROWS = 30000
PROBE_NEGATIVES_PER_CASE = 2
PROBE_POSITIVES_PER_CASE = 2


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_text_key(value: Any) -> str:
    return _clean_text(value).lower()


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
    if math.isnan(parsed) or math.isinf(parsed):
        return float(default)
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
        for item in value:
            try:
                out.append(float(item))
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


def _coerce_domain_list(values: Iterable[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in values:
        token = _clean_text(item).lower()
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


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


def _fetch_faculty_specs(
    *,
    min_spec_weight: float,
    limit: int,
) -> List[SpecRow]:
    sql = text(
        """
        SELECT
            f.faculty_id::text AS owner_id,
            lower(coalesce(f.email, '')) AS owner_email,
            fk.keywords AS keywords
        FROM faculty_keywords fk
        JOIN faculty f ON f.faculty_id = fk.faculty_id
        WHERE fk.keywords IS NOT NULL
        ORDER BY f.faculty_id ASC
        LIMIT :limit
        """
    )
    out: List[SpecRow] = []
    with SessionLocal() as sess:
        rows = sess.execute(sql, {"limit": int(max(1, limit))}).mappings().all()
    min_w = float(max(0.0, min_spec_weight))
    for row in rows:
        item = dict(row or {})
        owner_id = _clean_text(item.get("owner_id"))
        owner_email = _clean_text(item.get("owner_email")).lower()
        keywords = item.get("keywords") or {}
        domains = _extract_domains(keywords)
        for spec_text, spec_weight, section, idx in _extract_weighted_specs(keywords):
            if float(spec_weight) < min_w:
                continue
            out.append(
                SpecRow(
                    source="faculty",
                    owner_id=owner_id,
                    owner_email=owner_email,
                    spec_node_id=f"{section}:{idx}",
                    text=spec_text,
                    spec_weight=float(spec_weight),
                    domains=list(domains),
                    embedding=[],
                )
            )
    return out


def _fetch_grant_specs(
    *,
    min_spec_weight: float,
    limit: int,
) -> List[SpecRow]:
    sql = text(
        """
        SELECT
            o.opportunity_id::text AS owner_id,
            ok.keywords AS keywords
        FROM opportunity_keywords ok
        JOIN opportunity o ON o.opportunity_id = ok.opportunity_id
        WHERE ok.keywords IS NOT NULL
        ORDER BY o.opportunity_id ASC
        LIMIT :limit
        """
    )
    out: List[SpecRow] = []
    with SessionLocal() as sess:
        rows = sess.execute(sql, {"limit": int(max(1, limit))}).mappings().all()
    min_w = float(max(0.0, min_spec_weight))
    for row in rows:
        item = dict(row or {})
        owner_id = _clean_text(item.get("owner_id"))
        keywords = item.get("keywords") or {}
        domains = _extract_domains(keywords)
        for spec_text, spec_weight, section, idx in _extract_weighted_specs(keywords):
            if float(spec_weight) < min_w:
                continue
            out.append(
                SpecRow(
                    source="grant",
                    owner_id=owner_id,
                    owner_email="",
                    spec_node_id=f"{section}:{idx}",
                    text=spec_text,
                    spec_weight=float(spec_weight),
                    domains=list(domains),
                    embedding=[],
                )
            )
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
        section = _clean_text(item.get("section"))
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
        section = _clean_text(item.get("section"))
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


def _embed_spec_rows(
    specs: Sequence[SpecRow],
    *,
    batch_size: int,
    max_workers: int,
) -> List[SpecRow]:
    rows = list(specs or [])
    if not rows:
        return []
    safe_batch = _safe_limit(batch_size, default=64, minimum=1, maximum=512)
    safe_workers = resolve_pool_size(
        max_workers=max(1, int(max_workers)),
        task_count=max(1, int((len(rows) + safe_batch - 1) // safe_batch)),
    )

    unique_texts: List[str] = []
    seen = set()
    for row in rows:
        t = _clean_text(row.text)
        if not t or t in seen:
            continue
        seen.add(t)
        unique_texts.append(t)

    chunk_payloads: List[List[str]] = []
    for i in range(0, len(unique_texts), safe_batch):
        chunk_payloads.append(unique_texts[i : i + safe_batch])

    get_embedder = build_thread_local_getter(lambda: get_embedding_client().build())

    def _run_chunk(chunk: List[str]) -> List[Tuple[str, List[float]]]:
        vecs = embed_texts(chunk, embedding_client=get_embedder())
        if vecs.ndim != 2 or vecs.shape[0] != len(chunk):
            raise RuntimeError("Embedding batch returned invalid shape.")
        return [(t, [float(x) for x in vecs[j].tolist()]) for j, t in enumerate(chunk)]

    by_text: Dict[str, List[float]] = {}
    chunk_results = parallel_map(
        chunk_payloads,
        max_workers=safe_workers,
        run_item=_run_chunk,
    )
    for pairs in list(chunk_results or []):
        for t, v in list(pairs or []):
            by_text[t] = list(v or [])

    out: List[SpecRow] = []
    for row in rows:
        vec = by_text.get(_clean_text(row.text)) or []
        if not vec:
            continue
        out.append(replace(row, embedding=vec))
    return out


def _select_common_embedding_dim(
    faculty_specs: Sequence[SpecRow],
    grant_specs: Sequence[SpecRow],
) -> Tuple[int, List[SpecRow], List[SpecRow]]:
    fac_counter = Counter(len(x.embedding) for x in faculty_specs if x.embedding)
    grant_counter = Counter(len(x.embedding) for x in grant_specs if x.embedding)
    common_dims = [d for d in fac_counter if d in grant_counter]
    if not common_dims:
        return 0, [], []
    best_dim = max(common_dims, key=lambda d: (fac_counter[d] * grant_counter[d], fac_counter[d] + grant_counter[d], d))
    fac = [x for x in faculty_specs if len(x.embedding) == best_dim]
    grants = [x for x in grant_specs if len(x.embedding) == best_dim]
    return int(best_dim), fac, grants


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms <= 0.0, 1.0, norms)
    return vectors / norms


def _resolve_candidate_mix(
    *,
    top_k: int,
    hard_neg_per_query: int,
    random_neg_per_query: int,
    candidates_per_query: int,
) -> Dict[str, int]:
    safe_top_k = _safe_limit(top_k, default=DEFAULT_TOP_K_CANDIDATES, minimum=1, maximum=64)
    safe_hard = _safe_limit(hard_neg_per_query, default=DEFAULT_HARD_NEGATIVES_PER_GRANT, minimum=0, maximum=128)
    safe_easy = _safe_limit(random_neg_per_query, default=DEFAULT_RANDOM_NEGATIVES_PER_GRANT, minimum=0, maximum=32)
    safe_total = _safe_limit(candidates_per_query, default=DEFAULT_CANDIDATES_PER_QUERY, minimum=2, maximum=256)

    # Enforce hard-negative-heavy composition:
    # 2 positives : 4-6 hard negatives : 1-2 easy negatives.
    # We generalize from top_k while preserving the hard-negative emphasis.
    min_hard = max(4, safe_top_k * 2)
    safe_hard = max(safe_hard, min_hard)
    safe_easy = min(max(1, safe_easy), 2)
    safe_total = max(safe_total, safe_top_k + safe_hard + safe_easy)

    return {
        "top_k": int(safe_top_k),
        "hard_neg_per_query": int(safe_hard),
        "easy_neg_per_query": int(safe_easy),
        "candidates_per_query": int(safe_total),
    }


def _generate_query_candidate_sets(
    *,
    grant_specs: Sequence[SpecRow],
    faculty_specs: Sequence[SpecRow],
    top_k: int,
    hard_neg_per_query: int,
    random_neg_per_query: int,
    candidates_per_query: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed))
    fac_mat = np.asarray([x.embedding for x in faculty_specs], dtype=np.float32)
    grant_mat = np.asarray([x.embedding for x in grant_specs], dtype=np.float32)
    fac_mat = _normalize_rows(fac_mat)
    grant_mat = _normalize_rows(grant_mat)

    out: List[Dict[str, Any]] = []
    hard_pool_width = max(int(top_k) + int(hard_neg_per_query) * 5, int(top_k) + 20)

    for g_idx, grant_spec in enumerate(grant_specs):
        sims = np.dot(fac_mat, grant_mat[g_idx])
        ordered = np.argsort(-sims).tolist()
        if not ordered:
            continue

        top_idx = [int(x) for x in ordered[: min(len(ordered), int(top_k))]]
        hard_pool = [int(x) for x in ordered[len(top_idx): min(len(ordered), hard_pool_width)]]
        hard_idx = hard_pool[: min(len(hard_pool), int(max(0, hard_neg_per_query)))]

        selected = list(dict.fromkeys(top_idx + hard_idx))
        selected_set = set(selected)
        remaining = [int(x) for x in ordered if int(x) not in selected_set]
        easy_idx: List[int] = []
        if remaining and random_neg_per_query > 0:
            pick_n = min(len(remaining), int(random_neg_per_query))
            easy_idx = rng.sample(remaining, pick_n)

        combined = list(dict.fromkeys(selected + easy_idx))

        # Ensure at least two candidates, backfill from remaining.
        if len(combined) < 2:
            for idx in remaining:
                if idx in combined:
                    continue
                combined.append(int(idx))
                if len(combined) >= 2:
                    break
        if len(combined) < 2:
            continue

        if candidates_per_query > 0 and len(combined) > candidates_per_query:
            combined = combined[: int(candidates_per_query)]
        if len(combined) < 2:
            continue

        top_set = set(top_idx)
        hard_set = set(hard_idx)
        easy_set = set(easy_idx)
        candidates: List[Dict[str, Any]] = []
        for local_i, f_idx in enumerate(combined, start=1):
            fac_spec = faculty_specs[int(f_idx)]
            if int(f_idx) in top_set:
                ctype = "topk"
            elif int(f_idx) in hard_set:
                ctype = "hard_negative"
            elif int(f_idx) in easy_set:
                ctype = "easy_negative"
            else:
                ctype = "backfill"
            candidates.append(
                {
                    "i": int(local_i),
                    "candidate_type": ctype,
                    "faculty_spec_id": fac_spec.spec_id,
                    "faculty_id": fac_spec.owner_id,
                    "faculty_email": fac_spec.owner_email,
                    "faculty_domains": list(fac_spec.domains),
                    "faculty_spec_weight": float(fac_spec.spec_weight),
                    "text": fac_spec.text,
                    "cosine_sim": _safe_unit_float(float(sims[int(f_idx)]), default=0.0),
                }
            )

        out.append(
            {
                "query_spec_id": grant_spec.spec_id,
                "grant_id": grant_spec.owner_id,
                "grant_domains": list(grant_spec.domains),
                "grant_spec_weight": float(grant_spec.spec_weight),
                "query": grant_spec.text,
                "candidates": candidates,
            }
        )
    return out


def _downsample_query_sets(
    query_sets: Sequence[Dict[str, Any]],
    *,
    max_queries: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if max_queries <= 0 or len(query_sets) <= max_queries:
        return list(query_sets)
    rng = random.Random(int(seed))
    out = list(query_sets)
    rng.shuffle(out)
    return out[: int(max_queries)]


def _build_rank_chain(model_id: str):
    from langchain_core.prompts import ChatPromptTemplate

    rank_batch_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RANK_BATCH_SYSTEM_PROMPT),
            ("human", "Tasks JSON:\n{tasks_json}"),
        ]
    )
    llm = get_llm_client(model_id).build()
    return rank_batch_prompt | llm.with_structured_output(RankedQueryOut)


def _sanitize_ranking(
    ranked: Sequence[Any],
    *,
    valid_indices: Sequence[int],
    default_rank: Sequence[int],
) -> Tuple[List[int], bool]:
    valid_set = set(int(x) for x in valid_indices)
    out: List[int] = []
    seen = set()
    had_any_valid = False
    for raw in list(ranked or []):
        try:
            idx = int(raw)
        except Exception:
            continue
        if idx not in valid_set or idx in seen:
            continue
        had_any_valid = True
        seen.add(idx)
        out.append(idx)

    for idx in default_rank:
        i = int(idx)
        if i in valid_set and i not in seen:
            seen.add(i)
            out.append(i)
    return out, had_any_valid


def _label_query_rankings_with_llm(
    *,
    query_sets: Sequence[Dict[str, Any]],
    llm_batch_size: int,
    model_id: str,
    max_retries: int,
    llm_max_workers: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    total_queries = len(list(query_sets or []))
    batch_size = max(1, int(llm_batch_size))
    total_batches = 0 if total_queries <= 0 else ((total_queries + batch_size - 1) // batch_size)
    max_attempts = max(1, int(max_retries))
    safe_workers = resolve_pool_size(
        max_workers=max(1, int(llm_max_workers)),
        task_count=max(1, int(total_batches)),
    )

    batches = 0
    retries_used = 0
    fallback_query_count = 0
    partial_fallback_query_count = 0
    llm_calls = 0

    query_list = list(query_sets or [])

    batch_payloads: List[Dict[str, Any]] = []
    for start in range(0, total_queries, batch_size):
        end = min(total_queries, start + batch_size)
        batch = query_list[start:end]
        batches += 1

        prompt_tasks: List[Dict[str, Any]] = []
        valid_index_map: Dict[int, List[int]] = {}
        default_rank_map: Dict[int, List[int]] = {}
        for q_local, query_row in enumerate(batch, start=1):
            candidates = list(query_row.get("candidates") or [])
            default_sorted = sorted(candidates, key=lambda x: float(x.get("cosine_sim") or 0.0), reverse=True)
            default_rank = [int(c.get("i")) for c in default_sorted if int(c.get("i") or 0) > 0]
            valid_indices = [int(c.get("i")) for c in candidates if int(c.get("i") or 0) > 0]
            valid_index_map[q_local] = valid_indices
            default_rank_map[q_local] = default_rank

            prompt_tasks.append(
                {
                    "q": int(q_local),
                    "query": _clean_text(query_row.get("query")),
                    "candidates": [
                        {
                            "i": int(c.get("i") or 0),
                            "t": _clean_text(c.get("text")),
                        }
                        for c in candidates
                    ],
                }
            )

        tasks_json = json.dumps(prompt_tasks, ensure_ascii=False)
        batch_payloads.append(
            {
                "batch_number": int(batches),
                "batch": list(batch),
                "tasks_json": tasks_json,
                "valid_index_map": valid_index_map,
                "default_rank_map": default_rank_map,
            }
        )

    call_counter = {"value": 0}
    counter_lock = threading.Lock()
    get_chain = build_thread_local_getter(lambda: _build_rank_chain(model_id))

    def _run_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
        ranked_map: Dict[int, List[int]] = {}
        batch_ok = False
        retries_local = 0
        batch_rows: List[Dict[str, Any]] = []
        fallback_local = 0
        partial_local = 0

        batch_number = int(payload.get("batch_number") or 0)
        batch_rows_in = list(payload.get("batch") or [])
        valid_index_map = dict(payload.get("valid_index_map") or {})
        default_rank_map = dict(payload.get("default_rank_map") or {})
        tasks_json_local = _clean_text(payload.get("tasks_json"))

        for attempt in range(0, max_attempts):
            if attempt > 0:
                retries_local += 1
            try:
                with counter_lock:
                    call_counter["value"] += 1
                    call_id = int(call_counter["value"])
                print(
                    f"[llm_call] count={call_id} batch={batch_number}/{total_batches} "
                    f"attempt={attempt + 1}/{max_attempts} batch_queries={len(batch_rows_in)}"
                )
                result = get_chain().invoke({"tasks_json": tasks_json_local})
                for item in list(getattr(result, "items", []) or []):
                    row = item.model_dump() if hasattr(item, "model_dump") else dict(item or {})
                    q_local = int(row.get("q") or 0)
                    if q_local <= 0:
                        continue
                    ranked_map[q_local] = [int(x) for x in list(row.get("ranked") or [])]
                batch_ok = True
                break
            except Exception as e:
                print(
                    f"[llm_call_error] batch={batch_number}/{total_batches} "
                    f"attempt={attempt + 1}/{max_attempts} error={type(e).__name__}: {e}"
                )
                continue

        for q_local, query_row in enumerate(batch_rows_in, start=1):
            ranked_raw = ranked_map.get(q_local, [])
            ranking, had_any_valid = _sanitize_ranking(
                ranked_raw,
                valid_indices=valid_index_map.get(q_local, []),
                default_rank=default_rank_map.get(q_local, []),
            )
            out_row = dict(query_row)
            out_row["ranking"] = ranking
            if not batch_ok:
                out_row["label_source"] = "fallback_cosine_batch"
                fallback_local += 1
            elif not had_any_valid:
                out_row["label_source"] = "fallback_cosine_query"
                fallback_local += 1
            elif len(ranking) != len(valid_index_map.get(q_local, [])):
                out_row["label_source"] = "llm_partial_ranking"
                partial_local += 1
            else:
                out_row["label_source"] = "llm_ranking"
            batch_rows.append(out_row)

        return {
            "batch_number": batch_number,
            "rows": batch_rows,
            "retries_used": int(retries_local),
            "fallback_query_count": int(fallback_local),
            "partial_fallback_query_count": int(partial_local),
        }

    run_results = parallel_map(
        batch_payloads,
        max_workers=safe_workers,
        run_item=_run_batch,
    )

    run_results = sorted(
        list(run_results or []),
        key=lambda x: int((x or {}).get("batch_number") or 0),
    )

    labeled_queries: List[Dict[str, Any]] = []
    for item in run_results:
        row = dict(item or {})
        labeled_queries.extend(list(row.get("rows") or []))
        retries_used += int(row.get("retries_used") or 0)
        fallback_query_count += int(row.get("fallback_query_count") or 0)
        partial_fallback_query_count += int(row.get("partial_fallback_query_count") or 0)
    llm_calls = int(call_counter["value"])

    meta = {
        "queries_total": int(total_queries),
        "batches": int(batches),
        "llm_batch_size": int(batch_size),
        "llm_max_workers": int(safe_workers),
        "llm_model": str(model_id or ""),
        "llm_calls": int(llm_calls),
        "retries_used": int(retries_used),
        "fallback_query_count": int(fallback_query_count),
        "fallback_query_ratio": (0.0 if total_queries <= 0 else float(fallback_query_count) / float(total_queries)),
        "partial_fallback_query_count": int(partial_fallback_query_count),
    }
    return labeled_queries, meta


def _label_query_rankings_with_cosine(
    *,
    query_sets: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    labeled: List[Dict[str, Any]] = []
    for query_row in list(query_sets or []):
        row = dict(query_row or {})
        candidates = list(row.get("candidates") or [])
        ranked = sorted(
            candidates,
            key=lambda c: float(c.get("cosine_sim") or 0.0),
            reverse=True,
        )
        ranking = [int(c.get("i") or 0) for c in ranked if int(c.get("i") or 0) > 0]
        if len(ranking) < 2:
            continue
        row["ranking"] = ranking
        row["label_source"] = "cosine_ranking"
        labeled.append(row)

    meta = {
        "queries_total": int(len(list(query_sets or []))),
        "queries_labeled": int(len(labeled)),
        "labeler": "cosine_only",
        "llm_batch_size": 0,
        "llm_max_workers": 0,
        "llm_model": "",
        "llm_calls": 0,
        "retries_used": 0,
        "fallback_query_count": 0,
        "fallback_query_ratio": 0.0,
        "partial_fallback_query_count": 0,
    }
    return labeled, meta


def _expand_rankings_to_pairwise_rows(
    *,
    ranked_queries: Sequence[Dict[str, Any]],
    max_pairs: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pair_rows: List[Dict[str, Any]] = []
    pairs_generated = 0

    for query_row in list(ranked_queries or []):
        query_text = _clean_text(query_row.get("query"))
        query_spec_id = _clean_text(query_row.get("query_spec_id"))
        grant_id = _clean_text(query_row.get("grant_id"))
        grant_domains = list(query_row.get("grant_domains") or [])
        grant_spec_weight = _safe_unit_float(query_row.get("grant_spec_weight"), default=0.0)
        label_source = _clean_text(query_row.get("label_source"))

        candidates = list(query_row.get("candidates") or [])
        by_idx: Dict[int, Dict[str, Any]] = {}
        for c in candidates:
            idx = int(c.get("i") or 0)
            if idx > 0:
                by_idx[idx] = dict(c)
        ranking = [int(x) for x in list(query_row.get("ranking") or []) if int(x) in by_idx]
        n = len(ranking)
        if n < 2:
            continue

        denom = max(1, n - 1)
        for pos in range(0, n - 1):
            pos_idx = int(ranking[pos])
            pos_c = dict(by_idx.get(pos_idx) or {})
            pos_teacher_score = 1.0 - (float(pos) / float(denom))
            for neg in range(pos + 1, n):
                neg_idx = int(ranking[neg])
                neg_c = dict(by_idx.get(neg_idx) or {})
                neg_teacher_score = 1.0 - (float(neg) / float(denom))
                preference_strength = (float(neg - pos) / float(denom))
                margin = max(0.0, pos_teacher_score - neg_teacher_score)
                pair_rows.append(
                    {
                        "distill_method": "ranking_distillation",
                        "label": 1,
                        "query": query_text,
                        "positive": _clean_text(pos_c.get("text")),
                        "negative": _clean_text(neg_c.get("text")),
                        "query_spec_id": query_spec_id,
                        "grant_id": grant_id,
                        "grant_domains": grant_domains,
                        "grant_spec_weight": float(grant_spec_weight),
                        "positive_spec_id": _clean_text(pos_c.get("faculty_spec_id")),
                        "negative_spec_id": _clean_text(neg_c.get("faculty_spec_id")),
                        "positive_faculty_id": _clean_text(pos_c.get("faculty_id")),
                        "negative_faculty_id": _clean_text(neg_c.get("faculty_id")),
                        "positive_faculty_email": _clean_text(pos_c.get("faculty_email")).lower(),
                        "negative_faculty_email": _clean_text(neg_c.get("faculty_email")).lower(),
                        "positive_domains": list(pos_c.get("faculty_domains") or []),
                        "negative_domains": list(neg_c.get("faculty_domains") or []),
                        "positive_spec_weight": _safe_unit_float(pos_c.get("faculty_spec_weight"), default=0.0),
                        "negative_spec_weight": _safe_unit_float(neg_c.get("faculty_spec_weight"), default=0.0),
                        "positive_rank": int(pos + 1),
                        "negative_rank": int(neg + 1),
                        "candidate_count": int(n),
                        "positive_teacher_score": float(_safe_unit_float(pos_teacher_score, default=0.0)),
                        "negative_teacher_score": float(_safe_unit_float(neg_teacher_score, default=0.0)),
                        "teacher_margin": float(_safe_unit_float(margin, default=0.0)),
                        "preference_strength": float(_safe_unit_float(preference_strength, default=0.0)),
                        "positive_cosine_sim": float(_safe_unit_float(pos_c.get("cosine_sim"), default=0.0)),
                        "negative_cosine_sim": float(_safe_unit_float(neg_c.get("cosine_sim"), default=0.0)),
                        "positive_candidate_type": _clean_text(pos_c.get("candidate_type")),
                        "negative_candidate_type": _clean_text(neg_c.get("candidate_type")),
                        "label_source": label_source,
                    }
                )
                pairs_generated += 1

    if max_pairs > 0 and len(pair_rows) > max_pairs:
        rng = random.Random(int(seed))
        rng.shuffle(pair_rows)
        pair_rows = pair_rows[: int(max_pairs)]

    return pair_rows, {
        "pairs_generated_before_cap": int(pairs_generated),
        "pairs_saved_after_cap": int(len(pair_rows)),
    }


def _load_mined_false_positive_rows(
    *,
    probes_json_path: str,
    seed: int,
    max_rows: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = Path(_clean_text(probes_json_path)).expanduser().resolve() if _clean_text(probes_json_path) else None
    if path is None:
        return [], {
            "enabled": False,
            "probes_json_path": "",
            "probe_cases_seen": 0,
            "probe_cases_used": 0,
            "probe_rows_added_before_cap": 0,
            "probe_rows_added_after_cap": 0,
        }
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Probe-scored JSON file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse probes JSON: {path}. error={e}") from e

    cases = []
    if isinstance(payload, dict):
        raw_cases = payload.get("cases")
        if isinstance(raw_cases, list):
            cases = list(raw_cases)
        elif isinstance(payload.get("failure_examples_preview"), list):
            # Allow summary JSON as a fallback input.
            cases = list(payload.get("failure_examples_preview") or [])

    rng = random.Random(int(seed))
    pair_rows: List[Dict[str, Any]] = []
    cases_seen = len(cases)
    cases_used = 0

    for idx, case_raw in enumerate(cases):
        case = dict(case_raw or {})
        query = _clean_text(case.get("query"))
        if not query:
            continue

        positives = []
        for p in list(case.get("positives") or case.get("expected_positives") or []):
            t = _clean_text(p)
            if t:
                positives.append(t)
        if not positives:
            continue

        ranked = list(case.get("ranked") or case.get("top_ranked_preview") or [])
        if not ranked:
            continue

        negatives: List[Dict[str, Any]] = []
        for cand_raw in ranked:
            cand = dict(cand_raw or {})
            cand_text = _clean_text(cand.get("candidate"))
            if not cand_text:
                continue
            is_pos = int(cand.get("is_positive") or 0)
            if is_pos == 1:
                continue
            negatives.append(
                {
                    "candidate": cand_text,
                    "kind": _clean_text(cand.get("kind")) or "hard_negative",
                }
            )
        if not negatives:
            continue

        hard_first = sorted(
            negatives,
            key=lambda x: (0 if "hard" in _normalize_text_key(x.get("kind")) else 1),
        )
        neg_picks = hard_first[: max(1, int(PROBE_NEGATIVES_PER_CASE))]
        pos_picks = positives[: max(1, int(PROBE_POSITIVES_PER_CASE))]
        if not neg_picks or not pos_picks:
            continue

        query_id = f"probe:{_stable_text_id(query)}"
        for pos_text in pos_picks:
            for neg in neg_picks:
                neg_text = _clean_text(neg.get("candidate"))
                if not neg_text:
                    continue
                if _normalize_text_key(pos_text) == _normalize_text_key(neg_text):
                    continue
                kind = _clean_text(neg.get("kind")) or "hard_negative"
                if "hard" not in _normalize_text_key(kind):
                    kind = "hard_negative"
                pair_rows.append(
                    {
                        "distill_method": "probe_mined_false_positive",
                        "label": 1,
                        "query": query,
                        "positive": pos_text,
                        "negative": neg_text,
                        "query_spec_id": query_id,
                        "grant_id": f"probe_case_{idx}",
                        "grant_domains": [],
                        "grant_spec_weight": 1.0,
                        "positive_spec_id": f"probe_pos:{_stable_text_id(pos_text)}",
                        "negative_spec_id": f"probe_neg:{_stable_text_id(neg_text)}",
                        "positive_faculty_id": "",
                        "negative_faculty_id": "",
                        "positive_faculty_email": "",
                        "negative_faculty_email": "",
                        "positive_domains": [],
                        "negative_domains": [],
                        "positive_spec_weight": 1.0,
                        "negative_spec_weight": 1.0,
                        "positive_rank": 1,
                        "negative_rank": 2,
                        "candidate_count": int(len(ranked)),
                        "positive_teacher_score": float(MINED_FALSE_POSITIVE_POS_FLOOR),
                        "negative_teacher_score": float(MINED_FALSE_POSITIVE_NEG_CAP),
                        "teacher_margin": float(MINED_FALSE_POSITIVE_POS_FLOOR - MINED_FALSE_POSITIVE_NEG_CAP),
                        "preference_strength": 1.0,
                        "positive_cosine_sim": 0.0,
                        "negative_cosine_sim": 0.0,
                        "positive_candidate_type": "topk",
                        "negative_candidate_type": str(kind),
                        "label_source": "mined_false_positive",
                    }
                )
                cases_used += 1

    before_cap = len(pair_rows)
    safe_max = max(0, int(max_rows))
    if safe_max > 0 and len(pair_rows) > safe_max:
        rng.shuffle(pair_rows)
        pair_rows = pair_rows[:safe_max]

    return pair_rows, {
        "enabled": True,
        "probes_json_path": str(path),
        "probe_cases_seen": int(cases_seen),
        "probe_cases_used": int(cases_used),
        "probe_rows_added_before_cap": int(before_cap),
        "probe_rows_added_after_cap": int(len(pair_rows)),
    }


def _sharpen_teacher_labels(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    adjusted_total = 0
    adjusted_hard = 0
    adjusted_mined = 0

    for raw in list(rows or []):
        row = dict(raw or {})
        old_pos = _safe_unit_float(row.get("positive_teacher_score"), default=1.0)
        old_neg = _safe_unit_float(row.get("negative_teacher_score"), default=0.0)
        source = _normalize_text_key(row.get("label_source"))
        neg_type = _normalize_text_key(row.get("negative_candidate_type"))

        is_mined = "mined_false_positive" in source
        is_hard = ("hard_negative" in neg_type) or is_mined

        pos = float(old_pos)
        neg = float(old_neg)

        if is_mined:
            pos = max(pos, float(MINED_FALSE_POSITIVE_POS_FLOOR))
            neg = min(neg, float(MINED_FALSE_POSITIVE_NEG_CAP))
            min_margin = float(MINED_FALSE_POSITIVE_MIN_MARGIN)
        elif is_hard:
            pos = max(pos, float(HARD_NEGATIVE_POS_FLOOR))
            neg = min(neg, float(HARD_NEGATIVE_NEG_CAP))
            min_margin = float(HARD_NEGATIVE_MIN_MARGIN)
        else:
            min_margin = float(GENERAL_MIN_MARGIN)

        if (pos - neg) < min_margin:
            pos = min(1.0, max(pos, neg + min_margin))
            if (pos - neg) < min_margin:
                neg = max(0.0, pos - min_margin)
        if pos <= neg:
            pos = min(1.0, neg + 1e-3)

        pos = float(_safe_unit_float(pos, default=1.0))
        neg = float(_safe_unit_float(neg, default=0.0))
        margin = float(max(0.0, pos - neg))

        pref = _safe_unit_float(row.get("preference_strength"), default=0.0)
        if is_mined:
            pref = max(pref, 0.85)
        elif is_hard:
            pref = max(pref, 0.65)
        pref = max(pref, margin)

        if abs(pos - old_pos) > 1e-12 or abs(neg - old_neg) > 1e-12:
            adjusted_total += 1
            if is_hard:
                adjusted_hard += 1
            if is_mined:
                adjusted_mined += 1

        row["positive_teacher_score"] = float(pos)
        row["negative_teacher_score"] = float(neg)
        row["teacher_margin"] = float(_safe_unit_float(margin, default=0.0))
        row["preference_strength"] = float(_safe_unit_float(pref, default=0.0))
        out.append(row)

    return out, {
        "teacher_labels_adjusted_total": int(adjusted_total),
        "teacher_labels_adjusted_hard_negative": int(adjusted_hard),
        "teacher_labels_adjusted_mined_false_positive": int(adjusted_mined),
    }


def _clean_pair_rows(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # 1) Drop invalid rows and exact duplicates.
    invalid_dropped = 0
    exact_dupe_dropped = 0
    best_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    best_score: Dict[Tuple[str, str, str], Tuple[float, float, float]] = {}

    for raw in list(rows or []):
        row = dict(raw or {})
        query = _clean_text(row.get("query"))
        pos = _clean_text(row.get("positive"))
        neg = _clean_text(row.get("negative"))
        if not query or not pos or not neg:
            invalid_dropped += 1
            continue
        if _normalize_text_key(pos) == _normalize_text_key(neg):
            invalid_dropped += 1
            continue

        key = (_normalize_text_key(query), _normalize_text_key(pos), _normalize_text_key(neg))
        score_key = (
            float(_safe_unit_float(row.get("teacher_margin"), default=0.0)),
            float(_safe_unit_float(row.get("preference_strength"), default=0.0)),
            float(_safe_unit_float(row.get("positive_teacher_score"), default=0.0)),
        )
        old_score = best_score.get(key)
        if old_score is None or score_key > old_score:
            if old_score is not None:
                exact_dupe_dropped += 1
            best_by_key[key] = row
            best_score[key] = score_key
        else:
            exact_dupe_dropped += 1

    deduped = list(best_by_key.values())

    # 2) Resolve contradictory roles for same (query, doc).
    role_stats: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for row in deduped:
        qk = _normalize_text_key(row.get("query"))
        pk = _normalize_text_key(row.get("positive"))
        nk = _normalize_text_key(row.get("negative"))
        pos_score = float(_safe_unit_float(row.get("positive_teacher_score"), default=0.0))
        neg_score = float(_safe_unit_float(row.get("negative_teacher_score"), default=0.0))
        role_stats.setdefault((qk, pk), {"pos": [], "neg": []})["pos"].append(pos_score)
        role_stats.setdefault((qk, nk), {"pos": [], "neg": []})["neg"].append(neg_score)

    preferred_role: Dict[Tuple[str, str], str] = {}
    contradictions = 0
    for key, stats in role_stats.items():
        pos_vals = list(stats.get("pos") or [])
        neg_vals = list(stats.get("neg") or [])
        if not pos_vals or not neg_vals:
            continue
        contradictions += 1
        mean_pos = float(np.mean(np.asarray(pos_vals, dtype=np.float64)))
        mean_neg = float(np.mean(np.asarray(neg_vals, dtype=np.float64)))
        preferred_role[key] = "positive" if mean_pos >= mean_neg else "negative"

    contradiction_dropped = 0
    cleaned: List[Dict[str, Any]] = []
    for row in deduped:
        qk = _normalize_text_key(row.get("query"))
        pk = (qk, _normalize_text_key(row.get("positive")))
        nk = (qk, _normalize_text_key(row.get("negative")))

        drop = False
        if preferred_role.get(pk) == "negative":
            drop = True
        if preferred_role.get(nk) == "positive":
            drop = True
        if drop:
            contradiction_dropped += 1
            continue
        cleaned.append(row)

    return cleaned, {
        "invalid_rows_dropped": int(invalid_dropped),
        "exact_duplicate_rows_dropped": int(exact_dupe_dropped),
        "contradiction_keys": int(contradictions),
        "contradiction_rows_dropped": int(contradiction_dropped),
        "pair_rows_after_cleaning": int(len(cleaned)),
    }


def _cap_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    max_rows: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out = list(rows or [])
    before_cap = len(out)
    safe_max = max(0, int(max_rows))
    if safe_max > 0 and len(out) > safe_max:
        rng = random.Random(int(seed))
        rng.shuffle(out)
        out = out[:safe_max]
    return out, {
        "pair_rows_before_final_cap": int(before_cap),
        "pair_rows_after_final_cap": int(len(out)),
    }


def _save_dataset(
    *,
    rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    output_prefix: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = f"{output_prefix}_{ts}"
    jsonl_path = output_dir / f"{stem}.jsonl"
    meta_path = output_dir / f"{stem}.meta.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")

    meta_payload = {
        **dict(meta or {}),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "jsonl_path": str(jsonl_path),
        "row_count": int(len(list(rows or []))),
    }
    meta_path.write_text(json.dumps(json_ready(meta_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "jsonl_path": str(jsonl_path),
        "meta_path": str(meta_path),
    }


def build_dataset(
    *,
    top_k_candidates: int,
    hard_negatives_per_grant: int,
    random_negatives_per_grant: int,
    candidates_per_query: int,
    max_queries: int,
    max_pairs: int,
    llm_batch_size: int,
    llm_model: str,
    llm_max_retries: int,
    llm_max_workers: int,
    faculty_min_spec_weight: float,
    grant_min_spec_weight: float,
    faculty_limit: int,
    grant_limit: int,
    embed_batch_size: int,
    embed_max_workers: int,
    use_stored_spec_embeddings: bool,
    spec_embedding_model: str,
    use_llm_ranker: bool,
    mined_probes_json: str,
    mined_max_rows: int,
    seed: int,
    output_dir: Path,
    output_prefix: str,
) -> Dict[str, Any]:
    mix = _resolve_candidate_mix(
        top_k=top_k_candidates,
        hard_neg_per_query=hard_negatives_per_grant,
        random_neg_per_query=random_negatives_per_grant,
        candidates_per_query=candidates_per_query,
    )
    safe_top_k = int(mix["top_k"])
    safe_hard = int(mix["hard_neg_per_query"])
    safe_rand = int(mix["easy_neg_per_query"])
    safe_candidates_per_query = int(mix["candidates_per_query"])
    safe_max_queries = _safe_limit(max_queries, default=5000, minimum=1, maximum=2_000_000)
    safe_max_pairs = _safe_limit(max_pairs, default=DEFAULT_MAX_PAIRS, minimum=0, maximum=5_000_000)
    safe_batch = _safe_limit(llm_batch_size, default=8, minimum=1, maximum=64)
    safe_retries = _safe_limit(llm_max_retries, default=2, minimum=1, maximum=8)
    safe_workers = _safe_limit(llm_max_workers, default=4, minimum=1, maximum=64)
    safe_fac_limit = _safe_limit(faculty_limit, default=200000, minimum=1, maximum=2_000_000)
    safe_grant_limit = _safe_limit(grant_limit, default=200000, minimum=1, maximum=2_000_000)
    safe_embed_batch = _safe_limit(embed_batch_size, default=64, minimum=1, maximum=512)
    safe_embed_workers = _safe_limit(embed_max_workers, default=4, minimum=1, maximum=64)
    safe_use_stored_embeddings = bool(use_stored_spec_embeddings)
    safe_embedding_model = _clean_text(spec_embedding_model) or _clean_text(settings.bedrock_embed_model_id)
    safe_use_llm_ranker = bool(use_llm_ranker)
    safe_mined_max_rows = _safe_limit(mined_max_rows, default=DEFAULT_MINED_MAX_ROWS, minimum=0, maximum=2_000_000)
    safe_seed = int(seed)

    if safe_use_stored_embeddings:
        faculty_specs_raw = _fetch_faculty_specs_from_embedding_table(
            min_spec_weight=faculty_min_spec_weight,
            limit=safe_fac_limit,
            embedding_model=safe_embedding_model,
        )
        grant_specs_raw = _fetch_grant_specs_from_embedding_table(
            min_spec_weight=grant_min_spec_weight,
            limit=safe_grant_limit,
            embedding_model=safe_embedding_model,
        )
    else:
        faculty_specs_raw = _fetch_faculty_specs(
            min_spec_weight=faculty_min_spec_weight,
            limit=safe_fac_limit,
        )
        grant_specs_raw = _fetch_grant_specs(
            min_spec_weight=grant_min_spec_weight,
            limit=safe_grant_limit,
        )

        faculty_specs_raw = _embed_spec_rows(
            faculty_specs_raw,
            batch_size=safe_embed_batch,
            max_workers=safe_embed_workers,
        )
        grant_specs_raw = _embed_spec_rows(
            grant_specs_raw,
            batch_size=safe_embed_batch,
            max_workers=safe_embed_workers,
        )

    common_dim, faculty_specs, grant_specs = _select_common_embedding_dim(faculty_specs_raw, grant_specs_raw)
    if common_dim <= 0 or not faculty_specs or not grant_specs:
        fac_dim_counter = Counter(len(x.embedding) for x in faculty_specs_raw if x.embedding)
        grant_dim_counter = Counter(len(x.embedding) for x in grant_specs_raw if x.embedding)
        debug_payload = {
            "faculty_specs_raw_count": int(len(faculty_specs_raw)),
            "grant_specs_raw_count": int(len(grant_specs_raw)),
            "faculty_embedded_count": int(sum(fac_dim_counter.values())),
            "grant_embedded_count": int(sum(grant_dim_counter.values())),
            "faculty_embedding_dims": dict(fac_dim_counter),
            "grant_embedding_dims": dict(grant_dim_counter),
            "common_dims": sorted([int(d) for d in fac_dim_counter if d in grant_dim_counter]),
            "embed_model_id": _clean_text(settings.bedrock_embed_model_id),
            "aws_region": _clean_text(settings.aws_region),
            "use_stored_spec_embeddings": bool(safe_use_stored_embeddings),
            "spec_embedding_model": safe_embedding_model,
        }
        raise RuntimeError(
            "No shared embedding dimension found across faculty/grant specialization keywords. "
            f"Debug={json.dumps(json_ready(debug_payload), ensure_ascii=False)}"
        )

    query_sets = _generate_query_candidate_sets(
        grant_specs=grant_specs,
        faculty_specs=faculty_specs,
        top_k=safe_top_k,
        hard_neg_per_query=safe_hard,
        random_neg_per_query=safe_rand,
        candidates_per_query=safe_candidates_per_query,
        seed=safe_seed,
    )
    query_sets = _downsample_query_sets(query_sets, max_queries=safe_max_queries, seed=safe_seed)

    if safe_use_llm_ranker:
        ranked_queries, llm_meta = _label_query_rankings_with_llm(
            query_sets=query_sets,
            llm_batch_size=safe_batch,
            model_id=(llm_model or settings.haiku or "").strip(),
            max_retries=safe_retries,
            llm_max_workers=safe_workers,
        )
    else:
        ranked_queries, llm_meta = _label_query_rankings_with_cosine(
            query_sets=query_sets,
        )

    pair_rows_base, pair_meta = _expand_rankings_to_pairwise_rows(
        ranked_queries=ranked_queries,
        max_pairs=0,
        seed=safe_seed,
    )

    probe_rows, probe_meta = _load_mined_false_positive_rows(
        probes_json_path=_clean_text(mined_probes_json),
        seed=safe_seed,
        max_rows=safe_mined_max_rows,
    )

    pair_rows_all = list(pair_rows_base) + list(probe_rows)
    pair_rows_sharp, sharpen_meta = _sharpen_teacher_labels(pair_rows_all)
    pair_rows_clean, clean_meta = _clean_pair_rows(pair_rows_sharp)
    pair_rows, cap_meta = _cap_rows(
        pair_rows_clean,
        max_rows=safe_max_pairs,
        seed=safe_seed,
    )
    pair_meta = {
        **dict(pair_meta or {}),
        "pairs_generated_before_cap": int(len(pair_rows_all)),
        "pairs_saved_after_cap": int(len(pair_rows)),
        "pairs_generated_from_rankings": int(len(pair_rows_base)),
    }

    final_label_source_counts = Counter(_clean_text(x.get("label_source")) for x in pair_rows)

    candidate_type_counts = Counter()
    label_source_counts = Counter()
    for q in ranked_queries:
        label_source_counts[_clean_text(q.get("label_source"))] += 1
        for c in list(q.get("candidates") or []):
            candidate_type_counts[_clean_text(c.get("candidate_type"))] += 1

    paths = _save_dataset(
        rows=pair_rows,
        output_dir=output_dir,
        output_prefix=output_prefix,
        meta={
            "params": {
                "top_k_candidates": safe_top_k,
                "hard_negatives_per_grant": safe_hard,
                "random_negatives_per_grant": safe_rand,
                "candidates_per_query": safe_candidates_per_query,
                "max_queries": safe_max_queries,
                "max_pairs": safe_max_pairs,
                "mined_probes_json": _clean_text(mined_probes_json),
                "mined_max_rows": safe_mined_max_rows,
                "llm_batch_size": safe_batch,
                "llm_model": (llm_model or settings.haiku or "").strip(),
                "llm_max_retries": safe_retries,
                "llm_max_workers": safe_workers,
                "faculty_min_spec_weight": float(max(0.0, faculty_min_spec_weight)),
                "grant_min_spec_weight": float(max(0.0, grant_min_spec_weight)),
                "faculty_limit": safe_fac_limit,
                "grant_limit": safe_grant_limit,
                "embed_batch_size": safe_embed_batch,
                "embed_max_workers": safe_embed_workers,
                "use_stored_spec_embeddings": bool(safe_use_stored_embeddings),
                "spec_embedding_model": safe_embedding_model,
                "use_llm_ranker": bool(safe_use_llm_ranker),
                "seed": safe_seed,
            },
            "counts": {
                "faculty_specs_raw": len(faculty_specs_raw),
                "grant_specs_raw": len(grant_specs_raw),
                "embedding_common_dim": common_dim,
                "faculty_specs_used": len(faculty_specs),
                "grant_specs_used": len(grant_specs),
                "queries_generated": len(query_sets),
                "queries_ranked": len(ranked_queries),
                "candidate_type_counts": dict(candidate_type_counts),
                "query_label_source_counts": dict(label_source_counts),
                "final_pair_label_source_counts": dict(final_label_source_counts),
                **pair_meta,
                **probe_meta,
                **sharpen_meta,
                **clean_meta,
                **cap_meta,
            },
            "llm": llm_meta,
        },
    )

    return {
        "params": {
            "top_k_candidates": safe_top_k,
            "hard_negatives_per_grant": safe_hard,
            "random_negatives_per_grant": safe_rand,
            "candidates_per_query": safe_candidates_per_query,
            "max_queries": safe_max_queries,
            "max_pairs": safe_max_pairs,
            "mined_probes_json": _clean_text(mined_probes_json),
            "mined_max_rows": safe_mined_max_rows,
            "llm_batch_size": safe_batch,
            "llm_model": (llm_model or settings.haiku or "").strip(),
            "llm_max_workers": safe_workers,
            "faculty_min_spec_weight": float(max(0.0, faculty_min_spec_weight)),
            "grant_min_spec_weight": float(max(0.0, grant_min_spec_weight)),
            "faculty_limit": safe_fac_limit,
            "grant_limit": safe_grant_limit,
            "embed_batch_size": safe_embed_batch,
            "embed_max_workers": safe_embed_workers,
            "use_stored_spec_embeddings": bool(safe_use_stored_embeddings),
            "spec_embedding_model": safe_embedding_model,
            "use_llm_ranker": bool(safe_use_llm_ranker),
            "seed": safe_seed,
        },
        "counts": {
            "faculty_specs_raw": len(faculty_specs_raw),
            "grant_specs_raw": len(grant_specs_raw),
            "embedding_common_dim": common_dim,
            "faculty_specs_used": len(faculty_specs),
            "grant_specs_used": len(grant_specs),
            "queries_generated": len(query_sets),
            "queries_ranked": len(ranked_queries),
            "candidate_type_counts": dict(candidate_type_counts),
            "query_label_source_counts": dict(label_source_counts),
            "final_pair_label_source_counts": dict(final_label_source_counts),
            **pair_meta,
            **probe_meta,
            **sharpen_meta,
            **clean_meta,
            **cap_meta,
        },
        "llm": llm_meta,
        "output": paths,
    }


def _build_parser() -> argparse.ArgumentParser:
    default_output_dir = Path(__file__).resolve().parent / "dataset"
    parser = argparse.ArgumentParser(
        description=(
            "Build ranking-distilled specialization preference dataset for cross-encoder training "
            "using Grant specs as queries and Faculty specs as candidate docs."
        )
    )
    parser.add_argument("--top-k-candidates", type=int, default=DEFAULT_TOP_K_CANDIDATES, help="Top cosine candidates per query (positive-heavy pool).")
    parser.add_argument("--hard-negatives-per-grant", type=int, default=DEFAULT_HARD_NEGATIVES_PER_GRANT, help="Hard negatives per query.")
    parser.add_argument("--random-negatives-per-grant", type=int, default=DEFAULT_RANDOM_NEGATIVES_PER_GRANT, help="Easy/random negatives per query (kept small).")
    parser.add_argument("--candidates-per-query", type=int, default=DEFAULT_CANDIDATES_PER_QUERY, help="Total candidates fed to ranker per query.")
    parser.add_argument("--max-queries", type=int, default=5000, help="Max queries (grant specs) to label.")
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS, help="Max pairwise preference rows to save (0 = all).")
    parser.add_argument("--llm-batch-size", type=int, default=8, help="Queries per LLM ranking call.")
    parser.add_argument("--llm-model", type=str, default=(settings.haiku or "").strip(), help="Bedrock model id for ranking labels.")
    parser.add_argument("--llm-max-retries", type=int, default=2, help="Retries per LLM batch.")
    parser.add_argument("--llm-max-workers", type=int, default=4, help="Parallel LLM batch workers.")
    parser.add_argument("--faculty-min-spec-weight", type=float, default=0.0, help="Minimum faculty specialization edge weight.")
    parser.add_argument("--grant-min-spec-weight", type=float, default=0.0, help="Minimum grant specialization edge weight.")
    parser.add_argument("--faculty-limit", type=int, default=200000, help="Max faculty keyword rows to fetch from Postgres.")
    parser.add_argument("--grant-limit", type=int, default=200000, help="Max grant/opportunity keyword rows to fetch from Postgres.")
    parser.add_argument("--embed-batch-size", type=int, default=64, help="Specialization text embedding batch size.")
    parser.add_argument("--embed-max-workers", type=int, default=4, help="Parallel workers for embedding batches.")
    parser.add_argument(
        "--use-stored-spec-embeddings",
        action="store_true",
        help="Use precomputed rows from faculty_specialization_embedding/opportunity_specialization_embedding tables.",
    )
    parser.add_argument(
        "--spec-embedding-model",
        type=str,
        default=(settings.bedrock_embed_model_id or "").strip(),
        help="Embedding model id filter when --use-stored-spec-embeddings is enabled.",
    )
    parser.add_argument(
        "--no-llm-rank",
        action="store_true",
        help="Disable LLM ranking labels and use cosine-only ranking labels.",
    )
    parser.add_argument(
        "--mined-probes-json",
        type=str,
        default="",
        help="Optional path to probes_scored.json to mine false positives back into training rows.",
    )
    parser.add_argument(
        "--mined-max-rows",
        type=int,
        default=DEFAULT_MINED_MAX_ROWS,
        help="Cap mined probe rows added to dataset (0 = all).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir), help="Output directory for dataset files.")
    parser.add_argument("--output-prefix", type=str, default="spec_pair_rankdistill_train", help="Output file prefix.")
    parser.add_argument("--json-only", action="store_true", help="Print only JSON payload.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    payload = build_dataset(
        top_k_candidates=int(args.top_k_candidates),
        hard_negatives_per_grant=int(args.hard_negatives_per_grant),
        random_negatives_per_grant=int(args.random_negatives_per_grant),
        candidates_per_query=int(args.candidates_per_query),
        max_queries=int(args.max_queries),
        max_pairs=int(args.max_pairs),
        llm_batch_size=int(args.llm_batch_size),
        llm_model=_clean_text(args.llm_model),
        llm_max_retries=int(args.llm_max_retries),
        llm_max_workers=int(args.llm_max_workers),
        faculty_min_spec_weight=float(args.faculty_min_spec_weight),
        grant_min_spec_weight=float(args.grant_min_spec_weight),
        faculty_limit=int(args.faculty_limit),
        grant_limit=int(args.grant_limit),
        embed_batch_size=int(args.embed_batch_size),
        embed_max_workers=int(args.embed_max_workers),
        use_stored_spec_embeddings=bool(args.use_stored_spec_embeddings),
        spec_embedding_model=_clean_text(args.spec_embedding_model),
        use_llm_ranker=not bool(args.no_llm_rank),
        mined_probes_json=_clean_text(args.mined_probes_json),
        mined_max_rows=int(args.mined_max_rows),
        seed=int(args.seed),
        output_dir=Path(_clean_text(args.output_dir)),
        output_prefix=_clean_text(args.output_prefix) or "spec_pair_rankdistill_train",
    )

    if not args.json_only:
        print("Cross-encoder dataset build (ranking distillation) complete.")
        print(f"  faculty specs used      : {payload.get('counts', {}).get('faculty_specs_used', 0)}")
        print(f"  grant specs used        : {payload.get('counts', {}).get('grant_specs_used', 0)}")
        print(f"  queries ranked          : {payload.get('counts', {}).get('queries_ranked', 0)}")
        print(f"  pairs generated (raw)   : {payload.get('counts', {}).get('pairs_generated_before_cap', 0)}")
        print(f"  pairs saved             : {payload.get('counts', {}).get('pairs_saved_after_cap', 0)}")
        print(f"  output jsonl            : {payload.get('output', {}).get('jsonl_path', '')}")
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
