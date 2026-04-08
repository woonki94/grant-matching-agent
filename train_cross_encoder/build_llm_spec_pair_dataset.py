from __future__ import annotations

import argparse
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
from langchain_core.prompts import ChatPromptTemplate
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
from utils.embedder import embed_texts


class RankedQueryRow(BaseModel):
    q: int = Field(..., description="1-based query index within batch input")
    ranked: List[int] = Field(default_factory=list, description="Candidate indices in best->worst order")


class RankedQueryOut(BaseModel):
    items: List[RankedQueryRow] = Field(default_factory=list)


RANK_BATCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
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
        ),
        ("human", "Tasks JSON:\n{tasks_json}"),
    ]
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
    if math.isnan(parsed) or math.isinf(parsed):
        return float(default)
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _coerce_vector(value: Any) -> List[float]:
    if not isinstance(value, (list, tuple)):
        return []
    out: List[float] = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            return []
    return out


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


def _embed_spec_rows(
    specs: Sequence[SpecRow],
    *,
    batch_size: int,
) -> List[SpecRow]:
    rows = list(specs or [])
    if not rows:
        return []
    safe_batch = _safe_limit(batch_size, default=64, minimum=1, maximum=512)

    unique_texts: List[str] = []
    seen = set()
    for row in rows:
        t = _clean_text(row.text)
        if not t or t in seen:
            continue
        seen.add(t)
        unique_texts.append(t)

    by_text: Dict[str, List[float]] = {}
    for i in range(0, len(unique_texts), safe_batch):
        chunk = unique_texts[i: i + safe_batch]
        vecs = embed_texts(chunk)
        if vecs.ndim != 2 or vecs.shape[0] != len(chunk):
            raise RuntimeError("Embedding batch returned invalid shape.")
        for j, t in enumerate(chunk):
            by_text[t] = [float(x) for x in vecs[j].tolist()]

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
        remaining = [int(x) for x in ordered if int(x) not in set(selected)]
        random_idx: List[int] = []
        if remaining and random_neg_per_query > 0:
            pick_n = min(len(remaining), int(random_neg_per_query))
            random_idx = rng.sample(remaining, pick_n)

        combined = list(dict.fromkeys(selected + random_idx))

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
        random_set = set(random_idx)
        candidates: List[Dict[str, Any]] = []
        for local_i, f_idx in enumerate(combined, start=1):
            fac_spec = faculty_specs[int(f_idx)]
            if int(f_idx) in top_set:
                ctype = "topk"
            elif int(f_idx) in hard_set:
                ctype = "hard_negative"
            elif int(f_idx) in random_set:
                ctype = "random_negative"
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
    llm = get_llm_client(model_id).build()
    return RANK_BATCH_PROMPT | llm.with_structured_output(RankedQueryOut)


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
    seed: int,
    output_dir: Path,
    output_prefix: str,
) -> Dict[str, Any]:
    safe_top_k = _safe_limit(top_k_candidates, default=8, minimum=1, maximum=200)
    safe_hard = _safe_limit(hard_negatives_per_grant, default=4, minimum=0, maximum=200)
    safe_rand = _safe_limit(random_negatives_per_grant, default=4, minimum=0, maximum=200)
    safe_candidates_per_query = _safe_limit(candidates_per_query, default=16, minimum=2, maximum=128)
    safe_max_queries = _safe_limit(max_queries, default=5000, minimum=1, maximum=2_000_000)
    safe_max_pairs = _safe_limit(max_pairs, default=200000, minimum=1, maximum=5_000_000)
    safe_batch = _safe_limit(llm_batch_size, default=8, minimum=1, maximum=64)
    safe_retries = _safe_limit(llm_max_retries, default=2, minimum=1, maximum=8)
    safe_workers = _safe_limit(llm_max_workers, default=4, minimum=1, maximum=64)
    safe_fac_limit = _safe_limit(faculty_limit, default=200000, minimum=1, maximum=2_000_000)
    safe_grant_limit = _safe_limit(grant_limit, default=200000, minimum=1, maximum=2_000_000)
    safe_embed_batch = _safe_limit(embed_batch_size, default=64, minimum=1, maximum=512)
    safe_seed = int(seed)

    faculty_specs_raw = _fetch_faculty_specs(
        min_spec_weight=faculty_min_spec_weight,
        limit=safe_fac_limit,
    )
    grant_specs_raw = _fetch_grant_specs(
        min_spec_weight=grant_min_spec_weight,
        limit=safe_grant_limit,
    )

    faculty_specs_raw = _embed_spec_rows(faculty_specs_raw, batch_size=safe_embed_batch)
    grant_specs_raw = _embed_spec_rows(grant_specs_raw, batch_size=safe_embed_batch)

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

    ranked_queries, llm_meta = _label_query_rankings_with_llm(
        query_sets=query_sets,
        llm_batch_size=safe_batch,
        model_id=(llm_model or settings.haiku or "").strip(),
        max_retries=safe_retries,
        llm_max_workers=safe_workers,
    )

    pair_rows, pair_meta = _expand_rankings_to_pairwise_rows(
        ranked_queries=ranked_queries,
        max_pairs=safe_max_pairs,
        seed=safe_seed,
    )

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
                "llm_batch_size": safe_batch,
                "llm_model": (llm_model or settings.haiku or "").strip(),
                "llm_max_retries": safe_retries,
                "llm_max_workers": safe_workers,
                "faculty_min_spec_weight": float(max(0.0, faculty_min_spec_weight)),
                "grant_min_spec_weight": float(max(0.0, grant_min_spec_weight)),
                "faculty_limit": safe_fac_limit,
                "grant_limit": safe_grant_limit,
                "embed_batch_size": safe_embed_batch,
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
                **pair_meta,
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
            "llm_batch_size": safe_batch,
            "llm_model": (llm_model or settings.haiku or "").strip(),
            "llm_max_workers": safe_workers,
            "faculty_min_spec_weight": float(max(0.0, faculty_min_spec_weight)),
            "grant_min_spec_weight": float(max(0.0, grant_min_spec_weight)),
            "faculty_limit": safe_fac_limit,
            "grant_limit": safe_grant_limit,
            "embed_batch_size": safe_embed_batch,
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
            **pair_meta,
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
    parser.add_argument("--top-k-candidates", type=int, default=8, help="Top cosine candidates per query.")
    parser.add_argument("--hard-negatives-per-grant", type=int, default=10, help="Hard negatives per query.")
    parser.add_argument("--random-negatives-per-grant", type=int, default=10, help="Random negatives per query.")
    parser.add_argument("--candidates-per-query", type=int, default=20, help="Total candidates fed to ranker per query.")
    parser.add_argument("--max-queries", type=int, default=5000, help="Max queries (grant specs) to label.")
    parser.add_argument("--max-pairs", type=int, default=200000, help="Max pairwise preference rows to save.")
    parser.add_argument("--llm-batch-size", type=int, default=8, help="Queries per LLM ranking call.")
    parser.add_argument("--llm-model", type=str, default=(settings.haiku or "").strip(), help="Bedrock model id for ranking labels.")
    parser.add_argument("--llm-max-retries", type=int, default=2, help="Retries per LLM batch.")
    parser.add_argument("--llm-max-workers", type=int, default=4, help="Parallel LLM batch workers.")
    parser.add_argument("--faculty-min-spec-weight", type=float, default=0.0, help="Minimum faculty specialization edge weight.")
    parser.add_argument("--grant-min-spec-weight", type=float, default=0.0, help="Minimum grant specialization edge weight.")
    parser.add_argument("--faculty-limit", type=int, default=200000, help="Max faculty keyword rows to fetch from Postgres.")
    parser.add_argument("--grant-limit", type=int, default=200000, help="Max grant/opportunity keyword rows to fetch from Postgres.")
    parser.add_argument("--embed-batch-size", type=int, default=64, help="Specialization text embedding batch size.")
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
