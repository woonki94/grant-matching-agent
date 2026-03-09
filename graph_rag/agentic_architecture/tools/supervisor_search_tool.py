from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np

from graph_rag.agentic_architecture.tools.filter_tool import (
    filter_grant_ids_by_domain_threshold,
    hard_filter_open_grant_ids,
)
from utils.embedder import cosine_sim_matrix, embed_texts


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _coerce_vector(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=np.float32)
        return arr if arr.ndim == 1 else np.zeros((0,), dtype=np.float32)
    if not isinstance(value, (list, tuple)):
        return np.zeros((0,), dtype=np.float32)
    out: List[float] = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            return np.zeros((0,), dtype=np.float32)
    if not out:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def _token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(_clean_text(text).lower()))


def _lexical_score(query: str, text: str) -> float:
    q = _clean_text(query).lower()
    t = _clean_text(text).lower()
    if not q or not t:
        return 0.0
    if q == t:
        return 1.0
    score = 0.0
    if q in t or t in q:
        score = 0.72
    q_tokens = _token_set(q)
    t_tokens = _token_set(t)
    if q_tokens and t_tokens:
        score = max(score, len(q_tokens & t_tokens) / float(max(len(q_tokens), 1)))
    return float(max(0.0, min(1.0, score)))


def _fetch_neo4j_rows(query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        from neo4j import GraphDatabase, RoutingControl
        from graph_rag.common import load_dotenv_if_present, read_neo4j_settings
    except Exception:
        return []

    try:
        load_dotenv_if_present()
        settings = read_neo4j_settings()
    except Exception:
        return []

    try:
        with GraphDatabase.driver(
            settings.uri,
            auth=(settings.username, settings.password),
        ) as driver:
            records, _, _ = driver.execute_query(
                query,
                parameters_=params,
                routing_=RoutingControl.READ,
                database_=settings.database,
            )
    except Exception:
        return []
    return [dict(row or {}) for row in records]


def _rank_rows_by_query(
    *,
    query: str,
    rows: List[Dict[str, Any]],
    text_key: str = "text",
    embedding_key: str = "embedding",
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    safe_top_k = max(1, int(top_k or 1))
    query_text = _clean_text(query)
    query_vec = np.zeros((0,), dtype=np.float32)
    if query_text:
        try:
            emb = embed_texts([query_text])
            if emb.ndim == 2 and emb.shape[0] == 1:
                query_vec = np.asarray(emb[0], dtype=np.float32)
        except Exception:
            query_vec = np.zeros((0,), dtype=np.float32)

    ranked: List[Dict[str, Any]] = []
    for row in rows or []:
        text = _clean_text((row or {}).get(text_key))
        lex = _lexical_score(query_text, text)
        sem = 0.0
        vec = _coerce_vector((row or {}).get(embedding_key))
        if query_vec.size > 0 and vec.size > 0 and int(query_vec.shape[0]) == int(vec.shape[0]):
            try:
                sem = float(cosine_sim_matrix(query_vec.reshape(1, -1), vec.reshape(1, -1))[0, 0])
                sem = max(0.0, min(1.0, sem))
            except Exception:
                sem = 0.0
        payload = dict(row or {})
        payload["score"] = float(max(lex, 0.72 * sem + 0.28 * lex))
        ranked.append(payload)

    ranked.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return ranked[:safe_top_k]


# ------------------------------
# Primary: domain threshold + open-grant hard filter
# ------------------------------
def primary_prefilter_open_grants_for_faculty(
    *,
    faculty_email: str,
    threshold: float = 0.2,
    top_k: int = 100,
    include_closed: bool = False,
) -> List[str]:
    candidate_ids = filter_grant_ids_by_domain_threshold(
        faculty_email=_clean_text(faculty_email).lower(),
        threshold=float(threshold),
    )
    filtered_ids = hard_filter_open_grant_ids(
        opportunity_ids=[_clean_text(x) for x in candidate_ids],
        include_closed=bool(include_closed),
    )
    return filtered_ids[: max(1, int(top_k or 1))]


# ------------------------------
# Intermediary: specialization keyword embedding comparison in GraphRAG
# ------------------------------
def intermediary_compare_specialization_embeddings(
    *,
    faculty_email: str,
    opportunity_id: str,
    support_threshold: float = 0.45,
) -> Dict[str, Any]:
    faculty_rows = _fetch_neo4j_rows(
        """
        MATCH (f:Faculty {email: $email})-[r]->(k:FacultyKeyword)
        WHERE type(r) IN ['HAS_RESEARCH_SPECIALIZATION', 'HAS_APPLICATION_SPECIALIZATION']
          AND k.embedding IS NOT NULL
          AND k.value IS NOT NULL
        RETURN
            k.value AS value,
            k.embedding AS embedding,
            coalesce(r.weight, 0.5) AS weight,
            type(r) AS relation
        """,
        {"email": _clean_text(faculty_email).lower()},
    )
    grant_rows = _fetch_neo4j_rows(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(k:GrantKeyword)
        WHERE type(r) IN ['HAS_RESEARCH_SPECIALIZATION', 'HAS_APPLICATION_SPECIALIZATION']
          AND k.embedding IS NOT NULL
          AND k.value IS NOT NULL
        RETURN
            k.value AS value,
            k.embedding AS embedding,
            coalesce(r.weight, 0.5) AS weight,
            type(r) AS relation
        """,
        {"opportunity_id": _clean_text(opportunity_id)},
    )

    fac_terms = []
    for row in faculty_rows:
        vec = _coerce_vector(row.get("embedding"))
        value = _clean_text(row.get("value"))
        if value and vec.size > 0:
            fac_terms.append({"value": value, "embedding": vec})

    grant_terms = []
    for row in grant_rows:
        vec = _coerce_vector(row.get("embedding"))
        value = _clean_text(row.get("value"))
        if value and vec.size > 0:
            grant_terms.append({"value": value, "embedding": vec, "weight": float(row.get("weight") or 0.5)})

    if not fac_terms or not grant_terms:
        return {
            "mode": "specialization_embedding_compare",
            "support_map": {},
            "supported_required": [],
            "unsupported_required": [row["value"] for row in grant_terms],
            "avg_score_required": 0.0,
        }

    fac_vecs = np.asarray([x["embedding"] for x in fac_terms], dtype=np.float32)
    grant_vecs = np.asarray([x["embedding"] for x in grant_terms], dtype=np.float32)
    if fac_vecs.ndim != 2 or grant_vecs.ndim != 2 or fac_vecs.shape[1] != grant_vecs.shape[1]:
        return {
            "mode": "specialization_embedding_compare",
            "support_map": {},
            "supported_required": [],
            "unsupported_required": [row["value"] for row in grant_terms],
            "avg_score_required": 0.0,
            "warning": "Embedding dimension mismatch.",
        }

    sims = cosine_sim_matrix(grant_vecs, fac_vecs)
    support_map: Dict[str, Dict[str, Any]] = {}
    supported_required: List[str] = []
    unsupported_required: List[str] = []
    scores: List[float] = []

    for i, grant_term in enumerate(grant_terms):
        term = grant_term["value"]
        row_sims = sims[i]
        best_idx = int(np.argmax(row_sims))
        best_score = float(max(0.0, min(1.0, float(row_sims[best_idx]))))
        matched = fac_terms[best_idx]["value"]
        supported = bool(best_score >= float(support_threshold))
        support_map[term] = {
            "score": best_score,
            "supported": supported,
            "matched_faculty_term": matched,
            "grant_weight": float(grant_term.get("weight") or 0.5),
        }
        scores.append(best_score)
        if supported:
            supported_required.append(term)
        else:
            unsupported_required.append(term)

    avg_score = 0.0 if not scores else float(sum(scores) / len(scores))
    return {
        "mode": "specialization_embedding_compare",
        "support_map": support_map,
        "supported_required": supported_required,
        "unsupported_required": unsupported_required,
        "avg_score_required": avg_score,
    }


# ------------------------------
# Deep: query publication chunks / attachments / additional infos
# ------------------------------
def deep_query_publication_chunks(
    *,
    faculty_email: str,
    query: str,
    top_k: int = 10,
    candidate_limit: int = 300,
) -> List[Dict[str, Any]]:
    rows = _fetch_neo4j_rows(
        """
        MATCH (f:Faculty {email: $email})-[:AUTHORED]->(p:FacultyPublication)
        WHERE p.abstract IS NOT NULL
        RETURN
            p.publication_id AS source_id,
            p.title AS source_title,
            p.abstract AS text,
            p.abstract_embedding AS embedding,
            p.year AS year
        LIMIT $candidate_limit
        """,
        {
            "email": _clean_text(faculty_email).lower(),
            "candidate_limit": max(1, int(candidate_limit or 1)),
        },
    )
    for row in rows:
        row["source_type"] = "faculty_publication_abstract"
    return _rank_rows_by_query(query=query, rows=rows, top_k=top_k)


def deep_query_attachment_chunks(
    *,
    opportunity_id: str,
    query: str,
    top_k: int = 10,
    candidate_limit: int = 500,
) -> List[Dict[str, Any]]:
    rows = _fetch_neo4j_rows(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[:HAS_ATTACHMENT_CHUNK]->(c:GrantTextChunk)
        WHERE c.text IS NOT NULL
        RETURN
            c.chunk_id AS source_id,
            c.source_title AS source_title,
            c.source_ref_id AS source_ref_id,
            c.chunk_index AS chunk_index,
            c.text AS text,
            c.embedding AS embedding
        LIMIT $candidate_limit
        """,
        {
            "opportunity_id": _clean_text(opportunity_id),
            "candidate_limit": max(1, int(candidate_limit or 1)),
        },
    )
    for row in rows:
        row["source_type"] = "grant_attachment_chunk"
    return _rank_rows_by_query(query=query, rows=rows, top_k=top_k)


def deep_query_additional_info_chunks(
    *,
    opportunity_id: str,
    query: str,
    top_k: int = 10,
    candidate_limit: int = 500,
) -> List[Dict[str, Any]]:
    rows = _fetch_neo4j_rows(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[:HAS_ADDITIONAL_INFO_CHUNK]->(c:GrantTextChunk)
        WHERE c.text IS NOT NULL
        RETURN
            c.chunk_id AS source_id,
            c.source_title AS source_title,
            c.source_ref_id AS source_ref_id,
            c.chunk_index AS chunk_index,
            c.text AS text,
            c.embedding AS embedding
        LIMIT $candidate_limit
        """,
        {
            "opportunity_id": _clean_text(opportunity_id),
            "candidate_limit": max(1, int(candidate_limit or 1)),
        },
    )
    for row in rows:
        row["source_type"] = "grant_additional_info_chunk"
    return _rank_rows_by_query(query=query, rows=rows, top_k=top_k)
