from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from neo4j import GraphDatabase

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from graph_rag.common import Neo4jSettings, json_ready, load_dotenv_if_present, read_neo4j_settings

FACULTY_SPEC_RELATIONS = [
    "HAS_RESEARCH_SPECIALIZATION",
    "HAS_APPLICATION_SPECIALIZATION",
]
GRANT_SPEC_RELATIONS = [
    "HAS_RESEARCH_SPECIALIZATION",
    "HAS_APPLICATION_SPECIALIZATION",
]
GRANT_CHUNK_RELATIONS = [
    "HAS_SUMMARY_CHUNK",
    "HAS_ADDITIONAL_INFO_CHUNK",
    "HAS_ATTACHMENT_CHUNK",
]


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


def _dedupe_nonempty(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values or []:
        token = _clean_text(value)
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(token)
    return out


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


def _aggregate_support_embedding(
    *,
    supports: Any,
    fallback: np.ndarray,
) -> np.ndarray:
    target_dim: Optional[int] = None
    if fallback.size > 0:
        target_dim = int(fallback.shape[0])

    vecs: List[np.ndarray] = []
    weights: List[float] = []
    for item in list(supports or []):
        if not isinstance(item, dict):
            continue
        vec = _coerce_vector(item.get("embedding"))
        if vec.size == 0:
            continue
        if target_dim is None:
            target_dim = int(vec.shape[0])
        if int(vec.shape[0]) != int(target_dim):
            continue
        support_score = _safe_unit_float(item.get("score"), default=0.5)
        vecs.append(vec)
        weights.append(max(0.05, float(support_score)))

    if not vecs:
        return fallback

    matrix = np.asarray(vecs, dtype=np.float32)
    weight_vec = np.asarray(weights, dtype=np.float32)
    agg = (matrix * weight_vec.reshape(-1, 1)).sum(axis=0) / (float(weight_vec.sum()) + 1e-9)
    return np.asarray(agg, dtype=np.float32)


def _cosine_vec(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if int(a.shape[0]) != int(b.shape[0]):
        return 0.0
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 0.0 or b_norm <= 0.0:
        return 0.0
    sim = float(np.dot(a, b) / ((a_norm * b_norm) + 1e-9))
    return max(0.0, min(1.0, sim))


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


def _list_grant_ids(
    *,
    driver,
    settings: Neo4jSettings,
    include_closed: bool,
    limit: int,
    offset: int,
) -> List[str]:
    query = """
        MATCH (g:Grant)
        WHERE g.opportunity_id IS NOT NULL
        WITH
            g,
            toLower(coalesce(g.opportunity_status, "")) AS status_token,
            coalesce(toString(g.close_date), "") AS close_token
        WITH
            g,
            status_token,
            CASE
                WHEN close_token =~ '^\\d{4}-\\d{2}-\\d{2}.*$' THEN date(substring(close_token, 0, 10))
                ELSE NULL
            END AS close_dt
        WHERE
            $include_closed
            OR (
                NONE(token IN ['closed', 'archived', 'inactive', 'canceled'] WHERE status_token CONTAINS token)
                AND (close_dt IS NULL OR close_dt >= date())
            )
        RETURN g.opportunity_id AS opportunity_id
        ORDER BY g.opportunity_id ASC
        SKIP $offset
    """
    params: Dict[str, Any] = {
        "include_closed": bool(include_closed),
        "offset": max(0, int(offset or 0)),
    }
    if int(limit or 0) > 0:
        query += "\nLIMIT $limit"
        params["limit"] = max(1, int(limit))

    records, _, _ = driver.execute_query(
        query,
        parameters_=params,
        database_=settings.database,
    )
    return _dedupe_nonempty([row.get("opportunity_id") for row in records])


def _list_faculty_emails(
    *,
    driver,
    settings: Neo4jSettings,
    limit: int,
    offset: int,
) -> List[str]:
    query = """
        MATCH (f:Faculty)
        WHERE f.email IS NOT NULL
        RETURN f.email AS email
        ORDER BY f.email ASC
        SKIP $offset
    """
    params: Dict[str, Any] = {
        "offset": max(0, int(offset or 0)),
    }
    if int(limit or 0) > 0:
        query += "\nLIMIT $limit"
        params["limit"] = max(1, int(limit))

    records, _, _ = driver.execute_query(
        query,
        parameters_=params,
        database_=settings.database,
    )
    return _dedupe_nonempty([_clean_text(row.get("email")).lower() for row in records])


def _fetch_faculty_identity(
    *,
    driver,
    settings: Neo4jSettings,
    faculty_email: str,
) -> Optional[Dict[str, Any]]:
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})
        RETURN f.faculty_id AS faculty_id, f.email AS email
        LIMIT 1
        """,
        parameters_={"email": _clean_text(faculty_email).lower()},
        database_=settings.database,
    )
    if not records:
        return None

    row = dict(records[0] or {})
    try:
        faculty_id = int(row.get("faculty_id"))
    except Exception:
        return None
    return {
        "faculty_id": faculty_id,
        "email": _clean_text(row.get("email")).lower(),
    }


def _fetch_faculty_domain_vectors(
    *,
    driver,
    settings: Neo4jSettings,
    faculty_email: str,
) -> Dict[str, np.ndarray]:
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})-[r]->(k:FacultyKeyword)
        WHERE type(r) IN ['HAS_RESEARCH_DOMAIN', 'HAS_APPLICATION_DOMAIN']
          AND k.bucket = 'domain'
          AND k.embedding IS NOT NULL
        RETURN k.section AS section, k.embedding AS embedding
        """,
        parameters_={"email": _clean_text(faculty_email).lower()},
        database_=settings.database,
    )

    out: Dict[str, np.ndarray] = {}
    for row in records:
        section = _clean_text(row.get("section")).lower()
        if section not in {"research", "application"}:
            continue
        vec = _coerce_vector(row.get("embedding"))
        if vec.size == 0:
            continue
        out[section] = vec
    return out


def _fetch_grant_domain_vectors(
    *,
    driver,
    settings: Neo4jSettings,
    opportunity_id: str,
) -> Dict[str, np.ndarray]:
    records, _, _ = driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(k:GrantKeyword)
        WHERE type(r) IN ['HAS_RESEARCH_DOMAIN', 'HAS_APPLICATION_DOMAIN']
          AND k.bucket = 'domain'
          AND k.embedding IS NOT NULL
        RETURN k.section AS section, k.embedding AS embedding
        """,
        parameters_={"opportunity_id": _clean_text(opportunity_id)},
        database_=settings.database,
    )

    out: Dict[str, np.ndarray] = {}
    for row in records:
        section = _clean_text(row.get("section")).lower()
        if section not in {"research", "application"}:
            continue
        vec = _coerce_vector(row.get("embedding"))
        if vec.size == 0:
            continue
        out[section] = vec
    return out


def _domain_gate_score(
    faculty_domain: Dict[str, np.ndarray],
    grant_domain: Dict[str, np.ndarray],
) -> Dict[str, float]:
    r_sim = _cosine_vec(faculty_domain.get("research", np.zeros((0,), dtype=np.float32)), grant_domain.get("research", np.zeros((0,), dtype=np.float32)))
    a_sim = _cosine_vec(faculty_domain.get("application", np.zeros((0,), dtype=np.float32)), grant_domain.get("application", np.zeros((0,), dtype=np.float32)))
    return {
        "research_domain_sim": r_sim,
        "application_domain_sim": a_sim,
        "domain_gate_score": max(r_sim, a_sim),
    }


def _fetch_faculty_spec_keyword_rows(
    *,
    driver,
    settings: Neo4jSettings,
    faculty_email: str,
) -> List[Dict[str, Any]]:
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})-[r]->(k:FacultyKeyword)
        WHERE type(r) IN $relations
          AND k.bucket = 'specialization'
          AND k.embedding IS NOT NULL
          AND k.value IS NOT NULL
        CALL (f, k) {
            OPTIONAL MATCH (k)-[s:SUPPORTED_BY_FACULTY_CHUNK]->(:FacultyTextChunk)
            WHERE s.scope_faculty_id = f.faculty_id
            RETURN coalesce(max(s.score), 0.0) AS chunk_conf
        }
        CALL (f, k) {
            OPTIONAL MATCH (k)-[s:SUPPORTED_BY_FACULTY_PUBLICATION]->(p:FacultyPublication)
            WHERE p.faculty_id = f.faculty_id
            RETURN coalesce(max(s.score), 0.0) AS pub_conf
        }
        CALL (f, k) {
            OPTIONAL MATCH (k)-[s:SUPPORTED_BY_FACULTY_CHUNK]->(c:FacultyTextChunk)
            WHERE s.scope_faculty_id = f.faculty_id
              AND c.embedding IS NOT NULL
            RETURN collect({
                embedding: c.embedding,
                score: coalesce(s.score, 0.0)
            }) AS chunk_supports
        }
        CALL (f, k) {
            OPTIONAL MATCH (k)-[s:SUPPORTED_BY_FACULTY_PUBLICATION]->(p:FacultyPublication)
            WHERE p.faculty_id = f.faculty_id
              AND p.abstract_embedding IS NOT NULL
            RETURN collect({
                embedding: p.abstract_embedding,
                score: coalesce(s.score, 0.0)
            }) AS pub_supports
        }
        RETURN DISTINCT
            k.value AS keyword_value,
            k.section AS keyword_section,
            k.embedding AS embedding,
            coalesce(r.weight, 0.5) AS keyword_weight,
            CASE WHEN chunk_conf >= pub_conf THEN chunk_conf ELSE pub_conf END AS keyword_confidence,
            chunk_supports,
            pub_supports
        ORDER BY keyword_value ASC
        """,
        parameters_={
            "email": _clean_text(faculty_email).lower(),
            "relations": FACULTY_SPEC_RELATIONS,
        },
        database_=settings.database,
    )
    return [dict(row or {}) for row in records]


def _fetch_grant_spec_keyword_rows(
    *,
    driver,
    settings: Neo4jSettings,
    opportunity_id: str,
) -> List[Dict[str, Any]]:
    records, _, _ = driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(k:GrantKeyword)
        WHERE type(r) IN $relations
          AND k.bucket = 'specialization'
          AND k.embedding IS NOT NULL
          AND k.value IS NOT NULL
        CALL (g, k) {
            OPTIONAL MATCH (k)-[s:SUPPORTED_BY_GRANT_CHUNK]->(c:GrantTextChunk)
            WHERE c.opportunity_id = g.opportunity_id
            RETURN coalesce(max(s.score), 0.0) AS chunk_conf
        }
        CALL (g, k) {
            OPTIONAL MATCH (k)-[s:SUPPORTED_BY_GRANT_CHUNK]->(c:GrantTextChunk)
            WHERE c.opportunity_id = g.opportunity_id
              AND c.embedding IS NOT NULL
            RETURN collect({
                embedding: c.embedding,
                score: coalesce(s.score, 0.0)
            }) AS chunk_supports
        }
        RETURN DISTINCT
            k.value AS keyword_value,
            k.section AS keyword_section,
            k.embedding AS embedding,
            coalesce(r.weight, 0.5) AS keyword_weight,
            chunk_conf AS keyword_confidence,
            chunk_supports
        ORDER BY keyword_value ASC
        """,
        parameters_={
            "opportunity_id": _clean_text(opportunity_id),
            "relations": GRANT_SPEC_RELATIONS,
        },
        database_=settings.database,
    )
    return [dict(row or {}) for row in records]


def _fetch_grant_chunk_rows(
    *,
    driver,
    settings: Neo4jSettings,
    opportunity_id: str,
) -> List[Dict[str, Any]]:
    records, _, _ = driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(c:GrantTextChunk)
        WHERE type(r) IN $relations
          AND c.embedding IS NOT NULL
          AND c.chunk_id IS NOT NULL
        RETURN DISTINCT
            c.chunk_id AS chunk_id,
            c.source_type AS source_type,
            c.embedding AS embedding
        ORDER BY c.chunk_id ASC
        """,
        parameters_={
            "opportunity_id": _clean_text(opportunity_id),
            "relations": GRANT_CHUNK_RELATIONS,
        },
        database_=settings.database,
    )
    return [dict(row or {}) for row in records]


def _fetch_faculty_evidence_rows(
    *,
    driver,
    settings: Neo4jSettings,
    faculty_email: str,
) -> List[Dict[str, Any]]:
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})
        CALL (f) {
            WITH f
            MATCH (f)-[r]->(c:FacultyTextChunk)
            WHERE c.embedding IS NOT NULL
              AND c.chunk_id IS NOT NULL
            RETURN
                'chunk' AS evidence_kind,
                c.chunk_id AS evidence_id,
                c.source_type AS source_type,
                c.embedding AS embedding
            UNION ALL
            WITH f
            MATCH (f)-[:AUTHORED]->(p:FacultyPublication)
            WHERE p.publication_id IS NOT NULL
              AND p.abstract_embedding IS NOT NULL
            RETURN
                'publication' AS evidence_kind,
                toString(p.publication_id) AS evidence_id,
                'publication_abstract' AS source_type,
                p.abstract_embedding AS embedding
        }
        RETURN DISTINCT evidence_kind, evidence_id, source_type, embedding
        ORDER BY evidence_kind ASC, evidence_id ASC
        """,
        parameters_={"email": _clean_text(faculty_email).lower()},
        database_=settings.database,
    )
    return [dict(row or {}) for row in records]


def _prepare_faculty_spec_vectors(
    *,
    rows: List[Dict[str, Any]],
    min_keyword_confidence: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    safe_min_kw_conf = max(0.0, min(1.0, float(min_keyword_confidence)))
    for row in rows or []:
        keyword_value = _clean_text(row.get("keyword_value"))
        keyword_section = _clean_text(row.get("keyword_section")).lower()
        fallback_vec = _coerce_vector(row.get("embedding"))
        supports: List[Dict[str, Any]] = []
        supports.extend(list(row.get("chunk_supports") or []))
        supports.extend(list(row.get("pub_supports") or []))
        vec = _aggregate_support_embedding(supports=supports, fallback=fallback_vec)
        if not keyword_value or vec.size == 0:
            continue
        out.append(
            {
                "keyword_value": keyword_value,
                "keyword_section": keyword_section,
                "vector": vec,
                "keyword_weight": _safe_unit_float(row.get("keyword_weight"), default=0.5),
                "keyword_confidence": max(
                    safe_min_kw_conf,
                    _safe_unit_float(row.get("keyword_confidence"), default=safe_min_kw_conf),
                ),
            }
        )
    return out


def _prepare_grant_spec_vectors(
    *,
    rows: List[Dict[str, Any]],
    min_keyword_confidence: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    safe_min_kw_conf = max(0.0, min(1.0, float(min_keyword_confidence)))
    for row in rows or []:
        keyword_value = _clean_text(row.get("keyword_value"))
        keyword_section = _clean_text(row.get("keyword_section")).lower()
        fallback_vec = _coerce_vector(row.get("embedding"))
        vec = _aggregate_support_embedding(
            supports=row.get("chunk_supports"),
            fallback=fallback_vec,
        )
        if not keyword_value or vec.size == 0:
            continue
        out.append(
            {
                "keyword_value": keyword_value,
                "keyword_section": keyword_section,
                "vector": vec,
                "keyword_weight": _safe_unit_float(row.get("keyword_weight"), default=0.5),
                "keyword_confidence": max(
                    safe_min_kw_conf,
                    _safe_unit_float(row.get("keyword_confidence"), default=safe_min_kw_conf),
                ),
            }
        )
    return out


def _prepare_grant_chunks(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows or []:
        chunk_id = _clean_text(row.get("chunk_id"))
        vec = _coerce_vector(row.get("embedding"))
        if not chunk_id or vec.size == 0:
            continue
        out.append(
            {
                "chunk_id": chunk_id,
                "source_type": _clean_text(row.get("source_type")),
                "vector": vec,
            }
        )
    return out


def _prepare_faculty_evidence(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows or []:
        evidence_kind = _clean_text(row.get("evidence_kind")).lower()
        evidence_id = _clean_text(row.get("evidence_id"))
        vec = _coerce_vector(row.get("embedding"))
        if evidence_kind not in {"chunk", "publication"}:
            continue
        if not evidence_id or vec.size == 0:
            continue
        out.append(
            {
                "evidence_kind": evidence_kind,
                "evidence_id": evidence_id,
                "source_type": _clean_text(row.get("source_type")),
                "vector": vec,
            }
        )
    return out


def _rank_faculty_spec_to_grant_chunks(
    *,
    faculty_specs: List[Dict[str, Any]],
    grant_chunks: List[Dict[str, Any]],
    min_sim: float,
    top_k_per_keyword: int,
    domain_gate_score: float,
) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    safe_min_sim = max(0.0, min(1.0, float(min_sim)))
    safe_top_k = _safe_limit(top_k_per_keyword, default=3, minimum=1, maximum=50)
    safe_domain = max(0.0, min(1.0, float(domain_gate_score)))

    if not faculty_specs or not grant_chunks:
        return links

    prepared_chunks = [row for row in grant_chunks if _coerce_vector(row.get("vector")).size > 0]
    if not prepared_chunks:
        return links

    for fk in faculty_specs:
        fk_vec = _coerce_vector(fk.get("vector"))
        if fk_vec.size == 0:
            continue

        compatible = []
        for gc in prepared_chunks:
            gc_vec = _coerce_vector(gc.get("vector"))
            if int(gc_vec.shape[0]) != int(fk_vec.shape[0]):
                continue
            compatible.append(gc)
        if not compatible:
            continue

        matrix = np.asarray([_coerce_vector(x.get("vector")) for x in compatible], dtype=np.float32)
        sims = _cosine_matrix(fk_vec.reshape(1, -1), matrix)[0]
        order = np.argsort(-sims)

        kept = 0
        for idx in order:
            cosine_sim = float(sims[int(idx)])
            if cosine_sim < safe_min_sim:
                continue

            chunk_row = compatible[int(idx)]
            fk_weight = _safe_unit_float(fk.get("keyword_weight"), default=0.5)
            fk_conf = _safe_unit_float(fk.get("keyword_confidence"), default=0.5)
            score = max(0.0, min(1.0, cosine_sim * fk_weight * fk_conf * safe_domain))

            links.append(
                {
                    "faculty_keyword_value": _clean_text(fk.get("keyword_value")),
                    "faculty_keyword_section": _clean_text(fk.get("keyword_section")).lower(),
                    "grant_chunk_id": _clean_text(chunk_row.get("chunk_id")),
                    "grant_chunk_source_type": _clean_text(chunk_row.get("source_type")),
                    "cosine_sim": max(0.0, min(1.0, cosine_sim)),
                    "faculty_keyword_weight": fk_weight,
                    "faculty_keyword_confidence": fk_conf,
                    "domain_gate_score": safe_domain,
                    "score": score,
                }
            )

            kept += 1
            if kept >= safe_top_k:
                break

    return links


def _rank_grant_spec_to_faculty_evidence(
    *,
    grant_specs: List[Dict[str, Any]],
    faculty_evidence: List[Dict[str, Any]],
    min_sim: float,
    top_k_per_keyword: int,
    domain_gate_score: float,
) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    safe_min_sim = max(0.0, min(1.0, float(min_sim)))
    safe_top_k = _safe_limit(top_k_per_keyword, default=3, minimum=1, maximum=50)
    safe_domain = max(0.0, min(1.0, float(domain_gate_score)))

    if not grant_specs or not faculty_evidence:
        return links

    prepared_evidence = [row for row in faculty_evidence if _coerce_vector(row.get("vector")).size > 0]
    if not prepared_evidence:
        return links

    for gk in grant_specs:
        gk_vec = _coerce_vector(gk.get("vector"))
        if gk_vec.size == 0:
            continue

        compatible = []
        for ev in prepared_evidence:
            ev_vec = _coerce_vector(ev.get("vector"))
            if int(ev_vec.shape[0]) != int(gk_vec.shape[0]):
                continue
            compatible.append(ev)
        if not compatible:
            continue

        matrix = np.asarray([_coerce_vector(x.get("vector")) for x in compatible], dtype=np.float32)
        sims = _cosine_matrix(gk_vec.reshape(1, -1), matrix)[0]
        order = np.argsort(-sims)

        kept = 0
        for idx in order:
            cosine_sim = float(sims[int(idx)])
            if cosine_sim < safe_min_sim:
                continue

            ev_row = compatible[int(idx)]
            gk_weight = _safe_unit_float(gk.get("keyword_weight"), default=0.5)
            gk_conf = _safe_unit_float(gk.get("keyword_confidence"), default=0.5)
            score = max(0.0, min(1.0, cosine_sim * gk_weight * gk_conf * safe_domain))

            links.append(
                {
                    "grant_keyword_value": _clean_text(gk.get("keyword_value")),
                    "grant_keyword_section": _clean_text(gk.get("keyword_section")).lower(),
                    "faculty_evidence_kind": _clean_text(ev_row.get("evidence_kind")).lower(),
                    "faculty_evidence_id": _clean_text(ev_row.get("evidence_id")),
                    "faculty_evidence_source_type": _clean_text(ev_row.get("source_type")),
                    "cosine_sim": max(0.0, min(1.0, cosine_sim)),
                    "grant_keyword_weight": gk_weight,
                    "grant_keyword_confidence": gk_conf,
                    "domain_gate_score": safe_domain,
                    "score": score,
                }
            )

            kept += 1
            if kept >= safe_top_k:
                break

    return links


def _clear_scope_edges(
    *,
    driver,
    settings: Neo4jSettings,
    faculty_id: int,
    opportunity_id: str,
) -> None:
    driver.execute_query(
        """
        MATCH (:FacultyKeyword)-[r:FAC_SPEC_SUPPORTS_GRANT_CHUNK {
            scope_faculty_id: $faculty_id,
            scope_opportunity_id: $opportunity_id
        }]->(:GrantTextChunk)
        DELETE r
        """,
        parameters_={
            "faculty_id": int(faculty_id),
            "opportunity_id": _clean_text(opportunity_id),
        },
        database_=settings.database,
    )
    driver.execute_query(
        """
        MATCH (:GrantKeyword)-[r:GRANT_SPEC_SUPPORTS_FAC_EVIDENCE {
            scope_faculty_id: $faculty_id,
            scope_opportunity_id: $opportunity_id
        }]->()
        DELETE r
        """,
        parameters_={
            "faculty_id": int(faculty_id),
            "opportunity_id": _clean_text(opportunity_id),
        },
        database_=settings.database,
    )


def _write_fac_to_grant_chunk_edges(
    *,
    driver,
    settings: Neo4jSettings,
    faculty_id: int,
    faculty_email: str,
    opportunity_id: str,
    rows: List[Dict[str, Any]],
    min_chunk_sim: float,
    min_domain_sim: float,
) -> int:
    if not rows:
        return 0

    records, _, _ = driver.execute_query(
        """
        UNWIND $rows AS row
        MATCH (fk:FacultyKeyword {
            value: row.faculty_keyword_value,
            section: row.faculty_keyword_section,
            bucket: 'specialization'
        })
        MATCH (gc:GrantTextChunk {chunk_id: row.grant_chunk_id})
        WHERE gc.opportunity_id = $opportunity_id
        MERGE (fk)-[r:FAC_SPEC_SUPPORTS_GRANT_CHUNK {
            scope_faculty_id: $faculty_id,
            scope_faculty_email: $faculty_email,
            scope_opportunity_id: $opportunity_id,
            faculty_keyword_value: row.faculty_keyword_value,
            faculty_keyword_section: row.faculty_keyword_section,
            grant_chunk_id: row.grant_chunk_id
        }]->(gc)
        SET
            r.sim = row.cosine_sim,
            r.score = row.score,
            r.domain_gate_score = row.domain_gate_score,
            r.faculty_keyword_weight = row.faculty_keyword_weight,
            r.faculty_keyword_confidence = row.faculty_keyword_confidence,
            r.grant_chunk_source_type = row.grant_chunk_source_type,
            r.min_chunk_sim = $min_chunk_sim,
            r.min_domain_sim = $min_domain_sim,
            r.method = 'domain_prefilter_plus_spec_to_grant_chunk',
            r.updated_at = datetime()
        RETURN count(r) AS linked_count
        """,
        parameters_={
            "faculty_id": int(faculty_id),
            "faculty_email": _clean_text(faculty_email).lower(),
            "opportunity_id": _clean_text(opportunity_id),
            "rows": rows,
            "min_chunk_sim": max(0.0, min(1.0, float(min_chunk_sim))),
            "min_domain_sim": max(0.0, min(1.0, float(min_domain_sim))),
        },
        database_=settings.database,
    )
    if not records:
        return 0
    try:
        return int(records[0].get("linked_count") or 0)
    except Exception:
        return 0


def _write_grant_to_fac_evidence_edges(
    *,
    driver,
    settings: Neo4jSettings,
    faculty_id: int,
    faculty_email: str,
    opportunity_id: str,
    rows: List[Dict[str, Any]],
    min_chunk_sim: float,
    min_domain_sim: float,
) -> int:
    if not rows:
        return 0

    chunk_rows = [x for x in rows if _clean_text(x.get("faculty_evidence_kind")).lower() == "chunk"]
    pub_rows = [x for x in rows if _clean_text(x.get("faculty_evidence_kind")).lower() == "publication"]
    linked = 0

    if chunk_rows:
        records, _, _ = driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (gk:GrantKeyword {
                value: row.grant_keyword_value,
                section: row.grant_keyword_section,
                bucket: 'specialization'
            })
            MATCH (fc:FacultyTextChunk {chunk_id: row.faculty_evidence_id})
            MERGE (gk)-[r:GRANT_SPEC_SUPPORTS_FAC_EVIDENCE {
                scope_faculty_id: $faculty_id,
                scope_faculty_email: $faculty_email,
                scope_opportunity_id: $opportunity_id,
                grant_keyword_value: row.grant_keyword_value,
                grant_keyword_section: row.grant_keyword_section,
                faculty_evidence_kind: 'chunk',
                faculty_chunk_id: row.faculty_evidence_id
            }]->(fc)
            SET
                r.sim = row.cosine_sim,
                r.score = row.score,
                r.domain_gate_score = row.domain_gate_score,
                r.grant_keyword_weight = row.grant_keyword_weight,
                r.grant_keyword_confidence = row.grant_keyword_confidence,
                r.faculty_evidence_source_type = row.faculty_evidence_source_type,
                r.min_chunk_sim = $min_chunk_sim,
                r.min_domain_sim = $min_domain_sim,
                r.method = 'domain_prefilter_plus_grant_spec_to_fac_evidence',
                r.updated_at = datetime()
            RETURN count(r) AS linked_count
            """,
            parameters_={
                "faculty_id": int(faculty_id),
                "faculty_email": _clean_text(faculty_email).lower(),
                "opportunity_id": _clean_text(opportunity_id),
                "rows": chunk_rows,
                "min_chunk_sim": max(0.0, min(1.0, float(min_chunk_sim))),
                "min_domain_sim": max(0.0, min(1.0, float(min_domain_sim))),
            },
            database_=settings.database,
        )
        if records:
            try:
                linked += int(records[0].get("linked_count") or 0)
            except Exception:
                pass

    if pub_rows:
        normalized_rows: List[Dict[str, Any]] = []
        for row in pub_rows:
            pub_id_txt = _clean_text(row.get("faculty_evidence_id"))
            try:
                pub_id = int(pub_id_txt)
            except Exception:
                continue
            normalized = dict(row)
            normalized["faculty_publication_id"] = pub_id
            normalized_rows.append(normalized)

        if normalized_rows:
            records, _, _ = driver.execute_query(
                """
                UNWIND $rows AS row
                MATCH (gk:GrantKeyword {
                    value: row.grant_keyword_value,
                    section: row.grant_keyword_section,
                    bucket: 'specialization'
                })
                MATCH (fp:FacultyPublication {publication_id: row.faculty_publication_id})
                WHERE fp.faculty_id = $faculty_id
                MERGE (gk)-[r:GRANT_SPEC_SUPPORTS_FAC_EVIDENCE {
                    scope_faculty_id: $faculty_id,
                    scope_faculty_email: $faculty_email,
                    scope_opportunity_id: $opportunity_id,
                    grant_keyword_value: row.grant_keyword_value,
                    grant_keyword_section: row.grant_keyword_section,
                    faculty_evidence_kind: 'publication',
                    faculty_publication_id: row.faculty_publication_id
                }]->(fp)
                SET
                    r.sim = row.cosine_sim,
                    r.score = row.score,
                    r.domain_gate_score = row.domain_gate_score,
                    r.grant_keyword_weight = row.grant_keyword_weight,
                    r.grant_keyword_confidence = row.grant_keyword_confidence,
                    r.faculty_evidence_source_type = row.faculty_evidence_source_type,
                    r.min_chunk_sim = $min_chunk_sim,
                    r.min_domain_sim = $min_domain_sim,
                    r.method = 'domain_prefilter_plus_grant_spec_to_fac_evidence',
                    r.updated_at = datetime()
                RETURN count(r) AS linked_count
                """,
                parameters_={
                    "faculty_id": int(faculty_id),
                    "faculty_email": _clean_text(faculty_email).lower(),
                    "opportunity_id": _clean_text(opportunity_id),
                    "rows": normalized_rows,
                    "min_chunk_sim": max(0.0, min(1.0, float(min_chunk_sim))),
                    "min_domain_sim": max(0.0, min(1.0, float(min_domain_sim))),
                },
                database_=settings.database,
            )
            if records:
                try:
                    linked += int(records[0].get("linked_count") or 0)
                except Exception:
                    pass

    return linked


def run_domain_prefilter_spec_chunk_linker(
    *,
    faculty_emails: Sequence[str],
    grant_ids: Sequence[str],
    all_faculty: bool,
    all_grants: bool,
    include_closed: bool,
    limit: int,
    offset: int,
    min_domain_sim: float,
    domain_top_n: int,
    min_chunk_sim: float,
    top_k_fac_to_grant_chunk: int,
    top_k_grant_to_fac_evidence: int,
    min_keyword_confidence: float,
    write_edges: bool,
    uri: str = "",
    username: str = "",
    password: str = "",
    database: str = "",
) -> Dict[str, Any]:
    load_dotenv_if_present()
    settings = read_neo4j_settings(
        uri=uri,
        username=username,
        password=password,
        database=database,
    )

    safe_limit = _safe_limit(limit, default=0, minimum=0, maximum=100000)
    safe_offset = _safe_limit(offset, default=0, minimum=0, maximum=1000000)
    safe_min_domain_sim = max(0.0, min(1.0, float(min_domain_sim)))
    safe_domain_top_n = _safe_limit(domain_top_n, default=100, minimum=0, maximum=5000)
    safe_min_chunk_sim = max(0.0, min(1.0, float(min_chunk_sim)))
    safe_top_k_f2g = _safe_limit(top_k_fac_to_grant_chunk, default=5, minimum=1, maximum=50)
    safe_top_k_g2f = _safe_limit(top_k_grant_to_fac_evidence, default=5, minimum=1, maximum=50)
    safe_min_keyword_confidence = max(0.0, min(1.0, float(min_keyword_confidence)))

    targets_faculty = _dedupe_nonempty([_clean_text(x).lower() for x in list(faculty_emails or [])])
    targets_grants = _dedupe_nonempty([_clean_text(x) for x in list(grant_ids or [])])

    with GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    ) as driver:
        driver.verify_connectivity()

        if bool(all_faculty):
            targets_faculty = _list_faculty_emails(
                driver=driver,
                settings=settings,
                limit=safe_limit,
                offset=safe_offset,
            )

        if bool(all_grants) or not targets_grants:
            targets_grants = _list_grant_ids(
                driver=driver,
                settings=settings,
                include_closed=bool(include_closed),
                limit=safe_limit,
                offset=safe_offset,
            )

        if not targets_faculty:
            return {
                "status": "skipped",
                "reason": "no_faculty_targets",
            }
        if not targets_grants:
            return {
                "status": "skipped",
                "reason": "no_grant_targets",
            }

        grant_domain_cache: Dict[str, Dict[str, np.ndarray]] = {}
        grant_spec_cache: Dict[str, List[Dict[str, Any]]] = {}
        grant_chunk_cache: Dict[str, List[Dict[str, Any]]] = {}

        results: List[Dict[str, Any]] = []

        for faculty_email in targets_faculty:
            faculty = _fetch_faculty_identity(
                driver=driver,
                settings=settings,
                faculty_email=faculty_email,
            )
            if not faculty:
                results.append(
                    {
                        "faculty_email": _clean_text(faculty_email).lower(),
                        "status": "faculty_not_found",
                    }
                )
                continue

            faculty_domain = _fetch_faculty_domain_vectors(
                driver=driver,
                settings=settings,
                faculty_email=faculty_email,
            )
            faculty_specs = _prepare_faculty_spec_vectors(
                rows=_fetch_faculty_spec_keyword_rows(
                    driver=driver,
                    settings=settings,
                    faculty_email=faculty_email,
                ),
                min_keyword_confidence=safe_min_keyword_confidence,
            )
            faculty_evidence = _prepare_faculty_evidence(
                _fetch_faculty_evidence_rows(
                    driver=driver,
                    settings=settings,
                    faculty_email=faculty_email,
                )
            )

            if not faculty_specs:
                results.append(
                    {
                        "faculty_id": int(faculty["faculty_id"]),
                        "faculty_email": _clean_text(faculty_email).lower(),
                        "status": "no_faculty_specialization_keywords",
                    }
                )
                continue

            candidates: List[Dict[str, Any]] = []
            for opp_id in targets_grants:
                grant_id = _clean_text(opp_id)
                if not grant_id:
                    continue
                if grant_id not in grant_domain_cache:
                    grant_domain_cache[grant_id] = _fetch_grant_domain_vectors(
                        driver=driver,
                        settings=settings,
                        opportunity_id=grant_id,
                    )
                domain_scores = _domain_gate_score(faculty_domain, grant_domain_cache[grant_id])
                if float(domain_scores["domain_gate_score"]) < safe_min_domain_sim:
                    continue
                candidates.append(
                    {
                        "opportunity_id": grant_id,
                        **domain_scores,
                    }
                )

            candidates.sort(key=lambda x: float(x.get("domain_gate_score") or 0.0), reverse=True)
            if safe_domain_top_n > 0:
                candidates = candidates[:safe_domain_top_n]

            faculty_rows: List[Dict[str, Any]] = []
            for candidate in candidates:
                opp_id = _clean_text(candidate.get("opportunity_id"))
                if not opp_id:
                    continue

                if opp_id not in grant_spec_cache:
                    grant_spec_cache[opp_id] = _prepare_grant_spec_vectors(
                        rows=_fetch_grant_spec_keyword_rows(
                            driver=driver,
                            settings=settings,
                            opportunity_id=opp_id,
                        ),
                        min_keyword_confidence=safe_min_keyword_confidence,
                    )
                if opp_id not in grant_chunk_cache:
                    grant_chunk_cache[opp_id] = _prepare_grant_chunks(
                        _fetch_grant_chunk_rows(
                            driver=driver,
                            settings=settings,
                            opportunity_id=opp_id,
                        )
                    )

                grant_specs = grant_spec_cache[opp_id]
                grant_chunks = grant_chunk_cache[opp_id]

                fac_to_grant_rows = _rank_faculty_spec_to_grant_chunks(
                    faculty_specs=faculty_specs,
                    grant_chunks=grant_chunks,
                    min_sim=safe_min_chunk_sim,
                    top_k_per_keyword=safe_top_k_f2g,
                    domain_gate_score=float(candidate.get("domain_gate_score") or 0.0),
                )
                grant_to_fac_rows = _rank_grant_spec_to_faculty_evidence(
                    grant_specs=grant_specs,
                    faculty_evidence=faculty_evidence,
                    min_sim=safe_min_chunk_sim,
                    top_k_per_keyword=safe_top_k_g2f,
                    domain_gate_score=float(candidate.get("domain_gate_score") or 0.0),
                )

                linked_f2g = 0
                linked_g2f = 0
                if write_edges:
                    _clear_scope_edges(
                        driver=driver,
                        settings=settings,
                        faculty_id=int(faculty["faculty_id"]),
                        opportunity_id=opp_id,
                    )
                    linked_f2g = _write_fac_to_grant_chunk_edges(
                        driver=driver,
                        settings=settings,
                        faculty_id=int(faculty["faculty_id"]),
                        faculty_email=_clean_text(faculty_email).lower(),
                        opportunity_id=opp_id,
                        rows=fac_to_grant_rows,
                        min_chunk_sim=safe_min_chunk_sim,
                        min_domain_sim=safe_min_domain_sim,
                    )
                    linked_g2f = _write_grant_to_fac_evidence_edges(
                        driver=driver,
                        settings=settings,
                        faculty_id=int(faculty["faculty_id"]),
                        faculty_email=_clean_text(faculty_email).lower(),
                        opportunity_id=opp_id,
                        rows=grant_to_fac_rows,
                        min_chunk_sim=safe_min_chunk_sim,
                        min_domain_sim=safe_min_domain_sim,
                    )

                faculty_rows.append(
                    {
                        "opportunity_id": opp_id,
                        "domain_gate_score": float(candidate.get("domain_gate_score") or 0.0),
                        "research_domain_sim": float(candidate.get("research_domain_sim") or 0.0),
                        "application_domain_sim": float(candidate.get("application_domain_sim") or 0.0),
                        "faculty_to_grant_chunk_edges": int(linked_f2g) if write_edges else len(fac_to_grant_rows),
                        "grant_to_fac_evidence_edges": int(linked_g2f) if write_edges else len(grant_to_fac_rows),
                        "faculty_specs": len(faculty_specs),
                        "grant_specs": len(grant_specs),
                        "grant_chunks": len(grant_chunks),
                        "faculty_evidence": len(faculty_evidence),
                    }
                )

            results.append(
                {
                    "faculty_id": int(faculty["faculty_id"]),
                    "faculty_email": _clean_text(faculty_email).lower(),
                    "candidate_grants": len(candidates),
                    "matched_grants": len(faculty_rows),
                    "rows": faculty_rows,
                }
            )

    total_f2g = 0
    total_g2f = 0
    for item in results:
        for row in list(item.get("rows") or []):
            total_f2g += int(row.get("faculty_to_grant_chunk_edges", 0) or 0)
            total_g2f += int(row.get("grant_to_fac_evidence_edges", 0) or 0)

    return {
        "params": {
            "all_faculty": bool(all_faculty),
            "all_grants": bool(all_grants),
            "include_closed": bool(include_closed),
            "limit": safe_limit,
            "offset": safe_offset,
            "min_domain_sim": safe_min_domain_sim,
            "domain_top_n": safe_domain_top_n,
            "min_chunk_sim": safe_min_chunk_sim,
            "top_k_fac_to_grant_chunk": safe_top_k_f2g,
            "top_k_grant_to_fac_evidence": safe_top_k_g2f,
            "min_keyword_confidence": safe_min_keyword_confidence,
            "write_edges": bool(write_edges),
        },
        "totals": {
            "faculties_processed": len(results),
            "faculty_to_grant_chunk_edges": total_f2g,
            "grant_to_fac_evidence_edges": total_g2f,
            "all_edges": int(total_f2g + total_g2f),
        },
        "faculties": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage matcher: domain prefilter first, then write cross-chunk specialization edges "
            "(faculty keyword -> grant chunks, grant keyword -> faculty chunk/publication)."
        )
    )
    parser.add_argument("--faculty-email", action="append", default=[], help="Target faculty email (repeatable).")
    parser.add_argument("--grant-id", action="append", default=[], help="Target grant opportunity_id (repeatable).")
    parser.add_argument("--all-faculty", action="store_true", help="Process all faculty emails in Neo4j.")
    parser.add_argument("--all-grants", action="store_true", help="Process all grants in Neo4j.")
    parser.add_argument("--include-closed", action="store_true", help="Include closed grants when listing all grants.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit when using --all-* (0 = all).")
    parser.add_argument("--offset", type=int, default=0, help="Optional offset when using --all-*.")
    parser.add_argument("--min-domain-sim", type=float, default=0.35, help="Domain prefilter threshold.")
    parser.add_argument("--domain-top-n", type=int, default=100, help="Max candidate grants kept per faculty after domain prefilter (0 = keep all).")
    parser.add_argument("--min-chunk-sim", type=float, default=0.55, help="Minimum similarity for specialization->opposite-evidence links.")
    parser.add_argument("--top-k-fac-to-grant-chunk", type=int, default=5, help="Top grant chunks kept per faculty specialization keyword.")
    parser.add_argument("--top-k-grant-to-fac-evidence", type=int, default=5, help="Top faculty evidence items kept per grant specialization keyword.")
    parser.add_argument(
        "--min-keyword-confidence",
        type=float,
        default=0.5,
        help="Fallback confidence floor if keyword evidence confidence is missing.",
    )
    parser.add_argument("--skip-write", action="store_true", help="Compute only; do not write edges.")
    parser.add_argument("--json-only", action="store_true", help="Print only JSON output.")

    parser.add_argument("--uri", type=str, default="", help="Neo4j URI. Fallback: NEO4J_URI")
    parser.add_argument("--username", type=str, default="", help="Neo4j username. Fallback: NEO4J_USERNAME")
    parser.add_argument("--password", type=str, default="", help="Neo4j password. Fallback: NEO4J_PASSWORD")
    parser.add_argument("--database", type=str, default="", help="Neo4j database. Fallback: NEO4J_DATABASE or neo4j")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if not bool(args.all_faculty) and not list(args.faculty_email or []):
        raise SystemExit("Provide --faculty-email (repeatable) or use --all-faculty.")

    payload = run_domain_prefilter_spec_chunk_linker(
        faculty_emails=list(args.faculty_email or []),
        grant_ids=list(args.grant_id or []),
        all_faculty=bool(args.all_faculty),
        all_grants=bool(args.all_grants),
        include_closed=bool(args.include_closed),
        limit=int(args.limit or 0),
        offset=int(args.offset or 0),
        min_domain_sim=float(args.min_domain_sim),
        domain_top_n=int(args.domain_top_n or 100),
        min_chunk_sim=float(args.min_chunk_sim),
        top_k_fac_to_grant_chunk=int(args.top_k_fac_to_grant_chunk or 5),
        top_k_grant_to_fac_evidence=int(args.top_k_grant_to_fac_evidence or 5),
        min_keyword_confidence=float(args.min_keyword_confidence),
        write_edges=not bool(args.skip_write),
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    if not args.json_only:
        totals = payload.get("totals", {})
        print("Domain-prefilter specialization chunk linker complete.")
        print(f"  faculties processed            : {totals.get('faculties_processed', 0)}")
        print(f"  fac->grant_chunk edges         : {totals.get('faculty_to_grant_chunk_edges', 0)}")
        print(f"  grant->fac_evidence edges      : {totals.get('grant_to_fac_evidence_edges', 0)}")
        print(f"  total edges                    : {totals.get('all_edges', 0)}")
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

