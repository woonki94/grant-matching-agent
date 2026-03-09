from __future__ import annotations

import argparse
import json
import math
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
    "HAS_SPECIALIZATION_KEYWORD",
    "HAS_RESEARCH_SPECIALIZATION",
    "HAS_APPLICATION_SPECIALIZATION",
]
GRANT_SPEC_RELATIONS = [
    "HAS_SPECIALIZATION_KEYWORD",
    "HAS_RESEARCH_SPECIALIZATION",
    "HAS_APPLICATION_SPECIALIZATION",
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


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_unit_float(value: Any, *, default: float = 0.0) -> float:
    parsed = _safe_float(value, default=default)
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


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
    if sim < 0.0:
        return 0.0
    if sim > 1.0:
        return 1.0
    return sim


def _truncate_text(value: Any, max_chars: int) -> str:
    text = _clean_text(value)
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class _CrossEncoderScorer:
    def __init__(self, *, model_name: str, batch_size: int):
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            raise RuntimeError(
                "V3 hybrid linker requires sentence-transformers. "
                "Install with: pip install sentence-transformers torch"
            ) from exc

        self.batch_size = max(1, int(batch_size or 16))
        self.model_name = str(model_name or "").strip() or "BAAI/bge-reranker-v2-m3"
        self.model = CrossEncoder(self.model_name)

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        raw = self.model.predict(
            list(pairs),
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        vals = np.asarray(raw, dtype=np.float32).reshape(-1)
        out: List[float] = []
        for item in vals:
            out.append(_safe_unit_float(_sigmoid(float(item)), default=0.0))
        return out


def _fetch_faculty_identity_and_domain_gate(
    *,
    driver,
    settings: Neo4jSettings,
    faculty_email: str,
) -> Optional[Dict[str, Any]]:
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})-[:HAS_DOMAIN_GATE]->(dg:FacultyDomainGate)
        WHERE dg.embedding IS NOT NULL
        RETURN
            f.faculty_id AS faculty_id,
            f.email AS email,
            dg.embedding AS domain_embedding,
            dg.domain_terms AS domain_terms
        ORDER BY coalesce(dg.updated_at, datetime({epochMillis: 0})) DESC
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
    vec = _coerce_vector(row.get("domain_embedding"))
    if vec.size == 0:
        return None
    return {
        "faculty_id": faculty_id,
        "email": _clean_text(row.get("email")).lower(),
        "domain_embedding": vec,
        "domain_terms": list(row.get("domain_terms") or []),
    }


def _fetch_grants_with_domain_gate(
    *,
    driver,
    settings: Neo4jSettings,
    include_closed: bool,
) -> List[Dict[str, Any]]:
    records, _, _ = driver.execute_query(
        """
        MATCH (g:Grant)-[:HAS_DOMAIN_GATE]->(dg:GrantDomainGate)
        WHERE g.opportunity_id IS NOT NULL
          AND dg.embedding IS NOT NULL
        WITH
            g,
            dg,
            toLower(coalesce(g.opportunity_status, "")) AS status_token,
            coalesce(toString(g.close_date), "") AS close_token
        WITH
            g,
            dg,
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
        RETURN DISTINCT
            g.opportunity_id AS opportunity_id,
            coalesce(g.opportunity_title, g.title, '') AS title,
            dg.embedding AS domain_embedding
        ORDER BY opportunity_id ASC
        """,
        parameters_={"include_closed": bool(include_closed)},
        database_=settings.database,
    )
    out: List[Dict[str, Any]] = []
    for row in records:
        oid = _clean_text(row.get("opportunity_id"))
        if not oid:
            continue
        vec = _coerce_vector(row.get("domain_embedding"))
        if vec.size == 0:
            continue
        out.append(
            {
                "opportunity_id": oid,
                "title": _clean_text(row.get("title")),
                "domain_embedding": vec,
            }
        )
    return out


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
            OPTIONAL MATCH (:FacultyTextChunk)-[s:FACULTY_CHUNK_SUPPORTS_SPECIALIZATION]->(k)
            WHERE s.scope_faculty_id = f.faculty_id
            RETURN coalesce(max(s.score), 0.0) AS chunk_conf
        }
        CALL (f, k) {
            OPTIONAL MATCH (:FacultyPublication)-[s:FACULTY_PUBLICATION_SUPPORTS_SPECIALIZATION]->(k)
            WHERE s.scope_faculty_id = f.faculty_id
            RETURN coalesce(max(s.score), 0.0) AS pub_conf
        }
        RETURN DISTINCT
            elementId(k) AS keyword_element_id,
            k.value AS keyword_value,
            coalesce(k.section, 'general') AS keyword_section,
            k.embedding AS embedding,
            coalesce(r.weight, 0.5) AS keyword_weight,
            CASE WHEN chunk_conf >= pub_conf THEN chunk_conf ELSE pub_conf END AS keyword_confidence
        ORDER BY keyword_value ASC
        """,
        parameters_={
            "email": _clean_text(faculty_email).lower(),
            "relations": FACULTY_SPEC_RELATIONS,
        },
        database_=settings.database,
    )
    return [dict(row or {}) for row in records]


def _fetch_grant_spec_keyword_rows_for_ids(
    *,
    driver,
    settings: Neo4jSettings,
    opportunity_ids: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    ids = [str(x or "").strip() for x in list(opportunity_ids or []) if str(x or "").strip()]
    if not ids:
        return {}

    records, _, _ = driver.execute_query(
        """
        UNWIND $opportunity_ids AS oid
        MATCH (g:Grant {opportunity_id: oid})-[r]->(k:GrantKeyword)
        WHERE type(r) IN $relations
          AND k.bucket = 'specialization'
          AND k.embedding IS NOT NULL
          AND k.value IS NOT NULL
        CALL (oid, k) {
            OPTIONAL MATCH (:GrantTextChunk)-[s:GRANT_CHUNK_SUPPORTS_SPECIALIZATION]->(k)
            WHERE s.scope_opportunity_id = oid
            RETURN coalesce(max(s.score), 0.0) AS chunk_conf
        }
        RETURN DISTINCT
            oid AS opportunity_id,
            coalesce(g.opportunity_title, g.title, '') AS opportunity_title,
            elementId(k) AS keyword_element_id,
            k.value AS keyword_value,
            coalesce(k.section, 'general') AS keyword_section,
            k.embedding AS embedding,
            coalesce(r.weight, 0.5) AS keyword_weight,
            chunk_conf AS keyword_confidence
        ORDER BY opportunity_id ASC, keyword_value ASC
        """,
        parameters_={
            "opportunity_ids": ids,
            "relations": GRANT_SPEC_RELATIONS,
        },
        database_=settings.database,
    )

    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in records:
        item = dict(row or {})
        oid = _clean_text(item.get("opportunity_id"))
        if not oid:
            continue
        out.setdefault(oid, []).append(item)
    return out


def _prepare_keyword_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    min_keyword_confidence: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    safe_min_conf = _safe_unit_float(min_keyword_confidence, default=0.0)
    for row in list(rows or []):
        keyword_element_id = _clean_text(row.get("keyword_element_id"))
        keyword_value = _clean_text(row.get("keyword_value"))
        keyword_section = _clean_text(row.get("keyword_section")).lower() or "general"
        vector = _coerce_vector(row.get("embedding"))
        if not keyword_element_id or not keyword_value or vector.size == 0:
            continue
        out.append(
            {
                "keyword_element_id": keyword_element_id,
                "keyword_value": keyword_value,
                "keyword_section": keyword_section,
                "vector": vector,
                "keyword_weight": _safe_unit_float(row.get("keyword_weight"), default=0.5),
                "keyword_confidence": max(
                    safe_min_conf,
                    _safe_unit_float(row.get("keyword_confidence"), default=safe_min_conf),
                ),
            }
        )
    return out


def _score_spec_keyword_pairs(
    *,
    faculty_keywords: Sequence[Dict[str, Any]],
    grant_keywords: Sequence[Dict[str, Any]],
    scorer: _CrossEncoderScorer,
    min_cosine_sim: float,
    min_attention_score: float,
    max_pair_text_chars: int,
    max_pairs_per_grant: int,
    domain_gate_score: float,
    opportunity_id: str,
    opportunity_title: str,
) -> Dict[str, Any]:
    safe_min_cos = _safe_unit_float(min_cosine_sim, default=0.0)
    safe_min_att = _safe_unit_float(min_attention_score, default=0.0)
    safe_max_chars = _safe_limit(max_pair_text_chars, default=300, minimum=50, maximum=8000)
    safe_max_pairs = _safe_limit(max_pairs_per_grant, default=0, minimum=0, maximum=500000)
    safe_domain_gate = _safe_unit_float(domain_gate_score, default=0.0)

    stage_rows: List[Dict[str, Any]] = []
    pair_inputs: List[Tuple[str, str]] = []

    for fk in list(faculty_keywords or []):
        f_vec = _coerce_vector(fk.get("vector"))
        if f_vec.size == 0:
            continue
        for gk in list(grant_keywords or []):
            g_vec = _coerce_vector(gk.get("vector"))
            if g_vec.size == 0:
                continue
            if int(f_vec.shape[0]) != int(g_vec.shape[0]):
                continue

            cosine_sim = _cosine_vec(f_vec, g_vec)
            if cosine_sim < safe_min_cos:
                continue

            f_text = _truncate_text(fk.get("keyword_value"), safe_max_chars)
            g_text = _truncate_text(gk.get("keyword_value"), safe_max_chars)
            if not f_text or not g_text:
                continue

            stage_rows.append(
                {
                    "opportunity_id": _clean_text(opportunity_id),
                    "opportunity_title": _clean_text(opportunity_title),
                    "faculty_keyword_element_id": _clean_text(fk.get("keyword_element_id")),
                    "faculty_keyword_value": _clean_text(fk.get("keyword_value")),
                    "faculty_keyword_section": _clean_text(fk.get("keyword_section")).lower() or "general",
                    "faculty_keyword_weight": _safe_unit_float(fk.get("keyword_weight"), default=0.5),
                    "faculty_keyword_confidence": _safe_unit_float(fk.get("keyword_confidence"), default=0.5),
                    "grant_keyword_element_id": _clean_text(gk.get("keyword_element_id")),
                    "grant_keyword_value": _clean_text(gk.get("keyword_value")),
                    "grant_keyword_section": _clean_text(gk.get("keyword_section")).lower() or "general",
                    "grant_keyword_weight": _safe_unit_float(gk.get("keyword_weight"), default=0.5),
                    "grant_keyword_confidence": _safe_unit_float(gk.get("keyword_confidence"), default=0.5),
                    "domain_gate_score": safe_domain_gate,
                    "cosine_sim": cosine_sim,
                }
            )
            pair_inputs.append((f_text, g_text))

    if not stage_rows:
        return {"rows": [], "pair_count": 0, "scored_count": 0, "kept_count": 0}

    if safe_max_pairs > 0 and len(stage_rows) > safe_max_pairs:
        ordering = sorted(
            range(len(stage_rows)),
            key=lambda i: float(stage_rows[i].get("cosine_sim") or 0.0),
            reverse=True,
        )[:safe_max_pairs]
        stage_rows = [stage_rows[i] for i in ordering]
        pair_inputs = [pair_inputs[i] for i in ordering]

    attention_scores = scorer.score_pairs(pair_inputs)
    if not attention_scores:
        return {"rows": [], "pair_count": len(stage_rows), "scored_count": 0, "kept_count": 0}

    final_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(stage_rows):
        attention_score = _safe_unit_float(attention_scores[i], default=0.0)
        if attention_score < safe_min_att:
            continue
        cosine_sim = _safe_unit_float(row.get("cosine_sim"), default=0.0)
        hybrid_sim = _safe_unit_float((cosine_sim + attention_score) / 2.0, default=0.0)

        score = hybrid_sim
        score *= _safe_unit_float(row.get("faculty_keyword_weight"), default=0.5)
        score *= _safe_unit_float(row.get("grant_keyword_weight"), default=0.5)
        score *= _safe_unit_float(row.get("faculty_keyword_confidence"), default=0.5)
        score *= _safe_unit_float(row.get("grant_keyword_confidence"), default=0.5)
        score *= _safe_unit_float(row.get("domain_gate_score"), default=0.0)

        row["attention_score"] = attention_score
        row["hybrid_sim"] = hybrid_sim
        row["score"] = _safe_unit_float(score, default=0.0)
        final_rows.append(row)

    return {
        "rows": final_rows,
        "pair_count": len(stage_rows),
        "scored_count": len(attention_scores),
        "kept_count": len(final_rows),
    }


def _delete_existing_v3_edges_for_faculty(
    *,
    driver,
    settings: Neo4jSettings,
    faculty_id: int,
) -> None:
    driver.execute_query(
        """
        MATCH (:FacultyKeyword)-[r:FACULTY_SPEC_MATCHES_GRANT_SPEC_V3 {scope_faculty_id: $faculty_id}]->(:GrantKeyword)
        DELETE r
        """,
        parameters_={"faculty_id": int(faculty_id)},
        database_=settings.database,
    )


def _write_v3_edges(
    *,
    driver,
    settings: Neo4jSettings,
    faculty_id: int,
    faculty_email: str,
    rows: Sequence[Dict[str, Any]],
    min_domain_sim: float,
    min_cosine_sim: float,
    min_attention_score: float,
    batch_size: int,
) -> int:
    all_rows = list(rows or [])
    if not all_rows:
        return 0

    safe_batch_size = _safe_limit(batch_size, default=2000, minimum=100, maximum=20000)
    linked = 0

    for i in range(0, len(all_rows), safe_batch_size):
        batch = all_rows[i : i + safe_batch_size]
        records, _, _ = driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (fk) WHERE elementId(fk) = row.faculty_keyword_element_id
            MATCH (gk) WHERE elementId(gk) = row.grant_keyword_element_id
            MERGE (fk)-[r:FACULTY_SPEC_MATCHES_GRANT_SPEC_V3 {
                scope_faculty_id: $faculty_id,
                scope_faculty_email: $faculty_email,
                scope_opportunity_id: row.opportunity_id,
                faculty_keyword_element_id: row.faculty_keyword_element_id,
                grant_keyword_element_id: row.grant_keyword_element_id
            }]->(gk)
            SET
                r.faculty_keyword_value = row.faculty_keyword_value,
                r.faculty_keyword_section = row.faculty_keyword_section,
                r.grant_keyword_value = row.grant_keyword_value,
                r.grant_keyword_section = row.grant_keyword_section,
                r.opportunity_title = row.opportunity_title,
                r.cosine_sim = row.cosine_sim,
                r.attention_score = row.attention_score,
                r.hybrid_sim = row.hybrid_sim,
                r.score = row.score,
                r.domain_gate_score = row.domain_gate_score,
                r.faculty_keyword_weight = row.faculty_keyword_weight,
                r.grant_keyword_weight = row.grant_keyword_weight,
                r.faculty_keyword_confidence = row.faculty_keyword_confidence,
                r.grant_keyword_confidence = row.grant_keyword_confidence,
                r.min_domain_sim = $min_domain_sim,
                r.min_cosine_sim = $min_cosine_sim,
                r.min_attention_score = $min_attention_score,
                r.method = 'domain_gate_spec_keyword_hybrid_v3',
                r.updated_at = datetime()
            RETURN count(r) AS linked_count
            """,
            parameters_={
                "faculty_id": int(faculty_id),
                "faculty_email": _clean_text(faculty_email).lower(),
                "rows": batch,
                "min_domain_sim": _safe_unit_float(min_domain_sim, default=0.3),
                "min_cosine_sim": _safe_unit_float(min_cosine_sim, default=0.0),
                "min_attention_score": _safe_unit_float(min_attention_score, default=0.0),
            },
            database_=settings.database,
        )
        if records:
            linked += int(records[0].get("linked_count") or 0)

    return linked


def run_domain_gate_spec_keyword_hybrid_linker_v3(
    *,
    faculty_email: str,
    min_domain_sim: float,
    domain_top_n: int,
    include_closed: bool,
    min_cosine_sim: float,
    min_attention_score: float,
    min_keyword_confidence: float,
    max_pair_text_chars: int,
    max_pairs_per_grant: int,
    cross_encoder_model: str,
    cross_encoder_batch_size: int,
    write_edges: bool,
    write_batch_size: int,
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

    safe_min_domain = _safe_unit_float(min_domain_sim, default=0.3)
    safe_top_n = _safe_limit(domain_top_n, default=0, minimum=0, maximum=100000)

    scorer = _CrossEncoderScorer(
        model_name=_clean_text(cross_encoder_model) or "BAAI/bge-reranker-v2-m3",
        batch_size=_safe_limit(cross_encoder_batch_size, default=16, minimum=1, maximum=512),
    )

    with GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    ) as driver:
        driver.verify_connectivity()

        faculty = _fetch_faculty_identity_and_domain_gate(
            driver=driver,
            settings=settings,
            faculty_email=faculty_email,
        )
        if faculty is None:
            raise RuntimeError(
                "Faculty domain gate embedding not found. "
                "Run faculty keyword/domain sync first for this faculty."
            )

        all_grants = _fetch_grants_with_domain_gate(
            driver=driver,
            settings=settings,
            include_closed=bool(include_closed),
        )
        candidates: List[Dict[str, Any]] = []
        f_domain = _coerce_vector(faculty.get("domain_embedding"))
        for grant in all_grants:
            g_domain = _coerce_vector(grant.get("domain_embedding"))
            domain_sim = _cosine_vec(f_domain, g_domain)
            if domain_sim < safe_min_domain:
                continue
            candidates.append(
                {
                    "opportunity_id": _clean_text(grant.get("opportunity_id")),
                    "title": _clean_text(grant.get("title")),
                    "domain_gate_score": _safe_unit_float(domain_sim, default=0.0),
                }
            )

        candidates.sort(key=lambda x: float(x.get("domain_gate_score") or 0.0), reverse=True)
        if safe_top_n > 0:
            candidates = candidates[:safe_top_n]

        faculty_specs = _prepare_keyword_rows(
            _fetch_faculty_spec_keyword_rows(
                driver=driver,
                settings=settings,
                faculty_email=str(faculty.get("email") or ""),
            ),
            min_keyword_confidence=min_keyword_confidence,
        )

        candidate_ids = [str(x.get("opportunity_id") or "") for x in candidates if str(x.get("opportunity_id") or "")]
        grant_specs_by_opp = _fetch_grant_spec_keyword_rows_for_ids(
            driver=driver,
            settings=settings,
            opportunity_ids=candidate_ids,
        )

        all_link_rows: List[Dict[str, Any]] = []
        per_grant: List[Dict[str, Any]] = []
        total_pair_count = 0
        total_scored_count = 0

        for candidate in candidates:
            oid = str(candidate.get("opportunity_id") or "").strip()
            title = str(candidate.get("title") or "").strip()
            domain_score = _safe_unit_float(candidate.get("domain_gate_score"), default=0.0)
            raw_grant_specs = grant_specs_by_opp.get(oid, [])
            grant_specs = _prepare_keyword_rows(
                raw_grant_specs,
                min_keyword_confidence=min_keyword_confidence,
            )

            scored = _score_spec_keyword_pairs(
                faculty_keywords=faculty_specs,
                grant_keywords=grant_specs,
                scorer=scorer,
                min_cosine_sim=min_cosine_sim,
                min_attention_score=min_attention_score,
                max_pair_text_chars=max_pair_text_chars,
                max_pairs_per_grant=max_pairs_per_grant,
                domain_gate_score=domain_score,
                opportunity_id=oid,
                opportunity_title=title,
            )
            link_rows = list(scored.get("rows") or [])
            all_link_rows.extend(link_rows)
            total_pair_count += int(scored.get("pair_count") or 0)
            total_scored_count += int(scored.get("scored_count") or 0)

            per_grant.append(
                {
                    "opportunity_id": oid,
                    "opportunity_title": title,
                    "domain_gate_score": domain_score,
                    "grant_specialization_keywords": len(grant_specs),
                    "pair_count": int(scored.get("pair_count") or 0),
                    "scored_count": int(scored.get("scored_count") or 0),
                    "kept_edges": int(scored.get("kept_count") or 0),
                }
            )

        linked_edges = 0
        if write_edges:
            _delete_existing_v3_edges_for_faculty(
                driver=driver,
                settings=settings,
                faculty_id=int(faculty["faculty_id"]),
            )
            linked_edges = _write_v3_edges(
                driver=driver,
                settings=settings,
                faculty_id=int(faculty["faculty_id"]),
                faculty_email=str(faculty.get("email") or ""),
                rows=all_link_rows,
                min_domain_sim=safe_min_domain,
                min_cosine_sim=min_cosine_sim,
                min_attention_score=min_attention_score,
                batch_size=write_batch_size,
            )
        else:
            linked_edges = len(all_link_rows)

    payload = {
        "scope": {
            "faculty_email": str(faculty.get("email") or "").lower(),
            "faculty_id": int(faculty["faculty_id"]),
            "include_closed": bool(include_closed),
            "min_domain_sim": safe_min_domain,
            "domain_top_n": safe_top_n,
            "min_cosine_sim": _safe_unit_float(min_cosine_sim, default=0.0),
            "min_attention_score": _safe_unit_float(min_attention_score, default=0.0),
            "min_keyword_confidence": _safe_unit_float(min_keyword_confidence, default=0.0),
            "max_pair_text_chars": _safe_limit(max_pair_text_chars, default=300, minimum=50, maximum=8000),
            "max_pairs_per_grant": _safe_limit(max_pairs_per_grant, default=0, minimum=0, maximum=500000),
            "cross_encoder_model": str(cross_encoder_model or "").strip() or "BAAI/bge-reranker-v2-m3",
            "cross_encoder_batch_size": _safe_limit(cross_encoder_batch_size, default=16, minimum=1, maximum=512),
            "write_edges": bool(write_edges),
            "write_batch_size": _safe_limit(write_batch_size, default=2000, minimum=100, maximum=20000),
            "edge_label": "FACULTY_SPEC_MATCHES_GRANT_SPEC_V3",
        },
        "summary": {
            "grants_considered_with_domain_gate": len(all_grants),
            "grants_passing_domain_gate": len(candidates),
            "faculty_specialization_keywords": len(faculty_specs),
            "total_keyword_pairs_after_cosine_filter": int(total_pair_count),
            "total_keyword_pairs_scored_by_cross_encoder": int(total_scored_count),
            "edges_prepared": len(all_link_rows),
            "edges_linked": int(linked_edges),
        },
        "per_grant": per_grant,
    }
    return json_ready(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "V3 matcher: prefilter grants by faculty-vs-grant domain gate cosine similarity, "
            "then link faculty specialization keywords to grant specialization keywords "
            "with both cosine and cross-encoder scores."
        )
    )
    parser.add_argument("--faculty-email", type=str, required=True, help="Faculty email for matching.")
    parser.add_argument("--min-domain-sim", type=float, default=0.3, help="Domain gate cosine threshold.")
    parser.add_argument(
        "--domain-top-n",
        type=int,
        default=0,
        help="Max grants to keep after domain gate prefilter (0 = keep all).",
    )
    parser.add_argument("--include-closed", action="store_true", help="Include closed grants.")
    parser.add_argument("--min-cosine-sim", type=float, default=0.0, help="Optional pair-level cosine filter.")
    parser.add_argument(
        "--min-attention-score",
        type=float,
        default=0.0,
        help="Optional pair-level cross-encoder filter.",
    )
    parser.add_argument(
        "--min-keyword-confidence",
        type=float,
        default=0.0,
        help="Floor for keyword confidence values from support edges.",
    )
    parser.add_argument(
        "--max-pair-text-chars",
        type=int,
        default=300,
        help="Chars per side passed to cross-encoder.",
    )
    parser.add_argument(
        "--max-pairs-per-grant",
        type=int,
        default=0,
        help="Cap pair count per grant before cross-encoder (0 = no cap).",
    )
    parser.add_argument(
        "--cross-encoder-model",
        type=str,
        default="BAAI/bge-reranker-v2-m3",
        help="Cross-encoder model.",
    )
    parser.add_argument("--cross-encoder-batch-size", type=int, default=16, help="Cross-encoder batch size.")
    parser.add_argument("--write-batch-size", type=int, default=2000, help="Edge write batch size.")
    parser.add_argument("--skip-write", action="store_true", help="Compute only; do not write edges.")
    parser.add_argument("--json-only", action="store_true", help="Print JSON only.")
    parser.add_argument("--uri", type=str, default="", help="Neo4j URI. Fallback: NEO4J_URI")
    parser.add_argument("--username", type=str, default="", help="Neo4j username. Fallback: NEO4J_USERNAME")
    parser.add_argument("--password", type=str, default="", help="Neo4j password. Fallback: NEO4J_PASSWORD")
    parser.add_argument("--database", type=str, default="", help="Neo4j database. Fallback: NEO4J_DATABASE or neo4j")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    payload = run_domain_gate_spec_keyword_hybrid_linker_v3(
        faculty_email=str(args.faculty_email or "").strip().lower(),
        min_domain_sim=float(args.min_domain_sim),
        domain_top_n=int(args.domain_top_n),
        include_closed=bool(args.include_closed),
        min_cosine_sim=float(args.min_cosine_sim),
        min_attention_score=float(args.min_attention_score),
        min_keyword_confidence=float(args.min_keyword_confidence),
        max_pair_text_chars=int(args.max_pair_text_chars),
        max_pairs_per_grant=int(args.max_pairs_per_grant),
        cross_encoder_model=str(args.cross_encoder_model or "").strip(),
        cross_encoder_batch_size=int(args.cross_encoder_batch_size),
        write_edges=not bool(args.skip_write),
        write_batch_size=int(args.write_batch_size),
        uri=str(args.uri or "").strip(),
        username=str(args.username or "").strip(),
        password=str(args.password or "").strip(),
        database=str(args.database or "").strip(),
    )

    if not bool(args.json_only):
        print("V3 domain-gate spec-keyword hybrid linker complete.")
        summary = payload.get("summary") or {}
        print(f"  grants considered (domain gate) : {int(summary.get('grants_considered_with_domain_gate') or 0)}")
        print(f"  grants passing gate             : {int(summary.get('grants_passing_domain_gate') or 0)}")
        print(f"  faculty spec keywords           : {int(summary.get('faculty_specialization_keywords') or 0)}")
        print(f"  keyword pairs scored            : {int(summary.get('total_keyword_pairs_scored_by_cross_encoder') or 0)}")
        print(f"  edges linked                    : {int(summary.get('edges_linked') or 0)}")
        print()

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
