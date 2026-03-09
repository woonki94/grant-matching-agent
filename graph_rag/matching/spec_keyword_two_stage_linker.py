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
    "HAS_RESEARCH_SPECIALIZATION",
    "HAS_APPLICATION_SPECIALIZATION",
]
GRANT_SPEC_RELATIONS = [
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


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


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
                "Two-stage spec linker requires sentence-transformers. "
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
        RETURN DISTINCT
            k.value AS keyword_value,
            k.section AS keyword_section,
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
        RETURN DISTINCT
            k.value AS keyword_value,
            k.section AS keyword_section,
            k.embedding AS embedding,
            coalesce(r.weight, 0.5) AS keyword_weight,
            chunk_conf AS keyword_confidence
        ORDER BY keyword_value ASC
        """,
        parameters_={
            "opportunity_id": _clean_text(opportunity_id),
            "relations": GRANT_SPEC_RELATIONS,
        },
        database_=settings.database,
    )
    return [dict(row or {}) for row in records]


def _rank_spec_keyword_matches_two_stage(
    *,
    faculty_keyword_rows: List[Dict[str, Any]],
    grant_keyword_rows: List[Dict[str, Any]],
    scorer: _CrossEncoderScorer,
    min_cosine_sim: float,
    min_attention_score: float,
    top_k_cosine: int,
    top_k_final_per_faculty_keyword: int,
    min_keyword_confidence: float,
    same_section_only: bool,
    attention_alpha: float,
    max_pair_text_chars: int,
) -> Dict[str, Any]:
    links: List[Dict[str, Any]] = []
    safe_min_cosine = _safe_unit_float(min_cosine_sim, default=0.45)
    safe_min_attention = _safe_unit_float(min_attention_score, default=0.55)
    safe_top_k_cosine = _safe_limit(top_k_cosine, default=20, minimum=1, maximum=1000)
    safe_top_k_final = _safe_limit(top_k_final_per_faculty_keyword, default=10, minimum=1, maximum=1000)
    safe_min_kw_conf = _safe_unit_float(min_keyword_confidence, default=0.5)
    safe_alpha = _safe_unit_float(attention_alpha, default=0.5)
    safe_max_chars = _safe_limit(max_pair_text_chars, default=300, minimum=50, maximum=8000)

    prepared_grant: List[Dict[str, Any]] = []
    for row in grant_keyword_rows or []:
        keyword_value = _clean_text(row.get("keyword_value"))
        keyword_section = _clean_text(row.get("keyword_section")).lower()
        vec = _coerce_vector(row.get("embedding"))
        if not keyword_value or vec.size == 0:
            continue
        prepared_grant.append(
            {
                "keyword_value": keyword_value,
                "keyword_section": keyword_section,
                "embedding": vec,
                "keyword_weight": _safe_unit_float(row.get("keyword_weight"), default=0.5),
                "keyword_confidence": max(
                    safe_min_kw_conf,
                    _safe_unit_float(row.get("keyword_confidence"), default=safe_min_kw_conf),
                ),
            }
        )

    if not prepared_grant:
        return {
            "rows": [],
            "stats": {
                "cosine_candidates": 0,
                "attention_candidates": 0,
                "links_kept": 0,
            },
        }

    cosine_candidates = 0
    attention_candidates = 0

    for fac_row in faculty_keyword_rows or []:
        fac_keyword_value = _clean_text(fac_row.get("keyword_value"))
        fac_keyword_section = _clean_text(fac_row.get("keyword_section")).lower()
        fac_vec = _coerce_vector(fac_row.get("embedding"))
        if not fac_keyword_value or fac_vec.size == 0:
            continue

        fac_weight = _safe_unit_float(fac_row.get("keyword_weight"), default=0.5)
        fac_conf = max(
            safe_min_kw_conf,
            _safe_unit_float(fac_row.get("keyword_confidence"), default=safe_min_kw_conf),
        )

        compatible: List[Dict[str, Any]] = []
        for g in prepared_grant:
            if same_section_only and fac_keyword_section and g["keyword_section"] and fac_keyword_section != g["keyword_section"]:
                continue
            g_vec = _coerce_vector(g.get("embedding"))
            if int(g_vec.shape[0]) != int(fac_vec.shape[0]):
                continue
            compatible.append(g)

        if not compatible:
            continue

        grant_matrix = np.asarray([_coerce_vector(x.get("embedding")) for x in compatible], dtype=np.float32)
        sims = _cosine_matrix(fac_vec.reshape(1, -1), grant_matrix)[0]
        order = np.argsort(-sims)

        stage1_rows: List[Dict[str, Any]] = []
        pair_inputs: List[Tuple[str, str]] = []
        for idx in order:
            cosine_sim = float(sims[int(idx)])
            if cosine_sim < safe_min_cosine:
                continue

            grant_row = compatible[int(idx)]
            grant_weight = _safe_unit_float(grant_row.get("keyword_weight"), default=0.5)
            grant_conf = _safe_unit_float(grant_row.get("keyword_confidence"), default=safe_min_kw_conf)
            stage1_rows.append(
                {
                    "faculty_keyword_value": fac_keyword_value,
                    "faculty_keyword_section": fac_keyword_section,
                    "grant_keyword_value": _clean_text(grant_row.get("keyword_value")),
                    "grant_keyword_section": _clean_text(grant_row.get("keyword_section")).lower(),
                    "faculty_keyword_weight": fac_weight,
                    "grant_keyword_weight": grant_weight,
                    "faculty_keyword_confidence": fac_conf,
                    "grant_keyword_confidence": grant_conf,
                    "cosine_sim": _safe_unit_float(cosine_sim, default=0.0),
                }
            )
            pair_inputs.append(
                (
                    _truncate_text(fac_keyword_value, safe_max_chars),
                    _truncate_text(grant_row.get("keyword_value"), safe_max_chars),
                )
            )
            if len(stage1_rows) >= safe_top_k_cosine:
                break

        cosine_candidates += len(stage1_rows)
        if not stage1_rows:
            continue

        attention_scores = scorer.score_pairs(pair_inputs)
        if not attention_scores:
            continue

        stage2_rows: List[Dict[str, Any]] = []
        for i, row in enumerate(stage1_rows):
            attention_sim = _safe_unit_float(attention_scores[i], default=0.0)
            if attention_sim < safe_min_attention:
                continue

            attention_candidates += 1
            hybrid_sim = ((1.0 - safe_alpha) * float(row["cosine_sim"])) + (safe_alpha * attention_sim)
            score = hybrid_sim
            score *= _safe_unit_float(row.get("faculty_keyword_weight"), default=0.0)
            score *= _safe_unit_float(row.get("grant_keyword_weight"), default=0.0)
            score *= _safe_unit_float(row.get("faculty_keyword_confidence"), default=0.0)
            score *= _safe_unit_float(row.get("grant_keyword_confidence"), default=0.0)
            row["attention_score"] = attention_sim
            row["hybrid_sim"] = _safe_unit_float(hybrid_sim, default=0.0)
            row["score"] = _safe_unit_float(score, default=0.0)
            stage2_rows.append(row)

        if not stage2_rows:
            continue

        stage2_rows.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        links.extend(stage2_rows[:safe_top_k_final])

    return {
        "rows": links,
        "stats": {
            "cosine_candidates": int(cosine_candidates),
            "attention_candidates": int(attention_candidates),
            "links_kept": int(len(links)),
        },
    }


def _write_keyword_match_edges(
    *,
    driver,
    settings: Neo4jSettings,
    faculty_id: int,
    faculty_email: str,
    opportunity_id: str,
    rows: List[Dict[str, Any]],
    min_cosine_sim: float,
    min_attention_score: float,
    attention_alpha: float,
) -> None:
    driver.execute_query(
        """
        MATCH (:FacultyKeyword)-[r:FACULTY_SPEC_MATCHES_GRANT_SPEC {
            scope_faculty_id: $faculty_id,
            scope_opportunity_id: $opportunity_id
        }]->(:GrantKeyword)
        DELETE r
        """,
        parameters_={
            "faculty_id": int(faculty_id),
            "opportunity_id": _clean_text(opportunity_id),
        },
        database_=settings.database,
    )

    if not rows:
        return

    driver.execute_query(
        """
        UNWIND $rows AS row
        MATCH (fk:FacultyKeyword {
            value: row.faculty_keyword_value,
            section: row.faculty_keyword_section,
            bucket: 'specialization'
        })
        MATCH (gk:GrantKeyword {
            value: row.grant_keyword_value,
            section: row.grant_keyword_section,
            bucket: 'specialization'
        })
        MERGE (fk)-[r:FACULTY_SPEC_MATCHES_GRANT_SPEC {
            scope_faculty_id: $faculty_id,
            scope_faculty_email: $faculty_email,
            scope_opportunity_id: $opportunity_id,
            faculty_keyword_value: row.faculty_keyword_value,
            faculty_keyword_section: row.faculty_keyword_section,
            grant_keyword_value: row.grant_keyword_value,
            grant_keyword_section: row.grant_keyword_section
        }]->(gk)
        SET
            r.score = row.score,
            r.cosine_sim = row.cosine_sim,
            r.attention_score = row.attention_score,
            r.hybrid_sim = row.hybrid_sim,
            r.faculty_keyword_weight = row.faculty_keyword_weight,
            r.grant_keyword_weight = row.grant_keyword_weight,
            r.faculty_keyword_confidence = row.faculty_keyword_confidence,
            r.grant_keyword_confidence = row.grant_keyword_confidence,
            r.min_cosine_sim = $min_cosine_sim,
            r.min_attention_score = $min_attention_score,
            r.attention_alpha = $attention_alpha,
            r.method = 'specialization_keyword_two_stage_hybrid',
            r.updated_at = datetime()
        """,
        parameters_={
            "faculty_id": int(faculty_id),
            "faculty_email": _clean_text(faculty_email).lower(),
            "opportunity_id": _clean_text(opportunity_id),
            "rows": rows,
            "min_cosine_sim": _safe_unit_float(min_cosine_sim, default=0.45),
            "min_attention_score": _safe_unit_float(min_attention_score, default=0.55),
            "attention_alpha": _safe_unit_float(attention_alpha, default=0.5),
        },
        database_=settings.database,
    )


def run_specialization_keyword_two_stage_linking(
    *,
    faculty_emails: Sequence[str],
    grant_ids: Sequence[str],
    all_faculty: bool,
    all_grants: bool,
    include_closed: bool,
    limit: int,
    offset: int,
    min_cosine_sim: float,
    min_attention_score: float,
    top_k_cosine: int,
    top_k_final_per_faculty_keyword: int,
    min_keyword_confidence: float,
    same_section_only: bool,
    attention_alpha: float,
    cross_encoder_model: str,
    cross_encoder_batch_size: int,
    max_pair_text_chars: int,
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

    targets_faculty = _dedupe_nonempty([_clean_text(x).lower() for x in list(faculty_emails or [])])
    targets_grants = _dedupe_nonempty([_clean_text(x) for x in list(grant_ids or [])])

    scorer = _CrossEncoderScorer(
        model_name=_clean_text(cross_encoder_model) or "BAAI/bge-reranker-v2-m3",
        batch_size=_safe_limit(cross_encoder_batch_size, default=16, minimum=1, maximum=512),
    )

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
                "params": {
                    "all_faculty": bool(all_faculty),
                    "all_grants": bool(all_grants),
                },
            }

        grant_keyword_cache: Dict[str, List[Dict[str, Any]]] = {}
        pair_results: List[Dict[str, Any]] = []

        for faculty_email in targets_faculty:
            faculty = _fetch_faculty_identity(
                driver=driver,
                settings=settings,
                faculty_email=faculty_email,
            )
            if not faculty:
                pair_results.append(
                    {
                        "faculty_email": _clean_text(faculty_email).lower(),
                        "status": "faculty_not_found",
                    }
                )
                continue

            faculty_kw = _fetch_faculty_spec_keyword_rows(
                driver=driver,
                settings=settings,
                faculty_email=faculty_email,
            )

            for opportunity_id in targets_grants:
                opp_id = _clean_text(opportunity_id)
                if not opp_id:
                    continue

                if opp_id not in grant_keyword_cache:
                    grant_keyword_cache[opp_id] = _fetch_grant_spec_keyword_rows(
                        driver=driver,
                        settings=settings,
                        opportunity_id=opp_id,
                    )
                grant_kw = grant_keyword_cache[opp_id]

                ranked = _rank_spec_keyword_matches_two_stage(
                    faculty_keyword_rows=faculty_kw,
                    grant_keyword_rows=grant_kw,
                    scorer=scorer,
                    min_cosine_sim=float(min_cosine_sim),
                    min_attention_score=float(min_attention_score),
                    top_k_cosine=int(top_k_cosine),
                    top_k_final_per_faculty_keyword=int(top_k_final_per_faculty_keyword),
                    min_keyword_confidence=float(min_keyword_confidence),
                    same_section_only=bool(same_section_only),
                    attention_alpha=float(attention_alpha),
                    max_pair_text_chars=int(max_pair_text_chars),
                )
                rows = list(ranked.get("rows") or [])
                stats = dict(ranked.get("stats") or {})

                if write_edges:
                    _write_keyword_match_edges(
                        driver=driver,
                        settings=settings,
                        faculty_id=int(faculty["faculty_id"]),
                        faculty_email=_clean_text(faculty_email).lower(),
                        opportunity_id=opp_id,
                        rows=rows,
                        min_cosine_sim=float(min_cosine_sim),
                        min_attention_score=float(min_attention_score),
                        attention_alpha=float(attention_alpha),
                    )

                scores = [float(x.get("score") or 0.0) for x in rows]
                pair_results.append(
                    {
                        "faculty_id": int(faculty["faculty_id"]),
                        "faculty_email": _clean_text(faculty_email).lower(),
                        "opportunity_id": opp_id,
                        "faculty_keyword_count": len(faculty_kw),
                        "grant_keyword_count": len(grant_kw),
                        "cosine_candidates": int(stats.get("cosine_candidates") or 0),
                        "attention_candidates": int(stats.get("attention_candidates") or 0),
                        "edge_count": len(rows),
                        "max_score": max(scores) if scores else 0.0,
                        "mean_score": (sum(scores) / len(scores)) if scores else 0.0,
                    }
                )

    return {
        "params": {
            "all_faculty": bool(all_faculty),
            "all_grants": bool(all_grants),
            "include_closed": bool(include_closed),
            "limit": safe_limit,
            "offset": safe_offset,
            "min_cosine_sim": _safe_unit_float(min_cosine_sim, default=0.45),
            "min_attention_score": _safe_unit_float(min_attention_score, default=0.55),
            "top_k_cosine": _safe_limit(top_k_cosine, default=20, minimum=1, maximum=1000),
            "top_k_final_per_faculty_keyword": _safe_limit(top_k_final_per_faculty_keyword, default=10, minimum=1, maximum=1000),
            "min_keyword_confidence": _safe_unit_float(min_keyword_confidence, default=0.5),
            "same_section_only": bool(same_section_only),
            "attention_alpha": _safe_unit_float(attention_alpha, default=0.5),
            "cross_encoder_model": _clean_text(cross_encoder_model) or "BAAI/bge-reranker-v2-m3",
            "cross_encoder_batch_size": _safe_limit(cross_encoder_batch_size, default=16, minimum=1, maximum=512),
            "max_pair_text_chars": _safe_limit(max_pair_text_chars, default=300, minimum=50, maximum=8000),
            "write_edges": bool(write_edges),
        },
        "totals": {
            "pairs_processed": len(pair_results),
            "cosine_candidates": sum(int(item.get("cosine_candidates", 0) or 0) for item in pair_results),
            "attention_candidates": sum(int(item.get("attention_candidates", 0) or 0) for item in pair_results),
            "edges_written": sum(int(item.get("edge_count", 0) or 0) for item in pair_results),
        },
        "pairs": pair_results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage spec keyword linker: stage-1 cosine prefilter over all faculty<->grant "
            "specialization keyword pairs, then stage-2 cross-encoder scoring on filtered pairs."
        )
    )
    parser.add_argument("--faculty-email", action="append", default=[], help="Target faculty email (repeatable).")
    parser.add_argument("--grant-id", action="append", default=[], help="Target grant opportunity_id (repeatable).")
    parser.add_argument("--all-faculty", action="store_true", help="Process all faculty emails in Neo4j.")
    parser.add_argument("--all-grants", action="store_true", help="Process all grants in Neo4j.")
    parser.add_argument("--include-closed", action="store_true", help="Include closed grants when listing all grants.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit when using --all-* (0 = all).")
    parser.add_argument("--offset", type=int, default=0, help="Optional offset when using --all-*.")

    parser.add_argument("--min-cosine-sim", type=float, default=0.45, help="Stage-1 minimum cosine similarity.")
    parser.add_argument("--min-attention-score", type=float, default=0.55, help="Stage-2 minimum cross-encoder score.")
    parser.add_argument("--top-k-cosine", type=int, default=20, help="Stage-1 max candidates per faculty keyword.")
    parser.add_argument("--top-k-final", type=int, default=10, help="Final max kept links per faculty keyword.")
    parser.add_argument("--attention-alpha", type=float, default=0.5, help="Blend weight for attention in hybrid_sim (0..1).")
    parser.add_argument(
        "--min-keyword-confidence",
        type=float,
        default=0.5,
        help="Fallback confidence floor when keyword evidence confidence is missing.",
    )
    parser.add_argument("--same-section-only", action="store_true", help="Match research->research and application->application only.")

    parser.add_argument("--cross-encoder-model", type=str, default="BAAI/bge-reranker-v2-m3", help="Cross-encoder model name.")
    parser.add_argument("--cross-encoder-batch-size", type=int, default=16, help="Cross-encoder batch size.")
    parser.add_argument("--max-pair-text-chars", type=int, default=300, help="Max chars per keyword phrase sent to cross-encoder.")

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

    payload = run_specialization_keyword_two_stage_linking(
        faculty_emails=list(args.faculty_email or []),
        grant_ids=list(args.grant_id or []),
        all_faculty=bool(args.all_faculty),
        all_grants=bool(args.all_grants),
        include_closed=bool(args.include_closed),
        limit=int(args.limit or 0),
        offset=int(args.offset or 0),
        min_cosine_sim=float(args.min_cosine_sim),
        min_attention_score=float(args.min_attention_score),
        top_k_cosine=int(args.top_k_cosine),
        top_k_final_per_faculty_keyword=int(args.top_k_final),
        min_keyword_confidence=float(args.min_keyword_confidence),
        same_section_only=bool(args.same_section_only),
        attention_alpha=float(args.attention_alpha),
        cross_encoder_model=str(args.cross_encoder_model or ""),
        cross_encoder_batch_size=int(args.cross_encoder_batch_size or 16),
        max_pair_text_chars=int(args.max_pair_text_chars or 300),
        write_edges=not bool(args.skip_write),
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    if not args.json_only:
        totals = payload.get("totals", {})
        print("Two-stage specialization keyword linking complete.")
        print(f"  pairs processed     : {totals.get('pairs_processed', 0)}")
        print(f"  cosine candidates   : {totals.get('cosine_candidates', 0)}")
        print(f"  attention candidates: {totals.get('attention_candidates', 0)}")
        print(f"  edges written       : {totals.get('edges_written', 0)}")
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
