from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from graph_rag.common import (
    Neo4jSettings,
    json_ready,
    load_dotenv_if_present,
    read_neo4j_settings,
)

GRANT_SPEC_RELATIONS = [
    "HAS_RESEARCH_SPECIALIZATION",
    "HAS_APPLICATION_SPECIALIZATION",
]
GRANT_CHUNK_RELATIONS = [
    "HAS_SUMMARY_CHUNK",
    "HAS_ADDITIONAL_INFO_CHUNK",
    "HAS_ATTACHMENT_CHUNK",
]
FACULTY_SPEC_RELATIONS = [
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


def _rank_chunk_links_for_keywords(
    *,
    keyword_rows: List[Dict[str, Any]],
    chunk_rows: List[Dict[str, Any]],
    min_score: float,
    top_k_per_keyword: int,
) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    safe_top_k = max(1, int(top_k_per_keyword or 1))
    safe_min_score = float(min_score)

    prepared_chunks: List[Tuple[Dict[str, Any], np.ndarray]] = []
    for row in chunk_rows or []:
        vec = _coerce_vector(row.get("embedding"))
        if vec.size == 0:
            continue
        prepared_chunks.append((row, vec))

    for kw in keyword_rows or []:
        kw_vec = _coerce_vector(kw.get("embedding"))
        if kw_vec.size == 0:
            continue

        compatible: List[Tuple[Dict[str, Any], np.ndarray]] = [
            (chunk_row, chunk_vec)
            for chunk_row, chunk_vec in prepared_chunks
            if int(chunk_vec.shape[0]) == int(kw_vec.shape[0])
        ]
        if not compatible:
            continue

        chunk_matrix = np.asarray([vec for _, vec in compatible], dtype=np.float32)
        sims = _cosine_matrix(kw_vec.reshape(1, -1), chunk_matrix)[0]
        order = np.argsort(-sims)

        taken = 0
        for idx in order:
            score = float(sims[int(idx)])
            if score < safe_min_score:
                continue

            chunk_row = compatible[int(idx)][0]
            links.append(
                {
                    "keyword_value": _clean_text(kw.get("value")),
                    "keyword_section": _clean_text(kw.get("section")),
                    "keyword_bucket": _clean_text(kw.get("bucket")),
                    "chunk_id": _clean_text(chunk_row.get("chunk_id")),
                    "chunk_source_type": _clean_text(chunk_row.get("source_type")),
                    "score": max(0.0, min(1.0, score)),
                }
            )
            taken += 1
            if taken >= safe_top_k:
                break

    return links


def _list_grant_ids(
    *,
    driver,
    settings: Neo4jSettings,
    limit: int,
    offset: int,
) -> List[str]:
    query = """
        MATCH (g:Grant)
        WHERE g.opportunity_id IS NOT NULL
        RETURN g.opportunity_id AS opportunity_id
        ORDER BY g.opportunity_id ASC
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


def _fetch_grant_keyword_rows(
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
        RETURN DISTINCT
            k.value AS value,
            k.section AS section,
            k.bucket AS bucket,
            k.embedding AS embedding
        ORDER BY value ASC
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
        ORDER BY chunk_id ASC
        """,
        parameters_={
            "opportunity_id": _clean_text(opportunity_id),
            "relations": GRANT_CHUNK_RELATIONS,
        },
        database_=settings.database,
    )
    return [dict(row or {}) for row in records]


def _fetch_faculty_id(
    *,
    driver,
    settings: Neo4jSettings,
    faculty_email: str,
) -> Optional[int]:
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})
        RETURN f.faculty_id AS faculty_id
        LIMIT 1
        """,
        parameters_={"email": _clean_text(faculty_email).lower()},
        database_=settings.database,
    )
    if not records:
        return None
    try:
        return int(records[0].get("faculty_id"))
    except Exception:
        return None


def _fetch_faculty_keyword_rows(
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
        RETURN DISTINCT
            k.value AS value,
            k.section AS section,
            k.bucket AS bucket,
            k.embedding AS embedding
        ORDER BY value ASC
        """,
        parameters_={
            "email": _clean_text(faculty_email).lower(),
            "relations": FACULTY_SPEC_RELATIONS,
        },
        database_=settings.database,
    )
    return [dict(row or {}) for row in records]


def _fetch_faculty_chunk_rows(
    *,
    driver,
    settings: Neo4jSettings,
    faculty_email: str,
) -> List[Dict[str, Any]]:
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})-[:HAS_ADDITIONAL_INFO_CHUNK]->(c:FacultyTextChunk)
        WHERE c.embedding IS NOT NULL
          AND c.chunk_id IS NOT NULL
        RETURN DISTINCT
            c.chunk_id AS chunk_id,
            c.source_type AS source_type,
            c.embedding AS embedding
        ORDER BY chunk_id ASC
        """,
        parameters_={"email": _clean_text(faculty_email).lower()},
        database_=settings.database,
    )
    return [dict(row or {}) for row in records]


def link_grant_keyword_chunk_edges(
    *,
    driver,
    settings: Neo4jSettings,
    opportunity_id: str,
    min_score: float,
    top_k_per_keyword: int,
) -> Dict[str, Any]:
    grant_id = _clean_text(opportunity_id)
    if not grant_id:
        return {"opportunity_id": "", "status": "skipped", "reason": "missing_opportunity_id"}

    keyword_rows = _fetch_grant_keyword_rows(
        driver=driver,
        settings=settings,
        opportunity_id=grant_id,
    )
    chunk_rows = _fetch_grant_chunk_rows(
        driver=driver,
        settings=settings,
        opportunity_id=grant_id,
    )

    links = _rank_chunk_links_for_keywords(
        keyword_rows=keyword_rows,
        chunk_rows=chunk_rows,
        min_score=float(min_score),
        top_k_per_keyword=int(top_k_per_keyword),
    )

    driver.execute_query(
        """
        MATCH (:GrantKeyword)-[r:SUPPORTED_BY_GRANT_CHUNK {scope_grant_id: $opportunity_id}]->(:GrantTextChunk)
        DELETE r
        """,
        parameters_={"opportunity_id": grant_id},
        database_=settings.database,
    )

    if links:
        driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (k:GrantKeyword {
                value: row.keyword_value,
                section: row.keyword_section,
                bucket: row.keyword_bucket
            })
            MATCH (c:GrantTextChunk {chunk_id: row.chunk_id})
            MERGE (k)-[r:SUPPORTED_BY_GRANT_CHUNK {
                scope_grant_id: $opportunity_id,
                chunk_id: row.chunk_id,
                keyword_value: row.keyword_value,
                keyword_section: row.keyword_section,
                keyword_bucket: row.keyword_bucket
            }]->(c)
            SET
                r.score = row.score,
                r.method = 'embedding_cosine',
                r.chunk_source_type = row.chunk_source_type,
                r.updated_at = datetime()
            """,
            parameters_={
                "opportunity_id": grant_id,
                "rows": links,
            },
            database_=settings.database,
        )

    return {
        "opportunity_id": grant_id,
        "keyword_count": len(keyword_rows),
        "chunk_count": len(chunk_rows),
        "edge_count": len(links),
    }


def link_faculty_keyword_chunk_edges(
    *,
    driver,
    settings: Neo4jSettings,
    faculty_email: str,
    min_score: float,
    top_k_per_keyword: int,
) -> Dict[str, Any]:
    email = _clean_text(faculty_email).lower()
    if not email:
        return {"email": "", "status": "skipped", "reason": "missing_faculty_email"}

    faculty_id = _fetch_faculty_id(
        driver=driver,
        settings=settings,
        faculty_email=email,
    )
    if faculty_id is None:
        return {"email": email, "status": "not_found"}

    keyword_rows = _fetch_faculty_keyword_rows(
        driver=driver,
        settings=settings,
        faculty_email=email,
    )
    chunk_rows = _fetch_faculty_chunk_rows(
        driver=driver,
        settings=settings,
        faculty_email=email,
    )

    links = _rank_chunk_links_for_keywords(
        keyword_rows=keyword_rows,
        chunk_rows=chunk_rows,
        min_score=float(min_score),
        top_k_per_keyword=int(top_k_per_keyword),
    )

    driver.execute_query(
        """
        MATCH (:FacultyKeyword)-[r:SUPPORTED_BY_FACULTY_CHUNK {scope_faculty_id: $faculty_id}]->(:FacultyTextChunk)
        DELETE r
        """,
        parameters_={"faculty_id": int(faculty_id)},
        database_=settings.database,
    )

    if links:
        driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (k:FacultyKeyword {
                value: row.keyword_value,
                section: row.keyword_section,
                bucket: row.keyword_bucket
            })
            MATCH (c:FacultyTextChunk {chunk_id: row.chunk_id})
            MERGE (k)-[r:SUPPORTED_BY_FACULTY_CHUNK {
                scope_faculty_id: $faculty_id,
                chunk_id: row.chunk_id,
                keyword_value: row.keyword_value,
                keyword_section: row.keyword_section,
                keyword_bucket: row.keyword_bucket
            }]->(c)
            SET
                r.score = row.score,
                r.method = 'embedding_cosine',
                r.chunk_source_type = row.chunk_source_type,
                r.updated_at = datetime()
            """,
            parameters_={
                "faculty_id": int(faculty_id),
                "rows": links,
            },
            database_=settings.database,
        )

    return {
        "email": email,
        "faculty_id": int(faculty_id),
        "keyword_count": len(keyword_rows),
        "chunk_count": len(chunk_rows),
        "edge_count": len(links),
    }


def run_keyword_chunk_linker(
    *,
    grant_ids: Sequence[str],
    faculty_emails: Sequence[str],
    all_grants: bool,
    all_faculty: bool,
    limit: int,
    offset: int,
    min_score: float,
    top_k_per_keyword: int,
) -> Dict[str, Any]:
    from neo4j import GraphDatabase

    load_dotenv_if_present()
    settings = read_neo4j_settings()

    safe_limit = _safe_limit(limit, default=0, minimum=0, maximum=100000)
    safe_offset = _safe_limit(offset, default=0, minimum=0, maximum=1000000)
    safe_top_k = _safe_limit(top_k_per_keyword, default=3, minimum=1, maximum=20)
    safe_min_score = max(0.0, min(1.0, float(min_score)))

    targets_grant = _dedupe_nonempty([_clean_text(x) for x in list(grant_ids or [])])
    targets_faculty = _dedupe_nonempty([_clean_text(x).lower() for x in list(faculty_emails or [])])

    with GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    ) as driver:
        driver.verify_connectivity()

        if all_grants:
            targets_grant = _list_grant_ids(
                driver=driver,
                settings=settings,
                limit=safe_limit,
                offset=safe_offset,
            )
        if all_faculty:
            targets_faculty = _list_faculty_emails(
                driver=driver,
                settings=settings,
                limit=safe_limit,
                offset=safe_offset,
            )

        grant_results: List[Dict[str, Any]] = []
        for opportunity_id in targets_grant:
            grant_results.append(
                link_grant_keyword_chunk_edges(
                    driver=driver,
                    settings=settings,
                    opportunity_id=opportunity_id,
                    min_score=safe_min_score,
                    top_k_per_keyword=safe_top_k,
                )
            )

        faculty_results: List[Dict[str, Any]] = []
        for faculty_email in targets_faculty:
            faculty_results.append(
                link_faculty_keyword_chunk_edges(
                    driver=driver,
                    settings=settings,
                    faculty_email=faculty_email,
                    min_score=safe_min_score,
                    top_k_per_keyword=safe_top_k,
                )
            )

    return {
        "params": {
            "all_grants": bool(all_grants),
            "all_faculty": bool(all_faculty),
            "limit": safe_limit,
            "offset": safe_offset,
            "min_score": safe_min_score,
            "top_k_per_keyword": safe_top_k,
        },
        "totals": {
            "grants_processed": len(grant_results),
            "faculties_processed": len(faculty_results),
            "grant_edges_written": sum(int(item.get("edge_count", 0) or 0) for item in grant_results),
            "faculty_edges_written": sum(int(item.get("edge_count", 0) or 0) for item in faculty_results),
        },
        "grants": grant_results,
        "faculties": faculty_results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create keyword-to-chunk support edges using existing specialization keyword embeddings "
            "and chunk embeddings in Neo4j."
        )
    )
    parser.add_argument("--grant-id", action="append", default=[], help="Target grant opportunity_id. Repeatable.")
    parser.add_argument("--faculty-email", action="append", default=[], help="Target faculty email. Repeatable.")
    parser.add_argument("--all-grants", action="store_true", help="Process all grants.")
    parser.add_argument("--all-faculty", action="store_true", help="Process all faculties.")
    parser.add_argument("--limit", type=int, default=0, help="Optional list limit when using --all-* flags (0 = all).")
    parser.add_argument("--offset", type=int, default=0, help="Optional list offset when using --all-* flags.")
    parser.add_argument("--min-score", type=float, default=0.45, help="Minimum cosine score to keep an edge.")
    parser.add_argument("--top-k", type=int, default=3, help="Top chunks kept per keyword.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if (
        not bool(args.all_grants)
        and not bool(args.all_faculty)
        and not list(args.grant_id or [])
        and not list(args.faculty_email or [])
    ):
        raise SystemExit(
            "Provide at least one target: --grant-id, --faculty-email, --all-grants, or --all-faculty."
        )

    payload = run_keyword_chunk_linker(
        grant_ids=list(args.grant_id or []),
        faculty_emails=list(args.faculty_email or []),
        all_grants=bool(args.all_grants),
        all_faculty=bool(args.all_faculty),
        limit=int(args.limit or 0),
        offset=int(args.offset or 0),
        min_score=float(args.min_score),
        top_k_per_keyword=int(args.top_k or 3),
    )
    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
