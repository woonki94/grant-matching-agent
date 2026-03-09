from __future__ import annotations

from typing import Any, Dict, List


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


def get_faculty_chunks_by_id(*, faculty_id: int) -> List[Dict[str, Any]]:
    """
    Return all FacultyTextChunk rows linked to a Faculty node by faculty_id.
    """
    try:
        fid = int(faculty_id)
    except Exception:
        return []
    if fid <= 0:
        return []

    rows = _fetch_neo4j_rows(
        """
        MATCH (f:Faculty {faculty_id: $faculty_id})-[r]->(c:FacultyTextChunk)
        WHERE c.chunk_id IS NOT NULL
          AND c.text IS NOT NULL
        RETURN
            c.chunk_id AS chunk_id,
            type(r) AS relation,
            c.source_type AS source_type,
            c.source_ref_id AS source_ref_id,
            c.source_url AS source_url,
            c.chunk_index AS chunk_index,
            c.char_count AS char_count,
            c.text AS text
        ORDER BY
            c.source_type ASC,
            c.source_ref_id ASC,
            c.chunk_index ASC,
            c.chunk_id ASC
        """,
        {"faculty_id": fid},
    )

    out: List[Dict[str, Any]] = []
    seen_chunk_ids = set()
    for row in rows:
        chunk_id = str(row.get("chunk_id") or "").strip()
        if not chunk_id or chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        out.append(
            {
                "chunk_id": chunk_id,
                "relation": str(row.get("relation") or "").strip() or None,
                "source_type": str(row.get("source_type") or "").strip() or None,
                "source_ref_id": str(row.get("source_ref_id") or "").strip() or None,
                "source_url": str(row.get("source_url") or "").strip() or None,
                "chunk_index": row.get("chunk_index"),
                "char_count": row.get("char_count"),
                "text": str(row.get("text") or ""),
            }
        )
    return out
