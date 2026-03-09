from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from graph_rag.common import json_ready, load_dotenv_if_present, read_neo4j_settings


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


def retrieve_grants_by_cross_chunk_edges(
    *,
    faculty_id: Optional[int],
    faculty_email: str,
    top_k: int,
    pairs_per_grant: int,
    min_edge_score: float,
    coverage_bonus: float,
    include_closed: bool,
    uri: str = "",
    username: str = "",
    password: str = "",
    database: str = "",
) -> Dict[str, Any]:
    fid = int(faculty_id) if faculty_id is not None else None
    femail = _clean_text(faculty_email).lower()
    if fid is None and not femail:
        raise ValueError("Provide faculty_id or faculty_email.")

    safe_top_k = _safe_limit(top_k, default=20, minimum=1, maximum=2000)
    safe_pairs_per_grant = _safe_limit(pairs_per_grant, default=20, minimum=1, maximum=500)
    safe_min_edge_score = max(0.0, min(1.0, float(min_edge_score)))
    safe_coverage_bonus = max(0.0, float(coverage_bonus))

    load_dotenv_if_present()
    settings = read_neo4j_settings(
        uri=uri,
        username=username,
        password=password,
        database=database,
    )

    query = """
        MATCH (g:Grant)
        WHERE g.opportunity_id IS NOT NULL
        WITH
            g,
            toLower(coalesce(g.opportunity_status, '')) AS status_token,
            coalesce(toString(g.close_date), '') AS close_token
        WITH
            g,
            status_token,
            CASE
                WHEN close_token =~ '^\\\\d{4}-\\\\d{2}-\\\\d{2}.*$' THEN date(substring(close_token, 0, 10))
                ELSE NULL
            END AS close_dt
        WHERE
            $include_closed
            OR (
                NONE(token IN ['closed', 'archived', 'inactive', 'canceled'] WHERE status_token CONTAINS token)
                AND (close_dt IS NULL OR close_dt >= date())
            )
        CALL (g) {
            OPTIONAL MATCH ()-[m:FAC_SPEC_SUPPORTS_GRANT_CHUNK]->(gc:GrantTextChunk)
            WHERE m.scope_opportunity_id = g.opportunity_id
              AND ($faculty_id IS NULL OR m.scope_faculty_id = $faculty_id)
              AND ($faculty_email = '' OR toLower(m.scope_faculty_email) = $faculty_email)
              AND coalesce(m.score, 0.0) >= $min_edge_score
            WITH m, gc, coalesce(m.score, 0.0) AS s
            ORDER BY s DESC
            RETURN
                sum(s) AS f2g_score,
                count(m) AS f2g_edge_count,
                count(DISTINCT m.faculty_keyword_value) AS f2g_faculty_keyword_coverage,
                count(DISTINCT gc.chunk_id) AS f2g_grant_chunk_coverage,
                collect({
                    faculty_keyword_value: m.faculty_keyword_value,
                    faculty_keyword_section: m.faculty_keyword_section,
                    faculty_keyword_weight: coalesce(m.faculty_keyword_weight, 0.0),
                    faculty_keyword_confidence: coalesce(m.faculty_keyword_confidence, 0.0),
                    grant_chunk_id: coalesce(m.grant_chunk_id, gc.chunk_id),
                    grant_chunk_source_type: coalesce(m.grant_chunk_source_type, gc.source_type),
                    cosine_sim: coalesce(m.sim, 0.0),
                    edge_score: s,
                    domain_gate_score: coalesce(m.domain_gate_score, 0.0)
                })[0..$pairs_per_grant] AS fac_to_grant_chunk_pairs
        }
        CALL (g) {
            OPTIONAL MATCH ()-[m:GRANT_SPEC_SUPPORTS_FAC_EVIDENCE]->(ev)
            WHERE m.scope_opportunity_id = g.opportunity_id
              AND ($faculty_id IS NULL OR m.scope_faculty_id = $faculty_id)
              AND ($faculty_email = '' OR toLower(m.scope_faculty_email) = $faculty_email)
              AND coalesce(m.score, 0.0) >= $min_edge_score
            WITH m, ev, coalesce(m.score, 0.0) AS s
            ORDER BY s DESC
            RETURN
                sum(s) AS g2f_score,
                count(m) AS g2f_edge_count,
                count(DISTINCT m.grant_keyword_value) AS g2f_grant_keyword_coverage,
                count(DISTINCT CASE
                    WHEN coalesce(m.faculty_evidence_kind, '') = 'chunk' THEN coalesce(m.faculty_chunk_id, ev.chunk_id)
                    WHEN coalesce(m.faculty_evidence_kind, '') = 'publication' THEN coalesce(toString(m.faculty_publication_id), toString(ev.publication_id))
                    ELSE coalesce(m.faculty_chunk_id, toString(m.faculty_publication_id), ev.chunk_id, toString(ev.publication_id))
                END) AS g2f_faculty_evidence_coverage,
                collect({
                    grant_keyword_value: m.grant_keyword_value,
                    grant_keyword_section: m.grant_keyword_section,
                    grant_keyword_weight: coalesce(m.grant_keyword_weight, 0.0),
                    grant_keyword_confidence: coalesce(m.grant_keyword_confidence, 0.0),
                    faculty_evidence_kind: coalesce(m.faculty_evidence_kind, CASE WHEN 'FacultyPublication' IN labels(ev) THEN 'publication' ELSE 'chunk' END),
                    faculty_evidence_id: CASE
                        WHEN coalesce(m.faculty_evidence_kind, '') = 'chunk' THEN coalesce(m.faculty_chunk_id, ev.chunk_id)
                        WHEN coalesce(m.faculty_evidence_kind, '') = 'publication' THEN coalesce(toString(m.faculty_publication_id), toString(ev.publication_id))
                        ELSE coalesce(m.faculty_chunk_id, toString(m.faculty_publication_id), ev.chunk_id, toString(ev.publication_id))
                    END,
                    faculty_evidence_source_type: coalesce(m.faculty_evidence_source_type, ev.source_type, 'publication_abstract'),
                    cosine_sim: coalesce(m.sim, 0.0),
                    edge_score: s,
                    domain_gate_score: coalesce(m.domain_gate_score, 0.0)
                })[0..$pairs_per_grant] AS grant_to_fac_evidence_pairs
        }
        WITH
            g,
            coalesce(f2g_score, 0.0) AS f2g_score,
            coalesce(g2f_score, 0.0) AS g2f_score,
            coalesce(f2g_edge_count, 0) AS f2g_edge_count,
            coalesce(g2f_edge_count, 0) AS g2f_edge_count,
            coalesce(f2g_faculty_keyword_coverage, 0) AS f2g_faculty_keyword_coverage,
            coalesce(f2g_grant_chunk_coverage, 0) AS f2g_grant_chunk_coverage,
            coalesce(g2f_grant_keyword_coverage, 0) AS g2f_grant_keyword_coverage,
            coalesce(g2f_faculty_evidence_coverage, 0) AS g2f_faculty_evidence_coverage,
            coalesce(fac_to_grant_chunk_pairs, []) AS fac_to_grant_chunk_pairs,
            coalesce(grant_to_fac_evidence_pairs, []) AS grant_to_fac_evidence_pairs
        WHERE f2g_edge_count > 0 OR g2f_edge_count > 0
        WITH
            g,
            f2g_score,
            g2f_score,
            f2g_edge_count,
            g2f_edge_count,
            f2g_faculty_keyword_coverage,
            f2g_grant_chunk_coverage,
            g2f_grant_keyword_coverage,
            g2f_faculty_evidence_coverage,
            fac_to_grant_chunk_pairs,
            grant_to_fac_evidence_pairs,
            (f2g_score + g2f_score) AS base_score,
            (
                $coverage_bonus * toFloat(
                    f2g_faculty_keyword_coverage
                    + f2g_grant_chunk_coverage
                    + g2f_grant_keyword_coverage
                    + g2f_faculty_evidence_coverage
                )
            ) AS coverage_bonus_score
        WITH
            g,
            f2g_score,
            g2f_score,
            f2g_edge_count,
            g2f_edge_count,
            f2g_faculty_keyword_coverage,
            f2g_grant_chunk_coverage,
            g2f_grant_keyword_coverage,
            g2f_faculty_evidence_coverage,
            fac_to_grant_chunk_pairs,
            grant_to_fac_evidence_pairs,
            base_score,
            coverage_bonus_score,
            (base_score + coverage_bonus_score) AS rank_score
        RETURN
            g.opportunity_id AS opportunity_id,
            g.opportunity_title AS opportunity_title,
            g.agency_name AS agency_name,
            g.opportunity_status AS opportunity_status,
            g.close_date AS close_date,
            rank_score,
            base_score,
            coverage_bonus_score,
            f2g_score,
            g2f_score,
            f2g_edge_count,
            g2f_edge_count,
            (f2g_edge_count + g2f_edge_count) AS total_edge_count,
            f2g_faculty_keyword_coverage,
            f2g_grant_chunk_coverage,
            g2f_grant_keyword_coverage,
            g2f_faculty_evidence_coverage,
            fac_to_grant_chunk_pairs,
            grant_to_fac_evidence_pairs
        ORDER BY rank_score DESC, base_score DESC, total_edge_count DESC
        LIMIT $limit
    """

    with GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    ) as driver:
        driver.verify_connectivity()
        records, _, _ = driver.execute_query(
            query,
            parameters_={
                "faculty_id": fid,
                "faculty_email": femail,
                "min_edge_score": safe_min_edge_score,
                "pairs_per_grant": safe_pairs_per_grant,
                "coverage_bonus": float(safe_coverage_bonus),
                "include_closed": bool(include_closed),
                "limit": safe_top_k,
            },
            database_=settings.database,
        )

    rows: List[Dict[str, Any]] = []
    for row in records:
        item = dict(row or {})
        rows.append(
            {
                "opportunity_id": _clean_text(item.get("opportunity_id")),
                "opportunity_title": _clean_text(item.get("opportunity_title")),
                "agency_name": _clean_text(item.get("agency_name")),
                "opportunity_status": _clean_text(item.get("opportunity_status")),
                "close_date": _clean_text(item.get("close_date")),
                "rank_score": float(item.get("rank_score") or 0.0),
                "base_score": float(item.get("base_score") or 0.0),
                "coverage_bonus_score": float(item.get("coverage_bonus_score") or 0.0),
                "f2g_score": float(item.get("f2g_score") or 0.0),
                "g2f_score": float(item.get("g2f_score") or 0.0),
                "f2g_edge_count": int(item.get("f2g_edge_count") or 0),
                "g2f_edge_count": int(item.get("g2f_edge_count") or 0),
                "total_edge_count": int(item.get("total_edge_count") or 0),
                "f2g_faculty_keyword_coverage": int(item.get("f2g_faculty_keyword_coverage") or 0),
                "f2g_grant_chunk_coverage": int(item.get("f2g_grant_chunk_coverage") or 0),
                "g2f_grant_keyword_coverage": int(item.get("g2f_grant_keyword_coverage") or 0),
                "g2f_faculty_evidence_coverage": int(item.get("g2f_faculty_evidence_coverage") or 0),
                "fac_to_grant_chunk_pairs": list(item.get("fac_to_grant_chunk_pairs") or []),
                "grant_to_fac_evidence_pairs": list(item.get("grant_to_fac_evidence_pairs") or []),
            }
        )

    return {
        "params": {
            "faculty_id": fid,
            "faculty_email": femail,
            "top_k": safe_top_k,
            "pairs_per_grant": safe_pairs_per_grant,
            "min_edge_score": safe_min_edge_score,
            "coverage_bonus": float(safe_coverage_bonus),
            "include_closed": bool(include_closed),
        },
        "totals": {
            "grants": len(rows),
        },
        "grants": rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve grants ranked by cross-chunk matcher edges "
            "(FAC_SPEC_SUPPORTS_GRANT_CHUNK + GRANT_SPEC_SUPPORTS_FAC_EVIDENCE)."
        )
    )
    parser.add_argument("--faculty-id", type=int, default=0, help="Faculty ID filter.")
    parser.add_argument("--faculty-email", type=str, default="", help="Faculty email filter.")
    parser.add_argument("--top-k", type=int, default=20, help="Top K ranked grants.")
    parser.add_argument(
        "--pairs-per-grant",
        type=int,
        default=20,
        help="Max matched pair rows returned per edge direction per grant.",
    )
    parser.add_argument("--min-edge-score", type=float, default=0.0, help="Minimum edge score to include.")
    parser.add_argument(
        "--coverage-bonus",
        type=float,
        default=0.0,
        help="Optional additive bonus multiplier for coverage counts (default 0).",
    )
    parser.add_argument("--include-closed", action="store_true", help="Include closed grants in output.")
    parser.add_argument("--json-only", action="store_true", help="Print only JSON output.")

    parser.add_argument("--uri", type=str, default="", help="Neo4j URI. Fallback: NEO4J_URI")
    parser.add_argument("--username", type=str, default="", help="Neo4j username. Fallback: NEO4J_USERNAME")
    parser.add_argument("--password", type=str, default="", help="Neo4j password. Fallback: NEO4J_PASSWORD")
    parser.add_argument("--database", type=str, default="", help="Neo4j database. Fallback: NEO4J_DATABASE or neo4j")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    fid = int(args.faculty_id or 0)
    faculty_id = fid if fid > 0 else None

    payload = retrieve_grants_by_cross_chunk_edges(
        faculty_id=faculty_id,
        faculty_email=_clean_text(args.faculty_email).lower(),
        top_k=int(args.top_k or 20),
        pairs_per_grant=int(args.pairs_per_grant or 20),
        min_edge_score=float(args.min_edge_score),
        coverage_bonus=float(args.coverage_bonus),
        include_closed=bool(args.include_closed),
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    if not args.json_only:
        print("Grant retrieval from cross-chunk edges complete.")
        print(f"  grants returned : {payload.get('totals', {}).get('grants', 0)}")
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
