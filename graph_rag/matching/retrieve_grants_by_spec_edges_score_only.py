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


def retrieve_grants_ranked_by_spec_edge_score_only(
    *,
    faculty_id: Optional[int],
    faculty_email: str,
    top_k: int,
    pairs_per_grant: int,
    min_edge_score: float,
    include_closed: bool,
    edge_method: str,
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
    safe_edge_method = _clean_text(edge_method)

    load_dotenv_if_present()
    settings = read_neo4j_settings(
        uri=uri,
        username=username,
        password=password,
        database=database,
    )

    query = """
        MATCH ()-[m:FACULTY_SPEC_MATCHES_GRANT_SPEC]->()
        WHERE
            ($faculty_id IS NULL OR m.scope_faculty_id = $faculty_id)
            AND ($faculty_email = '' OR toLower(m.scope_faculty_email) = $faculty_email)
            AND m.scope_opportunity_id IS NOT NULL
            AND coalesce(m.score, 0.0) >= $min_edge_score
            AND ($edge_method = '' OR coalesce(m.method, '') = $edge_method)
        WITH
            m.scope_opportunity_id AS opportunity_id,
            m
        ORDER BY opportunity_id ASC, coalesce(m.score, 0.0) DESC
        WITH
            opportunity_id,
            sum(coalesce(m.score, 0.0)) AS rank_score,
            count(*) AS edge_count,
            collect({
                faculty_keyword_value: m.faculty_keyword_value,
                faculty_keyword_section: m.faculty_keyword_section,
                grant_keyword_value: m.grant_keyword_value,
                grant_keyword_section: m.grant_keyword_section,
                faculty_keyword_weight: coalesce(m.faculty_keyword_weight, 0.0),
                grant_keyword_weight: coalesce(m.grant_keyword_weight, 0.0),
                faculty_keyword_confidence: coalesce(m.faculty_keyword_confidence, 0.0),
                grant_keyword_confidence: coalesce(m.grant_keyword_confidence, 0.0),
                cosine_sim: coalesce(m.cosine_sim, 0.0),
                attention_score: coalesce(m.attention_score, 0.0),
                hybrid_sim: coalesce(m.hybrid_sim, 0.0),
                edge_score: coalesce(m.score, 0.0),
                method: coalesce(m.method, '')
            })[0..$pairs_per_grant] AS matched_pairs
        MATCH (g:Grant {opportunity_id: opportunity_id})
        WITH
            g,
            rank_score,
            edge_count,
            matched_pairs,
            toLower(coalesce(g.opportunity_status, '')) AS status_token,
            coalesce(toString(g.close_date), '') AS close_token
        WITH
            g,
            rank_score,
            edge_count,
            matched_pairs,
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
        RETURN
            g.opportunity_id AS opportunity_id,
            g.opportunity_title AS opportunity_title,
            g.agency_name AS agency_name,
            g.opportunity_status AS opportunity_status,
            g.close_date AS close_date,
            rank_score,
            edge_count,
            matched_pairs
        ORDER BY rank_score DESC
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
                "include_closed": bool(include_closed),
                "pairs_per_grant": safe_pairs_per_grant,
                "limit": safe_top_k,
                "edge_method": safe_edge_method,
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
                "edge_count": int(item.get("edge_count") or 0),
                "matched_pairs": list(item.get("matched_pairs") or []),
            }
        )

    return {
        "params": {
            "faculty_id": fid,
            "faculty_email": femail,
            "top_k": safe_top_k,
            "pairs_per_grant": safe_pairs_per_grant,
            "min_edge_score": safe_min_edge_score,
            "include_closed": bool(include_closed),
            "edge_method": safe_edge_method,
            "ranking": "sum(edge_score) only",
        },
        "totals": {
            "grants": len(rows),
        },
        "grants": rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve grants ranked only by sum of FACULTY_SPEC_MATCHES_GRANT_SPEC edge score."
        )
    )
    parser.add_argument("--faculty-id", type=int, default=0, help="Faculty ID filter.")
    parser.add_argument("--faculty-email", type=str, default="", help="Faculty email filter.")
    parser.add_argument("--top-k", type=int, default=20, help="Top K ranked grants.")
    parser.add_argument(
        "--pairs-per-grant",
        type=int,
        default=20,
        help="Max matched keyword pairs returned per grant (ordered by edge score).",
    )
    parser.add_argument("--min-edge-score", type=float, default=0.0, help="Minimum edge score to include.")
    parser.add_argument("--include-closed", action="store_true", help="Include closed grants in output.")
    parser.add_argument(
        "--edge-method",
        type=str,
        default="specialization_keyword_two_stage_hybrid",
        help="Filter by edge method. Use empty string to include all methods.",
    )
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

    payload = retrieve_grants_ranked_by_spec_edge_score_only(
        faculty_id=faculty_id,
        faculty_email=_clean_text(args.faculty_email).lower(),
        top_k=int(args.top_k or 20),
        pairs_per_grant=int(args.pairs_per_grant or 20),
        min_edge_score=float(args.min_edge_score),
        include_closed=bool(args.include_closed),
        edge_method=str(args.edge_method or ""),
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    if not args.json_only:
        print("Grant retrieval from spec edges (score-only) complete.")
        print(f"  grants returned : {payload.get('totals', {}).get('grants', 0)}")
        grants = list(payload.get("grants") or [])
        if grants:
            print("  top results:")
            for idx, row in enumerate(grants[:10], start=1):
                print(
                    f"    {idx:02d}. {row.get('opportunity_id', '')} | "
                    f"{row.get('rank_score', 0.0):.4f} | {row.get('opportunity_title', '')}"
                )
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
