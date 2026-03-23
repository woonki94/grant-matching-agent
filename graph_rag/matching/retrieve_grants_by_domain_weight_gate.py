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


def _fetch_specialization_coverage_scores(
    *,
    driver,
    database: str,
    faculty_id: Optional[int],
    faculty_email: str,
    candidate_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    if not candidate_ids:
        return {}

    records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty)
        WHERE
            ($faculty_id IS NULL OR f.faculty_id = $faculty_id)
            AND ($faculty_email = '' OR toLower(f.email) = $faculty_email)
        UNWIND $candidate_ids AS oid
        OPTIONAL MATCH (f)-[r:FACULTY_GRANT_SPEC_COVERAGE]->(g:Grant)
        WHERE toString(g.opportunity_id) = oid
        RETURN
            oid AS opportunity_id,
            coalesce(r.faculty_covers_grant_specs_coverage_weighted_avg, 0.0) AS f2g_coverage_weighted_avg,
            coalesce(r.grant_covers_faculty_specs_coverage_weighted_avg, 0.0) AS g2f_coverage_weighted_avg,
            coalesce(r.faculty_covers_grant_specs_coverage_avg, 0.0) AS f2g_coverage_avg,
            coalesce(r.grant_covers_faculty_specs_coverage_avg, 0.0) AS g2f_coverage_avg,
            coalesce(r.faculty_covers_grant_specs_hit_ratio, 0.0) AS f2g_hit_ratio,
            coalesce(r.grant_covers_faculty_specs_hit_ratio, 0.0) AS g2f_hit_ratio
        """,
        parameters_={
            "faculty_id": faculty_id,
            "faculty_email": _clean_text(faculty_email).lower(),
            "candidate_ids": [str(x) for x in candidate_ids if _clean_text(x)],
        },
        database_=database,
    )

    out: Dict[str, Dict[str, float]] = {}
    for row in records:
        item = dict(row or {})
        oid = _clean_text(item.get("opportunity_id"))
        if not oid:
            continue
        f2g_w = _safe_unit_float(item.get("f2g_coverage_weighted_avg"), default=0.0)
        g2f_w = _safe_unit_float(item.get("g2f_coverage_weighted_avg"), default=0.0)
        # Stage-2 specialization coverage score for rerank.
        spec_cov = (f2g_w + g2f_w) * 0.5
        out[oid] = {
            "specialization_coverage_score": float(spec_cov),
            "f2g_coverage_weighted_avg": float(f2g_w),
            "g2f_coverage_weighted_avg": float(g2f_w),
            "f2g_coverage_avg": _safe_unit_float(item.get("f2g_coverage_avg"), default=0.0),
            "g2f_coverage_avg": _safe_unit_float(item.get("g2f_coverage_avg"), default=0.0),
            "f2g_hit_ratio": _safe_unit_float(item.get("f2g_hit_ratio"), default=0.0),
            "g2f_hit_ratio": _safe_unit_float(item.get("g2f_hit_ratio"), default=0.0),
        }
    return out


def retrieve_grants_by_domain_weight_gate(
    *,
    faculty_id: Optional[int],
    faculty_email: str,
    min_domain_weight: float,
    top_k: int,
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

    safe_min_weight = _safe_unit_float(min_domain_weight, default=0.7)
    safe_top_k = _safe_limit(top_k, default=20, minimum=1, maximum=2000)

    load_dotenv_if_present()
    settings = read_neo4j_settings(
        uri=uri,
        username=username,
        password=password,
        database=database,
    )

    query = """
        MATCH (f:Faculty)
        WHERE
            ($faculty_id IS NULL OR f.faculty_id = $faculty_id)
            AND ($faculty_email = '' OR toLower(f.email) = $faculty_email)

        MATCH (f)-[fr:HAS_DOMAIN_KEYWORD]->(fk:FacultyKeyword {bucket: 'domain'})
        WHERE coalesce(fr.weight, 0.0) > $min_domain_weight

        MATCH (fk)-[:MAPS_TO_SHARED_DOMAIN]->(d:DomainKeywordShared {bucket: 'domain'})

        MATCH (gk:GrantKeyword {bucket: 'domain'})-[:MAPS_TO_SHARED_DOMAIN]->(d)
        MATCH (g:Grant)-[gr:HAS_DOMAIN_KEYWORD]->(gk)
        WHERE coalesce(gr.weight, 0.0) > $min_domain_weight

        WITH
            g, d, fr, gr,
            toLower(coalesce(g.opportunity_status, '')) AS status_token,
            coalesce(toString(g.close_date), '') AS close_token
        WITH
            g, d, fr, gr, status_token,
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

        WITH
            g,
            collect({
                domain: coalesce(d.value, d.value_norm, ''),
                domain_norm: coalesce(d.value_norm, toLower(coalesce(d.value, ''))),
                faculty_domain_weight: coalesce(fr.weight, 0.0),
                grant_domain_weight: coalesce(gr.weight, 0.0),
                pair_weight: CASE
                    WHEN coalesce(fr.weight, 0.0) < coalesce(gr.weight, 0.0) THEN coalesce(fr.weight, 0.0)
                    ELSE coalesce(gr.weight, 0.0)
                END
            }) AS matches,
            sum(
                CASE
                    WHEN coalesce(fr.weight, 0.0) < coalesce(gr.weight, 0.0) THEN coalesce(fr.weight, 0.0)
                    ELSE coalesce(gr.weight, 0.0)
                END
            ) AS rank_score,
            count(DISTINCT d.value_norm) AS shared_domain_count

        RETURN
            g.opportunity_id AS opportunity_id,
            coalesce(g.opportunity_title, g.title, '') AS opportunity_title,
            coalesce(g.agency_name, '') AS agency_name,
            coalesce(g.opportunity_status, '') AS opportunity_status,
            coalesce(toString(g.close_date), '') AS close_date,
            rank_score,
            shared_domain_count,
            matches
        ORDER BY rank_score DESC, shared_domain_count DESC, opportunity_id ASC
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
                "min_domain_weight": safe_min_weight,
                "include_closed": bool(include_closed),
                "limit": safe_top_k,
            },
            database_=settings.database,
        )

        domain_gate_candidates: List[Dict[str, Any]] = []
        candidate_ids: List[str] = []
        domain_rank_index = 0
        for row in records:
            item = dict(row or {})
            matched_domains = list(item.get("matches") or [])
            oid = _clean_text(item.get("opportunity_id"))
            domain_rank_index += 1
            candidate_ids.append(oid)
            domain_gate_candidates.append(
                {
                    "opportunity_id": oid,
                    "opportunity_title": _clean_text(item.get("opportunity_title")),
                    "agency_name": _clean_text(item.get("agency_name")),
                    "opportunity_status": _clean_text(item.get("opportunity_status")),
                    "close_date": _clean_text(item.get("close_date")),
                    "domain_rank": int(domain_rank_index),
                    "domain_rank_score": float(item.get("rank_score") or 0.0),
                    "rank_score": float(item.get("rank_score") or 0.0),
                    "shared_domain_count": int(item.get("shared_domain_count") or 0),
                    "matched_domains": matched_domains,
                }
            )

        coverage_by_opp = _fetch_specialization_coverage_scores(
            driver=driver,
            database=settings.database,
            faculty_id=fid,
            faculty_email=femail,
            candidate_ids=candidate_ids,
        )

    grants: List[Dict[str, Any]] = []
    for cand in domain_gate_candidates:
        oid = _clean_text(cand.get("opportunity_id"))
        cov = dict(coverage_by_opp.get(oid) or {})
        grants.append(
            {
                **cand,
                "specialization_coverage_score": float(cov.get("specialization_coverage_score") or 0.0),
                "rank_score": float(cov.get("specialization_coverage_score") or 0.0),
                "f2g_coverage_weighted_avg": float(cov.get("f2g_coverage_weighted_avg") or 0.0),
                "g2f_coverage_weighted_avg": float(cov.get("g2f_coverage_weighted_avg") or 0.0),
                "f2g_coverage_avg": float(cov.get("f2g_coverage_avg") or 0.0),
                "g2f_coverage_avg": float(cov.get("g2f_coverage_avg") or 0.0),
                "f2g_hit_ratio": float(cov.get("f2g_hit_ratio") or 0.0),
                "g2f_hit_ratio": float(cov.get("g2f_hit_ratio") or 0.0),
            }
        )
    grants.sort(
        key=lambda x: (
            float(x.get("rank_score") or 0.0),
            float(x.get("domain_rank_score") or 0.0),
            int(x.get("shared_domain_count") or 0),
            _clean_text(x.get("opportunity_id")),
        ),
        reverse=True,
    )

    coverage_edges_found = 0
    for grant in grants:
        if float(grant.get("specialization_coverage_score") or 0.0) > 0.0:
            coverage_edges_found += 1

    return {
        "params": {
            "faculty_id": fid,
            "faculty_email": femail,
            "min_domain_weight": safe_min_weight,
            "top_k": safe_top_k,
            "include_closed": bool(include_closed),
            "rule": "stage1 domain gate + stage2 specialization coverage rerank",
            "stage1_rule": "shared domain + both HAS_DOMAIN_KEYWORD edge weights > threshold",
            "stage2_rule": "reorder stage1 candidates by FACULTY_GRANT_SPEC_COVERAGE specialization coverage score",
        },
        "totals": {
            "grants": len(grants),
            "domain_gate_candidates": len(domain_gate_candidates),
            "coverage_edges_found": int(coverage_edges_found),
        },
        "domain_gate_candidates": domain_gate_candidates,
        "grants": grants,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simple faculty-domain-grant retrieval: return grants where both faculty and grant "
            "HAS_DOMAIN_KEYWORD edge weights are above threshold on shared domains."
        )
    )
    parser.add_argument("--faculty-id", type=int, default=0, help="Faculty ID filter.")
    parser.add_argument("--faculty-email", type=str, default="", help="Faculty email filter.")
    parser.add_argument("--min-domain-weight", type=float, default=0.6, help="Both-side domain edge weight threshold.")
    parser.add_argument("--top-k", type=int, default=20, help="Top K grants.")
    parser.add_argument("--include-closed", action="store_true", help="Include closed grants.")
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

    payload = retrieve_grants_by_domain_weight_gate(
        faculty_id=faculty_id,
        faculty_email=_clean_text(args.faculty_email).lower(),
        min_domain_weight=float(args.min_domain_weight),
        top_k=int(args.top_k or 20),
        include_closed=bool(args.include_closed),
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    if not args.json_only:
        print("Simple domain-weight grant retrieval complete.")
        print(f"  grants returned : {payload.get('totals', {}).get('grants', 0)}")
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
