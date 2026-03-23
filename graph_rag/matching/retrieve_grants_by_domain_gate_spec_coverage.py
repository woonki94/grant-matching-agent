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


def _spec_key(value: Any, section: Any) -> str:
    return f"{_clean_text(value).lower()}||{_clean_text(section).lower() or 'general'}"


def _weighted_jaccard(a: Dict[str, float], b: Dict[str, float]) -> float:
    left = {str(k): _safe_unit_float(v, default=0.0) for k, v in dict(a or {}).items() if _clean_text(k)}
    right = {str(k): _safe_unit_float(v, default=0.0) for k, v in dict(b or {}).items() if _clean_text(k)}
    if not left or not right:
        return 0.0
    keys = set(left.keys()) | set(right.keys())
    inter = 0.0
    union = 0.0
    for k in keys:
        lv = float(left.get(k, 0.0))
        rv = float(right.get(k, 0.0))
        inter += min(lv, rv)
        union += max(lv, rv)
    if union <= 0.0:
        return 0.0
    return _safe_unit_float(inter / union, default=0.0)


def _fetch_domain_gate_candidates(
    *,
    driver,
    database: str,
    faculty_id: Optional[int],
    faculty_email: str,
    min_domain_weight: float,
    include_closed: bool,
    candidate_limit: int,
) -> List[Dict[str, Any]]:
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
            }) AS matched_domains,
            sum(
                CASE
                    WHEN coalesce(fr.weight, 0.0) < coalesce(gr.weight, 0.0) THEN coalesce(fr.weight, 0.0)
                    ELSE coalesce(gr.weight, 0.0)
                END
            ) AS domain_rank_score,
            count(DISTINCT d.value_norm) AS shared_domain_count

        RETURN
            toString(g.opportunity_id) AS opportunity_id,
            coalesce(g.opportunity_title, g.title, '') AS opportunity_title,
            coalesce(g.agency_name, '') AS agency_name,
            coalesce(g.opportunity_status, '') AS opportunity_status,
            coalesce(toString(g.close_date), '') AS close_date,
            domain_rank_score,
            shared_domain_count,
            matched_domains
        ORDER BY domain_rank_score DESC, shared_domain_count DESC, opportunity_id ASC
        LIMIT $candidate_limit
    """
    records, _, _ = driver.execute_query(
        query,
        parameters_={
            "faculty_id": faculty_id,
            "faculty_email": _clean_text(faculty_email).lower(),
            "min_domain_weight": float(min_domain_weight),
            "include_closed": bool(include_closed),
            "candidate_limit": int(candidate_limit),
        },
        database_=database,
    )
    out: List[Dict[str, Any]] = []
    rank_idx = 0
    for row in records:
        item = dict(row or {})
        oid = _clean_text(item.get("opportunity_id"))
        if not oid:
            continue
        rank_idx += 1
        out.append(
            {
                "opportunity_id": oid,
                "opportunity_title": _clean_text(item.get("opportunity_title")),
                "agency_name": _clean_text(item.get("agency_name")),
                "opportunity_status": _clean_text(item.get("opportunity_status")),
                "close_date": _clean_text(item.get("close_date")),
                "domain_rank": int(rank_idx),
                "domain_rank_score": float(item.get("domain_rank_score") or 0.0),
                "shared_domain_count": int(item.get("shared_domain_count") or 0),
                "matched_domains": list(item.get("matched_domains") or []),
            }
        )
    return out


def _fetch_spec_coverage_for_candidates(
    *,
    driver,
    database: str,
    faculty_id: Optional[int],
    faculty_email: str,
    candidate_ids: List[str],
    min_pair_similarity: float,
    pairs_per_spec: int,
) -> Dict[str, Dict[str, Any]]:
    if not candidate_ids:
        return {}

    safe_pairs_per_spec = _safe_limit(pairs_per_spec, default=5, minimum=1, maximum=100)
    safe_min_sim = _safe_unit_float(min_pair_similarity, default=0.4)
    candidate_ids = [_clean_text(x) for x in candidate_ids if _clean_text(x)]
    if not candidate_ids:
        return {}

    fac_records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty)
        WHERE
            ($faculty_id IS NULL OR f.faculty_id = $faculty_id)
            AND ($faculty_email = '' OR toLower(f.email) = $faculty_email)
        RETURN f.faculty_id AS faculty_id, toLower(f.email) AS email
        LIMIT 1
        """,
        parameters_={
            "faculty_id": faculty_id,
            "faculty_email": _clean_text(faculty_email).lower(),
        },
        database_=database,
    )
    if not fac_records:
        return {}
    fac_row = dict(fac_records[0] or {})
    resolved_faculty_id = int(fac_row.get("faculty_id") or 0)

    faculty_specs: Dict[str, Dict[str, Any]] = {}
    fac_spec_records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty {faculty_id: $faculty_id})-[hs]->(fs:FacultyKeyword {bucket: 'specialization'})
        WHERE type(hs) IN ['HAS_SPECIALIZATION_KEYWORD','HAS_RESEARCH_SPECIALIZATION','HAS_APPLICATION_SPECIALIZATION']
          AND fs.value IS NOT NULL
        OPTIONAL MATCH (f)-[hd:HAS_DOMAIN_KEYWORD]->(fd:FacultyKeyword {bucket: 'domain'})-[ds:FACULTY_DOMAIN_HAS_SPECIALIZATION]->(fs)
        WHERE ds.scope_faculty_id = $faculty_id
        OPTIONAL MATCH (fd)-[:MAPS_TO_SHARED_DOMAIN]->(sd:DomainKeywordShared {bucket: 'domain'})
        RETURN
            fs.value AS spec_value,
            toLower(coalesce(fs.section, 'general')) AS spec_section,
            coalesce(hs.weight, 0.0) AS spec_weight,
            collect(DISTINCT {
                domain_norm: coalesce(sd.value_norm, toLower(coalesce(fd.value, ''))),
                domain_weight: coalesce(ds.domain_weight, hd.weight, 0.0)
            }) AS domain_links
        """,
        parameters_={"faculty_id": resolved_faculty_id},
        database_=database,
    )
    for raw in fac_spec_records:
        row = dict(raw or {})
        spec_value = _clean_text(row.get("spec_value"))
        spec_section = _clean_text(row.get("spec_section")).lower() or "general"
        key = _spec_key(spec_value, spec_section)
        if not spec_value:
            continue
        domain_weights: Dict[str, float] = {}
        for link in list(row.get("domain_links") or []):
            if not isinstance(link, dict):
                continue
            dom = _clean_text(link.get("domain_norm")).lower()
            if not dom:
                continue
            w = _safe_unit_float(link.get("domain_weight"), default=0.0)
            if w > float(domain_weights.get(dom, 0.0)):
                domain_weights[dom] = w
        faculty_specs[key] = {
            "spec_value": spec_value,
            "spec_section": spec_section,
            "spec_weight": _safe_unit_float(row.get("spec_weight"), default=0.0),
            "domain_weights": domain_weights,
        }

    grant_specs_by_opp: Dict[str, Dict[str, Dict[str, Any]]] = {}
    grant_spec_records, _, _ = driver.execute_query(
        """
        UNWIND $candidate_ids AS oid
        MATCH (g:Grant)
        WHERE toString(g.opportunity_id) = oid
        MATCH (g)-[hs]->(gs:GrantKeyword {bucket: 'specialization'})
        WHERE type(hs) IN ['HAS_SPECIALIZATION_KEYWORD','HAS_RESEARCH_SPECIALIZATION','HAS_APPLICATION_SPECIALIZATION']
          AND gs.value IS NOT NULL
        OPTIONAL MATCH (g)-[hd:HAS_DOMAIN_KEYWORD]->(gd:GrantKeyword {bucket: 'domain'})-[ds:GRANT_DOMAIN_HAS_SPECIALIZATION]->(gs)
        WHERE toString(ds.scope_opportunity_id) = oid
        OPTIONAL MATCH (gd)-[:MAPS_TO_SHARED_DOMAIN]->(sd:DomainKeywordShared {bucket: 'domain'})
        RETURN
            oid AS opportunity_id,
            gs.value AS spec_value,
            toLower(coalesce(gs.section, 'general')) AS spec_section,
            coalesce(hs.weight, 0.0) AS spec_weight,
            collect(DISTINCT {
                domain_norm: coalesce(sd.value_norm, toLower(coalesce(gd.value, ''))),
                domain_weight: coalesce(ds.domain_weight, hd.weight, 0.0)
            }) AS domain_links
        """,
        parameters_={"candidate_ids": candidate_ids},
        database_=database,
    )
    for raw in grant_spec_records:
        row = dict(raw or {})
        oid = _clean_text(row.get("opportunity_id"))
        spec_value = _clean_text(row.get("spec_value"))
        spec_section = _clean_text(row.get("spec_section")).lower() or "general"
        if not oid or not spec_value:
            continue
        domain_weights: Dict[str, float] = {}
        for link in list(row.get("domain_links") or []):
            if not isinstance(link, dict):
                continue
            dom = _clean_text(link.get("domain_norm")).lower()
            if not dom:
                continue
            w = _safe_unit_float(link.get("domain_weight"), default=0.0)
            if w > float(domain_weights.get(dom, 0.0)):
                domain_weights[dom] = w
        grant_specs_by_opp.setdefault(oid, {})[_spec_key(spec_value, spec_section)] = {
            "spec_value": spec_value,
            "spec_section": spec_section,
            "spec_weight": _safe_unit_float(row.get("spec_weight"), default=0.0),
            "domain_weights": domain_weights,
        }

    pairs_by_opp_spec: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    pair_records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty {faculty_id: $faculty_id})
        UNWIND $candidate_ids AS oid
        MATCH (fk:FacultyKeyword)-[m:FACULTY_SPEC_MATCHES_GRANT_SPEC]->(gs:GrantKeyword {bucket: 'specialization'})
        WHERE
            m.scope_faculty_id = f.faculty_id
            AND toString(m.scope_opportunity_id) = oid
            AND coalesce(m.model_score, m.cosine_sim, m.score, 0.0) >= $min_pair_similarity
        RETURN
            oid AS opportunity_id,
            gs.value AS grant_keyword_value,
            toLower(coalesce(gs.section, 'general')) AS grant_keyword_section,
            coalesce(m.faculty_keyword_value, '') AS faculty_keyword_value,
            toLower(coalesce(m.faculty_keyword_section, 'general')) AS faculty_keyword_section,
            coalesce(m.model_score, m.cosine_sim, m.score, 0.0) AS similarity,
            coalesce(m.grant_keyword_weight, 0.0) AS grant_keyword_weight,
            coalesce(m.faculty_keyword_weight, 0.0) AS faculty_keyword_weight,
            coalesce(m.score, 0.0) AS edge_score
        """,
        parameters_={
            "faculty_id": resolved_faculty_id,
            "candidate_ids": candidate_ids,
            "min_pair_similarity": float(safe_min_sim),
        },
        database_=database,
    )
    for raw in pair_records:
        row = dict(raw or {})
        oid = _clean_text(row.get("opportunity_id"))
        grant_key = _spec_key(row.get("grant_keyword_value"), row.get("grant_keyword_section"))
        if not oid or not grant_key:
            continue
        pairs_by_opp_spec.setdefault(oid, {}).setdefault(grant_key, []).append(
            {
                "faculty_keyword_value": _clean_text(row.get("faculty_keyword_value")),
                "faculty_keyword_section": _clean_text(row.get("faculty_keyword_section")).lower() or "general",
                "similarity": _safe_unit_float(row.get("similarity"), default=0.0),
                "faculty_keyword_weight": _safe_unit_float(row.get("faculty_keyword_weight"), default=0.0),
                "grant_keyword_weight": _safe_unit_float(row.get("grant_keyword_weight"), default=0.0),
                "edge_score": _safe_unit_float(row.get("edge_score"), default=0.0),
            }
        )

    out: Dict[str, Dict[str, Any]] = {}
    for oid in candidate_ids:
        grant_specs = dict(grant_specs_by_opp.get(oid) or {})
        total_specs = len(grant_specs)
        total_grant_weight = 0.0
        covered_specs = 0
        coverage_sum = 0.0
        matched_edges = 0
        details: List[Dict[str, Any]] = []

        for grant_key, grant_spec in grant_specs.items():
            grant_weight = _safe_unit_float(grant_spec.get("spec_weight"), default=0.0)
            total_grant_weight += grant_weight
            grant_domains = dict(grant_spec.get("domain_weights") or {})

            pairs = list((pairs_by_opp_spec.get(oid) or {}).get(grant_key) or [])
            matched_edges += len(pairs)

            best_score = 0.0
            spec_score_sum = 0.0
            best_pair: Dict[str, Any] = {}
            ranked_pairs: List[Dict[str, Any]] = []

            for pair in pairs:
                f_key = _spec_key(pair.get("faculty_keyword_value"), pair.get("faculty_keyword_section"))
                fac_spec = dict(faculty_specs.get(f_key) or {})
                fac_domains = dict(fac_spec.get("domain_weights") or {})
                fac_weight = _safe_unit_float(
                    fac_spec.get("spec_weight"),
                    default=_safe_unit_float(pair.get("faculty_keyword_weight"), default=0.0),
                )

                sim = _safe_unit_float(pair.get("similarity"), default=0.0)
                domain_overlap = _weighted_jaccard(grant_domains, fac_domains)
                pair_score = sim * (domain_overlap ** 2)

                row = {
                    "faculty_keyword_value": _clean_text(pair.get("faculty_keyword_value")),
                    "faculty_keyword_section": _clean_text(pair.get("faculty_keyword_section")).lower() or "general",
                    "similarity": sim,
                    "domain_overlap": domain_overlap,
                    "grant_keyword_weight": grant_weight,
                    "faculty_keyword_weight": fac_weight,
                    "pair_score": float(pair_score),
                }
                ranked_pairs.append(row)
                spec_score_sum += float(pair_score)
                if pair_score > best_score:
                    best_score = float(pair_score)
                    best_pair = row

            if spec_score_sum > 0.0:
                covered_specs += 1
            coverage_sum += float(spec_score_sum)

            ranked_pairs.sort(
                key=lambda x: (
                    float(x.get("pair_score") or 0.0),
                    float(x.get("similarity") or 0.0),
                    float(x.get("domain_overlap") or 0.0),
                ),
                reverse=True,
            )
            details.append(
                {
                    "grant_keyword_value": _clean_text(grant_spec.get("spec_value")),
                    "grant_keyword_section": _clean_text(grant_spec.get("spec_section")).lower() or "general",
                    "grant_spec_weight": float(grant_weight),
                    "matched_faculty_spec_count": int(len(pairs)),
                    "spec_score_sum": float(spec_score_sum),
                    "best_pair_score": float(best_score),
                    "best_pair": best_pair,
                    "matched_faculty_specs": ranked_pairs[:safe_pairs_per_spec],
                }
            )

        coverage_ratio = 0.0 if total_specs <= 0 else float(covered_specs) / float(total_specs)
        normalized_coverage = 0.0 if total_grant_weight <= 0.0 else float(coverage_sum) / float(total_grant_weight)
        details.sort(
            key=lambda x: (
                float((x or {}).get("best_pair_score") or 0.0),
                int((x or {}).get("matched_faculty_spec_count") or 0),
                _clean_text((x or {}).get("grant_keyword_value")),
            ),
            reverse=True,
        )

        out[oid] = {
            "total_grant_spec_keywords": int(total_specs),
            "covered_grant_spec_keywords": int(covered_specs),
            "grant_spec_coverage_ratio": float(coverage_ratio),
            "total_grant_spec_weight": float(total_grant_weight),
            "grant_spec_coverage_sum": float(coverage_sum),
            "normalized_coverage": float(normalized_coverage),
            "matched_faculty_spec_edges": int(matched_edges),
            "grant_spec_coverage_details": details,
        }
    return out


def retrieve_grants_by_domain_gate_spec_coverage(
    *,
    faculty_id: Optional[int],
    faculty_email: str,
    min_domain_weight: float,
    candidate_limit: int,
    top_k: int,
    include_closed: bool,
    min_pair_similarity: float,
    pairs_per_spec: int,
    uri: str = "",
    username: str = "",
    password: str = "",
    database: str = "",
) -> Dict[str, Any]:
    fid = int(faculty_id) if faculty_id is not None else None
    femail = _clean_text(faculty_email).lower()
    if fid is None and not femail:
        raise ValueError("Provide faculty_id or faculty_email.")

    safe_min_domain_weight = _safe_unit_float(min_domain_weight, default=0.6)
    safe_candidate_limit = _safe_limit(candidate_limit, default=20, minimum=1, maximum=2000)
    safe_top_k = _safe_limit(top_k, default=20, minimum=1, maximum=2000)
    safe_min_pair_similarity = _safe_unit_float(min_pair_similarity, default=0.4)
    safe_pairs_per_spec = _safe_limit(pairs_per_spec, default=5, minimum=1, maximum=100)

    load_dotenv_if_present()
    settings = read_neo4j_settings(uri=uri, username=username, password=password, database=database)

    with GraphDatabase.driver(settings.uri, auth=(settings.username, settings.password)) as driver:
        driver.verify_connectivity()
        candidates = _fetch_domain_gate_candidates(
            driver=driver,
            database=settings.database,
            faculty_id=fid,
            faculty_email=femail,
            min_domain_weight=safe_min_domain_weight,
            include_closed=bool(include_closed),
            candidate_limit=safe_candidate_limit,
        )
        candidate_ids = [_clean_text(x.get("opportunity_id")) for x in candidates if _clean_text(x.get("opportunity_id"))]
        spec_cov_by_opp = _fetch_spec_coverage_for_candidates(
            driver=driver,
            database=settings.database,
            faculty_id=fid,
            faculty_email=femail,
            candidate_ids=candidate_ids,
            min_pair_similarity=safe_min_pair_similarity,
            pairs_per_spec=safe_pairs_per_spec,
        )

    grants: List[Dict[str, Any]] = []
    for cand in candidates:
        oid = _clean_text(cand.get("opportunity_id"))
        cov = dict(spec_cov_by_opp.get(oid) or {})
        total_specs = int(cov.get("total_grant_spec_keywords") or 0)
        covered_specs = int(cov.get("covered_grant_spec_keywords") or 0)
        ratio = 0.0 if total_specs <= 0 else float(covered_specs) / float(total_specs)
        coverage_sum = float(cov.get("grant_spec_coverage_sum") or 0.0)
        domain_rank_score = _safe_float(cand.get("domain_rank_score"), default=0.0)
        final_score = coverage_sum * domain_rank_score

        grants.append(
            {
                **cand,
                "total_grant_spec_keywords": total_specs,
                "covered_grant_spec_keywords": covered_specs,
                "grant_spec_coverage_ratio": float(cov.get("grant_spec_coverage_ratio") or ratio),
                "grant_spec_coverage_sum": coverage_sum,
                "normalized_coverage": float(cov.get("normalized_coverage") or 0.0),
                "total_grant_spec_weight": float(cov.get("total_grant_spec_weight") or 0.0),
                "matched_faculty_spec_edges": int(cov.get("matched_faculty_spec_edges") or 0),
                "grant_spec_coverage_details": list(cov.get("grant_spec_coverage_details") or []),
                # score = sim * (domain_overlap^2)
                # coverage_sum = sum_i sum_j(score)
                # final_score = coverage_sum * domain_rank_score
                "final_score": final_score,
                "rank_score": final_score,
            }
        )

    grants.sort(
        key=lambda x: (
            float(x.get("rank_score") or 0.0),
            float(x.get("grant_spec_coverage_sum") or 0.0),
            float(x.get("grant_spec_coverage_ratio") or 0.0),
            int(x.get("covered_grant_spec_keywords") or 0),
            float(x.get("total_grant_spec_weight") or 0.0),
            float(x.get("domain_rank_score") or 0.0),
            int(x.get("shared_domain_count") or 0),
            _clean_text(x.get("opportunity_id")),
        ),
        reverse=True,
    )
    grants = grants[:safe_top_k]

    return {
        "params": {
            "faculty_id": fid,
            "faculty_email": femail,
            "min_domain_weight": safe_min_domain_weight,
            "candidate_limit": safe_candidate_limit,
            "top_k": safe_top_k,
            "include_closed": bool(include_closed),
            "min_pair_similarity": safe_min_pair_similarity,
            "pairs_per_spec": safe_pairs_per_spec,
            "rule": "stage1 domain gate candidates + stage2 grant-spec coverage over FACULTY_SPEC_MATCHES_GRANT_SPEC edges",
            "pair_rule": "score = similarity * (domain_overlap^2)",
            "coverage_rule": "coverage_sum = sum_i sum_j(score)",
            "final_score_rule": "final_score = coverage_sum * domain_rank_score",
            "rank_rule": "rank by final_score, then coverage_sum, then coverage ratio/count",
        },
        "totals": {
            "domain_gate_candidates": len(candidates),
            "grants": len(grants),
            "covered_grants": sum(1 for g in grants if int(g.get("covered_grant_spec_keywords") or 0) > 0),
        },
        "domain_gate_candidates": candidates,
        "grants": grants,
        "top10_titles": [
            {
                "opportunity_id": _clean_text(g.get("opportunity_id")),
                "opportunity_title": _clean_text(g.get("opportunity_title")),
                "covered_grant_spec_keywords": int(g.get("covered_grant_spec_keywords") or 0),
                "final_score": float(g.get("final_score") or 0.0),
                "grant_spec_coverage_sum": float(g.get("grant_spec_coverage_sum") or 0.0),
                "grant_spec_coverage_ratio": float(g.get("grant_spec_coverage_ratio") or 0.0),
                "domain_rank_score": float(g.get("domain_rank_score") or 0.0),
            }
            for g in grants[:10]
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve grants with stage1 domain gate and stage2 specialization coverage. "
            "Grant specialization keywords are the coverage target."
        )
    )
    parser.add_argument("--faculty-id", type=int, default=103, help="Faculty ID filter.")
    parser.add_argument("--faculty-email", type=str, default="", help="Faculty email filter.")
    parser.add_argument("--min-domain-weight", type=float, default=0.7, help="Stage1 domain gate threshold.")
    parser.add_argument("--candidate-limit", type=int, default=20, help="Stage1 candidate count.")
    parser.add_argument("--top-k", type=int, default=20, help="Final grants to return.")
    parser.add_argument("--include-closed", action="store_true", help="Include closed grants.")
    parser.add_argument("--min-pair-similarity", type=float, default=0.0, help="Minimum similarity on FACULTY_SPEC_MATCHES_GRANT_SPEC edge.")
    parser.add_argument("--pairs-per-spec", type=int, default=10, help="Matched faculty specs returned per grant spec keyword.")
    parser.add_argument("--json-only", action="store_true", help="Print only JSON.")

    parser.add_argument("--uri", type=str, default="", help="Neo4j URI. Fallback: NEO4J_URI")
    parser.add_argument("--username", type=str, default="", help="Neo4j username. Fallback: NEO4J_USERNAME")
    parser.add_argument("--password", type=str, default="", help="Neo4j password. Fallback: NEO4J_PASSWORD")
    parser.add_argument("--database", type=str, default="", help="Neo4j database. Fallback: NEO4J_DATABASE or neo4j")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    fid = int(args.faculty_id or 0)
    faculty_id = fid if fid > 0 else None

    payload = retrieve_grants_by_domain_gate_spec_coverage(
        faculty_id=faculty_id,
        faculty_email=_clean_text(args.faculty_email).lower(),
        min_domain_weight=float(args.min_domain_weight),
        candidate_limit=int(args.candidate_limit or 20),
        top_k=int(args.top_k or 20),
        include_closed=bool(args.include_closed),
        min_pair_similarity=float(args.min_pair_similarity),
        pairs_per_spec=int(args.pairs_per_spec or 5),
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    if not args.json_only:
        totals = payload.get("totals", {})
        print("Domain-gate + spec-coverage retrieval complete.")
        print(f"  domain candidates : {totals.get('domain_gate_candidates', 0)}")
        print(f"  grants returned   : {totals.get('grants', 0)}")
        print(f"  covered grants    : {totals.get('covered_grants', 0)}")
        print()
        print("Top 10 grants:")
        for idx, item in enumerate(list(payload.get("top10_titles") or [])[:10], start=1):
            title = _clean_text(item.get("opportunity_title"))
            oid = _clean_text(item.get("opportunity_id"))
            covered = int(item.get("covered_grant_spec_keywords") or 0)
            final_score = _safe_float(item.get("final_score"), default=0.0)
            coverage_sum = _safe_float(item.get("grant_spec_coverage_sum"), default=0.0)
            coverage_ratio = _safe_float(item.get("grant_spec_coverage_ratio"), default=0.0)
            domain_score = _safe_float(item.get("domain_rank_score"), default=0.0)
            print(
                f"{idx:>2}. {title} ({oid}) | final_score={final_score:.6f} "
                f"| coverage_sum={coverage_sum:.6f} | coverage_ratio={coverage_ratio:.4f} "
                f"| domain_rank_score={domain_score:.6f} | covered_spec={covered}"
            )
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
