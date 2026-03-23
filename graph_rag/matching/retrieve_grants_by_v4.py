from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from neo4j import GraphDatabase

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from graph_rag.common import json_ready, load_dotenv_if_present, read_neo4j_settings

DOMAIN_MATCH_MIN_COSINE = 0.0
TOP_DOMAIN_ADVANTAGE_MULT = 10.0


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
    if math.isnan(parsed) or math.isinf(parsed):
        return float(default)
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _safe_nonneg_float(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if math.isnan(parsed) or math.isinf(parsed):
        return float(default)
    if parsed < 0.0:
        return 0.0
    return parsed


def _coerce_vector(value: Any) -> List[float]:
    if not isinstance(value, (list, tuple)):
        return []
    out: List[float] = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            return []
    return out


def _cosine_vec(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for idx in range(len(a)):
        x = float(a[idx])
        y = float(b[idx])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    sim = dot / ((math.sqrt(na) * math.sqrt(nb)) + 1e-9)
    return _safe_unit_float(sim, default=0.0)


def _parse_domain_links(items: Any, *, spec_weight: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    safe_spec_weight = _safe_unit_float(spec_weight, default=0.0)
    max_keyword_weight = 0.0

    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        name = _clean_text(item.get("domain")).lower()
        if not name:
            continue

        domain_weight = _safe_unit_float(item.get("domain_weight"), default=1.0)
        keyword_weight = _safe_unit_float(item.get("keyword_weight"), default=0.5)
        embedding = _coerce_vector(item.get("embedding"))
        rows.append(
            {
                "domain": name,
                "domain_weight": domain_weight,
                "keyword_weight": keyword_weight,
                "spec_weight": safe_spec_weight,
                "embedding": embedding,
            }
        )
        if keyword_weight > max_keyword_weight:
            max_keyword_weight = keyword_weight

    by_name: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("domain") or "")
        domain_weight = float(row.get("domain_weight") or 0.0)
        keyword_weight = float(row.get("keyword_weight") or 0.0)
        embedding = list(row.get("embedding") or [])

        # Strongly favor only top HAS_DOMAIN_KEYWORD weights.
        is_top = abs(keyword_weight - max_keyword_weight) <= 1e-9
        if is_top:
            side_score = domain_weight * safe_spec_weight * float(TOP_DOMAIN_ADVANTAGE_MULT)
        else:
            side_score = 0.0

        prev = by_name.get(name)
        if prev is None:
            by_name[name] = {
                "domain": name,
                "domain_weight": domain_weight,
                "keyword_weight": keyword_weight,
                "spec_weight": safe_spec_weight,
                "side_score": side_score,
                "is_top_keyword_weight": bool(is_top),
                "embedding": embedding,
            }
            continue

        if side_score > float(prev.get("side_score") or 0.0):
            prev["domain_weight"] = domain_weight
            prev["keyword_weight"] = keyword_weight
            prev["side_score"] = side_score
            prev["is_top_keyword_weight"] = bool(is_top)
        if not prev.get("embedding") and embedding:
            prev["embedding"] = embedding

    return list(by_name.values())


def _compute_domain_similarity(
    *,
    faculty_domains: Sequence[Dict[str, Any]],
    grant_domains: Sequence[Dict[str, Any]],
    min_cosine: float = DOMAIN_MATCH_MIN_COSINE,
) -> Tuple[float, List[Dict[str, Any]], List[str], List[str]]:
    fac_list = list(faculty_domains or [])
    grant_list = list(grant_domains or [])
    if not fac_list or not grant_list:
        return 0.0, [], [str(_clean_text(x.get("domain")).lower()) for x in fac_list], [
            str(_clean_text(x.get("domain")).lower()) for x in grant_list
        ]

    threshold = _safe_unit_float(min_cosine, default=DOMAIN_MATCH_MIN_COSINE)

    matched_fac: set[int] = set()
    matched_grant: set[int] = set()
    matched_pairs: List[Dict[str, Any]] = []
    domain_sim = 0.0

    # Requested rule over all pairs:
    # if cosine >= 0.5 add (fac_domain_weight*fac_spec_weight + grant_domain_weight*grant_spec_weight)
    # else subtract the same term.
    for fi, f_row in enumerate(fac_list):
        f_vec = list(f_row.get("embedding") or [])
        f_side = _safe_unit_float(f_row.get("side_score"), default=0.0)
        f_name = _clean_text(f_row.get("domain")).lower()
        for gi, g_row in enumerate(grant_list):
            g_vec = list(g_row.get("embedding") or [])
            g_side = _safe_unit_float(g_row.get("side_score"), default=0.0)
            g_name = _clean_text(g_row.get("domain")).lower()

            sim = _cosine_vec(f_vec, g_vec)
            pair_term = f_side + g_side
            if sim >= threshold:
                domain_sim += pair_term
                matched_fac.add(fi)
                matched_grant.add(gi)
                matched_pairs.append(
                    {
                        "faculty_domain": f_name,
                        "grant_domain": g_name,
                        "cosine_sim": _safe_unit_float(sim, default=0.0),
                        "faculty_side": f_side,
                        "grant_side": g_side,
                        "pair_term": pair_term,
                    }
                )
            else:
                domain_sim -= pair_term

    faculty_unmatched = [
        _clean_text(fac_list[i].get("domain")).lower()
        for i in range(len(fac_list))
        if i not in matched_fac
    ]
    grant_unmatched = [
        _clean_text(grant_list[i].get("domain")).lower()
        for i in range(len(grant_list))
        if i not in matched_grant
    ]

    if len(matched_pairs) > 100:
        matched_pairs = matched_pairs[:100]

    return domain_sim, matched_pairs, sorted(set(faculty_unmatched)), sorted(set(grant_unmatched))


def _fetch_v4_pairs(
    *,
    driver,
    database: str,
    faculty_id: Optional[int],
    faculty_email: str,
    include_closed: bool,
) -> List[Dict[str, Any]]:
    records, _, _ = driver.execute_query(
        """
        MATCH (fk:FacultyKeyword)-[m:FACULTY_SPEC_MATCHES_GRANT_SPEC_V3]->(gk:GrantKeyword)
        WHERE
            ($faculty_id IS NULL OR m.scope_faculty_id = $faculty_id)
            AND ($faculty_email = '' OR toLower(m.scope_faculty_email) = $faculty_email)
            AND m.scope_opportunity_id IS NOT NULL
        MATCH (g:Grant {opportunity_id: m.scope_opportunity_id})
        WITH
            fk, gk, m, g,
            toLower(coalesce(g.opportunity_status, '')) AS status_token,
            coalesce(toString(g.close_date), '') AS close_token
        WITH
            fk, gk, m, g, status_token,
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
        CALL (fk, m) {
            MATCH (f:Faculty {faculty_id: m.scope_faculty_id})
            OPTIONAL MATCH (f)-[hk:HAS_DOMAIN_KEYWORD]->(fd:FacultyKeyword {bucket: 'domain'})-[r:FACULTY_DOMAIN_HAS_SPECIALIZATION]->(fk)
            WHERE r.scope_faculty_id = m.scope_faculty_id
            RETURN collect(DISTINCT {
                domain: toLower(coalesce(fd.value, '')),
                score: coalesce(r.score, 0.0),
                domain_weight: coalesce(r.domain_weight, 1.0),
                keyword_weight: coalesce(hk.weight, 0.5),
                embedding: fd.embedding
            }) AS faculty_domain_links
        }
        CALL (gk, m, g) {
            OPTIONAL MATCH (g)-[hk:HAS_DOMAIN_KEYWORD]->(gd:GrantKeyword {bucket: 'domain'})-[r:GRANT_DOMAIN_HAS_SPECIALIZATION]->(gk)
            WHERE r.scope_opportunity_id = m.scope_opportunity_id
            RETURN collect(DISTINCT {
                domain: toLower(coalesce(gd.value, '')),
                score: coalesce(r.score, 0.0),
                domain_weight: coalesce(r.domain_weight, 1.0),
                keyword_weight: coalesce(hk.weight, 0.5),
                embedding: gd.embedding
            }) AS grant_domain_links
        }
        RETURN
            m.scope_opportunity_id AS opportunity_id,
            coalesce(g.opportunity_title, g.title, '') AS opportunity_title,
            coalesce(g.agency_name, '') AS agency_name,
            coalesce(g.opportunity_status, '') AS opportunity_status,
            coalesce(toString(g.close_date), '') AS close_date,
            coalesce(m.faculty_keyword_value, fk.value, '') AS faculty_keyword_value,
            coalesce(m.faculty_keyword_section, fk.section, 'general') AS faculty_keyword_section,
            coalesce(m.grant_keyword_value, gk.value, '') AS grant_keyword_value,
            coalesce(m.grant_keyword_section, gk.section, 'general') AS grant_keyword_section,
            coalesce(m.cosine_sim, 0.0) AS cosine_sim,
            coalesce(m.attention_score, 0.0) AS attention_score,
            coalesce(m.faculty_keyword_weight, 0.0) AS faculty_keyword_weight,
            coalesce(m.grant_keyword_weight, 0.0) AS grant_keyword_weight,
            coalesce(m.domain_gate_score, 0.0) AS domain_gate_score,
            faculty_domain_links,
            grant_domain_links
        ORDER BY opportunity_id ASC
        """,
        parameters_={
            "faculty_id": int(faculty_id) if faculty_id is not None else None,
            "faculty_email": _clean_text(faculty_email).lower(),
            "include_closed": bool(include_closed),
        },
        database_=database,
    )
    return [dict(row or {}) for row in records]


def retrieve_grants_by_v4(
    *,
    faculty_id: Optional[int],
    faculty_email: str,
    top_k: int,
    top_pairs_for_rank: int,
    pairs_per_grant: int,
    cosine_weight: float,
    cross_weight: float,
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
    safe_top_pairs_for_rank = _safe_limit(top_pairs_for_rank, default=5, minimum=1, maximum=50)
    safe_pairs_per_grant = _safe_limit(pairs_per_grant, default=20, minimum=1, maximum=500)

    cw = _safe_nonneg_float(cosine_weight, default=0.5)
    xw = _safe_nonneg_float(cross_weight, default=0.5)
    weight_sum = cw + xw
    if weight_sum <= 0.0:
        cw = 0.5
        xw = 0.5
        weight_sum = 1.0

    load_dotenv_if_present()
    settings = read_neo4j_settings(
        uri=uri,
        username=username,
        password=password,
        database=database,
    )

    with GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    ) as driver:
        driver.verify_connectivity()
        pair_rows = _fetch_v4_pairs(
            driver=driver,
            database=settings.database,
            faculty_id=fid,
            faculty_email=femail,
            include_closed=bool(include_closed),
        )

    grouped: Dict[str, Dict[str, Any]] = {}

    for row in pair_rows:
        opportunity_id = _clean_text(row.get("opportunity_id"))
        if not opportunity_id:
            continue

        g = grouped.get(opportunity_id)
        if g is None:
            g = {
                "opportunity_id": opportunity_id,
                "opportunity_title": _clean_text(row.get("opportunity_title")),
                "agency_name": _clean_text(row.get("agency_name")),
                "opportunity_status": _clean_text(row.get("opportunity_status")),
                "close_date": _clean_text(row.get("close_date")),
                "pair_rows": [],
            }
            grouped[opportunity_id] = g

        cosine_sim = _safe_unit_float(row.get("cosine_sim"), default=0.0)
        cross_score = _safe_unit_float(row.get("attention_score"), default=0.0)

        fac_spec_weight = _safe_unit_float(row.get("faculty_keyword_weight"), default=0.0)
        grant_spec_weight = _safe_unit_float(row.get("grant_keyword_weight"), default=0.0)

        # Step 1: spec-to-spec similarity blended by cosine/cross, then weighted by both spec weights.
        spec_similarity = ((cw * cosine_sim) + (xw * cross_score)) / weight_sum
        s1 = spec_similarity * fac_spec_weight * grant_spec_weight

        fac_domains = _parse_domain_links(row.get("faculty_domain_links"), spec_weight=fac_spec_weight)
        grant_domains = _parse_domain_links(row.get("grant_domain_links"), spec_weight=grant_spec_weight)

        # Step 2: domain similarity with add/subtract rule by match existence.
        domain_sim, matched_domain_pairs, faculty_unmatched_domains, grant_unmatched_domains = _compute_domain_similarity(
            faculty_domains=fac_domains,
            grant_domains=grant_domains,
            min_cosine=DOMAIN_MATCH_MIN_COSINE,
        )

        pair_score = s1 * domain_sim

        g["pair_rows"].append(
            {
                "faculty_keyword_value": _clean_text(row.get("faculty_keyword_value")),
                "faculty_keyword_section": _clean_text(row.get("faculty_keyword_section")).lower() or "general",
                "grant_keyword_value": _clean_text(row.get("grant_keyword_value")),
                "grant_keyword_section": _clean_text(row.get("grant_keyword_section")).lower() or "general",
                "cosine_sim": cosine_sim,
                "cross_encoder_score": cross_score,
                "spec_similarity": spec_similarity,
                "faculty_spec_weight": fac_spec_weight,
                "grant_spec_weight": grant_spec_weight,
                "s1": s1,
                "domain_similarity": domain_sim,
                "domain_match_min_cosine": DOMAIN_MATCH_MIN_COSINE,
                "matched_domain_pairs": matched_domain_pairs,
                "faculty_unmatched_domains": faculty_unmatched_domains,
                "grant_unmatched_domains": grant_unmatched_domains,
                "pair_score": pair_score,
            }
        )

    grant_rows: List[Dict[str, Any]] = []
    for item in grouped.values():
        pairs_sorted = sorted(
            list(item.get("pair_rows") or []),
            key=lambda x: float(x.get("pair_score") or 0.0),
            reverse=True,
        )
        rank_pairs = pairs_sorted[:safe_top_pairs_for_rank]
        rank_score = float(sum(float(x.get("pair_score") or 0.0) for x in rank_pairs))

        grant_rows.append(
            {
                "opportunity_id": item["opportunity_id"],
                "opportunity_title": item["opportunity_title"],
                "agency_name": item["agency_name"],
                "opportunity_status": item["opportunity_status"],
                "close_date": item["close_date"],
                "rank_score": rank_score,
                "pair_count": len(pairs_sorted),
                "rank_pair_count_used": len(rank_pairs),
                "matched_pairs": pairs_sorted[:safe_pairs_per_grant],
            }
        )

    grant_rows.sort(
        key=lambda x: (
            float(x.get("rank_score") or 0.0),
            int(x.get("pair_count") or 0),
        ),
        reverse=True,
    )
    grant_rows = grant_rows[:safe_top_k]

    return {
        "params": {
            "faculty_id": fid,
            "faculty_email": femail,
            "top_k": safe_top_k,
            "top_pairs_for_rank": safe_top_pairs_for_rank,
            "pairs_per_grant": safe_pairs_per_grant,
            "cosine_weight": cw,
            "cross_weight": xw,
            "domain_match_min_cosine": DOMAIN_MATCH_MIN_COSINE,
            "top_domain_advantage_mult": TOP_DOMAIN_ADVANTAGE_MULT,
            "include_closed": bool(include_closed),
            "pair_score_formula": (
                "spec_similarity = (cosine_weight*cosine_sim + cross_weight*cross_encoder_score)/(cosine_weight+cross_weight); "
                "s1 = spec_similarity * faculty_spec_weight * grant_spec_weight; "
                "top-domain filter: keep only domains whose HAS_DOMAIN_KEYWORD.weight is maximal on each side; "
                "their side term = domain_weight * spec_weight * top_domain_advantage_mult, others side term = 0; "
                "for each faculty_domain x grant_domain pair: "
                "if cosine(domain,domain) >= 0.5 then domain_similarity += (faculty_side + grant_side) else domain_similarity -= (faculty_side + grant_side); "
                "pair_score = s1 * domain_similarity; "
                "grant_rank = sum(top_k(pair_score))"
            ),
        },
        "totals": {
            "pair_rows": len(pair_rows),
            "grants": len(grant_rows),
        },
        "grants": grant_rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve grants using V4 scoring over FACULTY_SPEC_MATCHES_GRANT_SPEC_V3 edges: "
            "(blended spec similarity * weighted domain similarity)."
        )
    )
    parser.add_argument("--faculty-id", type=int, default=103, help="Faculty ID filter.")
    parser.add_argument("--faculty-email", type=str, default="", help="Faculty email filter.")
    parser.add_argument("--top-k", type=int, default=20, help="Top K grants to return.")
    parser.add_argument(
        "--top-pairs-for-rank",
        type=int,
        default=5,
        help="Top N pair scores used when aggregating each grant rank.",
    )
    parser.add_argument(
        "--pairs-per-grant",
        type=int,
        default=20,
        help="Matched pairs included per returned grant.",
    )
    parser.add_argument(
        "--cosine-weight",
        type=float,
        default=0.5,
        help="Blend weight for cosine similarity in spec-to-spec similarity.",
    )
    parser.add_argument(
        "--cross-weight",
        type=float,
        default=0.5,
        help="Blend weight for cross-encoder score in spec-to-spec similarity.",
    )
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

    payload = retrieve_grants_by_v4(
        faculty_id=faculty_id,
        faculty_email=_clean_text(args.faculty_email).lower(),
        top_k=int(args.top_k or 20),
        top_pairs_for_rank=int(args.top_pairs_for_rank or 5),
        pairs_per_grant=int(args.pairs_per_grant or 20),
        cosine_weight=float(args.cosine_weight),
        cross_weight=float(args.cross_weight),
        include_closed=bool(args.include_closed),
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    if not args.json_only:
        print("Grant retrieval v4 complete.")
        print(f"  grants returned : {payload.get('totals', {}).get('grants', 0)}")
        print(f"  pair rows       : {payload.get('totals', {}).get('pair_rows', 0)}")
        print()

    print(json.dumps(json_ready(payload), indent=2))

    if not args.json_only:
        grants = list(payload.get("grants") or [])
        top10 = grants[:10]
        print()
        print("Top 10 Grants")
        if not top10:
            print("  (none)")
        else:
            for idx, row in enumerate(top10, start=1):
                opp_id = _clean_text(row.get("opportunity_id"))
                title = _clean_text(row.get("opportunity_title"))
                score = float(row.get("rank_score") or 0.0)
                print(f"{idx:2d}. {score:.6f} | {opp_id} | {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
