from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from neo4j import GraphDatabase, RoutingControl

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from tmp.neo4j_common import (
    Neo4jSettings,
    json_ready,
    load_dotenv_if_present,
    read_neo4j_settings,
    safe_text,
)

KEYWORD_RELATIONS = [
    "HAS_RESEARCH_DOMAIN",
    "HAS_RESEARCH_SPECIALIZATION",
    "HAS_APPLICATION_DOMAIN",
    "HAS_APPLICATION_SPECIALIZATION",
]
AGENCY_STOPWORDS = {
    "the",
    "of",
    "and",
    "for",
    "to",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
}


def _safe_limit(value: int, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _normalize_terms(raw: str) -> List[str]:
    if not raw:
        return []

    tokens = re.split(r"[,\n]+", raw)
    out: List[str] = []
    seen = set()
    for token in tokens:
        cleaned = str(token or "").strip().lower()
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _normalize_agency_filter(raw: str) -> Tuple[bool, str, str, str]:
    text = str(raw or "").strip().lower()
    if not text:
        return False, "", "", ""

    # Normalize to alphanumeric tokens only.
    normalized = re.sub(r"[^a-z0-9]+", " ", text).strip()
    if not normalized:
        return False, "", "", ""

    tokens_all = [tok for tok in normalized.split(" ") if tok]
    tokens_nostop = [tok for tok in tokens_all if tok not in AGENCY_STOPWORDS]
    compact = "".join(tokens_all)
    initials_all = "".join(tok[0] for tok in tokens_all if tok)
    initials_nostop = "".join(tok[0] for tok in tokens_nostop if tok)

    return True, compact, initials_all, initials_nostop


def retrieve_grants(
    *,
    settings: Neo4jSettings,
    keyword_terms: List[str],
    agency_name: str,
    broad_category: str,
    specific_category: str,
    opportunity_status: str,
    top_k: int,
    additional_info_limit: int,
    attachment_limit: int,
    excerpt_chars: int,
) -> Dict[str, Any]:
    (
        agency_filter_enabled,
        agency_query_compact,
        agency_query_initials_all,
        agency_query_initials_nostop,
    ) = _normalize_agency_filter(agency_name)

    query = """
        MATCH (g:Grant)
        OPTIONAL MATCH (g)-[agency_rel]->(agency:Agency)
        WHERE type(agency_rel) = 'FUNDED_BY'
        WITH
            g,
            agency,
            toLower(
                reduce(
                    s = trim(coalesce(g.agency_name, agency.name, '')),
                    ch IN ['.', ',', '-', '/', '(', ')', '&', '_']
                    | replace(s, ch, ' ')
                )
            ) AS candidate_spaced_raw
        WITH
            g,
            agency,
            [t IN split(candidate_spaced_raw, ' ') WHERE t <> ''] AS candidate_tokens_all
        WITH
            g,
            agency,
            candidate_tokens_all,
            [t IN candidate_tokens_all WHERE NOT t IN $agency_stopwords] AS candidate_tokens_nostop
        WITH
            g,
            agency,
            reduce(acc = '', t IN candidate_tokens_all | acc + t) AS candidate_compact,
            reduce(acc = '', t IN candidate_tokens_nostop | acc + t) AS candidate_compact_nostop,
            reduce(acc = '', t IN candidate_tokens_all | acc + substring(t, 0, 1)) AS candidate_initials_all,
            reduce(acc = '', t IN candidate_tokens_nostop | acc + substring(t, 0, 1)) AS candidate_initials_nostop
        WHERE (
            NOT $agency_filter_enabled
            OR candidate_compact CONTAINS $agency_query_compact
            OR candidate_compact_nostop CONTAINS $agency_query_compact
            OR $agency_query_compact CONTAINS candidate_compact
            OR $agency_query_compact CONTAINS candidate_compact_nostop
            OR candidate_initials_all CONTAINS $agency_query_compact
            OR candidate_initials_nostop CONTAINS $agency_query_compact
            OR ($agency_query_initials_all <> '' AND candidate_initials_all CONTAINS $agency_query_initials_all)
            OR (
                $agency_query_initials_nostop <> ''
                AND candidate_initials_nostop CONTAINS $agency_query_initials_nostop
            )
        )
        AND (
            $opportunity_status = ''
            OR toLower(coalesce(g.opportunity_status, '')) = toLower($opportunity_status)
        )

        CALL (g) {
            OPTIONAL MATCH (g)-[broad_rel]->(bc:GrantBroadCategory)
            WHERE type(broad_rel) = 'HAS_BROAD_CATEGORY'
            RETURN collect(DISTINCT bc.name) AS broad_categories
        }
        WITH g, agency, broad_categories
        WHERE (
            $broad_category = ''
            OR any(x IN broad_categories WHERE toLower(x) = toLower($broad_category))
        )

        CALL (g) {
            OPTIONAL MATCH (g)-[specific_rel]->(sc:GrantSpecificCategory)
            WHERE type(specific_rel) = 'HAS_SPECIFIC_CATEGORY'
            RETURN collect(DISTINCT sc.name) AS specific_categories
        }
        WITH g, agency, broad_categories, specific_categories
        WHERE (
            $specific_category = ''
            OR any(x IN specific_categories WHERE toLower(x) = toLower($specific_category))
        )

        CALL (g) {
            WITH g, $keyword_terms AS terms
            OPTIONAL MATCH (g)-[r]->(k:GrantKeyword)
            WHERE type(r) IN $relations
              AND (
                size(terms) = 0
                OR any(term IN terms WHERE toLower(k.value) CONTAINS term OR term CONTAINS toLower(k.value))
              )
            RETURN
              [x IN collect(DISTINCT CASE
                WHEN k IS NULL THEN NULL
                ELSE {
                    value: k.value,
                    section: k.section,
                    bucket: k.bucket,
                    relation: type(r),
                    weight: r.weight
                }
              END) WHERE x IS NOT NULL] AS keyword_matches,
              sum(CASE WHEN k IS NULL THEN 0.0 ELSE coalesce(r.weight, 0.5) END) AS keyword_score,
              count(DISTINCT k) AS keyword_hits
        }
        WITH g, agency, broad_categories, specific_categories, keyword_matches, keyword_score, keyword_hits
        WHERE size($keyword_terms) = 0 OR keyword_hits > 0

        CALL (g) {
            OPTIONAL MATCH (g)-[info_rel]->(ai:GrantAdditionalInfo)
            WHERE type(info_rel) = 'HAS_ADDITIONAL_INFO'
            WITH ai ORDER BY ai.extracted_at DESC, ai.additional_info_id DESC
            RETURN [x IN collect(ai)[0..$additional_info_limit] WHERE x IS NOT NULL | {
                additional_info_id: x.additional_info_id,
                url: x.additional_info_url,
                excerpt: CASE
                    WHEN x.extracted_text IS NULL THEN NULL
                    ELSE substring(x.extracted_text, 0, $excerpt_chars)
                END
            }] AS additional_info
        }

        CALL (g) {
            OPTIONAL MATCH (g)-[att_rel]->(att:GrantAttachment)
            WHERE type(att_rel) = 'HAS_ATTACHMENT'
            WITH att ORDER BY att.extracted_at DESC, att.attachment_id DESC
            RETURN [x IN collect(att)[0..$attachment_limit] WHERE x IS NOT NULL | {
                attachment_id: x.attachment_id,
                file_name: x.file_name,
                url: x.file_download_path,
                excerpt: CASE
                    WHEN x.extracted_text IS NULL THEN NULL
                    ELSE substring(x.extracted_text, 0, $excerpt_chars)
                END
            }] AS attachments
        }

        RETURN {
            opportunity_id: g.opportunity_id,
            opportunity_title: g.opportunity_title,
            agency_name: g.agency_name,
            opportunity_status: g.opportunity_status,
            category: g.category,
            award_ceiling: g.award_ceiling,
            award_floor: g.award_floor,
            close_date: g.close_date,
            summary_description: g.summary_description,
            broad_categories: broad_categories,
            specific_categories: specific_categories,
            keyword_score: keyword_score,
            keyword_hits: keyword_hits,
            keyword_matches: keyword_matches,
            additional_info: additional_info,
            attachments: attachments
        } AS grant
        ORDER BY
            CASE WHEN size($keyword_terms) = 0 THEN 0 ELSE keyword_score END DESC,
            CASE WHEN size($keyword_terms) = 0 THEN 0 ELSE keyword_hits END DESC,
            g.close_date ASC,
            g.opportunity_id ASC
        LIMIT $top_k
    """

    with GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    ) as driver:
        driver.verify_connectivity()
        records, _, _ = driver.execute_query(
            query,
            parameters_={
                "keyword_terms": keyword_terms,
                "agency_filter_enabled": agency_filter_enabled,
                "agency_query_compact": agency_query_compact,
                "agency_query_initials_all": agency_query_initials_all,
                "agency_query_initials_nostop": agency_query_initials_nostop,
                "agency_stopwords": sorted(AGENCY_STOPWORDS),
                "broad_category": str(broad_category or "").strip().lower(),
                "specific_category": str(specific_category or "").strip().lower(),
                "opportunity_status": str(opportunity_status or "").strip(),
                "relations": KEYWORD_RELATIONS,
                "top_k": _safe_limit(top_k, default=10, minimum=1, maximum=200),
                "additional_info_limit": _safe_limit(additional_info_limit, default=2, minimum=0, maximum=20),
                "attachment_limit": _safe_limit(attachment_limit, default=2, minimum=0, maximum=20),
                "excerpt_chars": _safe_limit(excerpt_chars, default=280, minimum=60, maximum=2000),
            },
            routing_=RoutingControl.READ,
            database_=settings.database,
        )

    grants = [record.get("grant") for record in records if record.get("grant")]

    return {
        "filters": {
            "keyword_terms": keyword_terms,
            "agency_name": safe_text(agency_name),
            "broad_category": safe_text(broad_category),
            "specific_category": safe_text(specific_category),
            "opportunity_status": safe_text(opportunity_status),
        },
        "found": len(grants) > 0,
        "count": len(grants),
        "grants": grants,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve grants from Neo4j Grant GraphRAG prototype.")
    parser.add_argument(
        "--query-keywords",
        type=str,
        default="",
        help="Comma-separated keyword terms (e.g. 'robotics,reinforcement learning').",
    )
    parser.add_argument(
        "--agency",
        type=str,
        default="",
        help="Agency filter (normalized/fuzzy; supports acronym inputs like NSF, NIH, DoD).",
    )
    parser.add_argument("--broad-category", type=str, default="", help="Filter by broad category.")
    parser.add_argument("--specific-category", type=str, default="", help="Filter by specific category.")
    parser.add_argument("--opportunity-status", type=str, default="", help="Exact opportunity status filter.")
    parser.add_argument("--top-k", type=int, default=10, help="Max grants returned.")
    parser.add_argument("--additional-info-limit", type=int, default=2, help="Additional info snippets per grant.")
    parser.add_argument("--attachment-limit", type=int, default=2, help="Attachment snippets per grant.")
    parser.add_argument("--excerpt-chars", type=int, default=280, help="Excerpt character cap for snippets.")
    parser.add_argument("--json-only", action="store_true", help="Print only JSON.")

    parser.add_argument("--uri", type=str, default="", help="Neo4j URI. Fallback: NEO4J_URI")
    parser.add_argument("--username", type=str, default="", help="Neo4j username. Fallback: NEO4J_USERNAME")
    parser.add_argument("--password", type=str, default="", help="Neo4j password. Fallback: NEO4J_PASSWORD")
    parser.add_argument("--database", type=str, default="", help="Neo4j database. Fallback: NEO4J_DATABASE or neo4j")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    load_dotenv_if_present()

    settings = read_neo4j_settings(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    result = retrieve_grants(
        settings=settings,
        keyword_terms=_normalize_terms(args.query_keywords),
        agency_name=args.agency,
        broad_category=args.broad_category,
        specific_category=args.specific_category,
        opportunity_status=args.opportunity_status,
        top_k=args.top_k,
        additional_info_limit=args.additional_info_limit,
        attachment_limit=args.attachment_limit,
        excerpt_chars=args.excerpt_chars,
    )

    if not args.json_only:
        print("Grant GraphRAG retrieval complete.")
        print(f"  found grants: {result['count']}")
        print()

    print(json.dumps(json_ready(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
