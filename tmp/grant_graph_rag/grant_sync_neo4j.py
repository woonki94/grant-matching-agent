from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase
from sqlalchemy import bindparam, text
from sqlalchemy.orm import selectinload

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from db.db_conn import SessionLocal
from db.models.opportunity import Opportunity, OpportunityAdditionalInfo, OpportunityAttachment
from tmp.neo4j_common import (
    Neo4jSettings,
    coerce_float,
    coerce_int,
    coerce_iso_datetime,
    coerce_str_list,
    json_ready,
    load_dotenv_if_present,
    read_neo4j_settings,
    safe_text,
)
from utils.content_extractor import load_extracted_content

KEYWORD_RELATIONS: Dict[str, Tuple[str, str]] = {
    "HAS_RESEARCH_DOMAIN": ("research", "domain"),
    "HAS_RESEARCH_SPECIALIZATION": ("research", "specialization"),
    "HAS_APPLICATION_DOMAIN": ("application", "domain"),
    "HAS_APPLICATION_SPECIALIZATION": ("application", "specialization"),
}


@dataclass(frozen=True)
class SyncLimits:
    max_additional_info: int
    max_attachments: int
    max_text_chars: int


def _coerce_weight(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


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


def _clip_text(value: Any, *, max_chars: int) -> Optional[str]:
    text = safe_text(value)
    if not text:
        return None

    cap = _safe_limit(max_chars, default=3000, minimum=100, maximum=50000)
    if len(text) <= cap:
        return text

    clipped = text[:cap].rstrip()
    sentence_end = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
    if sentence_end >= int(cap * 0.65):
        return clipped[: sentence_end + 1].rstrip()

    word_end = clipped.rfind(" ")
    if word_end >= int(cap * 0.75):
        return clipped[:word_end].rstrip()

    return clipped


def _load_category_map(session, opportunity_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not opportunity_ids:
        return {}

    stmt = (
        text(
            """
            SELECT opportunity_id, broad_category, specific_categories
            FROM opportunity_keywords
            WHERE opportunity_id IN :ids
            """
        )
        .bindparams(bindparam("ids", expanding=True))
    )

    out: Dict[str, Dict[str, Any]] = {}
    rows = session.execute(stmt, {"ids": [str(x) for x in opportunity_ids]}).mappings().all()
    for row in rows:
        opp_id = str(row.get("opportunity_id") or "").strip()
        if not opp_id:
            continue
        out[opp_id] = {
            "broad_category": safe_text(row.get("broad_category")),
            "specific_categories": coerce_str_list(row.get("specific_categories") or []),
        }
    return out


def _load_opportunities(
    *,
    opportunity_id: str,
    sync_all: bool,
    limit: int,
    offset: int,
) -> List[Opportunity]:
    with SessionLocal() as session:
        query = (
            session.query(Opportunity)
            .options(
                selectinload(Opportunity.additional_info),
                selectinload(Opportunity.attachments),
                selectinload(Opportunity.keyword),
            )
            .order_by(Opportunity.opportunity_id.asc())
        )

        cleaned_id = str(opportunity_id or "").strip()
        if cleaned_id:
            rows = query.filter(Opportunity.opportunity_id == cleaned_id).all()
        elif sync_all:
            if limit > 0:
                rows = query.offset(max(0, int(offset or 0))).limit(limit).all()
            else:
                rows = query.all()
        else:
            rows = query.limit(1).all()

        category_map = _load_category_map(
            session,
            [str(getattr(r, "opportunity_id", "") or "").strip() for r in rows],
        )

        for row in rows:
            oid = str(getattr(row, "opportunity_id", "") or "").strip()
            meta = category_map.get(oid, {})
            setattr(row, "_broad_category", safe_text(meta.get("broad_category")))
            setattr(row, "_specific_categories", coerce_str_list(meta.get("specific_categories") or []))

        return rows


def _grant_row(opp: Opportunity) -> Dict[str, Any]:
    return {
        "opportunity_id": safe_text(opp.opportunity_id),
        "agency_name": safe_text(opp.agency_name),
        "category": safe_text(opp.category),
        "opportunity_status": safe_text(opp.opportunity_status),
        "opportunity_title": safe_text(opp.opportunity_title),
        "agency_email_address": safe_text(opp.agency_email_address),
        "applicant_types": coerce_str_list(opp.applicant_types),
        "archive_date": safe_text(opp.archive_date),
        "award_ceiling": coerce_float(opp.award_ceiling),
        "award_floor": coerce_float(opp.award_floor),
        "close_date": safe_text(opp.close_date),
        "created_at": safe_text(opp.created_at),
        "estimated_total_program_funding": coerce_float(opp.estimated_total_program_funding),
        "expected_number_of_awards": coerce_int(opp.expected_number_of_awards),
        "forecasted_award_date": safe_text(opp.forecasted_award_date),
        "forecasted_close_date": safe_text(opp.forecasted_close_date),
        "forecasted_post_date": safe_text(opp.forecasted_post_date),
        "forecasted_project_start_date": safe_text(opp.forecasted_project_start_date),
        "funding_categories": coerce_str_list(opp.funding_categories),
        "funding_instruments": coerce_str_list(opp.funding_instruments),
        "is_cost_sharing": bool(opp.is_cost_sharing) if opp.is_cost_sharing is not None else None,
        "post_date": safe_text(opp.post_date),
        "summary_description": safe_text(opp.summary_description),
        "updated_at": None,
    }


def _additional_info_text_map(
    rows: List[OpportunityAdditionalInfo],
    *,
    include_extracted_text: bool,
    max_text_chars: int,
) -> Dict[str, str]:
    if not include_extracted_text:
        return {}

    items = load_extracted_content(
        rows,
        url_attr="additional_info_url",
    )
    out: Dict[str, str] = {}
    for item in items:
        url = safe_text(item.get("url"))
        text_value = _clip_text(item.get("content"), max_chars=max_text_chars)
        if not url or not text_value:
            continue
        out[url] = text_value
    return out


def _attachment_text_map(
    rows: List[OpportunityAttachment],
    *,
    include_extracted_text: bool,
    max_text_chars: int,
) -> Dict[Tuple[str, str], str]:
    if not include_extracted_text:
        return {}

    items = load_extracted_content(
        rows,
        url_attr="file_download_path",
        title_attr="file_name",
    )
    out: Dict[Tuple[str, str], str] = {}
    for item in items:
        url = safe_text(item.get("url")) or ""
        title = safe_text(item.get("title")) or ""
        text_value = _clip_text(item.get("content"), max_chars=max_text_chars)
        if not text_value:
            continue
        out[(url, title)] = text_value
    return out


def _additional_info_rows(
    opp: Opportunity,
    *,
    max_additional_info: int,
    include_extracted_text: bool,
    max_text_chars: int,
) -> List[Dict[str, Any]]:
    safe_max = _safe_limit(max_additional_info, default=50, minimum=1, maximum=1000)
    rows: List[OpportunityAdditionalInfo] = sorted(
        list(opp.additional_info or []),
        key=lambda x: (
            x.extracted_at.isoformat() if x.extracted_at is not None else "",
            int(x.id or 0),
        ),
        reverse=True,
    )[:safe_max]

    text_by_url = _additional_info_text_map(
        rows,
        include_extracted_text=include_extracted_text,
        max_text_chars=max_text_chars,
    )

    out: List[Dict[str, Any]] = []
    for row in rows:
        url = safe_text(row.additional_info_url)
        if not url:
            continue
        out.append(
            {
                "additional_info_id": int(row.id),
                "opportunity_id": safe_text(row.opportunity_id),
                "additional_info_url": url,
                "content_path": safe_text(row.content_path),
                "detected_type": safe_text(row.detected_type),
                "content_char_count": coerce_int(row.content_char_count),
                "extracted_at": coerce_iso_datetime(row.extracted_at),
                "extract_status": safe_text(row.extract_status),
                "extract_error": safe_text(row.extract_error),
                "extracted_text": text_by_url.get(url),
            }
        )
    return out


def _attachment_rows(
    opp: Opportunity,
    *,
    max_attachments: int,
    include_extracted_text: bool,
    max_text_chars: int,
) -> List[Dict[str, Any]]:
    safe_max = _safe_limit(max_attachments, default=50, minimum=1, maximum=2000)
    rows: List[OpportunityAttachment] = sorted(
        list(opp.attachments or []),
        key=lambda x: (
            x.extracted_at.isoformat() if x.extracted_at is not None else "",
            int(x.id or 0),
        ),
        reverse=True,
    )[:safe_max]

    text_by_key = _attachment_text_map(
        rows,
        include_extracted_text=include_extracted_text,
        max_text_chars=max_text_chars,
    )

    out: List[Dict[str, Any]] = []
    for row in rows:
        title = safe_text(row.file_name)
        download_url = safe_text(row.file_download_path)
        if not title and not download_url:
            continue

        out.append(
            {
                "attachment_id": int(row.id),
                "opportunity_id": safe_text(row.opportunity_id),
                "file_name": title,
                "file_download_path": download_url,
                "content_path": safe_text(row.content_path),
                "detected_type": safe_text(row.detected_type),
                "content_char_count": coerce_int(row.content_char_count),
                "extracted_at": coerce_iso_datetime(row.extracted_at),
                "extract_status": safe_text(row.extract_status),
                "extract_error": safe_text(row.extract_error),
                "extracted_text": text_by_key.get((download_url or "", title or "")),
            }
        )
    return out


def _keyword_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    out: List[Dict[str, Any]] = []
    seen = set()

    for relation, (section, bucket) in KEYWORD_RELATIONS.items():
        section_payload = payload.get(section)
        if not isinstance(section_payload, dict):
            continue

        raw_values = section_payload.get(bucket) or []
        if isinstance(raw_values, (str, int, float, dict)):
            raw_values = [raw_values]

        for item in raw_values:
            if isinstance(item, dict):
                value = safe_text(item.get("t") or item.get("text"))
                weight = _coerce_weight(item.get("w"))
            else:
                value = safe_text(item)
                weight = None

            if not value:
                continue

            dedupe_key = (value.lower(), section, bucket, relation)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            out.append(
                {
                    "value": value,
                    "section": section,
                    "bucket": bucket,
                    "relation": relation,
                    "weight": weight,
                }
            )
    return out


def _broad_category_row(opp: Opportunity) -> Optional[Dict[str, Any]]:
    broad = safe_text(getattr(opp, "_broad_category", None))
    if not broad:
        return None
    return {
        "name": broad.lower(),
    }


def _specific_category_rows(opp: Opportunity) -> List[Dict[str, Any]]:
    specific = coerce_str_list(getattr(opp, "_specific_categories", []) or [])
    out: List[Dict[str, Any]] = []
    for value in specific:
        cleaned = safe_text(value)
        if not cleaned:
            continue
        out.append({"name": cleaned.lower()})
    return out


def sync_grant_to_neo4j(
    *,
    driver,
    settings: Neo4jSettings,
    opp: Opportunity,
    limits: SyncLimits,
    include_extracted_text: bool,
) -> Dict[str, Any]:
    grant_row = _grant_row(opp)
    additional_info_rows = _additional_info_rows(
        opp,
        max_additional_info=limits.max_additional_info,
        include_extracted_text=include_extracted_text,
        max_text_chars=limits.max_text_chars,
    )
    attachment_rows = _attachment_rows(
        opp,
        max_attachments=limits.max_attachments,
        include_extracted_text=include_extracted_text,
        max_text_chars=limits.max_text_chars,
    )
    keyword_payload = (getattr(opp, "keyword", None) and getattr(opp.keyword, "keywords", None)) or {}
    keyword_rows = _keyword_rows(keyword_payload)
    broad_category_row = _broad_category_row(opp)
    specific_category_rows = _specific_category_rows(opp)

    opportunity_id = grant_row.get("opportunity_id")
    if not opportunity_id:
        raise ValueError("Opportunity has no opportunity_id; cannot sync.")

    driver.execute_query(
        """
        MERGE (g:Grant {opportunity_id: $row.opportunity_id})
        SET
            g.agency_name = $row.agency_name,
            g.category = $row.category,
            g.opportunity_status = $row.opportunity_status,
            g.opportunity_title = $row.opportunity_title,
            g.agency_email_address = $row.agency_email_address,
            g.applicant_types = $row.applicant_types,
            g.archive_date = $row.archive_date,
            g.award_ceiling = $row.award_ceiling,
            g.award_floor = $row.award_floor,
            g.close_date = $row.close_date,
            g.created_at = $row.created_at,
            g.estimated_total_program_funding = $row.estimated_total_program_funding,
            g.expected_number_of_awards = $row.expected_number_of_awards,
            g.forecasted_award_date = $row.forecasted_award_date,
            g.forecasted_close_date = $row.forecasted_close_date,
            g.forecasted_post_date = $row.forecasted_post_date,
            g.forecasted_project_start_date = $row.forecasted_project_start_date,
            g.funding_categories = $row.funding_categories,
            g.funding_instruments = $row.funding_instruments,
            g.is_cost_sharing = $row.is_cost_sharing,
            g.post_date = $row.post_date,
            g.summary_description = $row.summary_description,
            g.updated_at = datetime()
        """,
        parameters_={"row": grant_row},
        database_=settings.database,
    )

    # Refresh grant-owned nodes/edges.
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})
        OPTIONAL MATCH (g)-[:HAS_ADDITIONAL_INFO]->(ai:GrantAdditionalInfo)
        DETACH DELETE ai
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})
        OPTIONAL MATCH (g)-[:HAS_ATTACHMENT]->(att:GrantAttachment)
        DETACH DELETE att
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:GrantKeyword)
        WHERE type(r) IN [
            'HAS_RESEARCH_DOMAIN',
            'HAS_RESEARCH_SPECIALIZATION',
            'HAS_APPLICATION_DOMAIN',
            'HAS_APPLICATION_SPECIALIZATION'
        ]
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:Agency)
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:GrantBroadCategory)
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:GrantSpecificCategory)
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:OpportunityCategory)
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:ApplicantType)
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:FundingCategory)
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:FundingInstrument)
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings.database,
    )

    agency_name = grant_row.get("agency_name")
    if agency_name:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            MERGE (a:Agency {name: $agency_name})
            SET a.updated_at = datetime()
            MERGE (g)-[:FUNDED_BY]->(a)
            """,
            parameters_={"opportunity_id": opportunity_id, "agency_name": agency_name},
            database_=settings.database,
        )

    opportunity_category = safe_text(grant_row.get("category"))
    if opportunity_category:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            MERGE (c:OpportunityCategory {name: $category_name})
            SET c.updated_at = datetime()
            MERGE (g)-[:IN_OPPORTUNITY_CATEGORY]->(c)
            """,
            parameters_={
                "opportunity_id": opportunity_id,
                "category_name": opportunity_category,
            },
            database_=settings.database,
        )

    if broad_category_row:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            MERGE (bc:GrantBroadCategory {name: $row.name})
            SET bc.updated_at = datetime()
            MERGE (g)-[:HAS_BROAD_CATEGORY]->(bc)
            """,
            parameters_={"opportunity_id": opportunity_id, "row": broad_category_row},
            database_=settings.database,
        )

    if specific_category_rows:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            UNWIND $rows AS row
            MERGE (sc:GrantSpecificCategory {name: row.name})
            SET sc.updated_at = datetime()
            MERGE (g)-[:HAS_SPECIFIC_CATEGORY]->(sc)
            """,
            parameters_={"opportunity_id": opportunity_id, "rows": specific_category_rows},
            database_=settings.database,
        )

    applicant_types = coerce_str_list(grant_row.get("applicant_types") or [])
    if applicant_types:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            UNWIND $items AS name
            MERGE (a:ApplicantType {name: name})
            SET a.updated_at = datetime()
            MERGE (g)-[:HAS_APPLICANT_TYPE]->(a)
            """,
            parameters_={"opportunity_id": opportunity_id, "items": applicant_types},
            database_=settings.database,
        )

    funding_categories = coerce_str_list(grant_row.get("funding_categories") or [])
    if funding_categories:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            UNWIND $items AS name
            MERGE (fc:FundingCategory {name: name})
            SET fc.updated_at = datetime()
            MERGE (g)-[:HAS_FUNDING_CATEGORY]->(fc)
            """,
            parameters_={"opportunity_id": opportunity_id, "items": funding_categories},
            database_=settings.database,
        )

    funding_instruments = coerce_str_list(grant_row.get("funding_instruments") or [])
    if funding_instruments:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            UNWIND $items AS name
            MERGE (fi:FundingInstrument {name: name})
            SET fi.updated_at = datetime()
            MERGE (g)-[:HAS_FUNDING_INSTRUMENT]->(fi)
            """,
            parameters_={"opportunity_id": opportunity_id, "items": funding_instruments},
            database_=settings.database,
        )

    if additional_info_rows:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            UNWIND $rows AS row
            MERGE (ai:GrantAdditionalInfo {additional_info_id: row.additional_info_id})
            SET
                ai.opportunity_id = row.opportunity_id,
                ai.additional_info_url = row.additional_info_url,
                ai.content_path = row.content_path,
                ai.detected_type = row.detected_type,
                ai.content_char_count = row.content_char_count,
                ai.extracted_at = row.extracted_at,
                ai.extract_status = row.extract_status,
                ai.extract_error = row.extract_error,
                ai.extracted_text = row.extracted_text,
                ai.updated_at = datetime()
            MERGE (g)-[:HAS_ADDITIONAL_INFO]->(ai)
            """,
            parameters_={"opportunity_id": opportunity_id, "rows": additional_info_rows},
            database_=settings.database,
        )

    if attachment_rows:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            UNWIND $rows AS row
            MERGE (att:GrantAttachment {attachment_id: row.attachment_id})
            SET
                att.opportunity_id = row.opportunity_id,
                att.file_name = row.file_name,
                att.file_download_path = row.file_download_path,
                att.content_path = row.content_path,
                att.detected_type = row.detected_type,
                att.content_char_count = row.content_char_count,
                att.extracted_at = row.extracted_at,
                att.extract_status = row.extract_status,
                att.extract_error = row.extract_error,
                att.extracted_text = row.extracted_text,
                att.updated_at = datetime()
            MERGE (g)-[:HAS_ATTACHMENT]->(att)
            """,
            parameters_={"opportunity_id": opportunity_id, "rows": attachment_rows},
            database_=settings.database,
        )

    for relation in KEYWORD_RELATIONS:
        rows = [item for item in keyword_rows if item.get("relation") == relation]
        if not rows:
            continue

        driver.execute_query(
            f"""
            MATCH (g:Grant {{opportunity_id: $opportunity_id}})
            UNWIND $rows AS row
            MERGE (k:GrantKeyword {{
                value: row.value,
                section: row.section,
                bucket: row.bucket
            }})
            SET k.updated_at = datetime()
            MERGE (g)-[r:{relation}]->(k)
            SET
                r.weight = row.weight,
                r.updated_at = datetime()
            """,
            parameters_={"opportunity_id": opportunity_id, "rows": rows},
            database_=settings.database,
        )

    return {
        "opportunity_id": opportunity_id,
        "counts": {
            "additional_info": len(additional_info_rows),
            "attachments": len(attachment_rows),
            "keywords": len(keyword_rows),
            "has_broad_category": 1 if broad_category_row else 0,
            "specific_categories": len(specific_category_rows),
            "applicant_types": len(applicant_types),
            "funding_categories": len(funding_categories),
            "funding_instruments": len(funding_instruments),
        },
    }


def verify_grant_from_neo4j(
    *,
    driver,
    settings: Neo4jSettings,
    opportunity_id: str,
    additional_info_limit: int,
    attachment_limit: int,
) -> Dict[str, Any]:
    records, _, _ = driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})
        CALL (g) {
            OPTIONAL MATCH (g)-[:FUNDED_BY]->(a:Agency)
            RETURN a {.name} AS agency
        }
        CALL (g) {
            OPTIONAL MATCH (g)-[:HAS_BROAD_CATEGORY]->(bc:GrantBroadCategory)
            RETURN collect(DISTINCT bc.name) AS broad_categories
        }
        CALL (g) {
            OPTIONAL MATCH (g)-[:HAS_SPECIFIC_CATEGORY]->(sc:GrantSpecificCategory)
            RETURN collect(DISTINCT sc.name) AS specific_categories
        }
        CALL (g) {
            OPTIONAL MATCH (g)-[:HAS_APPLICANT_TYPE]->(at:ApplicantType)
            RETURN collect(DISTINCT at.name) AS applicant_types
        }
        CALL (g) {
            OPTIONAL MATCH (g)-[:HAS_FUNDING_CATEGORY]->(fc:FundingCategory)
            RETURN collect(DISTINCT fc.name) AS funding_categories
        }
        CALL (g) {
            OPTIONAL MATCH (g)-[:HAS_FUNDING_INSTRUMENT]->(fi:FundingInstrument)
            RETURN collect(DISTINCT fi.name) AS funding_instruments
        }
        CALL (g) {
            OPTIONAL MATCH (g)-[:HAS_ADDITIONAL_INFO]->(ai:GrantAdditionalInfo)
            WITH ai ORDER BY ai.extracted_at DESC, ai.additional_info_id DESC
            RETURN [x IN collect(ai)[0..$additional_info_limit] WHERE x IS NOT NULL | x {
                .additional_info_id,
                .additional_info_url,
                .extract_status,
                .content_char_count
            }] AS additional_info
        }
        CALL (g) {
            OPTIONAL MATCH (g)-[:HAS_ATTACHMENT]->(att:GrantAttachment)
            WITH att ORDER BY att.extracted_at DESC, att.attachment_id DESC
            RETURN [x IN collect(att)[0..$attachment_limit] WHERE x IS NOT NULL | x {
                .attachment_id,
                .file_name,
                .extract_status,
                .content_char_count
            }] AS attachments
        }
        CALL (g) {
            OPTIONAL MATCH (g)-[r]->(k:GrantKeyword)
            WHERE type(r) IN [
                'HAS_RESEARCH_DOMAIN',
                'HAS_RESEARCH_SPECIALIZATION',
                'HAS_APPLICATION_DOMAIN',
                'HAS_APPLICATION_SPECIALIZATION'
            ]
            RETURN collect(DISTINCT {
                relation: type(r),
                value: k.value,
                section: k.section,
                bucket: k.bucket,
                weight: r.weight
            }) AS keywords
        }
        RETURN
            g {
                .opportunity_id,
                .opportunity_title,
                .agency_name,
                .opportunity_status,
                .category,
                .award_ceiling,
                .award_floor,
                .close_date
            } AS grant,
            agency,
            broad_categories,
            specific_categories,
            applicant_types,
            funding_categories,
            funding_instruments,
            additional_info,
            attachments,
            keywords
        """,
        parameters_={
            "opportunity_id": str(opportunity_id or "").strip(),
            "additional_info_limit": _safe_limit(additional_info_limit, default=5, minimum=1, maximum=100),
            "attachment_limit": _safe_limit(attachment_limit, default=5, minimum=1, maximum=100),
        },
        database_=settings.database,
    )

    if not records:
        return {
            "opportunity_id": str(opportunity_id or "").strip(),
            "found": False,
        }

    row = records[0]
    additional_info = [item for item in (row.get("additional_info") or []) if item]
    attachments = [item for item in (row.get("attachments") or []) if item]
    keywords = [item for item in (row.get("keywords") or []) if item and item.get("value")]

    return {
        "opportunity_id": str(opportunity_id or "").strip(),
        "found": True,
        "grant": row.get("grant"),
        "counts": {
            "additional_info": len(additional_info),
            "attachments": len(attachments),
            "keywords": len(keywords),
            "broad_categories": len([x for x in (row.get("broad_categories") or []) if x]),
            "specific_categories": len([x for x in (row.get("specific_categories") or []) if x]),
            "applicant_types": len([x for x in (row.get("applicant_types") or []) if x]),
            "funding_categories": len([x for x in (row.get("funding_categories") or []) if x]),
            "funding_instruments": len([x for x in (row.get("funding_instruments") or []) if x]),
        },
        "preview": {
            "agency": row.get("agency"),
            "broad_categories": row.get("broad_categories"),
            "specific_categories": row.get("specific_categories"),
            "applicant_types": row.get("applicant_types"),
            "funding_categories": row.get("funding_categories"),
            "funding_instruments": row.get("funding_instruments"),
            "additional_info": additional_info,
            "attachments": attachments,
            "keywords": keywords,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync Grant GraphRAG data from Postgres into Neo4j.")
    parser.add_argument("--opportunity-id", type=str, default="", help="Sync one opportunity id.")
    parser.add_argument("--all", action="store_true", help="Sync all opportunities.")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows when using --all (0 = no limit).")
    parser.add_argument("--offset", type=int, default=0, help="Offset rows when using --all.")

    parser.add_argument("--max-additional-info", type=int, default=100, help="Max additional-info rows per grant.")
    parser.add_argument("--max-attachments", type=int, default=100, help="Max attachment rows per grant.")
    parser.add_argument("--max-text-chars", type=int, default=4000, help="Max chars for each extracted text block.")
    parser.add_argument(
        "--skip-extracted-text",
        action="store_true",
        help="Do not load extracted S3 text for additional info / attachments.",
    )

    parser.add_argument("--verify-opportunity-id", type=str, default="", help="Run verify query for this id after sync.")
    parser.add_argument("--verify-additional-info-limit", type=int, default=5, help="Additional-info rows in verify output.")
    parser.add_argument("--verify-attachment-limit", type=int, default=5, help="Attachment rows in verify output.")
    parser.add_argument("--json-only", action="store_true", help="Print only JSON output.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on first sync error.")

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

    rows = _load_opportunities(
        opportunity_id=args.opportunity_id,
        sync_all=bool(args.all),
        limit=max(0, int(args.limit or 0)),
        offset=max(0, int(args.offset or 0)),
    )

    if not rows:
        raise RuntimeError("No opportunities found for requested sync scope.")

    limits = SyncLimits(
        max_additional_info=_safe_limit(args.max_additional_info, default=100, minimum=1, maximum=1000),
        max_attachments=_safe_limit(args.max_attachments, default=100, minimum=1, maximum=2000),
        max_text_chars=_safe_limit(args.max_text_chars, default=4000, minimum=100, maximum=50000),
    )

    include_extracted_text = not bool(args.skip_extracted_text)

    synced: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    ) as driver:
        driver.verify_connectivity()

        for opp in rows:
            opportunity_id = safe_text(getattr(opp, "opportunity_id", None)) or ""
            try:
                result = sync_grant_to_neo4j(
                    driver=driver,
                    settings=settings,
                    opp=opp,
                    limits=limits,
                    include_extracted_text=include_extracted_text,
                )
                synced.append(result)
            except Exception as exc:
                errors.append(
                    {
                        "opportunity_id": opportunity_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if args.stop_on_error:
                    break

        verify_id = safe_text(args.verify_opportunity_id)
        if not verify_id and len(synced) == 1:
            verify_id = safe_text(synced[0].get("opportunity_id"))

        verify = None
        if verify_id:
            verify = verify_grant_from_neo4j(
                driver=driver,
                settings=settings,
                opportunity_id=verify_id,
                additional_info_limit=args.verify_additional_info_limit,
                attachment_limit=args.verify_attachment_limit,
            )

    totals = {
        "grants_synced": len(synced),
        "grants_failed": len(errors),
        "additional_info": sum(int(item.get("counts", {}).get("additional_info", 0)) for item in synced),
        "attachments": sum(int(item.get("counts", {}).get("attachments", 0)) for item in synced),
        "keywords": sum(int(item.get("counts", {}).get("keywords", 0)) for item in synced),
        "broad_categories": sum(int(item.get("counts", {}).get("has_broad_category", 0)) for item in synced),
        "specific_categories": sum(int(item.get("counts", {}).get("specific_categories", 0)) for item in synced),
        "applicant_types": sum(int(item.get("counts", {}).get("applicant_types", 0)) for item in synced),
        "funding_categories": sum(int(item.get("counts", {}).get("funding_categories", 0)) for item in synced),
        "funding_instruments": sum(int(item.get("counts", {}).get("funding_instruments", 0)) for item in synced),
    }

    payload = {
        "scope": {
            "opportunity_id": safe_text(args.opportunity_id),
            "all": bool(args.all),
            "limit": max(0, int(args.limit or 0)),
            "offset": max(0, int(args.offset or 0)),
            "include_extracted_text": include_extracted_text,
        },
        "totals": totals,
        "synced": synced,
        "errors": errors,
        "verify": verify,
    }

    if not args.json_only:
        print("Grant GraphRAG sync complete.")
        print(f"  synced grants      : {totals['grants_synced']}")
        print(f"  failed grants      : {totals['grants_failed']}")
        print(f"  additional info    : {totals['additional_info']}")
        print(f"  attachments        : {totals['attachments']}")
        print(f"  keyword edges      : {totals['keywords']}")
        print(f"  broad categories   : {totals['broad_categories']}")
        print(f"  specific categories: {totals['specific_categories']}")
        print(f"  applicant types    : {totals['applicant_types']}")
        print(f"  funding categories : {totals['funding_categories']}")
        print(f"  funding instruments: {totals['funding_instruments']}")
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
