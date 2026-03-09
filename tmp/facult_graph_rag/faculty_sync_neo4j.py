from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase
from sqlalchemy import func
from sqlalchemy.orm import selectinload

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from db.db_conn import SessionLocal
from db.models.faculty import Faculty, FacultyAdditionalInfo, FacultyPublication
from tmp.neo4j_common import (
    Neo4jSettings,
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
    max_publications: int
    max_additional_info: int
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


def _load_faculties(
    *,
    email: str,
    sync_all: bool,
    limit: int,
    offset: int,
) -> List[Faculty]:
    with SessionLocal() as session:
        query = (
            session.query(Faculty)
            .options(
                selectinload(Faculty.additional_info),
                selectinload(Faculty.publications),
                selectinload(Faculty.keyword),
            )
            .order_by(Faculty.faculty_id.asc())
        )

        cleaned_email = str(email or "").strip().lower()
        if cleaned_email:
            return query.filter(func.lower(Faculty.email) == cleaned_email).all()

        if sync_all:
            if limit > 0:
                return query.offset(max(0, int(offset or 0))).limit(limit).all()
            return query.all()

        return query.limit(1).all()


def _faculty_row(fac: Faculty) -> Dict[str, Any]:
    return {
        "faculty_id": int(fac.faculty_id),
        "email": str(fac.email or "").strip().lower(),
        "source_url": safe_text(fac.source_url),
        "name": safe_text(fac.name),
        "phone": safe_text(fac.phone),
        "position": safe_text(fac.position),
        "organization": safe_text(fac.organization),
        "organizations": coerce_str_list(fac.organizations),
        "address": safe_text(fac.address),
        "biography": safe_text(fac.biography),
        "degrees": coerce_str_list(fac.degrees),
        "expertise": coerce_str_list(fac.expertise),
        "profile_last_refreshed_at": coerce_iso_datetime(fac.profile_last_refreshed_at),
    }


def _additional_info_text_map(
    rows: List[FacultyAdditionalInfo],
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


def _additional_info_rows(
    fac: Faculty,
    *,
    max_additional_info: int,
    include_extracted_text: bool,
    max_text_chars: int,
) -> List[Dict[str, Any]]:
    safe_max = _safe_limit(max_additional_info, default=50, minimum=1, maximum=1000)
    rows: List[FacultyAdditionalInfo] = sorted(
        list(fac.additional_info or []),
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
                "faculty_id": int(fac.faculty_id),
                "additional_info_url": url,
                "content_path": safe_text(row.content_path),
                "detected_type": safe_text(row.detected_type),
                "content_char_count": int(row.content_char_count) if row.content_char_count is not None else None,
                "extracted_at": coerce_iso_datetime(row.extracted_at),
                "extract_status": safe_text(row.extract_status),
                "extract_error": safe_text(row.extract_error),
                "extracted_text": text_by_url.get(url),
            }
        )
    return out


def _publication_rows(
    fac: Faculty,
    *,
    max_publications: int,
) -> List[Dict[str, Any]]:
    safe_max = _safe_limit(max_publications, default=100, minimum=1, maximum=3000)
    rows: List[FacultyPublication] = sorted(
        list(fac.publications or []),
        key=lambda x: ((x.year or 0), (x.id or 0)),
        reverse=True,
    )[:safe_max]

    out: List[Dict[str, Any]] = []
    for row in rows:
        title = safe_text(row.title)
        if not title:
            continue

        out.append(
            {
                "publication_id": int(row.id),
                "faculty_id": int(fac.faculty_id),
                "title": title,
                "abstract": safe_text(row.abstract),
                "year": int(row.year) if row.year is not None else None,
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


def sync_faculty_to_neo4j(
    *,
    driver,
    settings: Neo4jSettings,
    fac: Faculty,
    limits: SyncLimits,
    include_extracted_text: bool,
) -> Dict[str, Any]:
    faculty_row = _faculty_row(fac)
    additional_info_rows = _additional_info_rows(
        fac,
        max_additional_info=limits.max_additional_info,
        include_extracted_text=include_extracted_text,
        max_text_chars=limits.max_text_chars,
    )
    publication_rows = _publication_rows(
        fac,
        max_publications=limits.max_publications,
    )
    keyword_payload = (getattr(fac, "keyword", None) and getattr(fac.keyword, "keywords", None)) or {}
    keyword_rows = _keyword_rows(keyword_payload)

    email = faculty_row.get("email")
    if not email:
        raise ValueError(f"Faculty {fac.faculty_id} has no email; cannot sync.")

    driver.execute_query(
        """
        MERGE (f:Faculty {email: $row.email})
        SET
            f.faculty_id = $row.faculty_id,
            f.source_url = $row.source_url,
            f.name = $row.name,
            f.phone = $row.phone,
            f.position = $row.position,
            f.organization = $row.organization,
            f.organizations = $row.organizations,
            f.address = $row.address,
            f.biography = $row.biography,
            f.degrees = $row.degrees,
            f.expertise = $row.expertise,
            f.profile_last_refreshed_at = $row.profile_last_refreshed_at,
            f.updated_at = datetime()
        """,
        parameters_={"row": faculty_row},
        database_=settings.database,
    )

    # Refresh faculty-owned nodes/edges.
    driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})
        OPTIONAL MATCH (f)-[:HAS_ADDITIONAL_INFO]->(ai:FacultyAdditionalInfo)
        DETACH DELETE ai
        """,
        parameters_={"email": email},
        database_=settings.database,
    )
    driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})
        OPTIONAL MATCH (f)-[:AUTHORED]->(p:FacultyPublication)
        DETACH DELETE p
        """,
        parameters_={"email": email},
        database_=settings.database,
    )
    driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})-[r]->(:FacultyKeyword)
        WHERE type(r) IN [
            'HAS_RESEARCH_DOMAIN',
            'HAS_RESEARCH_SPECIALIZATION',
            'HAS_APPLICATION_DOMAIN',
            'HAS_APPLICATION_SPECIALIZATION'
        ]
        DELETE r
        """,
        parameters_={"email": email},
        database_=settings.database,
    )

    if additional_info_rows:
        driver.execute_query(
            """
            MATCH (f:Faculty {email: $email})
            UNWIND $rows AS row
            MERGE (ai:FacultyAdditionalInfo {additional_info_id: row.additional_info_id})
            SET
                ai.faculty_id = row.faculty_id,
                ai.additional_info_url = row.additional_info_url,
                ai.content_path = row.content_path,
                ai.detected_type = row.detected_type,
                ai.content_char_count = row.content_char_count,
                ai.extracted_at = row.extracted_at,
                ai.extract_status = row.extract_status,
                ai.extract_error = row.extract_error,
                ai.extracted_text = row.extracted_text,
                ai.updated_at = datetime()
            MERGE (f)-[:HAS_ADDITIONAL_INFO]->(ai)
            """,
            parameters_={"email": email, "rows": additional_info_rows},
            database_=settings.database,
        )

    if publication_rows:
        driver.execute_query(
            """
            MATCH (f:Faculty {email: $email})
            UNWIND $rows AS row
            MERGE (p:FacultyPublication {publication_id: row.publication_id})
            SET
                p.faculty_id = row.faculty_id,
                p.title = row.title,
                p.abstract = row.abstract,
                p.year = row.year,
                p.updated_at = datetime()
            MERGE (f)-[:AUTHORED]->(p)
            """,
            parameters_={"email": email, "rows": publication_rows},
            database_=settings.database,
        )

    for relation in KEYWORD_RELATIONS:
        rows = [item for item in keyword_rows if item.get("relation") == relation]
        if not rows:
            continue

        driver.execute_query(
            f"""
            MATCH (f:Faculty {{email: $email}})
            UNWIND $rows AS row
            MERGE (k:FacultyKeyword {{
                value: row.value,
                section: row.section,
                bucket: row.bucket
            }})
            SET k.updated_at = datetime()
            MERGE (f)-[r:{relation}]->(k)
            SET
                r.weight = row.weight,
                r.updated_at = datetime()
            """,
            parameters_={"email": email, "rows": rows},
            database_=settings.database,
        )

    return {
        "faculty_id": int(faculty_row["faculty_id"]),
        "email": email,
        "counts": {
            "additional_info": len(additional_info_rows),
            "publications": len(publication_rows),
            "keywords": len(keyword_rows),
        },
    }


def verify_faculty_from_neo4j(
    *,
    driver,
    settings: Neo4jSettings,
    email: str,
    publication_limit: int,
    additional_info_limit: int,
) -> Dict[str, Any]:
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})
        CALL (f) {
            OPTIONAL MATCH (f)-[:HAS_ADDITIONAL_INFO]->(ai:FacultyAdditionalInfo)
            WITH ai ORDER BY ai.extracted_at DESC, ai.additional_info_id DESC
            RETURN [x IN collect(ai)[0..$additional_info_limit] WHERE x IS NOT NULL | x {
                .additional_info_id,
                .additional_info_url,
                .extract_status
            }] AS additional_info
        }
        CALL (f) {
            OPTIONAL MATCH (f)-[:AUTHORED]->(p:FacultyPublication)
            WITH p ORDER BY p.year DESC, p.publication_id DESC
            RETURN [x IN collect(p)[0..$publication_limit] WHERE x IS NOT NULL | x {
                .publication_id,
                .title,
                .abstract,
                .year
            }] AS publications
        }
        CALL (f) {
            OPTIONAL MATCH (f)-[r]->(k:FacultyKeyword)
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
            f {.faculty_id, .email, .name, .position, .organization} AS faculty,
            additional_info,
            publications,
            keywords
        """,
        parameters_={
            "email": str(email or "").strip().lower(),
            "publication_limit": _safe_limit(publication_limit, default=10, minimum=1, maximum=200),
            "additional_info_limit": _safe_limit(additional_info_limit, default=10, minimum=1, maximum=200),
        },
        database_=settings.database,
    )

    if not records:
        return {"email": str(email or "").strip().lower(), "found": False}

    row = records[0]
    additional_info = [item for item in (row.get("additional_info") or []) if item]
    publications = [item for item in (row.get("publications") or []) if item]
    keywords = [item for item in (row.get("keywords") or []) if item and item.get("value")]

    return {
        "email": str(email or "").strip().lower(),
        "found": True,
        "faculty": row.get("faculty"),
        "counts": {
            "additional_info": len(additional_info),
            "publications": len(publications),
            "keywords": len(keywords),
        },
        "preview": {
            "additional_info": additional_info,
            "publications": publications,
            "keywords": keywords,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync Faculty GraphRAG data from Postgres into Neo4j.")
    parser.add_argument("--email", type=str, default="", help="Sync one faculty by email.")
    parser.add_argument("--all", action="store_true", help="Sync all faculty rows.")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows when using --all (0 = no limit).")
    parser.add_argument("--offset", type=int, default=0, help="Offset rows when using --all.")

    parser.add_argument("--max-publications", type=int, default=300, help="Max publication rows per faculty.")
    parser.add_argument("--max-additional-info", type=int, default=150, help="Max additional-info rows per faculty.")
    parser.add_argument("--max-text-chars", type=int, default=4000, help="Max chars for each extracted text block.")
    parser.add_argument(
        "--skip-extracted-text",
        action="store_true",
        help="Do not load extracted S3 text for additional info.",
    )

    parser.add_argument("--verify-email", type=str, default="", help="Run verify query for this email after sync.")
    parser.add_argument("--verify-publication-limit", type=int, default=10, help="Publication rows in verify response.")
    parser.add_argument("--verify-additional-info-limit", type=int, default=10, help="Additional info rows in verify response.")
    parser.add_argument("--json-only", action="store_true", help="Print only JSON output.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on first faculty sync error.")

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

    rows = _load_faculties(
        email=args.email,
        sync_all=bool(args.all),
        limit=max(0, int(args.limit or 0)),
        offset=max(0, int(args.offset or 0)),
    )
    if not rows:
        raise RuntimeError("No faculty rows found for requested sync scope.")

    limits = SyncLimits(
        max_publications=_safe_limit(args.max_publications, default=300, minimum=1, maximum=3000),
        max_additional_info=_safe_limit(args.max_additional_info, default=150, minimum=1, maximum=2000),
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

        for fac in rows:
            email = str(getattr(fac, "email", "") or "").strip().lower()
            try:
                result = sync_faculty_to_neo4j(
                    driver=driver,
                    settings=settings,
                    fac=fac,
                    limits=limits,
                    include_extracted_text=include_extracted_text,
                )
                synced.append(result)
            except Exception as exc:
                errors.append(
                    {
                        "faculty_id": int(getattr(fac, "faculty_id", 0) or 0),
                        "email": email,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if args.stop_on_error:
                    break

        verify_email = str(args.verify_email or "").strip().lower()
        if not verify_email and len(synced) == 1:
            verify_email = str(synced[0].get("email") or "").strip().lower()

        verify = None
        if verify_email:
            verify = verify_faculty_from_neo4j(
                driver=driver,
                settings=settings,
                email=verify_email,
                publication_limit=args.verify_publication_limit,
                additional_info_limit=args.verify_additional_info_limit,
            )

    totals = {
        "faculties_synced": len(synced),
        "faculties_failed": len(errors),
        "additional_info": sum(int(item.get("counts", {}).get("additional_info", 0)) for item in synced),
        "publications": sum(int(item.get("counts", {}).get("publications", 0)) for item in synced),
        "keywords": sum(int(item.get("counts", {}).get("keywords", 0)) for item in synced),
    }

    payload = {
        "scope": {
            "email": str(args.email or "").strip().lower(),
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
        print("Faculty GraphRAG sync complete.")
        print(f"  synced faculties : {totals['faculties_synced']}")
        print(f"  failed faculties : {totals['faculties_failed']}")
        print(f"  additional info  : {totals['additional_info']}")
        print(f"  publications     : {totals['publications']}")
        print(f"  keywords         : {totals['keywords']}")
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
