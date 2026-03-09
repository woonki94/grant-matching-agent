from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Type

from neo4j import GraphDatabase

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import settings
from db.db_conn import SessionLocal
from db.models.opportunity import OpportunityAdditionalInfo, OpportunityAttachment
from services.extract_content import run_extraction_pipeline
from tmp.grant_graph_rag.grant_sync_neo4j import (
    SyncLimits,
    _load_opportunities,
    _safe_limit,
    sync_grant_to_neo4j,
)
from tmp.neo4j_common import (
    Neo4jSettings,
    json_ready,
    load_dotenv_if_present,
    read_neo4j_settings,
    safe_text,
)
from tmp.neo4j_schema import init_neo4j_schema


def _requeue_extraction_rows(
    *,
    model: Type[Any],
    force_reextract: bool,
    retry_failed_extraction: bool,
) -> int:
    """
    Mark rows as pending so run_extraction_pipeline can process them.

    - force_reextract=True: requeue every row and clear previous extracted metadata.
    - retry_failed_extraction=True: requeue rows with extract_status='failed'.
    """
    with SessionLocal() as sess:
        if force_reextract:
            rows = sess.query(model).all()
        elif retry_failed_extraction:
            rows = sess.query(model).filter(model.extract_status == "failed").all()
        else:
            return 0

        for row in rows:
            if force_reextract:
                row.content_path = None
                row.detected_type = None
                row.content_char_count = None
                row.extracted_at = None

            row.extract_status = "pending"
            row.extract_error = None

        sess.commit()
        return len(rows)


def _extract_opportunity_content(
    *,
    extract_batch_size: int,
    force_reextract: bool,
    retry_failed_extraction: bool,
) -> Dict[str, Any]:
    bucket = (settings.extracted_content_bucket or "").strip()
    if not bucket:
        raise RuntimeError("EXTRACTED_CONTENT_BUCKET must be set before extraction.")

    common = {
        "batch_size": _safe_limit(extract_batch_size, default=200, minimum=1, maximum=2000),
        "s3_bucket": bucket,
        "s3_prefix": settings.extracted_content_prefix_opportunity,
        "aws_region": settings.aws_region,
        "aws_profile": settings.aws_profile,
    }

    # Keep path layout consistent with existing opportunity import pipeline.
    attachment_subdir = "opportunity_attachments"
    additional_subdir = "opportunity_additional_links"

    reset_attachment = _requeue_extraction_rows(
        model=OpportunityAttachment,
        force_reextract=force_reextract,
        retry_failed_extraction=retry_failed_extraction,
    )
    reset_additional = _requeue_extraction_rows(
        model=OpportunityAdditionalInfo,
        force_reextract=force_reextract,
        retry_failed_extraction=retry_failed_extraction,
    )

    attachment_stats = run_extraction_pipeline(
        model=OpportunityAttachment,
        subdir=attachment_subdir,
        url_getter=lambda row: row.file_download_path,
        **common,
    )
    additional_stats = run_extraction_pipeline(
        model=OpportunityAdditionalInfo,
        subdir=additional_subdir,
        url_getter=lambda row: row.additional_info_url,
        **common,
    )

    return {
        "requeued": {
            "attachment_rows": int(reset_attachment),
            "additional_info_rows": int(reset_additional),
        },
        "extraction": {
            "attachments": attachment_stats,
            "additional_info": additional_stats,
        },
    }


def _sync_all_grants(
    *,
    settings_neo4j: Neo4jSettings,
    limit: int,
    offset: int,
    max_additional_info: int,
    max_attachments: int,
    max_text_chars: int,
    stop_on_error: bool,
) -> Dict[str, Any]:
    rows = _load_opportunities(
        opportunity_id="",
        sync_all=True,
        limit=max(0, int(limit or 0)),
        offset=max(0, int(offset or 0)),
    )
    if not rows:
        raise RuntimeError("No opportunities found to sync.")

    limits = SyncLimits(
        max_additional_info=_safe_limit(max_additional_info, default=100, minimum=1, maximum=1000),
        max_attachments=_safe_limit(max_attachments, default=100, minimum=1, maximum=2000),
        max_text_chars=_safe_limit(max_text_chars, default=4000, minimum=100, maximum=50000),
    )

    synced: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with GraphDatabase.driver(
        settings_neo4j.uri,
        auth=(settings_neo4j.username, settings_neo4j.password),
    ) as driver:
        driver.verify_connectivity()

        for opp in rows:
            opportunity_id = safe_text(getattr(opp, "opportunity_id", None)) or ""
            try:
                result = sync_grant_to_neo4j(
                    driver=driver,
                    settings=settings_neo4j,
                    opp=opp,
                    limits=limits,
                    include_extracted_text=True,
                )
                synced.append(result)
            except Exception as exc:
                errors.append(
                    {
                        "opportunity_id": opportunity_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if stop_on_error:
                    break

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

    return {
        "scope": {
            "all": True,
            "limit": max(0, int(limit or 0)),
            "offset": max(0, int(offset or 0)),
            "include_extracted_text": True,
        },
        "totals": totals,
        "synced": synced,
        "errors": errors,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end grant sync: extract attachment/additional-info text to S3 paths, "
            "then sync all grants to Neo4j with extracted text."
        )
    )

    parser.add_argument("--limit", type=int, default=0, help="Optional sync limit (0 = all).")
    parser.add_argument("--offset", type=int, default=0, help="Optional sync offset.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop Neo4j sync on first grant error.")

    parser.add_argument("--extract-batch-size", type=int, default=200, help="Extraction batch size.")
    parser.add_argument(
        "--retry-failed-extraction",
        action="store_true",
        help="Requeue previously failed extraction rows before extraction run.",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Requeue all extraction rows and overwrite existing extracted metadata.",
    )

    parser.add_argument("--max-additional-info", type=int, default=100, help="Max additional-info rows per grant synced.")
    parser.add_argument("--max-attachments", type=int, default=100, help="Max attachment rows per grant synced.")
    parser.add_argument("--max-text-chars", type=int, default=4000, help="Max chars per extracted text block loaded into graph.")

    parser.add_argument("--skip-schema-init", action="store_true", help="Skip Neo4j schema init step.")
    parser.add_argument("--json-only", action="store_true", help="Print only JSON summary.")

    parser.add_argument("--uri", type=str, default="", help="Neo4j URI. Fallback: NEO4J_URI")
    parser.add_argument("--username", type=str, default="", help="Neo4j username. Fallback: NEO4J_USERNAME")
    parser.add_argument("--password", type=str, default="", help="Neo4j password. Fallback: NEO4J_PASSWORD")
    parser.add_argument("--database", type=str, default="", help="Neo4j database. Fallback: NEO4J_DATABASE or neo4j")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    load_dotenv_if_present()

    settings_neo4j = read_neo4j_settings(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    if not args.skip_schema_init:
        init_neo4j_schema(
            settings_neo4j,
            include_grant=True,
            include_faculty=False,
        )

    extraction_report = _extract_opportunity_content(
        extract_batch_size=args.extract_batch_size,
        force_reextract=bool(args.force_reextract),
        retry_failed_extraction=bool(args.retry_failed_extraction),
    )

    sync_report = _sync_all_grants(
        settings_neo4j=settings_neo4j,
        limit=args.limit,
        offset=args.offset,
        max_additional_info=args.max_additional_info,
        max_attachments=args.max_attachments,
        max_text_chars=args.max_text_chars,
        stop_on_error=bool(args.stop_on_error),
    )

    payload = {
        "extraction": extraction_report,
        "sync": sync_report,
    }

    if not args.json_only:
        totals = sync_report.get("totals", {})
        print("Grant full pipeline complete.")
        print("  extraction attachment done :", extraction_report["extraction"]["attachments"].get("done", 0))
        print("  extraction attachment fail :", extraction_report["extraction"]["attachments"].get("failed", 0))
        print("  extraction additional done :", extraction_report["extraction"]["additional_info"].get("done", 0))
        print("  extraction additional fail :", extraction_report["extraction"]["additional_info"].get("failed", 0))
        print("  synced grants             :", totals.get("grants_synced", 0))
        print("  failed grants             :", totals.get("grants_failed", 0))
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
