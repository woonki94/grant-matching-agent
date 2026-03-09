from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from neo4j import GraphDatabase

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import settings
from db.db_conn import SessionLocal
from db.models.faculty import FacultyAdditionalInfo
from graph_rag.common import (
    Neo4jSettings,
    json_ready,
    load_dotenv_if_present,
    read_neo4j_settings,
    safe_text,
)
from graph_rag.faculty.sync_neo4j import (
    SyncLimits,
    _load_faculties,
    _safe_limit,
    sync_faculty_to_neo4j,
)
from graph_rag.schema import init_neo4j_schema
from services.extract_content import run_extraction_pipeline
from utils.thread_pool import parallel_map


def _requeue_extraction_rows(
    *,
    force_reextract: bool,
    retry_failed_extraction: bool,
) -> int:
    with SessionLocal() as sess:
        if force_reextract:
            rows = sess.query(FacultyAdditionalInfo).all()
        elif retry_failed_extraction:
            rows = sess.query(FacultyAdditionalInfo).filter(FacultyAdditionalInfo.extract_status == "failed").all()
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


def _extract_faculty_content(
    *,
    extract_batch_size: int,
    extract_workers: int,
    force_reextract: bool,
    retry_failed_extraction: bool,
) -> Dict[str, Any]:
    bucket = (settings.extracted_content_bucket or "").strip()
    if not bucket:
        raise RuntimeError("EXTRACTED_CONTENT_BUCKET must be set before extraction.")

    reset_count = _requeue_extraction_rows(
        force_reextract=force_reextract,
        retry_failed_extraction=retry_failed_extraction,
    )

    safe_extract_workers = _safe_limit(extract_workers, default=8, minimum=1, maximum=64)

    stats = run_extraction_pipeline(
        model=FacultyAdditionalInfo,
        subdir="faculties_additional_links",
        url_getter=lambda row: row.additional_info_url,
        batch_size=_safe_limit(extract_batch_size, default=200, minimum=1, maximum=2000),
        max_workers=safe_extract_workers,
        s3_bucket=bucket,
        s3_prefix=settings.extracted_content_prefix_faculty,
        aws_region=settings.aws_region,
        aws_profile=settings.aws_profile,
    )

    return {
        "requeued": int(reset_count),
        "extraction": {
            **dict(stats or {}),
            "workers": int(safe_extract_workers),
        },
    }


def _sync_all_faculties(
    *,
    settings_neo4j: Neo4jSettings,
    limit: int,
    offset: int,
    max_publications: int,
    max_additional_info: int,
    max_text_chars: int,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    max_chunks_per_source: int,
    include_embeddings: bool,
    embedding_batch_size: int,
    extract_workers: int,
    sync_workers: int,
    extracted_text_workers: int,
    stop_on_error: bool,
) -> Dict[str, Any]:
    rows = _load_faculties(
        email="",
        sync_all=True,
        limit=max(0, int(limit or 0)),
        offset=max(0, int(offset or 0)),
    )
    if not rows:
        raise RuntimeError("No faculty rows found to sync.")

    limits = SyncLimits(
        max_publications=_safe_limit(max_publications, default=300, minimum=1, maximum=3000),
        max_additional_info=_safe_limit(max_additional_info, default=150, minimum=1, maximum=2000),
        max_text_chars=_safe_limit(max_text_chars, default=4000, minimum=100, maximum=50000),
        chunk_size_chars=_safe_limit(chunk_size_chars, default=1200, minimum=200, maximum=5000),
        chunk_overlap_chars=_safe_limit(chunk_overlap_chars, default=150, minimum=0, maximum=2500),
        max_chunks_per_source=_safe_limit(max_chunks_per_source, default=24, minimum=1, maximum=1000),
    )

    synced: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    safe_sync_workers = _safe_limit(sync_workers, default=4, minimum=1, maximum=64)
    safe_s3_workers = _safe_limit(extracted_text_workers, default=8, minimum=1, maximum=64)

    with GraphDatabase.driver(
        settings_neo4j.uri,
        auth=(settings_neo4j.username, settings_neo4j.password),
    ) as driver:
        driver.verify_connectivity()

        def _run_one(fac) -> Dict[str, Any]:
            email = safe_text(getattr(fac, "email", None)) or ""
            result = sync_faculty_to_neo4j(
                driver=driver,
                settings_neo4j=settings_neo4j,
                fac=fac,
                limits=limits,
                include_extracted_text=True,
                include_embeddings=include_embeddings,
                embedding_batch_size=_safe_limit(embedding_batch_size, default=12, minimum=1, maximum=128),
                extracted_text_workers=safe_s3_workers,
            )
            return {
                "ok": True,
                "result": result,
                "faculty_id": int(getattr(fac, "faculty_id", 0) or 0),
                "email": str(email).strip().lower(),
            }

        def _on_error(index: int, fac, exc: Exception) -> Dict[str, Any]:
            email = safe_text(getattr(fac, "email", None)) or ""
            return {
                "ok": False,
                "result": None,
                "faculty_id": int(getattr(fac, "faculty_id", 0) or 0),
                "email": str(email).strip().lower(),
                "error": f"{type(exc).__name__}: {exc}",
            }

        if stop_on_error or safe_sync_workers <= 1:
            for fac in rows:
                try:
                    out = _run_one(fac)
                except Exception as exc:
                    out = _on_error(0, fac, exc)
                if out.get("ok"):
                    synced.append(dict(out.get("result") or {}))
                else:
                    errors.append(
                        {
                            "faculty_id": int(out.get("faculty_id") or 0),
                            "email": str(out.get("email") or "").strip().lower(),
                            "error": str(out.get("error") or ""),
                        }
                    )
                    if stop_on_error:
                        break
        else:
            results = parallel_map(
                rows,
                max_workers=min(safe_sync_workers, len(rows)),
                run_item=_run_one,
                on_error=_on_error,
            )
            for out in results:
                if out.get("ok"):
                    synced.append(dict(out.get("result") or {}))
                else:
                    errors.append(
                        {
                            "faculty_id": int(out.get("faculty_id") or 0),
                            "email": str(out.get("email") or "").strip().lower(),
                            "error": str(out.get("error") or ""),
                        }
                    )

    totals = {
        "faculties_synced": len(synced),
        "faculties_failed": len(errors),
        "additional_info": sum(int(item.get("counts", {}).get("additional_info", 0)) for item in synced),
        "publications": sum(int(item.get("counts", {}).get("publications", 0)) for item in synced),
        "keywords": sum(int(item.get("counts", {}).get("keywords", 0)) for item in synced),
        "text_chunks": sum(int(item.get("counts", {}).get("text_chunks", 0)) for item in synced),
        "embedded_specialization_keywords": sum(
            int(item.get("counts", {}).get("embedded_specialization_keywords", 0)) for item in synced
        ),
        "embedded_publication_abstracts": sum(
            int(item.get("counts", {}).get("embedded_publication_abstracts", 0)) for item in synced
        ),
        "embedded_text_chunks": sum(int(item.get("counts", {}).get("embedded_text_chunks", 0)) for item in synced),
    }

    return {
        "scope": {
            "all": True,
            "limit": max(0, int(limit or 0)),
            "offset": max(0, int(offset or 0)),
            "include_extracted_text": True,
            "include_embeddings": bool(include_embeddings),
            "extract_workers": _safe_limit(extract_workers, default=8, minimum=1, maximum=64),
            "sync_workers": safe_sync_workers,
            "extracted_text_workers": safe_s3_workers,
        },
        "totals": totals,
        "synced": synced,
        "errors": errors,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end faculty sync: extract additional-info text to S3 paths, "
            "then sync all faculties to Neo4j with extracted text + publication abstract embeddings + keyword embeddings."
        )
    )

    parser.add_argument("--limit", type=int, default=0, help="Optional sync limit (0 = all).")
    parser.add_argument("--offset", type=int, default=0, help="Optional sync offset.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop Neo4j sync on first faculty error.")

    parser.add_argument("--extract-batch-size", type=int, default=200, help="Extraction batch size.")
    parser.add_argument("--extract-workers", type=int, default=8, help="Parallel workers for extraction fetch/extract/upload.")
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

    parser.add_argument("--max-publications", type=int, default=300, help="Max publication rows per faculty synced.")
    parser.add_argument("--max-additional-info", type=int, default=150, help="Max additional-info rows per faculty synced.")
    parser.add_argument("--max-text-chars", type=int, default=4000, help="Max chars per extracted text block loaded into graph.")
    parser.add_argument("--chunk-size-chars", type=int, default=1200, help="Chunk size for additional-info text embeddings.")
    parser.add_argument("--chunk-overlap-chars", type=int, default=150, help="Chunk overlap for additional-info text embeddings.")
    parser.add_argument("--max-chunks-per-source", type=int, default=24, help="Max chunks for each additional-info document.")
    parser.add_argument("--embedding-batch-size", type=int, default=12, help="Embedding batch size.")
    parser.add_argument("--sync-workers", type=int, default=4, help="Parallel workers for per-faculty Neo4j sync.")
    parser.add_argument("--s3-read-workers", type=int, default=8, help="Parallel S3 read workers for loading extracted text.")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip specialization/publication/chunk embeddings.")

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
            include_grant=False,
            include_faculty=True,
        )

    extraction_report = _extract_faculty_content(
        extract_batch_size=args.extract_batch_size,
        extract_workers=args.extract_workers,
        force_reextract=bool(args.force_reextract),
        retry_failed_extraction=bool(args.retry_failed_extraction),
    )

    sync_report = _sync_all_faculties(
        settings_neo4j=settings_neo4j,
        limit=args.limit,
        offset=args.offset,
        max_publications=args.max_publications,
        max_additional_info=args.max_additional_info,
        max_text_chars=args.max_text_chars,
        chunk_size_chars=args.chunk_size_chars,
        chunk_overlap_chars=args.chunk_overlap_chars,
        max_chunks_per_source=args.max_chunks_per_source,
        include_embeddings=not bool(args.skip_embeddings),
        embedding_batch_size=args.embedding_batch_size,
        extract_workers=args.extract_workers,
        sync_workers=args.sync_workers,
        extracted_text_workers=args.s3_read_workers,
        stop_on_error=bool(args.stop_on_error),
    )

    payload = {
        "extraction": extraction_report,
        "sync": sync_report,
    }

    if not args.json_only:
        totals = sync_report.get("totals", {})
        print("Faculty full pipeline complete.")
        print("  extraction done              :", extraction_report["extraction"].get("done", 0))
        print("  extraction failed            :", extraction_report["extraction"].get("failed", 0))
        print("  synced faculties             :", totals.get("faculties_synced", 0))
        print("  failed faculties             :", totals.get("faculties_failed", 0))
        print("  embedded spec keywords       :", totals.get("embedded_specialization_keywords", 0))
        print("  embedded publication abstract:", totals.get("embedded_publication_abstracts", 0))
        print("  embedded text chunks         :", totals.get("embedded_text_chunks", 0))
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
