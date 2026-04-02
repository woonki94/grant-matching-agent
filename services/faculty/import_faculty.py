import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../root
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
from typing import Any, Dict

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from logging_setup import setup_logging
from config import get_embedding_client, settings

from dao.faculty_dao import FacultyDAO
from db.db_conn import SessionLocal
from db.models.faculty import FacultyAdditionalInfo
from mappers.page_to_faculty import map_faculty_profile_to_dto
from services.extract_content import run_extraction_pipeline
from services.faculty.faculty_page_crawler import crawl
from services.faculty.profile_parser import parse_profile
from utils.publication_enricher import get_publication_dtos_for_faculty
from utils.thread_pool import parallel_map, resolve_pool_size

# Bulk import currently enriches faculty with recent OpenAlex publications.

logger = logging.getLogger("import_faculty")
setup_logging()

UNIV_NAME = settings.university_name


def _prepare_faculty_payload(link: str, *, years_back: int) -> Dict[str, Any]:
    profile = parse_profile(link)
    dto = map_faculty_profile_to_dto(profile)
    full_name = str(dto.name or "").strip()

    pubs = []
    if full_name:
        _author_id, pubs = get_publication_dtos_for_faculty(
            full_name=full_name,
            university=UNIV_NAME,
            years_back=years_back,
        )
    return {
        "link": link,
        "dto": dto,
        "publications": pubs,
        "error": None,
    }


def import_faculty(
    max_pages: int,
    max_faculty: int,
    years_back: int,
    *,
    workers: int = 8,
    extract_workers: int = 4,
) -> None:
    # -------------------------
    # 1) Fetch links
    # -------------------------
    logger.info(
        "Starting faculty import: max_pages=%s, max_faculty=%s, years_back=%s, workers=%s, extract_workers=%s",
        max_pages,
        max_faculty,
        years_back,
        workers,
        extract_workers,
    )

    links = crawl(max_pages=max_pages, max_links=max_faculty)
    logger.info("[1/3 FETCH] Completed (%d links)", len(links))

    prep_pool_size = resolve_pool_size(max_workers=workers, task_count=len(links))
    logger.info("[2/3 PREP] Building faculty payloads with workers=%s", prep_pool_size)

    def _on_prepare_error(_idx: int, link: str, exc: Exception) -> Dict[str, Any]:
        logger.exception("Failed processing faculty link: %s", link)
        return {
            "link": link,
            "dto": None,
            "publications": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    prepared = parallel_map(
        links,
        max_workers=prep_pool_size,
        run_item=lambda link: _prepare_faculty_payload(link, years_back=years_back),
        on_error=_on_prepare_error,
    )

    # -------------------------
    # 2) Upsert faculty + additional_info + publications
    # -------------------------
    with SessionLocal() as sess:
        fac_dao = FacultyDAO(sess)
        embedding_client = get_embedding_client().build()

        with logging_redirect_tqdm():
            for payload in tqdm(prepared, desc="Upserting faculty", unit="faculty"):
                link = str(payload.get("link") or "")
                if payload.get("error") or payload.get("dto") is None:
                    logger.warning(
                        "Skipping faculty link due to pre-processing error: %s | %s",
                        link,
                        payload.get("error") or "unknown_error",
                    )
                    continue
                try:
                    dto = payload["dto"]
                    pubs = list(payload.get("publications") or [])

                    faculty = fac_dao.upsert_faculty(dto)
                    sess.flush()  # ensure faculty_id exists

                    fac_dao.upsert_additional_info(
                        faculty.faculty_id,
                        dto.additional_info,
                    )
                    fac_dao.upsert_publications(
                        faculty.faculty_id,
                        pubs,
                        embedding_client=embedding_client,
                    )

                except Exception:
                    logger.exception("Failed processing faculty link: %s", link)

        sess.commit()

    logger.info("[2/3 UPSERT] Completed")

    # -------------------------
    # 3) Extract content from additional links (S3 ONLY)
    # -------------------------
    if not settings.extracted_content_bucket:
        raise RuntimeError("EXTRACTED_CONTENT_BUCKET must be set")

    # Writes to:
    # s3://<bucket>/<EXTRACTED_CONTENT_PREFIX_FACULTY>/faculties_additional_links/<id>__<hash>.txt
    LINK_SUBDIR = "faculties_additional_links"

    stats = run_extraction_pipeline(
        model=FacultyAdditionalInfo,
        subdir=LINK_SUBDIR,
        url_getter=lambda a: a.additional_info_url,
        max_workers=extract_workers,
        s3_bucket=settings.extracted_content_bucket,
        s3_prefix=settings.extracted_content_prefix_faculty,
        aws_region=settings.aws_region,
        aws_profile=settings.aws_profile,
    )
    logger.info("[3/3 EXTRACT:LINKS] Completed %s", stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import faculty pipeline")

    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Number of pages to crawl (0 = first page only, depending on your crawler)",
    )
    parser.add_argument(
        "--years-back",
        type=int,
        default=5,
        help="How many years back to fetch publications from OpenAlex",
    )


    parser.add_argument(
        "--max-faculty",
        type=int,
        default=0,
        help="Maximum number of faculty links to crawl (0 = no limit)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Thread workers for faculty page/profile/publication pre-processing.",
    )
    parser.add_argument(
        "--extract-workers",
        type=int,
        default=4,
        help="Thread workers for extraction/chunking/embedding/S3 upload.",
    )

    args = parser.parse_args()

    import_faculty(
        max_pages=args.max_pages,
        years_back=args.years_back,
        max_faculty=args.max_faculty,
        workers=args.workers,
        extract_workers=args.extract_workers,
    )
