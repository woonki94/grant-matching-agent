import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../root
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from config import settings
from logging_setup import setup_logging

from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from db.models.opportunity import OpportunityAdditionalInfo, OpportunityAttachment
from services.extract_content import run_extraction_pipeline
from services.opportunity.call_opportunity import run_search_pipeline

logger = logging.getLogger("import_opportunity")
setup_logging()


def import_opportunity(page_size: int, query: str | None) -> None:
    # -------------------------
    # 1) Fetch
    # -------------------------
    logger.info("Starting opportunity pipeline (Fetching %s Opportunities)", page_size)
    opportunities = run_search_pipeline(page_size=page_size, q=query)
    logger.info("[1/3 FETCH] Completed (%d opportunities)", len(opportunities))

    # -------------------------
    # 2) Upsert
    # -------------------------
    with SessionLocal() as sess:
        opp_dao = OpportunityDAO(sess)
        with logging_redirect_tqdm():
            for opp in tqdm(opportunities, desc="Upserting opportunities", unit="opp"):
                opp_dao.upsert_opportunity(opp)
                opp_dao.upsert_attachments(opp.opportunity_id, opp.attachments)
                opp_dao.upsert_additional_info(opp.opportunity_id, opp.additional_info)
        sess.commit()
    logger.info("[2/3 UPSERT] Completed")

    # -------------------------
    # 3) Extract (Local or S3)
    # -------------------------
    backend = settings.extracted_content_backend

    # IMPORTANT: use clean, stable subdir *names* for S3 keys
    # (Avoid passing local filesystem paths like "data/opportunity_attachments" unless you want those in S3 keys.)
    ATT_SUBDIR = "opportunity_attachments"
    LINK_SUBDIR = "opportunity_additional_links"

    if backend == "local":
        if settings.extracted_content_path is None:
            raise RuntimeError(
                "EXTRACTED_CONTENT_PATH must be set when EXTRACTED_CONTENT_BACKEND=local"
            )

        base_dir = settings.extracted_content_path
        base_dir.mkdir(parents=True, exist_ok=True)

        stats = run_extraction_pipeline(
            model=OpportunityAttachment,
            base_dir=base_dir,
            subdir=ATT_SUBDIR,
            url_getter=lambda a: a.file_download_path,
            backend="local",
        )
        logger.info("[3/3 EXTRACT:ATTACHMENTS] Completed %s", stats)

        stats = run_extraction_pipeline(
            model=OpportunityAdditionalInfo,
            base_dir=base_dir,
            subdir=LINK_SUBDIR,
            url_getter=lambda a: a.additional_info_url,
            backend="local",
        )
        logger.info("[3/3 EXTRACT:LINKS] Completed %s", stats)

    elif backend == "s3":
        if not settings.extracted_content_bucket:
            raise RuntimeError(
                "EXTRACTED_CONTENT_BUCKET must be set when EXTRACTED_CONTENT_BACKEND=s3"
            )

        stats = run_extraction_pipeline(
            model=OpportunityAttachment,
            base_dir=None,
            subdir=ATT_SUBDIR,
            url_getter=lambda a: a.file_download_path,
            backend="s3",
            s3_bucket=settings.extracted_content_bucket,
            s3_prefix=settings.extracted_content_prefix,
            aws_region=settings.aws_region,
            aws_profile=settings.aws_profile,
        )
        logger.info("[3/3 EXTRACT:ATTACHMENTS] Completed %s", stats)

        stats = run_extraction_pipeline(
            model=OpportunityAdditionalInfo,
            base_dir=None,
            subdir=LINK_SUBDIR,
            url_getter=lambda a: a.additional_info_url,
            backend="s3",
            s3_bucket=settings.extracted_content_bucket,
            s3_prefix=settings.extracted_content_prefix,
            aws_region=settings.aws_region,
            aws_profile=settings.aws_profile,
        )
        logger.info("[3/3 EXTRACT:LINKS] Completed %s", stats)

    else:
        raise RuntimeError(f"Unsupported EXTRACTED_CONTENT_BACKEND={backend}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import opportunities pipeline")

    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Number of opportunities to fetch",
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Search query string",
    )

    args = parser.parse_args()

    import_opportunity(
        page_size=args.page_size,
        query=args.query,
    )
