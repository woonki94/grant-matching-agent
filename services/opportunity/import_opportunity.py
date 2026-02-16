import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
from services.opportunity.call_opportunity import OpportunitySearchService

logger = logging.getLogger("import_opportunity")
setup_logging()


def import_opportunity(page_size: int, query: str | None, opp_id: str | None = None) -> None:
    search_service = OpportunitySearchService()
    # -------------------------
    # 1) Fetch
    # -------------------------
    if opp_id:
        logger.info("Starting opportunity pipeline for single opportunity_id=%s", opp_id)
        opportunities = search_service.run_search_pipeline(opportunity_id=opp_id)
    else:
        logger.info("Starting opportunity pipeline (Fetching %s Opportunities)", page_size)
        opportunities = search_service.run_search_pipeline(page_size=page_size, q=query, agencies=["HHS-NIH11"])
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
    # 3) Extract (S3 ONLY)
    # -------------------------
    if not settings.extracted_content_bucket:
        raise RuntimeError("EXTRACTED_CONTENT_BUCKET must be set")

    # Keep S3 keys stable + clean
    att_subdir = "opportunity_attachments"
    link_subdir = "opportunity_additional_links"

    common = dict(
        s3_bucket=settings.extracted_content_bucket,
        s3_prefix=settings.extracted_content_prefix_opportunity,
        aws_region=settings.aws_region,
        aws_profile=settings.aws_profile,
    )

    stats = run_extraction_pipeline(
        model=OpportunityAttachment,
        subdir=att_subdir,
        url_getter=lambda a: a.file_download_path,
        **common,
    )
    logger.info("[3/3 EXTRACT:ATTACHMENTS] Completed %s", stats)

    stats = run_extraction_pipeline(
        model=OpportunityAdditionalInfo,
        subdir=link_subdir,
        url_getter=lambda a: a.additional_info_url,
        **common,
    )
    logger.info("[3/3 EXTRACT:LINKS] Completed %s", stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import opportunities pipeline")

    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--opp-id", type=str, default=None, help="Import a single opportunity by ID")

    args = parser.parse_args()

    import_opportunity(page_size=args.page_size, query=args.query, opp_id=args.opp_id)
