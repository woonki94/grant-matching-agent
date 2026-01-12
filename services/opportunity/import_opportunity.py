import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../root
sys.path.insert(0, str(PROJECT_ROOT))

import logging

from logging_setup import setup_logging

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from config import settings
from db.models.opportunity import OpportunityAdditionalInfo, OpportunityAttachment
from services.extract_content import run_extraction_pipeline

from services.opportunity.call_opportunity import run_search_pipeline
from db.db_conn import SessionLocal
from dao.opportunity_dao import OpportunityDAO

import argparse

logger = logging.getLogger("import_opportunity")
setup_logging()

CONTENT_BASE_DIR = settings.extracted_content_path
ATT_SUB_DIR = settings.opportunity_attachment_path
LINK_SUB_DIR = settings.opportunity_additional_link_path



def import_opportunity(page_size, query) -> None:

    # 0) Ensure base dir exists
    CONTENT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Fetch
    logger.info("Starting opportunity pipeline (Fetching %s Opportunities)", page_size)

    opportunities = run_search_pipeline(page_size=page_size ,q= query)
    logger.info("[1/3 FETCH] Completed (%d opportunities)", len(opportunities))

    with SessionLocal() as sess:
        opp_dao = OpportunityDAO(sess)
        with logging_redirect_tqdm():
            #TODO: Bulk updates(batch size)
            for opp in tqdm(opportunities, desc="Upserting opportunities", unit="opp"):
                opp_dao.upsert_opportunity(opp)
                opp_dao.upsert_attachments(opp.opportunity_id, opp.attachments)
                opp_dao.upsert_additional_info(opp.opportunity_id, opp.additional_info)
        sess.commit()
    logger.info("[2/3 UPSERT] Completed")

    stats = run_extraction_pipeline(
        model=OpportunityAttachment,
        base_dir=CONTENT_BASE_DIR,
        subdir=ATT_SUB_DIR,
        url_getter=lambda a: a.file_download_path,
    )
    logger.info("[3/3 EXTRACT:ATTACHMENTS] Completed %s", stats)

    stats = run_extraction_pipeline(
        model=OpportunityAdditionalInfo,
        base_dir=CONTENT_BASE_DIR,
        subdir=LINK_SUB_DIR,
        url_getter=lambda a: a.additional_info_url,
    )
    logger.info("[3/3 EXTRACT:LINKS] Completed %s", stats)


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