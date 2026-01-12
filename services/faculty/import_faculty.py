import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../root
sys.path.insert(0, str(PROJECT_ROOT))


import logging

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from logging_setup import setup_logging
from config import settings

import argparse


from dao.faculty_dao import FacultyDAO
from db.db_conn import SessionLocal
from db.models.faculty import FacultyAdditionalInfo
from mappers.page_to_faculty import map_faculty_profile_to_dto
from services.extract_content import run_extraction_pipeline
from services.faculty.faculty_page_crawler import crawl
from services.faculty.profile_parser import parse_profile
from utils.publication_enricher import get_publication_dtos_for_faculty


logger = logging.getLogger("import_faculty")
setup_logging()

CONTENT_BASE_DIR = settings.extracted_content_path
LINK_SUB_DIR = settings.faculty_additional_link_path
UNIV_NAME = settings.university_name

def import_faculty(max_pages, years_back) -> None:

    # 0) Ensure base dir exists
    CONTENT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Fetch links
    logger.info(
        "Starting faculty import: max_pages=%s, publications from last %s years",
        max_pages,
        years_back,
    )

    links = crawl(max_pages=max_pages)
    logger.info("[1/3 FETCH] Completed (%d links)", len(links))

    # 2) Upsert faculty + additional_info + publications
    with SessionLocal() as sess:
        fac_dao = FacultyDAO(sess)

        #TODO: Find out what makes fetching slow?
        with logging_redirect_tqdm():
            for link in tqdm(links, desc="Upserting faculty", unit="faculty"):
                try:
                    profile = parse_profile(link)
                    dto = map_faculty_profile_to_dto(profile)

                    faculty = fac_dao.upsert_faculty(dto)
                    sess.flush()  # ensure faculty_id exists

                    fac_dao.upsert_additional_info(faculty.faculty_id, dto.additional_info)

                    author_id, pubs = get_publication_dtos_for_faculty(
                        full_name=faculty.name,
                        university=UNIV_NAME,
                        years_back=years_back
                    )
                    fac_dao.upsert_publications(faculty.faculty_id, pubs)

                except Exception:
                    logger.exception("Failed processing faculty link: %s", link)

        sess.commit()

    logger.info("[2/3 UPSERT] Completed")

    # 3) Extract content from additional links
    stats = run_extraction_pipeline(
        model=FacultyAdditionalInfo,
        base_dir=CONTENT_BASE_DIR,
        subdir=LINK_SUB_DIR,
        url_getter=lambda a: a.additional_info_url,
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

    args = parser.parse_args()

    import_faculty(
        max_pages=args.max_pages,
        years_back=args.years_back,
    )