from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../root
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, List
from sqlalchemy.orm import Session, sessionmaker
from db.db_conn import engine

from services.faculty.crawler import crawl, save_profile_dict
from services.faculty.profile_parser import parse_profile
from services.faculty.publication_enricher import enrich_faculty_publications

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def main():
    """
    Args:
      max_pages: how many faculty listing pages to crawl (pagination depth)
      publication_years_back: how far back to fetch publications per faculty
    """
    max_faculty_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    publication_years_back = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Running faculty import:")
    print(f"  max_faculty_pages  = {max_faculty_pages}")
    print(f"  publicationyears_back = {publication_years_back}")
    print()

    links = crawl(max_pages=max_faculty_pages)

    for link in links:
        try:
            profile = parse_profile(link)
            faculty_id = save_profile_dict(profile)

            # Skip publications if name is missing
            if not profile.get("name"):
                print(f"[SKIP PUBS] No name for {link}")
                continue

            num_pubs = enrich_faculty_publications(
                faculty_id=faculty_id,
                years_back=publication_years_back
            )

            print(f"[OK] {profile.get('name')} (id={faculty_id}) -> {num_pubs} pubs")

        except Exception as e:
            # Don't let one bad profile kill the whole run
            print(f"[ERROR] link={link} error={e}")


if __name__ == '__main__':
    main()