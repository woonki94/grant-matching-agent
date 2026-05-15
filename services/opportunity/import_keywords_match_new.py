from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from db.db_conn import SessionLocal
from db.models.faculty import Faculty
from db.models.opportunity import Opportunity
from logging_setup import setup_logging
from services.context_retrieval.context_generator import ContextGenerator
from services.keywords.keyword_generator import OpportunityKeywordGenerator
from services.matching.faculty_grant_matcher import FacultyGrantMatcher
from services.opportunity.import_opportunity import import_opportunity

logger = logging.getLogger("import_keywords_match_new")
setup_logging()


def _parse_agencies(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    tokens = [x.strip() for x in str(raw).split(",")]
    agencies = [x for x in tokens if x]
    return agencies or None


def _all_opportunity_ids() -> Set[str]:
    with SessionLocal() as sess:
        rows = sess.query(Opportunity.opportunity_id).all()
    return {str(oid) for (oid,) in rows if str(oid or "").strip()}


def _all_faculty_ids() -> List[int]:
    with SessionLocal() as sess:
        rows = sess.query(Faculty.faculty_id).order_by(Faculty.faculty_id.asc()).all()
    return [int(fid) for (fid,) in rows if fid is not None]


def run_pipeline(
    *,
    page_size: int,
    query: Optional[str],
    agencies: Optional[List[str]],
    fetch_workers: int,
    extract_workers: int,
    min_domain: float,
    rerank_workers: int,
    rerank_chunk_workers: Optional[int],
) -> Dict[str, object]:
    before_ids = _all_opportunity_ids()
    logger.info("Snapshot before import: opportunities=%s", len(before_ids))

    import_opportunity(
        page_size=int(page_size),
        query=(str(query) if query else None),
        agencies=(list(agencies) if agencies else None),
        fetch_workers=int(fetch_workers),
        extract_workers=int(extract_workers),
    )

    after_ids = _all_opportunity_ids()
    new_ids = sorted(after_ids - before_ids)
    logger.info(
        "Snapshot after import: opportunities=%s new_in_this_run=%s",
        len(after_ids),
        len(new_ids),
    )

    if not new_ids:
        return {
            "imported_total_opportunities": int(len(after_ids)),
            "new_opportunity_count": 0,
            "new_opportunity_ids": [],
            "opportunity_keywords_generated": 0,
            "opportunity_keywords_failed": 0,
            "faculty_count_used_for_matching": int(len(_all_faculty_ids())),
            "matches_upserted_total": 0,
            "matches_upserted_by_opportunity": {},
            "min_domain": float(min_domain),
            "rerank_workers": int(rerank_workers),
            "rerank_chunk_workers": (
                int(rerank_chunk_workers) if rerank_chunk_workers is not None else None
            ),
        }

    context_generator = ContextGenerator()
    keyword_generator = OpportunityKeywordGenerator(context_generator=context_generator)

    generated = 0
    failed = 0
    for opportunity_id in new_ids:
        try:
            out = keyword_generator.generate_opportunity_keywords_for_id(
                str(opportunity_id),
                force_regenerate=False,
            )
            if out:
                generated += 1
            else:
                failed += 1
        except Exception:
            failed += 1
            logger.exception(
                "Keyword generation failed for new opportunity_id=%s",
                str(opportunity_id),
            )

    faculty_ids = _all_faculty_ids()
    matcher = FacultyGrantMatcher(session_factory=SessionLocal)

    match_rows_by_opp: Dict[str, int] = {}
    match_rows_total = 0
    for opportunity_id in new_ids:
        try:
            upserted = int(
                matcher.run_for_opportunity(
                    opportunity_id=str(opportunity_id),
                    faculty_ids=faculty_ids or None,
                    min_domain=float(min_domain),
                    rerank_workers=int(rerank_workers),
                    rerank_chunk_workers=(
                        int(rerank_chunk_workers)
                        if rerank_chunk_workers is not None
                        else None
                    ),
                )
            )
            match_rows_by_opp[str(opportunity_id)] = int(upserted)
            match_rows_total += int(upserted)
        except Exception:
            match_rows_by_opp[str(opportunity_id)] = 0
            logger.exception(
                "Matching failed for new opportunity_id=%s",
                str(opportunity_id),
            )

    return {
        "imported_total_opportunities": int(len(after_ids)),
        "new_opportunity_count": int(len(new_ids)),
        "new_opportunity_ids": [str(x) for x in new_ids],
        "opportunity_keywords_generated": int(generated),
        "opportunity_keywords_failed": int(failed),
        "faculty_count_used_for_matching": int(len(faculty_ids)),
        "matches_upserted_total": int(match_rows_total),
        "matches_upserted_by_opportunity": match_rows_by_opp,
        "min_domain": float(min_domain),
        "rerank_workers": int(rerank_workers),
        "rerank_chunk_workers": (
            int(rerank_chunk_workers) if rerank_chunk_workers is not None else None
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Fetch opportunities, then generate keywords and match rows "
            "only for opportunities newly inserted in this run."
        )
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument(
        "--agencies",
        type=str,
        default=None,
        help="Comma-separated agency codes (example: HHS-NIH11,NSF).",
    )
    parser.add_argument("--fetch-workers", type=int, default=8)
    parser.add_argument("--extract-workers", type=int, default=4)
    parser.add_argument(
        "--min-domain",
        type=float,
        default=0.3,
        help="Cosine-similarity threshold for matching.",
    )
    parser.add_argument(
        "--rerank-workers",
        type=int,
        default=4,
        help="Worker count for LLM reranking stage.",
    )
    parser.add_argument(
        "--rerank-chunk-workers",
        type=int,
        default=None,
        help="Optional chunk worker override for per-faculty rerank chunks.",
    )
    args = parser.parse_args()

    summary = run_pipeline(
        page_size=int(args.page_size),
        query=(str(args.query) if args.query else None),
        agencies=_parse_agencies(args.agencies),
        fetch_workers=int(args.fetch_workers),
        extract_workers=int(args.extract_workers),
        min_domain=float(args.min_domain),
        rerank_workers=int(args.rerank_workers),
        rerank_chunk_workers=(
            int(args.rerank_chunk_workers)
            if args.rerank_chunk_workers is not None
            else None
        ),
    )
    print(json.dumps(summary, ensure_ascii=False))
