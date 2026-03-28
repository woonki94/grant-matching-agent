from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from services.context_retrieval.context_generator import ContextGenerator
from services.keywords.keyword_generator import FacultyKeywordGenerator, OpportunityKeywordGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate keywords for faculty/opportunities (Bedrock-only)")
    parser.add_argument("--limit", type=int, default=0, help="Max number of records to process (0 = no limit)")
    parser.add_argument("--faculty-only", action="store_true", help="Only generate faculty keywords")
    parser.add_argument("--opp-only", action="store_true", help="Only generate opportunity keywords")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Thread workers per batch (faculty/opportunity).",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate keywords for rows that already have keyword rows.",
    )
    args = parser.parse_args()

    run_faculty = not args.opp_only
    run_opp = not args.faculty_only
    context_generator = ContextGenerator()
    faculty_keyword_service = FacultyKeywordGenerator(
        context_generator=context_generator,
        force_regenerate=args.force_regenerate,
    )
    opportunity_keyword_service = OpportunityKeywordGenerator(
        context_generator=context_generator,
        force_regenerate=args.force_regenerate,
    )
    if run_faculty:
        faculty_keyword_service.run_batch(
            limit=args.limit,
            force_regenerate=args.force_regenerate,
            workers=args.workers,
        )
    if run_opp:
        opportunity_keyword_service.run_batch(
            limit=args.limit,
            force_regenerate=args.force_regenerate,
            workers=args.workers,
        )
