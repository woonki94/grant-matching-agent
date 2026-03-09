from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from services.keywords.faculty_keyword_generator_v2 import FacultyKeywordGeneratorV2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate faculty keywords v2 from Neo4j chunks only")
    parser.add_argument("--faculty-id", type=int, default=0, help="Generate for one faculty_id only (0 = all).")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers when running all faculties.")
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist generated keywords to relational DB.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=40000,
        help="Max chars per faculty context payload.",
    )
    parser.add_argument(
        "--max-neo4j-chunks",
        type=int,
        default=4000,
        help="Max chunk rows fetched per faculty from Neo4j.",
    )
    parser.add_argument(
        "--reserve-prompt-chars",
        type=int,
        default=3000,
        help="Reserved chars for prompt/system overhead.",
    )
    args = parser.parse_args()

    keyword_service = FacultyKeywordGeneratorV2(
        max_context_chars=args.max_context_chars,
        max_neo4j_chunks=args.max_neo4j_chunks,
        reserve_prompt_chars=args.reserve_prompt_chars,
    )

    if int(args.faculty_id or 0) > 0:
        result = keyword_service.generate_faculty_keywords_for_id(
            int(args.faculty_id),
            max_context_chars=args.max_context_chars,
            persist=bool(args.persist),
        )
        print(json.dumps(result, ensure_ascii=False))
    else:
        summary = keyword_service.run_all_faculty_keyword_pipelines_parallel(
            max_workers=max(1, int(args.max_workers)),
            max_context_chars=args.max_context_chars,
            persist=bool(args.persist),
        )
        print(summary)
