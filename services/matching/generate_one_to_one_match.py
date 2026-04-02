from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dao.match_dao import MatchDAO
from db.db_conn import SessionLocal
from services.matching.faculty_grant_matcher import FacultyGrantMatcher
from services.matching.single_match_llm_reranker import OneToOneLLMReranker


def run_match_and_rerank(
    *,
    k: int,
    min_domain: float,
    limit_faculty: int,
    commit_every: int = 30,
) -> int:
    matcher = FacultyGrantMatcher()
    return matcher.run(
        k=int(k),
        min_domain=float(min_domain),
        limit_faculty=int(limit_faculty),
        commit_every=int(commit_every),
    )


def run_rerank_only(
    *,
    limit_faculty: int,
    max_context_chars: int,
    rerank_workers: int,
) -> dict:
    reranker = OneToOneLLMReranker()
    summary = reranker.run(
        limit_faculty=int(limit_faculty),
        max_context_chars=int(max_context_chars),
        workers=int(rerank_workers),
    )

    outputs = list(summary.get("results") or [])
    total_updated = 0
    faculty_updated = 0

    with SessionLocal() as sess:
        mdao = MatchDAO(sess)
        for row in outputs:
            try:
                fid = int((row or {}).get("faculty_id") or 0)
            except Exception:
                fid = 0
            if fid <= 0:
                continue

            grant_scores = {}
            for item in list((row or {}).get("reranked_grants") or []):
                oid = str((item or {}).get("opportunity_id") or "").strip()
                if not oid:
                    continue
                try:
                    grant_scores[oid] = float((item or {}).get("llm_score") or 0.0)
                except Exception:
                    continue
            if not grant_scores:
                row["updated_rows"] = int((row or {}).get("updated_rows") or 0)
                continue

            updated = mdao.update_llm_scores_for_faculty(
                faculty_id=int(fid),
                grant_scores=grant_scores,
            )
            row["updated_rows"] = int(updated)
            total_updated += int(updated)
            if int(updated) > 0:
                faculty_updated += 1
        sess.commit()

    summary["updated_match_rows"] = int(total_updated)
    summary["faculty_updated"] = int(faculty_updated)
    summary["persisted"] = True
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one-to-one faculty-grant matching or rerank-only."
    )
    parser.add_argument(
        "--mode",
        choices=["match_and_rerank", "rerank_only"],
        default="match_and_rerank",
        help="Pipeline mode. Default: match_and_rerank",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-domain", type=float, default=0.0)
    parser.add_argument("--limit-faculty", type=int, default=100)
    parser.add_argument("--commit-every", type=int, default=30)
    parser.add_argument("--rerank-workers", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=100000)
    args = parser.parse_args()

    if str(args.mode) == "rerank_only":
        summary = run_rerank_only(
            limit_faculty=args.limit_faculty,
            max_context_chars=args.max_context_chars,
            rerank_workers=args.rerank_workers,
        )
        print("One-to-one rerank-only completed.")
        print(json.dumps(summary, ensure_ascii=False))
    else:
        processed = run_match_and_rerank(
            k=args.k,
            min_domain=args.min_domain,
            limit_faculty=args.limit_faculty,
            commit_every=args.commit_every,
        )
        print(f"One-to-one matching completed. Faculty processed: {processed}")
