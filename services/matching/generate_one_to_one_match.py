from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from services.matching.faculty_grant_matcher import (
    FacultyGrantMatcher
)

def main(k: int, min_domain: float, limit_faculty: int, commit_every: int = 30):
    service = FacultyGrantMatcher()
    service.run(
        k=k,
        min_domain=min_domain,
        limit_faculty=limit_faculty,
        commit_every=commit_every,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legacy entrypoint. Use faculty_grant_matcher.py.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-domain", type=float, default=0.30)
    parser.add_argument("--limit-faculty", type=int, default=100)
    parser.add_argument("--commit-every", type=int, default=30)
    args = parser.parse_args()

    main(
        k=args.k,
        min_domain=args.min_domain,
        limit_faculty=args.limit_faculty,
        commit_every=args.commit_every,
    )
