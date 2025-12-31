# services/matching/sem_match_report.py
from __future__ import annotations

# ---- make imports robust if someone runs the file directly ----
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---------------------------------------------------------------

import argparse
import re
from typing import Dict, List, Tuple

# Optional: load env (safe even if already loaded)
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv("api.env"), override=False)
except Exception:
    pass

from sqlalchemy.orm import Session

from db.db_conn import SessionLocal

# 👇 Import ALL models that are referenced by relationships so SQLAlchemy sees them
# If your models package already imports them in db/models/__init__.py, these three imports are enough:
from db.models.keywords_opportunity import Keyword as GrantKeyword
from db.models.keywords_faculty import FacultyKeyword as FacKeyword
from db.models.opportunity import Opportunity
from db.models.faculty import Faculty

from services.matching.semantic_matcher import score_faculty_vs_grant, rank_pairs


# ---------------------------- Helpers ----------------------------

def _normalize_terms(terms: List[str]) -> List[str]:
    out, seen = [], set()
    for t in terms or []:
        k = re.sub(r"\s+", " ", (t or "").strip().lower())
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _flatten(kw: Dict) -> List[str]:
    """Flatten the keyword JSON (research/app × domain/specialization) to a single list."""
    out = []
    for top in ("research", "application"):
        for sub in ("domain", "specialization"):
            out.extend(kw.get(top, {}).get(sub, []) or [])
    return _normalize_terms(out)


def jaccard_string_baseline(f_kw: Dict, g_kw: Dict) -> float:
    A, B = set(_flatten(f_kw)), set(_flatten(g_kw))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def load_grant_kw(db: Session, grant_id: str) -> Dict:
    row = (
        db.query(GrantKeyword)
          .filter(GrantKeyword.opportunity_id == grant_id)
          .one_or_none()
    )
    return (row.keywords or {}) if row else {}


def load_faculty_kw_all(db: Session) -> List[Tuple[int, Dict]]:
    """
    Returns: List[(faculty_id, keywords_json)]
    """
    rows = db.query(FacKeyword.faculty_id, FacKeyword.keywords).all()
    return [(fid, kw) for (fid, kw) in rows if kw]


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opportunity-id", required=True)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--agg", choices=["max", "mean_top5"], default="max",
                    help="Aggregation for faculty↔opportunity score: "
                         "'max' (best single term) or 'mean_top5' (avg of top 5 matches)")
    args = ap.parse_args()

    with SessionLocal() as db:
        g_kw = load_grant_kw(db, args.grant_id)
        if not g_kw:
            print(f"No keywords found for opportunity '{args.grant_id}'. "
                  f"Run the grant keyword extractor first.")
            return

        fac_rows = load_faculty_kw_all(db)
        if not fac_rows:
            print("No faculty keywords found. Run the faculty keyword extractor first.")
            return

        # Score all faculty against the opportunity
        scored: List[Tuple[int, float, float]] = []
        for fid, f_kw in fac_rows:
            sem = score_faculty_vs_grant(f_kw, g_kw, agg=args.agg)
            base = jaccard_string_baseline(f_kw, g_kw)
            scored.append((fid, sem, base))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Print table
        top_n = min(args.top, len(scored))
        print(f"\nTop {top_n} faculty for grant {args.grant_id} (agg={args.agg})")
        print("faculty_id | semantic_score | jaccard_baseline")
        for fid, sem, base in scored[:top_n]:
            print(f"{fid:10} | {sem:14.3f} | {base:16.3f}")

        # Explain the top match with term pairs
        if scored:
            best_fid = scored[0][0]
            # recover the keywords dict for that faculty
            best_f_kw = next(kw for f_id, kw in fac_rows if f_id == best_fid)
            pairs = rank_pairs(best_f_kw, g_kw, top_k_pairs=10)
            print(f"\nTop term pairs for faculty {best_fid}:")
            for f, g, s in pairs:
                print(f"  {s:0.3f}  {f}  ↔  {g}")


if __name__ == "__main__":
    main()
