import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../root
sys.path.insert(0, str(PROJECT_ROOT))

from dao.match_dao import MatchDAO
from db.db_conn import SessionLocal


def _as_set(d: dict, key: str) -> Set[int]:
    xs = (d or {}).get(key) or []
    return set(int(x) for x in xs)


def greedy_group_from_match_results(
    match_rows: List[dict],
    team_size: int = 4,
    w_research: float = 0.5,
    min_llm_score: float = 0.0,
    max_app_per_faculty: int = 3,
    max_res_per_faculty: int = 3,
    min_team_size: int = 1,
) -> Tuple[List[int], dict, Dict[int, Dict[str, List[int]]]]:
    """
    Builds a team by greedily maximizing marginal coverage of remaining needs.

    Inputs:
      - match_rows: rows for a single opportunity, each with fields:
          faculty_id, llm_score, covered ({"application":[...],"research":[...]}),
          missing (same shape)
      - team_size: max number of faculty to select
      - w_research: weight for research needs coverage
      - min_llm_score: filter out candidates below this score
      - max_app_per_faculty / max_res_per_faculty: cap how many NEW needs
        a single faculty is allowed to claim per section (forces multi-faculty teams)
      - min_team_size: ensure at least this many faculty are selected (even if coverage completes)

    Returns:
      (selected_faculty_ids, debug_summary, faculty_need_map)
      faculty_need_map[fid] = {"application": [idx...], "research": [idx...]}  # marginal contributions
    """

    rows = [r for r in match_rows if float(r.get("llm_score", 0.0)) >= min_llm_score]
    if not rows:
        return [], {"reason": "no candidates"}, {}

    # Need universe: derived from first row (assumes completeness constraint in your LLM)
    all_app: Set[int] = set()
    all_res: Set[int] = set()

    for r in rows:
        all_app |= _as_set(r.get("covered"), "application")
        all_app |= _as_set(r.get("missing"), "application")
        all_res |= _as_set(r.get("covered"), "research")
        all_res |= _as_set(r.get("missing"), "research")

    remaining_app = set(all_app)
    remaining_res = set(all_res)

    selected: List[int] = []
    covered_app: Set[int] = set()
    covered_res: Set[int] = set()

    # Preparse candidate coverage once
    parsed = []
    for r in rows:
        cov = r.get("covered") or {}
        parsed.append(
            {
                "faculty_id": int(r["faculty_id"]),
                "llm_score": float(r.get("llm_score", 0.0)),
                "covered_app": _as_set(cov, "application"),
                "covered_res": _as_set(cov, "research"),
            }
        )

    faculty_need_map: Dict[int, Dict[str, List[int]]] = {}

    def capped_gain(cand: dict) -> Tuple[float, int, int]:
        """Returns (gain, new_app_count, new_res_count) with per-faculty caps applied."""
        new_app = min(len(cand["covered_app"] & remaining_app), max_app_per_faculty)
        new_res = min(len(cand["covered_res"] & remaining_res), max_res_per_faculty)
        gain = new_app + w_research * new_res
        return gain, new_app, new_res

    # Select until we hit team_size, and until needs are covered (unless min_team_size forces more)
    while len(selected) < team_size and ((remaining_app or remaining_res) or len(selected) < min_team_size):
        best = None
        best_gain = -1.0
        best_tiebreak = -1.0

        for cand in parsed:
            fid = cand["faculty_id"]
            if fid in selected:
                continue

            gain, _, _ = capped_gain(cand)
            tiebreak = cand["llm_score"]

            if gain > best_gain or (gain == best_gain and tiebreak > best_tiebreak):
                best_gain = gain
                best_tiebreak = tiebreak
                best = cand

        # If we're still below min_team_size, allow selecting even if gain==0 (best coverage already done)
        if best is None:
            break
        if best_gain <= 0 and len(selected) >= min_team_size:
            break

        fid = best["faculty_id"]

        # Choose which needs this faculty "claims" (capped)
        newly_app_all = sorted(best["covered_app"] & remaining_app)
        newly_res_all = sorted(best["covered_res"] & remaining_res)

        newly_app = newly_app_all[:max_app_per_faculty]
        newly_res = newly_res_all[:max_res_per_faculty]

        selected.append(fid)
        faculty_need_map[fid] = {
            "application": newly_app,
            "research": newly_res,
        }

        # Update covered/remaining using ONLY the claimed (capped) needs
        covered_app |= set(newly_app)
        covered_res |= set(newly_res)
        remaining_app -= set(newly_app)
        remaining_res -= set(newly_res)

    dbg = {
        "all_app": sorted(all_app),
        "all_res": sorted(all_res),
        "covered_app": sorted(covered_app),
        "covered_res": sorted(covered_res),
        "missing_app": sorted(remaining_app),
        "missing_res": sorted(remaining_res),
    }
    return selected, dbg, faculty_need_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opportunity-id", required=True)
    parser.add_argument("--team-size", type=int, default=4)
    parser.add_argument("--limit-faculty", type=int, default=300)
    parser.add_argument("--min-llm-score", type=float, default=0.1)
    parser.add_argument("--w-research", type=float, default=0.5)
    parser.add_argument("--max-app-per-faculty", type=int, default=3)
    parser.add_argument("--max-res-per-faculty", type=int, default=3)
    parser.add_argument("--min-team-size", type=int, default=2)
    args = parser.parse_args()

    with SessionLocal() as sess:
        match_dao = MatchDAO(sess)
        match_rows = match_dao.list_matches_for_opportunity(args.opportunity_id, limit=args.limit_faculty)

        selected_ids, dbg, faculty_need_map = greedy_group_from_match_results(
            match_rows,
            team_size=args.team_size,
            w_research=args.w_research,
            min_llm_score=args.min_llm_score,
            max_app_per_faculty=args.max_app_per_faculty,
            max_res_per_faculty=args.max_res_per_faculty,
            min_team_size=args.min_team_size,
        )

        print("Selected:", selected_ids)

        print("\nFaculty to newly covered needs (contribution):")
        for fid in selected_ids:
            m = faculty_need_map.get(fid, {})
            print(f"  Faculty {fid}:")
            print(f"    application: {m.get('application', [])}")
            print(f"    research:    {m.get('research', [])}")

        print("\nCoverage summary:")
        print("  covered application:", dbg["covered_app"])
        print("  covered research:   ", dbg["covered_res"])
        print("  missing application:", dbg["missing_app"])
        print("  missing research:   ", dbg["missing_res"])