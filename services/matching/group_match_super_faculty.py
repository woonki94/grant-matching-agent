import sys
from pathlib import Path
from typing import Dict, List, final

from typing import Dict, List, Tuple
from itertools import combinations


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dao.faculty_dao import FacultyDAO
from dao.group_match_dao import GroupMatchDAO
from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from services.matching.group_matcher_milp import build_milp_inputs_for_opportunity
from utils.keyword_accessor import build_req_text_indexed


def team_selection_super_faculty(
    *,
    faculty_ids: List[int],
    requirements: Dict[str, List[int]],
    coverage: Dict[int, Dict[str, Dict[int, float]]],
    K: int,
) -> Tuple[List[int], Dict[str, Dict[int, float]]]:

    if K < 0:
        raise ValueError("K must be >= 0")
    if K == 0:
        best_covered = {sec: {i: 0.0 for i in idxs} for sec, idxs in requirements.items()}
        return [], best_covered
    if K > len(faculty_ids):
        K = min(K, len(faculty_ids))

    def score(covered: Dict[str, Dict[int, float]]) -> float:
        return sum(covered[sec][i] for sec in covered for i in covered[sec])

    best_team: List[int] = []
    best_covered: Dict[str, Dict[int, float]] = {}
    best_score = float("-inf")

    # Enumerate all teams of size K
    for team in combinations(faculty_ids, K):
        # Start with 0s (use keys as the indices)
        covered = {sec: {i: 0.0 for i in req.keys()} for sec, req in requirements.items()}

        # Build "super faculty": elementwise max over team members
        for f in team:
            for sec, req in requirements.items():
                f_sec = coverage[f][sec]
                for i in req.keys():
                    val = f_sec.get(i, 0.0)
                    if val > covered[sec][i]:
                        covered[sec][i] = val

        team_score = score(covered)

        if team_score > best_score:
            best_score = team_score
            best_team = list(team)
            best_covered = covered

    return best_team, best_covered


def run_group_match(
        faculty_email: str,
        team_size: int=3,
        limit_rows: int = 500,
        #commit_every: int = 25,
):
    with SessionLocal() as sess:
        '''
        opp_dao = OpportunityDAO(sess)
        g_dao = GroupMatchDAO(sess)
        '''

        match_dao = MatchDAO(sess)
        fac_dao = FacultyDAO(sess)

        fac_id = fac_dao.get_faculty_id_by_email(faculty_email)
        opp_ids = match_dao.get_grant_ids_for_faculty(
            faculty_id= fac_id
        )

        results =[]

        for opp_id in opp_ids:

            # --- build inputs ---
            F, I_app, I_res, w, c = build_milp_inputs_for_opportunity(
                sess=sess,
                opportunity_id=opp_id,
                limit_rows=limit_rows,
            )
            if not F:
                print("  ↳ skipped (no faculty candidates)")
                continue

            team, final_coverage = team_selection_super_faculty(
                faculty_ids=F,
                requirements=w,
                coverage=c,
                K=team_size
            )

            results.append({
                "opp_id": opp_id,
                "team": team,
                "final_coverage": final_coverage,
            })

        anchor_faculty_id = fac_id  # 41

        filtered_results = [
            r for r in results
            if anchor_faculty_id in r["team"]
        ]
        '''
            if not team:
                print(f"{opp_id}: skipped (empty team)")
                continue

            g_dao.upsert(
                grant_id=opp_id,
                faculty_ids=team,
                team_size=len(team),  # safer than passing team_size
                final_coverage=final_coverage,
            )

            if (idx + 1) % commit_every == 0:
                sess.commit()
                print(f"Committed {idx + 1} opportunities")

        sess.commit()
        '''

        return filtered_results

if __name__ == "__main__":

    team_size = 3
    email = 'AbbasiB@oregonstate.edu'

    ret = run_group_match(team_size=team_size, faculty_email=email)

    print(ret)

