import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dao.faculty_dao import FacultyDAO
from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from services.matching.super_faculty_selector import team_selection_super_faculty
from services.matching.team_candidate_llm_selector import select_candidate_teams_with_llm
from utils.keyword_accessor import extract_specializations


def build_inputs_for_opportunity(
    *,
    sess,
    opportunity_id: str,
    limit_rows: int = 500,
) -> Tuple[List[int], List[int], List[int], Dict[str, Dict[int, float]], Dict[int, Dict[str, Dict[int, float]]]]:
    """
    Returns (F, I_app, I_res, w, c) for one opportunity.
    """
    match_dao = MatchDAO(sess)
    opp_dao = OpportunityDAO(sess)

    opps = opp_dao.read_opportunities_by_ids_with_relations([opportunity_id])
    if not opps:
        raise ValueError(f"Opportunity not found: {opportunity_id}")
    opp = opps[0]

    kw_raw = getattr(opp.keyword, "keywords", {}) or {}
    spec_items = extract_specializations(kw_raw)

    i_app = list(range(len(spec_items["application"])))
    i_res = list(range(len(spec_items["research"])))

    w: Dict[str, Dict[int, float]] = {"application": {}, "research": {}}
    for sec, i_sec in (("application", i_app), ("research", i_res)):
        for i in i_sec:
            w[sec][i] = float(spec_items[sec][i].get("w", 1.0))

    match_rows = match_dao.list_matches_for_opportunity(opportunity_id, limit=limit_rows)
    if not match_rows:
        raise ValueError("No match rows found.")

    f = sorted({int(r["faculty_id"]) for r in match_rows})
    c: Dict[int, Dict[str, Dict[int, float]]] = {fid: {"application": {}, "research": {}} for fid in f}

    for r in match_rows:
        fid = int(r["faculty_id"])
        cov = r.get("covered") or {}
        for sec in ("application", "research"):
            sec_map = cov.get(sec) or {}
            for k, v in sec_map.items():
                try:
                    idx = int(k)
                    cval = float(v)
                except Exception:
                    continue
                prev = c[fid][sec].get(idx, 0.0)
                c[fid][sec][idx] = max(prev, cval)

    return f, i_app, i_res, w, c


def run_group_match(
    faculty_emails: List[str],
    team_size: int = 3,
    limit_rows: int = 500,
    num_candidates: int = 1,
    opp_ids: Optional[List[str]] = None,
    use_llm_selection: bool = False,
    desired_team_count: int = 1,
):
    with SessionLocal() as sess:
        match_dao = MatchDAO(sess)
        fac_dao = FacultyDAO(sess)

        if not faculty_emails:
            raise ValueError("At least one faculty email is required.")

        # Deduplicate while preserving order.
        unique_emails = list(dict.fromkeys(faculty_emails))
        fac_ids = [fac_dao.get_faculty_id_by_email(email) for email in unique_emails]
        anchor_fac_id = fac_ids[0]

        target_opp_ids = opp_ids if opp_ids else match_dao.get_grant_ids_for_faculty(faculty_id=anchor_fac_id)

        results = []
        for opp_id in target_opp_ids:
            f, _i_app, _i_res, w, c = build_inputs_for_opportunity(
                sess=sess,
                opportunity_id=opp_id,
                limit_rows=limit_rows,
            )
            if not f:
                print("  ↳ skipped (no faculty candidates)")
                continue

            selection = team_selection_super_faculty(
                cand_faculty_ids=f,
                requirements=w,
                coverage=c,
                K=team_size,
                required_faculty_ids=fac_ids,
                num_candidates=num_candidates,
            )

            if num_candidates == 1:
                team, final_coverage = selection
                results.append(
                    {
                        "opp_id": opp_id,
                        "team": team,
                        "final_coverage": final_coverage,
                    }
                )
            else:
                candidates = selection
                llm_selection = None
                selected_candidates = candidates
                if use_llm_selection:
                    llm_candidates = []
                    for cand in candidates:
                        team_ids = list(cand["team"])
                        llm_candidates.append(
                            {
                                **cand,
                                "member_coverages": {
                                    int(fid): c.get(int(fid), {"application": {}, "research": {}})
                                    for fid in team_ids
                                },
                            }
                        )
                    llm_selection = select_candidate_teams_with_llm(
                        opportunity_id=opp_id,
                        desired_team_count=desired_team_count,
                        candidates=llm_candidates,
                        requirement_weights=w,
                    )

                    selected_candidates = llm_selection["selected_candidates"]

                top = selected_candidates[0]
                results.append(
                    {
                        "opp_id": opp_id,
                        "team": top["team"],
                        "final_coverage": top["final_coverage"],
                        "score": top["score"],
                        "candidates": candidates,
                        "selected_teams": selected_candidates,
                        "llm_selection": llm_selection,
                    }
                )

        return results


if __name__ == "__main__":
    team_size = 3
    email_list = ["houssam.abbas@oregonstate.edu","AbbasiB@oregonstate.edu"]
    wished_grant = "60b8b017-30ec-4f31-a160-f00b7ee384e7"
    ret = run_group_match(team_size=team_size, faculty_emails=email_list, opp_ids=[wished_grant], num_candidates=10, use_llm_selection = True, desired_team_count=1)
    print(ret)
