import sys
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from services.context.context_generator import ContextGenerator
from dao.faculty_dao import FacultyDAO
from dao.match_dao import MatchDAO
from db.db_conn import SessionLocal
from services.matching.group_match_llm_selector import GroupMatchLLMSelector
from services.matching.super_faculty_selector import SuperFacultySelector


class TeamGrantMatcherService:
    def __init__(self, *, session_factory=SessionLocal, context_generator: Optional[ContextGenerator] = None):
        self.session_factory = session_factory
        self.context_generator = context_generator or ContextGenerator()
        self.super_faculty_selector = SuperFacultySelector()
        self.group_match_llm_selector = GroupMatchLLMSelector()

    @staticmethod
    def _attach_member_coverages(
        candidates: List[Dict[str, object]],
        coverage_map: Dict[int, Dict[str, Dict[int, float]]],
    ) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for cand in candidates:
            team_ids = list(cand["team"])
            out.append(
                {
                    **cand,
                    "member_coverages": {
                        int(fid): coverage_map.get(int(fid), {"application": {}, "research": {}})
                        for fid in team_ids
                    },
                }
            )
        return out

    @staticmethod
    def _normalize_candidates(
        selection,
        requirements: Dict[str, Dict[int, float]],
    ) -> List[Dict[str, object]]:
        if isinstance(selection, tuple):
            team, final_coverage = selection
            score = 0.0
            for sec, sec_cov in final_coverage.items():
                for idx, cov in sec_cov.items():
                    score += float(requirements.get(sec, {}).get(idx, 0.0)) * float(cov)
            return [{"team": team, "final_coverage": final_coverage, "score": score}]
        return selection

    def run_group_match(
        self,
        *,
        faculty_emails: List[str],
        team_size: int = 3,
        limit_rows: int = 500,
        num_candidates: int = 10,
        opp_ids: Optional[List[str]] = None,
        use_llm_selection: bool = False,
        desired_team_count: int = 1,
        group_by_opp: bool = True,
    ):
        with self.session_factory() as sess:
            match_dao = MatchDAO(sess)
            fac_dao = FacultyDAO(sess)

            if not faculty_emails:
                raise ValueError("At least one faculty email is required.")
            if desired_team_count < 1:
                raise ValueError("desired_team_count must be >= 1")
            if num_candidates < 1:
                raise ValueError("num_candidates must be >= 1")

            unique_emails = list(dict.fromkeys(faculty_emails))
            fac_ids = [fac_dao.get_faculty_id_by_email(email) for email in unique_emails]
            if opp_ids:
                target_opp_ids = opp_ids
            else:
                per_fac_opp_lists: List[List[str]] = [
                    list(match_dao.get_grant_ids_for_faculty(faculty_id=fid) or [])
                    for fid in fac_ids
                ]
                if not per_fac_opp_lists:
                    target_opp_ids = []
                else:
                    common = set(per_fac_opp_lists[0])
                    for fac_list in per_fac_opp_lists[1:]:
                        common &= set(fac_list)
                    target_opp_ids = [oid for oid in per_fac_opp_lists[0] if oid in common]

            flat_results = []
            for opp_id in target_opp_ids:
                f, w, c = self.context_generator.build_matching_inputs_for_opportunity(
                    sess=sess,
                    opportunity_id=opp_id,
                    limit_rows=limit_rows,
                )
                if not f:
                    print("  ↳ skipped (no faculty candidates)")
                    continue

                missing_required = [fid for fid in fac_ids if fid not in f]
                if missing_required:
                    for fid in missing_required:
                        f.append(fid)
                        c[fid] = {
                            "application": {i: 0.0 for i in w.get("application", {}).keys()},
                            "research": {i: 0.0 for i in w.get("research", {}).keys()},
                        }
                    f = sorted(set(f))

                candidate_pool_size = max(num_candidates, desired_team_count) if use_llm_selection else desired_team_count

                selection = self.super_faculty_selector.team_selection_super_faculty(
                    cand_faculty_ids=f,
                    requirements=w,
                    coverage=c,
                    K=team_size,
                    required_faculty_ids=fac_ids,
                    num_candidates=candidate_pool_size,
                )
                candidates = self._normalize_candidates(selection, w)

                if use_llm_selection:
                    llm_selection = self.group_match_llm_selector.select_candidate_teams_with_llm(
                        opportunity_id=opp_id,
                        desired_team_count=desired_team_count,
                        candidates=self._attach_member_coverages(candidates, c),
                        requirement_weights=w,
                    )
                    selected_candidates = llm_selection["selected_candidates"]
                else:
                    selected_candidates = candidates[:desired_team_count]

                for cand in selected_candidates[:desired_team_count]:
                    flat_results.append(
                        {
                            "opp_id": opp_id,
                            "team": cand["team"],
                            "final_coverage": cand["final_coverage"],
                            "score": cand["score"],
                        }
                    )

            if not group_by_opp:
                return flat_results

            grouped: Dict[str, Dict[str, object]] = {}
            for row in flat_results:
                oid = row["opp_id"]
                if oid not in grouped:
                    grouped[oid] = {
                        "opp_id": oid,
                        "selected_teams": [],
                    }
                grouped[oid]["selected_teams"].append(
                    {
                        "team": row["team"],
                        "final_coverage": row["final_coverage"],
                        "score": row["score"],
                    }
                )

            return list(grouped.values())
