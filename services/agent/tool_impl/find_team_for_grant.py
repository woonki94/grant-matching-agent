from __future__ import annotations

from typing import List
import re

from db.db_conn import SessionLocal
from dao.opportunity_dao import OpportunityDAO
from services.matching.group_match_super_faculty import build_inputs_for_opportunity
from services.matching.super_faculty_selector import team_selection_super_faculty
from services.justification.generate_group_justification import (
    run_justifications_from_group_results,
)
from services.agent.tool_errors import ToolInputError


def _resolve_opp_ids(opp_ids: List[str], odao: OpportunityDAO) -> List[str]:
    if not opp_ids:
        return []
    uuid_re = re.compile(r"^[0-9a-fA-F-]{32,36}$")
    resolved: List[str] = []
    for v in opp_ids:
        val = str(v or "").strip()
        if not val:
            continue
        if uuid_re.match(val):
            resolved.append(val)
            continue
        resolved.extend(odao.find_opportunity_ids_by_title(val, limit=5))
    return list(dict.fromkeys(resolved))


def _score_team(requirements: dict, coverage: dict) -> float:
    total = 0.0
    for sec, sec_cov in (coverage or {}).items():
        if not isinstance(sec_cov, dict):
            continue
        for idx, cov in sec_cov.items():
            try:
                w = float(requirements.get(sec, {}).get(int(idx), 0.0))
                total += w * float(cov)
            except Exception:
                continue
    return total


def find_team_for_grant(
    opp_ids: list[str],
    team_size: int,
) -> str:
    opp_ids = [str(x).strip() for x in (opp_ids or []) if str(x).strip()]
    if not opp_ids:
        raise ToolInputError(
            tool_name="find_team_for_grant",
            message="Please provide the opportunity ID or title.",
            missing_fields=["opp_ids"],
        )
    if team_size is None or int(team_size) < 1:
        raise ToolInputError(
            tool_name="find_team_for_grant",
            message="Please provide a valid team size.",
            missing_fields=["team_size"],
        )

    sess = SessionLocal()
    try:
        odao = OpportunityDAO(sess)
        resolved = _resolve_opp_ids(opp_ids, odao)
        if not resolved:
            raise ToolInputError(
                tool_name="find_team_for_grant",
                message="No matching opportunity found for the provided title or ID.",
                missing_fields=["opp_ids"],
            )

        group_results = []
        for opp_id in resolved:
            f, _i_app, _i_res, w, c = build_inputs_for_opportunity(
                sess=sess,
                opportunity_id=opp_id,
                limit_rows=500,
            )
            if not f:
                continue

            selection = team_selection_super_faculty(
                cand_faculty_ids=f,
                requirements=w,
                coverage=c,
                K=int(team_size),
                required_faculty_ids=[],
                num_candidates=1,
            )
            if isinstance(selection, tuple):
                team, final_coverage = selection
                score = _score_team(w, final_coverage)
                group_results.append(
                    {
                        "opp_id": opp_id,
                        "selected_teams": [
                            {
                                "team": team,
                                "final_coverage": final_coverage,
                                "score": score,
                            }
                        ],
                    }
                )
            elif selection:
                top = selection[0]
                group_results.append(
                    {
                        "opp_id": opp_id,
                        "selected_teams": [top],
                    }
                )

        if not group_results:
            return "No suitable teams found for the provided opportunity."

        try:
            return run_justifications_from_group_results(
                group_results=group_results,
                limit_rows=500,
                include_trace=False,
            )
        except Exception as exc:
            return f"Error generating report: {type(exc).__name__}: {exc}"
    finally:
        sess.close()
