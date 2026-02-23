from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from dao.faculty_dao import FacultyDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from services.context.context_generator import ContextGenerator
from services.justification.group_justification_engine import GroupJustificationEngine
from services.matching.team_grant_matcher import TeamGrantMatcher
from utils.keyword_utils import extract_requirement_specs
from utils.report_renderer import render_markdown_report


class GroupJustificationGenerator:
    def __init__(
        self,
        *,
        session_factory=SessionLocal,
        context_generator: Optional[ContextGenerator] = None,
        team_grant_matcher: Optional[TeamGrantMatcher] = None,
    ):
        self.session_factory = session_factory
        self.context_generator = context_generator or ContextGenerator()
        self.team_grant_matcher = team_grant_matcher or TeamGrantMatcher(
            session_factory=session_factory,
            context_generator=self.context_generator,
        )

    @staticmethod
    def _expand_group_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        expanded: List[Dict[str, Any]] = []
        for row in rows:
            if "selected_teams" in row:
                opp_id = row.get("opp_id") or row.get("grant_id")
                for cand in row.get("selected_teams") or []:
                    expanded.append(
                        {
                            "opp_id": opp_id,
                            "team": cand.get("team"),
                            "final_coverage": cand.get("final_coverage"),
                            "score": cand.get("score"),
                        }
                    )
            else:
                expanded.append(
                    {
                        "opp_id": row.get("opp_id") or row.get("grant_id"),
                        "team": row.get("team"),
                        "final_coverage": row.get("final_coverage"),
                        "score": row.get("score"),
                    }
                )
        return expanded

    def run_justifications_from_group_results(
        self,
        *,
        faculty_emails: Union[str, List[str]],
        team_size: int,
        opp_ids: Optional[List[str]] = None,
        limit_rows: int = 500,
        include_trace: bool = False,
    ) -> List[Dict[str, Any]]:
        with self.session_factory() as sess:
            odao = OpportunityDAO(sess)
            fdao = FacultyDAO(sess)
            engine = GroupJustificationEngine(
                odao=odao,
                fdao=fdao,
                context_generator=self.context_generator,
            )

            email_list = [faculty_emails] if isinstance(faculty_emails, str) else list(faculty_emails)
            group_results = self.team_grant_matcher.run_group_match(
                faculty_emails=email_list,
                team_size=team_size,
                limit_rows=limit_rows,
                opp_ids=opp_ids,
            )
            normalized_rows = self._expand_group_results(group_results)
            if not normalized_rows:
                raise ValueError(f"No group matches found for {faculty_emails}")

            opp_cache: Dict[str, Dict[str, Any]] = {}
            fac_cache: Dict[int, Dict[str, Any]] = {}
            opp_member_cov_cache: Dict[str, Dict[int, Dict[str, Dict[int, float]]]] = {}

            results: List[Dict[str, Any]] = []

            for idx, row in enumerate(normalized_rows):
                opp_id = row["opp_id"]
                team: List[int] = row["team"]
                coverage = row.get("final_coverage")

                if opp_id not in opp_cache:
                    opp_cache[opp_id] = odao.read_opportunity_context(opp_id) or {}
                opp_ctx = opp_cache[opp_id]
                grant_title = opp_ctx.get("title") or opp_ctx.get("opportunity_title")
                agency_name = opp_ctx.get("agency") or opp_ctx.get("agency_name")

                if not opp_ctx:
                    results.append(
                        {
                            "index": idx,
                            "grant_id": opp_id,
                            "grant_title": None,
                            "agency_name": None,
                            "grant_link": f"https://simpler.grants.gov/opportunity/{opp_id}",
                            "team": team,
                            "error": "Opportunity not found",
                        }
                    )
                    continue

                fac_ctxs: List[Dict[str, Any]] = []
                for fid in team:
                    if fid not in fac_cache:
                        fac_cache[fid] = fdao.get_faculty_keyword_context(fid) or {}
                    if fac_cache[fid]:
                        fac_ctxs.append(fac_cache[fid])

                if opp_id not in opp_member_cov_cache:
                    opp_member_cov_cache[opp_id] = self.context_generator.build_member_coverages_for_opportunity(
                        sess=sess,
                        opportunity_id=opp_id,
                        limit_rows=limit_rows,
                    )

                member_coverages = {
                    int(fid): opp_member_cov_cache.get(opp_id, {}).get(int(fid), {"application": {}, "research": {}})
                    for fid in team
                }

                if not fac_ctxs:
                    results.append(
                        {
                            "index": idx,
                            "grant_id": opp_id,
                            "grant_title": grant_title,
                            "agency_name": agency_name,
                            "grant_link": f"https://simpler.grants.gov/opportunity/{opp_id}",
                            "team": team,
                            "error": "No faculty contexts found for team",
                        }
                    )
                    continue

                group_meta = {
                    "group_id": row.get("group_id") or row.get("id"),
                    "lambda": row.get("lambda"),
                    "k": row.get("k"),
                    "objective": row.get("objective"),
                    "redundancy": row.get("redundancy"),
                    "meta": row.get("meta"),
                }

                try:
                    justification, trace = engine.run_one(
                        opp_ctx=dict(opp_ctx),
                        fac_ctxs=[dict(f) for f in fac_ctxs],
                        coverage=coverage,
                        member_coverages=member_coverages,
                        group_meta=group_meta,
                        trace={"index": idx, "opp_id": opp_id, "team": team},
                    )

                    out = {
                        "index": idx,
                        "grant_id": opp_id,
                        "grant_title": grant_title,
                        "agency_name": agency_name,
                        #"grant_link": f"https://simpler.grants.gov/opportunity/{opp_id}",
                        "team": team,
                        "team_members": [
                            {
                                "faculty_id": f.get("faculty_id") or f.get("id"),
                                "faculty_name": f.get("name"),
                                "faculty_email": f.get("email"),
                            }
                            for f in fac_ctxs
                        ],
                        "team_score": float(row.get("score") or 0.0),
                        #"final_coverage": coverage,
                        #"requirement_specs": extract_requirement_specs(opp_ctx),
                        "justification": justification.model_dump(),
                    }
                    if include_trace:
                        out["trace"] = trace
                    results.append(out)
                except Exception as e:
                    results.append(
                        {
                            "index": idx,
                            "grant_id": opp_id,
                            "grant_title": grant_title,
                            "agency_name": agency_name,
                            "grant_link": f"https://simpler.grants.gov/opportunity/{opp_id}",
                            "team": team,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )

            return results
