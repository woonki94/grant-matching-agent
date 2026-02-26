from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from config import get_llm_client
from dao.faculty_dao import FacultyDAO
from dao.opportunity_dao import OpportunityDAO
from dto.llm_response_dto import (
    GrantBriefOut,
    GroupJustificationOut,
    RecommendationOut,
    TeamRoleOut,
    WhyNotWorkingOut,
    WhyWorkingOut,
)
from services.context.context_generator import ContextGenerator
from utils.keyword_utils import extract_requirement_specs
from services.prompts.group_match_prompt import (
    GRANT_BRIEF_PROMPT,
    RECOMMENDER_PROMPT,
    TEAM_ROLE_DECIDER_PROMPT,
    WHY_NOT_WORKING_DECIDER_PROMPT,
    WHY_WORKING_DECIDER_PROMPT,
)


class GroupJustificationEngine:
    INDEPENDENT_STAGE_WORKERS = 4

    def __init__(
        self,
        *,
        odao: OpportunityDAO,
        fdao: FacultyDAO,
        context_generator: Optional[ContextGenerator] = None,
    ):
        self.odao = odao
        self.fdao = fdao
        self.context_generator = context_generator or ContextGenerator()
        self.llm = get_llm_client().build()

        self.grant_brief_chain = GRANT_BRIEF_PROMPT | self.llm.with_structured_output(GrantBriefOut)
        self.team_role_chain = TEAM_ROLE_DECIDER_PROMPT | self.llm.with_structured_output(TeamRoleOut)
        self.why_working_chain = WHY_WORKING_DECIDER_PROMPT | self.llm.with_structured_output(WhyWorkingOut)
        self.why_not_working_chain = WHY_NOT_WORKING_DECIDER_PROMPT | self.llm.with_structured_output(WhyNotWorkingOut)
        self.recommender_chain = RECOMMENDER_PROMPT | self.llm.with_structured_output(RecommendationOut)

    @staticmethod
    def _safe_json(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2)

    def run_one(
        self,
        *,
        opp_ctx: Dict[str, Any],
        fac_ctxs: List[Dict[str, Any]],
        coverage: Any,
        member_coverages: Optional[Dict[int, Dict[str, Dict[int, float]]]] = None,
        group_meta: Optional[Dict[str, Any]] = None,
        trace: Optional[Dict[str, Any]] = None,
    ) -> Tuple[GroupJustificationOut, Dict[str, Any]]:
        trace = trace or {}
        trace.setdefault("steps", {})

        context = self.context_generator.build_group_matching_context(
            opp_ctx=opp_ctx,
            fac_ctxs=fac_ctxs,
            coverage=coverage,
            member_coverages=member_coverages,
            group_meta=group_meta,
        )
        requirements = extract_requirement_specs(opp_ctx)
        grant_block = context.get("grant") or {}
        team_block = context.get("team") or []
        coverage_block = context.get("coverage") or {}
        grant_link = f"https://simpler.grants.gov/opportunity/{grant_block.get('id')}" if grant_block.get("id") else ""


        try:
            grant_brief_input = {
                "grant": {
                    "id": grant_block.get("id"),
                    "title": grant_block.get("title"),
                    "agency": grant_block.get("agency"),
                    "summary": grant_block.get("summary"),
                    "link": grant_link,
                    "keywords": grant_block.get("keywords"),
                },
                "requirements": requirements,
            }
            team_role_input = {
                "grant": {
                    "id": grant_block.get("id"),
                    "title": grant_block.get("title"),
                    "keywords": grant_block.get("keywords"),
                },
                "requirements": requirements,
                "team": team_block,
            }
            why_working_input = {
                "grant": {
                    "id": grant_block.get("id"),
                    "title": grant_block.get("title"),
                    "keywords": grant_block.get("keywords"),
                },
                "requirements": requirements,
                "team": team_block,
                "team_final_coverage": coverage_block,
            }
            why_not_input = {
                "grant": {
                    "id": grant_block.get("id"),
                    "title": grant_block.get("title"),
                    "keywords": grant_block.get("keywords"),
                },
                "requirements": requirements,
                "team": team_block,
                "team_final_coverage": coverage_block,
            }
            section_jobs = {
                "grant_brief": (
                    self.grant_brief_chain,
                    {"input_json": self._safe_json(grant_brief_input)},
                ),
                "team_roles": (
                    self.team_role_chain,
                    {"input_json": self._safe_json(team_role_input)},
                ),
                "why_working": (
                    self.why_working_chain,
                    {"input_json": self._safe_json(why_working_input)},
                ),
                "why_not_working": (
                    self.why_not_working_chain,
                    {"input_json": self._safe_json(why_not_input)},
                ),
            }

            section_outputs: Dict[str, Any] = {}
            with ThreadPoolExecutor(max_workers=self.INDEPENDENT_STAGE_WORKERS) as ex:
                futures = {
                    ex.submit(chain.invoke, payload): section_name
                    for section_name, (chain, payload) in section_jobs.items()
                }
                for fut in as_completed(futures):
                    section_name = futures[fut]
                    section_outputs[section_name] = fut.result()

            grant_brief: GrantBriefOut = section_outputs["grant_brief"]
            team_roles: TeamRoleOut = section_outputs["team_roles"]
            why_working: WhyWorkingOut = section_outputs["why_working"]
            why_not: WhyNotWorkingOut = section_outputs["why_not_working"]

            trace["steps"]["grant_brief"] = {
                "status": "ok",
                "input": grant_brief_input,
                "output": grant_brief.model_dump(),
            }
            trace["steps"]["team_roles"] = {
                "status": "ok",
                "input": team_role_input,
                "output": team_roles.model_dump(),
            }
            trace["steps"]["why_working"] = {
                "status": "ok",
                "input": why_working_input,
                "output": why_working.model_dump(),
            }
            trace["steps"]["why_not_working"] = {
                "status": "ok",
                "input": why_not_input,
                "output": why_not.model_dump(),
            }

            rec_input = {
                "grant": {
                    "id": grant_block.get("id"),
                    "title": grant_block.get("title"),
                    "link": grant_link,
                },
                "team_roles": team_roles.model_dump(),
                "why_working": why_working.model_dump(),
                "why_not_working": why_not.model_dump(),
            }
            recommendation = self.recommender_chain.invoke({"input_json": self._safe_json(rec_input)})
            trace["steps"]["recommendation"] = {
                "status": "ok",
                "input": rec_input,
                "output": recommendation.model_dump(),
            }

            justification = GroupJustificationOut(
                one_paragraph=grant_brief.grant_quick_explanation or "",
                member_roles=team_roles.member_roles,
                coverage={
                    "strong": why_working.strong,
                    "partial": why_working.partial,
                    "missing": why_not.missing,
                },
                member_strengths=why_working.member_strengths,
                why_not_working=why_not.why_not_working,
                recommendation=recommendation.recommendation,
            )
            return justification, trace
        except Exception as e:
            trace["steps"]["error"] = {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            }
            fallback_roles = []
            for tm in team_block:
                fid = tm.get("faculty_id")
                if isinstance(fid, int):
                    fallback_roles.append(
                        {
                            "faculty_id": fid,
                            "role": "Contributor",
                            "why": "Insufficient LLM output; role inferred as general contributor.",
                        }
                    )
            justification = GroupJustificationOut(
                one_paragraph="Insufficient model output to generate a complete structured justification.",
                member_roles=fallback_roles,
                coverage={"strong": [], "partial": [], "missing": []},
                member_strengths=[],
                why_not_working=["Structured writer stages failed; verify model/service health and rerun."],
                recommendation="Review this result manually and rerun generation.",
            )
            return justification, trace
