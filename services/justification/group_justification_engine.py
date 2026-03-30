from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from config import get_llm_client, settings
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
from services.context_retrieval.context_generator import ContextGenerator
from utils.keyword_utils import extract_requirement_specs
from utils.thread_pool import parallel_map
from services.prompts.group_match_prompt import (
    GRANT_BRIEF_PROMPT,
    RECOMMENDER_PROMPT,
    TEAM_ROLE_DECIDER_PROMPT,
    WHY_NOT_WORKING_DECIDER_PROMPT,
    WHY_WORKING_DECIDER_PROMPT,
)

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _safe_json(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2)

    @staticmethod
    def _build_grant_brief_chain():
        model_id = (settings.haiku or settings.sonnet or settings.opus or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        return GRANT_BRIEF_PROMPT | llm.with_structured_output(GrantBriefOut)

    @staticmethod
    def _build_team_role_chain():
        model_id = (settings.opus or settings.sonnet or settings.haiku or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        return TEAM_ROLE_DECIDER_PROMPT | llm.with_structured_output(TeamRoleOut)

    @staticmethod
    def _build_why_working_chain():
        model_id = (settings.sonnet or settings.haiku or settings.opus or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        return WHY_WORKING_DECIDER_PROMPT | llm.with_structured_output(WhyWorkingOut)

    @staticmethod
    def _build_why_not_working_chain():
        model_id = (settings.sonnet or settings.haiku or settings.opus or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        return WHY_NOT_WORKING_DECIDER_PROMPT | llm.with_structured_output(WhyNotWorkingOut)

    @staticmethod
    def _build_recommender_chain():
        model_id = (settings.sonnet or settings.opus or settings.haiku or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        return RECOMMENDER_PROMPT | llm.with_structured_output(RecommendationOut)

    def run_one(
        self,
        *,
        opp_ctx: Dict[str, Any],
        grant_brief_context: Optional[Dict[str, Any]] = None,
        fac_ctxs: List[Dict[str, Any]],
        coverage: Any,
        member_coverages: Optional[Dict[int, Dict[str, Dict[int, float]]]] = None,
        group_meta: Optional[Dict[str, Any]] = None,
        evidence_text: str = "",
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
        grant_ctx = dict(grant_brief_context or {})

        logger.info(
            "GROUP_EVIDENCE_INPUT meta=%s payload=%s",
            json.dumps(
                {
                    "opportunity_id": grant_block.get("id"),
                    "team_size": len(team_block),
                    "team_faculty_ids": [m.get("faculty_id") for m in list(team_block or [])],
                    "evidence_chars": len(str(evidence_text or "")),
                },
                ensure_ascii=False,
            ),
            str(evidence_text or ""),
        )

        try:
            grant_brief_chain = self._build_grant_brief_chain()
            team_role_chain = self._build_team_role_chain()
            why_working_chain = self._build_why_working_chain()
            why_not_working_chain = self._build_why_not_working_chain()
            recommender_chain = self._build_recommender_chain()

            grant_brief_input = {
                "grant_context": (
                    grant_ctx
                    if grant_ctx
                    else {
                        "opportunity_id": grant_block.get("id"),
                        "title": grant_block.get("title"),
                        "agency": grant_block.get("agency"),
                        "opportunity_link": grant_link,
                        "summary": grant_block.get("summary"),
                    }
                ),
            }
            team_role_input = {
                "grant": {
                    "id": grant_block.get("id"),
                    "title": grant_block.get("title"),
                    "keywords": grant_block.get("keywords"),
                },
                "requirements": requirements,
                "team": team_block,
                "evidence_text": str(evidence_text or ""),
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
                "evidence_text": str(evidence_text or ""),
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
                "evidence_text": str(evidence_text or ""),
            }
            section_jobs: List[Tuple[str, Any, Dict[str, str]]] = [
                (
                    "grant_brief",
                    grant_brief_chain,
                    {"input_json": self._safe_json(grant_brief_input)},
                ),
                (
                    "team_roles",
                    team_role_chain,
                    {"input_json": self._safe_json(team_role_input)},
                ),
                (
                    "why_working",
                    why_working_chain,
                    {"input_json": self._safe_json(why_working_input)},
                ),
                (
                    "why_not_working",
                    why_not_working_chain,
                    {"input_json": self._safe_json(why_not_input)},
                ),
            ]

            section_outputs = {
                section_name: section_output
                for section_name, section_output in parallel_map(
                    section_jobs,
                    max_workers=self.INDEPENDENT_STAGE_WORKERS,
                    run_item=lambda job: (job[0], job[1].invoke(job[2])),
                )
            }

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
            logger.info(
                "GROUP_EVIDENCE_OUTPUT meta=%s output=%s",
                json.dumps(
                    {
                        "opportunity_id": grant_block.get("id"),
                        "team_size": len(team_block),
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "grant_brief": grant_brief.model_dump(),
                        "team_roles": team_roles.model_dump(),
                        "why_working": why_working.model_dump(),
                        "why_not_working": why_not.model_dump(),
                    },
                    ensure_ascii=False,
                ),
            )

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
            recommendation = recommender_chain.invoke({"input_json": self._safe_json(rec_input)})
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
            logger.exception(
                "GROUP_EVIDENCE_PIPELINE_FAILED meta=%s",
                json.dumps(
                    {
                        "opportunity_id": grant_block.get("id"),
                        "team_size": len(team_block),
                    },
                    ensure_ascii=False,
                ),
            )
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
