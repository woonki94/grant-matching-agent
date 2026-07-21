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
        odao: Optional[OpportunityDAO] = None,
        fdao: Optional[FacultyDAO] = None,
        context_generator: Optional[ContextGenerator] = None,
    ):
        self.odao = odao
        self.fdao = fdao
        self.context_generator = context_generator or ContextGenerator()

    @staticmethod
    def _safe_json(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2)

    @staticmethod
    def _norm(text: Any) -> str:
        return " ".join(str(text or "").split()).strip()

    @staticmethod
    def _build_grant_brief_chain():
        model_id = (settings.haiku or settings.sonnet or settings.opus or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        return GRANT_BRIEF_PROMPT | llm.with_structured_output(GrantBriefOut)

    @staticmethod
    def _build_team_role_chain():
        model_id = (settings.haiku or settings.sonnet or settings.opus or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        return TEAM_ROLE_DECIDER_PROMPT | llm.with_structured_output(TeamRoleOut)

    @staticmethod
    def _build_why_working_chain():
        model_id = (settings.sonnet or settings.opus or settings.haiku or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        return WHY_WORKING_DECIDER_PROMPT | llm.with_structured_output(WhyWorkingOut)

    @staticmethod
    def _build_why_not_working_chain():
        model_id = (settings.sonnet or settings.opus or settings.haiku or "").strip()
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
        team_ids: Optional[List[int]] = None,
        match_rows_by_faculty: Optional[Dict[int, Dict[str, Any]]] = None,
        member_coverages: Optional[Dict[int, Dict[str, Dict[int, float]]]] = None,
        group_meta: Optional[Dict[str, Any]] = None,
        evidence_text: str = "",
        trace: Optional[Dict[str, Any]] = None,
    ) -> Tuple[GroupJustificationOut, Dict[str, Any]]:
        trace = trace or {}
        trace.setdefault("steps", {})

        team_block_ids: List[int] = [int(x) for x in list(team_ids or []) if x is not None]
        faculty_contexts_by_id: Dict[int, Dict[str, Any]] = {}
        if not team_block_ids:
            for fctx in list(fac_ctxs or []):
                try:
                    fid = int((fctx or {}).get("faculty_id"))
                    team_block_ids.append(fid)
                    faculty_contexts_by_id[fid] = dict(fctx or {})
                except Exception:
                    continue
        else:
            allowed_ids = set(team_block_ids)
            for fctx in list(fac_ctxs or []):
                try:
                    fid = int((fctx or {}).get("faculty_id"))
                except Exception:
                    continue
                if fid in allowed_ids:
                    faculty_contexts_by_id[fid] = dict(fctx or {})

        stage_inputs = self.context_generator.build_group_justification_stage_inputs_from_contexts(
            opp_ctx=dict(opp_ctx or {}),
            team_ids=team_block_ids,
            match_rows_by_faculty=dict(match_rows_by_faculty or {}),
            faculty_contexts_by_id=dict(faculty_contexts_by_id or {}),
            grant_brief_context=dict(grant_brief_context or {}),
            coverage=dict(coverage or {}),
        )

        grant_ctx = dict((stage_inputs.get("grant_brief_input") or {}).get("grant_context") or {})
        grant_id = grant_ctx.get("opportunity_id")
        grant_title = grant_ctx.get("title")
        grant_link = grant_ctx.get("opportunity_link") or (
            f"https://simpler.grants.gov/opportunity/{grant_id}" if grant_id else ""
        )

        team_role_input = dict(stage_inputs.get("team_role_input") or {})
        why_working_input = dict(stage_inputs.get("why_working_input") or {})
        why_not_input = dict(stage_inputs.get("why_not_working_input") or {})
        rec_template = dict(stage_inputs.get("recommendation_input_template") or {})

        team_block = list(team_role_input.get("team_match_rows") or [])

        # Check grant_brief cache — skip Haiku call if already stored.
        cached_brief: Optional[str] = None
        if grant_id and self.odao is not None:
            kw_row = self.odao.get_opportunity_keyword(str(grant_id))
            cached_brief = self._norm(getattr(kw_row, "grant_brief", None) or "")

        try:
            team_role_chain = self._build_team_role_chain()
            why_working_chain = self._build_why_working_chain()
            why_not_working_chain = self._build_why_not_working_chain()
            recommender_chain = self._build_recommender_chain()

            grant_brief_input = dict(stage_inputs.get("grant_brief_input") or {})
            section_jobs: List[Tuple[str, Any, Dict[str, str]]] = []
            if not cached_brief:
                grant_brief_chain = self._build_grant_brief_chain()
                section_jobs.append(
                    (
                        "grant_brief",
                        grant_brief_chain,
                        {"input_json": self._safe_json(grant_brief_input)},
                    )
                )
            section_jobs += [
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

            if cached_brief:
                grant_brief = GrantBriefOut(grant_quick_explanation=cached_brief)
                logger.info("GROUP_JUSTIFICATION grant_brief_cache_hit opportunity_id=%s", grant_id)
            else:
                grant_brief = section_outputs["grant_brief"]
                # Persist for future requests.
                brief_text = self._norm(getattr(grant_brief, "grant_quick_explanation", "") or "")
                if brief_text and grant_id and self.odao is not None:
                    try:
                        self.odao.save_grant_brief(opportunity_id=str(grant_id), brief=brief_text)
                        self.odao.session.commit()
                    except Exception:
                        logger.warning("Failed to cache grant_brief for opportunity_id=%s", grant_id)

            team_roles: TeamRoleOut = section_outputs["team_roles"]
            why_working: WhyWorkingOut = section_outputs["why_working"]
            why_not: WhyNotWorkingOut = section_outputs["why_not_working"]

            faculty_lookup = list(team_role_input.get("faculty_lookup") or [])
            # LLM now returns plain role strings in faculty_lookup order.
            mapped_member_roles: List[Dict[str, Any]] = []
            raw_roles = [self._norm(x) for x in list(getattr(team_roles, "roles", None) or [])]
            for idx, fac in enumerate(faculty_lookup):
                try:
                    fid = int((fac or {}).get("faculty_id"))
                except Exception:
                    continue
                role = raw_roles[idx] if idx < len(raw_roles) and raw_roles[idx] else "Contributor"
                mapped_member_roles.append({"faculty_id": fid, "role": role})

            # WhyWorkingOut.member_strengths now returns faculty_name + bullets.
            name_to_id: Dict[str, int] = {}
            for fac in faculty_lookup:
                try:
                    fid = int((fac or {}).get("faculty_id"))
                except Exception:
                    continue
                fname = self._norm((fac or {}).get("faculty_name")).lower()
                if fname:
                    name_to_id[fname] = fid

            mapped_strengths_by_id: Dict[int, List[str]] = {}
            for idx, item in enumerate(list(why_working.member_strengths or [])):
                fname = self._norm(getattr(item, "faculty_name", "")).lower()
                fid = name_to_id.get(fname)
                if fid is None and idx < len(faculty_lookup):
                    try:
                        fid = int((faculty_lookup[idx] or {}).get("faculty_id"))
                    except Exception:
                        fid = None
                if fid is None:
                    continue
                bullets = [self._norm(b) for b in list(getattr(item, "bullets", []) or []) if self._norm(b)]
                if fid not in mapped_strengths_by_id:
                    mapped_strengths_by_id[fid] = []
                for b in bullets:
                    if b not in mapped_strengths_by_id[fid]:
                        mapped_strengths_by_id[fid].append(b)
            mapped_member_strengths = [
                {"faculty_id": fid, "bullets": bullets}
                for fid, bullets in mapped_strengths_by_id.items()
                if list(bullets or [])
            ]

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
            rec_input = dict(rec_template or {})
            rec_input.update(
                {
                    "grant": dict((rec_template or {}).get("grant") or {
                        "id": grant_id,
                        "title": grant_title,
                        "link": grant_link,
                    }),
                }
            )
            rec_input.update(
                {
                "team_roles": team_roles.model_dump(),
                "why_working": why_working.model_dump(),
                "why_not_working": why_not.model_dump(),
                }
            )
            recommendation = recommender_chain.invoke({"input_json": self._safe_json(rec_input)})
            trace["steps"]["recommendation"] = {
                "status": "ok",
                "input": rec_input,
                "output": recommendation.model_dump(),
            }

            justification = GroupJustificationOut(
                one_paragraph=grant_brief.grant_quick_explanation or "",
                why_working_summary=why_working.summary or "",
                member_roles=mapped_member_roles,
                coverage={
                    "strong": [],
                    "partial": [],
                    "missing": why_not.missing,
                },
                member_strengths=mapped_member_strengths,
                why_not_working=why_not.why_not_working,
                recommendation=recommendation.recommendation,
                team_grant_fit=float(getattr(recommendation, "team_grant_fit", 0.0) or 0.0),
            )
            return justification, trace
        except Exception as e:
            logger.exception(
                "GROUP_JUSTIFICATION_PIPELINE_FAILED meta=%s",
                json.dumps(
                    {
                        "opportunity_id": grant_id,
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
            for fac in list(team_role_input.get("faculty_lookup") or []):
                try:
                    fid = int((fac or {}).get("faculty_id"))
                except Exception:
                    continue
                fallback_roles.append(
                    {
                        "faculty_id": fid,
                        "role": "Contributor",
                    }
                )
            justification = GroupJustificationOut(
                one_paragraph="Insufficient model output to generate a complete structured justification.",
                why_working_summary="",
                member_roles=fallback_roles,
                coverage={"strong": [], "partial": [], "missing": []},
                member_strengths=[],
                why_not_working=["Structured writer stages failed; verify model/service health and rerun."],
                recommendation="Review this result manually and rerun generation.",
                team_grant_fit=0.0,
            )
            return justification, trace
