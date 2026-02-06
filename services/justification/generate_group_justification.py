from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_llm_client
from dao.opportunity_dao import OpportunityDAO
from dao.faculty_dao import FacultyDAO
from db.db_conn import SessionLocal
from dto.llm_response_dto import GroupJustificationOut
from services.matching.group_match_super_faculty import run_group_match


# -------------------------
# Structured models (Planner/Critic)
# -------------------------

class PlannerRequest(BaseModel):
    """What extra info the LLM wants before writing the justification."""
    opp_fields: List[str] = Field(default_factory=list, description="Additional opportunity fields to fetch if available.")
    faculty_fields: List[str] = Field(default_factory=list, description="Additional faculty fields to fetch if available.")
    ask_for_more_faculty: bool = Field(default=False, description="Whether writer needs richer faculty context than keywords.")
    ask_for_more_opp: bool = Field(default=False, description="Whether writer needs richer opportunity context than summary/keywords.")
    focus_points: List[str] = Field(default_factory=list, description="Key angles to emphasize in justification.")


class CriticVerdict(BaseModel):
    """Critic either approves or requests more info and a rewrite."""
    ok: bool = Field(..., description="True if justification is acceptable and grounded.")
    issues: List[str] = Field(default_factory=list, description="Problems found: vagueness, missing grounding, etc.")
    request_more: PlannerRequest = Field(default_factory=PlannerRequest, description="If not ok, what more to fetch.")


# -------------------------
# Prompts
# -------------------------

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a grant-team match planning agent. Your job is to decide what information is missing "
     "to write a specific, grounded justification. "
     "You must be conservative: only ask for fields that would materially improve specificity."),
    ("user",
     "Given this draft context JSON:\n{input_json}\n\n"
     "Return a PlannerRequest specifying what extra fields would help. "
     "If current info is sufficient, return empty lists and false flags, but still include 2-6 focus_points.")
])

WRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You write concise, high-specificity grant-team match justifications.\n"
     "Grounding rules:\n"
     "- Use ONLY facts present in the provided JSON context.\n"
     "- Do NOT invent publications, awards, institutions, methods, or outcomes.\n"
     "- If a needed detail is missing, speak generally and acknowledge uncertainty.\n"
     "- Prefer concrete mappings between grant keywords/aims and each faculty member's keywords.\n"
     "Output must match the GroupJustificationOut schema exactly."),
    ("user",
     "Context JSON:\n{input_json}\n\n"
     "Task:\n"
     "Write a justification that explains why this team matches the grant. "
     "Use coverage and keywords to justify complementarity and reduce redundancy. "
     "Include 1-3 actionable suggestions to improve fit (e.g., add skill X, clarify aim Y) if appropriate.")
])

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict reviewer. Check the justification for:\n"
     "- Grounding: any claims not supported by the provided context JSON\n"
     "- Specificity: too generic or templated\n"
     "- Coverage: does it explain team complementarity relative to grant keywords?\n"
     "- Actionability: are recommendations concrete?\n"
     "If problems exist, set ok=false and request more data via PlannerRequest.\n"
     "If it is solid, ok=true."),
    ("user",
     "Context JSON:\n{context_json}\n\n"
     "Justification JSON (already structured):\n{justification_json}\n\n"
     "Return CriticVerdict.")
])


# -------------------------
# Context building utilities
# -------------------------

def build_base_payload(
    *,
    opp_ctx: Dict[str, Any],
    fac_ctxs: List[Dict[str, Any]],
    coverage: Any,
    group_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Base payload sent to planner/writer/critic."""
    payload: Dict[str, Any] = {
        "grant": {
            "id": opp_ctx.get("opportunity_id") or opp_ctx.get("id"),
            "title": opp_ctx.get("title"),
            "agency": opp_ctx.get("agency"),
            "summary": opp_ctx.get("summary"),
            "keywords": opp_ctx.get("keywords"),
        },
        "team": [
            {
                "faculty_id": f.get("faculty_id") or f.get("id"),
                "name": f.get("name"),
                "email": f.get("email"),
                "keywords": f.get("keywords"),
                # Optional extra fields can be merged in later
            }
            for f in fac_ctxs
        ],
        "coverage": coverage,
    }
    if group_meta:
        payload["group_match"] = group_meta
    return payload


def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


# -------------------------
# Provider-agnostic LLM chain builder
# -------------------------

@dataclass
class LLMs:
    planner: Any
    writer: Any
    critic: Any


def build_llms() -> LLMs:
    """
    Uses your existing get_llm_client().build().
    For best results:
      - planner: lower temp
      - writer: moderate temp
      - critic: low temp
    If your builder supports params, set them there; otherwise it's fine as-is.
    """
    base = get_llm_client().build()
    # If your client supports .bind(temperature=...), uncomment and tune:
    # planner = base.bind(temperature=0.2)
    # writer  = base.bind(temperature=0.5)
    # critic  = base.bind(temperature=0.1)
    planner = base
    writer = base
    critic = base
    return LLMs(planner=planner, writer=writer, critic=critic)


# -------------------------
# Optional: deterministic enrichment hooks
# -------------------------

def enrich_opportunity_context(odao: OpportunityDAO, opp_ctx: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    """
    Attempt to add extra fields if your DAO/context supports them.
    Safe: will only add keys that already exist in returned context.
    """
    # If you have richer DAO methods, call them here.
    # For now, we re-fetch the same context and keep only requested extra fields if present.
    full = odao.get_opportunity_context(opp_ctx.get("opportunity_id") or opp_ctx.get("id")) or {}
    for k in fields:
        if k in full and k not in opp_ctx:
            opp_ctx[k] = full[k]
    return opp_ctx


def enrich_faculty_context(fdao: FacultyDAO, fac_ctx: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    """
    Attempt to add extra fields if your DAO/context supports them.
    If you later add methods like get_faculty_profile/get_recent_pubs, wire them here.
    """
    fid = fac_ctx.get("faculty_id") or fac_ctx.get("id")
    # Re-fetch keyword context (or swap to richer DAO calls later).
    full = fdao.get_faculty_keyword_context(fid) or {}
    # Merge requested fields if present
    for k in fields:
        if k in full and k not in fac_ctx:
            fac_ctx[k] = full[k]
    return fac_ctx


# -------------------------
# Agentic justification pipeline
# -------------------------

class JustificationEngine:
    def __init__(self, *, odao: OpportunityDAO, fdao: FacultyDAO):
        self.odao = odao
        self.fdao = fdao
        self.llms = build_llms()

        self.planner_chain = PLANNER_PROMPT | self.llms.planner.with_structured_output(PlannerRequest)
        self.writer_chain = WRITER_PROMPT | self.llms.writer.with_structured_output(GroupJustificationOut)
        self.critic_chain = CRITIC_PROMPT | self.llms.critic.with_structured_output(CriticVerdict)

    def run_one(
        self,
        *,
        opp_ctx: Dict[str, Any],
        fac_ctxs: List[Dict[str, Any]],
        coverage: Any,
        group_meta: Optional[Dict[str, Any]] = None,
        max_rewrite_rounds: int = 1,
        trace: Optional[Dict[str, Any]] = None,
    ) -> Tuple[GroupJustificationOut, Dict[str, Any]]:
        """
        Returns (justification, trace).
        trace includes planner/critic outputs and the final context used.
        """
        trace = trace or {}
        trace.setdefault("planner", None)
        trace.setdefault("critic", [])
        trace.setdefault("context_versions", [])

        # Build initial context
        context = build_base_payload(
            opp_ctx=opp_ctx,
            fac_ctxs=fac_ctxs,
            coverage=coverage,
            group_meta=group_meta,
        )
        trace["context_versions"].append(json.loads(safe_json(context)))

        # 1) Plan
        plan: PlannerRequest = self.planner_chain.invoke({"input_json": safe_json(context)})
        trace["planner"] = json.loads(plan.model_dump_json())

        # 2) Deterministic research/enrichment based on plan
        if plan.ask_for_more_opp or plan.opp_fields:
            opp_ctx = enrich_opportunity_context(self.odao, opp_ctx, plan.opp_fields)
        if plan.ask_for_more_faculty or plan.faculty_fields:
            fac_ctxs = [enrich_faculty_context(self.fdao, f, plan.faculty_fields) for f in fac_ctxs]

        # Rebuild context after enrichment
        context = build_base_payload(
            opp_ctx=opp_ctx,
            fac_ctxs=fac_ctxs,
            coverage=coverage,
            group_meta=group_meta,
        )
        # Include focus points explicitly for writer (helps a lot)
        context["focus_points"] = plan.focus_points
        trace["context_versions"].append(json.loads(safe_json(context)))

        # 3) Write + (optional) Critic + rewrite
        justification: GroupJustificationOut = self.writer_chain.invoke({"input_json": safe_json(context)})

        for _ in range(max_rewrite_rounds + 1):
            verdict: CriticVerdict = self.critic_chain.invoke({
                "context_json": safe_json(context),
                "justification_json": safe_json(justification.model_dump()),
            })
            trace["critic"].append(json.loads(verdict.model_dump_json()))

            if verdict.ok:
                return justification, trace

            # If critic requests more info, enrich and rewrite once
            req = verdict.request_more
            if req.ask_for_more_opp or req.opp_fields:
                opp_ctx = enrich_opportunity_context(self.odao, opp_ctx, req.opp_fields)
            if req.ask_for_more_faculty or req.faculty_fields:
                fac_ctxs = [enrich_faculty_context(self.fdao, f, req.faculty_fields) for f in fac_ctxs]

            context = build_base_payload(
                opp_ctx=opp_ctx,
                fac_ctxs=fac_ctxs,
                coverage=coverage,
                group_meta=group_meta,
            )
            context["focus_points"] = (req.focus_points or context.get("focus_points") or [])
            trace["context_versions"].append(json.loads(safe_json(context)))

            justification = self.writer_chain.invoke({"input_json": safe_json(context)})

        # If still not ok, return best attempt + trace (don’t fail the whole batch)
        return justification, trace



def run_justifications_from_group_results_agentic(
    *,
    faculty_email: str,
    team_size: int,
    limit_rows: int = 500,
    include_trace: bool = False,
) -> str:
    with SessionLocal() as sess:
        odao = OpportunityDAO(sess)
        fdao = FacultyDAO(sess)
        engine = JustificationEngine(odao=odao, fdao=fdao)

        group_results = run_group_match(
            faculty_email=faculty_email,
            team_size=team_size,
            limit_rows=limit_rows,
        )
        if not group_results:
            raise ValueError(f"No group matches found for {faculty_email}")

        # Caches
        opp_cache: Dict[str, Dict[str, Any]] = {}
        fac_cache: Dict[int, Dict[str, Any]] = {}

        results: List[Dict[str, Any]] = []

        for idx, r in enumerate(group_results):
            opp_id = r["opp_id"]
            team: List[int] = r["team"]
            coverage = r.get("final_coverage")

            # --- opportunity context ---
            if opp_id not in opp_cache:
                opp_cache[opp_id] = odao.get_opportunity_context(opp_id) or {}
            opp_ctx = opp_cache[opp_id]

            if not opp_ctx:
                results.append({
                    "index": idx,
                    "grant_id": opp_id,
                    "team": team,
                    "error": "Opportunity not found",
                })
                continue

            # --- faculty contexts ---
            fac_ctxs: List[Dict[str, Any]] = []
            for fid in team:
                if fid not in fac_cache:
                    fac_cache[fid] = fdao.get_faculty_keyword_context(fid) or {}
                if fac_cache[fid]:
                    fac_ctxs.append(fac_cache[fid])

            if not fac_ctxs:
                results.append({
                    "index": idx,
                    "grant_id": opp_id,
                    "team": team,
                    "error": "No faculty contexts found for team",
                })
                continue

            # Optional group metadata if you have it in r; safe default:
            group_meta = {
                "group_id": r.get("group_id") or r.get("id"),
                "lambda": r.get("lambda"),
                "k": r.get("k"),
                "objective": r.get("objective"),
                "redundancy": r.get("redundancy"),
                "meta": r.get("meta"),
            }

            try:
                justification, trace = engine.run_one(
                    opp_ctx=dict(opp_ctx),          # copy so enrichment doesn’t poison cache
                    fac_ctxs=[dict(f) for f in fac_ctxs],
                    coverage=coverage,
                    group_meta=group_meta,
                    max_rewrite_rounds=1,
                    trace={"index": idx, "opp_id": opp_id, "team": team},
                )

                out = {
                    "index": idx,
                    "grant_id": opp_id,
                    "team": team,
                    "justification": justification.model_dump(),
                }
                if include_trace:
                    out["trace"] = trace

                results.append(out)

            except Exception as e:
                results.append({
                    "index": idx,
                    "grant_id": opp_id,
                    "team": team,
                    "error": f"{type(e).__name__}: {e}",
                })

        return json.dumps(results, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    email = "AbbasiB@oregonstate.edu"
    print(
        run_justifications_from_group_results_agentic(
            faculty_email=email,
            team_size=3,
            limit_rows=200,
            include_trace=False,
        )
    )