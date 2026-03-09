from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from tmp.agentic_arch.faculty_agent import FacultyProfessionAgent
from tmp.agentic_arch.grant_agent import GrantAgent
from tmp.agentic_arch.matcher import OneToOneProfessionMatcher
from tmp.agentic_arch.tools import FacultyTools, GrantTools
from tmp.agentic_arch.models import OneToOneMatch


class ProfessionGrantMatchOrchestrator:
    """
    Independent agentic architecture:
    1) Faculty agent profiles profession focus
    2) Grant agent collects grant snapshots with parallel tool calls
    3) Matcher ranks one-to-one matches
    """

    def __init__(self, *, faculty_tools: FacultyTools, grant_tools: GrantTools):
        self.faculty_agent = FacultyProfessionAgent(tools=faculty_tools)
        self.grant_agent = GrantAgent(tools=grant_tools)
        self.matcher = OneToOneProfessionMatcher()

    async def run_one_to_one(
        self,
        *,
        faculty_email: str,
        candidate_grant_k: int = 20,
        result_top_k: int = 5,
    ) -> Dict[str, Any]:
        profile = await self.faculty_agent.profile_profession(email=faculty_email)

        grant_snapshots = await self.grant_agent.collect_candidate_grants(
            faculty_profile=profile,
            top_k=max(1, int(candidate_grant_k or 20)),
        )

        matches: List[OneToOneMatch] = self.matcher.rank(
            faculty_profile=profile,
            grants=grant_snapshots,
            top_k=max(1, int(result_top_k or 5)),
        )

        return {
            "faculty_profile": {
                "email": profile.email,
                "faculty_name": profile.basic_info.faculty_name,
                "position": profile.basic_info.position,
                "organizations": list(profile.basic_info.organizations or []),
                "profession_focus": list(profile.profession_focus or []),
            },
            "candidate_grants_considered": len(grant_snapshots),
            "matches": [
                {
                    "faculty_email": m.faculty_email,
                    "grant_id": m.grant_id,
                    "grant_name": m.grant_name,
                    "agency_name": m.agency_name,
                    "close_date": m.close_date,
                    "score": m.score,
                    "reason": m.reason,
                    "matched_professions": list(m.matched_professions or []),
                }
                for m in matches
            ],
            "agent_trace": [
                "faculty_agent.profile_profession",
                "grant_agent.collect_candidate_grants",
                "grant_agent._build_snapshot (parallel metadata + requirement)",
                "grant_requirement_agent.analyze (parallel sub-tools)",
                "one_to_one_matcher.rank",
            ],
        }

    def run_one_to_one_sync(
        self,
        *,
        faculty_email: str,
        candidate_grant_k: int = 20,
        result_top_k: int = 5,
    ) -> Dict[str, Any]:
        return asyncio.run(
            self.run_one_to_one(
                faculty_email=faculty_email,
                candidate_grant_k=candidate_grant_k,
                result_top_k=result_top_k,
            )
        )
