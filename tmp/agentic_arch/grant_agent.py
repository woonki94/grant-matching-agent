from __future__ import annotations

import asyncio
from typing import List

from tmp.agentic_arch.tools import GrantTools
from tmp.agentic_arch.models import FacultyProfessionProfile, GrantRequirement, GrantSnapshot


class GrantRequirementAgent:
    """
    Requirement agent: runs requirement tool calls in parallel.
    """

    def __init__(self, tools: GrantTools):
        self.tools = tools

    async def analyze(self, grant_id: str) -> GrantRequirement:
        domains_task = self.tools.fetch_requirement_domains(grant_id)
        specs_task = self.tools.fetch_requirement_specializations(grant_id)
        eligibility_task = self.tools.fetch_requirement_eligibility(grant_id)
        deliverables_task = self.tools.fetch_requirement_deliverables(grant_id)

        domains, specs, eligibility, deliverables = await asyncio.gather(
            domains_task,
            specs_task,
            eligibility_task,
            deliverables_task,
        )

        return GrantRequirement(
            domains=list(domains or []),
            specializations=list(specs or []),
            eligibility=list(eligibility or []),
            deliverables=list(deliverables or []),
        )


class GrantAgent:
    """
    Grant agent:
    1) find candidate grants for profession focus
    2) for each candidate grant, parallel calls:
       - metadata tool call
       - requirement agent call
    """

    def __init__(self, tools: GrantTools):
        self.tools = tools
        self.requirement_agent = GrantRequirementAgent(tools=tools)

    async def _build_snapshot(self, grant_id: str) -> GrantSnapshot:
        metadata_task = self.tools.fetch_metadata(grant_id)
        requirement_task = self.requirement_agent.analyze(grant_id)
        metadata, requirement = await asyncio.gather(metadata_task, requirement_task)
        return GrantSnapshot(metadata=metadata, requirement=requirement)

    async def collect_candidate_grants(
        self,
        *,
        faculty_profile: FacultyProfessionProfile,
        top_k: int = 10,
    ) -> List[GrantSnapshot]:
        candidate_ids = await self.tools.search_candidate_grants(
            profession_focus=list(faculty_profile.profession_focus or []),
            top_k=max(int(top_k or 10), 1),
        )
        if not candidate_ids:
            return []

        snapshots = await asyncio.gather(
            *(self._build_snapshot(grant_id) for grant_id in candidate_ids),
        )
        return list(snapshots)
