from __future__ import annotations

import asyncio
from typing import Dict, List, Tuple

from tmp.agentic_arch.grant_agent import GrantRequirementAgent
from tmp.agentic_arch.models import GrantMetadata, GrantRequirement, GrantSnapshot
from tmp.agentic_arch.tools import GrantTools
from tmp.agentic_arch_v2.models import QueryAnswer, QueryItem


class GrantConversationAgent:
    def __init__(self, tools: GrantTools):
        self.tools = tools
        self.requirement_agent = GrantRequirementAgent(tools=tools)

    async def discover_candidate_grants(self, *, profession_focus: List[str], top_k: int = 20) -> List[str]:
        return await self.tools.search_candidate_grants(
            profession_focus=list(profession_focus or []),
            top_k=max(1, int(top_k or 20)),
        )

    async def _metadata(self, grant_id: str) -> GrantMetadata:
        return await self.tools.fetch_metadata(grant_id)

    async def _requirement(self, grant_id: str) -> GrantRequirement:
        return await self.requirement_agent.analyze(grant_id)

    @staticmethod
    def _overlap(a: List[str], b: List[str]) -> Tuple[float, List[str], List[str]]:
        a_norm = [str(x or "").strip().lower() for x in (a or []) if str(x or "").strip()]
        b_norm = [str(x or "").strip().lower() for x in (b or []) if str(x or "").strip()]
        if not a_norm or not b_norm:
            return 0.0, [], list(b_norm)

        matched_requirements: List[str] = []
        for req in b_norm:
            if any(req in term or term in req for term in a_norm):
                matched_requirements.append(req)

        missing = [req for req in b_norm if req not in matched_requirements]

        matched_faculty_terms: List[str] = []
        for term in a_norm:
            if any(req in term or term in req for req in matched_requirements):
                matched_faculty_terms.append(term)

        score = len(matched_requirements) / max(len(b_norm), 1)
        score = min(1.0, max(0.0, float(score)))
        return score, matched_faculty_terms, missing

    @staticmethod
    def _confidence_for_fields(answer: Dict[str, object], expected_fields: List[str], base: float = 0.6) -> float:
        fields = [str(x).strip() for x in (expected_fields or []) if str(x).strip()]
        if not fields:
            return min(1.0, max(0.0, base))
        got = 0
        for f in fields:
            v = answer.get(f)
            if v is None:
                continue
            if isinstance(v, (list, tuple, dict, str)) and len(v) == 0:
                continue
            got += 1
        return min(1.0, max(0.0, got / max(len(fields), 1)))

    async def _answer_one(self, *, query: QueryItem, faculty_profile) -> QueryAnswer:
        intent = str(query.intent or "").strip().lower()
        grant_id = str(query.grant_id or "").strip()
        if not grant_id:
            return QueryAnswer(
                query_id=query.query_id,
                target_agent="grant",
                intent=intent,
                answer={"error": "missing grant_id"},
                confidence=0.0,
                evidence=[],
                followup_queries=[],
            )

        metadata_task = self._metadata(grant_id)
        requirement_task = self._requirement(grant_id)

        # independent grant info + requirement in parallel
        metadata, requirement = await asyncio.gather(metadata_task, requirement_task)
        snapshot = GrantSnapshot(metadata=metadata, requirement=requirement)

        if intent == "grant_metadata":
            answer = {
                "grant_id": snapshot.metadata.grant_id,
                "grant_name": snapshot.metadata.grant_name,
                "agency_name": snapshot.metadata.agency_name,
                "close_date": snapshot.metadata.close_date,
            }
            return QueryAnswer(
                query_id=query.query_id,
                target_agent="grant",
                intent=intent,
                answer=answer,
                confidence=self._confidence_for_fields(answer, query.expected_fields, base=0.8),
                evidence=[x for x in [snapshot.metadata.grant_name, snapshot.metadata.agency_name] if x],
                followup_queries=[],
            )

        if intent in {"grant_requirement", "grant_requirement_priority", "grant_profession_fit_probe"}:
            domains = list(snapshot.requirement.domains or [])
            specs = list(snapshot.requirement.specializations or [])
            eligibility = list(snapshot.requirement.eligibility or [])
            deliverables = list(snapshot.requirement.deliverables or [])

            all_reqs = [*domains, *specs]
            score, matched, missing = self._overlap(
                list(faculty_profile.profession_focus or []),
                all_reqs,
            )

            answer: Dict[str, object] = {
                "grant_id": grant_id,
                "grant_name": snapshot.metadata.grant_name,
                "agency_name": snapshot.metadata.agency_name,
                "close_date": snapshot.metadata.close_date,
                "domains": domains,
                "specializations": specs,
                "eligibility": eligibility,
                "deliverables": deliverables,
                "matched_terms": matched,
                "missing_terms": missing,
                "fit_score": score,
                "priority_specializations": specs[:5],
            }

            conf = 0.45 + 0.45 * self._confidence_for_fields(answer, query.expected_fields, base=0.6)
            if score > 0:
                conf = min(1.0, conf + 0.1)

            followups: List[QueryItem] = []
            if missing and score < query.confidence_threshold:
                followups.append(
                    QueryItem(
                        query_id=f"{query.query_id}_fb_faculty_terms",
                        target_agent="faculty",
                        intent="faculty_evidence_for_terms",
                        question="Do you have evidence for missing grant terms?",
                        expected_fields=["evidence_hits", "coverage"],
                        priority=0.9,
                        confidence_threshold=0.65,
                        grant_id=grant_id,
                        context={
                            "required_terms": missing[:8],
                            "from_grant_id": grant_id,
                        },
                    )
                )

            return QueryAnswer(
                query_id=query.query_id,
                target_agent="grant",
                intent=intent,
                answer=answer,
                confidence=conf,
                evidence=matched[:8],
                followup_queries=followups,
            )

        return QueryAnswer(
            query_id=query.query_id,
            target_agent="grant",
            intent=intent,
            answer={"grant_id": grant_id, "note": "unsupported intent"},
            confidence=0.2,
            evidence=[],
            followup_queries=[],
        )

    async def answer_queries(
        self,
        *,
        queries: List[QueryItem],
        faculty_profile,
    ) -> List[QueryAnswer]:
        tasks = [self._answer_one(query=q, faculty_profile=faculty_profile) for q in (queries or [])]
        if not tasks:
            return []
        return list(await asyncio.gather(*tasks))

    async def collect_snapshots_for_grants(self, grant_ids: List[str]) -> List[GrantSnapshot]:
        async def _one(gid: str) -> GrantSnapshot:
            md, req = await asyncio.gather(self._metadata(gid), self._requirement(gid))
            return GrantSnapshot(metadata=md, requirement=req)

        tasks = [_one(str(gid)) for gid in (grant_ids or []) if str(gid).strip()]
        if not tasks:
            return []
        return list(await asyncio.gather(*tasks))
