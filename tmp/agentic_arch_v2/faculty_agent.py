from __future__ import annotations

import asyncio
from typing import List

from tmp.agentic_arch.faculty_agent import FacultyProfessionAgent
from tmp.agentic_arch.tools import FacultyTools
from tmp.agentic_arch_v2.models import QueryAnswer, QueryItem


class FacultyConversationAgent:
    def __init__(self, tools: FacultyTools):
        self.tools = tools
        self.profile_agent = FacultyProfessionAgent(tools=tools)

    @staticmethod
    def _norm_list(values: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for x in values or []:
            token = str(x or "").strip()
            if not token:
                continue
            lowered = token.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            out.append(token)
        return out

    async def build_profile(self, *, email: str):
        return await self.profile_agent.profile_profession(email=email)

    async def _answer_one(self, *, query: QueryItem, faculty_profile) -> QueryAnswer:
        intent = str(query.intent or "").strip().lower()
        focus = self._norm_list(list(faculty_profile.profession_focus or []))
        keywords = self._norm_list(list(faculty_profile.keywords or []))
        pubs = list((faculty_profile.evidence or {}).get("publication_abstracts") or [])
        addl = list((faculty_profile.evidence or {}).get("additional_text") or [])

        if intent in {"faculty_profession_detail", "faculty_context"}:
            answer = {
                "profession_focus": focus,
                "keywords": keywords[:20],
                "evidence": {
                    "organizations": list((faculty_profile.evidence or {}).get("organizations") or []),
                    "publication_abstracts": pubs[:5],
                    "additional_text": addl[:3],
                },
            }
            conf = 0.9 if focus else 0.45
            return QueryAnswer(
                query_id=query.query_id,
                target_agent="faculty",
                intent=intent,
                answer=answer,
                confidence=conf,
                evidence=focus[:5],
                followup_queries=[],
            )

        # intent: faculty_evidence_for_terms
        required_terms = self._norm_list(list(query.context.get("required_terms") or []))
        if not required_terms:
            required_terms = self._norm_list([str(query.question or "")])

        evidence_hits: List[str] = []
        bag = "\n".join([*focus, *keywords, *pubs, *addl]).lower()
        for term in required_terms:
            t = str(term).lower()
            if t and t in bag:
                evidence_hits.append(term)

        coverage = len(evidence_hits) / max(len(required_terms), 1)
        confidence = min(1.0, 0.2 + 0.8 * coverage)

        followups: List[QueryItem] = []
        if coverage < 0.5 and query.grant_id:
            followups.append(
                QueryItem(
                    query_id=f"{query.query_id}_fb_grant_priority",
                    target_agent="grant",
                    intent="grant_requirement_priority",
                    question="Clarify highest-priority specialization requirements for this grant.",
                    expected_fields=["priority_specializations"],
                    priority=0.88,
                    confidence_threshold=0.65,
                    grant_id=query.grant_id,
                    context={
                        "source_query_id": query.query_id,
                    },
                )
            )

        return QueryAnswer(
            query_id=query.query_id,
            target_agent="faculty",
            intent=intent,
            answer={
                "required_terms": required_terms,
                "evidence_hits": evidence_hits,
                "coverage": coverage,
                "profession_focus": focus,
            },
            confidence=confidence,
            evidence=evidence_hits,
            followup_queries=followups,
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
