from __future__ import annotations

import asyncio
from typing import Dict, List, Set

from tmp.agentic_arch.matcher import OneToOneProfessionMatcher
from tmp.agentic_arch.tools import FacultyTools, GrantTools
from tmp.agentic_arch_v2.faculty_agent import FacultyConversationAgent
from tmp.agentic_arch_v2.grant_agent import GrantConversationAgent
from tmp.agentic_arch_v2.models import (
    AgenticRunResult,
    OrchestrationRound,
    QueryAnswer,
    QueryItem,
)
from tmp.agentic_arch_v2.planner import LLMQueryPlanner, expand_queries_for_grants


class ProfessionGrantConversationOrchestrator:
    def __init__(
        self,
        *,
        faculty_tools: FacultyTools,
        grant_tools: GrantTools,
        planner: LLMQueryPlanner | None = None,
    ):
        self.faculty_agent = FacultyConversationAgent(tools=faculty_tools)
        self.grant_agent = GrantConversationAgent(tools=grant_tools)
        self.planner = planner or LLMQueryPlanner()
        self.matcher = OneToOneProfessionMatcher()

    @staticmethod
    def _query_signature(q: QueryItem) -> str:
        parts = [
            str(q.target_agent or "").strip().lower(),
            str(q.intent or "").strip().lower(),
            str(q.grant_id or "").strip(),
            str(q.question or "").strip().lower(),
            str(sorted((q.context or {}).get("required_terms") or [])),
        ]
        return "|".join(parts)

    @staticmethod
    def _critical_failed(answers: List[QueryAnswer]) -> bool:
        for a in answers:
            if a.target_agent == "grant" and a.intent in {
                "grant_metadata",
                "grant_requirement",
                "grant_profession_fit_probe",
            }:
                if float(a.confidence) < 0.6:
                    return True
        return False

    async def run(
        self,
        *,
        faculty_email: str,
        candidate_grant_k: int = 20,
        result_top_k: int = 5,
        max_rounds: int = 3,
        max_queries_per_round: int = 120,
    ) -> AgenticRunResult:
        faculty_profile = await self.faculty_agent.build_profile(email=faculty_email)

        candidate_grant_ids = await self.grant_agent.discover_candidate_grants(
            profession_focus=list(faculty_profile.profession_focus or []),
            top_k=max(1, int(candidate_grant_k or 20)),
        )

        plan = self.planner.plan_initial_queries(
            faculty_profile={
                "email": faculty_profile.email,
                "faculty_name": faculty_profile.basic_info.faculty_name,
                "position": faculty_profile.basic_info.position,
                "organizations": list(faculty_profile.basic_info.organizations or []),
                "profession_focus": list(faculty_profile.profession_focus or []),
            },
            candidate_grant_ids=list(candidate_grant_ids or []),
        )

        pending_queries = expand_queries_for_grants(
            plan=plan,
            candidate_grant_ids=list(candidate_grant_ids or []),
        )

        rounds: List[OrchestrationRound] = []
        seen_signatures: Set[str] = set()

        stop_reason = "max_rounds_reached"

        for round_index in range(1, max(1, int(max_rounds or 1)) + 1):
            # Deduplicate + cap
            unique_pending: List[QueryItem] = []
            for q in pending_queries:
                sig = self._query_signature(q)
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)
                unique_pending.append(q)
                if len(unique_pending) >= max(1, int(max_queries_per_round or 1)):
                    break

            if not unique_pending:
                stop_reason = "no_more_queries"
                break

            grant_queries = [q for q in unique_pending if q.target_agent == "grant"]
            faculty_queries = [q for q in unique_pending if q.target_agent == "faculty"]

            grant_answers_task = self.grant_agent.answer_queries(
                queries=grant_queries,
                faculty_profile=faculty_profile,
            )
            faculty_answers_task = self.faculty_agent.answer_queries(
                queries=faculty_queries,
                faculty_profile=faculty_profile,
            )

            grant_answers, faculty_answers = await asyncio.gather(
                grant_answers_task,
                faculty_answers_task,
            )
            answers = [*grant_answers, *faculty_answers]

            rounds.append(
                OrchestrationRound(
                    round_index=round_index,
                    queries=unique_pending,
                    answers=answers,
                )
            )

            followups: List[QueryItem] = []
            for answer in answers:
                followups.extend(list(answer.followup_queries or []))

            if not followups and not self._critical_failed(answers):
                stop_reason = "confidence_converged"
                break

            pending_queries = followups

        grant_snapshots = await self.grant_agent.collect_snapshots_for_grants(list(candidate_grant_ids or []))
        matches = self.matcher.rank(
            faculty_profile=faculty_profile,
            grants=grant_snapshots,
            top_k=max(1, int(result_top_k or 5)),
        )

        return AgenticRunResult(
            faculty_profile=faculty_profile,
            candidate_grant_ids=list(candidate_grant_ids or []),
            grant_snapshots=grant_snapshots,
            rounds=rounds,
            matches=matches,
            stop_reason=stop_reason,
        )

    def run_sync(
        self,
        *,
        faculty_email: str,
        candidate_grant_k: int = 20,
        result_top_k: int = 5,
        max_rounds: int = 3,
        max_queries_per_round: int = 120,
    ) -> AgenticRunResult:
        return asyncio.run(
            self.run(
                faculty_email=faculty_email,
                candidate_grant_k=candidate_grant_k,
                result_top_k=result_top_k,
                max_rounds=max_rounds,
                max_queries_per_round=max_queries_per_round,
            )
        )
