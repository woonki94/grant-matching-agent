from __future__ import annotations

import asyncio
from typing import List, Set

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
from tmp.agentic_langgraph.state import LangGraphConversationState


class ProfessionGrantLangGraphOrchestrator:
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
        self.graph = self._build_graph()

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
    def _is_critical_failed_answer(a: QueryAnswer) -> bool:
        if a.target_agent != "grant":
            return False
        if a.intent not in {"grant_metadata", "grant_requirement", "grant_profession_fit_probe"}:
            return False
        return float(a.confidence) < 0.6

    async def _node_bootstrap(self, state: LangGraphConversationState) -> LangGraphConversationState:
        faculty_email = str(state.get("faculty_email") or "").strip()
        if not faculty_email:
            raise ValueError("faculty_email is required")

        candidate_grant_k = max(1, int(state.get("candidate_grant_k") or 20))
        faculty_profile = await self.faculty_agent.build_profile(email=faculty_email)
        candidate_grant_ids = await self.grant_agent.discover_candidate_grants(
            profession_focus=list(faculty_profile.profession_focus or []),
            top_k=candidate_grant_k,
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

        return {
            "faculty_profile": faculty_profile,
            "candidate_grant_ids": list(candidate_grant_ids or []),
            "pending_queries": pending_queries,
            "round_queries": [],
            "remaining_round_queries": [],
            "next_round_queries": [],
            "current_query": None,
            "current_answer": None,
            "current_round_answers": [],
            "rounds": [],
            "seen_signatures": set(),
            "round_index": 1,
            "round_critical_failed": False,
            "route_decision": "",
            "stop_reason": "",
            "grant_snapshots": [],
            "matches": [],
        }

    async def _node_round_prepare(self, state: LangGraphConversationState) -> LangGraphConversationState:
        pending_queries = list(state.get("pending_queries") or [])
        seen_signatures: Set[str] = set(state.get("seen_signatures") or set())
        max_queries = max(1, int(state.get("max_queries_per_round") or 120))

        unique_pending: List[QueryItem] = []
        for q in pending_queries:
            sig = self._query_signature(q)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            unique_pending.append(q)
            if len(unique_pending) >= max_queries:
                break

        return {
            "seen_signatures": seen_signatures,
            "round_queries": unique_pending,
            "remaining_round_queries": list(unique_pending),
            "pending_queries": [],
            "next_round_queries": [],
            "current_round_answers": [],
            "round_critical_failed": False,
            "current_query": None,
            "current_answer": None,
            "route_decision": "",
        }

    async def _node_router(self, state: LangGraphConversationState) -> LangGraphConversationState:
        remaining = list(state.get("remaining_round_queries") or [])
        if not remaining:
            return {"route_decision": "finalize_round", "current_query": None}

        query = remaining.pop(0)
        target = str(query.target_agent or "").strip().lower()
        if target == "grant":
            decision = "grant_answer"
        elif target == "faculty":
            decision = "faculty_answer"
        else:
            decision = "skip_query"

        return {
            "current_query": query,
            "remaining_round_queries": remaining,
            "route_decision": decision,
        }

    async def _node_grant_answer(self, state: LangGraphConversationState) -> LangGraphConversationState:
        query = state.get("current_query")
        faculty_profile = state.get("faculty_profile")
        if query is None or faculty_profile is None:
            return {"current_answer": None}

        answers = await self.grant_agent.answer_queries(
            queries=[query],
            faculty_profile=faculty_profile,
        )
        return {"current_answer": answers[0] if answers else None}

    async def _node_faculty_answer(self, state: LangGraphConversationState) -> LangGraphConversationState:
        query = state.get("current_query")
        faculty_profile = state.get("faculty_profile")
        if query is None or faculty_profile is None:
            return {"current_answer": None}

        answers = await self.faculty_agent.answer_queries(
            queries=[query],
            faculty_profile=faculty_profile,
        )
        return {"current_answer": answers[0] if answers else None}

    async def _node_skip_query(self, state: LangGraphConversationState) -> LangGraphConversationState:
        query = state.get("current_query")
        if query is None:
            return {"current_answer": None}
        return {
            "current_answer": QueryAnswer(
                query_id=query.query_id,
                target_agent="grant",
                intent=str(query.intent or "").strip().lower(),
                answer={"note": f"unsupported target_agent: {query.target_agent}"},
                confidence=0.0,
                evidence=[],
                followup_queries=[],
            )
        }

    async def _node_integrate_answer(self, state: LangGraphConversationState) -> LangGraphConversationState:
        answer = state.get("current_answer")
        next_round_queries = list(state.get("next_round_queries") or [])
        next_round_signatures = {self._query_signature(q) for q in next_round_queries}
        current_round_answers = list(state.get("current_round_answers") or [])
        seen_signatures: Set[str] = set(state.get("seen_signatures") or set())
        round_critical_failed = bool(state.get("round_critical_failed") or False)

        if answer is not None:
            current_round_answers.append(answer)
            if self._is_critical_failed_answer(answer):
                round_critical_failed = True

            for q in list(answer.followup_queries or []):
                sig = self._query_signature(q)
                if sig in seen_signatures:
                    continue
                if sig in next_round_signatures:
                    continue
                next_round_signatures.add(sig)
                next_round_queries.append(q)

        return {
            "current_query": None,
            "current_answer": None,
            "next_round_queries": next_round_queries,
            "current_round_answers": current_round_answers,
            "seen_signatures": seen_signatures,
            "round_critical_failed": round_critical_failed,
        }

    async def _node_finalize_round(self, state: LangGraphConversationState) -> LangGraphConversationState:
        round_index = max(1, int(state.get("round_index") or 1))
        max_rounds = max(1, int(state.get("max_rounds") or 3))

        rounds = list(state.get("rounds") or [])
        round_queries = list(state.get("round_queries") or [])
        current_round_answers = list(state.get("current_round_answers") or [])
        if round_queries or current_round_answers:
            rounds.append(
                OrchestrationRound(
                    round_index=round_index,
                    queries=round_queries,
                    answers=current_round_answers,
                )
            )

        next_round_queries = list(state.get("next_round_queries") or [])
        round_critical_failed = bool(state.get("round_critical_failed") or False)

        if round_index >= max_rounds:
            return {
                "rounds": rounds,
                "stop_reason": "max_rounds_reached",
                "route_decision": "rank",
            }

        if not next_round_queries and not round_critical_failed:
            return {
                "rounds": rounds,
                "stop_reason": "confidence_converged",
                "route_decision": "rank",
            }

        if not next_round_queries:
            return {
                "rounds": rounds,
                "stop_reason": "no_more_queries",
                "route_decision": "rank",
            }

        return {
            "rounds": rounds,
            "round_index": round_index + 1,
            "pending_queries": next_round_queries,
            "round_queries": [],
            "remaining_round_queries": [],
            "next_round_queries": [],
            "current_round_answers": [],
            "round_critical_failed": False,
            "route_decision": "round_prepare",
        }

    async def _node_rank_and_finish(self, state: LangGraphConversationState) -> LangGraphConversationState:
        faculty_profile = state.get("faculty_profile")
        if faculty_profile is None:
            raise RuntimeError("Missing faculty profile at rank stage")

        candidate_grant_ids = list(state.get("candidate_grant_ids") or [])
        grant_snapshots = await self.grant_agent.collect_snapshots_for_grants(candidate_grant_ids)

        result_top_k = max(1, int(state.get("result_top_k") or 5))
        matches = self.matcher.rank(
            faculty_profile=faculty_profile,
            grants=grant_snapshots,
            top_k=result_top_k,
        )

        return {
            "grant_snapshots": grant_snapshots,
            "matches": matches,
            "stop_reason": str(state.get("stop_reason") or "max_rounds_reached"),
        }

    @staticmethod
    def _edge_from_router(state: LangGraphConversationState) -> str:
        decision = str(state.get("route_decision") or "").strip()
        if decision in {"grant_answer", "faculty_answer", "skip_query", "finalize_round"}:
            return decision
        return "finalize_round"

    @staticmethod
    def _edge_from_finalize_round(state: LangGraphConversationState) -> str:
        decision = str(state.get("route_decision") or "").strip()
        if decision in {"round_prepare", "rank"}:
            return decision
        return "rank"

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception as e:  # pragma: no cover - dependency guard
            raise ImportError("langgraph is required. Install with: pip install langgraph") from e

        graph = StateGraph(LangGraphConversationState)

        graph.add_node("bootstrap", self._node_bootstrap)
        graph.add_node("round_prepare", self._node_round_prepare)
        graph.add_node("router", self._node_router)
        graph.add_node("grant_answer", self._node_grant_answer)
        graph.add_node("faculty_answer", self._node_faculty_answer)
        graph.add_node("skip_query", self._node_skip_query)
        graph.add_node("integrate_answer", self._node_integrate_answer)
        graph.add_node("finalize_round", self._node_finalize_round)
        graph.add_node("rank_and_finish", self._node_rank_and_finish)

        graph.set_entry_point("bootstrap")
        graph.add_edge("bootstrap", "round_prepare")
        graph.add_edge("round_prepare", "router")

        graph.add_conditional_edges(
            "router",
            self._edge_from_router,
            {
                "grant_answer": "grant_answer",
                "faculty_answer": "faculty_answer",
                "skip_query": "skip_query",
                "finalize_round": "finalize_round",
            },
        )

        graph.add_edge("grant_answer", "integrate_answer")
        graph.add_edge("faculty_answer", "integrate_answer")
        graph.add_edge("skip_query", "integrate_answer")
        graph.add_edge("integrate_answer", "router")

        graph.add_conditional_edges(
            "finalize_round",
            self._edge_from_finalize_round,
            {
                "round_prepare": "round_prepare",
                "rank": "rank_and_finish",
            },
        )
        graph.add_edge("rank_and_finish", END)

        return graph.compile()

    async def run(
        self,
        *,
        faculty_email: str,
        candidate_grant_k: int = 20,
        result_top_k: int = 5,
        max_rounds: int = 3,
        max_queries_per_round: int = 120,
    ) -> AgenticRunResult:
        final_state: LangGraphConversationState = await self.graph.ainvoke(
            {
                "faculty_email": faculty_email,
                "candidate_grant_k": max(1, int(candidate_grant_k or 20)),
                "result_top_k": max(1, int(result_top_k or 5)),
                "max_rounds": max(1, int(max_rounds or 3)),
                "max_queries_per_round": max(1, int(max_queries_per_round or 120)),
            }
        )

        faculty_profile = final_state.get("faculty_profile")
        if faculty_profile is None:
            raise RuntimeError("LangGraph run completed without faculty_profile")

        return AgenticRunResult(
            faculty_profile=faculty_profile,
            candidate_grant_ids=list(final_state.get("candidate_grant_ids") or []),
            grant_snapshots=list(final_state.get("grant_snapshots") or []),
            rounds=list(final_state.get("rounds") or []),
            matches=list(final_state.get("matches") or []),
            stop_reason=str(final_state.get("stop_reason") or "max_rounds_reached"),
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

    def draw_mermaid(self) -> str:
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception:
            return ""
