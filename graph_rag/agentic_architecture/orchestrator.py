from __future__ import annotations

from typing import Any, Dict

from graph_rag.agentic_architecture.planner import Planner
from graph_rag.agentic_architecture.router import IntentRouter
from graph_rag.agentic_architecture.state import AgenticRequest, AgenticState
from graph_rag.agentic_architecture.agents.filter_agents import PreFilterAgent


class GraphRagOrchestrator:
    def __init__(
        self,
        *,
        router: IntentRouter | None = None,
        planner: Planner | None = None,
        prefilter_agent: PreFilterAgent | None = None,
    ):
        self.router = router or IntentRouter()
        self.planner = planner or Planner()
        self.prefilter_agent = prefilter_agent or PreFilterAgent()

    def run(self, request: AgenticRequest) -> Dict[str, Any]:
        state = AgenticState(request=request)

        state.routed_scenario = self.router.route(state)
        if state.routed_scenario == "unsupported":
            state.errors.append(f"Unsupported scenario: {request.scenario}")
            return self._finalize(state)

        state.planner_action = self.planner.plan(state)
        if state.planner_action != "run_domain_cosine_filter":
            state.errors.append(f"Unsupported planner action: {state.planner_action}")
            return self._finalize(state)

        state.result = self.prefilter_agent.domain_cos(
            email=request.email,
            threshold=request.threshold,
            top_k=request.top_k,
            include_closed=request.include_closed,
        )
        return self._finalize(state)

    @staticmethod
    def _finalize(state: AgenticState) -> Dict[str, Any]:
        return {
            "scenario": state.routed_scenario,
            "action": state.planner_action,
            "result": state.result,
            "errors": state.errors,
        }
