from __future__ import annotations

from graph_rag.agentic_architecture.state import AgenticState


class Planner:
    def plan(self, state: AgenticState) -> str:
        if state.routed_scenario == "faculty_one_to_one_match":
            return "run_domain_cosine_filter"
        return "unsupported"
