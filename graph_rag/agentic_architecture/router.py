from __future__ import annotations

from graph_rag.agentic_architecture.state import AgenticState


class IntentRouter:
    SUPPORTED_SCENARIOS = {
        "faculty_one_to_one_match",
    }

    def route(self, state: AgenticState) -> str:
        scenario = str(state.request.scenario or "").strip().lower()
        if scenario in self.SUPPORTED_SCENARIOS:
            return scenario
        return "unsupported"
