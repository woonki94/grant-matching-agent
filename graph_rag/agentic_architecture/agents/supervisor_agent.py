from __future__ import annotations

from typing import Protocol

from graph_rag.agentic_architecture.state import SupervisorInput, SupervisorOutput


class SupervisorAgent(Protocol):
    def run(self, request: SupervisorInput) -> SupervisorOutput:
        """
        Orchestrate faculty + grant agent calls and return aggregated output.
        """
