from __future__ import annotations

from typing import Protocol

from graph_rag.agentic_architecture.state import FacultyContext


class FacultyAgent(Protocol):
    def resolve_faculty(self, *, faculty_email: str) -> FacultyContext:
        """
        Resolve the faculty identity and return base context.
        """

    def enrich_for_matching(self, *, faculty: FacultyContext, query: str = "") -> FacultyContext:
        """
        Attach additional matching evidence to faculty context.
        """
