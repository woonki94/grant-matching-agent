from __future__ import annotations

from typing import List, Optional, Protocol

from graph_rag.agentic_architecture.state import (
    FacultyContext,
    GrantCandidate,
    GrantContext,
    MatchAssessment,
)


class GrantAgent(Protocol):
    def prefilter_candidates(
        self,
        *,
        faculty: FacultyContext,
        threshold: float,
        top_k: int,
        include_closed: bool,
        agency_filter: Optional[str],
    ) -> List[GrantCandidate]:
        """
        Retrieve grant candidates from filter tools only.
        """

    def load_grant_context(self, *, opportunity_id: str) -> GrantContext:
        """
        Load one grant's detailed context for scoring.
        """

    def assess_match(self, *, faculty: FacultyContext, grant: GrantContext) -> MatchAssessment:
        """
        Produce final match assessment for one faculty/grant pair.
        """
