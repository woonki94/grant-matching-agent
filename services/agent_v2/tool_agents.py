"""Backward-compatible re-export for moved agent classes.

Prefer importing from services.agent_v2.agents.
"""

from services.agent_v2.agents import (
    FacultyContextAgent,
    GeneralConversationAgent,
    MatchingExecutionAgent,
    OpportunityContextAgent,
)

__all__ = [
    "FacultyContextAgent",
    "GeneralConversationAgent",
    "OpportunityContextAgent",
    "MatchingExecutionAgent",
]
