from __future__ import annotations

from typing import List, Set, TypedDict

from tmp.agentic_arch.models import FacultyProfessionProfile, GrantSnapshot, OneToOneMatch
from tmp.agentic_arch_v2.models import OrchestrationRound, QueryAnswer, QueryItem


class LangGraphConversationState(TypedDict, total=False):
    faculty_email: str
    candidate_grant_k: int
    result_top_k: int
    max_rounds: int
    max_queries_per_round: int

    faculty_profile: FacultyProfessionProfile
    candidate_grant_ids: List[str]

    pending_queries: List[QueryItem]
    round_queries: List[QueryItem]
    remaining_round_queries: List[QueryItem]
    next_round_queries: List[QueryItem]

    current_query: QueryItem | None
    current_answer: QueryAnswer | None
    current_round_answers: List[QueryAnswer]

    rounds: List[OrchestrationRound]
    seen_signatures: Set[str]
    round_index: int
    round_critical_failed: bool

    route_decision: str
    stop_reason: str

    grant_snapshots: List[GrantSnapshot]
    matches: List[OneToOneMatch]

