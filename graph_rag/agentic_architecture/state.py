from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgenticRequest:
    scenario: str
    email: str
    threshold: float | None = None
    top_k: int = 100
    include_closed: bool = False


@dataclass
class AgenticState:
    request: AgenticRequest
    routed_scenario: str = ""
    planner_action: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


Decision = Literal[
    "apply_now",
    "apply_with_internal_collab",
    "apply_with_external_collab",
    "watchlist",
    "skip_ineligible",
    "skip_low_relevance",
    "skip_capacity",
    "needs_human_review",
]


@dataclass
class SupervisorInput:
    faculty_email: str
    query: str = ""
    top_k: int = 100
    include_closed: bool = False
    agency_filter: Optional[str] = None
    threshold: float = 0.2


@dataclass
class GrantCandidate:
    opportunity_id: str
    domain_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FacultyContext:
    faculty_id: Optional[int] = None
    basic: Dict[str, Any] = field(default_factory=dict)
    keywords: Dict[str, Any] = field(default_factory=dict)
    publications: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GrantContext:
    opportunity_id: str
    basic: Dict[str, Any] = field(default_factory=dict)
    keywords: Dict[str, Any] = field(default_factory=dict)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchAssessment:
    opportunity_id: str
    decision: Decision
    score: float
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SupervisorOutput:
    faculty_email: str
    assessments: List[MatchAssessment]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
