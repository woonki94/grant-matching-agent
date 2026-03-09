from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from tmp.agentic_arch.models import FacultyProfessionProfile, GrantSnapshot, OneToOneMatch

TargetAgent = Literal["grant", "faculty"]


@dataclass(frozen=True)
class QueryItem:
    query_id: str
    target_agent: TargetAgent
    intent: str
    question: str
    expected_fields: List[str] = field(default_factory=list)
    priority: float = 0.5
    confidence_threshold: float = 0.7
    grant_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryAnswer:
    query_id: str
    target_agent: TargetAgent
    intent: str
    answer: Dict[str, Any]
    confidence: float
    evidence: List[str] = field(default_factory=list)
    followup_queries: List[QueryItem] = field(default_factory=list)


@dataclass(frozen=True)
class OrchestrationRound:
    round_index: int
    queries: List[QueryItem]
    answers: List[QueryAnswer]


@dataclass(frozen=True)
class AgenticRunResult:
    faculty_profile: FacultyProfessionProfile
    candidate_grant_ids: List[str]
    grant_snapshots: List[GrantSnapshot]
    rounds: List[OrchestrationRound]
    matches: List[OneToOneMatch]
    stop_reason: str


class PlannedQueryOut(BaseModel):
    target_agent: TargetAgent = Field(...)
    intent: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    expected_fields: List[str] = Field(default_factory=list)
    priority: float = Field(0.5, ge=0.0, le=1.0)
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)


class QueryPlanOut(BaseModel):
    queries: List[PlannedQueryOut] = Field(default_factory=list)
