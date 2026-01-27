# dto/llm_outputs.py
from __future__ import annotations
from typing import List, Optional, Literal, Dict, Union
from pydantic import BaseModel, Field


# ───────────────────────────────────────────────
# Keyword extraction
# ───────────────────────────────────────────────

class CandidatesOut(BaseModel):
    candidates: List[str] = Field(default_factory=list)


class KeywordBucket(BaseModel):
    domain: List[str] = Field(default_factory=list)
    specialization: List[str] = Field(default_factory=list)


class KeywordsOut(BaseModel):
    research: KeywordBucket = Field(default_factory=KeywordBucket)
    application: KeywordBucket = Field(default_factory=KeywordBucket)


class KeywordItem(BaseModel):
    t: str
    w: float = Field(1.0, ge=0.0, le=1.0)

class WeightedSpecsOut(BaseModel):
    research: List[KeywordItem] = Field(default_factory=list)
    application: List[KeywordItem] = Field(default_factory=list)


# ───────────────────────────────────────────────
# Matching / scoring
# ───────────────────────────────────────────────

class MissingItem(BaseModel):
    section: Literal["application", "research"]
    idx: int

class ScoredCoveredItem(BaseModel):
    section: Literal["application", "research"]
    idx: int
    c: float = Field(..., ge=0.0, le=1.0)

class LLMMatchOut(BaseModel):
    llm_score: float = Field(..., ge=0.0, le=1.0)
    reason: Optional[str] = None
    covered: List[ScoredCoveredItem] = Field(default_factory=list)
    missing: List[MissingItem] = Field(default_factory=list)


# ───────────────────────────────────────────────
# Justification
# ───────────────────────────────────────────────
class FacultyOpportunityRec(BaseModel):
    opportunity_id: str
    title: str
    agency: Optional[str] = None

    # scores (explicit, no derived final score)
    domain_score: float = Field(ge=0.0, le=1.0)
    llm_score: float = Field(ge=0.0, le=1.0)

    why_good_match: List[str] = Field(default_factory=list)  # 2–4 bullets
    suggested_pitch: str = Field(min_length=1, max_length=500)

class FacultyRecsOut(BaseModel):
    faculty_name: str
    recommendations: List[FacultyOpportunityRec] = Field(default_factory=list)

NeedKind = Literal["research_domain", "method", "application", "compliance"]

# ───────────────────────────────────────────────
# For group matcher
# ───────────────────────────────────────────────
class Need(BaseModel):
    need_id: str
    label: str
    description: str
    weight: int = Field(ge=1, le=5, default=3)
    kind: NeedKind = "research_domain"
    must_have: bool = False

class NeedsOut(BaseModel):
    opportunity_id: str
    opportunity_title: str
    scope_confidence: float = Field(ge=0.0, le=1.0)
    suggested_team_size: int = Field(ge=1, le=8, default=3)
    needs: List[Need] = Field(default_factory=list)

class TeamMember(BaseModel):
    faculty_id: int
    name: str | None = None
    email: str | None = None
    role: str | None = None

class TeamMatchOut(BaseModel):
    opportunity_id: str
    selected: List[TeamMember] = Field(default_factory=list)
    covered_need_ids: List[str] = Field(default_factory=list)
    missing_need_ids: List[str] = Field(default_factory=list)

#
class PairPenalty(BaseModel):
    f: int = Field(..., description="Faculty id")
    g: int = Field(..., description="Faculty id")
    p: float = Field(..., ge=0, description="Penalty magnitude in score units")
    why: Optional[str] = Field(None, description="Short reason")

class PairPenaltiesOut(BaseModel):
    pair_penalties: List[PairPenalty] = Field(default_factory=list)