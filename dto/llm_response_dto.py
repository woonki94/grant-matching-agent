# dto/llm_outputs.py
from __future__ import annotations
from typing import List, Optional, Literal, Dict, Union
from pydantic import BaseModel, Field, field_validator
import json

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
    
    @field_validator("research", "application", mode="before")
    @classmethod
    def _coerce_bucket(cls, v):
        if not isinstance(v, str):
            return v

        s = v.strip()

        # Remove ```json fences if present
        if s.startswith("```"):
            s = s.strip("`").strip()
            if s.lower().startswith("json"):
                s = s[4:].strip()

        # Find the first '{' and parse the first JSON object only
        i = s.find("{")
        if i == -1:
            return v

        s2 = s[i:]
        dec = json.JSONDecoder()
        try:
            obj, end = dec.raw_decode(s2)  # parses first JSON object, ignores trailing junk
            return obj
        except Exception:
            return v

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

# ───────────────────────────────────────────────
# Group justification
# ───────────────────────────────────────────────
class MemberRoleOut(BaseModel):
    faculty_id: int
    role: str = Field(..., description="Short label like 'AI/ML lead', 'Education/Outreach lead', etc.")
    why: str = Field(..., description="1-2 sentences describing their unique contribution")

class CoverageOut(BaseModel):
    strong: List[str] = Field(default_factory=list)
    partial: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)

class MemberStrengthOut(BaseModel):
    faculty_id: int
    bullets: List[str] = Field(default_factory=list)

class TeamRoleOut(BaseModel):
    member_roles: List[MemberRoleOut] = Field(default_factory=list)

class WhyWorkingOut(BaseModel):
    summary: str = ""
    member_strengths: List[MemberStrengthOut] = Field(default_factory=list)
    strong: List[str] = Field(default_factory=list)
    partial: List[str] = Field(default_factory=list)

class WhyNotWorkingOut(BaseModel):
    why_not_working: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)

class RecommendationOut(BaseModel):
    match_quality: Literal["good", "moderate", "bad"] = "moderate"
    recommendation: str = ""


class GroupJustificationOut(BaseModel):
    match_quality: Literal["good", "moderate", "bad"] = "moderate"
    one_paragraph: str
    member_roles: List[MemberRoleOut] = Field(default_factory=list)
    coverage: CoverageOut = Field(default_factory=CoverageOut)
    member_strengths: List[MemberStrengthOut] = Field(default_factory=list)
    why_not_working: List[str] = Field(default_factory=list)
    recommendation: str = ""

class PlannerRequest(BaseModel):
    opp_fields: List[str] = Field(default_factory=list, description="Additional opportunity fields to fetch if available.")
    faculty_fields: List[str] = Field(default_factory=list, description="Additional faculty fields to fetch if available.")
    ask_for_more_faculty: bool = Field(default=False, description="Whether writer needs richer faculty context than keywords.")
    ask_for_more_opp: bool = Field(default=False, description="Whether writer needs richer opportunity context than summary/keywords.")
    focus_points: List[str] = Field(default_factory=list, description="Key angles to emphasize in justification.")


class CriticVerdict(BaseModel):
    ok: bool = Field(..., description="True if justification is acceptable and grounded.")
    issues: List[str] = Field(default_factory=list, description="Problems found: vagueness, missing grounding, etc.")
    request_more: PlannerRequest = Field(default_factory=PlannerRequest, description="If not ok, what more to fetch.")


# ───────────────────────────────────────────────
# Candidate team selection
# ───────────────────────────────────────────────
class TeamCandidateSelectionOut(BaseModel):
    selected_candidate_indices: List[int] = Field(default_factory=list)
    reason: Optional[str] = None
