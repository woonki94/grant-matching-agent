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

class OpportunityCategoryOut(BaseModel):
    broad_category: Literal["basic_research", "applied_research", "educational", "unclear"] = "unclear"
    specific_categories: List[str] = Field(default_factory=list)


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
# One-to-One Match Justification
# ───────────────────────────────────────────────
FitLabel = Literal["mismatch", "bad", "good", "great", "fantastic"]


class WhyMatchOut(BaseModel):
    summary: str = ""
    alignment_points: List[str] = Field(default_factory=list)
    risk_gaps: List[str] = Field(default_factory=list)


class FacultyOpportunityRec(BaseModel):
    opportunity_id: str
    title: str
    agency: Optional[str] = None
    grant_explanation: str = ""

    # scores (explicit, no derived final score)
    domain_score: float = Field(ge=0.0, le=1.0)
    llm_score: float = Field(ge=0.0, le=1.0)
    fit_label: FitLabel = "mismatch"

    why_match: WhyMatchOut = Field(default_factory=WhyMatchOut)
    suggested_pitch: str = Field(min_length=1, max_length=500)

    @field_validator("fit_label", mode="before")
    @classmethod
    def _normalize_fit_label(cls, v):
        label = str(v or "").strip().lower()
        if label == "greate":
            return "great"
        if label in {"mismatch", "bad", "good", "great", "fantastic"}:
            return label
        return "mismatch"

class FacultyRecsOut(BaseModel):
    faculty_name: str
    grant_explanation: str = ""
    recommendations: List[FacultyOpportunityRec] = Field(default_factory=list)


class GrantExplanationOut(BaseModel):
    grant_explanation: str = ""

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

class GrantBriefOut(BaseModel):
    grant_title: str = ""
    grant_link: str = ""
    grant_quick_explanation: str = ""
    priority_themes: List[str] = Field(default_factory=list)

class WhyWorkingOut(BaseModel):
    summary: str = ""
    member_strengths: List[MemberStrengthOut] = Field(default_factory=list)
    strong: List[str] = Field(default_factory=list)
    partial: List[str] = Field(default_factory=list)

class WhyNotWorkingOut(BaseModel):
    why_not_working: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)

class RecommendationOut(BaseModel):
    recommendation: str = ""

class GroupJustificationOut(BaseModel):
    one_paragraph: str
    member_roles: List[MemberRoleOut] = Field(default_factory=list)
    coverage: CoverageOut = Field(default_factory=CoverageOut)
    member_strengths: List[MemberStrengthOut] = Field(default_factory=list)
    why_not_working: List[str] = Field(default_factory=list)
    recommendation: str = ""

# ───────────────────────────────────────────────
# Candidate team selection (Additional layer for selecting group)
# ───────────────────────────────────────────────
class TeamCandidateSelectionOut(BaseModel):
    selected_candidate_indices: List[int] = Field(default_factory=list)
    reason: Optional[str] = None
