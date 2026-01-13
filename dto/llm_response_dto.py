# dto/llm_outputs.py
from __future__ import annotations
from typing import List, Optional
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


# ───────────────────────────────────────────────
# Matching / scoring
# ───────────────────────────────────────────────

class LLMMatchOut(BaseModel):
    llm_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=256)


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