from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FacultyBasicInfo:
    email: str
    faculty_name: Optional[str]
    position: Optional[str]
    organizations: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class FacultyPublication:
    title: str
    abstract: Optional[str]
    year: Optional[int]


@dataclass(frozen=True)
class FacultyProfessionProfile:
    email: str
    basic_info: FacultyBasicInfo
    profession_focus: List[str]
    keywords: List[str]
    evidence: Dict[str, List[str]]


@dataclass(frozen=True)
class GrantMetadata:
    grant_id: str
    grant_name: Optional[str]
    agency_name: Optional[str]
    close_date: Optional[str]


@dataclass(frozen=True)
class GrantRequirement:
    domains: List[str]
    specializations: List[str]
    eligibility: List[str]
    deliverables: List[str]


@dataclass(frozen=True)
class GrantSnapshot:
    metadata: GrantMetadata
    requirement: GrantRequirement


@dataclass(frozen=True)
class OneToOneMatch:
    faculty_email: str
    grant_id: str
    score: float
    reason: str
    matched_professions: List[str]
    grant_name: Optional[str]
    agency_name: Optional[str]
    close_date: Optional[str]
