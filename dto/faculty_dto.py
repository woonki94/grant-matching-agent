from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field


class FacultyAdditionalInfoDTO(BaseModel):
    model_config = ConfigDict(extra="ignore")
    additional_info_url: str
    content_path: Optional[str] = None
    detected_type: Optional[str] = None
    content_char_count: Optional[int] = None
    extracted_at: Optional[str] = None
    extract_status: str = "pending"
    extract_error: Optional[str] = None


class FacultyPublicationDTO(BaseModel):
    model_config = ConfigDict(extra="ignore")
    openalex_work_id: Optional[str] = None
    scholar_author_id: Optional[str] = None
    title: str = ""
    abstract: Optional[str] = None
    year: Optional[int] = None


class FacultyKeywordDTO(BaseModel):
    model_config = ConfigDict(extra="ignore")
    keywords: Dict[str, Any]
    raw_json: Optional[Dict[str, Any]] = None
    source: str = "gpt-5"


class FacultyDTO(BaseModel):
    model_config = ConfigDict(extra="ignore")
    faculty_id: Optional[int] = None

    source_url: str = ""

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    position: Optional[str] = None

    organization: Optional[str] = None
    organizations: Optional[List[str]] = None

    address: Optional[str] = None
    biography: Optional[str] = None

    degrees: Optional[List[str]] = None
    expertise: Optional[List[str]] = None

    additional_info: List[FacultyAdditionalInfoDTO] = Field(default_factory=list) # 1–many
    publications: List[FacultyPublicationDTO] = Field(default_factory=list)  # 1–many
    keyword: Optional[FacultyKeywordDTO] = None                  # 1–1
