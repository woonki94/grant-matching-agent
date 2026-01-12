from __future__ import annotations

from typing import Optional, List, Any, Dict
from pydantic import BaseModel, ConfigDict, Field


class OpportunityAdditionalInfoDTO(BaseModel):
    model_config = ConfigDict(extra="ignore")
    additional_info_url: str
    content_path: Optional[str] = None
    detected_type: Optional[str] = None
    content_char_count: Optional[int] = None
    extracted_at: Optional[str] = None
    extract_status: str = "pending"
    extract_error: Optional[str] = None


class OpportunityKeywordDTO(BaseModel):
    model_config = ConfigDict(extra="ignore")
    keywords: Dict[str, Any]
    raw_json: Optional[Dict[str, Any]] = None
    source: str = "gpt-5"


class OpportunityAttachmentDTO(BaseModel):
    model_config = ConfigDict(extra="ignore")
    file_name: str
    file_download_path: str = Field(alias="download_path")
    content_path: Optional[str] = None
    detected_type: Optional[str] = None
    content_char_count: Optional[int] = None
    extracted_at: Optional[str] = None
    extract_status: str = "pending"
    extract_error: Optional[str] = None


class OpportunityDTO(BaseModel):
    model_config = ConfigDict(extra="ignore")

    opportunity_id: str

    agency_name: Optional[str] = None
    category: Optional[str] = None
    opportunity_status: Optional[str] = None
    opportunity_title: Optional[str] = None

    agency_email_address: Optional[str] = None
    applicant_types: Optional[List[str]] = None

    archive_date: Optional[str] = None
    award_ceiling: Optional[float] = None
    award_floor: Optional[float] = None
    close_date: Optional[str] = None
    created_at: Optional[str] = None
    estimated_total_program_funding: Optional[float] = None
    expected_number_of_awards: Optional[int] = None

    forecasted_award_date: Optional[str] = None
    forecasted_close_date: Optional[str] = None
    forecasted_post_date: Optional[str] = None
    forecasted_project_start_date: Optional[str] = None

    funding_categories: Optional[List[str]] = None
    funding_instruments: Optional[List[str]] = None

    is_cost_sharing: Optional[bool] = None
    post_date: Optional[str] = None
    summary_description: Optional[str] = None

    additional_info: List[OpportunityAdditionalInfoDTO] = Field(default_factory=list)
    keyword: Optional[OpportunityKeywordDTO] = None
    attachments: List[OpportunityAttachmentDTO] = Field(default_factory=list)