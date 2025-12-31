# dto_portal_search.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any


# ─────────────────────────────
# Leaf DTOs
# ─────────────────────────────

@dataclass
class AssistanceListingDTO:
    assistance_listing_number: Optional[str] = None
    program_title: Optional[str] = None

    @staticmethod
    def from_dict(d: dict) -> "AssistanceListingDTO":
        return AssistanceListingDTO(
            assistance_listing_number=d.get("assistance_listing_number"),
            program_title=d.get("program_title"),
        )

@dataclass
class SummaryDTO:
    additional_info_url: Optional[str] = None
    additional_info_url_description: Optional[str] = None
    agency_contact_description: Optional[str] = None
    agency_email_address: Optional[str] = None
    agency_email_address_description: Optional[str] = None
    applicant_eligibility_description: Optional[str] = None
    applicant_types: List[str] = field(default_factory=list)
    archive_date: Optional[str] = None                # e.g., "2026-03-19"
    award_ceiling: Optional[float] = None
    award_floor: Optional[float] = None
    close_date: Optional[str] = None
    close_date_description: Optional[str] = None
    created_at: Optional[str] = None                  # ISO8601
    estimated_total_program_funding: Optional[float] = None
    expected_number_of_awards: Optional[int] = None
    fiscal_year: Optional[int] = None
    forecasted_award_date: Optional[str] = None
    forecasted_close_date: Optional[str] = None
    forecasted_close_date_description: Optional[str] = None
    forecasted_post_date: Optional[str] = None
    forecasted_project_start_date: Optional[str] = None
    funding_categories: List[str] = field(default_factory=list)
    funding_category_description: Optional[str] = None
    funding_instruments: List[str] = field(default_factory=list)
    is_cost_sharing: Optional[bool] = None
    is_forecast: Optional[bool] = None
    post_date: Optional[str] = None                   # "2025-10-23"
    summary_description: Optional[str] = None         # HTML allowed
    updated_at: Optional[str] = None                  # ISO8601
    version_number: Optional[int] = None

    @staticmethod
    def from_dict(d: dict) -> "SummaryDTO":
        return SummaryDTO(
            additional_info_url=d.get("additional_info_url"),
            additional_info_url_description=d.get("additional_info_url_description"),
            agency_contact_description=d.get("agency_contact_description"),
            agency_email_address=d.get("agency_email_address"),
            agency_email_address_description=d.get("agency_email_address_description"),
            applicant_eligibility_description=d.get("applicant_eligibility_description"),
            applicant_types=list(d.get("applicant_types") or []),
            archive_date=d.get("archive_date"),
            award_ceiling=d.get("award_ceiling"),
            award_floor=d.get("award_floor"),
            close_date=d.get("close_date"),
            close_date_description=d.get("close_date_description"),
            created_at=d.get("created_at"),
            estimated_total_program_funding=d.get("estimated_total_program_funding"),
            expected_number_of_awards=d.get("expected_number_of_awards"),
            fiscal_year=d.get("fiscal_year"),
            forecasted_award_date=d.get("forecasted_award_date"),
            forecasted_close_date=d.get("forecasted_close_date"),
            forecasted_close_date_description=d.get("forecasted_close_date_description"),
            forecasted_post_date=d.get("forecasted_post_date"),
            forecasted_project_start_date=d.get("forecasted_project_start_date"),
            funding_categories=list(d.get("funding_categories") or []),
            funding_category_description=d.get("funding_category_description"),
            funding_instruments=list(d.get("funding_instruments") or []),
            is_cost_sharing=d.get("is_cost_sharing"),
            is_forecast=d.get("is_forecast"),
            post_date=d.get("post_date"),
            summary_description=d.get("summary_description"),
            updated_at=d.get("updated_at"),
            version_number=d.get("version_number"),
        )


# ─────────────────────────────
# Attachment row
# ─────────────────────────────
@dataclass
class AttachmentDTO:
    created_at: Optional[str]
    updated_at: Optional[str]
    download_path: Optional[str]
    file_description: Optional[str]
    file_name: Optional[str]
    file_size_bytes: Optional[int]
    mime_type: Optional[str]

    @staticmethod
    def from_dict(d: dict) -> "AttachmentDTO":
        return AttachmentDTO(
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
            download_path=d.get("download_path"),
            file_description=d.get("file_description"),
            file_name=d.get("file_name"),
            file_size_bytes=d.get("file_size_bytes"),
            mime_type=d.get("mime_type"),
        )

# ─────────────────────────────
# Opportunity row
# ─────────────────────────────

@dataclass
class PortalOpportunityDTO:
    agency: Optional[str] = None                         # "HHS-CDC-GHC"
    agency_code: Optional[str] = None                    # "HHS-CDC-GHC"
    agency_name: Optional[str] = None                    # "Centers for Disease Control-GHC"
    category: Optional[str] = None                       # "discretionary"
    category_explanation: Optional[str] = None
    legacy_opportunity_id: Optional[int] = None          # 360773
    opportunity_assistance_listings: List[AssistanceListingDTO] = field(default_factory=list)
    opportunity_id: Optional[str] = None                 # UUID "9f2de5ef-..."
    opportunity_number: Optional[str] = None             # "CDC-RFA-JG-26-0185"
    opportunity_status: Optional[str] = None             # "forecasted" | "posted" | ...
    opportunity_title: Optional[str] = None
    summary: Optional[SummaryDTO] = None
    top_level_agency_name: Optional[str] = None          # "Department of Health and Human Services"
    attachments: List[AttachmentDTO] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "PortalOpportunityDTO":
        return PortalOpportunityDTO(
            agency=d.get("agency"),
            agency_code=d.get("agency_code"),
            agency_name=d.get("agency_name"),
            category=d.get("category"),
            category_explanation=d.get("category_explanation"),
            legacy_opportunity_id=d.get("legacy_opportunity_id"),
            opportunity_assistance_listings=[
                AssistanceListingDTO.from_dict(x) for x in (d.get("opportunity_assistance_listings") or [])
            ],
            opportunity_id=d.get("opportunity_id"),
            opportunity_number=d.get("opportunity_number"),
            opportunity_status=d.get("opportunity_status"),
            opportunity_title=d.get("opportunity_title"),
            summary=SummaryDTO.from_dict(d.get("summary") or {}),
            top_level_agency_name=d.get("top_level_agency_name"),
        )



@dataclass
class PortalSearchResponseDTO:
    data: List[PortalOpportunityDTO] = field(default_factory=list)
    # facet_counts intentionally omitted
    message: Optional[str] = None
    status_code: Optional[int] = None


    @staticmethod
    def from_dict(d: dict) -> "PortalSearchResponseDTO":
        return PortalSearchResponseDTO(
            data=[PortalOpportunityDTO.from_dict(x) for x in (d.get("data") or [])],
            message=d.get("message"),
            status_code=d.get("status_code"),
        )