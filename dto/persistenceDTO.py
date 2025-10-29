from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Iterable
import re, html

# --- HTML stripper for summary_description ---
_TAG_RE = re.compile(r"<[^>]+>")

def strip_html(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = _TAG_RE.sub("", s)
    t = html.unescape(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or None


# ========== Persistence DTOs (DB-facing, minimal & normalized) ==========
@dataclass
class SummaryPersistenceDTO:
    additional_info_url: Optional[str] = None

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

    summary_description: Optional[str] = None  # will be stored text-only

    @staticmethod
    def from_portal_summary(s) -> "SummaryPersistenceDTO":
        if s is None:
            return SummaryPersistenceDTO()
        return SummaryPersistenceDTO(
            additional_info_url=getattr(s, "additional_info_url", None),

            agency_email_address=getattr(s, "agency_email_address", None),
            applicant_types=list(getattr(s, "applicant_types", []) or None) if getattr(s, "applicant_types", None) else None,

            archive_date=getattr(s, "archive_date", None),
            award_ceiling=getattr(s, "award_ceiling", None),
            award_floor=getattr(s, "award_floor", None),
            close_date=getattr(s, "close_date", None),
            created_at=getattr(s, "created_at", None),
            estimated_total_program_funding=getattr(s, "estimated_total_program_funding", None),
            expected_number_of_awards=getattr(s, "expected_number_of_awards", None),
            forecasted_award_date=getattr(s, "forecasted_award_date", None),
            forecasted_close_date=getattr(s, "forecasted_close_date", None),
            forecasted_post_date=getattr(s, "forecasted_post_date", None),
            forecasted_project_start_date=getattr(s, "forecasted_project_start_date", None),

            funding_categories=list(getattr(s, "funding_categories", []) or None) if getattr(s, "funding_categories", None) else None,
            funding_instruments=list(getattr(s, "funding_instruments", []) or None) if getattr(s, "funding_instruments", None) else None,

            is_cost_sharing=getattr(s, "is_cost_sharing", None),
            post_date=getattr(s, "post_date", None),

            summary_description=strip_html(getattr(s, "summary_description", None)),
        )


@dataclass
class OpportunityPersistenceDTO:
    # top-level (PK + fields)
    opportunity_id: str
    agency_name: Optional[str] = None
    category: Optional[str] = None
    opportunity_status: Optional[str] = None
    opportunity_title: Optional[str] = None

    # nested summary (flattened for DB)
    additional_info_url: Optional[str] = None
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

    @staticmethod
    def from_portal_row(row) -> "OpportunityPersistenceDTO":
        """
        row: PortalOpportunityDTO (API response row)
        """
        s = getattr(row, "summary", None)
        s_p = SummaryPersistenceDTO.from_portal_summary(s)
        return OpportunityPersistenceDTO(
            opportunity_id=getattr(row, "opportunity_id", None),
            agency_name=getattr(row, "agency_name", None),
            category=getattr(row, "category", None),
            opportunity_status=getattr(row, "opportunity_status", None),
            opportunity_title=getattr(row, "opportunity_title", None),

            additional_info_url=s_p.additional_info_url,
            agency_email_address=s_p.agency_email_address,
            applicant_types=s_p.applicant_types,
            archive_date=s_p.archive_date,
            award_ceiling=s_p.award_ceiling,
            award_floor=s_p.award_floor,
            close_date=s_p.close_date,
            created_at=s_p.created_at,
            estimated_total_program_funding=s_p.estimated_total_program_funding,
            expected_number_of_awards=s_p.expected_number_of_awards,
            forecasted_award_date=s_p.forecasted_award_date,
            forecasted_close_date=s_p.forecasted_close_date,
            forecasted_post_date=s_p.forecasted_post_date,
            forecasted_project_start_date=s_p.forecasted_project_start_date,
            funding_categories=s_p.funding_categories,
            funding_instruments=s_p.funding_instruments,
            is_cost_sharing=s_p.is_cost_sharing,
            post_date=s_p.post_date,
            summary_description=s_p.summary_description,
        )


@dataclass
class AttachmentPersistenceDTO:
    # pk is autogen in DB; not needed here
    opportunity_id: str          # FK
    file_name: str
    download_path: str

    @staticmethod
    def from_attachment_row(opportunity_id: str, a) -> "AttachmentPersistenceDTO":
        """
        a: AttachmentDTO or dict with file_name, download_path
        """
        if isinstance(a, dict):
            fn = a.get("file_name")
            dp = a.get("download_path")
        else:
            fn = getattr(a, "file_name", None)
            dp = getattr(a, "download_path", None)
        return AttachmentPersistenceDTO(opportunity_id=opportunity_id, file_name=fn, download_path=dp)


# ========== Convenience mappers for whole search results ==========
def build_opportunity_persistence_list(search_dto) -> List[OpportunityPersistenceDTO]:
    """
    search_dto: PortalSearchResponseDTO
    """
    out: List[OpportunityPersistenceDTO] = []
    for row in getattr(search_dto, "data", []) or []:
        if getattr(row, "opportunity_id", None):
            out.append(OpportunityPersistenceDTO.from_portal_row(row))
    return out


def build_attachment_persistence_list(search_dto) -> List[AttachmentPersistenceDTO]:
    """
    Assumes you've already enriched each row with row.attachments (list)
    """
    out: List[AttachmentPersistenceDTO] = []
    for row in getattr(search_dto, "data", []) or []:
        oid = getattr(row, "opportunity_id", None)
        if not oid:
            continue
        for a in getattr(row, "attachments", []) or []:
            ap = AttachmentPersistenceDTO.from_attachment_row(oid, a)
            if ap.file_name and ap.download_path:
                out.append(ap)
    return out