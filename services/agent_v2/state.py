from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

# email (lowercase) → raw PDF bytes for that faculty member's CV
CvPdfMap = Dict[str, bytes]


@dataclass
class GrantMatchRequest:
    user_input: str
    email: Optional[str] = None
    emails: List[str] = field(default_factory=list)
    faculty_in_db: Optional[bool] = None
    email_in_db: Optional[bool] = None
    grant_link: Optional[str] = None
    grant_title: Optional[str] = None
    grant_identifier_type: Optional[str] = None
    grant_in_db: Optional[bool] = None
    grant_link_valid: Optional[bool] = None
    grant_title_confirmed: Optional[bool] = None
    desired_broad_category: Optional[str | List[str]] = None
    topic_query: Optional[str] = None
    requested_team_size: Optional[int] = None
    requested_top_k_grants: Optional[int] = None
    # email → PDF bytes map; each entry triggers publication ingestion for that faculty.
    # Supports 0, 1, or N CVs in a single request.
    cv_pdf_map: Optional[CvPdfMap] = None


class GrantMatchWorkflowState(TypedDict, total=False):
    user_input: str
    email: Optional[str]
    emails: List[str]
    faculty_in_db: Optional[bool]
    email_in_db: Optional[bool]
    grant_link: Optional[str]
    grant_title: Optional[str]
    grant_identifier_type: Optional[str]
    grant_in_db: Optional[bool]
    grant_link_valid: Optional[bool]
    grant_title_confirmed: Optional[bool]
    desired_broad_category: Optional[str | List[str]]
    topic_query: Optional[str]
    requested_team_size: Optional[int]
    requested_top_k_grants: Optional[int]
    # email → PDF bytes map; passed through from GrantMatchRequest.
    cv_pdf_map: Optional[CvPdfMap]

    scenario: str
    decision: str
    has_faculty_signal: Optional[bool]
    has_group_signal: Optional[bool]
    has_grant_signal: Optional[bool]
    email_detected: Optional[str]
    emails_detected: List[str]
    grant_link_detected: Optional[str]
    grant_title_detected: Optional[str]
    desired_broad_category_detected: Optional[str | List[str]]
    topic_query_detected: Optional[str]
    requested_team_size_detected: Optional[int]
    requested_top_k_grants_detected: Optional[int]
    faculty_ids: List[int]
    missing_emails: List[str]
    opportunity_id: Optional[str]
    opportunity_title: Optional[str]
    result: Dict[str, Any]
