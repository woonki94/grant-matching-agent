from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


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

    scenario: str
    decision: str
    has_faculty_signal: Optional[bool]
    has_group_signal: Optional[bool]
    has_grant_signal: Optional[bool]
    email_detected: Optional[str]
    emails_detected: List[str]
    grant_link_detected: Optional[str]
    grant_title_detected: Optional[str]
    result: Dict[str, Any]
