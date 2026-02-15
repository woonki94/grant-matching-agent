from __future__ import annotations

from typing import Any, Dict, List, Optional

from db.models import Faculty
from dto.faculty_dto import (
    FacultyDTO,
    FacultyAdditionalInfoDTO,
    FacultyPublicationDTO,
)

# ─────────────────────────────
# 1) Profile/base payload → FacultyAggregateDTO (NO publications IO)
# ─────────────────────────────

def map_faculty_profile_to_dto(profile: Dict[str, Any]) -> FacultyDTO:
    url = profile.get("source_url") or profile.get("url")
    if not url:
        raise ValueError("Faculty profile missing source_url/url")

    merged = dict(profile)
    merged["source_url"] = url
    merged["publications"] = []
    merged["keyword"] = None

    # additional_info is stored separately from the opportunity row
    additional_info_urls = merged.get("additional_info") or []
    merged["additional_info"] = [
        FacultyAdditionalInfoDTO(additional_info_url=u.strip())
        for u in additional_info_urls
        if isinstance(u, str) and u.strip()
    ]

    return FacultyDTO.model_validate(merged)

