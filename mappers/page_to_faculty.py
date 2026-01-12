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


# ─────────────────────────────
# 2) Publications endpoint JSON → List[FacultyPublicationDTO]
# ─────────────────────────────

def map_publications_response(resp: Any) -> List[FacultyPublicationDTO]:
    """
    Maps a nested publications response into FacultyPublicationDTO list.

    Tolerates multiple common shapes:
    - {"data": [..]}
    - {"publications": [..]}
    - directly a list of dicts
    """
    if isinstance(resp, dict):
        items = resp.get("data") or resp.get("publications") or resp.get("results") or []
    elif isinstance(resp, list):
        items = resp
    else:
        items = []

    out: List[FacultyPublicationDTO] = []

    for p in items:
        if not isinstance(p, dict):
            continue

        title = (p.get("title") or "").strip()
        if not title:
            continue

        out.append(
            FacultyPublicationDTO(
                openalex_work_id=p.get("openalex_work_id"),
                scholar_author_id=p.get("scholar_author_id"),
                title=title,
                abstract=p.get("abstract"),
                year=p.get("year"),
            )
        )

    return out


# ─────────────────────────────
# 3) Helper: apply publications (pure-ish; mutates DTO)
# ─────────────────────────────

def with_publications(
    faculty: FacultyDTO,
    publications: List[FacultyPublicationDTO],
) -> FacultyDTO:
    """
    Attach publications to an existing FacultyAggregateDTO.
    """
    faculty.publications = publications
    return faculty