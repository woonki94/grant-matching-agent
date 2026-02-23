from __future__ import annotations

from typing import Any, Dict, List, Optional
import html
import re

from dto.opportunity_dto import (
    OpportunityDTO,
    OpportunityAdditionalInfoDTO,
    OpportunityAttachmentDTO,
)

_TAG_RE = re.compile(r"<[^>]+>")

def strip_html(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = _TAG_RE.sub("", s)
    t = html.unescape(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or None


# ─────────────────────────────
# 1) Search-row → OpportunityDTO (NO attachments IO)
# ─────────────────────────────

def map_portal_search_row_to_opportunity(row: Dict[str, Any]) -> Optional[OpportunityDTO]:
    """
    Map ONE search result row into OpportunityDTO.
    (Attachments come from a nested call).
    """
    oid = row.get("opportunity_id")
    if not oid:
        return None

    summary = row.get("summary") or {}

    # Auto-bind matching fields by merging dicts.
    # summary overwrites row if a key exists in both (fine here).
    merged: Dict[str, Any] = {**row, **summary}

    # Your one special transform (HTML → text)
    merged["summary_description"] = strip_html(summary.get("summary_description"))

    # Relationships you handle separately
    merged["attachments"] = []  # nested call fills later
    merged["keyword"] = None

    # additional_info is stored separately from the opportunity row
    merged["additional_info"] = []
    additional_info_url = summary.get("additional_info_url")
    if additional_info_url:
        merged["additional_info"] = [
            OpportunityAdditionalInfoDTO(additional_info_url=additional_info_url)
        ]

    # Let Pydantic do the rest (types, defaults, ignoring extra fields)
    return OpportunityDTO.model_validate(merged)


def map_portal_search_response(resp: Dict[str, Any]) -> List[OpportunityDTO]:
    return [
        dto
        for dto in (
            map_portal_search_row_to_opportunity(row)
            for row in (resp.get("data") or [])
        )
        if dto is not None
    ]


# ─────────────────────────────
# 2) Attachments endpoint JSON → List[AttachmentDTO]
# ─────────────────────────────
def map_portal_attachments_response(detail_data: dict) -> List[OpportunityAttachmentDTO]:
    """
       Map the nested attachments response JSON into AttachmentDTOs.
    """
    items = ((detail_data.get("data") or {}).get("attachments") or [])
    out: List[OpportunityAttachmentDTO] = []
    for a in items:
        try:
            out.append(OpportunityAttachmentDTO.model_validate(a))  # no renaming
        except Exception:
            continue
    return out

# ─────────────────────────────
# *) When searched with id
# ─────────────────────────────
def map_portal_detail_response_to_opportunity(detail_data: Dict[str, Any]) -> Optional[OpportunityDTO]:
    """
    Map detail endpoint payload to one OpportunityDTO.
    """
    data = (detail_data or {}).get("data")
    if not isinstance(data, dict):
        return None

    # Normalize detail payload into search-like shape so we can reuse search mapper.
    row: Dict[str, Any] = dict(data)
    summary = row.get("summary") if isinstance(row.get("summary"), dict) else {}
    if "summary_description" in row and "summary_description" not in summary:
        summary["summary_description"] = row.get("summary_description")
    if "additional_info_url" in row and "additional_info_url" not in summary:
        summary["additional_info_url"] = row.get("additional_info_url")
    row["summary"] = summary

    base = map_portal_search_response({"data": [row]})
    if not base:
        return None
    dto = base[0]

    dto.attachments = map_portal_attachments_response(detail_data)
    return dto
