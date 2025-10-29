#TODO: Make everything absolute path
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict
import os
import requests
from requests.adapters import HTTPAdapter, Retry
from dotenv import load_dotenv

# Request/Response DTOs (your existing ones)
from dto.requestDTO import SearchRequest, Pagination, SortOrder, OpportunityStatusFilter
from dto.responseDTO import PortalSearchResponseDTO, AttachmentDTO

# Persistence DTOs we built earlier
from dto.persistenceDTO import (
    OpportunityPersistenceDTO,
    AttachmentPersistenceDTO,
    build_opportunity_persistence_list,
    build_attachment_persistence_list,
)

#TODO: put links in hidden config file
SIMPLER_SEARCH = "https://api.simpler.grants.gov/v1/opportunities/search"
SIMPLER_DETAIL = "https://api.simpler.grants.gov/v1/opportunities/"

env_path = Path(__file__).resolve().parents[1] / "api.env"
loaded = load_dotenv(dotenv_path=env_path, override=True)
API_KEY = os.getenv("API_KEY")


# ─────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────
def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"Content-Type": "application/json"})
    return s

def _auth_headers() -> Dict[str, str]:
    if not API_KEY:
        raise RuntimeError("Missing SIMPLER_API_KEY (set in api_key.env or env)")
    return {"x-api-key": API_KEY}


# ─────────────────────────────────────────────────────────────
# Build request DTO
# ─────────────────────────────────────────────────────────────
def build_search_request(
    *,
    page_offset: int = 1,
    page_size: int = 5,
    order_by: str = "post_date",
    sort_direction: str = "descending",
    statuses: Optional[List[str]] = None,
    q: Optional[str] = None,
) -> SearchRequest:
    statuses = statuses or ["forecasted", "posted"]
    return SearchRequest(
        pagination=Pagination(
            page_offset=page_offset,
            page_size=page_size,
            sort_order=[SortOrder(order_by=order_by, sort_direction=sort_direction)],
            opportunity_status=OpportunityStatusFilter(one_of=statuses),
        ),
        q=q,
    )

# ─────────────────────────────────────────────────────────────
# Core upstream calls (search + detail)
# ─────────────────────────────────────────────────────────────
def search_opportunities(req: SearchRequest) -> PortalSearchResponseDTO:
    sess = _session()
    resp = sess.post(SIMPLER_SEARCH, headers=_auth_headers(), json=req.to_dict(), timeout=30)
    resp.raise_for_status()
    return PortalSearchResponseDTO.from_dict(resp.json())

def _fetch_attachments(opportunity_id: str) -> List[AttachmentDTO]:
    sess = _session()
    url = f"{SIMPLER_DETAIL}{opportunity_id}"
    resp = sess.get(url, headers=_auth_headers(), timeout=30)
    resp.raise_for_status()
    payload = resp.json() or {}
    raw = (payload.get("data") or {}).get("attachments") or []
    return [AttachmentDTO.from_dict(x) for x in raw]

# ─────────────────────────────────────────────────────────────
# Enrichment + conversion to Persistence DTOs
# ─────────────────────────────────────────────────────────────
def enrich_with_attachments(dto: PortalSearchResponseDTO) -> PortalSearchResponseDTO:
    if not dto or not dto.data:
        return dto
    for row in dto.data:
        oid = getattr(row, "opportunity_id", None)
        if not oid:
            continue
        try:
            row.attachments = _fetch_attachments(oid)
        except Exception:
            # leave attachments empty on failure
            row.attachments = row.attachments or []
    return dto


# ─────────────────────────────────────────────────────────────
# Public service functions consumed by controller
# ─────────────────────────────────────────────────────────────
def run_search_pipeline(
    *,
    page_offset: int = 1,
    page_size: int = 5,
    order_by: str = "post_date",
    sort_direction: str = "descending",
    statuses: Optional[List[str]] = None,
    q: Optional[str] = None,
    include_files: bool = True,
):

    req = build_search_request(
        page_offset=page_offset,
        page_size=page_size,
        order_by=order_by,
        sort_direction=sort_direction,
        statuses=statuses,
        q=q,
    )
    response_dto = search_opportunities(req)
    response_dto = enrich_with_attachments(response_dto)

    # Project into Persistence DTOs (flatten nested summary fields, strip HTML)
    opportunities_p: List[OpportunityPersistenceDTO] = build_opportunity_persistence_list(response_dto)
    attachments_p: List[AttachmentPersistenceDTO] = build_attachment_persistence_list(response_dto)

    return response_dto, opportunities_p, attachments_p

