from __future__ import annotations
import os
from typing import List, Dict, Optional
import requests
from requests.adapters import HTTPAdapter, Retry
from dotenv import load_dotenv
#TODO: Make everything absolute path
from dto.grant_request_dto import SearchRequest, Pagination, SortOrder, OpportunityStatusFilter
from dto.grant_response_dto import PortalSearchResponseDTO, AttachmentDTO  # ← import new DTO

SIMPLER_API = "https://api.simpler.grants.gov/v1/opportunities/search"
DETAIL_API_BASE = "https://api.simpler.grants.gov/v1/opportunities/"

load_dotenv(dotenv_path="api.env")
API_KEY = os.getenv("SIMPLER_API_KEY")

def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"Content-Type": "application/json"})
    return s

def _auth_headers() -> Dict[str, str]:
    if not API_KEY:
        raise RuntimeError("Missing SIMPLER_API_KEY (expected in api_key.env or env)")
    return {"x-api-key": API_KEY}

def build_search_request(
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

def search_opportunities(req: SearchRequest) -> PortalSearchResponseDTO:
    sess = _session()
    resp = sess.post(SIMPLER_API, headers=_auth_headers(), json=req.to_dict(), timeout=30)
    resp.raise_for_status()
    return PortalSearchResponseDTO.from_dict(resp.json())

# ─────────────────────────────────────────────────────────────
# Detail fetch + enrichment
# ─────────────────────────────────────────────────────────────
def _fetch_attachments(opportunity_id: str) -> List[AttachmentDTO]:
    """GET /v1/opportunities/{opportunity_id} and parse only attachments."""
    sess = _session()
    url = f"{DETAIL_API_BASE}{opportunity_id}"
    r = sess.get(url, headers=_auth_headers(), timeout=30)
    #print(url)
    r.raise_for_status()
    payload = r.json() or {}
    raw_atts = (payload.get("data") or {}).get("attachments") or []
    return [AttachmentDTO.from_dict(a) for a in raw_atts]

def enrich_search_with_attachments(dto: PortalSearchResponseDTO) -> PortalSearchResponseDTO:

    if not dto or not dto.data:
        return dto
    for row in dto.data:
        oid = getattr(row, "opportunity_id", None)
        if not oid:
            continue
        try:
            row.attachments = _fetch_attachments(oid)
        except Exception:
            # Don’t fail the whole response if one detail call errors; leave empty
            row.attachments = row.attachments or []
    return dto