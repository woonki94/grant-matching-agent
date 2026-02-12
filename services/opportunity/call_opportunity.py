from __future__ import annotations

from typing import List, Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter, Retry

from config import settings, Grant_API_KEY

# Request DTOs
from dto.opportunity_request_dto import (
    SearchRequest,
    Pagination,
    SortOrder,
    AgencyFilter,
    OpportunityStatusFilter,
    Filters,
)

# New simple DTOs (DB-shaped)
from dto.opportunity_dto import OpportunityDTO

# New mappers (pure)
from mappers.portal_to_opportunity import (
    map_portal_search_response,
    map_portal_attachments_response,
)

SIMPLER_SEARCH = settings.simpler_search_url
SIMPLER_DETAIL = settings.simpler_detail_base_url


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
    if not Grant_API_KEY:
        raise RuntimeError("Missing SIMPLER_API_KEY (set in api_key.env or env)")
    return {"x-api-key": Grant_API_KEY}


# ─────────────────────────────────────────────────────────────
# Build request DTO (still using your request dto module)
# ─────────────────────────────────────────────────────────────
def build_search_request(
    *,
    page_offset: int = 1,
    page_size: int = 50,
    order_by: str = "post_date",
    sort_direction: str = "descending",
    statuses: Optional[List[str]] = None,
    agencies: Optional[List[str]] = None,
    q: Optional[str] = None,
) -> SearchRequest:
    statuses = statuses or ["forecasted", "posted"]

    return SearchRequest(
        pagination=Pagination(
            page_offset=page_offset,
            page_size=page_size,
            sort_order=[SortOrder(order_by=order_by, sort_direction=sort_direction)],
        ),
        filters=Filters(
            opportunity_status=OpportunityStatusFilter(one_of=statuses),
            agency=AgencyFilter(one_of=agencies) if agencies else None,
        ),
        q=q,
    )


# ─────────────────────────────────────────────────────────────
# Raw upstream calls (return dict)
# ─────────────────────────────────────────────────────────────
def search_opportunities(req: SearchRequest) -> Dict[str, Any]:
    sess = _session()
    resp = sess.post(
        SIMPLER_SEARCH,
        headers=_auth_headers(),
        json=req.to_dict(),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json() or {}


def fetch_attachments(opportunity_id: str) -> Dict[str, Any]:
    sess = _session()
    url = f"{SIMPLER_DETAIL}{opportunity_id}"
    resp = sess.get(url, headers=_auth_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json() or {}


# ─────────────────────────────────────────────────────────────
# Public pipeline (new DTO + mapper)
# ─────────────────────────────────────────────────────────────
def run_search_pipeline(
    *,
    page_offset: int = 1,
    page_size: int = 50,
    order_by: str = "post_date",
    sort_direction: str = "descending",
    statuses: Optional[List[str]] = None,
    agencies: Optional[List[str]] = None,
    q: Optional[str] = None,
    include_files: bool = True,
) -> List[OpportunityDTO]:
    """
    Returns list[OpportunityDTO] using the new simple DB-shaped DTOs.
    Attachments are fetched via nested detail call if include_files=True.
    """

    req = build_search_request(
        page_offset=page_offset,
        page_size=page_size,
        order_by=order_by,
        sort_direction=sort_direction,
        statuses=statuses,
        agencies=agencies,
        q=q,
    )

    # 1) Search (raw json)
    search_payload = search_opportunities(req)

    # 2) Map search response -> OpportunityDTO list (no attachments yet)
    opportunities: List[OpportunityDTO] = map_portal_search_response(search_payload)

    # 3) Nested call per opportunity for attachments
    if include_files:
        for opp in opportunities:
            try:
                attachments = fetch_attachments(opp.opportunity_id)
                opp.attachments = map_portal_attachments_response(attachments)

            except Exception:
                # Don’t fail the whole pipeline if one detail call fails
                opp.attachments = opp.attachments or []

    return opportunities
