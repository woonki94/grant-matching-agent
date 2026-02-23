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
    map_portal_detail_response_to_opportunity,
    map_portal_search_response,
    map_portal_attachments_response,
)


class OpportunitySearchService:
    def __init__(
        self,
        *,
        search_url: Optional[str] = None,
        detail_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.search_url = search_url or settings.simpler_search_url
        self.detail_base_url = detail_base_url or settings.simpler_detail_base_url
        self.api_key = api_key or Grant_API_KEY

    def _session(self) -> requests.Session:
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

    def _auth_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise RuntimeError("Missing SIMPLER_API_KEY (set in api_key.env or env)")
        return {"x-api-key": self.api_key}

    def build_search_request(
        self,
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

    def search_opportunities(self, req: SearchRequest) -> Dict[str, Any]:
        sess = self._session()
        resp = sess.post(
            self.search_url,
            headers=self._auth_headers(),
            json=req.to_dict(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json() or {}

    def fetch_opportunity_detail(self, opportunity_id: str) -> Dict[str, Any]:
        """
        Fetch full opportunity detail payload by opportunity_id.
        This includes richer fields than search results (e.g., attachments, competitions, forms).
        """
        sess = self._session()
        url = f"{self.detail_base_url}{opportunity_id}"
        resp = sess.get(url, headers=self._auth_headers(), timeout=30)
        resp.raise_for_status()
        return resp.json() or {}

    def fetch_opportunity_by_id(self, opportunity_id: str) -> Dict[str, Any]:
        """
        Return the full `data` object from the opportunity detail API for one opp_id.
        Use this when user already knows the target opportunity.
        """
        payload = self.fetch_opportunity_detail(opportunity_id)
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError(f"Invalid detail payload for opportunity_id={opportunity_id}")
        return data

    def run_search_pipeline(
        self,
        *,
        opportunity_id: Optional[str] = None,
        page_offset: int = 1,
        page_size: int = 50,
        order_by: str = "post_date",
        sort_direction: str = "descending",
        statuses: Optional[List[str]] = None,
        agencies: Optional[List[str]] = None,
        q: Optional[str] = None,
        include_files: bool = True,
    ) -> List[OpportunityDTO]:
        # Exact-by-id path: retrieve full detail payload and map to one DTO.
        if opportunity_id:
            detail_payload = self.fetch_opportunity_detail(opportunity_id)
            dto = map_portal_detail_response_to_opportunity(detail_payload)
            return [dto] if dto is not None else []

        # Broad search path: query + filters + optional attachment enrichment.
        req = self.build_search_request(
            page_offset=page_offset,
            page_size=page_size,
            order_by=order_by,
            sort_direction=sort_direction,
            statuses=statuses,
            agencies=agencies,
            q=q,
        )
        search_payload = self.search_opportunities(req)
        opportunities: List[OpportunityDTO] = map_portal_search_response(search_payload)

        if include_files:
            for opp in opportunities:
                try:
                    attachments = self.fetch_opportunity_detail(opp.opportunity_id)
                    opp.attachments = map_portal_attachments_response(attachments)
                except Exception:
                    opp.attachments = opp.attachments or []

        return opportunities
