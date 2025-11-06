from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

# --- Request DTOs for clean body construction ---
@dataclass
class SortOrder:
    order_by: str
    sort_direction: str


@dataclass
class OpportunityStatusFilter:
    one_of: List[str]

@dataclass
class Filters:
    # Add other filters here as needed (agency, post_date, etc.)
    opportunity_status: Optional[OpportunityStatusFilter] = None

@dataclass
class Pagination:
    page_offset: int
    page_size: int
    sort_order: List[SortOrder]

@dataclass
class SearchRequest:
    pagination: Pagination
    filters: Filters
    q: Optional[str] = None  # caller uses q; JSON expects "query"

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "pagination": asdict(self.pagination),
            "filters": asdict(self.filters)
        }
        # Drop None filters so you don't send `"opportunity_status": null`
        payload["filters"] = {k: v for k, v in payload["filters"].items() if v is not None}

        if self.q:
            payload["query"] = self.q
        return payload