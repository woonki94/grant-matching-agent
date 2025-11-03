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
class Pagination:
    page_offset: int
    page_size: int
    sort_order: List[SortOrder]
    opportunity_status: OpportunityStatusFilter

@dataclass
class SearchRequest:
    pagination: Pagination
    q: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {"pagination": asdict(self.pagination)}
        if self.q:
            payload["query"] = self.q
        return payload