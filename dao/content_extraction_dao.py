from __future__ import annotations

from typing import Any, Dict, List, Optional, Type
from sqlalchemy.orm import Session


class ContentExtractionDAO:
    """
    Generic DAO for models that have:
      - id (PK)
      - extract_status
    And for updates:
      - content_path, detected_type, content_char_count, extracted_at
      - extract_status, extract_error
    """

    def __init__(self, session: Session):
        self.session = session

    def fetch_pending(
        self,
        model: Type[Any],
        limit: int = 200,
        ids: Optional[List[int]] = None,
    ) -> List[Any]:
        query = self.session.query(model).filter(
            (model.extract_status == "pending") | (model.extract_status.is_(None))
        )
        if ids:
            clean_ids = [int(x) for x in list(ids) if x is not None]
            if not clean_ids:
                return []
            query = query.filter(model.id.in_(clean_ids))
        # Process only base rows; chunk rows are extraction outputs.
        if hasattr(model, "chunk_index"):
            query = query.filter(model.chunk_index == 0)
        return query.order_by(model.id.asc()).limit(limit).all()

    def bulk_update(self, model: Type[Any], updates: List[Dict[str, Any]]) -> None:
        if not updates:
            return
        self.session.bulk_update_mappings(model, updates)
