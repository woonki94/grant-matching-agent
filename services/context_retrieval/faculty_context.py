from __future__ import annotations

from typing import Any, Dict, List

from db.models import Faculty
from services.context_retrieval.rag_chunk_retriever import retrieve_faculty_additional_info_chunks
from utils.content_extractor import load_extracted_content
from utils.keyword_utils import keyword_inventory_for_rerank


class FacultyContextBuilder:
    """
    faculty context builder.
    """

    DEFAULT_TOP_K_PER_SOURCE = 5
    DEFAULT_MAX_RECENT_PUB_TITLES = 5

    @staticmethod
    def _safe_list(value: Any) -> List[Any]:
        """Normalize nullable list-like values to a concrete list."""
        return list(value or [])

    @staticmethod
    def _to_keywords(fac: Faculty) -> Dict[str, Any]:
        """Read persisted faculty keywords (or empty payload if missing)."""
        return (getattr(fac, "keyword", None) and getattr(fac.keyword, "keywords", None)) or {}

    @staticmethod
    def _sorted_publications(fac: Faculty) -> List[Any]:
        """Sort publications by year (desc) for stable ordering in context payloads."""
        pubs = list(getattr(fac, "publications", None) or [])
        return sorted(pubs, key=lambda p: (getattr(p, "year", None) or 0), reverse=True)

    @classmethod
    def _publication_rows(cls, fac: Faculty) -> List[Dict[str, Any]]:
        """Map ORM publication rows into JSON-safe dictionaries."""
        rows: List[Dict[str, Any]] = []
        for pub in cls._sorted_publications(fac):
            rows.append(
                {
                    "id": getattr(pub, "id", None),
                    "title": getattr(pub, "title", None),
                    "year": getattr(pub, "year", None),
                    "abstract": getattr(pub, "abstract", None),
                }
            )
        return rows

    @staticmethod
    def _additional_info_rows(
        fac: Faculty,
        *,
        use_rag: bool,
        top_k_per_source: int,
        max_recent_pub_titles: int,
    ) -> List[Dict[str, Any]]:
        """Load faculty additional-info chunks via RAG, then fallback to all extracted chunks."""
        if use_rag:
            rag = retrieve_faculty_additional_info_chunks(
                fac,
                top_k_per_source=max(1, int(top_k_per_source)),
                max_recent_pub_titles=max(0, int(max_recent_pub_titles)),
            )
            rows = list(rag.get("additional_info_chunks") or [])
            if rows:
                return rows

        return load_extracted_content(
            list(getattr(fac, "additional_info", None) or []),
            url_attr="additional_info_url",
            group_chunks=False,
            include_row_meta=True,
        )

    @classmethod
    def _build_faculty_context_payload(
        cls,
        fac: Faculty,
        *,
        include_keywords: bool,
        use_rag: bool = True,
        top_k_per_source: int = DEFAULT_TOP_K_PER_SOURCE,
        max_recent_pub_titles: int = DEFAULT_MAX_RECENT_PUB_TITLES,
    ) -> Dict[str, Any]:
        """Build shared faculty payload used by both basic and full context variants."""
        additional_info_extracted = cls._additional_info_rows(
            fac,
            use_rag=use_rag,
            top_k_per_source=top_k_per_source,
            max_recent_pub_titles=max_recent_pub_titles,
        )
        publications = cls._publication_rows(fac)

        payload = {
            "faculty_id": getattr(fac, "faculty_id", None),
            "source_url": getattr(fac, "source_url", None),
            "profile_url": getattr(fac, "source_url", None),
            "name": getattr(fac, "name", None),
            "email": getattr(fac, "email", None),
            "phone": getattr(fac, "phone", None),
            "position": getattr(fac, "position", None),
            "organization": getattr(fac, "organization", None),
            "organizations": cls._safe_list(getattr(fac, "organizations", None)),
            "address": getattr(fac, "address", None),
            "biography": getattr(fac, "biography", None),
            # "profile_last_refreshed_at": getattr(fac, "profile_last_refreshed_at", None),
            "degrees": cls._safe_list(getattr(fac, "degrees", None)),
            "expertise": cls._safe_list(getattr(fac, "expertise", None)),
            "additional_info_count": len(additional_info_extracted),
            "publication_count": len(publications),
            "additional_info_extracted": additional_info_extracted,
            "publications": publications,
        }
        if include_keywords:
            payload["keywords"] = cls._to_keywords(fac)
        return payload

    @classmethod
    def build_faculty_basic_context(
        cls,
        fac: Faculty,
        *,
        use_rag: bool = True,
        top_k_per_source: int = DEFAULT_TOP_K_PER_SOURCE,
        max_recent_pub_titles: int = DEFAULT_MAX_RECENT_PUB_TITLES,
    ) -> Dict[str, Any]:
        """
        Build full faculty context for keyword generation (keywords excluded).
        """
        return cls._build_faculty_context_payload(
            fac,
            include_keywords=False,
            use_rag=use_rag,
            top_k_per_source=top_k_per_source,
            max_recent_pub_titles=max_recent_pub_titles,
        )

    @classmethod
    def build_faculty_full_context(
        cls,
        fac: Faculty,
        *,
        use_rag: bool = True,
        top_k_per_source: int = DEFAULT_TOP_K_PER_SOURCE,
        max_recent_pub_titles: int = DEFAULT_MAX_RECENT_PUB_TITLES,
    ) -> Dict[str, Any]:
        """
        Build full faculty context for matching/justification (keywords included).
        """
        return cls._build_faculty_context_payload(
            fac,
            include_keywords=True,
            use_rag=use_rag,
            top_k_per_source=top_k_per_source,
            max_recent_pub_titles=max_recent_pub_titles,
        )

    @classmethod
    def build_faculty_keyword_context(cls, fac: Faculty) -> Dict[str, Any]:
        """
        Build minimal keyword-only faculty context for reranking.
        """
        return {
            "faculty_id": getattr(fac, "faculty_id", None),
            "name": getattr(fac, "name", None),
            "email": getattr(fac, "email", None),
            "profile_url": getattr(fac, "source_url", None),
            "keywords": cls._to_keywords(fac),
        }

    @classmethod
    def build_faculty_keyword_inventory(cls, fac: Faculty) -> Dict[str, Any]:
        """Build faculty keyword inventory payload for reranking inputs."""
        keyword_ctx = cls.build_faculty_keyword_context(fac)
        kw_inv = keyword_inventory_for_rerank(dict(keyword_ctx.get("keywords") or {}))
        return {
            "faculty_id": keyword_ctx.get("faculty_id"),
            "name": keyword_ctx.get("name"),
            "domain_keywords": kw_inv.get("domain") or [],
            "specialization_keywords": kw_inv.get("specialization") or {},
        }

    @classmethod
    def build_faculty_source_linked_context(
        cls,
        fac: Faculty,
        *,
        use_rag: bool = True,
        top_k_per_source: int = DEFAULT_TOP_K_PER_SOURCE,
        max_recent_pub_titles: int = DEFAULT_MAX_RECENT_PUB_TITLES,
    ) -> Dict[str, Any]:
        """Build faculty context that includes source-linked chunks and publication rows."""
        return cls.build_faculty_full_context(
            fac,
            use_rag=use_rag,
            top_k_per_source=top_k_per_source,
            max_recent_pub_titles=max_recent_pub_titles,
        )
