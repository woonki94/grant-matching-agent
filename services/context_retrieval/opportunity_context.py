from __future__ import annotations

from typing import Any, Dict, List

from db.models import Opportunity
from services.context_retrieval.rag_chunk_retriever import retrieve_opportunity_supporting_chunks
from utils.content_extractor import load_extracted_content
from utils.keyword_utils import keyword_inventory_for_rerank


class OpportunityContextBuilder:
    """Standardized opportunity context builder.

    This builder is isolated from legacy opportunity context logic so we can
    evolve schema and retrieval behavior safely before replacement.
    """

    DEFAULT_TOP_K_PER_ADDITIONAL_SOURCE = 10
    DEFAULT_TOP_K_PER_ATTACHMENT_SOURCE = 10

    @staticmethod
    def _to_keywords(opp: Opportunity) -> Dict[str, Any]:
        """Read persisted opportunity keywords (or empty payload if missing)."""
        return (getattr(opp, "keyword", None) and getattr(opp.keyword, "keywords", None)) or {}

    @staticmethod
    def _first_additional_info_link(opp: Opportunity) -> Any:
        """Use the first additional-info URL as the opportunity link when available."""
        infos = list(getattr(opp, "additional_info", None) or [])
        if not infos:
            return None
        return getattr(infos[0], "additional_info_url", None)

    @staticmethod
    def _extracted_rows(
        opp: Opportunity,
        *,
        use_rag: bool,
        top_k_per_additional_source: int,
        top_k_per_attachment_source: int,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load additional-link and attachment chunks via RAG, then fallback to all chunks."""
        additional_blocks: List[Dict[str, Any]] = []
        attachment_blocks: List[Dict[str, Any]] = []

        if use_rag:
            rag = retrieve_opportunity_supporting_chunks(
                opp,
                top_k_per_additional_source=max(1, int(top_k_per_additional_source)),
                top_k_per_attachment_source=max(1, int(top_k_per_attachment_source)),
            )
            additional_blocks = list(rag.get("additional_info_chunks") or [])
            attachment_blocks = list(rag.get("attachment_chunks") or [])

        if not additional_blocks:
            additional_blocks = load_extracted_content(
                list(getattr(opp, "additional_info", None) or []),
                url_attr="additional_info_url",
                group_chunks=False,
                include_row_meta=True,
            )

        if not attachment_blocks:
            attachment_blocks = load_extracted_content(
                list(getattr(opp, "attachments", None) or []),
                url_attr="file_download_path",
                title_attr="file_name",
                group_chunks=False,
                include_row_meta=True,
            )

        return additional_blocks, attachment_blocks

    @classmethod
    def _build_opportunity_context_payload(
        cls,
        opp: Opportunity,
        *,
        include_keywords: bool,
        use_rag: bool = True,
        top_k_per_additional_source: int = DEFAULT_TOP_K_PER_ADDITIONAL_SOURCE,
        top_k_per_attachment_source: int = DEFAULT_TOP_K_PER_ATTACHMENT_SOURCE,
    ) -> Dict[str, Any]:
        """Build shared opportunity payload used by both basic and full variants."""
        additional_blocks, attachment_blocks = cls._extracted_rows(
            opp,
            use_rag=use_rag,
            top_k_per_additional_source=top_k_per_additional_source,
            top_k_per_attachment_source=top_k_per_attachment_source,
        )

        payload: Dict[str, Any] = {
            "opportunity_id": getattr(opp, "opportunity_id", None),
            "opportunity_title": getattr(opp, "opportunity_title", None),
            "agency_name": getattr(opp, "agency_name", None),
            "category": getattr(opp, "category", None),
            "opportunity_status": getattr(opp, "opportunity_status", None),
            "opportunity_link": cls._first_additional_info_link(opp),
            "summary_description": getattr(opp, "summary_description", None),
            "additional_info_count": len(additional_blocks),
            "attachment_count": len(attachment_blocks),
            "additional_info_extracted": additional_blocks,
            "attachments_extracted": attachment_blocks,
        }
        if include_keywords:
            payload["keywords"] = cls._to_keywords(opp)
        return payload

    @classmethod
    def build_opportunity_basic_context(
        cls,
        opp: Opportunity,
        *,
        use_rag: bool = True,
        top_k_per_additional_source: int = DEFAULT_TOP_K_PER_ADDITIONAL_SOURCE,
        top_k_per_attachment_source: int = DEFAULT_TOP_K_PER_ATTACHMENT_SOURCE,
    ) -> Dict[str, Any]:
        """Build full opportunity context for keyword generation (keywords excluded)."""
        return cls._build_opportunity_context_payload(
            opp,
            include_keywords=False,
            use_rag=use_rag,
            top_k_per_additional_source=top_k_per_additional_source,
            top_k_per_attachment_source=top_k_per_attachment_source,
        )

    @classmethod
    def build_opportunity_full_context(
        cls,
        opp: Opportunity,
        *,
        use_rag: bool = True,
        top_k_per_additional_source: int = DEFAULT_TOP_K_PER_ADDITIONAL_SOURCE,
        top_k_per_attachment_source: int = DEFAULT_TOP_K_PER_ATTACHMENT_SOURCE,
    ) -> Dict[str, Any]:
        """Build full opportunity context for matching/justification (keywords included)."""
        return cls._build_opportunity_context_payload(
            opp,
            include_keywords=True,
            use_rag=use_rag,
            top_k_per_additional_source=top_k_per_additional_source,
            top_k_per_attachment_source=top_k_per_attachment_source,
        )

    @classmethod
    def build_opportunity_keyword_context(cls, opp: Opportunity) -> Dict[str, Any]:
        """Build minimal keyword-only opportunity context for reranking."""
        return {
            "opportunity_id": getattr(opp, "opportunity_id", None),
            "opportunity_title": getattr(opp, "opportunity_title", None),
            "agency_name": getattr(opp, "agency_name", None),
            "opportunity_link": cls._first_additional_info_link(opp),
            "keywords": cls._to_keywords(opp),
        }

    @classmethod
    def build_opportunity_matching_context(cls, opp: Opportunity) -> Dict[str, Any]:
        """Build compact opportunity context for matching/group reasoning."""
        return {
            "opportunity_id": getattr(opp, "opportunity_id", None),
            "opportunity_title": getattr(opp, "opportunity_title", None),
            "agency_name": getattr(opp, "agency_name", None),
            "summary_description": getattr(opp, "summary_description", None),
            "keywords": cls._to_keywords(opp),
        }

    @classmethod
    def build_opportunity_keyword_inventory(cls, opp: Opportunity) -> Dict[str, Any]:
        """Build opportunity keyword inventory payload for reranking inputs."""
        keyword_ctx = cls.build_opportunity_keyword_context(opp)
        kw_inv = keyword_inventory_for_rerank(dict(keyword_ctx.get("keywords") or {}))
        return {
            "opportunity_id": keyword_ctx.get("opportunity_id"),
            "opportunity_title": keyword_ctx.get("opportunity_title"),
            "grant_domain_keywords": kw_inv.get("domain") or [],
            "grant_specialization_keywords": kw_inv.get("specialization") or {},
        }

    @classmethod
    def build_opportunity_source_linked_context(
        cls,
        opp: Opportunity,
        *,
        use_rag: bool = True,
        top_k_per_additional_source: int = DEFAULT_TOP_K_PER_ADDITIONAL_SOURCE,
        top_k_per_attachment_source: int = DEFAULT_TOP_K_PER_ATTACHMENT_SOURCE,
    ) -> Dict[str, Any]:
        """Build opportunity context that includes source-linked chunk blocks."""
        return cls.build_opportunity_full_context(
            opp,
            use_rag=use_rag,
            top_k_per_additional_source=top_k_per_additional_source,
            top_k_per_attachment_source=top_k_per_attachment_source,
        )
