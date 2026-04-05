from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from dao.faculty_dao import FacultyDAO
from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.models import Faculty, Opportunity
from services.context_retrieval.faculty_context import FacultyContextBuilder
from services.context_retrieval.justification_context import JustificationContextBuilder
from services.context_retrieval.keyword_context import KeywordContextBuilder
from services.context_retrieval.matching_context import MatchingContextBuilder
from services.context_retrieval.opportunity_context import OpportunityContextBuilder


class ContextGenerator:
    """Context retrieval facade.
    """

    def __init__(
        self,
        *,
        faculty_builder: Optional[FacultyContextBuilder] = None,
        opportunity_builder: Optional[OpportunityContextBuilder] = None,
        keyword_builder: Optional[KeywordContextBuilder] = None,
        matching_builder: Optional[MatchingContextBuilder] = None,
        justification_builder: Optional[JustificationContextBuilder] = None,
        faculty_dao_cls=FacultyDAO,
        opportunity_dao_cls=OpportunityDAO,
        match_dao_cls=MatchDAO,
    ):
        """Initialize with injectable builders and DAO classes."""
        self.faculty = faculty_builder or FacultyContextBuilder()
        self.opportunity = opportunity_builder or OpportunityContextBuilder()
        self.keyword = keyword_builder or KeywordContextBuilder()
        self.matching = matching_builder or MatchingContextBuilder()
        self.justification = justification_builder or JustificationContextBuilder()

        self.faculty_dao_cls = faculty_dao_cls
        self.opportunity_dao_cls = opportunity_dao_cls
        self.match_dao_cls = match_dao_cls

    # ===================================================
    # Faculty Context
    # ===================================================
    def build_faculty_basic_context(
        self,
        fac: Faculty,
        *,
        use_rag: bool = True,
        top_k_per_source: int = FacultyContextBuilder.DEFAULT_TOP_K_PER_SOURCE,
        max_recent_pub_titles: int = FacultyContextBuilder.DEFAULT_MAX_RECENT_PUB_TITLES,
    ) -> Dict[str, Any]:
        """Build faculty context for keyword generation (keywords excluded)."""
        return self.faculty.build_faculty_basic_context(
            fac,
            use_rag=use_rag,
            top_k_per_source=top_k_per_source,
            max_recent_pub_titles=max_recent_pub_titles,
        )

    def build_faculty_full_context(
        self,
        fac: Faculty,
        *,
        use_rag: bool = True,
        top_k_per_source: int = FacultyContextBuilder.DEFAULT_TOP_K_PER_SOURCE,
        max_recent_pub_titles: int = FacultyContextBuilder.DEFAULT_MAX_RECENT_PUB_TITLES,
    ) -> Dict[str, Any]:
        """Build faculty context for matching/justification (keywords included)."""
        return self.faculty.build_faculty_full_context(
            fac,
            use_rag=use_rag,
            top_k_per_source=top_k_per_source,
            max_recent_pub_titles=max_recent_pub_titles,
        )

    def build_faculty_keyword_context(self, fac: Faculty) -> Dict[str, Any]:
        """Build minimal faculty keyword-only context for reranking."""
        return self.faculty.build_faculty_keyword_context(fac)

    def build_faculty_merged_content_context(
        self,
        fac: Faculty,
        *,
        use_rag: bool = True,
        top_k_per_source: int = FacultyContextBuilder.DEFAULT_TOP_K_PER_SOURCE,
        max_recent_pub_titles: int = FacultyContextBuilder.DEFAULT_MAX_RECENT_PUB_TITLES,
    ) -> Dict[str, Any]:
        """Build flat faculty context with only name + merged content text."""
        return self.faculty.build_faculty_merged_content_context(
            fac,
            use_rag=use_rag,
            top_k_per_source=top_k_per_source,
            max_recent_pub_titles=max_recent_pub_titles,
        )

    # ===================================================
    # Opportunity Context
    # ===================================================
    def build_opportunity_basic_context(
        self,
        opp: Opportunity,
        *,
        use_rag: bool = True,
        top_k_per_additional_source: int = OpportunityContextBuilder.DEFAULT_TOP_K_PER_ADDITIONAL_SOURCE,
        top_k_per_attachment_source: int = OpportunityContextBuilder.DEFAULT_TOP_K_PER_ATTACHMENT_SOURCE,
    ) -> Dict[str, Any]:
        """Build opportunity context for keyword generation (keywords excluded)."""
        return self.opportunity.build_opportunity_basic_context(
            opp,
            use_rag=use_rag,
            top_k_per_additional_source=top_k_per_additional_source,
            top_k_per_attachment_source=top_k_per_attachment_source,
        )

    def build_opportunity_full_context(
        self,
        opp: Opportunity,
        *,
        use_rag: bool = True,
        top_k_per_additional_source: int = OpportunityContextBuilder.DEFAULT_TOP_K_PER_ADDITIONAL_SOURCE,
        top_k_per_attachment_source: int = OpportunityContextBuilder.DEFAULT_TOP_K_PER_ATTACHMENT_SOURCE,
    ) -> Dict[str, Any]:
        """Build opportunity context for matching/justification (keywords included)."""
        return self.opportunity.build_opportunity_full_context(
            opp,
            use_rag=use_rag,
            top_k_per_additional_source=top_k_per_additional_source,
            top_k_per_attachment_source=top_k_per_attachment_source,
        )

    def build_opportunity_keyword_context(self, opp: Opportunity) -> Dict[str, Any]:
        """Build minimal opportunity keyword-only context for reranking."""
        return self.opportunity.build_opportunity_keyword_context(opp)

    def build_opportunity_matching_context(self, opp: Opportunity) -> Dict[str, Any]:
        """Build compact opportunity context for matching/group reasoning."""
        return self.opportunity.build_opportunity_matching_context(opp)

    def build_opportunity_merged_content_context(
        self,
        opp: Opportunity,
        *,
        use_rag: bool = True,
        top_k_per_additional_source: int = OpportunityContextBuilder.DEFAULT_TOP_K_PER_ADDITIONAL_SOURCE,
        top_k_per_attachment_source: int = OpportunityContextBuilder.DEFAULT_TOP_K_PER_ATTACHMENT_SOURCE,
    ) -> Dict[str, Any]:
        """Build flat opportunity context with only title + merged summary/content text."""
        return self.opportunity.build_opportunity_merged_content_context(
            opp,
            use_rag=use_rag,
            top_k_per_additional_source=top_k_per_additional_source,
            top_k_per_attachment_source=top_k_per_attachment_source,
        )

    # ===================================================
    # Keyword Context Formatting
    # ===================================================
    def dedupe_keyword_texts(self, values: List[Any]) -> List[str]:
        return self.keyword.dedupe_texts(values)

    def build_keyword_context_batches(
        self,
        *,
        context: Dict[str, Any],
        max_chars: int,
    ) -> List[Dict[str, Any]]:
        return self.keyword.build_context_batches(
            context=context,
            max_chars=max_chars,
        )

    def build_keyword_weight_context_from_batches(
        self,
        *,
        batches: List[Dict[str, Any]],
        max_chars: int,
    ) -> Dict[str, Any]:
        return self.keyword.build_weight_context_from_batches(
            batches=batches,
            max_chars=max_chars,
        )

    def format_keyword_merge_input_row(
        self,
        *,
        batch_idx: int,
        candidates: List[Any],
        keyword_bucket: Dict[str, Any],
        max_batch_keywords: int = KeywordContextBuilder.DEFAULT_MAX_BATCH_KEYWORDS,
    ) -> Dict[str, Any]:
        return self.keyword.format_merge_input_row(
            batch_idx=batch_idx,
            candidates=candidates,
            keyword_bucket=keyword_bucket,
            max_batch_keywords=max_batch_keywords,
        )

    def normalize_keyword_merge_output(
        self,
        merged: Any,
        *,
        max_domain: int = KeywordContextBuilder.DEFAULT_MAX_MERGED_DOMAIN,
        max_specialization: int = KeywordContextBuilder.DEFAULT_MAX_MERGED_SPECIALIZATION,
    ) -> Dict[str, List[str]]:
        return self.keyword.normalize_merge_output(
            merged,
            max_domain=max_domain,
            max_specialization=max_specialization,
        )

    def fallback_keyword_merge_bucket(
        self,
        *,
        batch_domains: List[List[str]],
        batch_specializations: List[List[str]],
        max_domain: int = KeywordContextBuilder.DEFAULT_MAX_MERGED_DOMAIN,
        max_specialization: int = KeywordContextBuilder.DEFAULT_MAX_MERGED_SPECIALIZATION,
    ) -> Dict[str, List[str]]:
        return self.keyword.fallback_merge_bucket(
            batch_domains=batch_domains,
            batch_specializations=batch_specializations,
            max_domain=max_domain,
            max_specialization=max_specialization,
        )

    def build_keyword_source_catalog(
        self,
        *,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build specialization source catalog from keyword-generation context."""
        return self.keyword.build_source_catalog(context=context)

    def attach_keyword_sources_by_cosine(
        self,
        *,
        keywords: Dict[str, Any],
        context: Dict[str, Any],
        embedding_client: Any,
        max_sources_per_specialization: int = 4,
        min_similarity: float = 0.10,
    ) -> Dict[str, Any]:
        """Attach specialization sources using cosine mapping over context-derived source catalog."""
        return self.keyword.attach_sources_by_cosine(
            keywords=keywords,
            context=context,
            embedding_client=embedding_client,
            max_sources_per_specialization=max_sources_per_specialization,
            min_similarity=min_similarity,
        )

    # ===================================================
    # Justification Context (DAO orchestration here)
    # ===================================================
    def build_grant_context_only(
        self,
        *,
        sess,
        opportunity_id: str,
        preview_chars: int = 700,
    ) -> Dict[str, Any]:
        """Build one grant-only context payload used by grant explanation chain.

        This intentionally reuses the opportunity basic context directly.
        """
        oid = str(opportunity_id or "").strip()
        if not oid:
            raise ValueError("opportunity_id is required")
        _ = preview_chars

        odao = self.opportunity_dao_cls(sess)
        opps = odao.read_opportunities_by_ids_with_relations([oid])
        opp = opps[0] if opps else None
        if not opp:
            raise ValueError(f"Opportunity not found: {oid}")

        # Temporary experiment mode: use full extracted context without RAG filtering.
        return self.build_opportunity_basic_context(opp, use_rag=False)

    def build_faculty_recommendation_source_linked_payload(
        self,
        *,
        sess,
        fac: Faculty,
        top_rows: List[Tuple[str, float, float]],
    ) -> Dict[str, Any]:
        """Build source-linked faculty recommendation payload with evidence rows."""
        odao = self.opportunity_dao_cls(sess)
        mdao = self.match_dao_cls(sess)

        opportunity_ids: List[str] = []
        seen = set()
        for oid, _, _ in list(top_rows or []):
            value = str(oid or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            opportunity_ids.append(value)

        opportunities = odao.read_opportunities_by_ids_with_relations(opportunity_ids) if opportunity_ids else []

        try:
            faculty_id = int(getattr(fac, "faculty_id"))
        except Exception as exc:
            raise ValueError("fac.faculty_id is required") from exc
        match_rows: List[Dict[str, Any]] = []
        for oid in opportunity_ids:
            row = mdao.get_match_for_faculty_opportunity(
                faculty_id=faculty_id,
                opportunity_id=oid,
            )
            if row:
                match_rows.append(dict(row))

        return self.justification.build_faculty_recommendation_source_linked_payload_from_entities(
            fac=fac,
            opportunities=list(opportunities or []),
            top_rows=list(top_rows or []),
            match_rows=match_rows,
            build_faculty_keyword_context=self.build_faculty_keyword_context,
            build_opportunity_keyword_context=self.build_opportunity_keyword_context,
            # Temporary experiment mode: load full extracted chunks without RAG filtering.
            build_faculty_source_linked_context=lambda fac_obj: self.faculty.build_faculty_source_linked_context(
                fac_obj,
                use_rag=False,
            ),
            build_opportunity_source_linked_context=lambda opp_obj: self.opportunity.build_opportunity_source_linked_context(
                opp_obj,
                use_rag=False,
            ),
        )

    def build_faculty_recommendation_source_linked_text(
        self,
        *,
        sess,
        fac: Faculty,
        top_rows: List[Tuple[str, float, float]],
        max_requirements: int = 5,
        grant_evidence_per_requirement: int = 3,
        faculty_evidence_per_requirement: int = 3,
    ) -> str:
        """Render compact source-linked evidence text for one faculty's top grants."""
        payload = self.build_faculty_recommendation_source_linked_payload(
            sess=sess,
            fac=fac,
            top_rows=top_rows,
        )
        return self.justification.build_faculty_recommendation_source_linked_text_from_payload(
            payload=payload,
            max_requirements=max_requirements,
            grant_evidence_per_requirement=grant_evidence_per_requirement,
            faculty_evidence_per_requirement=faculty_evidence_per_requirement,
        )

    def build_group_justification_stage_inputs_from_contexts(
        self,
        *,
        opp_ctx: Dict[str, Any],
        team_ids: List[int],
        match_rows_by_faculty: Dict[int, Dict[str, Any]],
        faculty_contexts_by_id: Optional[Dict[int, Dict[str, Any]]] = None,
        grant_brief_context: Optional[Dict[str, Any]] = None,
        coverage: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build stage-specific group-justification inputs from prepared contexts/rows."""
        return self.justification.build_group_justification_stage_inputs_from_contexts(
            opp_ctx=dict(opp_ctx or {}),
            team_ids=[int(x) for x in list(team_ids or [])],
            match_rows_by_faculty=dict(match_rows_by_faculty or {}),
            faculty_contexts_by_id=dict(faculty_contexts_by_id or {}),
            grant_brief_context=dict(grant_brief_context or {}),
            coverage=dict(coverage or {}),
        )

    # ===================================================
    # Matching Context (DAO orchestration here)
    # ===================================================

    def build_group_matching_context_from_contexts(
        self,
        *,
        opp_ctx: Dict[str, Any],
        fac_ctxs: List[Dict[str, Any]],
        coverage: Any,
        member_coverages: Optional[Dict[int, Dict[str, Dict[int, float]]]] = None,
        group_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Pass-through variant when caller already prepared contexts externally."""
        return self.matching.build_group_matching_context(
            opp_ctx=opp_ctx,
            fac_ctxs=fac_ctxs,
            coverage=coverage,
            member_coverages=member_coverages,
            group_meta=group_meta,
        )

    def build_member_coverages_for_opportunity(
        self,
        *,
        sess,
        opportunity_id: str,
        limit_rows: int = 500,
    ) -> Dict[int, Dict[str, Dict[int, float]]]:
        """Aggregate per-faculty coverage maps for one opportunity from match rows."""
        mdao = self.match_dao_cls(sess)
        rows = mdao.list_matches_for_opportunity(str(opportunity_id), limit=int(limit_rows)) or []
        return self.matching.build_member_coverages_from_match_rows(match_rows=rows)

    def build_matching_inputs_payload_for_opportunity(
        self,
        *,
        sess,
        opportunity_id: str,
        limit_rows: int = 500,
    ) -> Dict[str, Any]:
        """Build matching input payload with faculty_ids, requirements, and coverage."""
        odao = self.opportunity_dao_cls(sess)
        mdao = self.match_dao_cls(sess)
        opps = odao.read_opportunities_by_ids_with_relations([str(opportunity_id)])
        if not opps:
            raise ValueError(f"Opportunity not found: {opportunity_id}")
        match_rows = mdao.list_matches_for_opportunity(str(opportunity_id), limit=int(limit_rows)) or []
        payload = self.matching.build_matching_inputs_payload_from_opportunity_and_match_rows(
            opp=opps[0],
            match_rows=match_rows,
            build_opportunity_keyword_context=self.build_opportunity_keyword_context,
        )
        if not dict(payload.get("coverage") or {}):
            raise ValueError("No match rows found.")
        return payload

    def build_top_match_payload(
        self,
        *,
        sess,
        top_rows: List[Tuple[str, float, float]],
    ) -> List[Dict[str, Any]]:
        """Build top-match display payloads by composing opportunity keyword contexts."""
        if not list(top_rows or []):
            return []
        odao = self.opportunity_dao_cls(sess)
        oids = [str(oid) for oid, _, _ in list(top_rows or [])]
        opps = odao.read_opportunities_by_ids_with_relations(oids)
        return self.matching.build_top_match_payload_from_entities(
            top_rows=list(top_rows or []),
            opportunities=list(opps or []),
            build_opportunity_matching_context=self.build_opportunity_matching_context,
        )

    def build_rerank_keyword_inventory_for_faculty(
        self,
        *,
        sess,
        faculty_id: int,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build faculty-vs-grants keyword inventory payload for LLM reranking.

        By default this returns all existing matched grants for the faculty.
        Set `k` (>0) to apply an optional row cap.
        """
        fid = int(faculty_id)
        limit = int(k) if k is not None and int(k) > 0 else None

        fdao = self.faculty_dao_cls(sess)
        mdao = self.match_dao_cls(sess)
        odao = self.opportunity_dao_cls(sess)

        fac = fdao.get_with_relations_by_id(fid)
        if not fac:
            raise ValueError(f"Faculty not found: {fid}")
        top_rows = mdao.top_matches_for_faculty(fid, k=limit)
        ordered_ids = [str(oid) for oid, _, _ in list(top_rows or [])]
        opps = odao.read_opportunities_by_ids_with_relations(ordered_ids)
        return self.matching.build_rerank_keyword_inventory_from_entities(
            faculty=fac,
            top_rows=list(top_rows or []),
            opportunities=list(opps or []),
            build_faculty_keyword_inventory=self.faculty.build_faculty_keyword_inventory,
            build_opportunity_keyword_inventory=self.opportunity.build_opportunity_keyword_inventory,
        )
