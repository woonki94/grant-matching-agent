from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from dao.opportunity_dao import OpportunityDAO
from db.models import Faculty, Opportunity
from services.context_retrieval.faculty_context import FacultyContextBuilder
from services.context_retrieval.justification_context import JustificationContextBuilder
from services.context_retrieval.matching_context import MatchingContextBuilder
from services.context_retrieval.opportunity_context import OpportunityContextBuilder


class ContextGenerator:
    """Facade that composes faculty and grant context_retrieval builders."""

    def __init__(self):
        self.faculty = FacultyContextBuilder()
        self.grant = OpportunityContextBuilder()
        self.matching = MatchingContextBuilder()
        self.justification = JustificationContextBuilder(
            faculty_builder=self.faculty,
            opportunity_builder=self.grant,
        )

    # ==============================
    # Opportunity Context
    # ==============================
    def build_opportunity_basic_context(self, opp: Opportunity) -> Dict[str, Any]:
        return self.grant.build_opportunity_context(opp, profile="basic")

    def build_opportunity_explanation_contexts(
        self,
        *,
        sess,
        opportunity_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        ids: List[str] = []
        seen = set()
        for x in opportunity_ids or []:
            oid = str(x or "").strip()
            if not oid or oid in seen:
                continue
            seen.add(oid)
            ids.append(oid)
        if not ids:
            return {}

        odao = OpportunityDAO(sess)
        opps = odao.read_opportunities_by_ids_with_relations(ids)
        by_id = {str(getattr(o, "opportunity_id", "") or ""): o for o in opps}

        out: Dict[str, Dict[str, Any]] = {}
        for oid in ids:
            opp = by_id.get(oid)
            if not opp:
                continue
            out[oid] = self.grant.build_opportunity_context(opp, profile="explanation")
        return out

    def build_grant_context_only(
        self,
        *,
        sess,
        opportunity_id: str,
        preview_chars: int = 700,
    ) -> Dict[str, Any]:
        return self.justification.build_grant_context_only(
            sess=sess,
            opportunity_id=opportunity_id,
            preview_chars=preview_chars,
        )

    def build_rerank_keyword_inventory_for_opportunity(
        self,
        *,
        sess,
        opportunity_id: str,
        k: int = 10,
    ) -> Dict[str, Any]:
        return self.justification.build_rerank_keyword_inventory_for_opportunity(
            sess=sess,
            opportunity_id=opportunity_id,
            k=k,
        )

    def build_rerank_keyword_inventory_for_faculty(
        self,
        *,
        sess,
        faculty_id: int,
        k: int = 10,
    ) -> Dict[str, Any]:
        return self.justification.build_rerank_keyword_inventory_for_faculty(
            sess=sess,
            faculty_id=faculty_id,
            k=k,
        )

    # ==============================
    # Faculty Context
    # ==============================
    def build_faculty_basic_context(self, fac: Faculty) -> Dict[str, Any]:
        return self.faculty.build_faculty_context(fac, profile="basic")

    def build_faculty_keyword_context(self, fac: Faculty) -> Dict[str, Any]:
        return self.faculty.build_faculty_context(fac, profile="keyword")

    # ==============================
    # Matching Context
    # ==============================
    def build_matching_inputs_for_opportunity(
        self,
        *,
        sess,
        opportunity_id: str,
        limit_rows: int = 500,
    ) -> Tuple[List[int], Dict[str, Dict[int, float]], Dict[int, Dict[str, Dict[int, float]]]]:
        return self.matching.build_inputs_for_opportunity(
            sess=sess,
            opportunity_id=opportunity_id,
            limit_rows=limit_rows,
        )

    def build_group_matching_context(
        self,
        *,
        opp_ctx: Dict[str, Any],
        fac_ctxs: List[Dict[str, Any]],
        coverage: Any,
        member_coverages: Optional[Dict[int, Dict[str, Dict[int, float]]]] = None,
        group_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.matching.build_group_matching_context(
            opp_ctx=opp_ctx,
            fac_ctxs=fac_ctxs,
            coverage=coverage,
            member_coverages=member_coverages,
            group_meta=group_meta,
        )

    def build_top_match_payload(
        self,
        *,
        sess,
        top_rows: List[Tuple[str, float, float]],
    ) -> List[Dict[str, Any]]:
        return self.matching.build_top_match_payload(sess=sess, top_rows=top_rows)

    # ==============================
    # Recommendation Context
    # ==============================
    def build_faculty_recommendation_payloads(
        self,
        *,
        sess,
        fac: Faculty,
        top_rows: List[Tuple[str, float, float]],
    ) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
        return self.justification.build_faculty_recommendation_payloads(
            sess=sess,
            fac=fac,
            top_rows=top_rows,
        )

    def build_faculty_recommendation_source_linked_payload(
        self,
        *,
        sess,
        fac: Faculty,
        top_rows: List[Tuple[str, float, float]],
    ) -> Dict[str, Any]:
        return self.justification.build_faculty_recommendation_source_linked_payload(
            sess=sess,
            fac=fac,
            top_rows=top_rows,
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
        return self.justification.build_faculty_recommendation_source_linked_text(
            sess=sess,
            fac=fac,
            top_rows=top_rows,
            max_requirements=max_requirements,
            grant_evidence_per_requirement=grant_evidence_per_requirement,
            faculty_evidence_per_requirement=faculty_evidence_per_requirement,
        )

    def build_member_coverages_for_opportunity(
        self,
        *,
        sess,
        opportunity_id: str,
        limit_rows: int = 500,
    ) -> Dict[int, Dict[str, Dict[int, float]]]:
        return self.matching.build_member_coverages_for_opportunity(
            sess=sess,
            opportunity_id=opportunity_id,
            limit_rows=limit_rows,
        )
