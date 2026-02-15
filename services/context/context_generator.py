from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from db.models import Faculty, Opportunity
from services.context.faculty_context import FacultyContextBuilder
from services.context.matching_context import MatchingContextBuilder
from services.context.opportunity_context import OpportunityContextBuilder


class ContextGenerator:
    """Facade that composes faculty and grant context builders."""

    def __init__(self):
        self.faculty = FacultyContextBuilder()
        self.grant = OpportunityContextBuilder()
        self.matching = MatchingContextBuilder()

    def build_opportunity_basic_context(self, opp: Opportunity) -> Dict[str, Any]:
        return self.grant.build_opportunity_basic_context(opp)

    def build_faculty_basic_context(self, fac: Faculty) -> Dict[str, Any]:
        return self.faculty.build_faculty_basic_context(fac)

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
