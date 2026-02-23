from __future__ import annotations

from typing import Dict, List, Tuple

from dao.opportunity_dao import OpportunityDAO
from db.models import Faculty
from services.context.faculty_context import FacultyContextBuilder
from services.context.opportunity_context import OpportunityContextBuilder


class JustificationContextBuilder:
    """Build minimal payloads used by faculty recommendation justification."""

    def __init__(
        self,
        *,
        faculty_builder: FacultyContextBuilder | None = None,
        opportunity_builder: OpportunityContextBuilder | None = None,
    ):
        self.faculty = faculty_builder or FacultyContextBuilder()
        self.opportunity = opportunity_builder or OpportunityContextBuilder()

    def build_faculty_recommendation_payloads(
        self,
        *,
        sess,
        fac: Faculty,
        top_rows: List[Tuple[str, float, float]],
    ) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
        fac_ctx = self.faculty.build_faculty_keyword_context(fac)

        opp_ids = [gid for (gid, _, _) in (top_rows or [])]
        score_map = {
            gid: {"domain_score": domain_score, "llm_score": llm_score}
            for (gid, domain_score, llm_score) in (top_rows or [])
        }
        if not opp_ids:
            return fac_ctx, []

        opp_dao = OpportunityDAO(sess)
        opps = opp_dao.read_opportunities_by_ids_for_keyword_context(opp_ids)
        opp_map = {o.opportunity_id: o for o in opps}

        opp_payloads: List[Dict[str, object]] = []
        for oid in opp_ids:
            opp = opp_map.get(oid)
            if not opp:
                continue
            opp_ctx = self.opportunity.build_opportunity_keyword_context(opp)
            scores = score_map.get(oid, {"domain_score": None, "llm_score": None})
            opp_payloads.append(
                {
                    "opportunity_id": opp_ctx.get("opportunity_id") or oid,
                    "opportunity_title": opp_ctx.get("opportunity_title"),
                    "agency_name": opp_ctx.get("agency_name"),
                    "opportunity_link": opp_ctx.get("opportunity_link"),
                    "keywords": opp_ctx.get("keywords") or {},
                    "domain_score": float(scores["domain_score"] or 0.0),
                    "llm_score": float(scores["llm_score"] or 0.0),
                }
            )

        return fac_ctx, opp_payloads
