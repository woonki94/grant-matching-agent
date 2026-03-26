from __future__ import annotations

from typing import Any, Dict, List, Tuple

from dao.opportunity_dao import OpportunityDAO
from db.models import Faculty
from services.context_retrieval.faculty_context import FacultyContextBuilder
from services.context_retrieval.opportunity_context import OpportunityContextBuilder


class JustificationContextBuilder:
    PROFILE_FIELDS: Dict[str, Tuple[str, ...]] = {
        "faculty_recommendation_faculty": (
            "faculty_id",
            "name",
            "email",
            "profile_url",
            "keywords",
        ),
        "faculty_recommendation_opportunity": (
            "opportunity_id",
            "opportunity_title",
            "agency_name",
            "opportunity_link",
            "keywords",
            "domain_score",
            "llm_score",
        ),
    }

    def __init__(
        self,
        *,
        faculty_builder: FacultyContextBuilder | None = None,
        opportunity_builder: OpportunityContextBuilder | None = None,
    ):
        self.faculty = faculty_builder or FacultyContextBuilder()
        self.opportunity = opportunity_builder or OpportunityContextBuilder()

    @staticmethod
    def _select_fields(payload: Dict[str, Any], fields: Tuple[str, ...]) -> Dict[str, Any]:
        return {k: payload.get(k) for k in fields}

    def build_justification_retrievable_context(
        self,
        *,
        sess,
        fac: Faculty,
        top_rows: List[Tuple[str, float, float]],
    ) -> Dict[str, Any]:
        fac_ctx = self.faculty.build_faculty_context(fac, profile="keyword")

        opp_ids = [gid for (gid, _, _) in (top_rows or [])]
        score_map = {
            gid: {"domain_score": domain_score, "llm_score": llm_score}
            for (gid, domain_score, llm_score) in (top_rows or [])
        }
        if not opp_ids:
            return {
                "faculty": fac_ctx,
                "opportunities": [],
            }

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
            payload: Dict[str, object] = {
                "opportunity_id": opp_ctx.get("opportunity_id") or oid,
                "opportunity_title": opp_ctx.get("opportunity_title"),
                "agency_name": opp_ctx.get("agency_name"),
                "opportunity_link": opp_ctx.get("opportunity_link"),
                "keywords": opp_ctx.get("keywords") or {},
                "domain_score": float(scores["domain_score"] or 0.0),
                "llm_score": float(scores["llm_score"] or 0.0),
            }
            opp_payloads.append(payload)

        return {
            "faculty": fac_ctx,
            "opportunities": opp_payloads,
        }

    def build_justification_context(
        self,
        *,
        profile: str,
        sess,
        fac: Faculty,
        top_rows: List[Tuple[str, float, float]],
    ) -> Dict[str, Any]:
        normalized = str(profile or "").strip().lower()
        faculty_fields = self.PROFILE_FIELDS.get(f"{normalized}_faculty")
        opportunity_fields = self.PROFILE_FIELDS.get(f"{normalized}_opportunity")
        if not faculty_fields or not opportunity_fields:
            raise ValueError(f"Unsupported justification context profile: {profile}")
        full = self.build_justification_retrievable_context(
            sess=sess,
            fac=fac,
            top_rows=top_rows,
        )
        return {
            "faculty": self._select_fields(dict(full.get("faculty") or {}), faculty_fields),
            "opportunities": [
                self._select_fields(dict(payload or {}), opportunity_fields)
                for payload in list(full.get("opportunities") or [])
            ],
        }

    def build_faculty_recommendation_payloads(
        self,
        *,
        sess,
        fac: Faculty,
        top_rows: List[Tuple[str, float, float]],
    ) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
        ctx = self.build_justification_context(
            profile="faculty_recommendation",
            sess=sess,
            fac=fac,
            top_rows=top_rows,
        )
        return (
            dict(ctx.get("faculty") or {}),
            list(ctx.get("opportunities") or []),
        )
