from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import selectinload

from config import get_llm_client
from dao.match_dao import MatchDAO
from db.db_conn import SessionLocal
from db.models.faculty import Faculty
from dto.llm_response_dto import FacultyOpportunityRec, FacultyRecsOut
from services.context.context_generator import ContextGenerator
from services.prompts.justification_prompts import FACULTY_RECS_PROMPT


class SingleJustificationGenerator:
    def __init__(self, *, context_generator: Optional[ContextGenerator] = None):
        self.context_generator = context_generator or ContextGenerator()

    @staticmethod
    def _build_chain():
        llm = get_llm_client().build()
        return FACULTY_RECS_PROMPT | llm.with_structured_output(FacultyRecsOut)

    @staticmethod
    def _llm_label(llm_score: float) -> str:
        if llm_score < 0.30:
            return "sucks"
        if llm_score < 0.50:
            return "bad"
        if llm_score < 0.70:
            return "good"
        if llm_score < 0.85:
            return "great"
        return "fantastic"

    def _fallback_recommendations_from_payloads(
        self,
        *,
        faculty_name: str,
        opp_payloads: List[Dict[str, Any]],
        k: int,
    ) -> FacultyRecsOut:
        recs: List[FacultyOpportunityRec] = []
        for p in (opp_payloads or [])[: max(1, int(k))]:
            llm_score = float(p.get("llm_score") or 0.0)
            domain_score = float(p.get("domain_score") or 0.0)
            title = str(p.get("opportunity_title") or p.get("title") or "Untitled opportunity")
            agency = p.get("agency_name") or p.get("agency")
            recs.append(
                FacultyOpportunityRec(
                    opportunity_id=str(p.get("opportunity_id") or p.get("grant_id") or ""),
                    title=title,
                    agency=str(agency) if agency else None,
                    domain_score=domain_score,
                    llm_score=llm_score,
                    why_good_match=[
                        f"Fit is {self._llm_label(llm_score)} by LLM score {llm_score:.2f}; domain overlap estimate is {domain_score:.2f}.",
                        "Keyword overlap between faculty and opportunity indicates practical topical alignment.",
                        "Potential gaps likely remain; tighten scope and add a targeted collaborator to reduce execution risk.",
                    ],
                    suggested_pitch=(
                        "Frame your prior work directly against this program's scope and milestones. "
                        "Highlight one concrete deliverable and one measurable outcome in year one."
                    ),
                )
            )
        return FacultyRecsOut(
            faculty_name=faculty_name or "Unknown Faculty",
            recommendations=recs,
        )

    def _merge_llm_with_db_payloads(
        self,
        *,
        llm_recs: List[FacultyOpportunityRec],
        payloads: List[Dict[str, Any]],
        faculty_name: str,
    ) -> List[FacultyOpportunityRec]:
        fallback = self._fallback_recommendations_from_payloads(
            faculty_name=faculty_name,
            opp_payloads=payloads,
            k=len(payloads),
        ).recommendations

        out: List[FacultyOpportunityRec] = []
        for i, fb in enumerate(fallback):
            llm_rec = llm_recs[i] if i < len(llm_recs) else None
            if llm_rec is None:
                out.append(fb)
                continue
            out.append(
                fb.model_copy(
                    update={
                        "why_good_match": llm_rec.why_good_match or fb.why_good_match,
                        "suggested_pitch": llm_rec.suggested_pitch or fb.suggested_pitch,
                    }
                )
            )
        return out

    @staticmethod
    def _get_faculty_by_email(sess, *, email: str) -> Faculty:
        fac = (
            sess.query(Faculty)
            .options(selectinload(Faculty.keyword))
            .filter(Faculty.email == email)
            .one_or_none()
        )
        if not fac:
            raise ValueError(f"No faculty found with email: {email}")
        return fac

    @staticmethod
    def _invoke_llm_recommendations(
        *,
        chain,
        fac_ctx: Dict[str, Any],
        opp_payloads: List[Dict[str, Any]],
    ) -> List[FacultyOpportunityRec]:
        try:
            out: FacultyRecsOut = chain.invoke(
                {
                    "faculty_json": json.dumps(fac_ctx, ensure_ascii=False),
                    "opps_json": json.dumps(opp_payloads, ensure_ascii=False),
                }
            )
            return list(getattr(out, "recommendations", None) or [])
        except Exception:
            return []

    def _build_recommendations_from_payloads(
        self,
        *,
        chain,
        fac_ctx: Dict[str, Any],
        opp_payloads: List[Dict[str, Any]],
        faculty_name: str,
        k: int,
        batch_size: int,
    ) -> FacultyRecsOut:
        size = max(1, int(batch_size))
        target_n = min(len(opp_payloads), max(1, int(k)))
        collected: List[FacultyOpportunityRec] = []

        for start in range(0, target_n, size):
            chunk = opp_payloads[start : start + size]
            chunk_recs = self._invoke_llm_recommendations(
                chain=chain,
                fac_ctx=fac_ctx,
                opp_payloads=chunk,
            )
            merged_chunk = self._merge_llm_with_db_payloads(
                llm_recs=chunk_recs,
                payloads=chunk,
                faculty_name=faculty_name,
            )
            if merged_chunk:
                collected.extend(merged_chunk)

        if not collected:
            return self._fallback_recommendations_from_payloads(
                faculty_name=faculty_name,
                opp_payloads=opp_payloads,
                k=target_n,
            )

        return FacultyRecsOut(
            faculty_name=faculty_name,
            recommendations=collected[:target_n],
        )

    def generate_faculty_recs(self, *, email: str, k: int) -> FacultyRecsOut:
        chain = self._build_chain()

        with SessionLocal() as sess:
            match_dao = MatchDAO(sess)
            fac = self._get_faculty_by_email(sess, email=email)

            rows = match_dao.top_matches_for_faculty(
                faculty_id=fac.faculty_id,
                k=k,
            )
            opp_ids = [gid for (gid, _, _) in rows]
            if not opp_ids:
                raise ValueError(f"No matches found for {fac.name} ({email}).")

            fac_ctx, opp_payloads = self.context_generator.build_faculty_recommendation_payloads(
                sess=sess,
                fac=fac,
                top_rows=rows,
            )
            if not opp_payloads:
                raise ValueError(
                    f"Top matches exist but opportunities are missing in DB for faculty {fac.faculty_id}. "
                    "Re-fetch opportunities or rebuild match_results."
                )
            faculty_name = getattr(fac, "name", None) or email

            return self._build_recommendations_from_payloads(
                chain=chain,
                fac_ctx=fac_ctx,
                opp_payloads=opp_payloads,
                faculty_name=faculty_name,
                k=k,
                batch_size=3,
            )

    def generate_for_specific_grant(self, *, email: str, opportunity_id: str) -> FacultyRecsOut:
        """Generate one-to-one justification for exactly one opportunity."""
        opp_id = str(opportunity_id or "").strip()
        if not opp_id:
            raise ValueError("opportunity_id is required")

        chain = self._build_chain()

        with SessionLocal() as sess:
            match_dao = MatchDAO(sess)
            fac = self._get_faculty_by_email(sess, email=email)

            pair_row = match_dao.get_match_for_faculty_opportunity(
                faculty_id=int(fac.faculty_id),
                opportunity_id=opp_id,
            )
            if not pair_row:
                raise ValueError(
                    f"No match row found for faculty_id={fac.faculty_id}, opportunity_id={opp_id}"
                )

            fac_ctx, opp_payloads = self.context_generator.build_faculty_recommendation_payloads(
                sess=sess,
                fac=fac,
                top_rows=[
                    (
                        str(pair_row.get("grant_id") or opp_id),
                        float(pair_row.get("domain_score") or 0.0),
                        float(pair_row.get("llm_score") or 0.0),
                    )
                ],
            )
            if not opp_payloads:
                raise ValueError(
                    f"Match row exists but opportunity payload is missing for opportunity_id={opp_id}"
                )
            faculty_name = getattr(fac, "name", None) or email

            return self._build_recommendations_from_payloads(
                chain=chain,
                fac_ctx=fac_ctx,
                opp_payloads=opp_payloads[:1],
                faculty_name=faculty_name,
                k=1,
                batch_size=1,
            )
