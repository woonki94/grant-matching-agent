from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from config import get_llm_client
from dao.faculty_dao import FacultyDAO
from dao.match_dao import MatchDAO
from db.db_conn import SessionLocal
from db.models.faculty import Faculty
from dto.llm_response_dto import FacultyOpportunityRec, FacultyRecsOut, GrantExplanationOut, WhyMatchOut
from services.context_retrieval.context_generator import ContextGenerator
from services.prompts.justification_prompts import FACULTY_RECS_PROMPT, GRANT_EXPLANATION_PROMPT
from utils.thread_pool import build_thread_local_getter, parallel_map, resolve_pool_size


class SingleJustificationGenerator:
    DEFAULT_RECOMMENDATION_WORKERS = 4

    def __init__(self, *, context_generator: Optional[ContextGenerator] = None):
        self.context_generator = context_generator or ContextGenerator()

    @staticmethod
    def _build_chain():
        llm = get_llm_client().build()
        return FACULTY_RECS_PROMPT | llm.with_structured_output(FacultyRecsOut)

    @staticmethod
    def _build_grant_explanation_chain():
        llm = get_llm_client().build()
        return GRANT_EXPLANATION_PROMPT | llm.with_structured_output(GrantExplanationOut)

    @staticmethod
    def _llm_label(llm_score: float) -> str:
        if llm_score < 0.15:
            return "mismatch"
        if llm_score < 0.50:
            return "bad"
        if llm_score < 0.70:
            return "good"
        if llm_score < 0.85:
            return "great"
        return "fantastic"

    @staticmethod
    def _first_sentence(text: Any) -> str:
        s = " ".join(str(text or "").split()).strip()
        if not s:
            return ""
        m = re.search(r"[.!?]", s)
        if not m:
            return s
        return s[: m.end()].strip()

    @staticmethod
    def _clean_grant_explanation(text: Any) -> str:
        s = str(text or "").replace("\u2026", "...")
        s = re.sub(r"\.\s*\.\s*\.", ".", s)
        s = re.sub(r"\.{2,}", ".", s)
        s = " ".join(s.split()).strip()
        return s

    def _fallback_grant_explanation_from_context(
        self,
        *,
        grant_context: Optional[Dict[str, Any]],
    ) -> str:
        ctx = dict(grant_context or {})
        if not ctx:
            return ""

        title = str(ctx.get("opportunity_title") or ctx.get("title") or "Grant opportunity").strip()
        agency = str(ctx.get("agency_name") or ctx.get("agency") or "").strip()
        summary = self._first_sentence(ctx.get("summary_description"))

        keywords = ctx.get("keywords") if isinstance(ctx.get("keywords"), dict) else {}
        r = (keywords.get("research") or {}) if isinstance(keywords, dict) else {}
        a = (keywords.get("application") or {}) if isinstance(keywords, dict) else {}
        themes: List[str] = []
        for v in list((r.get("domain") or [])) + list((a.get("domain") or [])):
            s = str(v).strip()
            if not s or s in themes:
                continue
            themes.append(s)
            if len(themes) >= 3:
                break

        if title and agency:
            lead = f"{title} from {agency} supports targeted research and development priorities."
        elif title:
            lead = f"{title} supports targeted research and development priorities."
        else:
            lead = "This opportunity supports targeted research and development priorities."

        parts: List[str] = [lead]
        if summary:
            parts.append(summary)
        if themes:
            parts.append(f"Priority themes include {', '.join(themes)}.")

        return self._clean_grant_explanation(" ".join(parts))

    def _build_grant_explanation(
        self,
        *,
        chain,
        grant_context: Optional[Dict[str, Any]],
    ) -> str:
        ctx = dict(grant_context or {})
        if not ctx:
            return ""
        try:
            out: GrantExplanationOut = chain.invoke(
                {"grant_json": json.dumps(ctx, ensure_ascii=False)}
            )
            text = str(getattr(out, "grant_explanation", "") or "").strip()
            if text:
                cleaned = self._clean_grant_explanation(text)
                if cleaned:
                    return cleaned
        except Exception:
            pass
        return self._clean_grant_explanation(
            self._fallback_grant_explanation_from_context(grant_context=ctx)
        )

    @staticmethod
    def _top_grant_explanation_from_recs(recs: List[FacultyOpportunityRec]) -> str:
        if not recs:
            return ""
        text = str(getattr(recs[0], "grant_explanation", "") or "").strip()
        return text

    def _build_grant_explanations_for_payloads(
        self,
        *,
        opp_payloads: List[Dict[str, Any]],
        grant_explanation_contexts: Dict[str, Dict[str, Any]],
    ) -> Dict[str, str]:
        ordered_ids: List[str] = []
        seen = set()
        for p in opp_payloads or []:
            oid = str(p.get("opportunity_id") or p.get("grant_id") or "").strip()
            if not oid or oid in seen:
                continue
            seen.add(oid)
            ordered_ids.append(oid)
        if not ordered_ids:
            return {}

        raw_workers = os.getenv("SINGLE_JUST_EXPL_WORKERS", "4")
        try:
            configured_workers = int(raw_workers)
        except Exception:
            configured_workers = 4
        get_chain = build_thread_local_getter(self._build_grant_explanation_chain)

        def _run_one(oid: str) -> Tuple[str, str]:
            return (
                oid,
                self._build_grant_explanation(
                    chain=get_chain(),
                    grant_context=(grant_explanation_contexts or {}).get(oid) or {},
                ),
            )

        pairs = parallel_map(
            ordered_ids,
            max_workers=resolve_pool_size(
                max_workers=configured_workers,
                task_count=len(ordered_ids),
            ),
            run_item=_run_one,
        )
        out = {oid: text for oid, text in pairs}
        return out

    @staticmethod
    def _attach_grant_explanations_to_recs(
        recs: List[FacultyOpportunityRec],
        *,
        grant_explanations_by_opp: Dict[str, str],
    ) -> List[FacultyOpportunityRec]:
        out: List[FacultyOpportunityRec] = []
        for rec in recs or []:
            oid = str(getattr(rec, "opportunity_id", "") or "").strip()
            text = str((grant_explanations_by_opp or {}).get(oid) or "").strip()
            out.append(rec.model_copy(update={"grant_explanation": text}))
        return out

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
            if llm_score < 0.15:
                recs.append(
                    FacultyOpportunityRec(
                        opportunity_id=str(p.get("opportunity_id") or p.get("grant_id") or ""),
                        title=title,
                        agency=str(agency) if agency else None,
                        domain_score=domain_score,
                        llm_score=llm_score,
                        fit_label="mismatch",
                        why_match=WhyMatchOut(
                            summary="No match found.",
                            alignment_points=[],
                            risk_gaps=[
                                f"LLM score {llm_score:.2f} indicates a fundamental topical mismatch.",
                                f"Domain overlap estimate is {domain_score:.2f}, which is insufficient for pursuit.",
                            ],
                        ),
                        suggested_pitch="Do not pursue.",
                    )
                )
                continue
            recs.append(
                FacultyOpportunityRec(
                    opportunity_id=str(p.get("opportunity_id") or p.get("grant_id") or ""),
                    title=title,
                    agency=str(agency) if agency else None,
                    domain_score=domain_score,
                    llm_score=llm_score,
                    fit_label=self._llm_label(llm_score),
                    why_match=WhyMatchOut(
                        summary=(
                            f"Fit is {self._llm_label(llm_score)} by LLM score {llm_score:.2f}; "
                            f"domain overlap estimate is {domain_score:.2f}."
                        ),
                        alignment_points=[
                            "Keyword overlap between faculty and opportunity indicates practical topical alignment.",
                        ],
                        risk_gaps=[
                            "Potential gaps likely remain; tighten scope and add a targeted collaborator to reduce execution risk.",
                        ],
                    ),
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
            fit_label = llm_rec.fit_label or fb.fit_label
            why_match = llm_rec.why_match
            if not (
                why_match.summary.strip()
                or list(why_match.alignment_points or [])
                or list(why_match.risk_gaps or [])
            ):
                why_match = fb.why_match
            suggested_pitch = (llm_rec.suggested_pitch or "").strip() or fb.suggested_pitch
            if fit_label == "mismatch":
                suggested_pitch = "Do not pursue."
                if not (why_match.summary or "").strip():
                    why_match = why_match.model_copy(update={"summary": "No match found."})
            out.append(
                fb.model_copy(
                    update={
                        "fit_label": fit_label,
                        "why_match": why_match,
                        "suggested_pitch": suggested_pitch,
                    }
                )
            )
        return out

    @staticmethod
    def _get_faculty_by_email(faculty_dao: FacultyDAO, *, email: str) -> Faculty:
        fac = faculty_dao.get_with_relations_by_email(email)
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
        grant_explanation_contexts: Dict[str, Dict[str, Any]],
        faculty_name: str,
        k: int,
        batch_size: int,
    ) -> FacultyRecsOut:
        size = max(1, int(batch_size))
        target_n = min(len(opp_payloads), max(1, int(k)))
        active_payloads = list(opp_payloads or [])[:target_n]
        chunks: List[Tuple[int, List[Dict[str, Any]]]] = [
            (chunk_idx, active_payloads[start : start + size])
            for chunk_idx, start in enumerate(range(0, target_n, size))
        ]

        raw_workers = os.getenv(
            "SINGLE_JUST_WORKERS",
            str(self.DEFAULT_RECOMMENDATION_WORKERS),
        )
        try:
            configured_workers = int(raw_workers)
        except Exception:
            configured_workers = self.DEFAULT_RECOMMENDATION_WORKERS
        workers = resolve_pool_size(
            max_workers=configured_workers,
            task_count=len(chunks),
        )
        get_chain = build_thread_local_getter(self._build_chain)

        def _run_chunk(
            task: Tuple[int, List[Dict[str, Any]]],
        ) -> Tuple[List[FacultyOpportunityRec], Dict[str, str]]:
            _, chunk = task

            def _run_stage(stage: str) -> Tuple[str, Any]:
                try:
                    if stage == "recs":
                        llm_recs = self._invoke_llm_recommendations(
                            chain=(chain if workers <= 1 else get_chain()),
                            fac_ctx=fac_ctx,
                            opp_payloads=chunk,
                        )
                        return (
                            "recs",
                            self._merge_llm_with_db_payloads(
                                llm_recs=llm_recs,
                                payloads=chunk,
                                faculty_name=faculty_name,
                            ),
                        )
                    return (
                        "expl",
                        self._build_grant_explanations_for_payloads(
                            opp_payloads=chunk,
                            grant_explanation_contexts=grant_explanation_contexts or {},
                        ),
                    )
                except Exception:
                    if stage == "recs":
                        return (
                            "recs",
                            self._merge_llm_with_db_payloads(
                                llm_recs=[],
                                payloads=chunk,
                                faculty_name=faculty_name,
                            ),
                        )
                    return "expl", {}

            stage_outputs = dict(
                parallel_map(
                    ["recs", "expl"],
                    max_workers=2,
                    run_item=_run_stage,
                )
            )
            recs = list(stage_outputs.get("recs") or [])
            expl = dict(stage_outputs.get("expl") or {})
            return recs, expl

        chunk_outputs = parallel_map(
            chunks,
            max_workers=workers,
            run_item=_run_chunk,
        )
        collected: List[FacultyOpportunityRec] = []
        grant_explanations_by_opp: Dict[str, str] = {}
        for merged, chunk_expl in chunk_outputs:
            if merged:
                collected.extend(merged)
            if chunk_expl:
                grant_explanations_by_opp.update(chunk_expl)

        if collected:
            collected = self._attach_grant_explanations_to_recs(
                list(collected[:target_n]),
                grant_explanations_by_opp=grant_explanations_by_opp,
            )
            return FacultyRecsOut(
                faculty_name=faculty_name,
                grant_explanation=self._top_grant_explanation_from_recs(collected),
                recommendations=collected,
            )

        fallback_out = self._fallback_recommendations_from_payloads(
            faculty_name=faculty_name,
            opp_payloads=active_payloads,
            k=target_n,
        )
        with_expl = self._attach_grant_explanations_to_recs(
            list(fallback_out.recommendations or []),
            grant_explanations_by_opp=grant_explanations_by_opp,
        )
        return fallback_out.model_copy(
            update={
                "grant_explanation": self._top_grant_explanation_from_recs(with_expl),
                "recommendations": with_expl,
            }
        )

    def _generate_faculty_recs_from_top_rows(
        self,
        *,
        sess,
        email: str,
        fac: Faculty,
        top_rows: List[Tuple[str, float, float]],
        k: int,
        missing_payload_error: str,
    ) -> FacultyRecsOut:
        chain = self._build_chain()
        fac_ctx, opp_payloads = self.context_generator.build_faculty_recommendation_payloads(
            sess=sess,
            fac=fac,
            top_rows=top_rows,
        )
        if not opp_payloads:
            raise ValueError(missing_payload_error)

        faculty_name = getattr(fac, "name", None) or email
        grant_explanation_contexts = self.context_generator.build_opportunity_explanation_contexts(
            sess=sess,
            opportunity_ids=[
                str(p.get("opportunity_id") or p.get("grant_id") or "").strip()
                for p in (opp_payloads or [])
            ],
        )

        return self._build_recommendations_from_payloads(
            chain=chain,
            fac_ctx=fac_ctx,
            opp_payloads=opp_payloads,
            grant_explanation_contexts=grant_explanation_contexts,
            faculty_name=faculty_name,
            k=k,
            batch_size=3,
        )

    def run(self, *, email: str, k: int) -> FacultyRecsOut:
        with SessionLocal() as sess:
            match_dao = MatchDAO(sess)
            faculty_dao = FacultyDAO(sess)
            fac = self._get_faculty_by_email(faculty_dao, email=email)

            rows = match_dao.top_matches_for_faculty(
                faculty_id=fac.faculty_id,
                k=k,
            )
            if not rows:
                raise ValueError(f"No matches found for {fac.name} ({email}).")

            return self._generate_faculty_recs_from_top_rows(
                sess=sess,
                email=email,
                fac=fac,
                top_rows=list(rows),
                k=k,
                missing_payload_error=(
                    f"Top matches exist but opportunities are missing in DB for faculty {fac.faculty_id}. "
                    "Re-fetch opportunities or rebuild match_results."
                ),
            )

    def run_specific_matches(
        self,
        *,
        email: str,
        matches: List[Dict[str, Any]],
        k: int,
    ) -> FacultyRecsOut:
        """Generate recommendations for an explicit ordered match list (already filtered/ranked)."""
        target_k = max(1, int(k))
        top_matches = list(matches or [])[:target_k]
        if not top_matches:
            raise ValueError("No matches provided for recommendation generation.")

        with SessionLocal() as sess:
            faculty_dao = FacultyDAO(sess)
            fac = self._get_faculty_by_email(faculty_dao, email=email)
            top_rows = []
            for m in top_matches:
                opp_id = str(m.get("opportunity_id") or m.get("grant_id") or "").strip()
                if not opp_id:
                    continue
                top_rows.append(
                    (
                        opp_id,
                        float(m.get("domain_score") or 0.0),
                        float(m.get("llm_score") or 0.0),
                    )
                )

            if not top_rows:
                raise ValueError("No valid opportunity IDs found in provided matches.")

            return self._generate_faculty_recs_from_top_rows(
                sess=sess,
                email=email,
                fac=fac,
                top_rows=top_rows,
                k=target_k,
                missing_payload_error=(
                    "Match rows exist but opportunity payloads are missing in DB for recommendation generation."
                ),
            )

    def run_specific_grant(self, *, email: str, opportunity_id: str) -> FacultyRecsOut:
        """Generate one-to-one justification for exactly one opportunity."""
        opp_id = str(opportunity_id or "").strip()
        if not opp_id:
            raise ValueError("opportunity_id is required")

        chain = self._build_chain()

        with SessionLocal() as sess:
            match_dao = MatchDAO(sess)
            faculty_dao = FacultyDAO(sess)
            fac = self._get_faculty_by_email(faculty_dao, email=email)

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
            grant_explanation_contexts = self.context_generator.build_opportunity_explanation_contexts(
                sess=sess,
                opportunity_ids=[opp_id],
            )

            return self._build_recommendations_from_payloads(
                chain=chain,
                fac_ctx=fac_ctx,
                opp_payloads=opp_payloads[:1],
                grant_explanation_contexts=grant_explanation_contexts,
                faculty_name=faculty_name,
                k=1,
                batch_size=1,
            )
