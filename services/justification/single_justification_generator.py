from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from config import get_llm_client, settings
from dao.faculty_dao import FacultyDAO
from dao.match_dao import MatchDAO
from db.db_conn import SessionLocal
from db.models.faculty import Faculty
from dto.llm_response_dto import GrantExplanationOut
from services.context_retrieval.context_generator import ContextGenerator
from services.prompts.justification_prompts import (
    FACULTY_RECS_PROMPT,
    GRANT_EXPLANATION_PROMPT,
)
from utils.thread_pool import build_thread_local_getter, parallel_map, resolve_pool_size

logger = logging.getLogger(__name__)


class SingleJustificationGenerator:
    """Single-justification helper with per-match final generation.

    Supported flows:
    - Grant explanation from grant context
    - Final recommendation generation: one match == one LLM call, append all results
    """
    FINAL_JUSTIFICATION_WORKERS = 8
    GRANT_EXPLANATION_WORKERS = 8

    def __init__(self, *, context_generator: Optional[ContextGenerator] = None):
        """Inject context generator dependency (defaults to the standard ContextGenerator)."""
        self.context_generator = context_generator or ContextGenerator()

    @staticmethod
    def _norm(text: Any) -> str:
        """Normalize whitespace to keep prompt/output text compact and consistent."""
        return " ".join(str(text or "").split()).strip()

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Parse a float safely, falling back to a default on bad or missing values."""
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _sanitize_context_text_for_final_llm(text: str) -> str:
        """Strip explicit numeric score fragments from context before final justification LLM."""
        s = str(text or "")
        # Remove explicit score markers from context lines.
        s = re.sub(r"\(\s*score\s*=\s*[-+]?\d*\.?\d+\s*\)", "", s, flags=re.IGNORECASE)
        s = re.sub(r",\s*src\s*=\s*[-+]?\d*\.?\d+", "", s, flags=re.IGNORECASE)
        return s

    @staticmethod
    def _sanitize_final_justification_text(text: str) -> str:
        """Post-process final justification text to remove score wording and normalize phrasing."""
        s = " ".join(str(text or "").split()).strip()
        # Remove explicit score notations.
        s = re.sub(r"\(\s*score\s*=\s*[-+]?\d*\.?\d+\s*\)", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(?:score|scored|scoring)\b\s*[:=]?\s*[-+]?\d*\.?\d+%?", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\bat\s+[-+]?\d*\.?\d+%?\b", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\bhighest[-\s]?scoring\b", "strong", s, flags=re.IGNORECASE)
        # Replace explicit requirement wording.
        s = re.sub(r"\brequirements?\b", "priority areas", s, flags=re.IGNORECASE)
        s = re.sub(r"\balignment\b", "fit", s, flags=re.IGNORECASE)
        return " ".join(s.split()).strip()

    @staticmethod
    def _faculty_name(fac: Optional[Faculty], *, email: str) -> str:
        """Best-effort faculty display name, falling back to email when name is missing."""
        return str(getattr(fac, "name", None) or email).strip()

    @staticmethod
    def _build_grant_explanation_chain():
        """Build the LLM chain used for grant-only explanation generation."""
        model_id = (settings.sonnet or settings.haiku or settings.opus or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        return GRANT_EXPLANATION_PROMPT | llm.with_structured_output(GrantExplanationOut)

    @staticmethod
    def _build_final_justification_chain():
        """Build the free-form LLM chain for final one-match justification writing."""
        model_id = (settings.opus or settings.sonnet or settings.haiku or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        return FACULTY_RECS_PROMPT | llm

    def _generate_grant_explanation(
        self,
        *,
        opportunity_id: str,
        preview_chars: int = 50_000,
    ) -> Dict[str, Any]:
        """Generate one grant explanation from grant-context payload only."""
        logger.info(
            "JUSTIFICATION_STEP grant_explanation_start opportunity_id=%s",
            str(opportunity_id),
        )
        with SessionLocal() as sess:
            grant_context = self.context_generator.build_grant_context_only(
                sess=sess,
                opportunity_id=str(opportunity_id),
                preview_chars=int(preview_chars),
            )

        chain = self._build_grant_explanation_chain()
        grant_json = json.dumps(dict(grant_context or {}), ensure_ascii=False)
        out = chain.invoke({"grant_json": grant_json})
        if isinstance(out, GrantExplanationOut):
            explanation = self._norm(out.grant_explanation)
        elif isinstance(out, dict):
            explanation = self._norm(out.get("grant_explanation"))
        else:
            explanation = self._norm(getattr(out, "grant_explanation", None) or str(out or ""))
        logger.info(
            "JUSTIFICATION_STEP grant_explanation_done opportunity_id=%s chars=%s",
            str(opportunity_id),
            len(self._norm(explanation)),
        )
        return {
            "grant_context": grant_context,
            "grant_explanation": explanation,
        }

    def _load_top_grants_for_faculty(
        self,
        *,
        email: str,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Load stored top-k grant matches for a faculty (no reranking)."""
        target_k = max(1, int(k))
        with SessionLocal() as sess:
            fdao = FacultyDAO(sess)
            mdao = MatchDAO(sess)
            faculty_id = fdao.get_faculty_id_by_email(str(email or "").strip())
            if not faculty_id:
                raise ValueError(f"No faculty found with email: {email}")

            top_rows = mdao.top_matches_for_faculty(int(faculty_id), k=target_k)
            payload = self.context_generator.build_top_match_payload(
                sess=sess,
                top_rows=[(str(oid), float(d), float(l)) for (oid, d, l) in list(top_rows or [])],
            )

        ranked_ids: List[str] = []
        ordered_grants: List[Dict[str, Any]] = []
        for row in list(payload or []):
            merged = dict(row or {})
            oid = self._norm(merged.get("opportunity_id") or merged.get("grant_id"))
            if not oid:
                continue
            merged["opportunity_id"] = oid
            ordered_grants.append(merged)
            ranked_ids.append(oid)

        return {
            "ranked_opportunity_ids": ranked_ids,
            "grants": ordered_grants,
        }

    def _invoke_final_one_match(
        self,
        *,
        chain,
        context_text: str,
        one_match_payload: Dict[str, Any],
        opportunity_id: str,
    ) -> Dict[str, Any]:
        """Invoke final justification LLM for a single grant-faculty match payload."""
        def _fallback(text: str = "") -> Dict[str, Any]:
            return {
                "justification": str(text or "").strip() or "No match explanation generated.",
            }

        context_text = str(context_text or "")
        score_context = json.dumps(
            {
                "llm_score": self._safe_float(
                    (one_match_payload or {}).get(
                        "reranked_llm_score",
                        (one_match_payload or {}).get("llm_score"),
                    ),
                    0.0,
                )
            },
            ensure_ascii=False,
        )

        try:
            out = chain.invoke(
                {
                    "context_text": context_text,
                    "score_context": score_context,
                }
            )
        except Exception:
            logger.exception(
                "LLM_CALL_FAILED[faculty_recommendations_one_match] meta=%s",
                json.dumps({"opportunity_id": str(opportunity_id)}, ensure_ascii=False),
            )
            return _fallback()

        if isinstance(out, str):
            text = out
        elif hasattr(out, "content"):
            text = getattr(out, "content", "")
        elif isinstance(out, dict):
            text = out.get("text") or out.get("output") or json.dumps(out, ensure_ascii=False)
        else:
            text = str(out or "")
        parsed = _fallback(text=self._norm(text))

        return parsed

    def _justification_pipeline(
        self,
        *,
        email: str,
        k: int,
        preview_chars: int = 50_000,
        specific_matches: Optional[List[Dict[str, Any]]] = None,
        specific_opportunity_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full justification pipeline and return merged per-grant outputs.

        Broad stages:
        1) Build candidate grants (stored top matches, provided matches, or one specific grant).
        2) Generate grant explanations in parallel.
        3) Generate final one-match justifications in parallel.
        4) Merge and return a stable response payload for callers/scripts.
        """
        target_k = max(1, int(k))
        email_norm = str(email or "").strip()
        if not email_norm:
            raise ValueError("email is required")
        if specific_matches is not None and specific_opportunity_id is not None:
            raise ValueError("Provide either specific_matches or specific_opportunity_id, not both.")

        logger.info(
            "JUSTIFICATION_STEP combined_start email=%s k=%s mode=%s",
            email_norm,
            target_k,
            "specific_grant"
            if specific_opportunity_id is not None
            else ("specific_matches" if specific_matches is not None else "default"),
        )

        ranked_ids: List[str] = []
        ordered_grants: List[Dict[str, Any]] = []
        rerank_meta: Dict[str, Any] = {}

        # Stage 1: pick/prepare grant candidates.
        # Mode A: default path (uses stored top matches only)
        if specific_matches is None and specific_opportunity_id is None:
            rerank_out = self._load_top_grants_for_faculty(email=email_norm, k=target_k)
            ranked_ids = [
                self._norm(x) for x in list(rerank_out.get("ranked_opportunity_ids") or []) if self._norm(x)
            ][:target_k]
            reranked_grants = [dict(x or {}) for x in list(rerank_out.get("grants") or [])]
            by_id = {
                self._norm(g.get("opportunity_id")): dict(g)
                for g in reranked_grants
                if self._norm(g.get("opportunity_id"))
            }
            ordered_grants = [dict(by_id[oid]) for oid in ranked_ids if oid in by_id]
            rerank_meta = {
                "source": "stored_top_matches",
                "ranked_opportunity_ids": ranked_ids,
            }

        # Mode B: explicit provided matches
        elif specific_matches is not None:
            provided = list(specific_matches or [])[:target_k]
            for row in provided:
                oid = self._norm((row or {}).get("opportunity_id") or (row or {}).get("grant_id"))
                if not oid:
                    continue
                merged = dict(row or {})
                merged["opportunity_id"] = oid
                ordered_grants.append(merged)
                ranked_ids.append(oid)
            rerank_meta = {
                "source": "provided_matches",
                "ranked_opportunity_ids": ranked_ids,
                "updated_match_rows": 0,
            }

        # Mode C: one specific grant
        else:
            oid = self._norm(specific_opportunity_id)
            if not oid:
                raise ValueError("specific_opportunity_id is required")
            with SessionLocal() as sess:
                fdao = FacultyDAO(sess)
                mdao = MatchDAO(sess)
                faculty_id = fdao.get_faculty_id_by_email(email_norm)
                if not faculty_id:
                    raise ValueError(f"No faculty found with email: {email_norm}")
                pair_row = mdao.get_match_for_faculty_opportunity(
                    faculty_id=int(faculty_id),
                    opportunity_id=oid,
                )
                if not pair_row:
                    raise ValueError(
                        f"No match row found for faculty_id={faculty_id}, opportunity_id={oid}"
                    )
                payload = self.context_generator.build_top_match_payload(
                    sess=sess,
                    top_rows=[
                        (
                            oid,
                            self._safe_float(pair_row.get("domain_score")),
                            self._safe_float(pair_row.get("llm_score")),
                        )
                    ],
                )
            first = dict((payload or [{}])[0] or {})
            first["opportunity_id"] = oid
            ordered_grants = [first]
            ranked_ids = [oid]
            rerank_meta = {
                "source": "specific_grant",
                "ranked_opportunity_ids": ranked_ids,
                "updated_match_rows": 0,
            }

        by_id = {
            self._norm(g.get("opportunity_id")): dict(g)
            for g in ordered_grants
            if self._norm(g.get("opportunity_id"))
        }

        if not ordered_grants:
            logger.info("JUSTIFICATION_STEP combined_done no_ranked_grants")
            return {
                "email": email_norm,
                "k": target_k,
                "selection": rerank_meta,
                "results": [],
            }

        explanation_map: Dict[str, str] = {}
        expl_ids = [self._norm(x) for x in list(ranked_ids or []) if self._norm(x)]
        # Stage 2: generate grant explanations concurrently.
        if expl_ids:
            expl_pool = resolve_pool_size(
                max_workers=int(self.GRANT_EXPLANATION_WORKERS),
                task_count=len(expl_ids),
            )
            logger.info(
                "JUSTIFICATION_STEP grant_explanation_parallel_start jobs=%s workers=%s",
                len(expl_ids),
                expl_pool,
            )

            def _run_expl(oid: str) -> Dict[str, str]:
                out = self._generate_grant_explanation(
                    opportunity_id=oid,
                    preview_chars=int(preview_chars),
                )
                return {
                    "opportunity_id": oid,
                    "grant_explanation": self._norm(out.get("grant_explanation")),
                }

            def _on_expl_error(_index: int, oid: str, exc: Exception) -> Dict[str, str]:
                logger.exception(
                    "GRANT_EXPLANATION_FAILED meta=%s error=%s",
                    json.dumps({"opportunity_id": oid}, ensure_ascii=False),
                    f"{type(exc).__name__}: {exc}",
                )
                return {"opportunity_id": oid, "grant_explanation": ""}

            expl_rows = parallel_map(
                expl_ids,
                max_workers=expl_pool,
                run_item=_run_expl,
                on_error=_on_expl_error,
            )
            for row in list(expl_rows or []):
                oid = self._norm((row or {}).get("opportunity_id"))
                if not oid:
                    continue
                explanation_map[oid] = self._norm((row or {}).get("grant_explanation"))

        logger.info(
            "JUSTIFICATION_STEP combined_justification_stage_start grants=%s",
            len(ordered_grants),
        )
        # Stage 3: build one-match final justifications concurrently.
        top_rows: List[Tuple[str, float, float]] = []
        for oid in ranked_ids:
            row = dict(by_id.get(oid) or {})
            top_rows.append(
                (
                    oid,
                    self._safe_float(row.get("domain_score")),
                    self._safe_float(row.get("reranked_llm_score", row.get("llm_score"))),
                )
            )

        justifications_by_id: Dict[str, str] = {}
        with SessionLocal() as sess:
            fdao = FacultyDAO(sess)
            fac = fdao.get_with_relations_by_email(email_norm)
            if not fac:
                raise ValueError(f"No faculty found with email: {email_norm}")

            source_payload = self.context_generator.build_faculty_recommendation_source_linked_payload(
                sess=sess,
                fac=fac,
                top_rows=top_rows,
            )
            opp_payloads = [dict(x or {}) for x in list(source_payload.get("opportunity_payloads") or [])]
            opp_payload_by_id = {
                self._norm(o.get("opportunity_id")): dict(o)
                for o in list(opp_payloads or [])
                if self._norm(o.get("opportunity_id"))
            }
            jobs: List[Dict[str, Any]] = []
            for oid, domain_score, llm_score in top_rows:
                one_match_payload = dict(opp_payload_by_id.get(oid) or by_id.get(oid) or {})
                if not one_match_payload:
                    logger.info(
                        "JUSTIFICATION_STEP final_justification_skip_missing_payload opportunity_id=%s",
                        oid,
                    )
                    justifications_by_id[oid] = ""
                    continue
                one_match_text = self.context_generator.build_faculty_recommendation_source_linked_text(
                    sess=sess,
                    fac=fac,
                    top_rows=[(oid, self._safe_float(domain_score), self._safe_float(llm_score))],
                )
                jobs.append(
                    {
                        "opportunity_id": oid,
                        "context_text": self._sanitize_context_text_for_final_llm(one_match_text),
                        "payload": one_match_payload,
                    }
                )

        if jobs:
            pool_size = resolve_pool_size(
                max_workers=int(self.FINAL_JUSTIFICATION_WORKERS),
                task_count=len(jobs),
            )
            logger.info(
                "JUSTIFICATION_STEP final_justification_parallel_start jobs=%s workers=%s",
                len(jobs),
                pool_size,
            )
            get_chain = build_thread_local_getter(self._build_final_justification_chain)

            def _run_job(job: Dict[str, Any]) -> Dict[str, Any]:
                oid = self._norm(job.get("opportunity_id"))
                logger.info(
                    "JUSTIFICATION_STEP final_justification_one_match_start opportunity_id=%s",
                    oid,
                )
                parsed = self._invoke_final_one_match(
                    chain=get_chain(),
                    context_text=str(job.get("context_text") or ""),
                    one_match_payload=dict(job.get("payload") or {}),
                    opportunity_id=oid,
                )
                jtext = self._sanitize_final_justification_text(parsed.get("justification"))
                return {
                    "opportunity_id": oid,
                    "justification": self._norm(jtext),
                }

            def _on_job_error(_index: int, job: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
                oid = self._norm((job or {}).get("opportunity_id"))
                logger.exception(
                    "JUSTIFICATION_STEP final_justification_one_match_failed opportunity_id=%s error=%s",
                    oid,
                    f"{type(exc).__name__}: {exc}",
                )
                return {"opportunity_id": oid, "justification": ""}

            job_results = parallel_map(
                jobs,
                max_workers=pool_size,
                run_item=_run_job,
                on_error=_on_job_error,
            )
            for row in list(job_results or []):
                oid = self._norm((row or {}).get("opportunity_id"))
                text = self._norm((row or {}).get("justification"))
                if oid:
                    justifications_by_id[oid] = text
                logger.info(
                    "JUSTIFICATION_STEP final_justification_one_match_done opportunity_id=%s chars=%s",
                    oid,
                    len(text),
                )

        merged_results: List[Dict[str, Any]] = []
        # Stage 4: merge all artifacts into final response rows.
        for oid in ranked_ids:
            grant_row = dict(by_id.get(oid) or {})
            merged_results.append(
                {
                    "opportunity_id": oid,
                    "title": grant_row.get("title") or grant_row.get("opportunity_title"),
                    "agency": grant_row.get("agency") or grant_row.get("agency_name"),
                    "domain_score": self._safe_float(grant_row.get("domain_score")),
                    "llm_score": self._safe_float(
                        grant_row.get("reranked_llm_score", grant_row.get("llm_score"))
                    ),
                    "grant_explanation": explanation_map.get(oid, ""),
                    "justification": justifications_by_id.get(oid, ""),
                }
            )

        logger.info(
            "JUSTIFICATION_STEP combined_done results=%s",
            len(merged_results),
        )
        return {
            "email": email_norm,
            "k": target_k,
            "selection": rerank_meta,
            "results": merged_results,
        }

    def run(self, *, email: str, k: int, preview_chars: int = 50_000) -> Dict[str, Any]:
        """Public entrypoint for the default end-to-end justification pipeline."""
        return self._justification_pipeline(
            email=email,
            k=k,
            preview_chars=preview_chars,
        )

    def run_specific_matches(
        self,
        *,
        email: str,
        matches: List[Dict[str, Any]],
        k: int,
    ) -> Dict[str, Any]:
        """Public entrypoint when caller already has explicit match rows."""
        logger.info(
            "JUSTIFICATION_STEP run_specific_matches_start email=%s k=%s provided_matches=%s",
            str(email or "").strip(),
            max(1, int(k)),
            len(list(matches or [])),
        )
        return self._justification_pipeline(
            email=email,
            k=k,
            specific_matches=matches,
        )

    def run_specific_grant(self, *, email: str, opportunity_id: str) -> Dict[str, Any]:
        """Public entrypoint for one faculty + one explicitly provided grant id."""
        oid = self._norm(opportunity_id)
        if not oid:
            raise ValueError("opportunity_id is required")
        logger.info(
            "JUSTIFICATION_STEP run_specific_grant_start email=%s opportunity_id=%s",
            str(email or "").strip(),
            oid,
        )
        return self._justification_pipeline(
            email=email,
            k=1,
            specific_opportunity_id=oid,
        )
