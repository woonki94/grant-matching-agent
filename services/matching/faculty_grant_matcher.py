from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from dao.faculty_dao import FacultyDAO
from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from db.models.faculty import Faculty
from services.matching.single_match_llm_reranker import OneToOneLLMReranker
from services.matching.specialization_cross_encoder import SpecializationCrossEncoderScorer
from sqlalchemy.orm import selectinload
from utils.keyword_utils import extract_specializations
from utils.thread_pool import parallel_map, resolve_pool_size

logger = logging.getLogger(__name__)


class FacultyGrantMatcher:
    """One-to-one faculty↔grant matcher using cross-encoder specialization coverage."""

    DEFAULT_COVERED_THRESHOLD = 0.30
    DEFAULT_RERANK_WORKERS = 4

    def __init__(
        self,
        *,
        session_factory=SessionLocal,
        spec_scorer: Optional[SpecializationCrossEncoderScorer] = None,
        llm_reranker: Optional[OneToOneLLMReranker] = None,
    ):
        self.session_factory = session_factory
        self.spec_scorer = spec_scorer or SpecializationCrossEncoderScorer()
        self.llm_reranker = llm_reranker or OneToOneLLMReranker(session_factory=session_factory)

    @staticmethod
    def _safe_weight(value: Any) -> float:
        try:
            w = float(value)
        except Exception:
            w = 1.0
        if w < 0.0:
            return 0.0
        return w

    @classmethod
    def _resolve_covered_threshold(cls) -> float:
        raw = os.getenv("MATCH_COVERED_THRESHOLD", str(cls.DEFAULT_COVERED_THRESHOLD))
        try:
            val = float(raw)
        except Exception:
            val = cls.DEFAULT_COVERED_THRESHOLD
        if val < 0.0:
            return 0.0
        if val > 1.0:
            return 1.0
        return float(val)

    @staticmethod
    def _faculty_specs_by_section(fac: Faculty) -> Dict[str, List[str]]:
        kw = getattr(getattr(fac, "keyword", None), "keywords", {}) or {}
        specs = extract_specializations(kw)
        out: Dict[str, List[str]] = {"application": [], "research": []}
        for sec in ("application", "research"):
            for item in specs.get(sec, []):
                text = str(item.get("t") or "").strip()
                if text:
                    out[sec].append(text)
        return out

    def _opportunity_requirements_by_section(self, opp) -> Dict[str, List[Dict[str, Any]]]:
        kw = getattr(getattr(opp, "keyword", None), "keywords", {}) or {}
        specs = extract_specializations(kw)
        out: Dict[str, List[Dict[str, Any]]] = {"application": [], "research": []}
        for sec in ("application", "research"):
            for idx, item in enumerate(specs.get(sec, [])):
                text = str(item.get("t") or "").strip()
                if not text:
                    continue
                out[sec].append(
                    {
                        "idx": int(idx),
                        "text": text,
                        "weight": self._safe_weight(item.get("w", 0.0)),
                    }
                )
        return out

    def _score_specialization_coverage(
        self,
        *,
        requirements: Dict[str, List[Dict[str, Any]]],
        faculty_specs: Dict[str, List[str]],
        covered_threshold: float,
    ) -> Dict[str, Any]:
        covered: Dict[str, Dict[str, float]] = {"application": {}, "research": {}}
        missing: Dict[str, List[int]] = {"application": [], "research": []}

        evidence_sections: Dict[str, Dict[str, Dict[str, Any]]] = {"application": {}, "research": {}}
        weighted_pair_sum = 0.0
        total_pair_weight = 0.0
        pair_count = 0

        for sec in ("application", "research"):
            req_rows = list(requirements.get(sec) or [])
            fac_rows = list(faculty_specs.get(sec) or [])

            pair_scores_by_req: Dict[int, List[Tuple[int, float]]] = {}
            if req_rows and fac_rows:
                pairs: List[Tuple[str, str]] = []
                pair_index_meta: List[Tuple[int, int]] = []
                for req in req_rows:
                    req_idx = int(req["idx"])
                    req_text = str(req.get("text") or "").strip()
                    if not req_text:
                        continue
                    for fac_idx, fac_text in enumerate(fac_rows):
                        ftxt = str(fac_text or "").strip()
                        if not ftxt:
                            continue
                        pairs.append((req_text, ftxt))
                        pair_index_meta.append((req_idx, int(fac_idx)))

                pair_scores = self.spec_scorer.score_pairs(pairs)
                for (req_idx, fac_idx), score in zip(pair_index_meta, pair_scores):
                    score_val = max(0.0, min(1.0, float(score)))
                    pair_scores_by_req.setdefault(req_idx, []).append((int(fac_idx), score_val))

            for req in req_rows:
                req_idx = int(req["idx"])
                req_text = str(req.get("text") or "").strip()
                weight = self._safe_weight(req.get("weight", 1.0))
                req_pair_scores = list(pair_scores_by_req.get(req_idx) or [])

                if req_pair_scores:
                    c = sum(float(s) for _, s in req_pair_scores) / float(len(req_pair_scores))
                else:
                    c = 0.0

                if c >= float(covered_threshold):
                    covered[sec][str(req_idx)] = float(round(c, 6))
                else:
                    missing[sec].append(int(req_idx))

                if weight > 0.0 and req_pair_scores:
                    for _, score_val in req_pair_scores:
                        weighted_pair_sum += float(weight) * float(score_val)
                        total_pair_weight += float(weight)
                        pair_count += 1

                evidence_sections[sec][str(req_idx)] = {
                    "text": req_text,
                    "weight": float(round(weight, 6)),
                    "score": float(round(c, 6)),
                    "pair_count": int(len(req_pair_scores)),
                    "pair_scores": [
                        {
                            "fac_spec_idx": int(fac_idx),
                            "fac_spec": str(fac_rows[fac_idx]) if 0 <= fac_idx < len(fac_rows) else "",
                            "score": float(round(score_val, 6)),
                        }
                        for fac_idx, score_val in req_pair_scores
                    ],
                }

        spec_score = (weighted_pair_sum / total_pair_weight) if total_pair_weight > 0.0 else 0.0
        evidence = {
            "method": "cross_encoder_pairwise_weighted_average",
            "model_dir": self.spec_scorer.resolved_model_dir,
            "score_summary": {
                "weighted_pair_sum": float(round(weighted_pair_sum, 6)),
                "total_pair_weight": float(round(total_pair_weight, 6)),
                "pair_count": int(pair_count),
                "normalized_score": float(round(spec_score, 6)),
            },
            "sections": evidence_sections,
        }
        return {
            "llm_score": float(round(spec_score, 6)),
            "covered": covered,
            "missing": missing,
            "evidence": evidence,
        }

    @staticmethod
    def _build_match_row(
        *,
        grant_id: str,
        faculty_id: int,
        domain_sim: float,
        spec_score_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        cross_score = float(spec_score_payload.get("llm_score") or 0.0)
        cross_score = max(0.0, min(1.0, cross_score))

        domain_score_for_penalty = float(domain_sim)
        domain_score_for_penalty = max(0.0, min(1.0, domain_score_for_penalty))

        final_score = float(round(cross_score * domain_score_for_penalty, 6))

        evidence = dict(spec_score_payload.get("evidence") or {})
        score_summary = dict(evidence.get("score_summary") or {})
        score_summary["cross_score"] = float(round(cross_score, 6))
        score_summary["domain_similarity"] = float(round(domain_score_for_penalty, 6))
        score_summary["final_score"] = float(round(final_score, 6))
        evidence["score_summary"] = score_summary

        return {
            "grant_id": str(grant_id),
            "faculty_id": int(faculty_id),
            "domain_score": float(domain_sim),
            # Kept as llm_score column for compatibility with existing readers.
            "llm_score": final_score,
            "covered": spec_score_payload.get("covered") or {"application": {}, "research": {}},
            "missing": spec_score_payload.get("missing") or {"application": [], "research": []},
            "evidence": evidence,
        }

    def _build_rows_for_faculty_candidates(
        self,
        *,
        fac: Faculty,
        faculty_id: int,
        candidates: List[Tuple[str, float]],
        opp_map: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        faculty_specs = self._faculty_specs_by_section(fac)
        covered_threshold = self._resolve_covered_threshold()
        out_rows: List[Dict[str, Any]] = []

        for opp_id, domain_sim in candidates:
            opp = opp_map.get(str(opp_id))
            if not opp:
                continue
            reqs = self._opportunity_requirements_by_section(opp)
            spec_score_payload = self._score_specialization_coverage(
                requirements=reqs,
                faculty_specs=faculty_specs,
                covered_threshold=covered_threshold,
            )
            out_rows.append(
                self._build_match_row(
                    grant_id=str(opp_id),
                    faculty_id=int(faculty_id),
                    domain_sim=float(domain_sim),
                    spec_score_payload=spec_score_payload,
                )
            )
        return out_rows

    def _build_rows_for_opportunity_candidates(
        self,
        *,
        opportunity_id: str,
        candidates: List[Tuple[int, float]],
        fac_map: Dict[int, Faculty],
        opportunity_requirements: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        covered_threshold = self._resolve_covered_threshold()
        out_rows: List[Dict[str, Any]] = []

        for fid, domain_sim in candidates:
            fac = fac_map.get(int(fid))
            if not fac:
                continue
            faculty_specs = self._faculty_specs_by_section(fac)
            spec_score_payload = self._score_specialization_coverage(
                requirements=opportunity_requirements,
                faculty_specs=faculty_specs,
                covered_threshold=covered_threshold,
            )
            out_rows.append(
                self._build_match_row(
                    grant_id=str(opportunity_id),
                    faculty_id=int(fid),
                    domain_sim=float(domain_sim),
                    spec_score_payload=spec_score_payload,
                )
            )
        return out_rows

    @staticmethod
    def _extract_reranked_scores(rerank_result: Dict[str, Any]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for row in list((rerank_result or {}).get("reranked_grants") or []):
            oid = str((row or {}).get("opportunity_id") or "").strip()
            if not oid:
                continue
            try:
                scores[oid] = float((row or {}).get("llm_score") or 0.0)
            except Exception:
                continue
        return scores

    def _apply_reranked_scores_for_faculty(
        self,
        *,
        sess,
        match_dao: MatchDAO,
        faculty_id: int,
    ) -> int:
        try:
            rerank_result = self.llm_reranker.rerank_for_faculty(faculty_id=int(faculty_id))
        except Exception as exc:
            logger.exception(
                "LLM reranker failed for faculty_id=%s: %s",
                int(faculty_id),
                f"{type(exc).__name__}: {exc}",
            )
            return 0

        grant_scores = self._extract_reranked_scores(rerank_result)
        if not grant_scores:
            logger.info(
                "LLM reranker returned no scores for faculty_id=%s status=%s",
                int(faculty_id),
                str((rerank_result or {}).get("status") or ""),
            )
            return 0

        updated = match_dao.update_llm_scores_for_faculty(
            faculty_id=int(faculty_id),
            grant_scores=grant_scores,
        )
        logger.info(
            "Applied LLM reranked scores for faculty_id=%s updated_rows=%s",
            int(faculty_id),
            int(updated),
        )
        return int(updated)

    def _apply_reranked_scores_for_faculties(
        self,
        *,
        faculty_ids: List[int],
        workers: int = DEFAULT_RERANK_WORKERS,
    ) -> int:
        """Apply one-to-one LLM reranked scores for multiple faculty in parallel."""
        target_ids = sorted({int(x) for x in list(faculty_ids or []) if x is not None})
        if not target_ids:
            return 0

        pool_size = resolve_pool_size(max_workers=int(workers or 0), task_count=len(target_ids))
        logger.info(
            "Rerank batch start faculty_targets=%s workers=%s",
            len(target_ids),
            int(pool_size),
        )

        def _run_one(fid: int) -> Dict[str, Any]:
            with self.session_factory() as sess:
                match_dao = MatchDAO(sess)
                updated = self._apply_reranked_scores_for_faculty(
                    sess=sess,
                    match_dao=match_dao,
                    faculty_id=int(fid),
                )
                sess.commit()
                return {
                    "faculty_id": int(fid),
                    "updated_rows": int(updated),
                    "status": "done",
                }

        def _on_error(_idx: int, fid: int, exc: Exception) -> Dict[str, Any]:
            logger.exception(
                "Rerank batch failed for faculty_id=%s: %s",
                int(fid),
                f"{type(exc).__name__}: {exc}",
            )
            return {
                "faculty_id": int(fid),
                "updated_rows": 0,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

        outputs = parallel_map(
            target_ids,
            max_workers=pool_size,
            run_item=_run_one,
            on_error=_on_error,
        )
        total_updated = sum(int((row or {}).get("updated_rows") or 0) for row in list(outputs or []))
        logger.info(
            "Rerank batch done faculty_targets=%s updated_rows=%s",
            len(target_ids),
            int(total_updated),
        )
        return int(total_updated)

    def run(
        self,
        *,
        k: int = 10,
        min_domain: float = 0.30,
        limit_faculty: int = 100,
        commit_every: int = 30,
        rerank_workers: int = DEFAULT_RERANK_WORKERS,
    ) -> int:
        with self.session_factory() as sess:
            fac_dao = FacultyDAO(sess)
            opp_dao = OpportunityDAO(sess)
            match_dao = MatchDAO(sess)

            faculty_iter = fac_dao.iter_faculty_with_relations(stream=False)
            processed = 0
            rerank_faculty_ids: List[int] = []

            for fac in faculty_iter:
                if limit_faculty and limit_faculty > 0 and processed >= limit_faculty:
                    break

                candidates = match_dao.topk_opps_for_faculty(faculty_id=fac.faculty_id, k=k)
                candidates = [(oid, s) for (oid, s) in candidates if float(s) >= float(min_domain)]
                if not candidates:
                    processed += 1
                    continue

                opp_ids = [opp_id for opp_id, _ in candidates]
                opps = opp_dao.read_opportunities_by_ids_with_relations(opp_ids)
                opp_map = {o.opportunity_id: o for o in opps}

                out_rows = self._build_rows_for_faculty_candidates(
                    fac=fac,
                    faculty_id=int(fac.faculty_id),
                    candidates=candidates,
                    opp_map=opp_map,
                )

                if out_rows:
                    match_dao.upsert_matches(out_rows)
                    rerank_faculty_ids.append(int(fac.faculty_id))

                processed += 1
                if commit_every and processed % commit_every == 0:
                    sess.commit()
                    logger.info("Committed after %d faculty processed", processed)

            sess.commit()
            self._apply_reranked_scores_for_faculties(
                faculty_ids=rerank_faculty_ids,
                workers=int(rerank_workers),
            )
            logger.info("Faculty-grant matching completed. Total faculty processed: %d", processed)
            return processed

    def run_for_faculty(
        self,
        *,
        faculty_id: int,
        k: int = 10,
        min_domain: float = 0.30,
    ) -> int:
        """
        Generate one-to-one match rows for exactly one faculty.
        Returns number of upserted match rows.
        """
        if not faculty_id:
            return 0

        with self.session_factory() as sess:
            opp_dao = OpportunityDAO(sess)
            match_dao = MatchDAO(sess)
            fac = sess.get(Faculty, int(faculty_id))
            if not fac:
                return 0

            candidates = match_dao.topk_opps_for_faculty(faculty_id=int(faculty_id), k=int(k))
            candidates = [(oid, s) for (oid, s) in candidates if float(s) >= float(min_domain)]
            if not candidates:
                return 0

            opp_ids = [opp_id for opp_id, _ in candidates]
            opps = opp_dao.read_opportunities_by_ids_with_relations(opp_ids)
            opp_map = {o.opportunity_id: o for o in opps}

            out_rows = self._build_rows_for_faculty_candidates(
                fac=fac,
                faculty_id=int(faculty_id),
                candidates=candidates,
                opp_map=opp_map,
            )

            if out_rows:
                match_dao.upsert_matches(out_rows)
                # Commit first so reranker (separate session) can read the fresh rows.
                sess.commit()
                self._apply_reranked_scores_for_faculty(
                    sess=sess,
                    match_dao=match_dao,
                    faculty_id=int(faculty_id),
                )
                sess.commit()

            return len(out_rows)

    def run_for_opportunity(
        self,
        *,
        opportunity_id: str,
        faculty_ids: Optional[List[int]] = None,
        k: int = 200,
        min_domain: float = 0.30,
        rerank_workers: int = DEFAULT_RERANK_WORKERS,
    ) -> int:
        """
        Generate one-to-one match rows for exactly one grant.
        If faculty_ids is provided, compute only for those faculty.
        Otherwise compute for top-k faculty by embedding similarity.
        Returns number of upserted match rows.
        """
        if not opportunity_id:
            return 0

        with self.session_factory() as sess:
            opp_dao = OpportunityDAO(sess)
            match_dao = MatchDAO(sess)

            opps = opp_dao.read_opportunities_by_ids_with_relations([str(opportunity_id)])
            if not opps:
                return 0
            opp = opps[0]
            opportunity_requirements = self._opportunity_requirements_by_section(opp)

            candidates: List[Tuple[int, float]] = []
            if faculty_ids:
                for fid in sorted({int(x) for x in faculty_ids if x is not None}):
                    sim = match_dao.domain_similarity_for_faculty_opportunity(
                        faculty_id=fid,
                        opportunity_id=str(opportunity_id),
                    )
                    if sim is None:
                        continue
                    if float(sim) >= float(min_domain):
                        candidates.append((fid, float(sim)))
            else:
                cands = match_dao.topk_faculties_for_opportunity(
                    opportunity_id=str(opportunity_id),
                    k=max(int(k), 1),
                )
                candidates = [(fid, sim) for (fid, sim) in cands if float(sim) >= float(min_domain)]

            if not candidates:
                return 0

            fac_ids = [fid for (fid, _) in candidates]
            fac_rows = (
                sess.query(Faculty)
                .options(selectinload(Faculty.keyword))
                .filter(Faculty.faculty_id.in_(fac_ids))
                .all()
            )
            fac_map = {int(f.faculty_id): f for f in fac_rows}

            out_rows = self._build_rows_for_opportunity_candidates(
                opportunity_id=str(opportunity_id),
                candidates=candidates,
                fac_map=fac_map,
                opportunity_requirements=opportunity_requirements,
            )

            if out_rows:
                match_dao.upsert_matches(out_rows)
                # Commit first so reranker (separate session) can read the fresh rows.
                sess.commit()
                rerank_faculty_ids = sorted({int(row.get("faculty_id")) for row in out_rows if row.get("faculty_id") is not None})
                self._apply_reranked_scores_for_faculties(
                    faculty_ids=rerank_faculty_ids,
                    workers=int(rerank_workers),
                )
            return len(out_rows)
