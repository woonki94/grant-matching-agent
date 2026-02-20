from __future__ import annotations

import argparse
import json
import logging
from typing import List, Optional

from config import get_llm_client
from dao.faculty_dao import FacultyDAO
from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from db.models.faculty import Faculty
from sqlalchemy.orm import selectinload
from dto.llm_response_dto import LLMMatchOut, MissingItem, ScoredCoveredItem
from services.prompts.matching_prompt import MATCH_PROMPT
from utils.keyword_utils import keywords_for_matching, requirements_indexed

logger = logging.getLogger(__name__)



class FacultyGrantMatcher:
    """One-to-one faculty↔grant matching service using vector retrieval + LLM scoring."""

    def __init__(self, *, session_factory=SessionLocal):
        self.session_factory = session_factory

    @staticmethod
    def _build_chain():
        llm = get_llm_client().build()
        return MATCH_PROMPT | llm.with_structured_output(LLMMatchOut)

    @staticmethod
    def _covered_to_grouped(items: List[ScoredCoveredItem]):
        out = {"application": {}, "research": {}}
        for it in items or []:
            sec = it.section
            idx = str(int(it.idx))
            c = float(it.c)
            prev = out[sec].get(idx)
            out[sec][idx] = c if prev is None else max(prev, c)
        return out

    @staticmethod
    def _missing_to_grouped(items: List[MissingItem]):
        out = {"application": [], "research": []}
        for it in items or []:
            out[it.section].append(int(it.idx))
        for sec in out:
            seen = set()
            out[sec] = [x for x in out[sec] if not (x in seen or seen.add(x))]
        return out

    @staticmethod
    def _faculty_keywords_json(fac: Faculty) -> str:
        fac_kw = keywords_for_matching(getattr(fac.keyword, "keywords", {}) or {})
        return json.dumps(fac_kw, ensure_ascii=False)

    @staticmethod
    def _opportunity_requirements_json(opp) -> str:
        opp_kw = keywords_for_matching(getattr(opp.keyword, "keywords", {}) or {})
        req_idx = requirements_indexed(opp_kw)
        return json.dumps(req_idx, ensure_ascii=False)

    @staticmethod
    def _score_pair(
        *,
        chain,
        fac_json: str,
        opp_req_idx_json: str,
    ) -> LLMMatchOut:
        return chain.invoke(
            {
                "faculty_kw_json": fac_json,
                "requirements_indexed": opp_req_idx_json,
            }
        )

    def _build_match_row(
        self,
        *,
        grant_id: str,
        faculty_id: int,
        domain_sim: float,
        scored: LLMMatchOut,
    ) -> dict:
        return {
            "grant_id": str(grant_id),
            "faculty_id": int(faculty_id),
            "domain_score": float(domain_sim),
            "llm_score": float(scored.llm_score),
            "reason": (scored.reason or "").strip(),
            "covered": self._covered_to_grouped(scored.covered),
            "missing": self._missing_to_grouped(scored.missing),
        }

    def _build_rows_for_faculty_candidates(
        self,
        *,
        chain,
        fac: Faculty,
        faculty_id: int,
        candidates: List[tuple[str, float]],
        opp_map: dict,
    ) -> List[dict]:
        fac_json = self._faculty_keywords_json(fac)
        opp_req_cache: dict[str, str] = {}
        out_rows: List[dict] = []

        for opp_id, domain_sim in candidates:
            opp = opp_map.get(opp_id)
            if not opp:
                continue
            if opp_id not in opp_req_cache:
                opp_req_cache[opp_id] = self._opportunity_requirements_json(opp)
            scored = self._score_pair(
                chain=chain,
                fac_json=fac_json,
                opp_req_idx_json=opp_req_cache[opp_id],
            )
            out_rows.append(
                self._build_match_row(
                    grant_id=opp_id,
                    faculty_id=faculty_id,
                    domain_sim=domain_sim,
                    scored=scored,
                )
            )

        return out_rows

    def run(
        self,
        *,
        k: int = 10,
        min_domain: float = 0.30,
        limit_faculty: int = 100,
        commit_every: int = 30,
    ) -> int:
        chain = self._build_chain()

        with self.session_factory() as sess:
            fac_dao = FacultyDAO(sess)
            opp_dao = OpportunityDAO(sess)
            match_dao = MatchDAO(sess)

            faculty_iter = fac_dao.iter_faculty_with_relations(stream=False)
            processed = 0

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
                    chain=chain,
                    fac=fac,
                    faculty_id=int(fac.faculty_id),
                    candidates=candidates,
                    opp_map=opp_map,
                )

                if out_rows:
                    match_dao.upsert_matches(out_rows)

                processed += 1
                if commit_every and processed % commit_every == 0:
                    sess.commit()
                    logger.info("Committed after %d faculty processed", processed)

            sess.commit()
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

        chain = self._build_chain()

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
                chain=chain,
                fac=fac,
                faculty_id=int(faculty_id),
                candidates=candidates,
                opp_map=opp_map,
            )

            if out_rows:
                match_dao.upsert_matches(out_rows)
                sess.commit()

            return len(out_rows)

    def run_for_opportunity(
        self,
        *,
        opportunity_id: str,
        faculty_ids: Optional[List[int]] = None,
        k: int = 200,
        min_domain: float = 0.30,
    ) -> int:
        """
        Generate one-to-one match rows for exactly one grant.
        If faculty_ids is provided, compute only for those faculty.
        Otherwise compute for top-k faculty by embedding similarity.
        Returns number of upserted match rows.
        """
        if not opportunity_id:
            return 0

        chain = self._build_chain()

        with self.session_factory() as sess:
            opp_dao = OpportunityDAO(sess)
            match_dao = MatchDAO(sess)

            opps = opp_dao.read_opportunities_by_ids_with_relations([str(opportunity_id)])
            if not opps:
                return 0
            opp = opps[0]

            opp_req_idx_json = self._opportunity_requirements_json(opp)

            candidates: List[tuple[int, float]] = []
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

            out_rows = []
            for fid, domain_sim in candidates:
                fac = fac_map.get(int(fid))
                if not fac:
                    continue

                fac_json = self._faculty_keywords_json(fac)
                scored = self._score_pair(
                    chain=chain,
                    fac_json=fac_json,
                    opp_req_idx_json=opp_req_idx_json,
                )
                out_rows.append(
                    self._build_match_row(
                        grant_id=str(opportunity_id),
                        faculty_id=int(fid),
                        domain_sim=float(domain_sim),
                        scored=scored,
                    )
                )

            if out_rows:
                match_dao.upsert_matches(out_rows)
                sess.commit()
            return len(out_rows)

    def run_for_opp(
        self,
        *,
        opportunity_id: str,
        faculty_ids: Optional[List[int]] = None,
        k: int = 200,
        min_domain: float = 0.30,
    ) -> int:
        """Compatibility alias. Use run_for_opportunity for new code."""
        return self.run_for_opportunity(
            opportunity_id=opportunity_id,
            faculty_ids=faculty_ids,
            k=k,
            min_domain=min_domain,
        )
