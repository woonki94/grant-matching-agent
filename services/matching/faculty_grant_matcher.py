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
from dto.llm_response_dto import LLMMatchOut, MissingItem, ScoredCoveredItem
from services.prompts.matching_prompt import MATCH_PROMPT
from utils.keyword_utils import keywords_for_matching, requirements_indexed

logger = logging.getLogger(__name__)



class FacultyGrantMatcherService:
    """One-to-one faculty↔grant matching service using vector retrieval + LLM scoring."""

    def __init__(self, *, session_factory=SessionLocal):
        self.session_factory = session_factory

    @staticmethod
    def _build_chain():
        llm = get_llm_client().build()
        return MATCH_PROMPT | llm.with_structured_output(LLMMatchOut)

    def _covered_to_grouped(items: List[ScoredCoveredItem]):
        out = {"application": {}, "research": {}}
        for it in items or []:
            sec = it.section
            idx = str(int(it.idx))
            c = float(it.c)
            prev = out[sec].get(idx)
            out[sec][idx] = c if prev is None else max(prev, c)
        return out

    def _missing_to_grouped(items: List[MissingItem]):
        out = {"application": [], "research": []}
        for it in items or []:
            out[it.section].append(int(it.idx))
        for sec in out:
            seen = set()
            out[sec] = [x for x in out[sec] if not (x in seen or seen.add(x))]
        return out

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

                fac_kw = keywords_for_matching(getattr(fac.keyword, "keywords", {}) or {})
                fac_json = json.dumps(fac_kw, ensure_ascii=False)

                opp_ids = [opp_id for opp_id, _ in candidates]
                opps = opp_dao.read_opportunities_by_ids_with_relations(opp_ids)
                opp_map = {o.opportunity_id: o for o in opps}

                out_rows = []
                for opp_id, domain_sim in candidates:
                    opp = opp_map.get(opp_id)
                    if not opp:
                        continue

                    opp_kw = keywords_for_matching(getattr(opp.keyword, "keywords", {}) or {})
                    req_idx = requirements_indexed(opp_kw)
                    opp_req_idx_json = json.dumps(req_idx, ensure_ascii=False)

                    scored: LLMMatchOut = chain.invoke(
                        {
                            "faculty_kw_json": fac_json,
                            "requirements_indexed": opp_req_idx_json,
                        }
                    )

                    out_rows.append(
                        {
                            "grant_id": opp_id,
                            "faculty_id": fac.faculty_id,
                            "domain_score": float(domain_sim),
                            "llm_score": float(scored.llm_score),
                            "reason": (scored.reason or "").strip(),
                            "covered": self._covered_to_grouped(scored.covered),
                            "missing": self._missing_to_grouped(scored.missing),
                        }
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

