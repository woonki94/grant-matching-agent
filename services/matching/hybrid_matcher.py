# services/matching/hybrid_matcher.py
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../root
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging
from typing import List

from config import get_llm_client  # Bedrock-only in your new setup
from dao.faculty_dao import FacultyDAO
from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from dto.llm_response_dto import LLMMatchOut, MissingItem, ScoredCoveredItem
from services.prompts.matching_prompt import MATCH_PROMPT
from utils.keyword_accessor import keywords_for_matching, requirements_indexed

logger = logging.getLogger(__name__)


def covered_to_grouped(items: List[ScoredCoveredItem]):
    out = {"application": {}, "research": {}}
    for it in items or []:
        sec = it.section
        idx = str(int(it.idx))
        c = float(it.c)
        prev = out[sec].get(idx)
        out[sec][idx] = c if prev is None else max(prev, c)
    return out


def missing_to_grouped(items: List[MissingItem]):
    out = {"application": [], "research": []}
    for it in items or []:
        out[it.section].append(int(it.idx))
    # stable dedupe
    for sec in out:
        seen = set()
        out[sec] = [x for x in out[sec] if not (x in seen or seen.add(x))]
    return out


def main(k: int, min_domain: float, limit_faculty: int, commit_every: int = 30):
    # Bedrock LLM (no OpenAI)
    llm = get_llm_client().build()
    chain = MATCH_PROMPT | llm.with_structured_output(LLMMatchOut)

    with SessionLocal() as sess:
        fac_dao = FacultyDAO(sess)
        opp_dao = OpportunityDAO(sess)
        match_dao = MatchDAO(sess)

        # NOTE: Your DAO controls streaming semantics. We keep as-is.
        faculty_iter = fac_dao.iter_faculty_with_relations(stream=False)

        processed = 0
        for fac in faculty_iter:
            if limit_faculty and limit_faculty > 0 and processed >= limit_faculty:
                break

            # 1) Candidate opps from vector similarity (domain)
            cand = match_dao.topk_opps_for_faculty(faculty_id=fac.faculty_id, k=k)
            cand = [(oid, s) for (oid, s) in cand if float(s) >= float(min_domain)]
            if not cand:
                processed += 1
                continue

            # 2) Faculty keywords (from DB, no local files)
            fac_kw = keywords_for_matching(getattr(fac.keyword, "keywords", {}) or {})
            fac_json = json.dumps(fac_kw, ensure_ascii=False)

            # 3) Load candidate opps (DB)
            opp_ids = [opp_id for opp_id, _ in cand]
            opps = opp_dao.read_opportunities_by_ids_with_relations(opp_ids)
            opp_map = {o.opportunity_id: o for o in opps}

            out_rows = []
            for opp_id, domain_sim in cand:
                opp = opp_map.get(opp_id)
                if not opp:
                    continue

                # 4) Opportunity requirements indexed (from DB keywords, no local files)
                opp_kw = keywords_for_matching(getattr(opp.keyword, "keywords", {}) or {})
                req_idx = requirements_indexed(opp_kw)
                opp_req_idx_json = json.dumps(req_idx, ensure_ascii=False)

                # 5) Bedrock LLM scoring
                scored: LLMMatchOut = chain.invoke(
                    {
                        "faculty_kw_json": fac_json,
                        "requirements_indexed": opp_req_idx_json,
                    }
                )

                covered_grouped = covered_to_grouped(scored.covered)
                missing_grouped = missing_to_grouped(scored.missing)

                out_rows.append(
                    {
                        "grant_id": opp_id,
                        "faculty_id": fac.faculty_id,
                        "domain_score": float(domain_sim),
                        "llm_score": float(scored.llm_score),
                        "reason": (scored.reason or "").strip(),
                        "covered": covered_grouped,
                        "missing": missing_grouped,
                    }
                )

            if out_rows:
                match_dao.upsert_matches(out_rows)

            processed += 1
            if commit_every and processed % commit_every == 0:
                sess.commit()
                logger.info("Committed after %d faculty processed", processed)

        sess.commit()
        logger.info("Hybrid matching completed. Total faculty processed: %d", processed)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hybrid matcher (Bedrock-only, DB-only IO)")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--min-domain", type=float, default=0.30)
    p.add_argument("--limit-faculty", type=int, default=100)
    p.add_argument("--commit-every", type=int, default=30)
    args = p.parse_args()

    main(
        k=args.k,
        min_domain=args.min_domain,
        limit_faculty=args.limit_faculty,
        commit_every=args.commit_every,
    )
