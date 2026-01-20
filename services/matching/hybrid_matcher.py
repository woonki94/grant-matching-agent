from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../root
sys.path.insert(0, str(PROJECT_ROOT))

import json
import argparse
from sqlalchemy import text
from langchain_openai import ChatOpenAI

from config import OPENAI_MODEL, OPENAI_API_KEY
from db.db_conn import SessionLocal
from dao.faculty_dao import FacultyDAO
from dao.opportunity_dao import OpportunityDAO
from dao.match_dao import MatchDAO
from services.keywords.generate_context import faculty_to_keyword_context, opportunity_to_keyword_context
from services.prompts.matching_prompt import MATCH_PROMPT
from utils.content_compressor import cap_extracted_blocks, cap_fac, cap_opp
from dto.llm_response_dto import LLMMatchOut, ScoredCoveredItem, MissingItem  # make this

from utils.keyword_accessor import (
    keywords_for_matching,
    requirements_indexed,
)

def covered_to_grouped(items: list[ScoredCoveredItem]):
    out = {"application": {}, "research": {}}
    for it in items or []:
        sec = it.section
        idx = str(int(it.idx))
        c = float(it.c)
        # if duplicate idx appears, keep the max confidence
        prev = out[sec].get(idx)
        out[sec][idx] = c if prev is None else max(prev, c)
    return out

def missing_to_grouped(items: list[MissingItem]):
    out = {"application": [], "research": []}
    for it in items or []:
        out[it.section].append(int(it.idx))
    # stable dedupe
    for sec in out:
        seen = set()
        out[sec] = [x for x in out[sec] if not (x in seen or seen.add(x))]
    return out

def main(k: int, min_domain: float, limit_faculty: int):
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    chain = MATCH_PROMPT | llm.with_structured_output(LLMMatchOut)

    with SessionLocal() as sess:
        fac_dao = FacultyDAO(sess)
        opp_dao = OpportunityDAO(sess)
        match_dao = MatchDAO(sess)

        faculty_iter = fac_dao.iter_faculty_with_relations(stream=False)
        if limit_faculty and limit_faculty > 0:
            faculty_iter = faculty_iter[:limit_faculty]

        batch = 0
        for fac in faculty_iter:
            cand = match_dao.topk_opps_for_faculty(faculty_id=fac.faculty_id, k=k)
            cand = [(oid, s) for (oid, s) in cand if s >= min_domain]
            if not cand:
                continue

            fac_kw = keywords_for_matching(getattr(fac.keyword, "keywords", {}) or {})
            fac_json = json.dumps(fac_kw, ensure_ascii=False)

            opp_ids = [opp_id for opp_id, _ in cand]
            opps = opp_dao.read_opportunities_by_ids_with_relations(opp_ids)
            opp_map = {o.opportunity_id: o for o in opps}

            out_rows = []
            for opp_id, domain_sim in cand:
                opp = opp_map.get(opp_id)
                if not opp:
                    continue

                # Build indexed requirements from normalized (string-only) keywords
                opp_kw = keywords_for_matching(getattr(opp.keyword, "keywords", {}) or {})
                req_idx = requirements_indexed(opp_kw)
                opp_req_idx_json = json.dumps(req_idx, ensure_ascii=False)

                scored: LLMMatchOut = chain.invoke({
                    "faculty_kw_json": fac_json,
                    "requirements_indexed": opp_req_idx_json,
                })

                covered_grouped = covered_to_grouped(scored.covered)
                missing_grouped = missing_to_grouped(scored.missing)

                out_rows.append({
                    "grant_id": opp_id,
                    "faculty_id": fac.faculty_id,
                    "domain_score": float(domain_sim),
                    "llm_score": float(scored.llm_score),
                    "reason": scored.reason.strip(),
                    "covered": covered_grouped,
                    "missing": missing_grouped,
                })

            if out_rows:
                match_dao.upsert_matches(out_rows)

            batch += 1
            if batch % 30 == 0:
                sess.commit()

        sess.commit()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--min-domain", type=float, default=0.30)
    p.add_argument("--limit-faculty", type=int, default=10)
    args = p.parse_args()
    main(k=args.k, min_domain=args.min_domain, limit_faculty=args.limit_faculty)