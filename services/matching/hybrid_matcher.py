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
from dto.llm_response_dto import LLMMatchOut  # make this


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
            # Stage 1: candidates by cosine similarity (top-K)
            cand = match_dao.topk_opps_for_faculty(faculty_id=fac.faculty_id, k=k)
            # threshold filter
            cand = [(oid, s) for (oid, s) in cand if s >= min_domain]
            if not cand:
                continue
            #TODO: necessary to pass all the context?? both for fac and opp
            #fac_ctx = cap_fac(faculty_to_keyword_context(fac))
            fac_ctx = {"name": fac.name, "keywords": (fac.keyword.keywords if fac.keyword else {})}
            fac_json = json.dumps(fac_ctx, ensure_ascii=False)

            # Stage 2: LLM score only the filtered candidates
            opp_ids = [opp_id for opp_id, _ in cand]
            opps = opp_dao.read_opportunities_by_ids_with_relations(opp_ids)
            opp_map = {o.opportunity_id: o for o in opps}

            out_rows = []
            for opp_id, domain_sim in cand:
                opp = opp_map.get(opp_id)
                if opp is None:
                    continue

                #opp_ctx = cap_opp(opportunity_to_keyword_context(opp))
                opp_ctx = {"opportunity_id": opp.opportunity_id,
                           "keywords": (opp.keyword.keywords if opp.keyword else {})}
                opp_json = json.dumps(opp_ctx, ensure_ascii=False)

                kw = opp.keyword.keywords or {}

                app_specs = (kw.get("application") or {}).get("specialization") or []
                res_specs = (kw.get("research") or {}).get("specialization") or []

                requirements_indexed = {
                    "application": {str(i): t for i, t in enumerate(app_specs)},
                    "research": {str(i): t for i, t in enumerate(res_specs)},
                }

                requirements_indexed_json = json.dumps(requirements_indexed, ensure_ascii=False)

                scored: LLMMatchOut = chain.invoke({
                    "faculty_kw_json": fac_json,
                    "requirements_indexed": requirements_indexed_json,
                })

                def group_by_section(items):
                    out = {"application": [], "research": []}
                    for it in items or []:
                        out[it.section].append(int(it.idx))
                    # stable dedupe
                    for k in out:
                        seen = set()
                        out[k] = [x for x in out[k] if not (x in seen or seen.add(x))]
                    return out

                covered_grouped = group_by_section(scored.covered)
                missing_grouped = group_by_section(scored.missing)

                out_rows.append({
                    "grant_id": opp_id,
                    "faculty_id": fac.faculty_id,
                    "domain_score": float(domain_sim),
                    "llm_score": float(scored.llm_score),
                    "reason": scored.reason.strip(),
                    "covered": covered_grouped,   # JSONB object
                    "missing": missing_grouped,   # JSONB object
                    #"evidence": getattr(scored, "evidence", {}) or {},
                })

            print(out_rows)
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