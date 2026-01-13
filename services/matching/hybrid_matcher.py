from __future__ import annotations
import json
import argparse
from sqlalchemy import text
from langchain_openai import ChatOpenAI

from config import OPENAI_MODEL, OPENAI_API_KEY
from db.db_conn import SessionLocal
from dao.faculty_dao import FacultyDAO
from dao.opportunity_dao import OpportunityDAO
from dao.match_dao import MatchDAO
from services.keywords.generate_context import faculty_to_keyword_context, opportunity_to_prompt_payload
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

            fac_ctx = cap_fac(faculty_to_keyword_context(fac))
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

                opp_ctx = cap_opp(opportunity_to_prompt_payload(opp))
                opp_json = json.dumps(opp_ctx, ensure_ascii=False)

                scored: LLMMatchOut = chain.invoke({
                    "faculty_json": fac_json,
                    "opp_json": opp_json,
                })

                out_rows.append({
                    "grant_id": opp_id,
                    "faculty_id": fac.faculty_id,
                    "domain_score": float(domain_sim),
                    "llm_score": float(scored.llm_score),
                    "reason": scored.reason.strip(),
                })

            match_dao.upsert_matches(out_rows)
            batch += 1
            if batch % 30 == 0:
                sess.commit()
        sess.commit()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--min-domain", type=float, default=0.30)
    p.add_argument("--limit-faculty", type=int, default=0)
    args = p.parse_args()
    main(k=args.k, min_domain=args.min_domain, limit_faculty=args.limit_faculty)