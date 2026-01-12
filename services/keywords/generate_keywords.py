from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings, OPENAI_MODEL, OPENAI_API_KEY
from dao.faculty_dao import FacultyDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from db.models import Faculty
from llm_schemas.keywords import KeywordsOut, CandidatesOut
from services.keywords.generate_context import faculty_to_keyword_context, opportunity_to_prompt_payload
from utils.content_extractor import load_extracted_content


import json
from langchain_core.prompts import ChatPromptTemplate
from typing import Tuple
from langchain_openai import ChatOpenAI



FACULTY_CANDIDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Extract concise candidate keyword phrases from the faculty context..."),
    ("human", "Context (JSON):\n{context_json}")
])

FACULTY_KEYWORDS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Generate research/application keywords for faculty using ONLY context..."),
    ("human", "Context (JSON):\n{context_json}\n\nCandidate phrases:\n{candidates}")
])

OPP_CANDIDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Extract concise candidate keyword phrases from the funding opportunity context..."),
    ("human", "Context (JSON):\n{context_json}")
])

OPP_KEYWORDS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Generate research/application keywords for a funding opportunity using ONLY context..."),
    ("human", "Context (JSON):\n{context_json}\n\nCandidate phrases:\n{candidates}")
])

def build_keyword_chain(candidate_prompt: ChatPromptTemplate, keywords_prompt: ChatPromptTemplate):
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)

    candidates_chain = candidate_prompt | llm.with_structured_output(CandidatesOut)
    keywords_chain = keywords_prompt | llm.with_structured_output(KeywordsOut)
    return candidates_chain, keywords_chain

def generate_keywords(obj, *, context_builder, candidates_chain, keywords_chain) -> Tuple[dict, dict]:
    context = context_builder(obj)
    context_json = json.dumps(context, ensure_ascii=False)

    cand_out: CandidatesOut = candidates_chain.invoke({"context_json": context_json})
    candidates = cand_out.candidates[:50]

    kw_out: KeywordsOut = keywords_chain.invoke({
        "context_json": context_json,
        "candidates": "\n".join(f"- {c}" for c in candidates),
    })

    return kw_out.model_dump(), {"context_used": context, "candidates": candidates}


if __name__ == '__main__':
    faculty_cand_chain, faculty_kw_chain = build_keyword_chain(FACULTY_CANDIDATE_PROMPT, FACULTY_KEYWORDS_PROMPT)
    opp_cand_chain, opp_kw_chain = build_keyword_chain(OPP_CANDIDATE_PROMPT, OPP_KEYWORDS_PROMPT)

    with SessionLocal() as sess:
        fac_dao = FacultyDAO(sess)
        opp_dao = OpportunityDAO(sess)

        for fac in fac_dao.iter_faculty_with_relations():
            faculty_keywords, faculty_keywords_raw = generate_keywords(
                fac,
                context_builder=faculty_to_keyword_context,
                candidates_chain=faculty_cand_chain,
                keywords_chain=faculty_kw_chain,
            )

            fac_rows = [{
                "faculty_id": fac.faculty_id,
                "keywords": faculty_keywords,
                "raw_json": faculty_keywords_raw,
                "source": OPENAI_MODEL,
            }]

            fac_dao.upsert_keywords_json(fac_rows)
        sess.commit()

        for opp in opp_dao.iter_opportunities_with_relations():
            opportunity_keywords, opportunity_keywords_raw = generate_keywords(
                opp,
                context_builder=opportunity_to_prompt_payload,
                candidates_chain=opp_cand_chain,
                keywords_chain=opp_kw_chain,
            )
            opp_rows = [{
                "opportunity_id": opp.opportunity_id,
                "keywords": opportunity_keywords,
                "raw_json": opportunity_keywords_raw,
                "source": OPENAI_MODEL,
            }]

            opp_dao.upsert_keywords_json(opp_rows)
        sess.commit()



