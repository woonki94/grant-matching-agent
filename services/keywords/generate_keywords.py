from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../root
sys.path.insert(0, str(PROJECT_ROOT))

from services.prompts.keyword_prompts import (
    FACULTY_CANDIDATE_PROMPT,
    FACULTY_KEYWORDS_PROMPT,
    OPP_CANDIDATE_PROMPT,
    OPP_KEYWORDS_PROMPT, OPP_SPECIALIZATION_WEIGHT_PROMPT, FACULTY_SPECIALIZATION_WEIGHT_PROMPT
)
from utils.qwen_embedder import embed_domain_bucket, extract_domains


from config import settings, OPENAI_MODEL, OPENAI_API_KEY, QWEN_MODEL
from dao.faculty_dao import FacultyDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from db.models import Faculty
from dto.llm_response_dto import KeywordsOut, CandidatesOut, WeightedSpecsOut
from services.keywords.generate_context import faculty_to_keyword_context, opportunity_to_keyword_context
from utils.content_compressor import cap_extracted_blocks


import json
from langchain_core.prompts import ChatPromptTemplate
from typing import Tuple
from langchain_openai import ChatOpenAI


def build_keyword_chain(
    candidate_prompt: ChatPromptTemplate,
    keywords_prompt: ChatPromptTemplate,
    weight_prompt: ChatPromptTemplate,
):
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)

    candidates_chain = candidate_prompt | llm.with_structured_output(CandidatesOut)
    keywords_chain = keywords_prompt | llm.with_structured_output(KeywordsOut)
    weight_chain = weight_prompt | llm.with_structured_output(WeightedSpecsOut)

    return candidates_chain, keywords_chain, weight_chain


def _apply_weighted_specializations(*, keywords: dict, weighted: WeightedSpecsOut) -> dict:
    """
    Keeps domains unchanged (List[str]).
    Replaces specialization lists (List[str]) with List[{"t": str, "w": float}].
    """
    out = dict(keywords)
    out["research"] = dict(out.get("research") or {})
    out["application"] = dict(out.get("application") or {})

    out["research"]["specialization"] = [x.model_dump() for x in (weighted.research or [])]
    out["application"]["specialization"] = [x.model_dump() for x in (weighted.application or [])]
    return out


def generate_keywords(
    obj,
    *,
    context_builder,
    candidates_chain,
    keywords_chain,
    weight_chain,
) -> Tuple[dict, dict]:
    """
    3-step pipeline:
      1) context -> candidates
      2) context + candidates -> structured keywords (domains + specialization strings)
      3) context + specialization lists -> weighted specialization objects
    Returns:
      (keywords_weighted_json, raw_debug_json)
    """
    context = context_builder(obj)

    if "attachments_extracted" in context:
        context["attachments_extracted"] = cap_extracted_blocks(
            context["attachments_extracted"],
            max_total_chars=18_000,
            max_per_doc_chars=2_000,
        )

    context_json = json.dumps(context, ensure_ascii=False)

    # Step 1: candidates
    cand_out: CandidatesOut = candidates_chain.invoke({"context_json": context_json})
    candidates = (cand_out.candidates or [])[:50]

    # Step 2: structured keywords (specialization as strings)
    kw_out: KeywordsOut = keywords_chain.invoke({
        "context_json": context_json,
        "candidates": "\n".join(f"- {c}" for c in candidates),
    })
    kw_dict = kw_out.model_dump()

    # Step 3: weight specializations (faculty expertise OR opp requirement criticality)
    spec_in = {
        "research": (kw_dict.get("research") or {}).get("specialization") or [],
        "application": (kw_dict.get("application") or {}).get("specialization") or [],
    }
    weighted_out: WeightedSpecsOut = weight_chain.invoke({
        "context_json": context_json,
        "spec_json": json.dumps(spec_in, ensure_ascii=False),
    })

    kw_weighted = _apply_weighted_specializations(keywords=kw_dict, weighted=weighted_out)

    raw_debug = {
        "context_used": context,
        "candidates": candidates,
        "keywords_unweighted": kw_dict,          # helpful for debugging
        "specializations_input": spec_in,        # what was weighted
        "weighted_specializations": {
            "research": [x.model_dump() for x in weighted_out.research],
            "application": [x.model_dump() for x in weighted_out.application],
        },
    }

    return kw_weighted, raw_debug

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate keywords for faculty/opportunities")
    parser.add_argument("--limit", type=int, default=0, help="Max number of records to process (0 = no limit)")
    parser.add_argument("--faculty-only", action="store_true", help="Only generate faculty keywords")
    parser.add_argument("--opp-only", action="store_true", help="Only generate opportunity keywords")

    args = parser.parse_args()

    # If both flags set, treat as "do both" (or you can error out)
    run_faculty = not args.opp_only
    run_opp = not args.faculty_only

    faculty_cand_chain, faculty_kw_chain, faculty_w_chain = build_keyword_chain(
        FACULTY_CANDIDATE_PROMPT,
        FACULTY_KEYWORDS_PROMPT,
        FACULTY_SPECIALIZATION_WEIGHT_PROMPT,
    )

    opp_cand_chain, opp_kw_chain, opp_w_chain = build_keyword_chain(
        OPP_CANDIDATE_PROMPT,
        OPP_KEYWORDS_PROMPT,
        OPP_SPECIALIZATION_WEIGHT_PROMPT,
    )

    def _apply_limit(iterable):
        if args.limit and args.limit > 0:
            # assumes iterable is a generator; slice manually
            count = 0
            for x in iterable:
                yield x
                count += 1
                if count >= args.limit:
                    break
        else:
            yield from iterable

    with SessionLocal() as sess:
        fac_dao = FacultyDAO(sess)
        opp_dao = OpportunityDAO(sess)

        if run_faculty:
            for fac in _apply_limit(fac_dao.iter_faculty_with_relations()):
                faculty_keywords, faculty_keywords_raw = generate_keywords(
                    fac,
                    context_builder=faculty_to_keyword_context,
                    candidates_chain=faculty_cand_chain,
                    keywords_chain=faculty_kw_chain,
                    weight_chain=faculty_w_chain
                )

                # 1) upsert keywords
                fac_dao.upsert_keywords_json([{
                    "faculty_id": fac.faculty_id,
                    "keywords": faculty_keywords,
                    "raw_json": faculty_keywords_raw,
                    "source": OPENAI_MODEL,
                }])

                # 2) embed domains from freshly-generated keywords
                r_domains, a_domains = extract_domains(faculty_keywords)

                r_vec = embed_domain_bucket(r_domains)  # List[float] | None
                a_vec = embed_domain_bucket(a_domains)

                # upsert embedding row (only if at least one exists)
                if r_vec is not None or a_vec is not None:
                    fac_dao.upsert_keyword_embedding({
                        "faculty_id": fac.faculty_id,
                        "model": QWEN_MODEL,
                        "research_domain_vec": r_vec,
                        "application_domain_vec": a_vec,
                    })

            sess.commit()

        if run_opp:
            for opp in _apply_limit(opp_dao.iter_opportunities_with_relations()):
                opportunity_keywords, opportunity_keywords_raw = generate_keywords(
                    opp,
                    context_builder=opportunity_to_keyword_context,
                    candidates_chain=opp_cand_chain,
                    keywords_chain=opp_kw_chain,
                    weight_chain=opp_w_chain,
                )

                opp_dao.upsert_keywords_json([{
                    "opportunity_id": opp.opportunity_id,
                    "keywords": opportunity_keywords,
                    "raw_json": opportunity_keywords_raw,
                    "source": OPENAI_MODEL,
                }])

                r_domains, a_domains = extract_domains(opportunity_keywords)

                r_vec = embed_domain_bucket(r_domains)
                a_vec = embed_domain_bucket(a_domains)

                if r_vec is not None or a_vec is not None:
                    opp_dao.upsert_keyword_embedding({
                        "opportunity_id": opp.opportunity_id,
                        "model": QWEN_MODEL,
                        "research_domain_vec": r_vec,
                        "application_domain_vec": a_vec,
                    })

            sess.commit()



