from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from config import get_llm_client, settings
from dao.faculty_dao import FacultyDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from db.models.faculty import Faculty
from dto.llm_response_dto import CandidatesOut, KeywordsOut, WeightedSpecsOut
from services.context.context_generator import ContextGenerator
from services.prompts.keyword_prompts import (
    FACULTY_CANDIDATE_PROMPT,
    FACULTY_KEYWORDS_PROMPT,
    FACULTY_SPECIALIZATION_WEIGHT_PROMPT,
    OPP_CANDIDATE_PROMPT,
    OPP_KEYWORDS_PROMPT,
    OPP_SPECIALIZATION_WEIGHT_PROMPT,
)
from utils.content_compressor import cap_extracted_blocks
from utils.embedder import embed_domain_bucket
from utils.keyword_utils import (
    apply_weighted_specializations,
    coerce_keyword_sections,
    extract_domains_from_keywords,
)
from utils.payload_sanitizer import sanitize_for_postgres

ContextBuilder = Callable[[Any], Dict[str, Any]]


class KeywordGenerationService:
    def __init__(self, *, context_generator: ContextGenerator):
        self.context_generator = context_generator

    @staticmethod
    def build_keyword_chain(
        candidate_prompt: ChatPromptTemplate,
        keywords_prompt: ChatPromptTemplate,
        weight_prompt: ChatPromptTemplate,
    ):
        llm = get_llm_client().build()
        candidates_chain = candidate_prompt | llm.with_structured_output(CandidatesOut)
        keywords_chain = keywords_prompt | llm.with_structured_output(KeywordsOut)
        weight_chain = weight_prompt | llm.with_structured_output(WeightedSpecsOut)
        return candidates_chain, keywords_chain, weight_chain

    @staticmethod
    def _apply_limit(iterable, limit: int):
        if limit and limit > 0:
            count = 0
            for item in iterable:
                yield item
                count += 1
                if count >= limit:
                    break
            return
        yield from iterable

    def generate_keywords(
        self,
        obj: Any,
        *,
        context_builder: ContextBuilder,
        candidates_chain,
        keywords_chain,
        weight_chain,
    ) -> Tuple[dict, dict]:
        context = context_builder(obj)
        if "attachments_extracted" in context:
            context["attachments_extracted"] = cap_extracted_blocks(
                context["attachments_extracted"],
                max_total_chars=18_000,
                max_per_doc_chars=2_000,
            )

        context = sanitize_for_postgres(context)
        context_json = json.dumps(context, ensure_ascii=False)

        cand_out: CandidatesOut = candidates_chain.invoke({"context_json": context_json})
        candidates = (cand_out.candidates or [])[:50]

        kw_out: KeywordsOut = keywords_chain.invoke(
            {
                "context_json": context_json,
                "candidates": "\n".join(f"- {c}" for c in candidates),
            }
        )
        kw_dict = coerce_keyword_sections(kw_out.model_dump())

        spec_in = {
            "research": (kw_dict.get("research") or {}).get("specialization") or [],
            "application": (kw_dict.get("application") or {}).get("specialization") or [],
        }
        weighted_out: WeightedSpecsOut = weight_chain.invoke(
            {
                "context_json": context_json,
                "spec_json": json.dumps(spec_in, ensure_ascii=False),
            }
        )

        kw_weighted = apply_weighted_specializations(keywords=kw_dict, weighted=weighted_out)
        raw_debug = {
            "context_used": context,
            "candidates": candidates,
            "keywords_unweighted": kw_dict,
            "specializations_input": spec_in,
            "weighted_specializations": {
                "research": [x.model_dump() for x in weighted_out.research],
                "application": [x.model_dump() for x in weighted_out.application],
            },
        }
        return sanitize_for_postgres(kw_weighted), sanitize_for_postgres(raw_debug)

    def generate_faculty_keywords_for_id(self, faculty_id: int) -> Optional[dict]:
        if not faculty_id:
            return None

        faculty_cand_chain, faculty_kw_chain, faculty_w_chain = self.build_keyword_chain(
            FACULTY_CANDIDATE_PROMPT,
            FACULTY_KEYWORDS_PROMPT,
            FACULTY_SPECIALIZATION_WEIGHT_PROMPT,
        )

        with SessionLocal() as sess:
            fac_dao = FacultyDAO(sess)
            fac = sess.get(Faculty, faculty_id)
            if not fac:
                return None

            faculty_keywords, faculty_keywords_raw = self.generate_keywords(
                fac,
                context_builder=self.context_generator.build_faculty_basic_context,
                candidates_chain=faculty_cand_chain,
                keywords_chain=faculty_kw_chain,
                weight_chain=faculty_w_chain,
            )

            source_model = settings.bedrock_model_id
            embed_model = settings.bedrock_embed_model_id
            fac_dao.upsert_keywords_json(
                [
                    {
                        "faculty_id": fac.faculty_id,
                        "keywords": faculty_keywords,
                        "raw_json": faculty_keywords_raw,
                        "source": source_model,
                    }
                ]
            )

            r_domains, a_domains = extract_domains_from_keywords(faculty_keywords)
            r_vec = embed_domain_bucket(r_domains)
            a_vec = embed_domain_bucket(a_domains)
            if r_vec is not None or a_vec is not None:
                fac_dao.upsert_keyword_embedding(
                    {
                        "faculty_id": fac.faculty_id,
                        "model": embed_model,
                        "research_domain_vec": r_vec,
                        "application_domain_vec": a_vec,
                    }
                )
            sess.commit()
            return faculty_keywords

    def run_batch(self, *, run_faculty: bool, run_opp: bool, limit: int) -> None:
        faculty_cand_chain, faculty_kw_chain, faculty_w_chain = self.build_keyword_chain(
            FACULTY_CANDIDATE_PROMPT,
            FACULTY_KEYWORDS_PROMPT,
            FACULTY_SPECIALIZATION_WEIGHT_PROMPT,
        )
        opp_cand_chain, opp_kw_chain, opp_w_chain = self.build_keyword_chain(
            OPP_CANDIDATE_PROMPT,
            OPP_KEYWORDS_PROMPT,
            OPP_SPECIALIZATION_WEIGHT_PROMPT,
        )

        with SessionLocal() as sess:
            fac_dao = FacultyDAO(sess)
            opp_dao = OpportunityDAO(sess)
            source_model = settings.bedrock_model_id
            embed_model = settings.bedrock_embed_model_id

            if run_faculty:
                for fac in self._apply_limit(fac_dao.iter_faculty_missing_keywords(), limit):
                    faculty_keywords, faculty_keywords_raw = self.generate_keywords(
                        fac,
                        context_builder=self.context_generator.build_faculty_basic_context,
                        candidates_chain=faculty_cand_chain,
                        keywords_chain=faculty_kw_chain,
                        weight_chain=faculty_w_chain,
                    )
                    fac_dao.upsert_keywords_json(
                        [
                            {
                                "faculty_id": fac.faculty_id,
                                "keywords": faculty_keywords,
                                "raw_json": faculty_keywords_raw,
                                "source": source_model,
                            }
                        ]
                    )

                    r_domains, a_domains = extract_domains_from_keywords(faculty_keywords)
                    r_vec = embed_domain_bucket(r_domains)
                    a_vec = embed_domain_bucket(a_domains)
                    if r_vec is not None or a_vec is not None:
                        fac_dao.upsert_keyword_embedding(
                            {
                                "faculty_id": fac.faculty_id,
                                "model": embed_model,
                                "research_domain_vec": r_vec,
                                "application_domain_vec": a_vec,
                            }
                        )
                sess.commit()

            if run_opp:
                for opp in self._apply_limit(opp_dao.iter_opportunity_missing_keywords(), limit):
                    opportunity_keywords, opportunity_keywords_raw = self.generate_keywords(
                        opp,
                        context_builder=self.context_generator.build_opportunity_basic_context,
                        candidates_chain=opp_cand_chain,
                        keywords_chain=opp_kw_chain,
                        weight_chain=opp_w_chain,
                    )
                    opp_dao.upsert_keywords_json(
                        [
                            {
                                "opportunity_id": opp.opportunity_id,
                                "keywords": opportunity_keywords,
                                "raw_json": opportunity_keywords_raw,
                                "source": source_model,
                            }
                        ]
                    )

                    r_domains, a_domains = extract_domains_from_keywords(opportunity_keywords)
                    r_vec = embed_domain_bucket(r_domains)
                    a_vec = embed_domain_bucket(a_domains)
                    if r_vec is not None or a_vec is not None:
                        opp_dao.upsert_keyword_embedding(
                            {
                                "opportunity_id": opp.opportunity_id,
                                "model": embed_model,
                                "research_domain_vec": r_vec,
                                "application_domain_vec": a_vec,
                            }
                        )
                sess.commit()
