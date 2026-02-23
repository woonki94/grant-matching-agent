from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from config import get_llm_client, settings
from dao.faculty_dao import FacultyDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from db.models.faculty import Faculty
from dto.llm_response_dto import CandidatesOut, KeywordsOut, OpportunityCategoryOut, WeightedSpecsOut
from services.context.context_generator import ContextGenerator
from services.prompts.keyword_prompts import (
    FACULTY_CANDIDATE_PROMPT,
    FACULTY_KEYWORDS_PROMPT,
    FACULTY_SPECIALIZATION_WEIGHT_PROMPT,
    OPP_CATEGORY_PROMPT,
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


class KeywordGenerator:
    def __init__(self, *, context_generator: ContextGenerator, force_regenerate: bool = False):
        self.context_generator = context_generator
        self.force_regenerate = bool(force_regenerate)

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
    def build_opportunity_category_chain():
        llm = get_llm_client().build()
        return OPP_CATEGORY_PROMPT | llm.with_structured_output(OpportunityCategoryOut)

    @staticmethod
    def _normalize_specific_categories(values: list[str]) -> list[str]:
        out: list[str] = []
        seen = set()
        for raw in values or []:
            token = str(raw or "").strip().lower()
            if not token:
                continue
            token = re.sub(r"[^a-z0-9_]+", "_", token)
            token = re.sub(r"_+", "_", token).strip("_")
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    def classify_opportunity_category(
        self,
        *,
        category_chain,
        context: Dict[str, Any],
        keywords: Dict[str, Any],
    ) -> Dict[str, Any]:
        valid_broad = {"basic_research", "applied_research", "educational", "unclear"}
        try:
            out: OpportunityCategoryOut = category_chain.invoke(
                {
                    "context_json": json.dumps(sanitize_for_postgres(context or {}), ensure_ascii=False),
                    "keywords_json": json.dumps(sanitize_for_postgres(keywords or {}), ensure_ascii=False),
                }
            )
            broad = str(getattr(out, "broad_category", "unclear") or "unclear").strip().lower()
            if broad not in valid_broad:
                broad = "unclear"
            specific = self._normalize_specific_categories(getattr(out, "specific_categories", []) or [])
            return {
                "broad_category": broad,
                "specific_categories": specific,
            }
        except Exception as e:
            return {
                "broad_category": "unclear",
                "specific_categories": [],
                "error": f"{type(e).__name__}: {e}",
            }

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

    def _resolve_force(self, force_regenerate: Optional[bool]) -> bool:
        if force_regenerate is None:
            return self.force_regenerate
        return bool(force_regenerate)

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

    def generate_faculty_keywords_for_id(self, faculty_id: int, *, force_regenerate: Optional[bool] = None) -> Optional[dict]:
        if not faculty_id:
            return None

        force = self._resolve_force(force_regenerate)
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
            if not force and fac_dao.has_keyword_row(int(faculty_id)):
                existing = (getattr(getattr(fac, "keyword", None), "keywords", None) or {})
                return existing if existing else None

            faculty_keywords, faculty_keywords_raw = self.generate_keywords(
                fac,
                context_builder=self.context_generator.build_faculty_basic_context,
                candidates_chain=faculty_cand_chain,
                keywords_chain=faculty_kw_chain,
                weight_chain=faculty_w_chain,
            )

            source_model = settings.haiku
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

    def generate_opportunity_keywords_for_id(
        self,
        opportunity_id: str,
        *,
        force_regenerate: Optional[bool] = None,
    ) -> Optional[dict]:
        if not opportunity_id:
            return None

        force = self._resolve_force(force_regenerate)
        opp_cand_chain, opp_kw_chain, opp_w_chain = self.build_keyword_chain(
            OPP_CANDIDATE_PROMPT,
            OPP_KEYWORDS_PROMPT,
            OPP_SPECIALIZATION_WEIGHT_PROMPT,
        )
        opp_cat_chain = self.build_opportunity_category_chain()

        with SessionLocal() as sess:
            opp_dao = OpportunityDAO(sess)
            opps = opp_dao.read_opportunities_by_ids_with_relations([opportunity_id])
            if not opps:
                return None
            opp = opps[0]
            if not force and opp_dao.has_keyword_row(str(opportunity_id)):
                existing = (getattr(getattr(opp, "keyword", None), "keywords", None) or {})
                return existing if existing else None

            opportunity_keywords, opportunity_keywords_raw = self.generate_keywords(
                opp,
                context_builder=self.context_generator.build_opportunity_basic_context,
                candidates_chain=opp_cand_chain,
                keywords_chain=opp_kw_chain,
                weight_chain=opp_w_chain,
            )
            ctx_used = (opportunity_keywords_raw or {}).get("context_used") or self.context_generator.build_opportunity_basic_context(opp)
            category = self.classify_opportunity_category(
                category_chain=opp_cat_chain,
                context=ctx_used,
                keywords=opportunity_keywords,
            )
            opportunity_keywords_raw = dict(opportunity_keywords_raw or {})
            opportunity_keywords_raw["category"] = category

            source_model = settings.haiku
            embed_model = settings.bedrock_embed_model_id
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
            opp_dao.update_keyword_categories(
                opportunity_id=opp.opportunity_id,
                broad_category=category.get("broad_category"),
                specific_categories=category.get("specific_categories") or [],
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
            return opportunity_keywords

    def run_batch(
        self,
        *,
        run_faculty: bool,
        run_opp: bool,
        limit: int,
        force_regenerate: Optional[bool] = None,
    ) -> None:
        commit_every = 10
        force = self._resolve_force(force_regenerate)
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
        opp_cat_chain = self.build_opportunity_category_chain()

        with SessionLocal() as sess:
            fac_dao = FacultyDAO(sess)
            opp_dao = OpportunityDAO(sess)
            source_model = settings.haiku
            embed_model = settings.bedrock_embed_model_id

            if run_faculty:
                fac_processed = 0
                fac_iter = fac_dao.iter_faculty_with_relations() if force else fac_dao.iter_faculty_missing_keywords()
                for fac in self._apply_limit(fac_iter, limit):
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
                    fac_processed += 1
                    if fac_processed % commit_every == 0:
                        sess.commit()
                if fac_processed % commit_every != 0:
                    sess.commit()

            if run_opp:
                opp_processed = 0
                opp_iter = (
                    opp_dao.iter_opportunities_with_relations()
                    if force
                    else opp_dao.iter_opportunity_missing_keywords()
                )
                for opp in self._apply_limit(opp_iter, limit):
                    opportunity_keywords, opportunity_keywords_raw = self.generate_keywords(
                        opp,
                        context_builder=self.context_generator.build_opportunity_basic_context,
                        candidates_chain=opp_cand_chain,
                        keywords_chain=opp_kw_chain,
                        weight_chain=opp_w_chain,
                    )
                    ctx_used = (opportunity_keywords_raw or {}).get("context_used") or self.context_generator.build_opportunity_basic_context(opp)
                    category = self.classify_opportunity_category(
                        category_chain=opp_cat_chain,
                        context=ctx_used,
                        keywords=opportunity_keywords,
                    )
                    opportunity_keywords_raw = dict(opportunity_keywords_raw or {})
                    opportunity_keywords_raw["category"] = category

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
                    opp_dao.update_keyword_categories(
                        opportunity_id=opp.opportunity_id,
                        broad_category=category.get("broad_category"),
                        specific_categories=category.get("specific_categories") or [],
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
                    opp_processed += 1
                    if opp_processed % commit_every == 0:
                        sess.commit()
                if opp_processed % commit_every != 0:
                    sess.commit()
