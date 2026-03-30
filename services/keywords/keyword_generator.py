from __future__ import annotations

import json
import hashlib
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import text

from config import get_embedding_client, get_llm_client, settings
from dao.faculty_dao import FacultyDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from db.models.faculty import Faculty, FacultyKeyword
from db.models.opportunity import Opportunity, OpportunityKeyword
from dto.llm_response_dto import (
    CandidatesOut,
    KeywordsOut,
    OpportunityCategoryOut,
    SpecializationSourcesOut,
    WeightedSpecsOut,
)
from services.context_retrieval.context_generator import ContextGenerator
from services.prompts.keyword_prompts import (
    FACULTY_CANDIDATE_PROMPT,
    FACULTY_KEYWORDS_PROMPT,
    FACULTY_SPECIALIZATION_SOURCE_PROMPT,
    FACULTY_SPECIALIZATION_WEIGHT_PROMPT,
    OPP_CATEGORY_PROMPT,
    OPP_CANDIDATE_PROMPT,
    OPP_KEYWORDS_PROMPT,
    OPP_SPECIALIZATION_SOURCE_PROMPT,
    OPP_SPECIALIZATION_WEIGHT_PROMPT,
)
from utils.embedder import embed_domain_bucket
from utils.keyword_utils import (
    attach_specialization_sources_from_llm,
    apply_weighted_specializations,
    build_specialization_source_catalog,
    coerce_keyword_sections,
    extract_domains_from_keywords,
)
from utils.payload_sanitizer import sanitize_for_postgres
from utils.thread_pool import build_thread_local_getter, parallel_map, resolve_pool_size

ContextBuilder = Callable[[Any], Dict[str, Any]]
logger = logging.getLogger(__name__)


class _KeywordGeneratorBase:
    def __init__(self, *, context_generator: ContextGenerator, force_regenerate: bool = False):
        self.context_generator = context_generator
        self.force_regenerate = bool(force_regenerate)

    @staticmethod
    def build_keyword_chain(
        candidate_prompt: ChatPromptTemplate,
        keywords_prompt: ChatPromptTemplate,
        weight_prompt: ChatPromptTemplate,
    ):
        model_id = (settings.haiku or settings.sonnet or settings.opus or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        candidates_chain = candidate_prompt | llm.with_structured_output(CandidatesOut)
        keywords_chain = keywords_prompt | llm.with_structured_output(KeywordsOut)
        weight_chain = weight_prompt | llm.with_structured_output(WeightedSpecsOut)
        return candidates_chain, keywords_chain, weight_chain

    @staticmethod
    def build_opportunity_category_chain():
        model_id = (settings.haiku or settings.sonnet or settings.opus or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        return OPP_CATEGORY_PROMPT | llm.with_structured_output(OpportunityCategoryOut)

    #TODO: can be in the main chain.
    @staticmethod
    def build_specialization_source_chain(source_prompt: ChatPromptTemplate):
        model_id = (settings.haiku or settings.sonnet or settings.opus or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        return source_prompt | llm.with_structured_output(SpecializationSourcesOut)

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

    @staticmethod
    def _dedupe_in_order(values: List[Any]) -> Tuple[List[Any], int]:
        seen = set()
        out: List[Any] = []
        dup = 0
        for v in list(values or []):
            if v in seen:
                dup += 1
                continue
            seen.add(v)
            out.append(v)
        return out, dup

    @staticmethod
    def _advisory_key(namespace: str, value: str) -> int:
        digest = hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).digest()
        raw = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return raw - (1 << 64) if raw >= (1 << 63) else raw

    def _try_claim_job_lock(self, sess, *, namespace: str, value: str) -> bool:
        key = self._advisory_key(namespace, value)
        got = sess.execute(
            text("SELECT pg_try_advisory_xact_lock(:k) AS locked"),
            {"k": int(key)},
        ).scalar()
        return bool(got)

    def _classify_opportunity_category(
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

    def generate_keywords(
        self,
        obj: Any,
        *,
        context_builder: ContextBuilder,
        candidates_chain,
        keywords_chain,
        weight_chain,
        source_chain=None,
    ) -> Tuple[dict, dict]:
        context = context_builder(obj)

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
        source_catalog = build_specialization_source_catalog(context)
        source_map_raw: Dict[str, Any] = {}
        source_error: Optional[str] = None
        kw_with_sources = kw_weighted
        if source_catalog:
            if source_chain is not None:
                try:
                    source_out: SpecializationSourcesOut = source_chain.invoke(
                        {
                            "spec_json": json.dumps(
                                {
                                    "research": (kw_weighted.get("research") or {}).get("specialization") or [],
                                    "application": (kw_weighted.get("application") or {}).get("specialization") or [],
                                },
                                ensure_ascii=False,
                            ),
                            "source_catalog_json": json.dumps(source_catalog, ensure_ascii=False),
                        }
                    )
                    source_map_raw = source_out.model_dump()
                except Exception as e:
                    source_error = f"{type(e).__name__}: {e}"
            kw_with_sources = attach_specialization_sources_from_llm(
                keywords=kw_weighted,
                llm_sources=source_map_raw,
                source_catalog=source_catalog,
            )

        raw_debug = {
            "context_used": context,
            "candidates": candidates,
            "keywords_unweighted": kw_dict,
            "specializations_input": spec_in,
            "weighted_specializations": {
                "research": [x.model_dump() for x in weighted_out.research],
                "application": [x.model_dump() for x in weighted_out.application],
            },
            "specialization_source_catalog_count": len(source_catalog),
            "specialization_sources_raw": source_map_raw,
        }
        if source_error:
            raw_debug["specialization_sources_error"] = source_error
        return sanitize_for_postgres(kw_with_sources), sanitize_for_postgres(raw_debug)


class FacultyKeywordGenerator(_KeywordGeneratorBase):
    def generate_faculty_keywords_for_id(self, faculty_id: int, *, force_regenerate: Optional[bool] = None) -> Optional[dict]:
        if not faculty_id:
            return None

        force = self._resolve_force(force_regenerate)
        faculty_cand_chain, faculty_kw_chain, faculty_w_chain = self.build_keyword_chain(
            FACULTY_CANDIDATE_PROMPT,
            FACULTY_KEYWORDS_PROMPT,
            FACULTY_SPECIALIZATION_WEIGHT_PROMPT,
        )
        faculty_source_chain = self.build_specialization_source_chain(FACULTY_SPECIALIZATION_SOURCE_PROMPT)

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
                source_chain=faculty_source_chain,
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

    def run_batch(
        self,
        *,
        limit: int,
        force_regenerate: Optional[bool] = None,
        commit_every: int = 10,
        workers: int = 4,
    ) -> None:
        _ = commit_every  # Per-item commits are used in threaded mode.
        force = self._resolve_force(force_regenerate)
        safe_limit = max(0, int(limit or 0))
        with SessionLocal() as sess:
            q = sess.query(Faculty.faculty_id)
            if not force:
                q = (
                    q.outerjoin(FacultyKeyword, FacultyKeyword.faculty_id == Faculty.faculty_id)
                    .filter(FacultyKeyword.faculty_id.is_(None))
                )
            q = q.order_by(Faculty.faculty_id.asc())
            if safe_limit > 0:
                q = q.limit(safe_limit)
            target_ids_raw = [int(fid) for (fid,) in q.all()]
            target_ids, duplicate_targets = self._dedupe_in_order(target_ids_raw)
            if duplicate_targets:
                logger.warning("Faculty keyword batch deduped duplicate targets=%s", duplicate_targets)

        pool_size = resolve_pool_size(max_workers=int(workers or 0), task_count=len(target_ids))
        logger.info(
            "Faculty keyword batch start force=%s limit=%s targets=%s workers=%s",
            force,
            safe_limit,
            len(target_ids),
            pool_size,
        )
        if pool_size == 0:
            return

        source_model = settings.haiku
        embed_model = settings.bedrock_embed_model_id

        def _build_thread_state() -> Dict[str, Any]:
            fac_cand_chain, fac_kw_chain, fac_w_chain = self.build_keyword_chain(
                FACULTY_CANDIDATE_PROMPT,
                FACULTY_KEYWORDS_PROMPT,
                FACULTY_SPECIALIZATION_WEIGHT_PROMPT,
            )
            fac_source_chain = self.build_specialization_source_chain(FACULTY_SPECIALIZATION_SOURCE_PROMPT)
            return {
                "candidates_chain": fac_cand_chain,
                "keywords_chain": fac_kw_chain,
                "weight_chain": fac_w_chain,
                "source_chain": fac_source_chain,
                "embedding_client": get_embedding_client().build(),
            }

        get_thread_state = build_thread_local_getter(_build_thread_state)

        def _run_one(faculty_id: int) -> Dict[str, Any]:
            state = get_thread_state()
            with SessionLocal() as sess:
                fac_dao = FacultyDAO(sess)
                fac = sess.get(Faculty, int(faculty_id))
                if not fac:
                    return {"status": "missing", "faculty_id": int(faculty_id)}

                if not force and fac_dao.has_keyword_row(int(faculty_id)):
                    return {"status": "skipped_existing", "faculty_id": int(faculty_id)}

                if not self._try_claim_job_lock(
                    sess,
                    namespace="faculty_keyword",
                    value=str(int(faculty_id)),
                ):
                    return {"status": "skipped_locked", "faculty_id": int(faculty_id)}

                if not force and fac_dao.has_keyword_row(int(faculty_id)):
                    return {"status": "skipped_existing", "faculty_id": int(faculty_id)}

                faculty_keywords, faculty_keywords_raw = self.generate_keywords(
                    fac,
                    context_builder=self.context_generator.build_faculty_basic_context,
                    candidates_chain=state["candidates_chain"],
                    keywords_chain=state["keywords_chain"],
                    weight_chain=state["weight_chain"],
                    source_chain=state["source_chain"],
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
                r_vec = embed_domain_bucket(r_domains, embedding_client=state["embedding_client"])
                a_vec = embed_domain_bucket(a_domains, embedding_client=state["embedding_client"])
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
                return {"status": "processed", "faculty_id": int(faculty_id)}

        def _on_error(_idx: int, faculty_id: int, exc: Exception) -> Dict[str, Any]:
            logger.exception("Faculty keyword generation failed for faculty_id=%s", faculty_id)
            return {
                "status": "failed",
                "faculty_id": int(faculty_id),
                "error": f"{type(exc).__name__}: {exc}",
            }

        results = parallel_map(
            target_ids,
            max_workers=pool_size,
            run_item=_run_one,
            on_error=_on_error,
        )

        processed = sum(1 for r in results if r.get("status") == "processed")
        skipped = sum(1 for r in results if r.get("status") == "skipped_existing")
        skipped_locked = sum(1 for r in results if r.get("status") == "skipped_locked")
        missing = sum(1 for r in results if r.get("status") == "missing")
        failed = sum(1 for r in results if r.get("status") == "failed")
        logger.info(
            "Faculty keyword batch done processed=%s skipped=%s skipped_locked=%s missing=%s failed=%s",
            processed,
            skipped,
            skipped_locked,
            missing,
            failed,
        )


class OpportunityKeywordGenerator(_KeywordGeneratorBase):
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
        opp_source_chain = self.build_specialization_source_chain(OPP_SPECIALIZATION_SOURCE_PROMPT)
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
                source_chain=opp_source_chain,
            )
            ctx_used = (opportunity_keywords_raw or {}).get("context_used") or self.context_generator.build_opportunity_basic_context(opp)
            category = self._classify_opportunity_category(
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
        limit: int,
        force_regenerate: Optional[bool] = None,
        commit_every: int = 10,
        workers: int = 4,
    ) -> None:
        _ = commit_every  # Per-item commits are used in threaded mode.
        force = self._resolve_force(force_regenerate)
        safe_limit = max(0, int(limit or 0))
        with SessionLocal() as sess:
            q = sess.query(Opportunity.opportunity_id)
            if not force:
                q = (
                    q.outerjoin(OpportunityKeyword, OpportunityKeyword.opportunity_id == Opportunity.opportunity_id)
                    .filter(OpportunityKeyword.opportunity_id.is_(None))
                )
            q = q.order_by(Opportunity.opportunity_id.asc())
            if safe_limit > 0:
                q = q.limit(safe_limit)
            target_ids_raw = [str(oid) for (oid,) in q.all()]
            target_ids, duplicate_targets = self._dedupe_in_order(target_ids_raw)
            if duplicate_targets:
                logger.warning("Opportunity keyword batch deduped duplicate targets=%s", duplicate_targets)

        pool_size = resolve_pool_size(max_workers=int(workers or 0), task_count=len(target_ids))
        logger.info(
            "Opportunity keyword batch start force=%s limit=%s targets=%s workers=%s",
            force,
            safe_limit,
            len(target_ids),
            pool_size,
        )
        if pool_size == 0:
            return

        source_model = settings.haiku
        embed_model = settings.bedrock_embed_model_id

        def _build_thread_state() -> Dict[str, Any]:
            opp_cand_chain, opp_kw_chain, opp_w_chain = self.build_keyword_chain(
                OPP_CANDIDATE_PROMPT,
                OPP_KEYWORDS_PROMPT,
                OPP_SPECIALIZATION_WEIGHT_PROMPT,
            )
            opp_source_chain = self.build_specialization_source_chain(OPP_SPECIALIZATION_SOURCE_PROMPT)
            opp_cat_chain = self.build_opportunity_category_chain()
            return {
                "candidates_chain": opp_cand_chain,
                "keywords_chain": opp_kw_chain,
                "weight_chain": opp_w_chain,
                "source_chain": opp_source_chain,
                "category_chain": opp_cat_chain,
                "embedding_client": get_embedding_client().build(),
            }

        get_thread_state = build_thread_local_getter(_build_thread_state)

        def _run_one(opportunity_id: str) -> Dict[str, Any]:
            oid = str(opportunity_id)
            state = get_thread_state()
            with SessionLocal() as sess:
                opp_dao = OpportunityDAO(sess)
                opps = opp_dao.read_opportunities_by_ids_with_relations([oid])
                if not opps:
                    return {"status": "missing", "opportunity_id": oid}
                opp = opps[0]

                if not force and opp_dao.has_keyword_row(oid):
                    return {"status": "skipped_existing", "opportunity_id": oid}

                if not self._try_claim_job_lock(
                    sess,
                    namespace="opportunity_keyword",
                    value=oid,
                ):
                    return {"status": "skipped_locked", "opportunity_id": oid}

                if not force and opp_dao.has_keyword_row(oid):
                    return {"status": "skipped_existing", "opportunity_id": oid}

                opportunity_keywords, opportunity_keywords_raw = self.generate_keywords(
                    opp,
                    context_builder=self.context_generator.build_opportunity_basic_context,
                    candidates_chain=state["candidates_chain"],
                    keywords_chain=state["keywords_chain"],
                    weight_chain=state["weight_chain"],
                    source_chain=state["source_chain"],
                )
                ctx_used = (
                    (opportunity_keywords_raw or {}).get("context_used")
                    or self.context_generator.build_opportunity_basic_context(opp)
                )
                category = self._classify_opportunity_category(
                    category_chain=state["category_chain"],
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
                r_vec = embed_domain_bucket(r_domains, embedding_client=state["embedding_client"])
                a_vec = embed_domain_bucket(a_domains, embedding_client=state["embedding_client"])
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
                return {"status": "processed", "opportunity_id": oid}

        def _on_error(_idx: int, opportunity_id: str, exc: Exception) -> Dict[str, Any]:
            logger.exception("Opportunity keyword generation failed for opportunity_id=%s", opportunity_id)
            return {
                "status": "failed",
                "opportunity_id": str(opportunity_id),
                "error": f"{type(exc).__name__}: {exc}",
            }

        results = parallel_map(
            target_ids,
            max_workers=pool_size,
            run_item=_run_one,
            on_error=_on_error,
        )

        processed = sum(1 for r in results if r.get("status") == "processed")
        skipped = sum(1 for r in results if r.get("status") == "skipped_existing")
        skipped_locked = sum(1 for r in results if r.get("status") == "skipped_locked")
        missing = sum(1 for r in results if r.get("status") == "missing")
        failed = sum(1 for r in results if r.get("status") == "failed")
        logger.info(
            "Opportunity keyword batch done processed=%s skipped=%s skipped_locked=%s missing=%s failed=%s",
            processed,
            skipped,
            skipped_locked,
            missing,
            failed,
        )
