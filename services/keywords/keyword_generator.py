from __future__ import annotations

import json
import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor
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
    KeywordBucket,
    OpportunityCategoryOut,
    WeightedSpecsOut,
)
from services.context_retrieval.context_generator import ContextGenerator
from services.prompts.keyword_prompts import (
    FACULTY_APPLICATION_CANDIDATE_PROMPT,
    FACULTY_APPLICATION_KEYWORDS_PROMPT,
    FACULTY_APPLICATION_WEIGHTED_MERGE_PROMPT,
    FACULTY_APPLICATION_SPECIALIZATION_WEIGHT_PROMPT,
    FACULTY_APPLICATION_MERGE_PROMPT,
    FACULTY_RESEARCH_CANDIDATE_PROMPT,
    FACULTY_RESEARCH_KEYWORDS_PROMPT,
    FACULTY_RESEARCH_WEIGHTED_MERGE_PROMPT,
    FACULTY_RESEARCH_SPECIALIZATION_WEIGHT_PROMPT,
    FACULTY_RESEARCH_MERGE_PROMPT,
    OPP_CATEGORY_PROMPT,
    OPP_APPLICATION_CANDIDATE_PROMPT,
    OPP_APPLICATION_KEYWORDS_PROMPT,
    OPP_APPLICATION_WEIGHTED_MERGE_PROMPT,
    OPP_APPLICATION_SPECIALIZATION_WEIGHT_PROMPT,
    OPP_APPLICATION_MERGE_PROMPT,
    OPP_RESEARCH_CANDIDATE_PROMPT,
    OPP_RESEARCH_KEYWORDS_PROMPT,
    OPP_RESEARCH_WEIGHTED_MERGE_PROMPT,
    OPP_RESEARCH_SPECIALIZATION_WEIGHT_PROMPT,
    OPP_RESEARCH_MERGE_PROMPT,
)
from utils.embedder import embed_domain_bucket
from utils.keyword_utils import (
    coerce_keyword_sections,
    extract_domains_from_keywords,
)
from utils.payload_sanitizer import sanitize_for_postgres
from utils.thread_pool import build_thread_local_getter, parallel_map, resolve_pool_size

ContextBuilder = Callable[[Any], Dict[str, Any]]
logger = logging.getLogger(__name__)


class _KeywordGeneratorBase:
    """Shared keyword-generation pipeline used by faculty and opportunity generators."""

    KEYWORD_MAX_CONTEXT_CHARS = 50_000
    KEYWORD_MAX_CANDIDATES_PER_BATCH = 50
    KEYWORD_MAX_BATCH_KEYWORDS = 20
    KEYWORD_BATCH_WORKERS = 2

    def __init__(self, *, context_generator: ContextGenerator, force_regenerate: bool = False):
        self.context_generator = context_generator
        self.force_regenerate = bool(force_regenerate)

    @staticmethod
    def build_keyword_chain(
        research_candidate_prompt: ChatPromptTemplate,
        application_candidate_prompt: ChatPromptTemplate,
        research_keywords_prompt: ChatPromptTemplate,
        application_keywords_prompt: ChatPromptTemplate,
        weight_prompt: ChatPromptTemplate,
    ):
        """Build chains for candidate extraction, keyword extraction, and weighting."""
        model_id = (settings.haiku or settings.sonnet or settings.opus or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        research_candidates_chain = research_candidate_prompt | llm.with_structured_output(CandidatesOut)
        application_candidates_chain = application_candidate_prompt | llm.with_structured_output(CandidatesOut)
        research_keywords_chain = research_keywords_prompt | llm.with_structured_output(KeywordBucket)
        application_keywords_chain = application_keywords_prompt | llm.with_structured_output(KeywordBucket)
        weight_chain = weight_prompt | llm.with_structured_output(WeightedSpecsOut)
        return (
            research_candidates_chain,
            application_candidates_chain,
            research_keywords_chain,
            application_keywords_chain,
            weight_chain,
        )

    @staticmethod
    def build_keyword_chain_split_weight(
        research_candidate_prompt: ChatPromptTemplate,
        application_candidate_prompt: ChatPromptTemplate,
        research_keywords_prompt: ChatPromptTemplate,
        application_keywords_prompt: ChatPromptTemplate,
        research_weight_prompt: ChatPromptTemplate,
        application_weight_prompt: ChatPromptTemplate,
    ):
        """Build chains with split weighting prompts for research and application."""
        model_id = (settings.haiku or settings.sonnet or settings.opus or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        research_candidates_chain = research_candidate_prompt | llm.with_structured_output(CandidatesOut)
        application_candidates_chain = application_candidate_prompt | llm.with_structured_output(CandidatesOut)
        research_keywords_chain = research_keywords_prompt | llm.with_structured_output(KeywordBucket)
        application_keywords_chain = application_keywords_prompt | llm.with_structured_output(KeywordBucket)
        research_weight_chain = research_weight_prompt | llm.with_structured_output(WeightedSpecsOut)
        application_weight_chain = application_weight_prompt | llm.with_structured_output(WeightedSpecsOut)
        return (
            research_candidates_chain,
            application_candidates_chain,
            research_keywords_chain,
            application_keywords_chain,
            research_weight_chain,
            application_weight_chain,
        )

    @staticmethod
    def build_keyword_merge_chain(
        research_merge_prompt: ChatPromptTemplate,
        application_merge_prompt: ChatPromptTemplate,
    ):
        """Build chains that merge per-batch keyword outputs into one section-level output."""
        model_id = (settings.haiku or settings.sonnet or settings.opus or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        research_merge_chain = research_merge_prompt | llm.with_structured_output(KeywordBucket)
        application_merge_chain = application_merge_prompt | llm.with_structured_output(KeywordBucket)
        return research_merge_chain, application_merge_chain

    @staticmethod
    def build_weighted_merge_chain(
        research_weighted_merge_prompt: ChatPromptTemplate,
        application_weighted_merge_prompt: ChatPromptTemplate,
    ):
        """Build chains that consolidate and reweight batch-level weighted specializations."""
        model_id = (settings.haiku or settings.sonnet or settings.opus or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        research_weighted_merge_chain = (
            research_weighted_merge_prompt | llm.with_structured_output(WeightedSpecsOut)
        )
        application_weighted_merge_chain = (
            application_weighted_merge_prompt | llm.with_structured_output(WeightedSpecsOut)
        )
        return research_weighted_merge_chain, application_weighted_merge_chain

    @staticmethod
    def build_opportunity_category_chain():
        """Build chain that classifies broad/specific opportunity categories from context + keywords."""
        model_id = (settings.haiku or settings.sonnet or settings.opus or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        return OPP_CATEGORY_PROMPT | llm.with_structured_output(OpportunityCategoryOut)

    @staticmethod
    def _normalize_specific_categories(values: list[str]) -> list[str]:
        """Normalize and dedupe specific category tags into snake_case tokens."""
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
        """Yield at most `limit` items (or all items when limit is not positive)."""
        if limit and limit > 0:
            count = 0
            for item in iterable:
                yield item
                count += 1
                if count >= limit:
                    break
            return
        yield from iterable

    @staticmethod
    def _norm_text_key(value: Any) -> str:
        """Create a stable, case-insensitive key for text dedupe/merge operations."""
        return " ".join(str(value or "").strip().lower().split())

    def _resolve_force(self, force_regenerate: Optional[bool]) -> bool:
        """Resolve per-call force flag, defaulting to the instance-level setting."""
        if force_regenerate is None:
            return self.force_regenerate
        return bool(force_regenerate)

    @staticmethod
    def _dedupe_in_order(values: List[Any]) -> Tuple[List[Any], int]:
        """Dedupe list while preserving first-seen order and reporting duplicate count."""
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
        """Create deterministic 64-bit key used for DB advisory locks."""
        digest = hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).digest()
        raw = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return raw - (1 << 64) if raw >= (1 << 63) else raw

    def _try_claim_job_lock(self, sess, *, namespace: str, value: str) -> bool:
        """Try to claim transaction-scoped advisory lock; returns False when already taken."""
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
        """Classify opportunity category with a safe fallback to `unclear` on failure."""
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
        research_candidates_chain,
        application_candidates_chain,
        research_keywords_chain,
        application_keywords_chain,
        research_weight_chain=None,
        application_weight_chain=None,
        weight_chain=None,
        research_merge_chain=None,
        application_merge_chain=None,
        research_weighted_merge_chain=None,
        application_weighted_merge_chain=None,
        source_embedding_client: Optional[Any] = None,
    ) -> Tuple[dict, dict]:
        """Run full keyword pipeline and return (keywords_with_sources, raw_debug_payload)."""
        obj_id = (
            getattr(obj, "faculty_id", None)
            or getattr(obj, "opportunity_id", None)
            or getattr(obj, "id", None)
            or "unknown"
        )
        obj_tag = f"{type(obj).__name__}:{obj_id}"

        #=================================
        # 1. Build/normalize context and split into LLM-sized batches
        #=================================
        full_context = sanitize_for_postgres(context_builder(obj))
        context_batches = self.context_generator.build_keyword_context_batches(
            context=full_context,
            max_chars=self.KEYWORD_MAX_CONTEXT_CHARS,
        )
        logger.info(
            "Keyword generation batching obj=%s batches=%s max_chars=%s",
            obj_tag,
            len(context_batches),
            self.KEYWORD_MAX_CONTEXT_CHARS,
        )

        #=================================
        # 2. Run each batch in parallel (batch worker count is fixed to 2)
        #    Each batch executes: candidate -> keyword -> weight
        #=================================
        batch_items: List[Tuple[int, Dict[str, Any]]] = [
            (int(i), dict(ctx or {}))
            for i, ctx in enumerate(list(context_batches), start=1)
        ]
        batch_workers = resolve_pool_size(
            max_workers=int(self.KEYWORD_BATCH_WORKERS),
            task_count=len(batch_items),
        )
        logger.info(
            "Keyword generation batch parallel obj=%s workers=%s batches=%s",
            obj_tag,
            batch_workers,
            len(batch_items),
        )

        def _run_one_batch(item: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
            batch_idx, batch_ctx = item
            batch_json = json.dumps(batch_ctx, ensure_ascii=False)

            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_r_cand = pool.submit(research_candidates_chain.invoke, {"context_json": batch_json})
                fut_a_cand = pool.submit(application_candidates_chain.invoke, {"context_json": batch_json})
                candidates_research = self.context_generator.dedupe_keyword_texts(
                    (fut_r_cand.result().candidates or [])[: self.KEYWORD_MAX_CANDIDATES_PER_BATCH]
                )
                candidates_application = self.context_generator.dedupe_keyword_texts(
                    (fut_a_cand.result().candidates or [])[: self.KEYWORD_MAX_CANDIDATES_PER_BATCH]
                )
            logger.info(
                "KW_CHAIN candidates obj=%s batch=%s research=%s application=%s",
                obj_tag,
                int(batch_idx),
                len(candidates_research),
                len(candidates_application),
            )

            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_r_kw = pool.submit(
                    research_keywords_chain.invoke,
                    {
                        "context_json": batch_json,
                        "candidates": "\n".join(f"- {c}" for c in candidates_research),
                    },
                )
                fut_a_kw = pool.submit(
                    application_keywords_chain.invoke,
                    {
                        "context_json": batch_json,
                        "candidates": "\n".join(f"- {c}" for c in candidates_application),
                    },
                )
                research_out: KeywordBucket = fut_r_kw.result()
                application_out: KeywordBucket = fut_a_kw.result()

            research_kw = research_out.model_dump() if hasattr(research_out, "model_dump") else {}
            application_kw = application_out.model_dump() if hasattr(application_out, "model_dump") else {}
            logger.info(
                "KW_CHAIN keywords obj=%s batch=%s research(domain=%s,spec=%s) application(domain=%s,spec=%s)",
                obj_tag,
                int(batch_idx),
                len(list(research_kw.get("domain") or [])),
                len(list(research_kw.get("specialization") or [])),
                len(list(application_kw.get("domain") or [])),
                len(list(application_kw.get("specialization") or [])),
            )

            keyword_row_research = self.context_generator.format_keyword_merge_input_row(
                batch_idx=int(batch_idx),
                candidates=candidates_research,
                keyword_bucket=research_kw,
                max_batch_keywords=self.KEYWORD_MAX_BATCH_KEYWORDS,
            )
            keyword_row_application = self.context_generator.format_keyword_merge_input_row(
                batch_idx=int(batch_idx),
                candidates=candidates_application,
                keyword_bucket=application_kw,
                max_batch_keywords=self.KEYWORD_MAX_BATCH_KEYWORDS,
            )

            batch_spec_in = {
                "research": list((keyword_row_research or {}).get("specialization") or []),
                "application": list((keyword_row_application or {}).get("specialization") or []),
            }
            logger.info(
                "KW_CHAIN weight_input obj=%s batch=%s research_specs=%s application_specs=%s",
                obj_tag,
                int(batch_idx),
                len(list(batch_spec_in.get("research") or [])),
                len(list(batch_spec_in.get("application") or [])),
            )

            batch_weighted_payload: Dict[str, List[Dict[str, Any]]]
            weight_error: Optional[str] = None
            if batch_spec_in["research"] or batch_spec_in["application"]:
                try:
                    batch_weighted_out: Optional[WeightedSpecsOut] = None
                    if research_weight_chain is not None or application_weight_chain is not None:
                        context_json = json.dumps(batch_ctx, ensure_ascii=False)
                        research_rows: List[Dict[str, Any]] = []
                        application_rows: List[Dict[str, Any]] = []

                        with ThreadPoolExecutor(max_workers=2) as pool:
                            fut_r = None
                            fut_a = None
                            if research_weight_chain is not None and batch_spec_in["research"]:
                                fut_r = pool.submit(
                                    research_weight_chain.invoke,
                                    {
                                        "context_json": context_json,
                                        "spec_json": json.dumps(
                                            {
                                                "research": list(batch_spec_in.get("research") or []),
                                                "application": [],
                                            },
                                            ensure_ascii=False,
                                        ),
                                    },
                                )
                            if application_weight_chain is not None and batch_spec_in["application"]:
                                fut_a = pool.submit(
                                    application_weight_chain.invoke,
                                    {
                                        "context_json": context_json,
                                        "spec_json": json.dumps(
                                            {
                                                "research": [],
                                                "application": list(batch_spec_in.get("application") or []),
                                            },
                                            ensure_ascii=False,
                                        ),
                                    },
                                )

                            if fut_r is not None:
                                research_out_weighted: WeightedSpecsOut = fut_r.result()
                                research_rows = [x.model_dump() for x in (research_out_weighted.research or [])]
                            if fut_a is not None:
                                application_out_weighted: WeightedSpecsOut = fut_a.result()
                                application_rows = [x.model_dump() for x in (application_out_weighted.application or [])]

                        batch_weighted_payload = {
                            "research": research_rows,
                            "application": application_rows,
                        }
                    else:
                        if weight_chain is None:
                            raise ValueError("No weighting chain configured for keyword generation.")
                        batch_weighted_out = weight_chain.invoke(
                            {
                                "context_json": json.dumps(batch_ctx, ensure_ascii=False),
                                "spec_json": json.dumps(batch_spec_in, ensure_ascii=False),
                            }
                        )
                        batch_weighted_payload = {
                            "research": [x.model_dump() for x in (batch_weighted_out.research or [])],
                            "application": [x.model_dump() for x in (batch_weighted_out.application or [])],
                        }
                    r_rows = list(batch_weighted_payload.get("research") or [])
                    a_rows = list(batch_weighted_payload.get("application") or [])
                    r_nonzero = sum(1 for x in r_rows if float((x or {}).get("w") or 0.0) > 0.0)
                    a_nonzero = sum(1 for x in a_rows if float((x or {}).get("w") or 0.0) > 0.0)
                    logger.info(
                        "KW_CHAIN weight_output obj=%s batch=%s research_nonzero=%s/%s application_nonzero=%s/%s sample_application=%s",
                        obj_tag,
                        int(batch_idx),
                        r_nonzero,
                        len(r_rows),
                        a_nonzero,
                        len(a_rows),
                        [
                            {"t": str((x or {}).get("t") or ""), "w": float((x or {}).get("w") or 0.0)}
                            for x in a_rows[:3]
                        ],
                    )
                except Exception as e:
                    weight_error = f"batch={int(batch_idx)} {type(e).__name__}: {e}"
                    logger.exception(
                        "KW_CHAIN weight_failed obj=%s batch=%s error=%s",
                        obj_tag,
                        int(batch_idx),
                        weight_error,
                    )
                    batch_weighted_payload = {
                        "research": [{"t": str(t), "w": 0.0} for t in list(batch_spec_in.get("research") or [])],
                        "application": [{"t": str(t), "w": 0.0} for t in list(batch_spec_in.get("application") or [])],
                    }
            else:
                batch_weighted_payload = {"research": [], "application": []}

            return {
                "batch_idx": int(batch_idx),
                "batch_ctx": batch_ctx,
                "candidates_research": list(candidates_research),
                "candidates_application": list(candidates_application),
                "keyword_row_research": dict(keyword_row_research or {}),
                "keyword_row_application": dict(keyword_row_application or {}),
                "specializations_input": batch_spec_in,
                "weighted": batch_weighted_payload,
                "weight_error": weight_error,
            }

        def _on_batch_error(
            _idx: int,
            item: Tuple[int, Dict[str, Any]],
            exc: Exception,
        ) -> Dict[str, Any]:
            batch_idx, batch_ctx = item
            err = f"batch={int(batch_idx)} {type(exc).__name__}: {exc}"
            logger.exception("Keyword generation batch failed %s", err)
            return {
                "batch_idx": int(batch_idx),
                "batch_ctx": batch_ctx,
                "candidates_research": [],
                "candidates_application": [],
                "keyword_row_research": {
                    "batch_idx": int(batch_idx),
                    "candidates": [],
                    "domain": [],
                    "specialization": [],
                },
                "keyword_row_application": {
                    "batch_idx": int(batch_idx),
                    "candidates": [],
                    "domain": [],
                    "specialization": [],
                },
                "specializations_input": {"research": [], "application": []},
                "weighted": {"research": [], "application": []},
                "weight_error": err,
            }

        batch_results = parallel_map(
            batch_items,
            max_workers=max(1, int(batch_workers or 1)),
            run_item=_run_one_batch,
            on_error=_on_batch_error,
        )
        batch_results = sorted(batch_results, key=lambda x: int((x or {}).get("batch_idx") or 0))

        candidate_batches_research: List[List[str]] = [
            list((row or {}).get("candidates_research") or [])
            for row in batch_results
        ]
        candidate_batches_application: List[List[str]] = [
            list((row or {}).get("candidates_application") or [])
            for row in batch_results
        ]
        keyword_batches_research: List[Dict[str, Any]] = [
            dict((row or {}).get("keyword_row_research") or {})
            for row in batch_results
        ]
        keyword_batches_application: List[Dict[str, Any]] = [
            dict((row or {}).get("keyword_row_application") or {})
            for row in batch_results
        ]

        #=================================
        # 3. Merge per-batch keywords into one R bucket and one A bucket
        #=================================
        fallback_research = self.context_generator.fallback_keyword_merge_bucket(
            batch_domains=[list(item.get("domain") or []) for item in keyword_batches_research],
            batch_specializations=[list(item.get("specialization") or []) for item in keyword_batches_research],
        )
        fallback_application = self.context_generator.fallback_keyword_merge_bucket(
            batch_domains=[list(item.get("domain") or []) for item in keyword_batches_application],
            batch_specializations=[list(item.get("specialization") or []) for item in keyword_batches_application],
        )

        if research_merge_chain is not None:
            try:
                merged_research_out: KeywordBucket = research_merge_chain.invoke(
                    {"batch_json": json.dumps(keyword_batches_research, ensure_ascii=False)}
                )
                merged_research = self.context_generator.normalize_keyword_merge_output(merged_research_out)
            except Exception:
                merged_research = fallback_research
        else:
            merged_research = fallback_research

        if application_merge_chain is not None:
            try:
                merged_application_out: KeywordBucket = application_merge_chain.invoke(
                    {"batch_json": json.dumps(keyword_batches_application, ensure_ascii=False)}
                )
                merged_application = self.context_generator.normalize_keyword_merge_output(merged_application_out)
            except Exception:
                merged_application = fallback_application
        else:
            merged_application = fallback_application
        logger.info(
            "KW_CHAIN merged_keywords obj=%s research(domain=%s,spec=%s) application(domain=%s,spec=%s)",
            obj_tag,
            len(list((merged_research or {}).get("domain") or [])),
            len(list((merged_research or {}).get("specialization") or [])),
            len(list((merged_application or {}).get("domain") or [])),
            len(list((merged_application or {}).get("specialization") or [])),
        )

        kw_dict = coerce_keyword_sections(
            {
                "research": merged_research,
                "application": merged_application,
            }
        )

        #=================================
        # 4. Consolidate/reweight batch specializations with LLM weighted-merge
        #=================================
        spec_in = {
            "research": list((kw_dict.get("research") or {}).get("specialization") or []),
            "application": list((kw_dict.get("application") or {}).get("specialization") or []),
        }
        weighted_batches: List[Dict[str, Any]] = []
        weight_errors: List[str] = []
        merged_weighted_by_section_fallback: Dict[str, Dict[str, Dict[str, Any]]] = {
            "research": {},
            "application": {},
        }
        weighted_merge_batches: Dict[str, List[Dict[str, Any]]] = {
            "research": [],
            "application": [],
        }

        for batch in list(batch_results):
            batch_idx = int((batch or {}).get("batch_idx") or 0)
            batch_spec_in = dict((batch or {}).get("specializations_input") or {"research": [], "application": []})
            batch_weighted_payload = dict((batch or {}).get("weighted") or {"research": [], "application": []})
            batch_err = str((batch or {}).get("weight_error") or "").strip()
            if batch_err:
                weight_errors.append(batch_err)

            weighted_batches.append(
                {
                    "batch_idx": batch_idx,
                    "specializations_input": batch_spec_in,
                    "weighted": batch_weighted_payload,
                }
            )
            weighted_merge_batches["research"].append(
                {
                    "batch_idx": batch_idx,
                    "domain": list(((batch or {}).get("keyword_row_research") or {}).get("domain") or []),
                    "specialization": list(((batch or {}).get("keyword_row_research") or {}).get("specialization") or []),
                    "weighted_specialization": list(batch_weighted_payload.get("research") or []),
                }
            )
            weighted_merge_batches["application"].append(
                {
                    "batch_idx": batch_idx,
                    "domain": list(((batch or {}).get("keyword_row_application") or {}).get("domain") or []),
                    "specialization": list(((batch or {}).get("keyword_row_application") or {}).get("specialization") or []),
                    "weighted_specialization": list(batch_weighted_payload.get("application") or []),
                }
            )

            for sec in ("research", "application"):
                seen_in_batch = set()
                for item in list(batch_weighted_payload.get(sec) or []):
                    text = str((item or {}).get("t") or "").strip()
                    if not text:
                        continue
                    key = self._norm_text_key(text)
                    if not key or key in seen_in_batch:
                        continue
                    seen_in_batch.add(key)
                    try:
                        w = float((item or {}).get("w", 0.0))
                    except Exception:
                        w = 0.0
                    w = max(0.0, min(1.0, w))
                    existing = merged_weighted_by_section_fallback[sec].get(key)
                    if not existing:
                        merged_weighted_by_section_fallback[sec][key] = {
                            "t": text,
                            "w": w,
                            "support_count": 1,
                        }
                    else:
                        if w > float(existing.get("w", 0.0)):
                            existing["w"] = w
                            existing["t"] = text
                        existing["support_count"] = int(existing.get("support_count", 0)) + 1

        # Ensure merged specializations are preserved even if weighting skipped/fails.
        for sec in ("research", "application"):
            for text in list((kw_dict.get(sec) or {}).get("specialization") or []):
                t = str(text or "").strip()
                if not t:
                    continue
                key = self._norm_text_key(t)
                if not key:
                    continue
                if key not in merged_weighted_by_section_fallback[sec]:
                    merged_weighted_by_section_fallback[sec][key] = {
                        "t": t,
                        "w": 0.0,
                        "support_count": 0,
                    }

        #=================================
        # 5. Build final weighted keyword object
        #=================================
        fallback_merged_weighted_specs = {
            "research": sorted(
                list(merged_weighted_by_section_fallback["research"].values()),
                key=lambda x: (-float(x.get("w", 0.0)), -int(x.get("support_count", 0)), str(x.get("t") or "")),
            ),
            "application": sorted(
                list(merged_weighted_by_section_fallback["application"].values()),
                key=lambda x: (-float(x.get("w", 0.0)), -int(x.get("support_count", 0)), str(x.get("t") or "")),
            ),
        }
        merged_weighted_specs = {
            "research": list(fallback_merged_weighted_specs["research"]),
            "application": list(fallback_merged_weighted_specs["application"]),
        }
        context_json_for_weighted_merge = json.dumps(full_context, ensure_ascii=False)
        if research_weighted_merge_chain is not None:
            try:
                merged_research_weighted_out: WeightedSpecsOut = research_weighted_merge_chain.invoke(
                    {
                        "context_json": context_json_for_weighted_merge,
                        "batch_json": json.dumps(weighted_merge_batches.get("research") or [], ensure_ascii=False),
                    }
                )
                llm_research_weighted = [x.model_dump() for x in (merged_research_weighted_out.research or [])]
                if llm_research_weighted:
                    merged_weighted_specs["research"] = llm_research_weighted
            except Exception as e:
                logger.exception(
                    "KW_CHAIN weighted_merge_failed obj=%s section=research error=%s: %s",
                    obj_tag,
                    type(e).__name__,
                    e,
                )
        if application_weighted_merge_chain is not None:
            try:
                merged_application_weighted_out: WeightedSpecsOut = application_weighted_merge_chain.invoke(
                    {
                        "context_json": context_json_for_weighted_merge,
                        "batch_json": json.dumps(weighted_merge_batches.get("application") or [], ensure_ascii=False),
                    }
                )
                llm_application_weighted = [x.model_dump() for x in (merged_application_weighted_out.application or [])]
                if llm_application_weighted:
                    merged_weighted_specs["application"] = llm_application_weighted
            except Exception as e:
                logger.exception(
                    "KW_CHAIN weighted_merge_failed obj=%s section=application error=%s: %s",
                    obj_tag,
                    type(e).__name__,
                    e,
                )
        logger.info(
            "KW_CHAIN merged_weights obj=%s research_nonzero=%s/%s application_nonzero=%s/%s",
            obj_tag,
            sum(1 for x in list(merged_weighted_specs.get("research") or []) if float((x or {}).get("w") or 0.0) > 0.0),
            len(list(merged_weighted_specs.get("research") or [])),
            sum(1 for x in list(merged_weighted_specs.get("application") or []) if float((x or {}).get("w") or 0.0) > 0.0),
            len(list(merged_weighted_specs.get("application") or [])),
        )
        if (
            len(list(merged_weighted_specs.get("application") or [])) > 0
            and sum(1 for x in list(merged_weighted_specs.get("application") or []) if float((x or {}).get("w") or 0.0) > 0.0) == 0
        ):
            logger.warning(
                "KW_CHAIN application_all_zero obj=%s batches=%s weight_errors=%s",
                obj_tag,
                len(batch_results),
                len(weight_errors),
            )

        kw_weighted = {
            "research": dict(kw_dict.get("research") or {}),
            "application": dict(kw_dict.get("application") or {}),
        }
        kw_weighted["research"]["specialization"] = list(merged_weighted_specs["research"])
        kw_weighted["application"]["specialization"] = list(merged_weighted_specs["application"])

        #=================================
        # 6. Attach evidence sources using cosine similarity over source catalog
        #=================================
        source_attach = self.context_generator.attach_keyword_sources_by_cosine(
            keywords=kw_weighted,
            context=full_context,
            embedding_client=source_embedding_client,
            max_sources_per_specialization=4,
            min_similarity=0.10,
        )
        source_catalog = list(source_attach.get("source_catalog") or [])
        source_map_raw = dict(source_attach.get("source_map_raw") or {})
        source_error = source_attach.get("source_error")
        kw_with_sources = dict(source_attach.get("keywords") or kw_weighted)

        #=================================
        # 7. Build raw debug payload for traceability and return
        #=================================
        raw_debug = {
            "context_used": full_context,
            "source_mapping_method": "cosine_similarity",
            "context_batches_used": context_batches,
            "candidates": self.context_generator.dedupe_keyword_texts(
                [x for row in candidate_batches_research for x in row]
                + [x for row in candidate_batches_application for x in row]
            ),
            "candidates_research": self.context_generator.dedupe_keyword_texts(
                [x for row in candidate_batches_research for x in row]
            ),
            "candidates_application": self.context_generator.dedupe_keyword_texts(
                [x for row in candidate_batches_application for x in row]
            ),
            "keyword_batches_research": keyword_batches_research,
            "keyword_batches_application": keyword_batches_application,
            "keywords_unweighted": kw_dict,
            "specializations_input": spec_in,
            "weight_strategy": "llm_weighted_merge_over_batch_keywords_and_weights",
            "weighted_specializations_batches": weighted_batches,
            "weighted_merge_batches_input": weighted_merge_batches,
            "weighted_specializations": merged_weighted_specs,
            "specialization_source_catalog_count": len(source_catalog),
            "specialization_sources_raw": source_map_raw,
        }
        if weight_errors:
            raw_debug["weighted_specializations_errors"] = weight_errors
        if source_error:
            raw_debug["specialization_sources_error"] = source_error
        return sanitize_for_postgres(kw_with_sources), sanitize_for_postgres(raw_debug)


class FacultyKeywordGenerator(_KeywordGeneratorBase):
    def generate_faculty_keywords_for_id(self, faculty_id: int, *, force_regenerate: Optional[bool] = None) -> Optional[dict]:
        """Generate and persist keywords for one faculty id."""
        if not faculty_id:
            return None

        #=================================
        # 1. Build chains and load target faculty row
        #=================================
        force = self._resolve_force(force_regenerate)
        (
            faculty_r_cand_chain,
            faculty_a_cand_chain,
            faculty_r_kw_chain,
            faculty_a_kw_chain,
            faculty_r_w_chain,
            faculty_a_w_chain,
        ) = self.build_keyword_chain_split_weight(
            FACULTY_RESEARCH_CANDIDATE_PROMPT,
            FACULTY_APPLICATION_CANDIDATE_PROMPT,
            FACULTY_RESEARCH_KEYWORDS_PROMPT,
            FACULTY_APPLICATION_KEYWORDS_PROMPT,
            FACULTY_RESEARCH_SPECIALIZATION_WEIGHT_PROMPT,
            FACULTY_APPLICATION_SPECIALIZATION_WEIGHT_PROMPT,
        )
        faculty_r_merge_chain, faculty_a_merge_chain = self.build_keyword_merge_chain(
            FACULTY_RESEARCH_MERGE_PROMPT,
            FACULTY_APPLICATION_MERGE_PROMPT,
        )
        faculty_r_weighted_merge_chain, faculty_a_weighted_merge_chain = self.build_weighted_merge_chain(
            FACULTY_RESEARCH_WEIGHTED_MERGE_PROMPT,
            FACULTY_APPLICATION_WEIGHTED_MERGE_PROMPT,
        )

        with SessionLocal() as sess:
            fac_dao = FacultyDAO(sess)
            fac = sess.get(Faculty, faculty_id)
            if not fac:
                return None
            if not force and fac_dao.has_keyword_row(int(faculty_id)):
                existing = (getattr(getattr(fac, "keyword", None), "keywords", None) or {})
                return existing if existing else None

            #=================================
            # 2. Run keyword pipeline and upsert keyword JSON
            #=================================
            faculty_keywords, faculty_keywords_raw = self.generate_keywords(
                fac,
                context_builder=lambda fac_obj: self.context_generator.build_faculty_basic_context(
                    fac_obj,
                    use_rag=False,
                ),
                research_candidates_chain=faculty_r_cand_chain,
                application_candidates_chain=faculty_a_cand_chain,
                research_keywords_chain=faculty_r_kw_chain,
                application_keywords_chain=faculty_a_kw_chain,
                research_weight_chain=faculty_r_w_chain,
                application_weight_chain=faculty_a_w_chain,
                research_merge_chain=faculty_r_merge_chain,
                application_merge_chain=faculty_a_merge_chain,
                research_weighted_merge_chain=faculty_r_weighted_merge_chain,
                application_weighted_merge_chain=faculty_a_weighted_merge_chain,
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

            #=================================
            # 3. Build and persist domain embeddings for fast matching
            #=================================
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
        """Batch-generate faculty keywords in parallel with per-item transaction boundaries."""
        _ = commit_every  # Per-item commits are used in threaded mode.

        #=================================
        # 1. Resolve target faculty ids
        #=================================
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

        #=================================
        # 2. Prepare worker pool and thread-local chain state
        #=================================
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
            (
                fac_r_cand_chain,
                fac_a_cand_chain,
                fac_r_kw_chain,
                fac_a_kw_chain,
                fac_r_w_chain,
                fac_a_w_chain,
            ) = self.build_keyword_chain_split_weight(
                FACULTY_RESEARCH_CANDIDATE_PROMPT,
                FACULTY_APPLICATION_CANDIDATE_PROMPT,
                FACULTY_RESEARCH_KEYWORDS_PROMPT,
                FACULTY_APPLICATION_KEYWORDS_PROMPT,
                FACULTY_RESEARCH_SPECIALIZATION_WEIGHT_PROMPT,
                FACULTY_APPLICATION_SPECIALIZATION_WEIGHT_PROMPT,
            )
            fac_r_merge_chain, fac_a_merge_chain = self.build_keyword_merge_chain(
                FACULTY_RESEARCH_MERGE_PROMPT,
                FACULTY_APPLICATION_MERGE_PROMPT,
            )
            fac_r_weighted_merge_chain, fac_a_weighted_merge_chain = self.build_weighted_merge_chain(
                FACULTY_RESEARCH_WEIGHTED_MERGE_PROMPT,
                FACULTY_APPLICATION_WEIGHTED_MERGE_PROMPT,
            )
            emb_client = get_embedding_client().build()
            return {
                "research_candidates_chain": fac_r_cand_chain,
                "application_candidates_chain": fac_a_cand_chain,
                "research_keywords_chain": fac_r_kw_chain,
                "application_keywords_chain": fac_a_kw_chain,
                "research_weight_chain": fac_r_w_chain,
                "application_weight_chain": fac_a_w_chain,
                "research_merge_chain": fac_r_merge_chain,
                "application_merge_chain": fac_a_merge_chain,
                "research_weighted_merge_chain": fac_r_weighted_merge_chain,
                "application_weighted_merge_chain": fac_a_weighted_merge_chain,
                "embedding_client": emb_client,
            }

        get_thread_state = build_thread_local_getter(_build_thread_state)

        #=================================
        # 3. Per-item worker job: lock, generate, persist, embed
        #=================================
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
                    context_builder=lambda fac_obj: self.context_generator.build_faculty_basic_context(
                        fac_obj,
                        use_rag=False,
                    ),
                    research_candidates_chain=state["research_candidates_chain"],
                    application_candidates_chain=state["application_candidates_chain"],
                    research_keywords_chain=state["research_keywords_chain"],
                    application_keywords_chain=state["application_keywords_chain"],
                    research_weight_chain=state["research_weight_chain"],
                    application_weight_chain=state["application_weight_chain"],
                    research_merge_chain=state["research_merge_chain"],
                    application_merge_chain=state["application_merge_chain"],
                    research_weighted_merge_chain=state["research_weighted_merge_chain"],
                    application_weighted_merge_chain=state["application_weighted_merge_chain"],
                    source_embedding_client=state["embedding_client"],
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

        #=================================
        # 4. Execute workers and report summary
        #=================================
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
        """Generate and persist keywords + category for one opportunity id."""
        if not opportunity_id:
            return None

        #=================================
        # 1. Build chains and load target opportunity row
        #=================================
        force = self._resolve_force(force_regenerate)
        (
            opp_r_cand_chain,
            opp_a_cand_chain,
            opp_r_kw_chain,
            opp_a_kw_chain,
            opp_r_w_chain,
            opp_a_w_chain,
        ) = self.build_keyword_chain_split_weight(
            OPP_RESEARCH_CANDIDATE_PROMPT,
            OPP_APPLICATION_CANDIDATE_PROMPT,
            OPP_RESEARCH_KEYWORDS_PROMPT,
            OPP_APPLICATION_KEYWORDS_PROMPT,
            OPP_RESEARCH_SPECIALIZATION_WEIGHT_PROMPT,
            OPP_APPLICATION_SPECIALIZATION_WEIGHT_PROMPT,
        )
        opp_r_merge_chain, opp_a_merge_chain = self.build_keyword_merge_chain(
            OPP_RESEARCH_MERGE_PROMPT,
            OPP_APPLICATION_MERGE_PROMPT,
        )
        opp_r_weighted_merge_chain, opp_a_weighted_merge_chain = self.build_weighted_merge_chain(
            OPP_RESEARCH_WEIGHTED_MERGE_PROMPT,
            OPP_APPLICATION_WEIGHTED_MERGE_PROMPT,
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

            #=================================
            # 2. Run keyword pipeline and classify opportunity category
            #=================================
            opportunity_keywords, opportunity_keywords_raw = self.generate_keywords(
                opp,
                context_builder=lambda opp_obj: self.context_generator.build_opportunity_basic_context(
                    opp_obj,
                    use_rag=False,
                ),
                research_candidates_chain=opp_r_cand_chain,
                application_candidates_chain=opp_a_cand_chain,
                research_keywords_chain=opp_r_kw_chain,
                application_keywords_chain=opp_a_kw_chain,
                research_weight_chain=opp_r_w_chain,
                application_weight_chain=opp_a_w_chain,
                research_merge_chain=opp_r_merge_chain,
                application_merge_chain=opp_a_merge_chain,
                research_weighted_merge_chain=opp_r_weighted_merge_chain,
                application_weighted_merge_chain=opp_a_weighted_merge_chain,
            )
            ctx_used = (opportunity_keywords_raw or {}).get("context_used") or self.context_generator.build_opportunity_basic_context(opp)
            category = self._classify_opportunity_category(
                category_chain=opp_cat_chain,
                context=ctx_used,
                keywords=opportunity_keywords,
            )
            opportunity_keywords_raw = dict(opportunity_keywords_raw or {})
            opportunity_keywords_raw["category"] = category

            #=================================
            # 3. Persist keyword JSON + categories
            #=================================
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

            #=================================
            # 4. Build and persist domain embeddings for fast matching
            #=================================
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
        """Batch-generate opportunity keywords in parallel with per-item transaction boundaries."""
        _ = commit_every  # Per-item commits are used in threaded mode.

        #=================================
        # 1. Resolve target opportunity ids
        #=================================
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

        #=================================
        # 2. Prepare worker pool and thread-local chain state
        #=================================
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
            (
                opp_r_cand_chain,
                opp_a_cand_chain,
                opp_r_kw_chain,
                opp_a_kw_chain,
                opp_r_w_chain,
                opp_a_w_chain,
            ) = self.build_keyword_chain_split_weight(
                OPP_RESEARCH_CANDIDATE_PROMPT,
                OPP_APPLICATION_CANDIDATE_PROMPT,
                OPP_RESEARCH_KEYWORDS_PROMPT,
                OPP_APPLICATION_KEYWORDS_PROMPT,
                OPP_RESEARCH_SPECIALIZATION_WEIGHT_PROMPT,
                OPP_APPLICATION_SPECIALIZATION_WEIGHT_PROMPT,
            )
            opp_r_merge_chain, opp_a_merge_chain = self.build_keyword_merge_chain(
                OPP_RESEARCH_MERGE_PROMPT,
                OPP_APPLICATION_MERGE_PROMPT,
            )
            opp_r_weighted_merge_chain, opp_a_weighted_merge_chain = self.build_weighted_merge_chain(
                OPP_RESEARCH_WEIGHTED_MERGE_PROMPT,
                OPP_APPLICATION_WEIGHTED_MERGE_PROMPT,
            )
            opp_cat_chain = self.build_opportunity_category_chain()
            emb_client = get_embedding_client().build()
            return {
                "research_candidates_chain": opp_r_cand_chain,
                "application_candidates_chain": opp_a_cand_chain,
                "research_keywords_chain": opp_r_kw_chain,
                "application_keywords_chain": opp_a_kw_chain,
                "research_weight_chain": opp_r_w_chain,
                "application_weight_chain": opp_a_w_chain,
                "research_merge_chain": opp_r_merge_chain,
                "application_merge_chain": opp_a_merge_chain,
                "research_weighted_merge_chain": opp_r_weighted_merge_chain,
                "application_weighted_merge_chain": opp_a_weighted_merge_chain,
                "category_chain": opp_cat_chain,
                "embedding_client": emb_client,
            }

        get_thread_state = build_thread_local_getter(_build_thread_state)

        #=================================
        # 3. Per-item worker job: lock, generate, classify, persist, embed
        #=================================
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
                    context_builder=lambda opp_obj: self.context_generator.build_opportunity_basic_context(
                        opp_obj,
                        use_rag=False,
                    ),
                    research_candidates_chain=state["research_candidates_chain"],
                    application_candidates_chain=state["application_candidates_chain"],
                    research_keywords_chain=state["research_keywords_chain"],
                    application_keywords_chain=state["application_keywords_chain"],
                    research_weight_chain=state["research_weight_chain"],
                    application_weight_chain=state["application_weight_chain"],
                    research_merge_chain=state["research_merge_chain"],
                    application_merge_chain=state["application_merge_chain"],
                    research_weighted_merge_chain=state["research_weighted_merge_chain"],
                    application_weighted_merge_chain=state["application_weighted_merge_chain"],
                    source_embedding_client=state["embedding_client"],
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

        #=================================
        # 4. Execute workers and report summary
        #=================================
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
