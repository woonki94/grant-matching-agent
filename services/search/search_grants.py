from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from config import get_llm_client
from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from dto.llm_response_dto import LLMMatchOut
from services.keywords.keyword_service import KeywordGenerationService
from services.prompts.keyword_prompts import (
    QUERY_CANDIDATE_PROMPT,
    QUERY_KEYWORDS_PROMPT,
    QUERY_SPECIALIZATION_WEIGHT_PROMPT,
)
from services.prompts.matching_prompt import MATCH_PROMPT
from utils.embedder import embed_domain_bucket
from utils.keyword_utils import (
    apply_weighted_specializations,
    extract_domains_from_keywords,
    keywords_for_matching,
    requirements_indexed,
)
from utils.payload_sanitizer import sanitize_for_postgres

logger = logging.getLogger(__name__)


def _query_context(query_text: str, user_urls: Optional[List[str]]) -> Dict[str, Any]:
    return {
        "query_text": query_text,
        "user_urls": user_urls or [],
    }


def _matches_filters(
    opp,
    *,
    agency: Optional[str],
    category: Optional[str],
    status: Optional[str],
) -> bool:
    if agency and agency.lower() not in (opp.agency_name or "").lower():
        return False
    if category and category.lower() not in (opp.category or "").lower():
        return False
    if status and status.lower() not in (opp.opportunity_status or "").lower():
        return False
    return True


def generate_query_keywords(
    query_text: str,
    user_urls: Optional[List[str]],
) -> Dict[str, Any]:
    candidates_chain, keywords_chain, weight_chain = KeywordGenerationService.build_keyword_chain(
        QUERY_CANDIDATE_PROMPT,
        QUERY_KEYWORDS_PROMPT,
        QUERY_SPECIALIZATION_WEIGHT_PROMPT,
    )

    context = _query_context(query_text, user_urls)
    context = sanitize_for_postgres(context)
    context_json = json.dumps(context, ensure_ascii=False)

    cand_out = candidates_chain.invoke({"context_json": context_json})
    candidates = (cand_out.candidates or [])[:50]

    kw_out = keywords_chain.invoke(
        {
            "context_json": context_json,
            "candidates": "\n".join(f"- {c}" for c in candidates),
        }
    )
    kw_dict = kw_out.model_dump()
    for k in ("research", "application"):
        if isinstance(kw_dict.get(k), str):
            kw_dict[k] = json.loads(kw_dict[k])

    spec_in = {
        "research": (kw_dict.get("research") or {}).get("specialization") or [],
        "application": (kw_dict.get("application") or {}).get("specialization") or [],
    }
    weighted_out = weight_chain.invoke(
        {
            "context_json": context_json,
            "spec_json": json.dumps(spec_in, ensure_ascii=False),
        }
    )

    kw_weighted = apply_weighted_specializations(keywords=kw_dict, weighted=weighted_out)
    kw_weighted = sanitize_for_postgres(kw_weighted)
    return kw_weighted


def search_grants(
    *,
    query_text: str,
    top_k: int = 10,
    user_urls: Optional[List[str]] = None,
    agency: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Two-stage semantic search:
    1) Vector similarity on query domain embeddings vs opportunity embeddings
    2) LLM scoring using opportunity requirements and query keywords
    """
    if not query_text or not query_text.strip():
        return {
            "query_text": query_text,
            "results": [],
            "meta": {"reason": "empty query"},
        }

    top_k = int(top_k) if top_k and int(top_k) > 0 else 10

    with SessionLocal() as sess:
        match_dao = MatchDAO(sess)
        opp_dao = OpportunityDAO(sess)

        query_keywords = generate_query_keywords(query_text, user_urls)
        r_domains, a_domains = extract_domains_from_keywords(query_keywords)
        r_vec = embed_domain_bucket(r_domains)
        a_vec = embed_domain_bucket(a_domains)

        if r_vec is None and a_vec is None:
            return {
                "query_text": query_text,
                "results": [],
                "meta": {"reason": "no query embeddings produced"},
            }

        candidates = match_dao.topk_opps_for_query(
            research_vec=r_vec,
            application_vec=a_vec,
            k=max(top_k * 5, top_k),
        )

        if not candidates:
            return {
                "query_text": query_text,
                "results": [],
                "meta": {"reason": "no candidates from embeddings"},
            }

        llm = get_llm_client().build()
        chain = MATCH_PROMPT | llm.with_structured_output(LLMMatchOut)

        query_kw = keywords_for_matching(query_keywords)
        query_kw_json = json.dumps(query_kw, ensure_ascii=False)

        opp_ids = [oid for (oid, _) in candidates]
        opps = opp_dao.read_opportunities_by_ids_with_relations(opp_ids)
        opp_map = {o.opportunity_id: o for o in opps}

        rows = []
        for opp_id, domain_sim in candidates:
            opp = opp_map.get(opp_id)
            if not opp or not opp.keyword:
                continue
            if not _matches_filters(opp, agency=agency, category=category, status=status):
                continue

            opp_kw = keywords_for_matching(getattr(opp.keyword, "keywords", {}) or {})
            req_idx = requirements_indexed(opp_kw)
            req_idx_json = json.dumps(req_idx, ensure_ascii=False)

            scored = chain.invoke(
                {
                    "faculty_kw_json": query_kw_json,
                    "requirements_indexed": req_idx_json,
                }
            )

            rows.append(
                {
                    "opportunity_id": opp_id,
                    "title": opp.opportunity_title,
                    "agency": opp.agency_name,
                    "category": opp.category,
                    "status": opp.opportunity_status,
                    "domain_score": float(domain_sim),
                    "llm_score": float(scored.llm_score),
                    "reason": (scored.reason or "").strip(),
                }
            )

        rows.sort(key=lambda r: (r.get("llm_score", 0.0), r.get("domain_score", 0.0)), reverse=True)
        results = rows[:top_k]

        return {
            "query_text": query_text,
            "results": results,
            "meta": {
                "top_k": top_k,
                "filters": {
                    "agency": agency,
                    "category": category,
                    "status": status,
                    "user_urls": user_urls or [],
                },
            },
        }
