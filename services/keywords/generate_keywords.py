from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../root
sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import re

import boto3
from botocore.exceptions import ClientError
from langchain_core.prompts import ChatPromptTemplate

from config import settings, get_llm_client, get_embedding_client
from dao.faculty_dao import FacultyDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from dto.llm_response_dto import KeywordsOut, CandidatesOut, WeightedSpecsOut
from services.keywords.generate_context import (
    faculty_to_keyword_context,
    opportunity_to_keyword_context,
)
from services.prompts.keyword_prompts import (
    FACULTY_CANDIDATE_PROMPT,
    FACULTY_KEYWORDS_PROMPT,
    FACULTY_SPECIALIZATION_WEIGHT_PROMPT,
    OPP_CANDIDATE_PROMPT,
    OPP_KEYWORDS_PROMPT,
    OPP_SPECIALIZATION_WEIGHT_PROMPT,
)
from utils.content_compressor import cap_extracted_blocks

logger = logging.getLogger(__name__)

# -----------------------------
# Postgres JSONB safety:
# Remove NUL (\x00) and other control chars that Postgres rejects in text/jsonb.
# -----------------------------
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def sanitize_for_postgres(obj: Any) -> Any:
    if isinstance(obj, str):
        obj = obj.replace("\x00", "")
        return _CTRL_RE.sub("", obj)
    if isinstance(obj, list):
        return [sanitize_for_postgres(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize_for_postgres(v) for k, v in obj.items()}
    return obj


# -----------------------------
# S3 loader that does NOT depend on a global prefix
# It uses content_path stored in DB:
#   - "s3://bucket/key"
#   - or "key"
# -----------------------------
def _get_s3_client():
    session = (
        boto3.Session(
            profile_name=settings.aws_profile,
            region_name=settings.aws_region,
        )
        if settings.aws_profile
        else boto3.Session(region_name=settings.aws_region)
    )
    return session.client("s3")


def _parse_s3_bucket_key(content_path: str) -> Tuple[str, str]:
    """
    Returns (bucket, key).
    Supports:
      - s3://bucket/key
      - key  (uses settings.extracted_content_bucket)
    """
    cp = (content_path or "").strip()
    if not cp:
        raise ValueError("empty content_path")

    if cp.startswith("s3://"):
        rest = cp[5:]
        parts = rest.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"invalid s3 uri: {content_path}")
        return parts[0], parts[1]

    if not settings.extracted_content_bucket:
        raise RuntimeError(
            "extracted_content_bucket is required to load S3 content when content_path is not an s3:// URI"
        )
    return settings.extracted_content_bucket, cp.lstrip("/")


def _load_extracted_content_s3(
    rows: List[Any],
    url_attr: str,
    title_attr: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Loads extracted text for rows whose extract_status='done' and extract_error is empty.
    Reads the actual text from S3 using r.content_path as key (or s3:// URI).
    """
    if not rows:
        return []

    s3 = _get_s3_client()
    out: List[Dict[str, Any]] = []

    for r in rows:
        if getattr(r, "extract_status", None) != "done":
            continue
        if getattr(r, "extract_error", None):
            continue

        content_path = getattr(r, "content_path", None)
        if not content_path:
            continue

        try:
            bucket, key = _parse_s3_bucket_key(str(content_path))
            resp = s3.get_object(Bucket=bucket, Key=key)
            text = resp["Body"].read().decode("utf-8", errors="ignore").strip()
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404"):
                continue
            raise
        except Exception:
            continue

        if not text:
            continue

        item: Dict[str, Any] = {
            "url": getattr(r, url_attr, None),
            "content": text,
        }
        if title_attr:
            item["title"] = getattr(r, title_attr, None)

        out.append(item)

    return out


def _refresh_extracted_blocks_from_s3(context: dict, obj: Any) -> dict:
    """
    context_builder() may have tried to load extracted content using older logic.
    This function ensures the extracted blocks come from S3 using content_path keys.
    Works for:
      - Opportunity: attachments_extracted, additional_info_extracted
      - Faculty: additional_links_extracted / additional_info_extracted (depending on your context keys)
    """
    out = dict(context)

    # Opportunity convention
    if hasattr(obj, "attachments") and "attachments_extracted" in out:
        out["attachments_extracted"] = _load_extracted_content_s3(
            getattr(obj, "attachments") or [],
            url_attr="file_download_path",
            title_attr="file_name",
        )

    if hasattr(obj, "additional_info") and "additional_info_extracted" in out:
        out["additional_info_extracted"] = _load_extracted_content_s3(
            getattr(obj, "additional_info") or [],
            url_attr="additional_info_url",
            title_attr=None,
        )

    # Faculty conventions (depending on your schema/context)
    if hasattr(obj, "additional_info") and "additional_links_extracted" in out:
        out["additional_links_extracted"] = _load_extracted_content_s3(
            getattr(obj, "additional_info") or [],
            url_attr="additional_info_url",
            title_attr=None,
        )

    return out


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


def _apply_weighted_specializations(*, keywords: dict, weighted: WeightedSpecsOut) -> dict:
    out = dict(keywords)
    out["research"] = dict(out.get("research") or {})
    out["application"] = dict(out.get("application") or {})

    out["research"]["specialization"] = [x.model_dump() for x in (weighted.research or [])]
    out["application"]["specialization"] = [x.model_dump() for x in (weighted.application or [])]
    return out


def _extract_domains_from_keywords(kw: dict) -> Tuple[List[str], List[str]]:
    """
    Returns (research_domains, application_domains) from your KeywordsOut structure.
    """
    r = (kw.get("research") or {}).get("domain") or []
    a = (kw.get("application") or {}).get("domain") or []
    return list(r), list(a)


def _embed_domain_bucket(domains: List[str]) -> Optional[List[float]]:
    """
    Bedrock-only embeddings.
    Uses a single query embedding of the joined domains.
    """
    domains = [d.strip() for d in (domains or []) if d and str(d).strip()]
    if not domains:
        return None

    emb = get_embedding_client().build()
    text = " ; ".join(domains)
    vec = emb.embed_query(text)
    return vec if vec else None


def generate_keywords(
    obj: Any,
    *,
    context_builder,
    candidates_chain,
    keywords_chain,
    weight_chain,
) -> Tuple[dict, dict]:
    # Build context
    context = context_builder(obj)

    # Force extracted blocks to come from S3 (content_path keys)
    context = _refresh_extracted_blocks_from_s3(context, obj)

    # Cap huge extracted blobs (especially attachments)
    if "attachments_extracted" in context:
        context["attachments_extracted"] = cap_extracted_blocks(
            context["attachments_extracted"],
            max_total_chars=18_000,
            max_per_doc_chars=2_000,
        )

    # Sanitize context before serializing / sending to LLM (safe + avoids hidden NULs)
    context = sanitize_for_postgres(context)

    context_json = json.dumps(context, ensure_ascii=False)

    # Step 1: candidates
    cand_out: CandidatesOut = candidates_chain.invoke({"context_json": context_json})
    candidates = (cand_out.candidates or [])[:50]

    # Step 2: structured keywords
    kw_out: KeywordsOut = keywords_chain.invoke(
        {
            "context_json": context_json,
            "candidates": "\n".join(f"- {c}" for c in candidates),
        }
    )
    kw_dict = kw_out.model_dump()

    for k in ("research", "application"):
        if isinstance(kw_dict.get(k), str):
            kw_dict[k] = json.loads(kw_dict[k])

    # Step 3: weight specializations
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

    kw_weighted = _apply_weighted_specializations(keywords=kw_dict, weighted=weighted_out)

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

    # Final sanitization for Postgres JSONB safety
    kw_weighted = sanitize_for_postgres(kw_weighted)
    raw_debug = sanitize_for_postgres(raw_debug)

    return kw_weighted, raw_debug


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate keywords for faculty/opportunities (Bedrock-only)")
    parser.add_argument("--limit", type=int, default=0, help="Max number of records to process (0 = no limit)")
    parser.add_argument("--faculty-only", action="store_true", help="Only generate faculty keywords")
    parser.add_argument("--opp-only", action="store_true", help="Only generate opportunity keywords")

    args = parser.parse_args()

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

        source_model = settings.bedrock_model_id
        embed_model = settings.bedrock_embed_model_id

        if run_faculty:
            for fac in _apply_limit(fac_dao.iter_faculty_missing_keywords()):
                faculty_keywords, faculty_keywords_raw = generate_keywords(
                    fac,
                    context_builder=faculty_to_keyword_context,
                    candidates_chain=faculty_cand_chain,
                    keywords_chain=faculty_kw_chain,
                    weight_chain=faculty_w_chain,
                )

                # Extra safety (in case dao serializes differently)
                faculty_keywords = sanitize_for_postgres(faculty_keywords)
                faculty_keywords_raw = sanitize_for_postgres(faculty_keywords_raw)

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

                r_domains, a_domains = _extract_domains_from_keywords(faculty_keywords)
                r_vec = _embed_domain_bucket(r_domains)
                a_vec = _embed_domain_bucket(a_domains)

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
            for opp in _apply_limit(opp_dao.iter_opportunity_missing_keywords()):
                opportunity_keywords, opportunity_keywords_raw = generate_keywords(
                    opp,
                    context_builder=opportunity_to_keyword_context,
                    candidates_chain=opp_cand_chain,
                    keywords_chain=opp_kw_chain,
                    weight_chain=opp_w_chain,
                )

                # Extra safety (in case dao serializes differently)
                opportunity_keywords = sanitize_for_postgres(opportunity_keywords)
                opportunity_keywords_raw = sanitize_for_postgres(opportunity_keywords_raw)

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

                r_domains, a_domains = _extract_domains_from_keywords(opportunity_keywords)
                r_vec = _embed_domain_bucket(r_domains)
                a_vec = _embed_domain_bucket(a_domains)

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
