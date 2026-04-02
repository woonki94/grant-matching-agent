from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dao.faculty_dao import FacultyDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from services.context_retrieval.context_generator import ContextGenerator
from services.keywords.keyword_generator import _KeywordGeneratorBase
from services.prompts.keyword_prompts import (
    FACULTY_APPLICATION_CANDIDATE_PROMPT,
    FACULTY_APPLICATION_KEYWORDS_PROMPT,
    FACULTY_RESEARCH_CANDIDATE_PROMPT,
    FACULTY_RESEARCH_KEYWORDS_PROMPT,
    FACULTY_SPECIALIZATION_WEIGHT_PROMPT,
    OPP_APPLICATION_CANDIDATE_PROMPT,
    OPP_APPLICATION_KEYWORDS_PROMPT,
    OPP_RESEARCH_CANDIDATE_PROMPT,
    OPP_RESEARCH_KEYWORDS_PROMPT,
    OPP_SPECIALIZATION_WEIGHT_PROMPT,
)
from test.context_retrieval_test._llm_input_common import norm
from utils.keyword_utils import (
    apply_weighted_specializations,
    build_specialization_source_catalog,
    coerce_keyword_sections,
    specialization_text_sections,
)
from utils.payload_sanitizer import sanitize_for_postgres


def _build_source_mapping_input_from_live_keyword_pipeline(
    *,
    context: Dict[str, Any],
    prompt_pack: Tuple[Any, Any, Any, Any, Any],
    max_catalog_items: int,
    max_excerpt_chars: int,
) -> Dict[str, Any]:
    (
        research_candidate_prompt,
        application_candidate_prompt,
        research_keywords_prompt,
        application_keywords_prompt,
        weight_prompt,
    ) = prompt_pack
    (
        research_candidates_chain,
        application_candidates_chain,
        research_keywords_chain,
        application_keywords_chain,
        weight_chain,
    ) = _KeywordGeneratorBase.build_keyword_chain(
        research_candidate_prompt,
        application_candidate_prompt,
        research_keywords_prompt,
        application_keywords_prompt,
        weight_prompt,
    )

    sanitized_context = sanitize_for_postgres(dict(context or {}))
    context_json = json.dumps(sanitized_context, ensure_ascii=False)

    research_candidates = (
        getattr(research_candidates_chain.invoke({"context_json": context_json}), "candidates", None) or []
    )[:50]
    application_candidates = (
        getattr(application_candidates_chain.invoke({"context_json": context_json}), "candidates", None) or []
    )[:50]

    research_out = research_keywords_chain.invoke(
        {
            "context_json": context_json,
            "candidates": "\n".join(f"- {c}" for c in research_candidates),
        }
    )
    application_out = application_keywords_chain.invoke(
        {
            "context_json": context_json,
            "candidates": "\n".join(f"- {c}" for c in application_candidates),
        }
    )
    kw_dict = coerce_keyword_sections(
        {
            "research": research_out.model_dump(),
            "application": application_out.model_dump(),
        }
    )
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

    # This is the exact specialization text shape used before source-evidence mapping.
    spec_json = specialization_text_sections(kw_weighted)
    # max_excerpt_chars <= 0 means "no truncation" for this smoke test.
    excerpt_cap = int(max_excerpt_chars) if int(max_excerpt_chars) > 0 else 1_000_000_000
    source_catalog = build_specialization_source_catalog(
        dict(sanitized_context or {}),
        max_items=max(1, int(max_catalog_items)),
        max_excerpt_chars=excerpt_cap,
    )

    return {
        "spec_json": spec_json,
        "source_catalog_json": source_catalog,
        "meta": {
            "candidate_research_count": len(research_candidates),
            "candidate_application_count": len(application_candidates),
            "specialization_count": len(spec_json.get("research") or [])
            + len(spec_json.get("application") or []),
            "source_catalog_count": len(source_catalog),
        },
    }


def _build_faculty_payload(
    *,
    cgen: ContextGenerator,
    sess,
    faculty_id: int,
    max_catalog_items: int,
    max_excerpt_chars: int,
) -> Dict[str, Any]:
    fdao = FacultyDAO(sess)
    fac = fdao.get_with_relations_by_id(int(faculty_id))
    if not fac:
        raise ValueError(f"Faculty not found: {faculty_id}")

    context = cgen.build_faculty_basic_context(fac)
    source_input = _build_source_mapping_input_from_live_keyword_pipeline(
        context=context,
        prompt_pack=(
            FACULTY_RESEARCH_CANDIDATE_PROMPT,
            FACULTY_APPLICATION_CANDIDATE_PROMPT,
            FACULTY_RESEARCH_KEYWORDS_PROMPT,
            FACULTY_APPLICATION_KEYWORDS_PROMPT,
            FACULTY_SPECIALIZATION_WEIGHT_PROMPT,
        ),
        max_catalog_items=max_catalog_items,
        max_excerpt_chars=max_excerpt_chars,
    )
    meta = dict(source_input.get("meta") or {})

    return {
        "entity": "faculty",
        "faculty_id": int(faculty_id),
        "specialization_count": int(meta.get("specialization_count") or 0),
        "source_catalog_count": int(meta.get("source_catalog_count") or 0),
        "llm_source_mapping_input": {
            "spec_json": source_input.get("spec_json") or {"research": [], "application": []},
            "source_catalog_json": source_input.get("source_catalog_json") or [],
        },
        "meta": meta,
    }


def _build_opportunity_payload(
    *,
    cgen: ContextGenerator,
    sess,
    opportunity_id: str,
    max_catalog_items: int,
    max_excerpt_chars: int,
) -> Dict[str, Any]:
    oid = norm(opportunity_id)
    if not oid:
        raise ValueError("--opportunity-id is required")

    odao = OpportunityDAO(sess)
    opps = odao.read_opportunities_by_ids_with_relations([oid])
    opp = opps[0] if opps else None
    if not opp:
        raise ValueError(f"Opportunity not found: {oid}")

    context = cgen.build_opportunity_basic_context(opp)
    source_input = _build_source_mapping_input_from_live_keyword_pipeline(
        context=context,
        prompt_pack=(
            OPP_RESEARCH_CANDIDATE_PROMPT,
            OPP_APPLICATION_CANDIDATE_PROMPT,
            OPP_RESEARCH_KEYWORDS_PROMPT,
            OPP_APPLICATION_KEYWORDS_PROMPT,
            OPP_SPECIALIZATION_WEIGHT_PROMPT,
        ),
        max_catalog_items=max_catalog_items,
        max_excerpt_chars=max_excerpt_chars,
    )
    meta = dict(source_input.get("meta") or {})

    return {
        "entity": "opportunity",
        "opportunity_id": oid,
        "specialization_count": int(meta.get("specialization_count") or 0),
        "source_catalog_count": int(meta.get("source_catalog_count") or 0),
        "llm_source_mapping_input": {
            "spec_json": source_input.get("spec_json") or {"research": [], "application": []},
            "source_catalog_json": source_input.get("source_catalog_json") or [],
        },
        "meta": meta,
    }


def main(
    *,
    faculty_id: int | None,
    opportunity_id: str | None,
    max_catalog_items: int,
    max_excerpt_chars: int,
) -> int:
    if faculty_id is None and not opportunity_id:
        raise ValueError("Provide either --faculty-id or --opportunity-id")

    cgen = ContextGenerator()
    with SessionLocal() as sess:
        if faculty_id is not None:
            out = _build_faculty_payload(
                cgen=cgen,
                sess=sess,
                faculty_id=int(faculty_id),
                max_catalog_items=int(max_catalog_items),
                max_excerpt_chars=int(max_excerpt_chars),
            )
        else:
            out = _build_opportunity_payload(
                cgen=cgen,
                sess=sess,
                opportunity_id=str(opportunity_id),
                max_catalog_items=int(max_catalog_items),
                max_excerpt_chars=int(max_excerpt_chars),
            )

    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke test: print live keyword source-mapping input from the same generation chain steps.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--faculty-id", type=int)
    group.add_argument("--opportunity-id")
    parser.add_argument("--max-catalog-items", type=int, default=120)
    parser.add_argument(
        "--max-excerpt-chars",
        type=int,
        default=0,
        help="Max chars per source excerpt; use 0 (default) for no truncation in this smoke test.",
    )
    args = parser.parse_args()

    raise SystemExit(
        main(
            faculty_id=args.faculty_id,
            opportunity_id=args.opportunity_id,
            max_catalog_items=int(args.max_catalog_items),
            max_excerpt_chars=int(args.max_excerpt_chars),
        )
    )
