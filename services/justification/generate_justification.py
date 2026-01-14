from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from textwrap import fill
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text
from sqlalchemy.orm import selectinload

from dao.match_dao import MatchDAO
from dto.llm_response_dto import FacultyRecsOut
from services.prompts.justification_prompts import FACULTY_RECS_PROMPT



from config import OPENAI_MODEL, OPENAI_API_KEY
from db.db_conn import SessionLocal
from db.models.faculty import Faculty
from dao.opportunity_dao import OpportunityDAO
from services.keywords.generate_context import (
    faculty_to_keyword_context,
    opportunity_to_keyword_context,
)
from utils.content_compressor import cap_extracted_blocks, cap_fac, cap_opp

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

def llm_label(llm_score: float) -> str:
    if llm_score < 0.30:
        return "sucks"
    if llm_score < 0.50:
        return "bad"
    if llm_score < 0.70:
        return "good"
    if llm_score < 0.85:
        return "great"
    return "fantastic"


def _w(s: str, width: int = 92, indent: str = "      ") -> str:
    return fill((s or "").strip(), width=width, subsequent_indent=indent)


def print_faculty_recs(out, email: str, *, width: int = 92, show_full_id: bool = True) -> None:
    print("\n" + "=" * width)
    print(f"Faculty: {out.faculty_name}  <{email}>")
    print(f"Top {len(out.recommendations)} opportunities:")
    print("-" * width)

    for i, rec in enumerate(out.recommendations, start=1):
        label = llm_label(float(rec.llm_score))

        oid = rec.opportunity_id
        oid_disp = oid if show_full_id else (oid[:8] + "…" + oid[-6:])

        score_line = f"domain={rec.domain_score:.2f} | llm={rec.llm_score:.2f} | label={label}"
        print(f"{i:>2}. {rec.title}")
        print(f"    ID: {oid_disp}")
        if getattr(rec, "agency", None):
            print(f"    Agency: {rec.agency}")
        print(f"    Scores: {score_line}")

        print("    Why it matches:")
        for b in rec.why_good_match:
            print("      • " + _w(b, width=width, indent="        ").lstrip())

        print("    Suggested pitch:")
        print("      " + _w(rec.suggested_pitch, width=width, indent="        ").lstrip())
        print("-" * width)

def main(email: str, k: int) -> None:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    chain = FACULTY_RECS_PROMPT | llm.with_structured_output(FacultyRecsOut)

    with SessionLocal() as sess:
        opp_dao = OpportunityDAO(sess)
        match_dao = MatchDAO(sess)

        # 1) Fetch faculty by email (+ relations, since your context builder uses them)
        fac = (
            sess.query(Faculty)
            .options(
                selectinload(Faculty.additional_info),
                selectinload(Faculty.publications),
                selectinload(Faculty.keyword),
            )
            .filter(Faculty.email == email)
            .one_or_none()
        )

        if not fac:
            print(f"No faculty found with email: {email}")
            return

        # 2) Get top-K opp ids from match_results
        rows = match_dao.top_matches_for_faculty(
            faculty_id=fac.faculty_id,
            k=k,
        )

        opp_ids = [gid for (gid, _, _) in rows]
        score_map = {gid: {"domain_score": d, "llm_score": l} for (gid, d, l) in rows}

        if not opp_ids:
            print(f"No matches found for {fac.name} ({email}).")
            return

        # 3) Batch fetch opportunities (+ relations)
        opps = opp_dao.read_opportunities_by_ids_with_relations(opp_ids)
        opp_map = {o.opportunity_id: o for o in opps}

        # 4) Build payloads
        fac_ctx = cap_fac(faculty_to_keyword_context(fac))
        opp_payloads: List[Dict[str, Any]] = []

        for oid in opp_ids:
            opp = opp_map.get(oid)
            if not opp:
                continue
            ctx = cap_opp(opportunity_to_keyword_context(opp))
            scores = score_map.get(oid, {"domain_score": None, "llm_score": None})

            opp_payloads.append({
                "opportunity_id": oid,
                "opportunity_title": ctx.get("opportunity_title"),
                "agency_name": ctx.get("agency_name"),
                "category": ctx.get("category"),
                "opportunity_status": ctx.get("opportunity_status"),
                "summary_description": ctx.get("summary_description"),
                "attachments_extracted": ctx.get("attachments_extracted", []),
                "additional_info_extracted": ctx.get("additional_info_extracted", []),

                "domain_score": scores["domain_score"],
                "llm_score": scores["llm_score"],
            })

        # 5) One LLM call per faculty
        out: FacultyRecsOut = chain.invoke({
            "faculty_json": json.dumps(fac_ctx, ensure_ascii=False),
            "opps_json": json.dumps(opp_payloads, ensure_ascii=False),
        })

        print_faculty_recs(out, email, show_full_id=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate top opportunity recommendations for a faculty by email")
    parser.add_argument("--email", required=True, help="Faculty email address (must exist in DB)")
    parser.add_argument("--k", type=int, default=5, help="Top-K opportunities to explain (default=5)")
    args = parser.parse_args()

    main(email=args.email.strip(), k=args.k)