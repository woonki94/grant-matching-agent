from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dao.faculty_dao import FacultyDAO
from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from services.context_retrieval.context_generator import ContextGenerator
from test.context_retrieval_test._llm_input_common import (
    merge_member_coverages,
    norm,
    parse_team_ids,
    safe_float,
)
from utils.keyword_utils import extract_requirement_specs


def _short(text: Any, max_chars: int) -> str:
    s = norm(text)
    if len(s) <= int(max_chars):
        return s
    return s[: int(max_chars)].rstrip()


def _build_group_evidence_text(
    *,
    sess,
    cgen: ContextGenerator,
    fdao: FacultyDAO,
    mdao: MatchDAO,
    opportunity_id: str,
    team_ids: List[int],
    max_chars_per_member: int,
) -> str:
    rows_by_fid = mdao.list_matches_for_opportunity_by_faculty_ids(
        opportunity_id=str(opportunity_id),
        faculty_ids=[int(x) for x in list(team_ids or [])],
    )

    lines: List[str] = [f"GROUP EVIDENCE PACK: opportunity_id={opportunity_id}"]
    for fid in list(team_ids or []):
        fac = fdao.get_with_relations_by_id(int(fid))
        if not fac:
            lines.append(f"\nMEMBER {int(fid)} EVIDENCE\n(missing faculty)")
            continue

        score_row = dict(rows_by_fid.get(int(fid)) or {})
        domain_score = safe_float(score_row.get("domain_score"), 0.0)
        llm_score = safe_float(score_row.get("llm_score"), 0.0)

        text = cgen.build_faculty_recommendation_source_linked_text(
            sess=sess,
            fac=fac,
            top_rows=[(str(opportunity_id), float(domain_score), float(llm_score))],
            max_requirements=3,
            grant_evidence_per_requirement=2,
            faculty_evidence_per_requirement=2,
        )

        lines.append(
            f"\nMEMBER {int(fid)} EVIDENCE\n{_short(text, max_chars=max_chars_per_member)}"
        )

    return "\n".join(lines).strip()


def main(
    *,
    opportunity_id: str,
    team: str,
    limit_rows: int,
    preview_chars: int,
    evidence_chars_per_member: int,
) -> int:
    oid = norm(opportunity_id)
    team_ids = parse_team_ids(team)
    if not oid:
        raise ValueError("--opportunity-id is required")
    if not team_ids:
        raise ValueError("--team must contain at least one faculty id")

    cgen = ContextGenerator()
    with SessionLocal() as sess:
        odao = OpportunityDAO(sess)
        fdao = FacultyDAO(sess)
        mdao = MatchDAO(sess)

        opp_ctx = odao.read_opportunity_context(oid)
        if not opp_ctx:
            raise ValueError(f"Opportunity not found: {oid}")

        fac_ctxs: List[Dict[str, Any]] = []
        for fid in list(team_ids or []):
            fac_ctx = fdao.get_faculty_keyword_context(int(fid))
            if fac_ctx:
                fac_ctxs.append(dict(fac_ctx))

        member_cov_all = cgen.build_member_coverages_for_opportunity(
            sess=sess,
            opportunity_id=oid,
            limit_rows=max(1, int(limit_rows)),
        )
        member_coverages = {
            int(fid): dict(member_cov_all.get(int(fid)) or {"application": {}, "research": {}})
            for fid in list(team_ids or [])
        }
        merged_coverage = merge_member_coverages(member_coverages)

        group_ctx = cgen.build_group_matching_context_from_contexts(
            opp_ctx=dict(opp_ctx),
            fac_ctxs=list(fac_ctxs or []),
            coverage=merged_coverage,
            member_coverages=member_coverages,
            group_meta={},
        )

        requirements = extract_requirement_specs(dict(opp_ctx or {}))
        grant_block = dict(group_ctx.get("grant") or {})
        team_block = list(group_ctx.get("team") or [])
        coverage_block = dict(group_ctx.get("coverage") or {})
        grant_id = grant_block.get("opportunity_id") or grant_block.get("id")
        grant_title = grant_block.get("opportunity_title") or grant_block.get("title")
        grant_keywords = grant_block.get("keywords")
        grant_link = (
            f"https://simpler.grants.gov/opportunity/{grant_id}"
            if grant_id
            else ""
        )

        grant_context = cgen.build_grant_context_only(
            sess=sess,
            opportunity_id=oid,
            preview_chars=max(100, int(preview_chars)),
        )

        evidence_text = _build_group_evidence_text(
            sess=sess,
            cgen=cgen,
            fdao=fdao,
            mdao=mdao,
            opportunity_id=oid,
            team_ids=team_ids,
            max_chars_per_member=max(500, int(evidence_chars_per_member)),
        )

    payload = {
        "meta": {
            "opportunity_id": grant_id,
            "team_faculty_ids": [m.get("faculty_id") for m in list(team_block or [])],
            "team_size": len(list(team_block or [])),
        },
        "llm_inputs": {
            "grant_brief_input": {
                "grant_context": grant_context,
            },
            "team_role_input": {
                "grant": {
                    "id": grant_id,
                    "title": grant_title,
                    "keywords": grant_keywords,
                },
                "requirements": requirements,
                "team": team_block,
                "evidence_text": evidence_text,
            },
            "why_working_input": {
                "grant": {
                    "id": grant_id,
                    "title": grant_title,
                    "keywords": grant_keywords,
                },
                "requirements": requirements,
                "team": team_block,
                "team_final_coverage": coverage_block,
                "evidence_text": evidence_text,
            },
            "why_not_working_input": {
                "grant": {
                    "id": grant_id,
                    "title": grant_title,
                    "keywords": grant_keywords,
                },
                "requirements": requirements,
                "team": team_block,
                "team_final_coverage": coverage_block,
                "evidence_text": evidence_text,
            },
            "recommender_input_template": {
                "grant": {
                    "id": grant_id,
                    "title": grant_title,
                    "link": grant_link,
                },
                "team_roles": "<output of team_role_input LLM>",
                "why_working": "<output of why_working_input LLM>",
                "why_not_working": "<output of why_not_working_input LLM>",
            },
        },
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke test: retrieval-only LLM input payloads for group justification.",
    )
    parser.add_argument("--opportunity-id", required=True)
    parser.add_argument("--team", required=True, help="Comma-separated faculty ids, e.g. 71,73,108")
    parser.add_argument("--limit-rows", type=int, default=500)
    parser.add_argument("--preview-chars", type=int, default=50_000)
    parser.add_argument("--evidence-chars-per-member", type=int, default=12_000)
    args = parser.parse_args()

    raise SystemExit(
        main(
            opportunity_id=str(args.opportunity_id),
            team=str(args.team),
            limit_rows=int(args.limit_rows),
            preview_chars=int(args.preview_chars),
            evidence_chars_per_member=int(args.evidence_chars_per_member),
        )
    )
