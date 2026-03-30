from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dao.faculty_dao import FacultyDAO
from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from services.context_retrieval.context_generator import ContextGenerator
from utils.keyword_utils import extract_requirement_specs


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _short(text: Any, max_chars: int) -> str:
    s = _norm(text)
    if len(s) <= int(max_chars):
        return s
    return s[: int(max_chars)].rstrip()


def _parse_team(raw: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            fid = int(token)
        except Exception:
            continue
        if fid in seen:
            continue
        seen.add(fid)
        out.append(fid)
    return out


def _merge_team_coverage(
    *,
    team_ids: List[int],
    member_coverages: Dict[int, Dict[str, Dict[int, float]]],
) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {"application": {}, "research": {}}
    for fid in list(team_ids or []):
        cov = dict(member_coverages.get(int(fid)) or {})
        for sec in ("application", "research"):
            sec_cov = dict(cov.get(sec) or {})
            for idx, score in list(sec_cov.items()):
                try:
                    ridx = int(idx)
                except Exception:
                    continue
                prev = float(out[sec].get(ridx, 0.0))
                out[sec][ridx] = max(prev, _safe_float(score, 0.0))
    return out


def _build_group_evidence_text(
    *,
    sess,
    context_generator: ContextGenerator,
    faculty_dao: FacultyDAO,
    match_dao: MatchDAO,
    opp_id: str,
    team_ids: List[int],
    per_member_max_chars: int,
) -> str:
    score_map = match_dao.list_matches_for_opportunity_by_faculty_ids(
        opportunity_id=str(opp_id),
        faculty_ids=[int(x) for x in list(team_ids or [])],
    )

    lines: List[str] = [f"GROUP EVIDENCE PACK: opportunity_id={str(opp_id)}"]
    for fid in list(team_ids or []):
        fac = faculty_dao.get_with_relations_by_id(int(fid))
        if not fac:
            lines.append(f"\nMEMBER {int(fid)} EVIDENCE\n(missing faculty record)")
            continue

        score_row = dict(score_map.get(int(fid)) or {})
        domain_score = _safe_float(score_row.get("domain_score"), 0.0)
        llm_score = _safe_float(score_row.get("llm_score"), 0.0)
        evidence_text = context_generator.build_faculty_recommendation_source_linked_text(
            sess=sess,
            fac=fac,
            top_rows=[(str(opp_id), float(domain_score), float(llm_score))],
            max_requirements=3,
            grant_evidence_per_requirement=2,
            faculty_evidence_per_requirement=2,
        )
        lines.append(
            f"\nMEMBER {int(fid)} EVIDENCE\n{_short(evidence_text, max_chars=int(per_member_max_chars))}"
        )

    return "\n".join(lines).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test: build group justification input payloads only (no LLM calls)."
    )
    parser.add_argument("--opp-id", required=True, help="Opportunity ID")
    parser.add_argument(
        "--team",
        required=True,
        help="Comma-separated faculty IDs, e.g. 71,73,108",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=500,
        help="Rows used when building per-opportunity member coverage (default: 500)",
    )
    parser.add_argument(
        "--per-member-evidence-chars",
        type=int,
        default=12_000,
        help="Max chars per member evidence block in evidence_text (default: 12000)",
    )
    args = parser.parse_args()

    opp_id = _norm(args.opp_id)
    team_ids = _parse_team(args.team)
    if not opp_id:
        raise SystemExit("--opp-id is required")
    if not team_ids:
        raise SystemExit("--team must contain at least one valid faculty id")

    cg = ContextGenerator()
    with SessionLocal() as sess:
        odao = OpportunityDAO(sess)
        fdao = FacultyDAO(sess)
        mdao = MatchDAO(sess)

        opp_ctx = odao.read_opportunity_context(opp_id)
        if not opp_ctx:
            raise SystemExit(f"Opportunity not found: {opp_id}")

        fac_ctxs: List[Dict[str, Any]] = []
        for fid in list(team_ids or []):
            ctx = fdao.get_faculty_keyword_context(int(fid))
            if ctx:
                fac_ctxs.append(ctx)

        member_cov_full = cg.build_member_coverages_for_opportunity(
            sess=sess,
            opportunity_id=opp_id,
            limit_rows=max(1, int(args.limit_rows)),
        )
        member_coverages = {
            int(fid): dict(member_cov_full.get(int(fid)) or {"application": {}, "research": {}})
            for fid in list(team_ids or [])
        }
        final_coverage = _merge_team_coverage(team_ids=team_ids, member_coverages=member_coverages)

        group_ctx = cg.build_group_matching_context(
            opp_ctx=dict(opp_ctx),
            fac_ctxs=[dict(x) for x in fac_ctxs],
            coverage=final_coverage,
            member_coverages=member_coverages,
            group_meta={},
        )
        requirements = extract_requirement_specs(dict(opp_ctx or {}))
        grant_block = dict(group_ctx.get("grant") or {})
        team_block = list(group_ctx.get("team") or [])
        coverage_block = dict(group_ctx.get("coverage") or {})
        grant_link = (
            f"https://simpler.grants.gov/opportunity/{grant_block.get('id')}"
            if grant_block.get("id")
            else ""
        )

        evidence_text = _build_group_evidence_text(
            sess=sess,
            context_generator=cg,
            faculty_dao=fdao,
            match_dao=mdao,
            opp_id=opp_id,
            team_ids=team_ids,
            per_member_max_chars=max(500, int(args.per_member_evidence_chars)),
        )

    payload = {
        "meta": {
            "opportunity_id": grant_block.get("id"),
            "team_faculty_ids": [m.get("faculty_id") for m in team_block],
            "team_size": len(team_block),
            "evidence_chars": len(evidence_text),
        },
        "evidence_text": evidence_text,
        "inputs": {
            "grant_brief_input": {
                "grant": {
                    "id": grant_block.get("id"),
                    "title": grant_block.get("title"),
                    "agency": grant_block.get("agency"),
                    "summary": grant_block.get("summary"),
                    "link": grant_link,
                    "keywords": grant_block.get("keywords"),
                },
                "requirements": requirements,
                "evidence_text": evidence_text,
            },
            "team_role_input": {
                "grant": {
                    "id": grant_block.get("id"),
                    "title": grant_block.get("title"),
                    "keywords": grant_block.get("keywords"),
                },
                "requirements": requirements,
                "team": team_block,
                "evidence_text": evidence_text,
            },
            "why_working_input": {
                "grant": {
                    "id": grant_block.get("id"),
                    "title": grant_block.get("title"),
                    "keywords": grant_block.get("keywords"),
                },
                "requirements": requirements,
                "team": team_block,
                "team_final_coverage": coverage_block,
                "evidence_text": evidence_text,
            },
            "why_not_input": {
                "grant": {
                    "id": grant_block.get("id"),
                    "title": grant_block.get("title"),
                    "keywords": grant_block.get("keywords"),
                },
                "requirements": requirements,
                "team": team_block,
                "team_final_coverage": coverage_block,
                "evidence_text": evidence_text,
            },
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

