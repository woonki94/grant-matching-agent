from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dao.faculty_dao import FacultyDAO
from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from services.justification.group_justification_engine import GroupJustificationEngine
from utils.payload_utils import extract_requirement_specs
from utils.report_renderer import render_markdown_report, write_markdown_report
from services.matching.group_match_super_faculty import run_group_match


def _expand_group_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize run_group_match outputs into flat rows:
    {opp_id, team, final_coverage, score}
    Supports both grouped output (selected_teams per opp) and legacy flat rows.
    """
    expanded: List[Dict[str, Any]] = []
    for row in rows:
        if "selected_teams" in row:
            opp_id = row.get("opp_id") or row.get("grant_id")
            for cand in row.get("selected_teams") or []:
                expanded.append(
                    {
                        "opp_id": opp_id,
                        "team": cand.get("team"),
                        "final_coverage": cand.get("final_coverage"),
                        "score": cand.get("score"),
                    }
                )
        else:
            expanded.append(
                {
                    "opp_id": row.get("opp_id") or row.get("grant_id"),
                    "team": row.get("team"),
                    "final_coverage": row.get("final_coverage"),
                    "score": row.get("score"),
                }
            )
    return expanded


def run_justifications_from_group_results_agentic(
    *,
    faculty_emails: Union[str, List[str]],
    team_size: int,
    opp_ids: Optional[List[str]] = None,
    limit_rows: int = 500,
    include_trace: bool = False,
) -> str:
    with SessionLocal() as sess:
        odao = OpportunityDAO(sess)
        fdao = FacultyDAO(sess)
        mdao = MatchDAO(sess)
        engine = GroupJustificationEngine(odao=odao, fdao=fdao)

        email_list = [faculty_emails] if isinstance(faculty_emails, str) else list(faculty_emails)
        group_results = run_group_match(
            faculty_emails=email_list,
            team_size=team_size,
            limit_rows=limit_rows,
            opp_ids=opp_ids,
        )
        normalized_rows = _expand_group_results(group_results)
        if not normalized_rows:
            raise ValueError(f"No group matches found for {faculty_emails}")

        opp_cache: Dict[str, Dict[str, Any]] = {}
        fac_cache: Dict[int, Dict[str, Any]] = {}
        opp_member_cov_cache: Dict[str, Dict[int, Dict[str, Dict[int, float]]]] = {}

        results: List[Dict[str, Any]] = []

        for idx, row in enumerate(normalized_rows):
            opp_id = row["opp_id"]
            team: List[int] = row["team"]
            coverage = row.get("final_coverage")

            if opp_id not in opp_cache:
                opp_cache[opp_id] = odao.get_opportunity_context(opp_id) or {}
            opp_ctx = opp_cache[opp_id]

            if not opp_ctx:
                results.append(
                    {
                        "index": idx,
                        "grant_id": opp_id,
                        "grant_title": None,
                        "grant_link": f"https://simpler.grants.gov/opportunity/{opp_id}",
                        "team": team,
                        "error": "Opportunity not found",
                    }
                )
                continue

            fac_ctxs: List[Dict[str, Any]] = []
            for fid in team:
                if fid not in fac_cache:
                    fac_cache[fid] = fdao.get_faculty_keyword_context(fid) or {}
                if fac_cache[fid]:
                    fac_ctxs.append(fac_cache[fid])

            if opp_id not in opp_member_cov_cache:
                rows = mdao.list_matches_for_opportunity(opp_id, limit=limit_rows) or []
                fac_cov: Dict[int, Dict[str, Dict[int, float]]] = {}
                for match in rows:
                    try:
                        fid = int(match.get("faculty_id"))
                    except Exception:
                        continue
                    covered = match.get("covered") or {}
                    if fid not in fac_cov:
                        fac_cov[fid] = {"application": {}, "research": {}}
                    for sec in ("application", "research"):
                        sec_map = covered.get(sec) if isinstance(covered, dict) else None
                        if not isinstance(sec_map, dict):
                            continue
                        for k, v in sec_map.items():
                            try:
                                req_idx = int(k)
                                cov_val = float(v)
                            except Exception:
                                continue
                            prev = fac_cov[fid][sec].get(req_idx, 0.0)
                            fac_cov[fid][sec][req_idx] = max(prev, cov_val)
                opp_member_cov_cache[opp_id] = fac_cov

            member_coverages = {
                int(fid): opp_member_cov_cache.get(opp_id, {}).get(int(fid), {"application": {}, "research": {}})
                for fid in team
            }

            if not fac_ctxs:
                results.append(
                    {
                        "index": idx,
                        "grant_id": opp_id,
                        "grant_title": opp_ctx.get("title") or opp_ctx.get("opportunity_title"),
                        "grant_link": f"https://simpler.grants.gov/opportunity/{opp_id}",
                        "team": team,
                        "error": "No faculty contexts found for team",
                    }
                )
                continue

            group_meta = {
                "group_id": row.get("group_id") or row.get("id"),
                "lambda": row.get("lambda"),
                "k": row.get("k"),
                "objective": row.get("objective"),
                "redundancy": row.get("redundancy"),
                "meta": row.get("meta"),
            }

            try:
                justification, trace = engine.run_one(
                    opp_ctx=dict(opp_ctx),
                    fac_ctxs=[dict(f) for f in fac_ctxs],
                    coverage=coverage,
                    member_coverages=member_coverages,
                    group_meta=group_meta,
                    trace={"index": idx, "opp_id": opp_id, "team": team},
                )

                out = {
                    "index": idx,
                    "grant_id": opp_id,
                    "grant_title": opp_ctx.get("title") or opp_ctx.get("opportunity_title"),
                    "grant_link": f"https://simpler.grants.gov/opportunity/{opp_id}",
                    "team": team,
                    "team_members": [
                        {
                            "faculty_id": f.get("faculty_id") or f.get("id"),
                            "faculty_name": f.get("name"),
                            "faculty_email": f.get("email"),
                        }
                        for f in fac_ctxs
                    ],
                    "score": row.get("score"),
                    "final_coverage": coverage,
                    "requirement_specs": extract_requirement_specs(opp_ctx),
                    "justification": justification.model_dump(),
                }
                if include_trace:
                    out["trace"] = trace
                results.append(out)
            except Exception as e:
                results.append(
                    {
                        "index": idx,
                        "grant_id": opp_id,
                        "grant_title": opp_ctx.get("title") or opp_ctx.get("opportunity_title"),
                        "grant_link": f"https://simpler.grants.gov/opportunity/{opp_id}",
                        "team": team,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

        return render_markdown_report(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run group justification generation")
    parser.add_argument(
        "--email",
        action="append",
        required=True,
        help="Faculty email. Repeat --email for multiple entries.",
    )
    parser.add_argument("--team-size", type=int, default=3, help="Team size (default: 3)")
    parser.add_argument("--limit-rows", type=int, default=200, help="Max number of match rows to scan")
    parser.add_argument("--opp-id", action="append", help="Target opportunity id. Repeatable.")
    parser.add_argument(
        "--out-md",
        type=str,
        default=None,
        help="Output markdown file path (default: outputs/justification_reports/auto-generated)",
    )
    parser.add_argument("--include-trace", action="store_true", help="Include trace output in result payload")

    args = parser.parse_args()

    rendered = run_justifications_from_group_results_agentic(
        faculty_emails=args.email,
        team_size=args.team_size,
        opp_ids=args.opp_id,
        limit_rows=args.limit_rows,
        include_trace=args.include_trace,
    )
    out_path = write_markdown_report(PROJECT_ROOT, rendered, args.out_md)
    print(f"Saved markdown report to: {out_path}")
