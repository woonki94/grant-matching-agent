from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dao.faculty_dao import FacultyDAO
from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from services.context_retrieval.context_generator import ContextGenerator
from test.context_retrieval_test._llm_input_common import norm, parse_team_ids


def main(
    *,
    opportunity_id: str,
    team: str,
    preview_chars: int,
    only: str = "",
    why_only: bool = False,
) -> int:
    oid = norm(opportunity_id)
    team_ids = parse_team_ids(team)
    if not oid:
        raise ValueError("--opportunity-id is required")
    if not team_ids:
        raise ValueError("--team must contain at least one faculty id")

    cgen = ContextGenerator()
    with SessionLocal() as sess:
        fdao = FacultyDAO(sess)
        odao = OpportunityDAO(sess)
        mdao = MatchDAO(sess)

        opp_ctx = odao.read_opportunity_context(oid)
        if not opp_ctx:
            raise ValueError(f"Opportunity not found: {oid}")

        grant_brief_context = cgen.build_grant_context_only(
            sess=sess,
            opportunity_id=oid,
            preview_chars=max(100, int(preview_chars)),
        )

        match_rows_by_faculty: Dict[int, Dict[str, Any]] = {}
        for fid in list(team_ids or []):
            row = mdao.get_match_for_faculty_opportunity(
                faculty_id=int(fid),
                opportunity_id=str(oid),
            )
            if row:
                match_rows_by_faculty[int(fid)] = dict(row)

        faculty_contexts_by_id: Dict[int, Dict[str, Any]] = {}
        for fid in list(team_ids or []):
            fac_ctx = dict(fdao.get_faculty_keyword_context(int(fid)) or {})
            fac_obj = fdao.get_with_relations_by_id(int(fid))
            pub_title_by_id: Dict[int, str] = {}
            pub_year_by_id: Dict[int, int] = {}
            if fac_obj is not None:
                for pub in list(getattr(fac_obj, "publications", []) or []):
                    try:
                        pid = int(getattr(pub, "id"))
                    except Exception:
                        continue
                    title = str(getattr(pub, "title", "") or "").strip()
                    year = getattr(pub, "year", None)
                    if title:
                        pub_title_by_id[pid] = title
                    try:
                        if year is not None:
                            pub_year_by_id[pid] = int(year)
                    except Exception:
                        pass
            if pub_title_by_id:
                fac_ctx["publication_title_by_id"] = pub_title_by_id
            if pub_year_by_id:
                fac_ctx["publication_year_by_id"] = pub_year_by_id
            if fac_ctx:
                faculty_contexts_by_id[int(fid)] = dict(fac_ctx)

        stage_inputs = cgen.build_group_justification_stage_inputs_from_contexts(
            opp_ctx=dict(opp_ctx or {}),
            team_ids=[int(x) for x in list(team_ids or [])],
            match_rows_by_faculty=dict(match_rows_by_faculty or {}),
            faculty_contexts_by_id=dict(faculty_contexts_by_id or {}),
            grant_brief_context=dict(grant_brief_context or {}),
        )

    key = norm(only)
    if why_only or key in {"why_input", "why_inputs", "why_only"}:
        payload = {
            "why_working_input": stage_inputs.get("why_working_input") or {},
            "why_not_working_input": stage_inputs.get("why_not_working_input") or {},
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0

    if key:
        print(json.dumps(stage_inputs.get(key) or {}, ensure_ascii=False, indent=2, default=str))
        return 0

    payload = {
        "meta": {
            "opportunity_id": oid,
            "team_faculty_ids": [int(x) for x in list(team_ids or [])],
            "team_size": len(list(team_ids or [])),
            "matched_rows_found": len(dict(match_rows_by_faculty or {})),
        },
        "stage_inputs": stage_inputs,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke test: retrieval-only stage inputs for group justification.",
    )
    parser.add_argument("--opportunity-id", required=True)
    parser.add_argument("--team", required=True, help="Comma-separated faculty ids, e.g. 71,73,108")
    parser.add_argument("--preview-chars", type=int, default=50_000)
    parser.add_argument(
        "--only",
        default="",
        help=(
            "Optional stage key to print only one block. "
            "Examples: grant_brief_input, team_role_input, why_working_input, "
            "why_not_working_input, recommendation_input_template"
        ),
    )
    parser.add_argument(
        "--why-only",
        action="store_true",
        help="Print only why_working_input and why_not_working_input.",
    )
    args = parser.parse_args()

    raise SystemExit(
        main(
            opportunity_id=str(args.opportunity_id),
            team=str(args.team),
            preview_chars=int(args.preview_chars),
            only=str(args.only or ""),
            why_only=bool(args.why_only),
        )
    )
