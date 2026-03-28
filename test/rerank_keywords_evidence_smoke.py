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
from logging_setup import setup_logging
from utils.keyword_utils import keyword_inventory_for_rerank


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _grant_keyword_payload(grant_ctx: Dict[str, Any]) -> Dict[str, Any]:
    inv = keyword_inventory_for_rerank(dict((grant_ctx or {}).get("keywords") or {}))
    return {
        "opportunity_id": grant_ctx.get("opportunity_id"),
        "title": grant_ctx.get("title"),
        "grant_domain_keywords": inv.get("domain") or [],
        "grant_specialization_keywords": inv.get("specialization") or {},
    }


def _run_opportunity_mode(*, sess, opportunity_id: str, k: int) -> Dict[str, Any]:
    oid = _norm(opportunity_id)
    if not oid:
        raise ValueError("--opportunity-id is required")

    top_k = max(1, int(k))
    odao = OpportunityDAO(sess)
    mdao = MatchDAO(sess)
    fdao = FacultyDAO(sess)

    grant_ctx = odao.read_opportunity_context(oid)
    if not grant_ctx:
        raise ValueError(f"Opportunity not found: {oid}")

    match_rows = mdao.list_matches_for_opportunity(oid, limit=top_k)

    payload = {
        "grant": _grant_keyword_payload(grant_ctx),
        "faculty": [],
    }

    for row in list(match_rows or []):
        faculty_id = int(row.get("faculty_id"))
        fac_ctx = fdao.get_faculty_keyword_context(faculty_id) or {}
        fkw = keyword_inventory_for_rerank(dict((fac_ctx or {}).get("keywords") or {}))
        payload["faculty"].append(
            {
                "domain_score": float(row.get("domain_score") or 0.0),
                "llm_score": float(row.get("llm_score") or 0.0),
                "domain_keywords": fkw.get("domain") or [],
                "specialization_keywords": fkw.get("specialization") or {},
            }
        )

    return payload


def _run_faculty_mode(*, sess, faculty_id: int, k: int) -> Dict[str, Any]:
    fid = int(faculty_id)
    top_k = max(1, int(k))

    fdao = FacultyDAO(sess)
    mdao = MatchDAO(sess)
    odao = OpportunityDAO(sess)

    fac_ctx = fdao.get_faculty_keyword_context(fid)
    if not fac_ctx:
        raise ValueError(f"Faculty not found: {fid}")

    fkw = keyword_inventory_for_rerank(dict((fac_ctx or {}).get("keywords") or {}))
    top_rows = mdao.top_matches_for_faculty(fid, k=top_k)

    grants: List[Dict[str, Any]] = []
    for grant_id, domain_score, llm_score in list(top_rows or []):
        oid = str(grant_id)
        grant_ctx = odao.read_opportunity_context(oid)
        if not grant_ctx:
            continue
        g = _grant_keyword_payload(grant_ctx)
        g["domain_score"] = float(domain_score or 0.0)
        g["llm_score"] = float(llm_score or 0.0)
        grants.append(g)

    return {
        "faculty": {
            "faculty_id": fid,
            "name": fac_ctx.get("name"),
            "domain_keywords": fkw.get("domain") or [],
            "specialization_keywords": fkw.get("specialization") or {},
        },
        "grants": grants,
    }


def main(*, opportunity_id: str | None, faculty_id: int | None, k: int) -> int:
    with SessionLocal() as sess:
        if faculty_id is not None:
            payload = _run_faculty_mode(sess=sess, faculty_id=int(faculty_id), k=k)
        else:
            payload = _run_opportunity_mode(sess=sess, opportunity_id=str(opportunity_id or ""), k=k)

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    setup_logging("justification")
    parser = argparse.ArgumentParser(
        description="Keyword inventory smoke test: opportunity->matches or faculty->top grants"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--opportunity-id")
    group.add_argument("--faculty-id", type=int)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    raise SystemExit(
        main(
            opportunity_id=args.opportunity_id,
            faculty_id=args.faculty_id,
            k=args.k,
        )
    )
