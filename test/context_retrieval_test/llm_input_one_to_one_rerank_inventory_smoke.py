from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.db_conn import SessionLocal
from services.context_retrieval.context_generator import ContextGenerator
from test.context_retrieval_test._llm_input_common import norm


def main(*, faculty_id: int, opportunity_id: str, k: int | None) -> int:
    oid = norm(opportunity_id)
    if not oid:
        raise ValueError("--opportunity-id is required")

    cgen = ContextGenerator()
    limit_k = int(k) if k is not None and int(k) > 0 else None
    with SessionLocal() as sess:
        faculty_inventory = cgen.build_rerank_keyword_inventory_for_faculty(
            sess=sess,
            faculty_id=int(faculty_id),
            k=limit_k,
        )

    faculty_grants = list((faculty_inventory.get("grants") or []))
    pair_in_faculty_view = any(str(g.get("opportunity_id") or "") == oid for g in faculty_grants)
    target_grant = next(
        (dict(g or {}) for g in faculty_grants if str((g or {}).get("opportunity_id") or "") == oid),
        None,
    )

    payload = {
        "input": {
            "faculty_id": int(faculty_id),
            "opportunity_id": oid,
            "k_limit": limit_k,
        },
        "faculty_inventory": faculty_inventory,
        "pair_presence": {
            "pair_in_faculty_inventory": pair_in_faculty_view,
        },
        "target_grant_from_faculty_inventory": target_grant,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke test: one-to-one rerank inventory from faculty side only.",
    )
    parser.add_argument("--faculty-id", type=int, required=True)
    parser.add_argument("--opportunity-id", required=True)
    parser.add_argument("--k", type=int, default=0, help="Optional cap (>0). Default 0 means all matches.")
    args = parser.parse_args()

    raise SystemExit(
        main(
            faculty_id=int(args.faculty_id),
            opportunity_id=str(args.opportunity_id),
            k=int(args.k),
        )
    )
