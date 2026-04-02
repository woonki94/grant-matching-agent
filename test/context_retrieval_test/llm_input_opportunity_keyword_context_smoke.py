from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from services.context_retrieval.context_generator import ContextGenerator
from test.context_retrieval_test._llm_input_common import norm


def main(*, opportunity_id: str) -> int:
    oid = norm(opportunity_id)
    if not oid:
        raise ValueError("--opportunity-id is required")

    cgen = ContextGenerator()
    with SessionLocal() as sess:
        odao = OpportunityDAO(sess)
        opps = odao.read_opportunities_by_ids_with_relations([oid])
        opp = opps[0] if opps else None
        if not opp:
            raise ValueError(f"Opportunity not found: {oid}")

        payload = cgen.build_opportunity_merged_content_context(opp)

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke test: flat opportunity merged-content context.",
    )
    parser.add_argument("--opportunity-id", required=True)
    args = parser.parse_args()

    raise SystemExit(main(opportunity_id=str(args.opportunity_id)))
