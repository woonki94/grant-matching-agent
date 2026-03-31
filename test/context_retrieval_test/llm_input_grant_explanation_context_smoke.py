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


def main(*, opportunity_id: str, preview_chars: int) -> int:
    oid = norm(opportunity_id)
    if not oid:
        raise ValueError("--opportunity-id is required")

    cgen = ContextGenerator()
    with SessionLocal() as sess:
        payload = cgen.build_grant_context_only(
            sess=sess,
            opportunity_id=oid,
            preview_chars=max(100, int(preview_chars)),
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke test: LLM input context for grant explanation.",
    )
    parser.add_argument("--opportunity-id", required=True)
    parser.add_argument("--preview-chars", type=int, default=50_000)
    args = parser.parse_args()

    raise SystemExit(
        main(
            opportunity_id=str(args.opportunity_id),
            preview_chars=int(args.preview_chars),
        )
    )
