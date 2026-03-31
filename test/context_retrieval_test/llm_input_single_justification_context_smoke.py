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
from db.db_conn import SessionLocal
from services.context_retrieval.context_generator import ContextGenerator
from test.context_retrieval_test._llm_input_common import (
    build_top_rows_for_faculty,
    norm,
    safe_float,
)


def main(
    *,
    email: str,
    k: int,
    opportunity_id: str | None,
    preview_chars: int,
) -> int:
    email_norm = norm(email)
    if not email_norm:
        raise ValueError("--email is required")

    cgen = ContextGenerator()
    with SessionLocal() as sess:
        fdao = FacultyDAO(sess)
        fac = fdao.get_with_relations_by_email(email_norm)
        if not fac:
            raise ValueError(f"Faculty not found for email: {email_norm}")

        top_rows = build_top_rows_for_faculty(
            sess=sess,
            faculty_id=int(fac.faculty_id),
            k=max(1, int(k)),
            opportunity_id=(norm(opportunity_id) if opportunity_id else None),
        )

        grant_explanation_inputs: List[Dict[str, Any]] = []
        final_justification_context_text: List[Dict[str, Any]] = []
        for oid, d_score, l_score in list(top_rows or []):
            grant_context = cgen.build_grant_context_only(
                sess=sess,
                opportunity_id=str(oid),
                preview_chars=max(100, int(preview_chars)),
            )
            grant_explanation_inputs.append(
                {
                    "opportunity_id": str(oid),
                    "domain_score": safe_float(d_score),
                    "llm_score": safe_float(l_score),
                    "grant_context": grant_context,
                }
            )

            one_match_text = cgen.build_faculty_recommendation_source_linked_text(
                sess=sess,
                fac=fac,
                top_rows=[(str(oid), safe_float(d_score), safe_float(l_score))],
            )
            final_justification_context_text.append(
                {
                    "opportunity_id": str(oid),
                    "context_text": str(one_match_text or ""),
                }
            )

        final_payload = cgen.build_faculty_recommendation_source_linked_payload(
            sess=sess,
            fac=fac,
            top_rows=list(top_rows or []),
        )

        match_payload_rows = cgen.build_top_match_payload(
            sess=sess,
            top_rows=list(top_rows or []),
        )

    out = {
        "email": email_norm,
        "faculty_id": int(fac.faculty_id),
        "top_rows": [
            {
                "opportunity_id": str(oid),
                "domain_score": safe_float(d),
                "llm_score": safe_float(l),
            }
            for oid, d, l in list(top_rows or [])
        ],
        "llm_inputs": {
            "grant_explanation": grant_explanation_inputs,
            "final_justification_source_payload": final_payload,
            "final_justification_context_text": final_justification_context_text,
            "top_match_payload_rows": match_payload_rows,
        },
    }

    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke test: LLM input contexts for single justification pipeline.",
    )
    parser.add_argument("--email", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--opportunity-id", default=None)
    parser.add_argument("--preview-chars", type=int, default=50_000)
    args = parser.parse_args()

    raise SystemExit(
        main(
            email=str(args.email),
            k=int(args.k),
            opportunity_id=args.opportunity_id,
            preview_chars=int(args.preview_chars),
        )
    )
