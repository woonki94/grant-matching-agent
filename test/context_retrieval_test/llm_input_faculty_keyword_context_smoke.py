from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dao.faculty_dao import FacultyDAO
from db.db_conn import SessionLocal
from services.context_retrieval.context_generator import ContextGenerator


def main(*, faculty_id: int) -> int:
    cgen = ContextGenerator()
    with SessionLocal() as sess:
        fdao = FacultyDAO(sess)
        fac = fdao.get_with_relations_by_id(int(faculty_id))
        if not fac:
            raise ValueError(f"Faculty not found: {faculty_id}")

        payload = cgen.build_faculty_basic_context(fac)

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke test: LLM input context for faculty keyword generation.",
    )
    parser.add_argument("--faculty-id", type=int, required=True)
    args = parser.parse_args()

    raise SystemExit(main(faculty_id=int(args.faculty_id)))
