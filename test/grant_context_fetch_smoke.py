from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from logging_setup import setup_logging
from utils.content_extractor import load_extracted_content


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _short(text: Any, max_chars: int) -> str:
    s = _norm(text)
    if len(s) <= int(max_chars):
        return s
    return s[: int(max_chars)].rstrip()


def _trim_rows(rows: list[Dict[str, Any]], *, preview_chars: int) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for row in list(rows or []):
        item = dict(row or {})
        if "content" in item:
            item["content"] = _short(item.get("content"), max_chars=preview_chars)
        out.append(item)
    return out


def main(*, opportunity_id: str, preview_chars: int) -> int:
    oid = _norm(opportunity_id)
    if not oid:
        raise ValueError("--opportunity-id is required")

    with SessionLocal() as sess:
        odao = OpportunityDAO(sess)
        opps = odao.read_opportunities_by_ids_with_relations([oid])
        opp = opps[0] if opps else None
        if not opp:
            raise ValueError(f"Opportunity not found: {oid}")

        add_rows = load_extracted_content(
            list(getattr(opp, "additional_info", None) or []),
            url_attr="additional_info_url",
            group_chunks=False,
            include_row_meta=True,
        )
        att_rows = load_extracted_content(
            list(getattr(opp, "attachments", None) or []),
            url_attr="file_download_path",
            title_attr="file_name",
            group_chunks=False,
            include_row_meta=True,
        )

        payload = {
            "opportunity_id": getattr(opp, "opportunity_id", None),
            "title": getattr(opp, "opportunity_title", None),
            "agency": getattr(opp, "agency_name", None),
            "status": getattr(opp, "opportunity_status", None),
            "opportunity_link": None,
            "summary": _norm(getattr(opp, "summary_description", None)),
            "additional_info_count": len(add_rows),
            "attachment_count": len(att_rows),
            "additional_info_extracted": _trim_rows(add_rows, preview_chars=preview_chars),
            "attachments_extracted": _trim_rows(att_rows, preview_chars=preview_chars),
        }

        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0


if __name__ == "__main__":
    setup_logging("justification")
    parser = argparse.ArgumentParser(description="Smoke test: fetch one grant context only (no faculty)")
    parser.add_argument("--opportunity-id", required=True)
    parser.add_argument("--preview-chars", type=int, default=20000)
    args = parser.parse_args()

    raise SystemExit(
        main(
            opportunity_id=args.opportunity_id,
            preview_chars=max(100, int(args.preview_chars)),
        )
    )
