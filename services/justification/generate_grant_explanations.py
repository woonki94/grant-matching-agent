from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from logging_setup import setup_logging
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from services.justification.single_justification_generator import SingleJustificationGenerator
from utils.thread_pool import parallel_map, resolve_pool_size

logger = logging.getLogger(__name__)

WORKERS = 8


def _generate_and_save(opportunity_id: str) -> Dict[str, Any]:
    """Generate grant_explanation and grant_brief in parallel, then save both."""
    generator = SingleJustificationGenerator()
    oid = str(opportunity_id)

    with SessionLocal() as sess:
        kw_row = OpportunityDAO(sess).get_opportunity_keyword(oid)
        needs_explanation = not (getattr(kw_row, "grant_explanation", None) or "").strip()
        needs_brief = not (getattr(kw_row, "grant_brief", None) or "").strip()

    if not needs_explanation and not needs_brief:
        return {"opportunity_id": oid, "status": "already_cached"}

    explanation = ""
    brief = ""
    errors = []

    def _run_explanation():
        try:
            result = generator._generate_grant_explanation(opportunity_id=oid)
            return result.get("grant_explanation") or ""
        except Exception as exc:
            errors.append(f"explanation: {exc}")
            return ""

    def _run_brief():
        try:
            return generator._generate_grant_brief(opportunity_id=oid)
        except Exception as exc:
            errors.append(f"brief: {exc}")
            return ""

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_expl = pool.submit(_run_explanation) if needs_explanation else None
        fut_brief = pool.submit(_run_brief) if needs_brief else None
        if fut_expl:
            explanation = fut_expl.result()
        if fut_brief:
            brief = fut_brief.result()

    if errors:
        logger.error("Errors for opportunity_id=%s: %s", oid, errors)
        return {"opportunity_id": oid, "status": "failed", "errors": errors}

    if not explanation and needs_explanation:
        logger.warning("Empty explanation for opportunity_id=%s", oid)
    if not brief and needs_brief:
        logger.warning("Empty brief for opportunity_id=%s", oid)

    with SessionLocal() as sess:
        dao = OpportunityDAO(sess)
        if explanation:
            dao.save_grant_explanation(opportunity_id=oid, explanation=explanation)
        if brief:
            dao.save_grant_brief(opportunity_id=oid, brief=brief)
        sess.commit()

    logger.info(
        "Saved for opportunity_id=%s explanation_chars=%s brief_chars=%s",
        oid, len(explanation), len(brief),
    )
    return {"opportunity_id": oid, "status": "done"}


def main(force: bool = False) -> None:
    with SessionLocal() as sess:
        dao = OpportunityDAO(sess)
        if force:
            from db.models.opportunity import OpportunityKeyword
            rows = sess.query(OpportunityKeyword).all()
            for row in rows:
                row.grant_explanation = None
                row.grant_brief = None
            sess.commit()
            pending = rows
        else:
            pending = dao.list_opportunities_needing_generation()

    total = len(pending)
    if total == 0:
        print("All grants already have explanation and brief cached. Use --force to regenerate.")
        return

    opportunity_ids = [row.opportunity_id for row in pending]
    pool_size = resolve_pool_size(max_workers=WORKERS, task_count=total)
    print(f"Generating for {total} grants (workers={pool_size})...")

    def _on_error(_idx: int, oid: str, exc: Exception) -> Dict[str, Any]:
        logger.exception("Unhandled error for opportunity_id=%s: %s", oid, exc)
        return {"opportunity_id": oid, "status": "failed", "error": str(exc)}

    results = parallel_map(
        opportunity_ids,
        max_workers=pool_size,
        run_item=_generate_and_save,
        on_error=_on_error,
    )

    done = sum(1 for r in results if (r or {}).get("status") == "done")
    cached = sum(1 for r in results if (r or {}).get("status") == "already_cached")
    failed = sum(1 for r in results if (r or {}).get("status") == "failed")

    print(f"\nDone: {done}  Already cached: {cached}  Failed: {failed}  Total: {total}")


if __name__ == "__main__":
    setup_logging("grant_explanations")
    parser = argparse.ArgumentParser(
        description="Pre-generate and cache grant_explanation and grant_brief for all grants."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even for grants that already have cached values.",
    )
    args = parser.parse_args()
    main(force=args.force)
