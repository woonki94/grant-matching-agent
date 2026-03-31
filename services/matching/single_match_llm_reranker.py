from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_llm_client, settings
from dao.match_dao import MatchDAO
from db.db_conn import SessionLocal
from dto.llm_response_dto import FacultyGrantRerankOut
from services.context_retrieval.context_generator import ContextGenerator
from services.prompts.justification_prompts import FACULTY_TOP_GRANT_RERANK_PROMPT
from utils.thread_pool import parallel_map, resolve_pool_size

logger = logging.getLogger(__name__)


class OneToOneLLMReranker:
    """LLM reranker for existing one-to-one faculty-grant match rows."""

    DEFAULT_MAX_CONTEXT_CHARS = 50_000

    def __init__(
        self,
        *,
        session_factory=SessionLocal,
        context_generator: Optional[ContextGenerator] = None,
    ):
        self.session_factory = session_factory
        self.context_generator = context_generator or ContextGenerator()

    @staticmethod
    def _norm(text: Any) -> str:
        return " ".join(str(text or "").split()).strip()

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _to_json(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _build_grant_rerank_chain():
        model_id = (settings.opus or settings.sonnet or settings.haiku or "").strip()
        llm = get_llm_client(model_id=model_id).build()
        return FACULTY_TOP_GRANT_RERANK_PROMPT | llm.with_structured_output(FacultyGrantRerankOut)

    @staticmethod
    def _parse_rerank_output(out: Any) -> FacultyGrantRerankOut:
        if isinstance(out, FacultyGrantRerankOut):
            return out
        if isinstance(out, dict):
            return FacultyGrantRerankOut.model_validate(out)
        if hasattr(out, "model_dump"):
            return FacultyGrantRerankOut.model_validate(out.model_dump())
        return FacultyGrantRerankOut(ranked_opportunity_ids=[], reranked_grants=[])

    def _chunk_grants_for_context_limit(
        self,
        *,
        faculty_payload: Dict[str, Any],
        grants: List[Dict[str, Any]],
        max_context_chars: int,
    ) -> List[List[Dict[str, Any]]]:
        """Chunk grants so faculty_json + grants_json stays under max_context_chars per call."""
        limit = max(5_000, int(max_context_chars or self.DEFAULT_MAX_CONTEXT_CHARS))
        faculty_chars = len(self._to_json(dict(faculty_payload or {})))
        grants_budget = max(1_000, limit - faculty_chars)

        chunks: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_json_chars = 2  # "[]"

        for grant in list(grants or []):
            g = dict(grant or {})
            g_chars = len(self._to_json(g))
            separator_chars = 1 if current else 0
            projected = current_json_chars + separator_chars + g_chars

            if current and projected > grants_budget:
                chunks.append(current)
                current = [g]
                current_json_chars = 2 + g_chars
            else:
                current.append(g)
                current_json_chars = projected

        if current:
            chunks.append(current)

        if not chunks and grants:
            chunks = [list(grants)]

        return chunks

    def rerank_for_faculty(
        self,
        *,
        faculty_id: int,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> Dict[str, Any]:
        """Rerank all stored grants for one faculty and persist llm_score updates."""
        fid = int(faculty_id)
        logger.info(
            "One-to-one rerank faculty_start faculty_id=%s max_context_chars=%s",
            fid,
            int(max_context_chars),
        )

        with self.session_factory() as sess:
            mdao = MatchDAO(sess)
            match_count = mdao.count_matches_for_faculty(faculty_id=fid)
            if match_count <= 0:
                return {
                    "faculty_id": fid,
                    "match_count": 0,
                    "updated_rows": 0,
                    "status": "no_matches",
                }
            # Fetch all matched grants for this faculty via context builder.
            inventory = self.context_generator.build_rerank_keyword_inventory_for_faculty(
                sess=sess,
                faculty_id=fid,
            )

        faculty_payload = dict(inventory.get("faculty") or {})
        grants = [dict(x or {}) for x in list(inventory.get("grants") or [])]
        if not grants:
            return {
                "faculty_id": fid,
                "match_count": 0,
                "updated_rows": 0,
                "status": "no_grant_inventory",
            }

        grant_chunks = self._chunk_grants_for_context_limit(
            faculty_payload=faculty_payload,
            grants=grants,
            max_context_chars=int(max_context_chars),
        )
        logger.info(
            "One-to-one rerank chunking faculty_id=%s grants=%s chunks=%s max_context_chars=%s",
            fid,
            len(grants),
            len(grant_chunks),
            int(max_context_chars),
        )

        chain = self._build_grant_rerank_chain()
        score_by_id: Dict[str, float] = {}

        for idx, chunk in enumerate(list(grant_chunks or []), start=1):
            try:
                payload = {
                    "faculty_json": self._to_json(faculty_payload),
                    "grants_json": self._to_json(chunk),
                }
                logger.info(
                    "One-to-one rerank chunk_start faculty_id=%s chunk=%s/%s chunk_grants=%s faculty_json_chars=%s grants_json_chars=%s",
                    fid,
                    idx,
                    len(grant_chunks),
                    len(chunk),
                    len(payload["faculty_json"]),
                    len(payload["grants_json"]),
                )
                out = chain.invoke(
                    payload
                )
                parsed = self._parse_rerank_output(out)
                chunk_scores = {
                    self._norm(getattr(item, "opportunity_id", None)): self._safe_float(
                        getattr(item, "llm_score", 0.0)
                    )
                    for item in list(parsed.reranked_grants or [])
                    if self._norm(getattr(item, "opportunity_id", None))
                }
                score_by_id.update(chunk_scores)
                logger.info(
                    "One-to-one rerank chunk_done faculty_id=%s chunk=%s/%s chunk_grants=%s scored=%s",
                    fid,
                    idx,
                    len(grant_chunks),
                    len(chunk),
                    len(chunk_scores),
                )
            except Exception as exc:
                logger.exception(
                    "One-to-one rerank chunk failed faculty_id=%s chunk=%s/%s error=%s",
                    fid,
                    idx,
                    len(grant_chunks),
                    f"{type(exc).__name__}: {exc}",
                )

        base_by_id = {
            self._norm(g.get("opportunity_id")): dict(g)
            for g in grants
            if self._norm(g.get("opportunity_id"))
        }
        grant_scores: Dict[str, float] = {}
        for oid, base in list(base_by_id.items()):
            fallback = self._safe_float(base.get("llm_score"), 0.0)
            grant_scores[oid] = self._safe_float(score_by_id.get(oid), fallback)

        if not grant_scores:
            return {
                "faculty_id": fid,
                "match_count": len(grants),
                "updated_rows": 0,
                "status": "no_scores",
            }

        # Dry-run mode for inspection: do not write back to DB.
        reranked_rows = [
            {
                "opportunity_id": oid,
                "llm_score": self._safe_float(score),
            }
            for oid, score in sorted(grant_scores.items(), key=lambda x: float(x[1]), reverse=True)
        ]
        return {
            "faculty_id": fid,
            "match_count": len(grants),
            "chunk_count": len(grant_chunks),
            "updated_rows": 0,
            "status": "dry_run",
            "reranked_grants": reranked_rows,
        }

    def run(
        self,
        *,
        limit_faculty: int = 0,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        workers: int = 4,
    ) -> Dict[str, Any]:
        """Rerank llm_score for all faculty that currently have one-to-one match rows."""
        with self.session_factory() as sess:
            mdao = MatchDAO(sess)
            faculty_ids = mdao.list_faculty_ids_with_matches(
                limit=(int(limit_faculty) if int(limit_faculty or 0) > 0 else None)
            )
        pool_size = resolve_pool_size(max_workers=int(workers or 0), task_count=len(faculty_ids))
        logger.info(
            "One-to-one rerank run_start faculty_targets=%s limit_faculty=%s max_context_chars=%s workers=%s",
            len(faculty_ids),
            int(limit_faculty or 0),
            int(max_context_chars),
            int(pool_size),
        )

        def _run_one(fid: int) -> Dict[str, Any]:
            result = self.rerank_for_faculty(
                faculty_id=int(fid),
                max_context_chars=int(max_context_chars),
            )
            logger.info(
                "One-to-one rerank processed faculty_id=%s status=%s updated_rows=%s",
                int(fid),
                str(result.get("status") or ""),
                int(result.get("updated_rows") or 0),
            )
            return result

        def _on_error(_idx: int, fid: int, exc: Exception) -> Dict[str, Any]:
            logger.exception(
                "One-to-one rerank failed for faculty_id=%s: %s",
                int(fid),
                f"{type(exc).__name__}: {exc}",
            )
            return {
                "faculty_id": int(fid),
                "status": "failed",
                "updated_rows": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }

        outputs: List[Dict[str, Any]] = parallel_map(
            list(faculty_ids or []),
            max_workers=pool_size,
            run_item=_run_one,
            on_error=_on_error,
        )

        processed = len(outputs)
        failed = sum(1 for r in outputs if str((r or {}).get("status") or "").lower() == "failed")
        total_updated_rows = sum(int((r or {}).get("updated_rows") or 0) for r in outputs)

        summary = {
            "faculty_target_count": len(faculty_ids),
            "faculty_processed": int(processed),
            "faculty_failed": int(failed),
            "updated_match_rows": int(total_updated_rows),
            "results": outputs,
        }
        logger.info("One-to-one rerank done: %s", json.dumps(summary, ensure_ascii=False))
        return summary
