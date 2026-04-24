from __future__ import annotations

import copy
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional


logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FacultyProfileUpdateJobManager:
    """In-memory async job registry for faculty profile post-processing."""

    DEFAULT_MAX_WORKERS = 4
    DEFAULT_MAX_JOB_HISTORY = 200

    def __init__(
        self,
        *,
        max_workers: Optional[int] = None,
        max_job_history: Optional[int] = None,
    ):
        env_workers = self._safe_int_or_none(os.getenv("FACULTY_PROFILE_JOB_WORKERS"))
        env_history = self._safe_int_or_none(os.getenv("FACULTY_PROFILE_JOB_HISTORY"))
        workers = int(env_workers or max_workers or self.DEFAULT_MAX_WORKERS)
        history_limit = int(env_history or max_job_history or self.DEFAULT_MAX_JOB_HISTORY)
        workers = max(1, min(workers, 32))
        history_limit = max(20, min(history_limit, 5000))

        self._executor = ThreadPoolExecutor(
            max_workers=int(workers),
            thread_name_prefix="faculty-profile-job",
        )
        self._max_job_history = int(history_limit)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._job_order: list[str] = []
        self._lock = threading.Lock()

    @staticmethod
    def _safe_int_or_none(value: Any) -> Optional[int]:
        try:
            return int(value)
        except Exception:
            return None

    def submit(
        self,
        *,
        job_type: str,
        payload: Optional[Dict[str, Any]],
        run_fn: Callable[[], Dict[str, Any]],
    ) -> str:
        job_id = f"fpu-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        record = {
            "job_id": str(job_id),
            "job_type": str(job_type or "faculty_profile_postprocess"),
            "status": "queued",
            "payload": dict(payload or {}),
            "result": None,
            "error": None,
            "created_at": _utc_now_iso(),
            "started_at": None,
            "finished_at": None,
        }
        with self._lock:
            self._jobs[job_id] = record
            self._job_order.append(job_id)
            self._prune_locked()

        self._executor.submit(self._run_job, job_id, run_fn)
        return job_id

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        key = str(job_id or "").strip()
        if not key:
            return None
        with self._lock:
            row = self._jobs.get(key)
            return copy.deepcopy(row) if row else None

    def _run_job(self, job_id: str, run_fn: Callable[[], Dict[str, Any]]) -> None:
        with self._lock:
            row = self._jobs.get(job_id)
            if not row:
                return
            row["status"] = "running"
            row["started_at"] = _utc_now_iso()

        try:
            result = run_fn() or {}
        except Exception as exc:
            logger.exception("Faculty profile async job failed job_id=%s", job_id)
            with self._lock:
                row = self._jobs.get(job_id)
                if row:
                    row["status"] = "failed"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    row["finished_at"] = _utc_now_iso()
        else:
            with self._lock:
                row = self._jobs.get(job_id)
                if row:
                    row["status"] = "succeeded"
                    row["result"] = dict(result or {})
                    row["finished_at"] = _utc_now_iso()
        finally:
            with self._lock:
                self._prune_locked()

    def _prune_locked(self) -> None:
        overflow = len(self._job_order) - int(self._max_job_history)
        if overflow <= 0:
            return
        keep_ids = set(self._job_order[overflow:])
        for old_id in list(self._job_order[:overflow]):
            row = self._jobs.get(old_id)
            status = str((row or {}).get("status") or "")
            if status in {"queued", "running"}:
                keep_ids.add(old_id)
        self._job_order = [x for x in self._job_order if x in keep_ids]
        self._jobs = {k: v for k, v in self._jobs.items() if k in keep_ids}
