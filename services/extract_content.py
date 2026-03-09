from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Type

import boto3

from config import settings
from dao.content_extraction_dao import ContentExtractionDAO
from db.db_conn import SessionLocal
from utils.content_extractor import fetch_and_extract_one
from utils.thread_pool import build_thread_local_getter, parallel_map


def short_hash(value: str, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _safe_subdir(subdir: str) -> str:
    return str(subdir).strip("/").replace("\\", "/")


def run_extraction_pipeline(
    *,
    model: Type[Any],
    subdir: str,
    url_getter: Callable[[Any], str],
    batch_size: int = 200,
    max_workers: int = 1,
    # Overrides (recommended to pass explicitly from caller)
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    aws_region: Optional[str] = None,
    aws_profile: Optional[str] = None,
) -> Dict[str, int]:
    """
    Extracts text for pending DB rows and stores it in S3 ONLY:

      s3://<bucket>/<prefix>/<subdir>/<id>__<hash>.txt

    Stores the S3 *key* (not s3:// URI) in DB as content_path.
    """

    subdir = _safe_subdir(subdir)

    # -------------------------
    # S3 setup (required)
    # -------------------------
    bucket = (s3_bucket or settings.extracted_content_bucket or "").strip()
    if not bucket:
        raise RuntimeError("S3 storage requires EXTRACTED_CONTENT_BUCKET (extracted_content_bucket).")

    prefix = (s3_prefix or "").strip().strip("/")

    region = aws_region or settings.aws_region
    profile = aws_profile or settings.aws_profile

    safe_workers = max(1, int(max_workers or 1))

    def _build_s3_client():
        session = (
            boto3.Session(profile_name=profile, region_name=region)
            if profile
            else boto3.Session(region_name=region)
        )
        return session.client("s3")

    get_s3_client = build_thread_local_getter(_build_s3_client)

    # -------------------------
    # Process pending items
    # -------------------------
    processed = 0
    done = 0
    failed = 0

    with SessionLocal() as sess:
        dao = ContentExtractionDAO(sess)

        while True:
            items = dao.fetch_pending(model, limit=batch_size)
            if not items:
                break

            jobs = []
            for item in items:
                jobs.append(
                    {
                        "id": int(item.id),
                        "url": str(url_getter(item) or "").strip(),
                    }
                )

            def _run_one(job: Dict[str, Any]) -> Dict[str, Any]:
                item_id = int(job.get("id") or 0)
                url = str(job.get("url") or "").strip()
                print(url)
                extracted_at = datetime.now(timezone.utc)
                if item_id <= 0 or not url:
                    return {
                        "id": item_id,
                        "content_path": None,
                        "detected_type": None,
                        "content_char_count": None,
                        "extracted_at": extracted_at,
                        "extract_status": "failed",
                        "extract_error": "missing_id_or_url",
                    }

                result = fetch_and_extract_one(url)
                print(result)
                text = result.get("text")
                if not text:
                    err = str(result.get("error") or "no_text")
                    err = (err[:5000] + "…") if len(err) > 5000 else err
                    return {
                        "id": item_id,
                        "content_path": None,
                        "detected_type": None,
                        "content_char_count": None,
                        "extracted_at": extracted_at,
                        "extract_status": "failed",
                        "extract_error": err,
                    }

                detected_type = result.get("detected_type") or "unknown"
                hash_part = short_hash(url)
                filename = f"{item_id}__{hash_part}.txt"
                key = f"{prefix}/{subdir}/{filename}" if prefix else f"{subdir}/{filename}"

                s3 = get_s3_client()
                s3.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=str(text).encode("utf-8", errors="ignore"),
                    ContentType="text/plain; charset=utf-8",
                )
                return {
                    "id": item_id,
                    "content_path": key,
                    "detected_type": detected_type,
                    "content_char_count": len(str(text)),
                    "extracted_at": extracted_at,
                    "extract_status": "success",
                    "extract_error": None,
                }

            def _on_error(index: int, job: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
                item_id = int((job or {}).get("id") or 0)
                extracted_at = datetime.now(timezone.utc)
                err = f"{type(exc).__name__}: {exc}"
                err = (err[:5000] + "…") if len(err) > 5000 else err
                return {
                    "id": item_id,
                    "content_path": None,
                    "detected_type": None,
                    "content_char_count": None,
                    "extracted_at": extracted_at,
                    "extract_status": "failed",
                    "extract_error": err,
                }

            updates = parallel_map(
                jobs,
                max_workers=min(safe_workers, len(jobs)),
                run_item=_run_one,
                on_error=_on_error,
            )

            for row in updates:
                processed += 1
                if str(row.get("extract_status") or "").strip().lower() == "success":
                    done += 1
                else:
                    failed += 1

            dao.bulk_update(model, updates)
            sess.commit()

    return {"processed": processed, "done": done, "failed": failed}
