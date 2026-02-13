from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Type

import boto3

from config import settings
from dao.content_extraction_dao import ContentExtractionDAO
from db.db_conn import SessionLocal
from utils.content_extractor import fetch_and_extract_one


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

    session = (
        boto3.Session(profile_name=profile, region_name=region)
        if profile
        else boto3.Session(region_name=region)
    )
    s3_client = session.client("s3")

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

            updates = []
            for item in items:
                extracted_at = datetime.now(timezone.utc)
                url = url_getter(item)

                result = fetch_and_extract_one(url)
                text = result.get("text")

                if not text:
                    err = result.get("error") or "no_text"
                    err = (err[:5000] + "…") if len(err) > 5000 else err
                    updates.append(
                        {
                            "id": item.id,
                            "content_path": None,
                            "detected_type": None,
                            "content_char_count": None,
                            "extracted_at": extracted_at,
                            "extract_status": "failed",
                            "extract_error": err,
                        }
                    )
                    processed += 1
                    failed += 1
                    continue

                detected_type = result.get("detected_type") or "unknown"
                hash_part = short_hash(url)
                filename = f"{item.id}__{hash_part}.txt"

                # Build key: <prefix>/<subdir>/<filename> (or <subdir>/<filename> if prefix empty)
                key = f"{prefix}/{subdir}/{filename}" if prefix else f"{subdir}/{filename}"

                s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=text.encode("utf-8", errors="ignore"),
                    ContentType="text/plain; charset=utf-8",
                )

                updates.append(
                    {
                        "id": item.id,
                        "content_path": key,  # store S3 key in DB
                        "detected_type": detected_type,
                        "content_char_count": len(text),
                        "extracted_at": extracted_at,
                        "extract_status": "success",
                        "extract_error": None,
                    }
                )

                processed += 1
                done += 1

            dao.bulk_update(model, updates)
            sess.commit()

    return {"processed": processed, "done": done, "failed": failed}
