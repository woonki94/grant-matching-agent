from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type
import os
import boto3
from botocore.exceptions import ClientError

from config import settings
from db.db_conn import SessionLocal
from dao.content_extraction_dao import ContentExtractionDAO
from db.models.opportunity import OpportunityAttachment, OpportunityAdditionalInfo
from utils.content_extractor import fetch_and_extract_one
from utils.extracted_content_store import put_text

def short_hash(value: str, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def run_extraction_pipeline(
    *,
    model: Type[Any],
    base_dir: Path,
    subdir: str,
    url_getter: Callable[[Any], str],
    batch_size: int = 200,
) -> Dict[str, int]:
    """
    Generic pipeline for any DB model with:
      id, extract_status, extract_error, extracted_at, content_path, detected_type, content_char_count
    """

    processed = 0
    done = 0
    failed = 0

    bucket = os.getenv("EXTRACTED_CONTENT_BUCKET", "grant-matcher")
    prefix = os.getenv("EXTRACTED_CONTENT_PREFIX", "extracted-context-opportunities").strip("/")

    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-west-2"))


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
                    updates.append({
                        "id": item.id,
                        "content_path": None,
                        "detected_type": None,
                        "content_char_count": None,
                        "extracted_at": extracted_at,
                        "extract_status": "failed",
                        "extract_error": (err[:5000] + "…") if len(err) > 5000 else err,
                    })
                    processed += 1
                    failed += 1
                    continue

                detected_type = result.get("detected_type") or "unknown"

                hash_part = short_hash(url)

                # Store under: s3://grant-matcher/extracted-context-opportunities/<subdir>/<id>__<hash>.txt
                s3_key = f"{prefix}/{subdir}/{item.id}__{hash_part}.txt"

                s3.put_object(
                    Bucket=bucket,
                    Key=s3_key,
                    Body=text.encode("utf-8", errors="ignore"),
                    ContentType="text/plain; charset=utf-8",
                )

                updates.append({
                    "id": item.id,
                    # store either the key OR full s3:// url; key is usually better
                    "content_path": s3_key,
                    "detected_type": detected_type,
                    "content_char_count": len(text),
                    "extracted_at": extracted_at,
                    "extract_status": "done",
                    "extract_error": None,
                })

                processed += 1
                done += 1

            dao.bulk_update(model, updates)
            sess.commit()

    return {"processed": processed, "done": done, "failed": failed}
