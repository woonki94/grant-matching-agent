from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

import boto3

from db.db_conn import SessionLocal
from dao.content_extraction_dao import ContentExtractionDAO
from utils.content_extractor import fetch_and_extract_one


def short_hash(value: str, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _safe_subdir(subdir: str) -> str:
    # Make sure we don't accidentally create double slashes in S3 keys
    return str(subdir).strip("/").replace("\\", "/")


def run_extraction_pipeline(
    *,
    model: Type[Any],
    base_dir: Optional[Path],
    subdir: str,
    url_getter: Callable[[Any], str],
    batch_size: int = 200,
    # ✅ New: backend config
    backend: str = "local",  # "local" or "s3"
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "",
    aws_region: Optional[str] = None,
    aws_profile: Optional[str] = None,
) -> Dict[str, int]:
    """
    Generic pipeline for any DB model with:
      id, extract_status, extract_error, extracted_at, content_path, detected_type, content_char_count

    Local backend:
      Writes: <base_dir>/<subdir>/<id>__<hash>.txt
      Stores content_path as relative path (same convention as you choose; here we store the full local path string)

    S3 backend:
      Writes: s3://<bucket>/<prefix>/<subdir>/<id>__<hash>.txt
      Stores content_path as the S3 key (recommended).
    """

    backend = (backend or "local").lower().strip()
    if backend not in {"local", "s3"}:
        raise ValueError(f"Invalid backend={backend}. Must be 'local' or 's3'.")

    subdir = _safe_subdir(subdir)

    # Resolve S3 parameters (prefer explicit args; fallback to env vars for safety)
    if backend == "s3":
        bucket = s3_bucket or os.getenv("EXTRACTED_CONTENT_BUCKET")
        if not bucket:
            raise RuntimeError(
                "S3 backend requires EXTRACTED_CONTENT_BUCKET (or s3_bucket arg)."
            )

        prefix = (s3_prefix or os.getenv("EXTRACTED_CONTENT_PREFIX", "")).strip("/")
        region = aws_region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")

        if aws_profile:
            session = boto3.Session(profile_name=aws_profile, region_name=region)
        else:
            session = boto3.Session(region_name=region)

        s3 = session.client("s3")

    else:
        # Local backend requires a base_dir
        if base_dir is None:
            raise RuntimeError(
                "Local backend requires base_dir (EXTRACTED_CONTENT_PATH)."
            )
        base_dir = Path(base_dir)
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)

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

                if backend == "s3":
                    # Store under: s3://<bucket>/<prefix>/<subdir>/<id>__<hash>.txt
                    if prefix:
                        s3_key = f"{prefix}/{subdir}/{filename}"
                    else:
                        s3_key = f"{subdir}/{filename}"

                    s3.put_object(
                        Bucket=bucket,  # type: ignore[name-defined]
                        Key=s3_key,
                        Body=text.encode("utf-8", errors="ignore"),
                        ContentType="text/plain; charset=utf-8",
                    )

                    content_path = s3_key  # store key in DB (recommended)

                else:
                    # Local: <base_dir>/<subdir>/<id>__<hash>.txt
                    out_path = (base_dir / subdir / filename)  # type: ignore[operator]
                    out_path.write_text(text, encoding="utf-8", errors="ignore")
                    content_path = str(out_path)

                updates.append(
                    {
                        "id": item.id,
                        "content_path": content_path,
                        "detected_type": detected_type,
                        "content_char_count": len(text),
                        "extracted_at": extracted_at,
                        "extract_status": "done",
                        "extract_error": None,
                    }
                )

                processed += 1
                done += 1

            dao.bulk_update(model, updates)
            sess.commit()

    return {"processed": processed, "done": done, "failed": failed}
