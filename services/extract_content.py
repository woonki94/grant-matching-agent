from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
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
    base_dir: Optional[Path],
    subdir: str,
    url_getter: Callable[[Any], str],
    batch_size: int = 200,
    backend: str = "local",  # "local" or "s3"
    # Optional overrides (if not passed, uses config settings)
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    aws_region: Optional[str] = None,
    aws_profile: Optional[str] = None,
) -> Dict[str, int]:
    """
    Extracts text for pending DB rows and stores it either:
      - Local: <base_dir>/<subdir>/<id>__<hash>.txt
      - S3:    s3://<bucket>/<prefix>/<subdir>/<id>__<hash>.txt  (stores the S3 key in DB)
    """

    backend = (backend or "local").lower().strip()
    if backend not in {"local", "s3"}:
        raise ValueError(f"Invalid backend={backend}. Must be 'local' or 's3'.")

    subdir = _safe_subdir(subdir)

    # -------------------------
    # Backend setup
    # -------------------------
    s3_client = None
    bucket = None
    prefix = ""

    if backend == "s3":
        bucket = s3_bucket or settings.extracted_content_bucket
        if not bucket:
            raise RuntimeError(
                "S3 backend requires extracted_content_bucket (EXTRACTED_CONTENT_BUCKET)."
            )

        prefix = (s3_prefix if s3_prefix is not None else settings.extracted_content_prefix or "").strip("/")

        region = aws_region or settings.aws_region
        profile = aws_profile or settings.aws_profile

        session = boto3.Session(profile_name=profile, region_name=region) if profile else boto3.Session(region_name=region)
        s3_client = session.client("s3")

    else:
        if base_dir is None:
            raise RuntimeError("Local backend requires base_dir (EXTRACTED_CONTENT_PATH).")
        base_dir = Path(base_dir)
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)

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

                if backend == "s3":
                    # Build key: <prefix>/<subdir>/<filename>
                    key = f"{prefix}/{subdir}/{filename}" if prefix else f"{subdir}/{filename}"

                    s3_client.put_object(  # type: ignore[union-attr]
                        Bucket=bucket,  # type: ignore[arg-type]
                        Key=key,
                        Body=text.encode("utf-8", errors="ignore"),
                        ContentType="text/plain; charset=utf-8",
                    )
                    content_path = key  # store S3 key in DB

                else:
                    out_path = base_dir / subdir / filename  # type: ignore[operator]
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
