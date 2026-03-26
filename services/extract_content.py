from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Type

import boto3
import numpy as np
from sqlalchemy import and_, delete

from config import settings
from dao.content_extraction_dao import ContentExtractionDAO
from db.db_conn import SessionLocal
from utils.content_extractor import (
    chunk_text_for_embedding,
    fetch_and_extract_one,
    safe_filename,
)
from utils.embedder import embed_texts


def short_hash(value: str, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _safe_subdir(subdir: str) -> str:
    return str(subdir).strip("/").replace("\\", "/")


def _build_chunk_identity_filters(model: Type[Any], item: Any):
    """Build filters identifying all chunk rows of one logical file/link."""
    if hasattr(model, "faculty_id") and hasattr(model, "additional_info_url"):
        return [
            model.faculty_id == int(getattr(item, "faculty_id")),
            model.additional_info_url == str(getattr(item, "additional_info_url")),
        ]
    if hasattr(model, "opportunity_id") and hasattr(model, "additional_info_url"):
        return [
            model.opportunity_id == str(getattr(item, "opportunity_id")),
            model.additional_info_url == str(getattr(item, "additional_info_url")),
        ]
    if hasattr(model, "opportunity_id") and hasattr(model, "file_download_path"):
        return [
            model.opportunity_id == str(getattr(item, "opportunity_id")),
            model.file_download_path == str(getattr(item, "file_download_path")),
        ]
    raise RuntimeError(f"Unsupported model for chunk identity: {getattr(model, '__name__', model)}")


def _build_chunk_row(model: Type[Any], item: Any, *, chunk_index: int) -> Any:
    """Clone required identity fields from base row into a new chunk row."""
    if hasattr(model, "faculty_id") and hasattr(model, "additional_info_url"):
        return model(
            faculty_id=int(getattr(item, "faculty_id")),
            additional_info_url=str(getattr(item, "additional_info_url")),
            chunk_index=int(chunk_index),
        )
    if hasattr(model, "opportunity_id") and hasattr(model, "additional_info_url"):
        return model(
            opportunity_id=str(getattr(item, "opportunity_id")),
            additional_info_url=str(getattr(item, "additional_info_url")),
            chunk_index=int(chunk_index),
        )
    if hasattr(model, "opportunity_id") and hasattr(model, "file_name") and hasattr(model, "file_download_path"):
        return model(
            opportunity_id=str(getattr(item, "opportunity_id")),
            file_name=str(getattr(item, "file_name")),
            file_download_path=str(getattr(item, "file_download_path")),
            chunk_index=int(chunk_index),
        )
    raise RuntimeError(f"Unsupported model for chunk cloning: {getattr(model, '__name__', model)}")


def run_extraction_pipeline(
    *,
    model: Type[Any],
    subdir: str,
    url_getter: Callable[[Any], str],
    batch_size: int = 200,
    chunk_chars: int = 3000,
    # Overrides (recommended to pass explicitly from caller)
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    aws_region: Optional[str] = None,
    aws_profile: Optional[str] = None,
) -> Dict[str, int]:
    """
    Extracts text for pending base rows (chunk_index=0) and stores chunk files in S3.

      s3://<bucket>/<prefix>/<subdir>/<id>__<hash>__<source_name>__chunk_XXXX.txt

    The base row (chunk_index=0) stores chunk 1. Additional chunks are inserted
    as new rows in the same table with chunk_index 1..N-1.
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
            new_chunk_rows = []
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
                source_name = safe_filename(str(result.get("filename") or "source"))
                base_stem = f"{item.id}__{hash_part}__{source_name}"

                chunks = chunk_text_for_embedding(str(text), max_chars=int(chunk_chars))
                if not chunks:
                    err = "no_chunks_after_cleanup"
                    updates.append(
                        {
                            "id": item.id,
                            "content_path": None,
                            "detected_type": None,
                            "content_char_count": None,
                            "content_embedding": None,
                            "extracted_at": extracted_at,
                            "extract_status": "failed",
                            "extract_error": err,
                        }
                    )
                    processed += 1
                    failed += 1
                    continue

                try:
                    emb = embed_texts(chunks)
                    if emb.size <= 0 or emb.ndim != 2:
                        raise RuntimeError("invalid_embedding_shape")
                except Exception as e:
                    err = f"embedding_error: {type(e).__name__}: {e}"
                    err = (err[:5000] + "…") if len(err) > 5000 else err
                    updates.append(
                        {
                            "id": item.id,
                            "content_path": None,
                            "detected_type": None,
                            "content_char_count": None,
                            "content_embedding": None,
                            "extracted_at": extracted_at,
                            "extract_status": "failed",
                            "extract_error": err,
                        }
                    )
                    processed += 1
                    failed += 1
                    continue

                # Cleanup old extra chunk rows for this source before rebuilding.
                try:
                    filters = _build_chunk_identity_filters(model, item)
                    sess.execute(
                        delete(model).where(
                            and_(
                                *filters,
                                model.chunk_index > 0,
                            )
                        )
                    )
                except Exception:
                    pass

                # Store chunk files and map each chunk to one DB row.
                for idx, chunk in enumerate(chunks, start=1):
                    chunk_name = f"{base_stem}__chunk_{idx:04d}.txt"
                    chunk_key = f"{prefix}/{subdir}/{chunk_name}" if prefix else f"{subdir}/{chunk_name}"
                    s3_client.put_object(
                        Bucket=bucket,
                        Key=chunk_key,
                        Body=chunk.encode("utf-8", errors="ignore"),
                        ContentType="text/plain; charset=utf-8",
                    )

                    chunk_vec = emb[idx - 1].astype(np.float32).tolist()
                    chunk_idx_zero_based = int(idx - 1)
                    if idx == 1:
                        updates.append(
                            {
                                "id": item.id,
                                "chunk_index": int(0),
                                "content_path": chunk_key,
                                "detected_type": detected_type,
                                "content_char_count": len(chunk),
                                "content_embedding": chunk_vec,
                                "extracted_at": extracted_at,
                                "extract_status": "success",
                                "extract_error": None,
                            }
                        )
                    else:
                        row = _build_chunk_row(model, item, chunk_index=chunk_idx_zero_based)
                        row.content_path = chunk_key
                        row.detected_type = detected_type
                        row.content_char_count = len(chunk)
                        row.content_embedding = chunk_vec
                        row.extracted_at = extracted_at
                        row.extract_status = "success"
                        row.extract_error = None
                        new_chunk_rows.append(row)

                processed += 1
                done += 1

            if new_chunk_rows:
                sess.add_all(new_chunk_rows)
            dao.bulk_update(model, updates)
            sess.commit()

    return {"processed": processed, "done": done, "failed": failed}
