from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Type

import boto3
import numpy as np
import requests
from sqlalchemy import and_, delete

from config import get_embedding_client, settings
from dao.content_extraction_dao import ContentExtractionDAO
from db.db_conn import SessionLocal
from utils.content_extractor import (
    chunk_text_for_embedding,
    fetch_and_extract_one,
    safe_filename,
)
from utils.embedder import embed_texts
from utils.thread_pool import build_thread_local_getter, parallel_map, resolve_pool_size


logger = logging.getLogger(__name__)


def short_hash(value: str, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _safe_subdir(subdir: str) -> str:
    return str(subdir).strip("/").replace("\\", "/")


def _item_attr(item: Any, field: str) -> Any:
    if isinstance(item, dict):
        return item.get(field)
    return getattr(item, field)


def _trim_error(err: Any) -> str:
    msg = str(err or "unknown_error")
    return (msg[:5000] + "…") if len(msg) > 5000 else msg


def _snapshot_item(model: Type[Any], item: Any, *, url_getter: Callable[[Any], str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": int(getattr(item, "id")),
        "url": str(url_getter(item)),
    }
    if hasattr(model, "faculty_id") and hasattr(model, "additional_info_url"):
        payload["faculty_id"] = int(getattr(item, "faculty_id"))
        payload["additional_info_url"] = str(getattr(item, "additional_info_url"))
        return payload
    if hasattr(model, "opportunity_id") and hasattr(model, "additional_info_url"):
        payload["opportunity_id"] = str(getattr(item, "opportunity_id"))
        payload["additional_info_url"] = str(getattr(item, "additional_info_url"))
        return payload
    if hasattr(model, "opportunity_id") and hasattr(model, "file_name") and hasattr(model, "file_download_path"):
        payload["opportunity_id"] = str(getattr(item, "opportunity_id"))
        payload["file_name"] = str(getattr(item, "file_name"))
        payload["file_download_path"] = str(getattr(item, "file_download_path"))
        return payload
    raise RuntimeError(f"Unsupported model for extraction snapshot: {getattr(model, '__name__', model)}")


def _build_chunk_identity_filters(model: Type[Any], item: Any):
    """Build filters identifying all chunk rows of one logical file/link."""
    if hasattr(model, "faculty_id") and hasattr(model, "additional_info_url"):
        return [
            model.faculty_id == int(_item_attr(item, "faculty_id")),
            model.additional_info_url == str(_item_attr(item, "additional_info_url")),
        ]
    if hasattr(model, "opportunity_id") and hasattr(model, "additional_info_url"):
        return [
            model.opportunity_id == str(_item_attr(item, "opportunity_id")),
            model.additional_info_url == str(_item_attr(item, "additional_info_url")),
        ]
    if hasattr(model, "opportunity_id") and hasattr(model, "file_download_path"):
        return [
            model.opportunity_id == str(_item_attr(item, "opportunity_id")),
            model.file_download_path == str(_item_attr(item, "file_download_path")),
        ]
    raise RuntimeError(f"Unsupported model for chunk identity: {getattr(model, '__name__', model)}")


def _build_chunk_row(model: Type[Any], item: Any, *, chunk_index: int) -> Any:
    """Clone required identity fields from base row into a new chunk row."""
    if hasattr(model, "faculty_id") and hasattr(model, "additional_info_url"):
        return model(
            faculty_id=int(_item_attr(item, "faculty_id")),
            additional_info_url=str(_item_attr(item, "additional_info_url")),
            chunk_index=int(chunk_index),
        )
    if hasattr(model, "opportunity_id") and hasattr(model, "additional_info_url"):
        return model(
            opportunity_id=str(_item_attr(item, "opportunity_id")),
            additional_info_url=str(_item_attr(item, "additional_info_url")),
            chunk_index=int(chunk_index),
        )
    if hasattr(model, "opportunity_id") and hasattr(model, "file_name") and hasattr(model, "file_download_path"):
        return model(
            opportunity_id=str(_item_attr(item, "opportunity_id")),
            file_name=str(_item_attr(item, "file_name")),
            file_download_path=str(_item_attr(item, "file_download_path")),
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
    max_workers: int = 4,
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

    def _build_s3_client():
        session = (
            boto3.Session(profile_name=profile, region_name=region)
            if profile
            else boto3.Session(region_name=region)
        )
        return session.client("s3")

    thread_s3_client = build_thread_local_getter(_build_s3_client)
    thread_http_session = build_thread_local_getter(requests.Session)
    thread_embedder = build_thread_local_getter(lambda: get_embedding_client().build())

    def _failed_result(payload: Dict[str, Any], *, extracted_at: datetime, err: Any) -> Dict[str, Any]:
        return {
            "ok": False,
            "payload": payload,
            "extracted_at": extracted_at,
            "error": _trim_error(err),
        }

    def _process_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        extracted_at = datetime.now(timezone.utc)
        url = str(payload.get("url") or "")

        result = fetch_and_extract_one(url, session=thread_http_session())
        text = result.get("text")
        if not text:
            return _failed_result(
                payload,
                extracted_at=extracted_at,
                err=result.get("error") or "no_text",
            )

        detected_type = result.get("detected_type") or "unknown"
        hash_part = short_hash(url)
        source_name = safe_filename(str(result.get("filename") or "source"))
        base_stem = f"{payload['id']}__{hash_part}__{source_name}"

        chunks = chunk_text_for_embedding(str(text), max_chars=int(chunk_chars))
        if not chunks:
            return _failed_result(
                payload,
                extracted_at=extracted_at,
                err="no_chunks_after_cleanup",
            )

        try:
            emb = embed_texts(chunks, embedding_client=thread_embedder())
            if emb.size <= 0 or emb.ndim != 2 or emb.shape[0] != len(chunks):
                raise RuntimeError("invalid_embedding_shape")
        except Exception as e:
            return _failed_result(
                payload,
                extracted_at=extracted_at,
                err=f"embedding_error: {type(e).__name__}: {e}",
            )

        s3_client = thread_s3_client()
        chunk_rows = []
        for idx, chunk in enumerate(chunks, start=1):
            chunk_name = f"{base_stem}__chunk_{idx:04d}.txt"
            chunk_key = f"{prefix}/{subdir}/{chunk_name}" if prefix else f"{subdir}/{chunk_name}"

            s3_client.put_object(
                Bucket=bucket,
                Key=chunk_key,
                Body=chunk.encode("utf-8", errors="ignore"),
                ContentType="text/plain; charset=utf-8",
            )

            chunk_rows.append(
                {
                    "chunk_index": int(idx - 1),
                    "content_path": chunk_key,
                    "content_char_count": int(len(chunk)),
                    "content_embedding": emb[idx - 1].astype(np.float32).tolist(),
                }
            )

        return {
            "ok": True,
            "payload": payload,
            "extracted_at": extracted_at,
            "detected_type": str(detected_type),
            "chunks": chunk_rows,
        }

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

            payloads = [_snapshot_item(model, item, url_getter=url_getter) for item in items]
            pool_size = resolve_pool_size(max_workers=max_workers, task_count=len(payloads))

            def _on_worker_error(_idx: int, payload: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
                return _failed_result(
                    payload,
                    extracted_at=datetime.now(timezone.utc),
                    err=f"pipeline_error: {type(exc).__name__}: {exc}",
                )

            logger.info(
                "Extraction batch model=%s size=%s workers=%s",
                getattr(model, "__name__", str(model)),
                len(payloads),
                pool_size,
            )
            results = parallel_map(
                payloads,
                max_workers=pool_size,
                run_item=_process_payload,
                on_error=_on_worker_error,
            )

            updates = []
            new_chunk_rows = []
            for out in results:
                payload = dict(out.get("payload") or {})
                extracted_at = out.get("extracted_at") or datetime.now(timezone.utc)

                if not out.get("ok"):
                    updates.append(
                        {
                            "id": int(payload.get("id")),
                            "content_path": None,
                            "detected_type": None,
                            "content_char_count": None,
                            "content_embedding": None,
                            "extracted_at": extracted_at,
                            "extract_status": "failed",
                            "extract_error": _trim_error(out.get("error") or "unknown_error"),
                        }
                    )
                    processed += 1
                    failed += 1
                    continue

                chunks = list(out.get("chunks") or [])
                if not chunks:
                    updates.append(
                        {
                            "id": int(payload.get("id")),
                            "content_path": None,
                            "detected_type": None,
                            "content_char_count": None,
                            "content_embedding": None,
                            "extracted_at": extracted_at,
                            "extract_status": "failed",
                            "extract_error": "no_chunks_after_upload",
                        }
                    )
                    processed += 1
                    failed += 1
                    continue

                try:
                    filters = _build_chunk_identity_filters(model, payload)
                    sess.execute(
                        delete(model).where(
                            and_(
                                *filters,
                                model.chunk_index > 0,
                            )
                        )
                    )
                except Exception:
                    logger.exception(
                        "Failed removing old chunk rows model=%s source_id=%s",
                        getattr(model, "__name__", str(model)),
                        payload.get("id"),
                    )

                detected_type = str(out.get("detected_type") or "unknown")
                item_proxy = SimpleNamespace(**payload)
                for chunk in chunks:
                    chunk_index = int(chunk.get("chunk_index") or 0)
                    chunk_path = str(chunk.get("content_path") or "")
                    chunk_char_count = int(chunk.get("content_char_count") or 0)
                    chunk_vec = chunk.get("content_embedding")

                    if chunk_index == 0:
                        updates.append(
                            {
                                "id": int(payload.get("id")),
                                "chunk_index": int(0),
                                "content_path": chunk_path,
                                "detected_type": detected_type,
                                "content_char_count": chunk_char_count,
                                "content_embedding": chunk_vec,
                                "extracted_at": extracted_at,
                                "extract_status": "success",
                                "extract_error": None,
                            }
                        )
                        continue

                    row = _build_chunk_row(model, item_proxy, chunk_index=chunk_index)
                    row.content_path = chunk_path
                    row.detected_type = detected_type
                    row.content_char_count = chunk_char_count
                    row.content_embedding = chunk_vec
                    row.extracted_at = extracted_at
                    row.extract_status = "success"
                    row.extract_error = None
                    new_chunk_rows.append(row)

                processed += 1
                done += 1

            if new_chunk_rows:
                sess.add_all(new_chunk_rows)
            if updates:
                dao.bulk_update(model, updates)
            sess.commit()

    return {"processed": processed, "done": done, "failed": failed}
