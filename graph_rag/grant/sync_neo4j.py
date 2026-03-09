from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase
from sqlalchemy import bindparam, text
from sqlalchemy.orm import selectinload

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import settings
from config import get_embedding_client
from db.db_conn import SessionLocal
from db.models.opportunity import Opportunity, OpportunityAdditionalInfo, OpportunityAttachment
from graph_rag.common import (
    Neo4jSettings,
    coerce_float,
    coerce_int,
    coerce_iso_datetime,
    coerce_str_list,
    json_ready,
    load_dotenv_if_present,
    read_neo4j_settings,
    safe_text,
)
from utils.content_extractor import load_extracted_content
from utils.thread_pool import build_thread_local_getter, parallel_map

KEYWORD_RELATIONS: Dict[str, Tuple[str, str]] = {
    "HAS_RESEARCH_DOMAIN": ("research", "domain"),
    "HAS_RESEARCH_SPECIALIZATION": ("research", "specialization"),
    "HAS_APPLICATION_DOMAIN": ("application", "domain"),
    "HAS_APPLICATION_SPECIALIZATION": ("application", "specialization"),
}

SPECIALIZATION_RELATIONS = {
    "HAS_RESEARCH_SPECIALIZATION",
    "HAS_APPLICATION_SPECIALIZATION",
}

TEXT_CHUNK_RELATIONS: Dict[str, str] = {
    "summary": "HAS_SUMMARY_CHUNK",
    "additional_info": "HAS_ADDITIONAL_INFO_CHUNK",
    "attachment": "HAS_ATTACHMENT_CHUNK",
}

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class SyncLimits:
    max_additional_info: int
    max_attachments: int
    max_text_chars: int
    chunk_size_chars: int
    chunk_overlap_chars: int
    max_chunks_per_source: int


def _coerce_weight(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _safe_limit(value: int, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _clip_text(value: str, *, max_chars: int) -> str:
    cap = _safe_limit(max_chars, default=3000, minimum=100, maximum=50000)
    if len(value) <= cap:
        return value

    clipped = value[:cap].rstrip()
    sentence_end = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
    if sentence_end >= int(cap * 0.65):
        return clipped[: sentence_end + 1].rstrip()

    word_end = clipped.rfind(" ")
    if word_end >= int(cap * 0.75):
        return clipped[:word_end].rstrip()

    return clipped


def _clean_text(value: Any, *, max_chars: Optional[int] = None) -> Optional[str]:
    raw = safe_text(value)
    if not raw:
        return None

    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS_RE.sub("", text)
    lines: List[str] = []
    for line in text.split("\n"):
        compact = MULTI_SPACE_RE.sub(" ", line).strip()
        lines.append(compact)
    text = "\n".join(lines).strip()
    text = MULTI_BLANK_LINE_RE.sub("\n\n", text)
    if not text:
        return None
    if max_chars is not None:
        text = _clip_text(text, max_chars=max_chars)
    return text


def _chunk_text(
    text: str,
    *,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    max_chunks: int,
) -> List[str]:
    size = _safe_limit(chunk_size_chars, default=1200, minimum=200, maximum=5000)
    overlap = _safe_limit(chunk_overlap_chars, default=150, minimum=0, maximum=max(0, size // 2))
    limit = _safe_limit(max_chunks, default=20, minimum=1, maximum=1000)

    cleaned = _clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= size:
        return [cleaned]

    out: List[str] = []
    cursor = 0
    length = len(cleaned)
    while cursor < length and len(out) < limit:
        window_end = min(length, cursor + size)
        end = window_end

        if window_end < length:
            min_break = cursor + int(size * 0.60)
            newline_break = cleaned.rfind("\n", min_break, window_end)
            space_break = cleaned.rfind(" ", min_break, window_end)
            best_break = max(newline_break, space_break)
            if best_break > cursor:
                end = best_break

        chunk = cleaned[cursor:end].strip()
        if chunk:
            out.append(chunk)

        if end >= length:
            break

        next_cursor = max(end - overlap, cursor + 1)
        if next_cursor <= cursor:
            next_cursor = end
        cursor = next_cursor

    return out


def _load_category_map(session, opportunity_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not opportunity_ids:
        return {}

    stmt = (
        text(
            """
            SELECT opportunity_id, broad_category, specific_categories
            FROM opportunity_keywords
            WHERE opportunity_id IN :ids
            """
        )
        .bindparams(bindparam("ids", expanding=True))
    )

    out: Dict[str, Dict[str, Any]] = {}
    rows = session.execute(stmt, {"ids": [str(x) for x in opportunity_ids]}).mappings().all()
    for row in rows:
        opp_id = str(row.get("opportunity_id") or "").strip()
        if not opp_id:
            continue
        out[opp_id] = {
            "broad_category": safe_text(row.get("broad_category")),
            "specific_categories": coerce_str_list(row.get("specific_categories") or []),
        }
    return out


def _load_opportunities(
    *,
    opportunity_id: str,
    sync_all: bool,
    limit: int,
    offset: int,
) -> List[Opportunity]:
    with SessionLocal() as session:
        query = (
            session.query(Opportunity)
            .options(
                selectinload(Opportunity.additional_info),
                selectinload(Opportunity.attachments),
                selectinload(Opportunity.keyword),
            )
            .order_by(Opportunity.opportunity_id.asc())
        )

        cleaned_id = str(opportunity_id or "").strip()
        if cleaned_id:
            rows = query.filter(Opportunity.opportunity_id == cleaned_id).all()
        elif sync_all:
            if limit > 0:
                rows = query.offset(max(0, int(offset or 0))).limit(limit).all()
            else:
                rows = query.all()
        else:
            rows = query.limit(1).all()

        category_map = _load_category_map(
            session,
            [str(getattr(r, "opportunity_id", "") or "").strip() for r in rows],
        )

        for row in rows:
            oid = str(getattr(row, "opportunity_id", "") or "").strip()
            meta = category_map.get(oid, {})
            setattr(row, "_broad_category", safe_text(meta.get("broad_category")))
            setattr(row, "_specific_categories", coerce_str_list(meta.get("specific_categories") or []))

        return rows


def _grant_row(opp: Opportunity, *, max_text_chars: int) -> Dict[str, Any]:
    return {
        "opportunity_id": safe_text(opp.opportunity_id),
        "agency_name": safe_text(opp.agency_name),
        "category": safe_text(opp.category),
        "opportunity_status": safe_text(opp.opportunity_status),
        "opportunity_title": safe_text(opp.opportunity_title),
        "agency_email_address": safe_text(opp.agency_email_address),
        "applicant_types": coerce_str_list(opp.applicant_types),
        "archive_date": safe_text(opp.archive_date),
        "award_ceiling": coerce_float(opp.award_ceiling),
        "award_floor": coerce_float(opp.award_floor),
        "close_date": safe_text(opp.close_date),
        "created_at": safe_text(opp.created_at),
        "estimated_total_program_funding": coerce_float(opp.estimated_total_program_funding),
        "expected_number_of_awards": coerce_int(opp.expected_number_of_awards),
        "forecasted_award_date": safe_text(opp.forecasted_award_date),
        "forecasted_close_date": safe_text(opp.forecasted_close_date),
        "forecasted_post_date": safe_text(opp.forecasted_post_date),
        "forecasted_project_start_date": safe_text(opp.forecasted_project_start_date),
        "funding_categories": coerce_str_list(opp.funding_categories),
        "funding_instruments": coerce_str_list(opp.funding_instruments),
        "is_cost_sharing": bool(opp.is_cost_sharing) if opp.is_cost_sharing is not None else None,
        "post_date": safe_text(opp.post_date),
        # Save cleaned summary text in graph.
        "summary_description": _clean_text(opp.summary_description, max_chars=max_text_chars),
        "updated_at": None,
    }


def _additional_info_text_map(
    rows: List[OpportunityAdditionalInfo],
    *,
    include_extracted_text: bool,
    max_text_chars: int,
    extracted_text_workers: int = 1,
) -> Dict[str, str]:
    if not include_extracted_text:
        return {}

    items = load_extracted_content(
        rows,
        url_attr="additional_info_url",
        max_workers=_safe_limit(extracted_text_workers, default=1, minimum=1, maximum=64),
    )
    out: Dict[str, str] = {}
    for item in items:
        url = safe_text(item.get("url"))
        text_value = _clean_text(item.get("content"), max_chars=max_text_chars)
        if not url or not text_value:
            continue
        out[url] = text_value
    return out


def _attachment_text_map(
    rows: List[OpportunityAttachment],
    *,
    include_extracted_text: bool,
    max_text_chars: int,
    extracted_text_workers: int = 1,
) -> Dict[Tuple[str, str], str]:
    if not include_extracted_text:
        return {}

    items = load_extracted_content(
        rows,
        url_attr="file_download_path",
        title_attr="file_name",
        max_workers=_safe_limit(extracted_text_workers, default=1, minimum=1, maximum=64),
    )
    out: Dict[Tuple[str, str], str] = {}
    for item in items:
        url = safe_text(item.get("url")) or ""
        title = safe_text(item.get("title")) or ""
        text_value = _clean_text(item.get("content"), max_chars=max_text_chars)
        if not text_value:
            continue
        out[(url, title)] = text_value
    return out


def _additional_info_rows(
    opp: Opportunity,
    *,
    max_additional_info: int,
    include_extracted_text: bool,
    max_text_chars: int,
    extracted_text_workers: int = 1,
) -> List[Dict[str, Any]]:
    safe_max = _safe_limit(max_additional_info, default=50, minimum=1, maximum=1000)
    rows: List[OpportunityAdditionalInfo] = sorted(
        list(opp.additional_info or []),
        key=lambda x: (
            x.extracted_at.isoformat() if x.extracted_at is not None else "",
            int(x.id or 0),
        ),
        reverse=True,
    )[:safe_max]

    text_by_url = _additional_info_text_map(
        rows,
        include_extracted_text=include_extracted_text,
        max_text_chars=max_text_chars,
        extracted_text_workers=extracted_text_workers,
    )

    out: List[Dict[str, Any]] = []
    for row in rows:
        url = safe_text(row.additional_info_url)
        if not url:
            continue
        out.append(
            {
                "additional_info_id": int(row.id),
                "opportunity_id": safe_text(row.opportunity_id),
                "additional_info_url": url,
                "content_path": safe_text(row.content_path),
                "detected_type": safe_text(row.detected_type),
                "content_char_count": coerce_int(row.content_char_count),
                "extracted_at": coerce_iso_datetime(row.extracted_at),
                "extract_status": safe_text(row.extract_status),
                "extract_error": safe_text(row.extract_error),
                "extracted_text": text_by_url.get(url),
            }
        )
    return out


def _attachment_rows(
    opp: Opportunity,
    *,
    max_attachments: int,
    include_extracted_text: bool,
    max_text_chars: int,
    extracted_text_workers: int = 1,
) -> List[Dict[str, Any]]:
    safe_max = _safe_limit(max_attachments, default=50, minimum=1, maximum=2000)
    rows: List[OpportunityAttachment] = sorted(
        list(opp.attachments or []),
        key=lambda x: (
            x.extracted_at.isoformat() if x.extracted_at is not None else "",
            int(x.id or 0),
        ),
        reverse=True,
    )[:safe_max]

    text_by_key = _attachment_text_map(
        rows,
        include_extracted_text=include_extracted_text,
        max_text_chars=max_text_chars,
        extracted_text_workers=extracted_text_workers,
    )

    out: List[Dict[str, Any]] = []
    for row in rows:
        title = safe_text(row.file_name)
        download_url = safe_text(row.file_download_path)
        if not title and not download_url:
            continue

        out.append(
            {
                "attachment_id": int(row.id),
                "opportunity_id": safe_text(row.opportunity_id),
                "file_name": title,
                "file_download_path": download_url,
                "content_path": safe_text(row.content_path),
                "detected_type": safe_text(row.detected_type),
                "content_char_count": coerce_int(row.content_char_count),
                "extracted_at": coerce_iso_datetime(row.extracted_at),
                "extract_status": safe_text(row.extract_status),
                "extract_error": safe_text(row.extract_error),
                "extracted_text": text_by_key.get((download_url or "", title or "")),
            }
        )
    return out


def _keyword_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    out: List[Dict[str, Any]] = []
    seen = set()

    for relation, (section, bucket) in KEYWORD_RELATIONS.items():
        section_payload = payload.get(section)
        if not isinstance(section_payload, dict):
            continue

        raw_values = section_payload.get(bucket) or []
        if isinstance(raw_values, (str, int, float, dict)):
            raw_values = [raw_values]

        for item in raw_values:
            raw_weight = None
            if isinstance(item, dict):
                value = _clean_text(item.get("t") or item.get("text"), max_chars=300)
                raw_weight = item.get("w")
            else:
                value = _clean_text(item, max_chars=300)

            if not value:
                continue

            # Domain keywords are filter-only: no weight.
            if bucket == "domain":
                weight = None
            else:
                weight = _coerce_weight(raw_weight)
                if weight is None:
                    weight = 0.5

            dedupe_key = (value.lower(), section, bucket, relation)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            out.append(
                {
                    "value": value,
                    "section": section,
                    "bucket": bucket,
                    "relation": relation,
                    "weight": weight,
                    "embedding": None,
                    "embedding_model": None,
                }
            )
    return out


def _broad_category_row(opp: Opportunity) -> Optional[Dict[str, Any]]:
    broad = _clean_text(getattr(opp, "_broad_category", None), max_chars=120)
    if not broad:
        return None
    return {
        "name": broad.lower(),
    }


def _specific_category_rows(opp: Opportunity) -> List[Dict[str, Any]]:
    specific = coerce_str_list(getattr(opp, "_specific_categories", []) or [])
    out: List[Dict[str, Any]] = []
    for value in specific:
        cleaned = _clean_text(value, max_chars=120)
        if not cleaned:
            continue
        out.append({"name": cleaned.lower()})
    return out


def _build_text_chunk_rows(
    *,
    opportunity_id: str,
    grant_row: Optional[Dict[str, Any]],
    summary_text: Optional[str],
    additional_info_rows: List[Dict[str, Any]],
    attachment_rows: List[Dict[str, Any]],
    limits: SyncLimits,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def _to_line(label: str, value: Any) -> Optional[str]:
        cleaned = _clean_text(value, max_chars=400)
        if not cleaned:
            return None
        return f"{label}: {cleaned}"

    def _to_list_line(label: str, values: Any) -> Optional[str]:
        raw = coerce_str_list(values or [])
        cleaned_vals = []
        for item in raw:
            cleaned = _clean_text(item, max_chars=120)
            if cleaned:
                cleaned_vals.append(cleaned)
        if not cleaned_vals:
            return None
        return f"{label}: {', '.join(cleaned_vals)}"

    basic_lines: List[str] = []
    g = grant_row or {}
    for line in [
        _to_line("opportunity_title", g.get("opportunity_title")),
        _to_line("agency_name", g.get("agency_name")),
        _to_line("category", g.get("category")),
        _to_line("opportunity_status", g.get("opportunity_status")),
        _to_line("close_date", g.get("close_date")),
        _to_line("post_date", g.get("post_date")),
        _to_line("archive_date", g.get("archive_date")),
        _to_line("award_ceiling", g.get("award_ceiling")),
        _to_line("award_floor", g.get("award_floor")),
        _to_line("estimated_total_program_funding", g.get("estimated_total_program_funding")),
        _to_line("expected_number_of_awards", g.get("expected_number_of_awards")),
        _to_list_line("applicant_types", g.get("applicant_types")),
        _to_list_line("funding_categories", g.get("funding_categories")),
        _to_list_line("funding_instruments", g.get("funding_instruments")),
    ]:
        if line:
            basic_lines.append(line)

    basic_text = _clean_text("\n".join(basic_lines), max_chars=limits.max_text_chars)
    summary_body = _clean_text(summary_text, max_chars=limits.max_text_chars)
    summary_parts: List[str] = []
    if basic_text:
        summary_parts.append(f"[grant_basic_info]\n{basic_text}")
    if summary_body:
        summary_parts.append(f"[grant_summary]\n{summary_body}")
    summary_combined = _clean_text("\n\n".join(summary_parts), max_chars=limits.max_text_chars)

    if summary_combined:
        chunks = _chunk_text(
            summary_combined,
            chunk_size_chars=limits.chunk_size_chars,
            chunk_overlap_chars=limits.chunk_overlap_chars,
            max_chunks=limits.max_chunks_per_source,
        )
        for idx, chunk_text in enumerate(chunks):
            rows.append(
                {
                    "chunk_id": f"{opportunity_id}|summary|main|{idx}",
                    "opportunity_id": opportunity_id,
                    "source_type": "summary",
                    "source_ref_id": "main",
                    "source_url": None,
                    "source_title": None,
                    "chunk_index": idx,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "relation": TEXT_CHUNK_RELATIONS["summary"],
                    "embedding": None,
                    "embedding_model": None,
                }
            )

    for row in additional_info_rows:
        text_value = row.get("extracted_text")
        if not text_value:
            continue
        source_ref_id = str(row.get("additional_info_id") or "")
        chunks = _chunk_text(
            str(text_value),
            chunk_size_chars=limits.chunk_size_chars,
            chunk_overlap_chars=limits.chunk_overlap_chars,
            max_chunks=limits.max_chunks_per_source,
        )
        for idx, chunk_text in enumerate(chunks):
            rows.append(
                {
                    "chunk_id": f"{opportunity_id}|ai|{source_ref_id}|{idx}",
                    "opportunity_id": opportunity_id,
                    "source_type": "additional_info",
                    "source_ref_id": source_ref_id,
                    "source_url": row.get("additional_info_url"),
                    "source_title": None,
                    "chunk_index": idx,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "relation": TEXT_CHUNK_RELATIONS["additional_info"],
                    "embedding": None,
                    "embedding_model": None,
                }
            )

    for row in attachment_rows:
        text_value = row.get("extracted_text")
        if not text_value:
            continue
        source_ref_id = str(row.get("attachment_id") or "")
        chunks = _chunk_text(
            str(text_value),
            chunk_size_chars=limits.chunk_size_chars,
            chunk_overlap_chars=limits.chunk_overlap_chars,
            max_chunks=limits.max_chunks_per_source,
        )
        for idx, chunk_text in enumerate(chunks):
            rows.append(
                {
                    "chunk_id": f"{opportunity_id}|att|{source_ref_id}|{idx}",
                    "opportunity_id": opportunity_id,
                    "source_type": "attachment",
                    "source_ref_id": source_ref_id,
                    "source_url": row.get("file_download_path"),
                    "source_title": row.get("file_name"),
                    "chunk_index": idx,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "relation": TEXT_CHUNK_RELATIONS["attachment"],
                    "embedding": None,
                    "embedding_model": None,
                }
            )

    return rows


def _embed_text_values(
    texts: List[str],
    *,
    batch_size: int,
    max_workers: int = 4,
) -> Dict[str, List[float]]:
    unique: List[str] = []
    seen = set()
    for raw in texts or []:
        text_value = _clean_text(raw)
        if not text_value:
            continue
        key = text_value.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(text_value)

    if not unique:
        return {}

    safe_batch = _safe_limit(batch_size, default=12, minimum=1, maximum=128)
    safe_workers = _safe_limit(max_workers, default=4, minimum=1, maximum=32)
    batches = [unique[start : start + safe_batch] for start in range(0, len(unique), safe_batch)]

    def _embed_batch(model, batch: List[str]) -> Dict[str, List[float]]:
        local_out: Dict[str, List[float]] = {}
        vectors = model.embed_documents(batch)
        for text_value, vec in zip(batch, vectors):
            if vec is None:
                continue
            local_out[text_value] = [float(x) for x in vec]
        return local_out

    if len(batches) <= 1 or safe_workers <= 1:
        model = get_embedding_client().build()
        out: Dict[str, List[float]] = {}
        for batch in batches:
            out.update(_embed_batch(model, batch))
        return out

    get_model = build_thread_local_getter(lambda: get_embedding_client().build())
    batch_maps = parallel_map(
        batches,
        max_workers=min(safe_workers, len(batches)),
        run_item=lambda batch: _embed_batch(get_model(), batch),
    )
    out: Dict[str, List[float]] = {}
    for mapping in batch_maps:
        out.update(mapping)
    return out


def sync_grant_to_neo4j(
    *,
    driver,
    settings_neo4j: Neo4jSettings,
    opp: Opportunity,
    limits: SyncLimits,
    include_extracted_text: bool,
    include_embeddings: bool,
    embedding_batch_size: int,
    extracted_text_workers: int = 1,
) -> Dict[str, Any]:
    grant_row = _grant_row(opp, max_text_chars=limits.max_text_chars)
    additional_info_rows = _additional_info_rows(
        opp,
        max_additional_info=limits.max_additional_info,
        include_extracted_text=include_extracted_text,
        max_text_chars=limits.max_text_chars,
        extracted_text_workers=extracted_text_workers,
    )
    attachment_rows = _attachment_rows(
        opp,
        max_attachments=limits.max_attachments,
        include_extracted_text=include_extracted_text,
        max_text_chars=limits.max_text_chars,
        extracted_text_workers=extracted_text_workers,
    )
    # Initial GraphRAG sync phase: do not upload keyword nodes/edges yet.
    keyword_rows: List[Dict[str, Any]] = []
    broad_category_row = _broad_category_row(opp)
    specific_category_rows = _specific_category_rows(opp)

    opportunity_id = grant_row.get("opportunity_id")
    if not opportunity_id:
        raise ValueError("Opportunity has no opportunity_id; cannot sync.")

    text_chunk_rows = _build_text_chunk_rows(
        opportunity_id=str(opportunity_id),
        grant_row=grant_row,
        summary_text=grant_row.get("summary_description"),
        additional_info_rows=additional_info_rows,
        attachment_rows=attachment_rows,
        limits=limits,
    )

    embedding_model = (settings.bedrock_embed_model_id or "").strip() if include_embeddings else ""

    if include_embeddings:
        spec_texts = [str(x.get("value") or "") for x in keyword_rows if x.get("relation") in SPECIALIZATION_RELATIONS]
        spec_embeddings = _embed_text_values(spec_texts, batch_size=embedding_batch_size)
        for row in keyword_rows:
            if row.get("relation") in SPECIALIZATION_RELATIONS:
                key_text = _clean_text(row.get("value"))
                emb = spec_embeddings.get(key_text or "")
                if emb:
                    row["embedding"] = emb
                    row["embedding_model"] = embedding_model

        chunk_embeddings = _embed_text_values(
            [str(x.get("text") or "") for x in text_chunk_rows],
            batch_size=embedding_batch_size,
        )
        for row in text_chunk_rows:
            key_text = _clean_text(row.get("text"))
            emb = chunk_embeddings.get(key_text or "")
            if emb:
                row["embedding"] = emb
                row["embedding_model"] = embedding_model

    driver.execute_query(
        """
        MERGE (g:Grant {opportunity_id: $row.opportunity_id})
        SET
            g.agency_name = $row.agency_name,
            g.category = $row.category,
            g.opportunity_status = $row.opportunity_status,
            g.opportunity_title = $row.opportunity_title,
            g.agency_email_address = $row.agency_email_address,
            g.applicant_types = $row.applicant_types,
            g.archive_date = $row.archive_date,
            g.award_ceiling = $row.award_ceiling,
            g.award_floor = $row.award_floor,
            g.close_date = $row.close_date,
            g.created_at = $row.created_at,
            g.estimated_total_program_funding = $row.estimated_total_program_funding,
            g.expected_number_of_awards = $row.expected_number_of_awards,
            g.forecasted_award_date = $row.forecasted_award_date,
            g.forecasted_close_date = $row.forecasted_close_date,
            g.forecasted_post_date = $row.forecasted_post_date,
            g.forecasted_project_start_date = $row.forecasted_project_start_date,
            g.funding_categories = $row.funding_categories,
            g.funding_instruments = $row.funding_instruments,
            g.is_cost_sharing = $row.is_cost_sharing,
            g.post_date = $row.post_date,
            g.summary_description = $row.summary_description,
            g.updated_at = datetime()
        """,
        parameters_={"row": grant_row},
        database_=settings_neo4j.database,
    )

    # Refresh grant-owned nodes/edges.
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})
        OPTIONAL MATCH (g)-[:HAS_ADDITIONAL_INFO]->(ai:GrantAdditionalInfo)
        DETACH DELETE ai
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings_neo4j.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})
        OPTIONAL MATCH (g)-[:HAS_ATTACHMENT]->(att:GrantAttachment)
        DETACH DELETE att
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings_neo4j.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})
        OPTIONAL MATCH (g)-[r]->(c:GrantTextChunk)
        WHERE type(r) IN [
            'HAS_BASIC_INFO_CHUNK',
            'HAS_SUMMARY_CHUNK',
            'HAS_ADDITIONAL_INFO_CHUNK',
            'HAS_ATTACHMENT_CHUNK'
        ]
        DETACH DELETE c
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings_neo4j.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:GrantKeyword)
        WHERE type(r) IN [
            'HAS_RESEARCH_DOMAIN',
            'HAS_RESEARCH_SPECIALIZATION',
            'HAS_APPLICATION_DOMAIN',
            'HAS_APPLICATION_SPECIALIZATION'
        ]
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings_neo4j.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:Agency)
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings_neo4j.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:GrantBroadCategory)
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings_neo4j.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:GrantSpecificCategory)
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings_neo4j.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:OpportunityCategory)
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings_neo4j.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:ApplicantType)
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings_neo4j.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:FundingCategory)
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings_neo4j.database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:FundingInstrument)
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=settings_neo4j.database,
    )

    agency_name = grant_row.get("agency_name")
    if agency_name:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            MERGE (a:Agency {name: $agency_name})
            SET a.updated_at = datetime()
            MERGE (g)-[:FUNDED_BY]->(a)
            """,
            parameters_={"opportunity_id": opportunity_id, "agency_name": agency_name},
            database_=settings_neo4j.database,
        )

    opportunity_category = safe_text(grant_row.get("category"))
    if opportunity_category:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            MERGE (c:OpportunityCategory {name: $category_name})
            SET c.updated_at = datetime()
            MERGE (g)-[:IN_OPPORTUNITY_CATEGORY]->(c)
            """,
            parameters_={
                "opportunity_id": opportunity_id,
                "category_name": opportunity_category,
            },
            database_=settings_neo4j.database,
        )

    if broad_category_row:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            MERGE (bc:GrantBroadCategory {name: $row.name})
            SET bc.updated_at = datetime()
            MERGE (g)-[:HAS_BROAD_CATEGORY]->(bc)
            """,
            parameters_={"opportunity_id": opportunity_id, "row": broad_category_row},
            database_=settings_neo4j.database,
        )

    if specific_category_rows:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            UNWIND $rows AS row
            MERGE (sc:GrantSpecificCategory {name: row.name})
            SET sc.updated_at = datetime()
            MERGE (g)-[:HAS_SPECIFIC_CATEGORY]->(sc)
            """,
            parameters_={"opportunity_id": opportunity_id, "rows": specific_category_rows},
            database_=settings_neo4j.database,
        )

    applicant_types = coerce_str_list(grant_row.get("applicant_types") or [])
    if applicant_types:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            UNWIND $items AS name
            MERGE (a:ApplicantType {name: name})
            SET a.updated_at = datetime()
            MERGE (g)-[:HAS_APPLICANT_TYPE]->(a)
            """,
            parameters_={"opportunity_id": opportunity_id, "items": applicant_types},
            database_=settings_neo4j.database,
        )

    funding_categories = coerce_str_list(grant_row.get("funding_categories") or [])
    if funding_categories:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            UNWIND $items AS name
            MERGE (fc:FundingCategory {name: name})
            SET fc.updated_at = datetime()
            MERGE (g)-[:HAS_FUNDING_CATEGORY]->(fc)
            """,
            parameters_={"opportunity_id": opportunity_id, "items": funding_categories},
            database_=settings_neo4j.database,
        )

    funding_instruments = coerce_str_list(grant_row.get("funding_instruments") or [])
    if funding_instruments:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            UNWIND $items AS name
            MERGE (fi:FundingInstrument {name: name})
            SET fi.updated_at = datetime()
            MERGE (g)-[:HAS_FUNDING_INSTRUMENT]->(fi)
            """,
            parameters_={"opportunity_id": opportunity_id, "items": funding_instruments},
            database_=settings_neo4j.database,
        )

    if additional_info_rows:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            UNWIND $rows AS row
            MERGE (ai:GrantAdditionalInfo {additional_info_id: row.additional_info_id})
            SET
                ai.opportunity_id = row.opportunity_id,
                ai.additional_info_url = row.additional_info_url,
                ai.content_path = row.content_path,
                ai.detected_type = row.detected_type,
                ai.content_char_count = row.content_char_count,
                ai.extracted_at = row.extracted_at,
                ai.extract_status = row.extract_status,
                ai.extract_error = row.extract_error,
                ai.extracted_text = row.extracted_text,
                ai.updated_at = datetime()
            MERGE (g)-[:HAS_ADDITIONAL_INFO]->(ai)
            """,
            parameters_={"opportunity_id": opportunity_id, "rows": additional_info_rows},
            database_=settings_neo4j.database,
        )

    if attachment_rows:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            UNWIND $rows AS row
            MERGE (att:GrantAttachment {attachment_id: row.attachment_id})
            SET
                att.opportunity_id = row.opportunity_id,
                att.file_name = row.file_name,
                att.file_download_path = row.file_download_path,
                att.content_path = row.content_path,
                att.detected_type = row.detected_type,
                att.content_char_count = row.content_char_count,
                att.extracted_at = row.extracted_at,
                att.extract_status = row.extract_status,
                att.extract_error = row.extract_error,
                att.extracted_text = row.extracted_text,
                att.updated_at = datetime()
            MERGE (g)-[:HAS_ATTACHMENT]->(att)
            """,
            parameters_={"opportunity_id": opportunity_id, "rows": attachment_rows},
            database_=settings_neo4j.database,
        )

    for relation in KEYWORD_RELATIONS:
        rows = [item for item in keyword_rows if item.get("relation") == relation]
        if not rows:
            continue

        driver.execute_query(
            f"""
            MATCH (g:Grant {{opportunity_id: $opportunity_id}})
            UNWIND $rows AS row
            MERGE (k:GrantKeyword {{
                value: row.value,
                section: row.section,
                bucket: row.bucket
            }})
            SET
                k.embedding = row.embedding,
                k.embedding_model = row.embedding_model,
                k.updated_at = datetime()
            MERGE (g)-[r:{relation}]->(k)
            SET
                r.weight = row.weight,
                r.updated_at = datetime()
            """,
            parameters_={"opportunity_id": opportunity_id, "rows": rows},
            database_=settings_neo4j.database,
        )

    for relation in TEXT_CHUNK_RELATIONS.values():
        rows = [item for item in text_chunk_rows if item.get("relation") == relation]
        if not rows:
            continue
        driver.execute_query(
            f"""
            MATCH (g:Grant {{opportunity_id: $opportunity_id}})
            UNWIND $rows AS row
            MERGE (c:GrantTextChunk {{chunk_id: row.chunk_id}})
            SET
                c.opportunity_id = row.opportunity_id,
                c.source_type = row.source_type,
                c.source_ref_id = row.source_ref_id,
                c.source_url = row.source_url,
                c.source_title = row.source_title,
                c.chunk_index = row.chunk_index,
                c.text = row.text,
                c.char_count = row.char_count,
                c.embedding = row.embedding,
                c.embedding_model = row.embedding_model,
                c.updated_at = datetime()
            MERGE (g)-[r:{relation}]->(c)
            SET r.updated_at = datetime()
            """,
            parameters_={"opportunity_id": opportunity_id, "rows": rows},
            database_=settings_neo4j.database,
        )

    return {
        "opportunity_id": opportunity_id,
        "counts": {
            "additional_info": len(additional_info_rows),
            "attachments": len(attachment_rows),
            "keywords": len(keyword_rows),
            "text_chunks": len(text_chunk_rows),
            "embedded_specialization_keywords": len(
                [x for x in keyword_rows if x.get("relation") in SPECIALIZATION_RELATIONS and x.get("embedding")]
            ),
            "embedded_text_chunks": len([x for x in text_chunk_rows if x.get("embedding")]),
            "has_broad_category": 1 if broad_category_row else 0,
            "specific_categories": len(specific_category_rows),
            "applicant_types": len(applicant_types),
            "funding_categories": len(funding_categories),
            "funding_instruments": len(funding_instruments),
        },
    }


def verify_grant_from_neo4j(
    *,
    driver,
    settings_neo4j: Neo4jSettings,
    opportunity_id: str,
    additional_info_limit: int,
    attachment_limit: int,
    chunk_limit: int,
) -> Dict[str, Any]:
    records, _, _ = driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})
        CALL (g) {
            OPTIONAL MATCH (g)-[:FUNDED_BY]->(a:Agency)
            RETURN a {.name} AS agency
        }
        CALL (g) {
            OPTIONAL MATCH (g)-[:HAS_BROAD_CATEGORY]->(bc:GrantBroadCategory)
            RETURN collect(DISTINCT bc.name) AS broad_categories
        }
        CALL (g) {
            OPTIONAL MATCH (g)-[:HAS_SPECIFIC_CATEGORY]->(sc:GrantSpecificCategory)
            RETURN collect(DISTINCT sc.name) AS specific_categories
        }
        CALL (g) {
            OPTIONAL MATCH (g)-[:HAS_APPLICANT_TYPE]->(at:ApplicantType)
            RETURN collect(DISTINCT at.name) AS applicant_types
        }
        CALL (g) {
            OPTIONAL MATCH (g)-[:HAS_FUNDING_CATEGORY]->(fc:FundingCategory)
            RETURN collect(DISTINCT fc.name) AS funding_categories
        }
        CALL (g) {
            OPTIONAL MATCH (g)-[:HAS_FUNDING_INSTRUMENT]->(fi:FundingInstrument)
            RETURN collect(DISTINCT fi.name) AS funding_instruments
        }
        CALL (g) {
            OPTIONAL MATCH (g)-[:HAS_ADDITIONAL_INFO]->(ai:GrantAdditionalInfo)
            WITH ai ORDER BY ai.extracted_at DESC, ai.additional_info_id DESC
            RETURN [x IN collect(ai)[0..$additional_info_limit] WHERE x IS NOT NULL | x {
                .additional_info_id,
                .additional_info_url,
                .extract_status,
                .content_char_count
            }] AS additional_info
        }
        CALL (g) {
            OPTIONAL MATCH (g)-[:HAS_ATTACHMENT]->(att:GrantAttachment)
            WITH att ORDER BY att.extracted_at DESC, att.attachment_id DESC
            RETURN [x IN collect(att)[0..$attachment_limit] WHERE x IS NOT NULL | x {
                .attachment_id,
                .file_name,
                .extract_status,
                .content_char_count
            }] AS attachments
        }
        CALL (g) {
            OPTIONAL MATCH (g)-[r]->(k:GrantKeyword)
            WHERE type(r) IN [
                'HAS_RESEARCH_DOMAIN',
                'HAS_RESEARCH_SPECIALIZATION',
                'HAS_APPLICATION_DOMAIN',
                'HAS_APPLICATION_SPECIALIZATION'
            ]
            RETURN collect(DISTINCT {
                relation: type(r),
                value: k.value,
                section: k.section,
                bucket: k.bucket,
                weight: r.weight,
                has_embedding: k.embedding IS NOT NULL
            }) AS keywords
        }
        CALL (g) {
            OPTIONAL MATCH (g)-[r]->(c:GrantTextChunk)
            WHERE type(r) IN [
                'HAS_SUMMARY_CHUNK',
                'HAS_ADDITIONAL_INFO_CHUNK',
                'HAS_ATTACHMENT_CHUNK'
            ]
            WITH r, c ORDER BY c.source_type ASC, c.source_ref_id ASC, c.chunk_index ASC
            RETURN [x IN collect({relation: type(r), chunk: c})[0..$chunk_limit] | {
                relation: x.relation,
                chunk_id: x.chunk.chunk_id,
                source_type: x.chunk.source_type,
                source_ref_id: x.chunk.source_ref_id,
                chunk_index: x.chunk.chunk_index,
                char_count: x.chunk.char_count,
                has_embedding: x.chunk.embedding IS NOT NULL
            }] AS chunks
        }
        RETURN
            g {
                .opportunity_id,
                .opportunity_title,
                .agency_name,
                .opportunity_status,
                .category,
                .award_ceiling,
                .award_floor,
                .close_date
            } AS grant,
            agency,
            broad_categories,
            specific_categories,
            applicant_types,
            funding_categories,
            funding_instruments,
            additional_info,
            attachments,
            keywords,
            chunks
        """,
        parameters_={
            "opportunity_id": str(opportunity_id or "").strip(),
            "additional_info_limit": _safe_limit(additional_info_limit, default=5, minimum=1, maximum=100),
            "attachment_limit": _safe_limit(attachment_limit, default=5, minimum=1, maximum=100),
            "chunk_limit": _safe_limit(chunk_limit, default=12, minimum=1, maximum=200),
        },
        database_=settings_neo4j.database,
    )

    if not records:
        return {
            "opportunity_id": str(opportunity_id or "").strip(),
            "found": False,
        }

    row = records[0]
    additional_info = [item for item in (row.get("additional_info") or []) if item]
    attachments = [item for item in (row.get("attachments") or []) if item]
    keywords = [item for item in (row.get("keywords") or []) if item and item.get("value")]
    chunks = [item for item in (row.get("chunks") or []) if item]

    return {
        "opportunity_id": str(opportunity_id or "").strip(),
        "found": True,
        "grant": row.get("grant"),
        "counts": {
            "additional_info": len(additional_info),
            "attachments": len(attachments),
            "keywords": len(keywords),
            "specialization_keywords": len([k for k in keywords if k.get("bucket") == "specialization"]),
            "specialization_keywords_embedded": len(
                [k for k in keywords if k.get("bucket") == "specialization" and k.get("has_embedding")]
            ),
            "text_chunks": len(chunks),
            "text_chunks_embedded": len([c for c in chunks if c.get("has_embedding")]),
            "broad_categories": len([x for x in (row.get("broad_categories") or []) if x]),
            "specific_categories": len([x for x in (row.get("specific_categories") or []) if x]),
            "applicant_types": len([x for x in (row.get("applicant_types") or []) if x]),
            "funding_categories": len([x for x in (row.get("funding_categories") or []) if x]),
            "funding_instruments": len([x for x in (row.get("funding_instruments") or []) if x]),
        },
        "preview": {
            "agency": row.get("agency"),
            "broad_categories": row.get("broad_categories"),
            "specific_categories": row.get("specific_categories"),
            "applicant_types": row.get("applicant_types"),
            "funding_categories": row.get("funding_categories"),
            "funding_instruments": row.get("funding_instruments"),
            "additional_info": additional_info,
            "attachments": attachments,
            "keywords": keywords,
            "chunks": chunks,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync Grant GraphRAG data from Postgres into Neo4j.")
    parser.add_argument("--opportunity-id", type=str, default="", help="Sync one opportunity id.")
    parser.add_argument("--all", action="store_true", help="Sync all opportunities.")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows when using --all (0 = no limit).")
    parser.add_argument("--offset", type=int, default=0, help="Offset rows when using --all.")

    parser.add_argument("--max-additional-info", type=int, default=100, help="Max additional-info rows per grant.")
    parser.add_argument("--max-attachments", type=int, default=100, help="Max attachment rows per grant.")
    parser.add_argument("--max-text-chars", type=int, default=4000, help="Max chars for each cleaned text block.")
    parser.add_argument("--chunk-size-chars", type=int, default=1200, help="Chunk size for text embeddings.")
    parser.add_argument("--chunk-overlap-chars", type=int, default=150, help="Chunk overlap for text embeddings.")
    parser.add_argument("--max-chunks-per-source", type=int, default=24, help="Max chunks for each source document.")
    parser.add_argument("--embedding-batch-size", type=int, default=12, help="Embedding batch size.")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip specialization/chunk embedding writes.")
    parser.add_argument(
        "--skip-extracted-text",
        action="store_true",
        help="Do not load extracted S3 text for additional info / attachments.",
    )

    parser.add_argument("--verify-opportunity-id", type=str, default="", help="Run verify query for this id after sync.")
    parser.add_argument("--verify-additional-info-limit", type=int, default=5, help="Additional-info rows in verify output.")
    parser.add_argument("--verify-attachment-limit", type=int, default=5, help="Attachment rows in verify output.")
    parser.add_argument("--verify-chunk-limit", type=int, default=12, help="Text chunk rows in verify output.")
    parser.add_argument("--json-only", action="store_true", help="Print only JSON output.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on first sync error.")

    parser.add_argument("--uri", type=str, default="", help="Neo4j URI. Fallback: NEO4J_URI")
    parser.add_argument("--username", type=str, default="", help="Neo4j username. Fallback: NEO4J_USERNAME")
    parser.add_argument("--password", type=str, default="", help="Neo4j password. Fallback: NEO4J_PASSWORD")
    parser.add_argument("--database", type=str, default="", help="Neo4j database. Fallback: NEO4J_DATABASE or neo4j")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    load_dotenv_if_present()

    settings_neo4j = read_neo4j_settings(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    rows = _load_opportunities(
        opportunity_id=args.opportunity_id,
        sync_all=bool(args.all),
        limit=max(0, int(args.limit or 0)),
        offset=max(0, int(args.offset or 0)),
    )

    if not rows:
        raise RuntimeError("No opportunities found for requested sync scope.")

    limits = SyncLimits(
        max_additional_info=_safe_limit(args.max_additional_info, default=100, minimum=1, maximum=1000),
        max_attachments=_safe_limit(args.max_attachments, default=100, minimum=1, maximum=2000),
        max_text_chars=_safe_limit(args.max_text_chars, default=4000, minimum=100, maximum=50000),
        chunk_size_chars=_safe_limit(args.chunk_size_chars, default=1200, minimum=200, maximum=5000),
        chunk_overlap_chars=_safe_limit(args.chunk_overlap_chars, default=150, minimum=0, maximum=2500),
        max_chunks_per_source=_safe_limit(args.max_chunks_per_source, default=24, minimum=1, maximum=1000),
    )

    include_extracted_text = not bool(args.skip_extracted_text)
    include_embeddings = not bool(args.skip_embeddings)
    embedding_batch_size = _safe_limit(args.embedding_batch_size, default=12, minimum=1, maximum=128)

    synced: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with GraphDatabase.driver(
        settings_neo4j.uri,
        auth=(settings_neo4j.username, settings_neo4j.password),
    ) as driver:
        driver.verify_connectivity()

        for opp in rows:
            opportunity_id = safe_text(getattr(opp, "opportunity_id", None)) or ""
            try:
                result = sync_grant_to_neo4j(
                    driver=driver,
                    settings_neo4j=settings_neo4j,
                    opp=opp,
                    limits=limits,
                    include_extracted_text=include_extracted_text,
                    include_embeddings=include_embeddings,
                    embedding_batch_size=embedding_batch_size,
                )
                synced.append(result)
            except Exception as exc:
                errors.append(
                    {
                        "opportunity_id": opportunity_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if args.stop_on_error:
                    break

        verify_id = safe_text(args.verify_opportunity_id)
        if not verify_id and len(synced) == 1:
            verify_id = safe_text(synced[0].get("opportunity_id"))

        verify = None
        if verify_id:
            verify = verify_grant_from_neo4j(
                driver=driver,
                settings_neo4j=settings_neo4j,
                opportunity_id=verify_id,
                additional_info_limit=args.verify_additional_info_limit,
                attachment_limit=args.verify_attachment_limit,
                chunk_limit=args.verify_chunk_limit,
            )

    totals = {
        "grants_synced": len(synced),
        "grants_failed": len(errors),
        "additional_info": sum(int(item.get("counts", {}).get("additional_info", 0)) for item in synced),
        "attachments": sum(int(item.get("counts", {}).get("attachments", 0)) for item in synced),
        "keywords": sum(int(item.get("counts", {}).get("keywords", 0)) for item in synced),
        "text_chunks": sum(int(item.get("counts", {}).get("text_chunks", 0)) for item in synced),
        "embedded_specialization_keywords": sum(
            int(item.get("counts", {}).get("embedded_specialization_keywords", 0)) for item in synced
        ),
        "embedded_text_chunks": sum(int(item.get("counts", {}).get("embedded_text_chunks", 0)) for item in synced),
        "broad_categories": sum(int(item.get("counts", {}).get("has_broad_category", 0)) for item in synced),
        "specific_categories": sum(int(item.get("counts", {}).get("specific_categories", 0)) for item in synced),
        "applicant_types": sum(int(item.get("counts", {}).get("applicant_types", 0)) for item in synced),
        "funding_categories": sum(int(item.get("counts", {}).get("funding_categories", 0)) for item in synced),
        "funding_instruments": sum(int(item.get("counts", {}).get("funding_instruments", 0)) for item in synced),
    }

    payload = {
        "scope": {
            "opportunity_id": safe_text(args.opportunity_id),
            "all": bool(args.all),
            "limit": max(0, int(args.limit or 0)),
            "offset": max(0, int(args.offset or 0)),
            "include_extracted_text": include_extracted_text,
            "include_embeddings": include_embeddings,
            "embedding_model": (settings.bedrock_embed_model_id or "").strip() if include_embeddings else None,
        },
        "totals": totals,
        "synced": synced,
        "errors": errors,
        "verify": verify,
    }

    if not args.json_only:
        print("Grant GraphRAG sync complete.")
        print(f"  synced grants                    : {totals['grants_synced']}")
        print(f"  failed grants                    : {totals['grants_failed']}")
        print(f"  additional info                  : {totals['additional_info']}")
        print(f"  attachments                      : {totals['attachments']}")
        print(f"  keyword edges                    : {totals['keywords']}")
        print(f"  text chunks                      : {totals['text_chunks']}")
        print(f"  embedded specialization keywords : {totals['embedded_specialization_keywords']}")
        print(f"  embedded text chunks             : {totals['embedded_text_chunks']}")
        print(f"  broad categories                 : {totals['broad_categories']}")
        print(f"  specific categories              : {totals['specific_categories']}")
        print(f"  applicant types                  : {totals['applicant_types']}")
        print(f"  funding categories               : {totals['funding_categories']}")
        print(f"  funding instruments              : {totals['funding_instruments']}")
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
