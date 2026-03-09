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
from sqlalchemy import func
from sqlalchemy.orm import selectinload

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import get_embedding_client
from config import settings
from db.db_conn import SessionLocal
from db.models.faculty import Faculty, FacultyAdditionalInfo, FacultyPublication
from graph_rag.common import (
    Neo4jSettings,
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
    "biography": "HAS_BIO_CHUNK",
    "additional_info": "HAS_ADDITIONAL_INFO_CHUNK",
}

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class SyncLimits:
    max_publications: int
    max_additional_info: int
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


def _load_faculties(
    *,
    email: str,
    sync_all: bool,
    limit: int,
    offset: int,
) -> List[Faculty]:
    with SessionLocal() as session:
        query = (
            session.query(Faculty)
            .options(
                selectinload(Faculty.additional_info),
                selectinload(Faculty.publications),
                selectinload(Faculty.keyword),
            )
            .order_by(Faculty.faculty_id.asc())
        )

        cleaned_email = str(email or "").strip().lower()
        if cleaned_email:
            return query.filter(func.lower(Faculty.email) == cleaned_email).all()

        if sync_all:
            if limit > 0:
                return query.offset(max(0, int(offset or 0))).limit(limit).all()
            return query.all()

        return query.limit(1).all()


def _faculty_row(fac: Faculty, *, max_text_chars: int) -> Dict[str, Any]:
    return {
        "faculty_id": int(fac.faculty_id),
        "email": str(fac.email or "").strip().lower(),
        "source_url": safe_text(fac.source_url),
        "name": _clean_text(fac.name, max_chars=300),
        "phone": safe_text(fac.phone),
        "position": _clean_text(fac.position, max_chars=300),
        "organization": _clean_text(fac.organization, max_chars=600),
        "organizations": [_clean_text(x, max_chars=200) for x in coerce_str_list(fac.organizations)],
        "address": _clean_text(fac.address, max_chars=max_text_chars),
        "biography": _clean_text(fac.biography, max_chars=max_text_chars),
        "degrees": [_clean_text(x, max_chars=200) for x in coerce_str_list(fac.degrees)],
        "expertise": [_clean_text(x, max_chars=200) for x in coerce_str_list(fac.expertise)],
        "profile_last_refreshed_at": coerce_iso_datetime(fac.profile_last_refreshed_at),
    }


def _additional_info_text_map(
    rows: List[FacultyAdditionalInfo],
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


def _additional_info_rows(
    fac: Faculty,
    *,
    max_additional_info: int,
    include_extracted_text: bool,
    max_text_chars: int,
    extracted_text_workers: int = 1,
) -> List[Dict[str, Any]]:
    safe_max = _safe_limit(max_additional_info, default=50, minimum=1, maximum=1000)
    rows: List[FacultyAdditionalInfo] = sorted(
        list(fac.additional_info or []),
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
                "faculty_id": int(fac.faculty_id),
                "additional_info_url": url,
                "content_path": safe_text(row.content_path),
                "detected_type": safe_text(row.detected_type),
                "content_char_count": int(row.content_char_count) if row.content_char_count is not None else None,
                "extracted_at": coerce_iso_datetime(row.extracted_at),
                "extract_status": safe_text(row.extract_status),
                "extract_error": safe_text(row.extract_error),
                "extracted_text": text_by_url.get(url),
            }
        )
    return out


def _publication_rows(
    fac: Faculty,
    *,
    max_publications: int,
    max_text_chars: int,
) -> List[Dict[str, Any]]:
    safe_max = _safe_limit(max_publications, default=100, minimum=1, maximum=3000)
    rows: List[FacultyPublication] = sorted(
        list(fac.publications or []),
        key=lambda x: ((x.year or 0), (x.id or 0)),
        reverse=True,
    )[:safe_max]

    out: List[Dict[str, Any]] = []
    for row in rows:
        title = _clean_text(row.title, max_chars=500)
        if not title:
            continue

        out.append(
            {
                "publication_id": int(row.id),
                "faculty_id": int(fac.faculty_id),
                "title": title,
                "abstract": _clean_text(row.abstract, max_chars=max_text_chars),
                "year": int(row.year) if row.year is not None else None,
                "abstract_embedding": None,
                "embedding_model": None,
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

            # Domain keywords are filter-only.
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


def _build_additional_info_chunk_rows(
    *,
    email: str,
    additional_info_rows: List[Dict[str, Any]],
    limits: SyncLimits,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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
                    "chunk_id": f"{email}|ai|{source_ref_id}|{idx}",
                    "email": email,
                    "source_type": "additional_info",
                    "source_ref_id": source_ref_id,
                    "source_url": row.get("additional_info_url"),
                    "chunk_index": idx,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "relation": TEXT_CHUNK_RELATIONS["additional_info"],
                    "embedding": None,
                    "embedding_model": None,
                }
            )
    return rows


def _build_biography_chunk_rows(
    *,
    email: str,
    faculty_row: Dict[str, Any],
) -> List[Dict[str, Any]]:
    text_value = _clean_text((faculty_row or {}).get("biography"))
    if not text_value:
        return []
    return [
        {
            "chunk_id": f"{email}|bio|main|0",
            "email": email,
            "source_type": "biography",
            "source_ref_id": "main",
            "source_url": (faculty_row or {}).get("source_url"),
            "chunk_index": 0,
            "text": text_value,
            "char_count": len(text_value),
            "relation": TEXT_CHUNK_RELATIONS["biography"],
            "embedding": None,
            "embedding_model": None,
        }
    ]


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


def sync_faculty_to_neo4j(
    *,
    driver,
    settings_neo4j: Neo4jSettings,
    fac: Faculty,
    limits: SyncLimits,
    include_extracted_text: bool,
    include_embeddings: bool,
    embedding_batch_size: int,
    extracted_text_workers: int = 1,
) -> Dict[str, Any]:
    faculty_row = _faculty_row(fac, max_text_chars=limits.max_text_chars)
    additional_info_rows = _additional_info_rows(
        fac,
        max_additional_info=limits.max_additional_info,
        include_extracted_text=include_extracted_text,
        max_text_chars=limits.max_text_chars,
        extracted_text_workers=extracted_text_workers,
    )
    publication_rows = _publication_rows(
        fac,
        max_publications=limits.max_publications,
        max_text_chars=limits.max_text_chars,
    )
    # Initial GraphRAG sync phase: do not upload keyword nodes/edges yet.
    keyword_rows: List[Dict[str, Any]] = []

    email = faculty_row.get("email")
    if not email:
        raise ValueError(f"Faculty {fac.faculty_id} has no email; cannot sync.")

    text_chunk_rows = _build_additional_info_chunk_rows(
        email=str(email),
        additional_info_rows=additional_info_rows,
        limits=limits,
    )
    text_chunk_rows.extend(
        _build_biography_chunk_rows(
            email=str(email),
            faculty_row=faculty_row,
        )
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

        abs_embeddings = _embed_text_values(
            [str(x.get("abstract") or "") for x in publication_rows if x.get("abstract")],
            batch_size=embedding_batch_size,
        )
        for row in publication_rows:
            key_text = _clean_text(row.get("abstract"))
            emb = abs_embeddings.get(key_text or "")
            if emb:
                row["abstract_embedding"] = emb
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
        MERGE (f:Faculty {email: $row.email})
        SET
            f.faculty_id = $row.faculty_id,
            f.source_url = $row.source_url,
            f.name = $row.name,
            f.phone = $row.phone,
            f.position = $row.position,
            f.organization = $row.organization,
            f.organizations = [x IN $row.organizations WHERE x IS NOT NULL],
            f.address = $row.address,
            f.biography = $row.biography,
            f.degrees = [x IN $row.degrees WHERE x IS NOT NULL],
            f.expertise = [x IN $row.expertise WHERE x IS NOT NULL],
            f.profile_last_refreshed_at = $row.profile_last_refreshed_at,
            f.updated_at = datetime()
        """,
        parameters_={"row": faculty_row},
        database_=settings_neo4j.database,
    )

    # Refresh faculty-owned nodes/edges.
    driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})
        OPTIONAL MATCH (f)-[:HAS_ADDITIONAL_INFO]->(ai:FacultyAdditionalInfo)
        DETACH DELETE ai
        """,
        parameters_={"email": email},
        database_=settings_neo4j.database,
    )
    driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})
        OPTIONAL MATCH (f)-[:AUTHORED]->(p:FacultyPublication)
        DETACH DELETE p
        """,
        parameters_={"email": email},
        database_=settings_neo4j.database,
    )
    driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})
        OPTIONAL MATCH (f)-[r]->(c:FacultyTextChunk)
        WHERE type(r) IN [
            'HAS_BIO_CHUNK',
            'HAS_ADDITIONAL_INFO_CHUNK'
        ]
        DETACH DELETE c
        """,
        parameters_={"email": email},
        database_=settings_neo4j.database,
    )
    if additional_info_rows:
        driver.execute_query(
            """
            MATCH (f:Faculty {email: $email})
            UNWIND $rows AS row
            MERGE (ai:FacultyAdditionalInfo {additional_info_id: row.additional_info_id})
            SET
                ai.faculty_id = row.faculty_id,
                ai.additional_info_url = row.additional_info_url,
                ai.content_path = row.content_path,
                ai.detected_type = row.detected_type,
                ai.content_char_count = row.content_char_count,
                ai.extracted_at = row.extracted_at,
                ai.extract_status = row.extract_status,
                ai.extract_error = row.extract_error,
                ai.extracted_text = row.extracted_text,
                ai.updated_at = datetime()
            MERGE (f)-[:HAS_ADDITIONAL_INFO]->(ai)
            """,
            parameters_={"email": email, "rows": additional_info_rows},
            database_=settings_neo4j.database,
        )

    if publication_rows:
        driver.execute_query(
            """
            MATCH (f:Faculty {email: $email})
            UNWIND $rows AS row
            MERGE (p:FacultyPublication {publication_id: row.publication_id})
            SET
                p.faculty_id = row.faculty_id,
                p.title = row.title,
                p.abstract = row.abstract,
                p.year = row.year,
                p.abstract_embedding = row.abstract_embedding,
                p.embedding_model = row.embedding_model,
                p.updated_at = datetime()
            MERGE (f)-[:AUTHORED]->(p)
            """,
            parameters_={"email": email, "rows": publication_rows},
            database_=settings_neo4j.database,
        )

    for relation in KEYWORD_RELATIONS:
        rows = [item for item in keyword_rows if item.get("relation") == relation]
        if not rows:
            continue

        driver.execute_query(
            f"""
            MATCH (f:Faculty {{email: $email}})
            UNWIND $rows AS row
            MERGE (k:FacultyKeyword {{
                value: row.value,
                section: row.section,
                bucket: row.bucket
            }})
            SET
                k.embedding = row.embedding,
                k.embedding_model = row.embedding_model,
                k.updated_at = datetime()
            MERGE (f)-[r:{relation}]->(k)
            SET
                r.weight = row.weight,
                r.updated_at = datetime()
            """,
            parameters_={"email": email, "rows": rows},
            database_=settings_neo4j.database,
        )

    for relation in TEXT_CHUNK_RELATIONS.values():
        rows = [item for item in text_chunk_rows if item.get("relation") == relation]
        if not rows:
            continue
        driver.execute_query(
            f"""
            MATCH (f:Faculty {{email: $email}})
            UNWIND $rows AS row
            MERGE (c:FacultyTextChunk {{chunk_id: row.chunk_id}})
            SET
                c.email = row.email,
                c.source_type = row.source_type,
                c.source_ref_id = row.source_ref_id,
                c.source_url = row.source_url,
                c.chunk_index = row.chunk_index,
                c.text = row.text,
                c.char_count = row.char_count,
                c.embedding = row.embedding,
                c.embedding_model = row.embedding_model,
                c.updated_at = datetime()
            MERGE (f)-[r:{relation}]->(c)
            SET r.updated_at = datetime()
            """,
            parameters_={"email": email, "rows": rows},
            database_=settings_neo4j.database,
        )

    return {
        "faculty_id": int(faculty_row["faculty_id"]),
        "email": email,
        "counts": {
            "additional_info": len(additional_info_rows),
            "publications": len(publication_rows),
            "keywords": len(keyword_rows),
            "text_chunks": len(text_chunk_rows),
            "embedded_specialization_keywords": len(
                [x for x in keyword_rows if x.get("relation") in SPECIALIZATION_RELATIONS and x.get("embedding")]
            ),
            "embedded_publication_abstracts": len([x for x in publication_rows if x.get("abstract_embedding")]),
            "embedded_text_chunks": len([x for x in text_chunk_rows if x.get("embedding")]),
        },
    }


def verify_faculty_from_neo4j(
    *,
    driver,
    settings_neo4j: Neo4jSettings,
    email: str,
    publication_limit: int,
    additional_info_limit: int,
    chunk_limit: int,
) -> Dict[str, Any]:
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})
        CALL (f) {
            OPTIONAL MATCH (f)-[:HAS_ADDITIONAL_INFO]->(ai:FacultyAdditionalInfo)
            WITH ai ORDER BY ai.extracted_at DESC, ai.additional_info_id DESC
            RETURN [x IN collect(ai)[0..$additional_info_limit] WHERE x IS NOT NULL | x {
                .additional_info_id,
                .additional_info_url,
                .extract_status
            }] AS additional_info
        }
        CALL (f) {
            OPTIONAL MATCH (f)-[:AUTHORED]->(p:FacultyPublication)
            WITH p ORDER BY p.year DESC, p.publication_id DESC
            RETURN [x IN collect(p)[0..$publication_limit] WHERE x IS NOT NULL | {
                publication_id: x.publication_id,
                title: x.title,
                abstract: x.abstract,
                year: x.year,
                has_abstract_embedding: x.abstract_embedding IS NOT NULL
            }] AS publications
        }
        CALL (f) {
            OPTIONAL MATCH (f)-[r]->(k:FacultyKeyword)
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
        CALL (f) {
            OPTIONAL MATCH (f)-[r]->(c:FacultyTextChunk)
            WHERE type(r) IN [
                'HAS_BIO_CHUNK',
                'HAS_ADDITIONAL_INFO_CHUNK'
            ]
            WITH r, c ORDER BY c.source_type ASC, c.source_ref_id ASC, c.chunk_index ASC
            RETURN [x IN collect({relation: type(r), chunk: c})[0..$chunk_limit] WHERE x.chunk IS NOT NULL | {
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
            f {.faculty_id, .email, .name, .position, .organization} AS faculty,
            additional_info,
            publications,
            keywords,
            chunks
        """,
        parameters_={
            "email": str(email or "").strip().lower(),
            "publication_limit": _safe_limit(publication_limit, default=10, minimum=1, maximum=200),
            "additional_info_limit": _safe_limit(additional_info_limit, default=10, minimum=1, maximum=200),
            "chunk_limit": _safe_limit(chunk_limit, default=12, minimum=1, maximum=200),
        },
        database_=settings_neo4j.database,
    )

    if not records:
        return {"email": str(email or "").strip().lower(), "found": False}

    row = records[0]
    additional_info = [item for item in (row.get("additional_info") or []) if item]
    publications = [item for item in (row.get("publications") or []) if item]
    keywords = [item for item in (row.get("keywords") or []) if item and item.get("value")]
    chunks = [item for item in (row.get("chunks") or []) if item]

    return {
        "email": str(email or "").strip().lower(),
        "found": True,
        "faculty": row.get("faculty"),
        "counts": {
            "additional_info": len(additional_info),
            "publications": len(publications),
            "publication_abstracts_embedded": len([x for x in publications if x.get("has_abstract_embedding")]),
            "keywords": len(keywords),
            "specialization_keywords": len([k for k in keywords if k.get("bucket") == "specialization"]),
            "specialization_keywords_embedded": len(
                [k for k in keywords if k.get("bucket") == "specialization" and k.get("has_embedding")]
            ),
            "text_chunks": len(chunks),
            "text_chunks_embedded": len([c for c in chunks if c.get("has_embedding")]),
        },
        "preview": {
            "additional_info": additional_info,
            "publications": publications,
            "keywords": keywords,
            "chunks": chunks,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync Faculty GraphRAG data from Postgres into Neo4j.")
    parser.add_argument("--email", type=str, default="", help="Sync one faculty by email.")
    parser.add_argument("--all", action="store_true", help="Sync all faculty rows.")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows when using --all (0 = no limit).")
    parser.add_argument("--offset", type=int, default=0, help="Offset rows when using --all.")

    parser.add_argument("--max-publications", type=int, default=300, help="Max publication rows per faculty.")
    parser.add_argument("--max-additional-info", type=int, default=150, help="Max additional-info rows per faculty.")
    parser.add_argument("--max-text-chars", type=int, default=4000, help="Max chars for each cleaned text block.")
    parser.add_argument("--chunk-size-chars", type=int, default=1200, help="Chunk size for additional-info text embeddings.")
    parser.add_argument("--chunk-overlap-chars", type=int, default=150, help="Chunk overlap for additional-info text embeddings.")
    parser.add_argument("--max-chunks-per-source", type=int, default=24, help="Max chunks for each additional-info document.")
    parser.add_argument("--embedding-batch-size", type=int, default=12, help="Embedding batch size.")
    parser.add_argument("--s3-read-workers", type=int, default=1, help="Parallel workers for loading extracted S3 text.")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip specialization/publication/chunk embeddings.")
    parser.add_argument(
        "--skip-extracted-text",
        action="store_true",
        help="Do not load extracted S3 text for additional info.",
    )

    parser.add_argument("--verify-email", type=str, default="", help="Run verify query for this email after sync.")
    parser.add_argument("--verify-publication-limit", type=int, default=10, help="Publication rows in verify response.")
    parser.add_argument("--verify-additional-info-limit", type=int, default=10, help="Additional info rows in verify response.")
    parser.add_argument("--verify-chunk-limit", type=int, default=12, help="Chunk rows in verify response.")
    parser.add_argument("--json-only", action="store_true", help="Print only JSON output.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on first faculty sync error.")

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

    rows = _load_faculties(
        email=args.email,
        sync_all=bool(args.all),
        limit=max(0, int(args.limit or 0)),
        offset=max(0, int(args.offset or 0)),
    )
    if not rows:
        raise RuntimeError("No faculty rows found for requested sync scope.")

    limits = SyncLimits(
        max_publications=_safe_limit(args.max_publications, default=300, minimum=1, maximum=3000),
        max_additional_info=_safe_limit(args.max_additional_info, default=150, minimum=1, maximum=2000),
        max_text_chars=_safe_limit(args.max_text_chars, default=4000, minimum=100, maximum=50000),
        chunk_size_chars=_safe_limit(args.chunk_size_chars, default=1200, minimum=200, maximum=5000),
        chunk_overlap_chars=_safe_limit(args.chunk_overlap_chars, default=150, minimum=0, maximum=2500),
        max_chunks_per_source=_safe_limit(args.max_chunks_per_source, default=24, minimum=1, maximum=1000),
    )

    include_extracted_text = not bool(args.skip_extracted_text)
    include_embeddings = not bool(args.skip_embeddings)
    embedding_batch_size = _safe_limit(args.embedding_batch_size, default=12, minimum=1, maximum=128)
    extracted_text_workers = _safe_limit(args.s3_read_workers, default=1, minimum=1, maximum=64)

    synced: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with GraphDatabase.driver(
        settings_neo4j.uri,
        auth=(settings_neo4j.username, settings_neo4j.password),
    ) as driver:
        driver.verify_connectivity()

        for fac in rows:
            email = str(getattr(fac, "email", "") or "").strip().lower()
            try:
                result = sync_faculty_to_neo4j(
                    driver=driver,
                    settings_neo4j=settings_neo4j,
                    fac=fac,
                    limits=limits,
                    include_extracted_text=include_extracted_text,
                    include_embeddings=include_embeddings,
                    embedding_batch_size=embedding_batch_size,
                    extracted_text_workers=extracted_text_workers,
                )
                synced.append(result)
            except Exception as exc:
                errors.append(
                    {
                        "faculty_id": int(getattr(fac, "faculty_id", 0) or 0),
                        "email": email,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if args.stop_on_error:
                    break

        verify_email = str(args.verify_email or "").strip().lower()
        if not verify_email and len(synced) == 1:
            verify_email = str(synced[0].get("email") or "").strip().lower()

        verify = None
        if verify_email:
            verify = verify_faculty_from_neo4j(
                driver=driver,
                settings_neo4j=settings_neo4j,
                email=verify_email,
                publication_limit=args.verify_publication_limit,
                additional_info_limit=args.verify_additional_info_limit,
                chunk_limit=args.verify_chunk_limit,
            )

    totals = {
        "faculties_synced": len(synced),
        "faculties_failed": len(errors),
        "additional_info": sum(int(item.get("counts", {}).get("additional_info", 0)) for item in synced),
        "publications": sum(int(item.get("counts", {}).get("publications", 0)) for item in synced),
        "keywords": sum(int(item.get("counts", {}).get("keywords", 0)) for item in synced),
        "text_chunks": sum(int(item.get("counts", {}).get("text_chunks", 0)) for item in synced),
        "embedded_specialization_keywords": sum(
            int(item.get("counts", {}).get("embedded_specialization_keywords", 0)) for item in synced
        ),
        "embedded_publication_abstracts": sum(
            int(item.get("counts", {}).get("embedded_publication_abstracts", 0)) for item in synced
        ),
        "embedded_text_chunks": sum(int(item.get("counts", {}).get("embedded_text_chunks", 0)) for item in synced),
    }

    payload = {
        "scope": {
            "email": str(args.email or "").strip().lower(),
            "all": bool(args.all),
            "limit": max(0, int(args.limit or 0)),
            "offset": max(0, int(args.offset or 0)),
            "include_extracted_text": include_extracted_text,
            "include_embeddings": include_embeddings,
            "extracted_text_workers": extracted_text_workers,
            "embedding_model": (settings.bedrock_embed_model_id or "").strip() if include_embeddings else None,
        },
        "totals": totals,
        "synced": synced,
        "errors": errors,
        "verify": verify,
    }

    if not args.json_only:
        print("Faculty GraphRAG sync complete.")
        print(f"  synced faculties               : {totals['faculties_synced']}")
        print(f"  failed faculties               : {totals['faculties_failed']}")
        print(f"  additional info                : {totals['additional_info']}")
        print(f"  publications                   : {totals['publications']}")
        print(f"  keyword edges                  : {totals['keywords']}")
        print(f"  text chunks                    : {totals['text_chunks']}")
        print(f"  embedded specialization kws    : {totals['embedded_specialization_keywords']}")
        print(f"  embedded publication abstracts : {totals['embedded_publication_abstracts']}")
        print(f"  embedded text chunks           : {totals['embedded_text_chunks']}")
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
