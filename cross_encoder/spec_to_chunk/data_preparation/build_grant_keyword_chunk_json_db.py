from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import bindparam, text

# Ensure project root on sys.path for direct script execution.
def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "cross_encoder").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.db_conn import SessionLocal
from db.models.opportunity import OpportunityAdditionalInfo, OpportunityAttachment
from utils.content_extractor import load_extracted_content


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_path(value: Any) -> Path:
    path = Path(_clean_text(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _safe_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _safe_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _truncate(value: str, *, max_chars: int) -> str:
    text = _clean_text(value)
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _coerce_keywords_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            obj = json.loads(value)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {}
    return {}


def _extract_specs_from_keywords(keywords: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for section in ("research", "application"):
        sec = keywords.get(section)
        if not isinstance(sec, dict):
            continue
        specs = sec.get("specialization")
        if not isinstance(specs, list):
            continue
        for item in specs:
            spec_text = ""
            spec_weight: Optional[float] = None
            if isinstance(item, dict):
                spec_text = _clean_text(item.get("t") or item.get("text"))
                try:
                    if item.get("w") is not None:
                        spec_weight = float(item.get("w"))
                    elif item.get("weight") is not None:
                        spec_weight = float(item.get("weight"))
                except Exception:
                    spec_weight = None
            else:
                spec_text = _clean_text(item)
            if not spec_text:
                continue
            out.append(
                {
                    "section": section,
                    "text": spec_text,
                    "weight": spec_weight,
                    "model": None,
                    "source": "opportunity_keywords",
                }
            )
    return out


def _load_base_grants(*, limit_grants: int) -> List[Dict[str, Any]]:
    limit = int(max(0, limit_grants))
    sql = text(
        """
        SELECT
            o.opportunity_id AS opportunity_id,
            COALESCE(o.opportunity_title, '') AS opportunity_title,
            COALESCE(o.agency_name, '') AS agency_name,
            COALESCE(o.summary_description, '') AS summary_description,
            ok.keywords AS keywords_raw
        FROM opportunity o
        LEFT JOIN opportunity_keywords ok
            ON ok.opportunity_id = o.opportunity_id
        WHERE
            EXISTS (
                SELECT 1
                FROM opportunity_specialization_embedding ose
                WHERE ose.opportunity_id = o.opportunity_id
            )
            OR EXISTS (
                SELECT 1
                FROM opportunity_additional_info oai
                WHERE oai.opportunity_id = o.opportunity_id
            )
            OR EXISTS (
                SELECT 1
                FROM opportunity_attachment oat
                WHERE oat.opportunity_id = o.opportunity_id
            )
        ORDER BY o.opportunity_id ASC
        """
    )

    with SessionLocal() as sess:
        rows = sess.execute(sql).mappings().all()

    out = [dict(r or {}) for r in rows]
    if limit > 0:
        out = out[:limit]
    return out


def _load_embedded_specializations(
    *,
    opportunity_ids: List[str],
    min_spec_weight: float,
) -> Dict[str, List[Dict[str, Any]]]:
    if not opportunity_ids:
        return {}

    sql = text(
        """
        SELECT
            ose.opportunity_id AS opportunity_id,
            COALESCE(ose.section, '') AS section,
            COALESCE(ose.spec_text, '') AS spec_text,
            COALESCE(ose.spec_weight, 1.0) AS spec_weight,
            COALESCE(ose.model, '') AS model
        FROM opportunity_specialization_embedding ose
        WHERE ose.opportunity_id IN :opportunity_ids
          AND COALESCE(ose.spec_text, '') <> ''
          AND COALESCE(ose.spec_weight, 1.0) >= :min_spec_weight
        ORDER BY
            ose.opportunity_id ASC,
            ose.section ASC,
            COALESCE(ose.spec_weight, 1.0) DESC,
            ose.id ASC
        """
    ).bindparams(bindparam("opportunity_ids", expanding=True))

    out: Dict[str, List[Dict[str, Any]]] = {}
    with SessionLocal() as sess:
        rows = sess.execute(
            sql,
            {
                "opportunity_ids": list(opportunity_ids),
                "min_spec_weight": float(max(0.0, min_spec_weight)),
            },
        ).mappings().all()

    for row in rows:
        item = dict(row or {})
        opp_id = _clean_text(item.get("opportunity_id"))
        if not opp_id:
            continue
        text_value = _clean_text(item.get("spec_text"))
        if not text_value:
            continue
        out.setdefault(opp_id, []).append(
            {
                "section": _clean_text(item.get("section")) or "unknown",
                "text": text_value,
                "weight": float(item.get("spec_weight") if item.get("spec_weight") is not None else 1.0),
                "model": _clean_text(item.get("model")) or None,
                "source": "opportunity_specialization_embedding",
            }
        )
    return out


def _load_chunk_rows(*, opportunity_ids: List[str]) -> Dict[str, Dict[str, List[Any]]]:
    out: Dict[str, Dict[str, List[Any]]] = {}
    if not opportunity_ids:
        return out

    with SessionLocal() as sess:
        additional_rows = (
            sess.query(OpportunityAdditionalInfo)
            .filter(OpportunityAdditionalInfo.opportunity_id.in_(opportunity_ids))
            .order_by(
                OpportunityAdditionalInfo.opportunity_id.asc(),
                OpportunityAdditionalInfo.chunk_index.asc(),
                OpportunityAdditionalInfo.id.asc(),
            )
            .all()
        )
        attachment_rows = (
            sess.query(OpportunityAttachment)
            .filter(OpportunityAttachment.opportunity_id.in_(opportunity_ids))
            .order_by(
                OpportunityAttachment.opportunity_id.asc(),
                OpportunityAttachment.chunk_index.asc(),
                OpportunityAttachment.id.asc(),
            )
            .all()
        )

    for row in additional_rows:
        opp_id = _clean_text(getattr(row, "opportunity_id", ""))
        if not opp_id:
            continue
        bucket = out.setdefault(opp_id, {"additional": [], "attachment": []})
        bucket["additional"].append(row)

    for row in attachment_rows:
        opp_id = _clean_text(getattr(row, "opportunity_id", ""))
        if not opp_id:
            continue
        bucket = out.setdefault(opp_id, {"additional": [], "attachment": []})
        bucket["attachment"].append(row)

    return out


def _build_chunk_payload(
    *,
    additional_rows: List[Any],
    attachment_rows: List[Any],
    include_content: bool,
    max_chars_per_chunk: int,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    chunk_items: List[Dict[str, Any]] = []
    additional_text_by_row_id: Dict[int, str] = {}
    attachment_text_by_row_id: Dict[int, str] = {}

    if include_content:
        try:
            additional_loaded = load_extracted_content(
                additional_rows,
                url_attr="additional_info_url",
                group_chunks=False,
                include_row_meta=True,
            )
            for item in additional_loaded:
                row_id = _safe_int(item.get("row_id"), default=0, minimum=0, maximum=2_147_483_647)
                if row_id <= 0:
                    continue
                additional_text_by_row_id[row_id] = _truncate(
                    str(item.get("content") or ""),
                    max_chars=max_chars_per_chunk,
                )
        except Exception as e:
            warnings.append(f"additional_content_load_failed: {type(e).__name__}: {e}")

        try:
            attachment_loaded = load_extracted_content(
                attachment_rows,
                url_attr="file_download_path",
                title_attr="file_name",
                group_chunks=False,
                include_row_meta=True,
            )
            for item in attachment_loaded:
                row_id = _safe_int(item.get("row_id"), default=0, minimum=0, maximum=2_147_483_647)
                if row_id <= 0:
                    continue
                attachment_text_by_row_id[row_id] = _truncate(
                    str(item.get("content") or ""),
                    max_chars=max_chars_per_chunk,
                )
        except Exception as e:
            warnings.append(f"attachment_content_load_failed: {type(e).__name__}: {e}")

    for row in additional_rows:
        row_id = _safe_int(getattr(row, "id", 0), default=0, minimum=0, maximum=2_147_483_647)
        chunk_items.append(
            {
                "source_type": "additional_info",
                "row_id": row_id,
                "source_url": _clean_text(getattr(row, "additional_info_url", "")),
                "file_name": None,
                "chunk_index": _safe_int(getattr(row, "chunk_index", 0), default=0, minimum=0, maximum=1_000_000),
                "content_path": _clean_text(getattr(row, "content_path", "")),
                "detected_type": _clean_text(getattr(row, "detected_type", "")) or None,
                "content_char_count": _safe_int(
                    getattr(row, "content_char_count", 0),
                    default=0,
                    minimum=0,
                    maximum=1_000_000_000,
                ),
                "extract_status": _clean_text(getattr(row, "extract_status", "")) or None,
                "extract_error": _clean_text(getattr(row, "extract_error", "")) or None,
                "content": additional_text_by_row_id.get(row_id, "") if include_content else "",
            }
        )

    for row in attachment_rows:
        row_id = _safe_int(getattr(row, "id", 0), default=0, minimum=0, maximum=2_147_483_647)
        chunk_items.append(
            {
                "source_type": "attachment",
                "row_id": row_id,
                "source_url": _clean_text(getattr(row, "file_download_path", "")),
                "file_name": _clean_text(getattr(row, "file_name", "")) or None,
                "chunk_index": _safe_int(getattr(row, "chunk_index", 0), default=0, minimum=0, maximum=1_000_000),
                "content_path": _clean_text(getattr(row, "content_path", "")),
                "detected_type": _clean_text(getattr(row, "detected_type", "")) or None,
                "content_char_count": _safe_int(
                    getattr(row, "content_char_count", 0),
                    default=0,
                    minimum=0,
                    maximum=1_000_000_000,
                ),
                "extract_status": _clean_text(getattr(row, "extract_status", "")) or None,
                "extract_error": _clean_text(getattr(row, "extract_error", "")) or None,
                "content": attachment_text_by_row_id.get(row_id, "") if include_content else "",
            }
        )

    chunk_items.sort(
        key=lambda x: (
            str(x.get("source_type") or ""),
            str(x.get("source_url") or ""),
            int(x.get("chunk_index") or 0),
            int(x.get("row_id") or 0),
        )
    )
    return chunk_items


def build_json_db(
    *,
    output_path: Path,
    limit_grants: int,
    min_spec_weight: float,
    include_content: bool,
    max_chars_per_chunk: int,
    indent: int,
) -> Dict[str, Any]:
    warnings: List[str] = []
    base_grants = _load_base_grants(limit_grants=limit_grants)
    grant_ids = [_clean_text(x.get("opportunity_id")) for x in base_grants if _clean_text(x.get("opportunity_id"))]

    embedded_specs_by_grant: Dict[str, List[Dict[str, Any]]] = {}
    try:
        embedded_specs_by_grant = _load_embedded_specializations(
            opportunity_ids=grant_ids,
            min_spec_weight=min_spec_weight,
        )
    except Exception as e:
        warnings.append(f"embedded_specialization_load_failed: {type(e).__name__}: {e}")

    chunk_rows_by_grant = _load_chunk_rows(opportunity_ids=grant_ids)

    grants_out: List[Dict[str, Any]] = []
    total_specs = 0
    total_chunks = 0
    total_chunks_with_text = 0

    for row in base_grants:
        grant_id = _clean_text(row.get("opportunity_id"))
        if not grant_id:
            continue

        keywords_raw = _coerce_keywords_json(row.get("keywords_raw"))
        embedded_specs = list(embedded_specs_by_grant.get(grant_id, []))
        fallback_specs = _extract_specs_from_keywords(keywords_raw)
        specs = embedded_specs if embedded_specs else fallback_specs

        chunk_bucket = chunk_rows_by_grant.get(grant_id, {"additional": [], "attachment": []})
        additional_rows = list(chunk_bucket.get("additional", []) or [])
        attachment_rows = list(chunk_bucket.get("attachment", []) or [])
        chunk_payload = _build_chunk_payload(
            additional_rows=additional_rows,
            attachment_rows=attachment_rows,
            include_content=include_content,
            max_chars_per_chunk=max_chars_per_chunk,
            warnings=warnings,
        )

        total_specs += len(specs)
        total_chunks += len(chunk_payload)
        total_chunks_with_text += sum(1 for c in chunk_payload if _clean_text(c.get("content")))

        grants_out.append(
            {
                "opportunity_id": grant_id,
                "opportunity_title": _clean_text(row.get("opportunity_title")),
                "agency_name": _clean_text(row.get("agency_name")),
                "summary_description": _clean_text(row.get("summary_description")),
                "specializations": specs,
                "keywords_raw": keywords_raw,
                "chunks": chunk_payload,
            }
        )

    payload = {
        "meta": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "postgresql(+optional_s3_content)",
            "grant_count": len(grants_out),
            "specialization_count": int(total_specs),
            "chunk_count": int(total_chunks),
            "chunks_with_text_count": int(total_chunks_with_text),
            "include_content": bool(include_content),
            "max_chars_per_chunk": int(max_chars_per_chunk),
            "min_spec_weight": float(min_spec_weight),
            "limit_grants": int(limit_grants),
            "warnings": warnings,
        },
        "grants": grants_out,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=max(0, int(indent))),
        encoding="utf-8",
    )
    return payload


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export grant specialization keywords + grant chunk rows from Postgres into a single JSON file "
            "that can be used as a lightweight local DB snapshot."
        )
    )
    p.add_argument(
        "--output",
        type=str,
        default="cross_encoder/spec_to_chunk/dataset/grant_keyword_chunk_db.json",
        help="Output JSON file path.",
    )
    p.add_argument(
        "--limit-grants",
        type=int,
        default=0,
        help="Max number of grants to export (0 = all matched grants).",
    )
    p.add_argument(
        "--min-spec-weight",
        type=float,
        default=0.0,
        help="Minimum specialization weight for opportunity_specialization_embedding rows.",
    )
    p.add_argument(
        "--no-content",
        action="store_true",
        help="Do not read chunk text bodies from S3; export chunk metadata only.",
    )
    p.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=6000,
        help="Trim loaded chunk text to this many chars per chunk (0 = no trim).",
    )
    p.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation spaces.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    output_path = _resolve_path(args.output)
    limit_grants = _safe_int(args.limit_grants, default=0, minimum=0, maximum=5_000_000)
    min_spec_weight = _safe_float(args.min_spec_weight, default=0.0, minimum=0.0, maximum=1.0)
    include_content = not bool(args.no_content)
    max_chars_per_chunk = _safe_int(args.max_chars_per_chunk, default=6000, minimum=0, maximum=500_000)
    indent = _safe_int(args.indent, default=2, minimum=0, maximum=16)

    payload = build_json_db(
        output_path=output_path,
        limit_grants=limit_grants,
        min_spec_weight=min_spec_weight,
        include_content=include_content,
        max_chars_per_chunk=max_chars_per_chunk,
        indent=indent,
    )

    meta = dict(payload.get("meta") or {})
    print(f"output={output_path}")
    print(f"grant_count={meta.get('grant_count')}")
    print(f"specialization_count={meta.get('specialization_count')}")
    print(f"chunk_count={meta.get('chunk_count')}")
    print(f"chunks_with_text_count={meta.get('chunks_with_text_count')}")
    warning_count = len(list(meta.get("warnings") or []))
    print(f"warning_count={warning_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
