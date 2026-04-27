from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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
from db.models.faculty import FacultyAdditionalInfo, FacultyPublication
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


def _truncate(text_value: str, *, max_chars: int) -> str:
    text_clean = _clean_text(text_value)
    if max_chars <= 0:
        return text_clean
    if len(text_clean) <= max_chars:
        return text_clean
    return text_clean[:max_chars]


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in list(items or []):
        value = _clean_text(item)
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _coerce_keywords_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _extract_keyword_texts(raw_items: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(raw_items, list):
        return out

    for item in raw_items:
        if isinstance(item, dict):
            text_value = _clean_text(item.get("t") or item.get("text"))
        else:
            text_value = _clean_text(item)
        if text_value:
            out.append(text_value)
    return out


def _extract_domain_keywords_from_keywords_json(keywords: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for section in ("research", "application"):
        section_obj = keywords.get(section)
        if not isinstance(section_obj, dict):
            continue
        out.extend(_extract_keyword_texts(section_obj.get("domain")))
    return _dedupe_keep_order(out)


def _extract_spec_keywords_from_keywords_json(keywords: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for section in ("research", "application"):
        section_obj = keywords.get(section)
        if not isinstance(section_obj, dict):
            continue
        out.extend(_extract_keyword_texts(section_obj.get("specialization")))
    return _dedupe_keep_order(out)


def _load_grant_base_rows(*, limit_grants: int) -> List[Dict[str, Any]]:
    limit = int(max(0, limit_grants))
    sql_body = """
        SELECT
            o.opportunity_id AS grant_id,
            ok.keywords AS keywords_raw
        FROM opportunity o
        LEFT JOIN opportunity_keywords ok
            ON ok.opportunity_id = o.opportunity_id
        WHERE
            EXISTS (
                SELECT 1
                FROM opportunity_keywords ok2
                WHERE ok2.opportunity_id = o.opportunity_id
            )
            OR EXISTS (
                SELECT 1
                FROM opportunity_specialization_embedding ose
                WHERE ose.opportunity_id = o.opportunity_id
            )
        ORDER BY o.opportunity_id ASC
    """
    if limit > 0:
        sql_body += "\nLIMIT :limit_grants"

    sql = text(sql_body)
    params: Dict[str, Any] = {}
    if limit > 0:
        params["limit_grants"] = limit

    with SessionLocal() as sess:
        rows = sess.execute(sql, params).mappings().all()
    return [dict(r or {}) for r in rows]


def _load_grant_spec_keywords_from_embedding(
    *,
    grant_ids: List[str],
    min_spec_weight: float,
) -> Dict[str, List[str]]:
    if not grant_ids:
        return {}

    sql = text(
        """
        SELECT
            ose.opportunity_id AS grant_id,
            COALESCE(ose.spec_text, '') AS spec_text
        FROM opportunity_specialization_embedding ose
        WHERE ose.opportunity_id IN :grant_ids
          AND COALESCE(ose.spec_text, '') <> ''
          AND COALESCE(ose.spec_weight, 1.0) >= :min_spec_weight
        ORDER BY
            ose.opportunity_id ASC,
            COALESCE(ose.spec_weight, 1.0) DESC,
            ose.id ASC
        """
    ).bindparams(bindparam("grant_ids", expanding=True))

    with SessionLocal() as sess:
        rows = sess.execute(
            sql,
            {
                "grant_ids": list(grant_ids),
                "min_spec_weight": float(max(0.0, min_spec_weight)),
            },
        ).mappings().all()

    out: Dict[str, List[str]] = {}
    for row in rows:
        grant_id = _clean_text(row.get("grant_id"))
        spec_text = _clean_text(row.get("spec_text"))
        if not grant_id or not spec_text:
            continue
        out.setdefault(grant_id, []).append(spec_text)

    for grant_id, spec_list in list(out.items()):
        out[grant_id] = _dedupe_keep_order(spec_list)
    return out


def _build_grant_keyword_records(
    *,
    limit_grants: int,
    min_spec_weight: float,
) -> Dict[str, Any]:
    base_rows = _load_grant_base_rows(limit_grants=limit_grants)
    grant_ids = [_clean_text(r.get("grant_id")) for r in base_rows if _clean_text(r.get("grant_id"))]
    spec_by_grant = _load_grant_spec_keywords_from_embedding(
        grant_ids=grant_ids,
        min_spec_weight=min_spec_weight,
    )

    records: List[Dict[str, Any]] = []
    total_keywords = 0
    total_spec_keywords = 0

    for row in base_rows:
        grant_id = _clean_text(row.get("grant_id"))
        if not grant_id:
            continue
        keywords_json = _coerce_keywords_json(row.get("keywords_raw"))
        grant_keywords = _extract_domain_keywords_from_keywords_json(keywords_json)
        fallback_spec_keywords = _extract_spec_keywords_from_keywords_json(keywords_json)
        grant_spec_keywords = list(spec_by_grant.get(grant_id) or fallback_spec_keywords)

        total_keywords += len(grant_keywords)
        total_spec_keywords += len(grant_spec_keywords)
        records.append(
            {
                "grant_id": grant_id,
                "grant_keywords": grant_keywords,
                "grant_spec_keywords": grant_spec_keywords,
            }
        )

    return {
        "grants": records,
        "meta": {
            "grant_count": len(records),
            "grant_keyword_count": int(total_keywords),
            "grant_spec_keyword_count": int(total_spec_keywords),
            "min_spec_weight": float(min_spec_weight),
            "limit_grants": int(limit_grants),
        },
    }


def _load_faculty_ids(*, limit_faculties: int) -> List[int]:
    sql_body = """
        SELECT f.faculty_id AS faculty_id
        FROM faculty f
        ORDER BY f.faculty_id ASC
    """
    limit = int(max(0, limit_faculties))
    params: Dict[str, Any] = {}
    if limit > 0:
        sql_body += "\nLIMIT :limit_faculties"
        params["limit_faculties"] = limit

    sql = text(sql_body)
    with SessionLocal() as sess:
        rows = sess.execute(sql, params).mappings().all()

    out: List[int] = []
    for row in rows:
        fac_id = _safe_int(row.get("faculty_id"), default=0, minimum=0, maximum=2_147_483_647)
        if fac_id > 0:
            out.append(fac_id)
    return out


def _load_faculty_chunk_rows_for_ids(*, faculty_ids: List[int]) -> List[Any]:
    if not faculty_ids:
        return []
    with SessionLocal() as sess:
        rows = (
            sess.query(FacultyAdditionalInfo)
            .filter(FacultyAdditionalInfo.faculty_id.in_(faculty_ids))
            .order_by(
                FacultyAdditionalInfo.faculty_id.asc(),
                FacultyAdditionalInfo.chunk_index.asc(),
                FacultyAdditionalInfo.id.asc(),
            )
            .all()
        )
    return list(rows or [])


def _load_faculty_publication_rows_for_ids(*, faculty_ids: List[int]) -> List[Any]:
    if not faculty_ids:
        return []
    with SessionLocal() as sess:
        rows = (
            sess.query(FacultyPublication)
            .filter(FacultyPublication.faculty_id.in_(faculty_ids))
            .filter(FacultyPublication.abstract.isnot(None))
            .order_by(
                FacultyPublication.faculty_id.asc(),
                FacultyPublication.year.asc().nulls_last(),
                FacultyPublication.id.asc(),
            )
            .all()
        )
    return list(rows or [])


def _build_faculty_chunk_records(
    *,
    limit_faculties: int,
    batch_faculty_size: int,
    max_chars_per_chunk: int,
) -> Dict[str, Any]:
    faculty_ids = _load_faculty_ids(limit_faculties=limit_faculties)
    safe_batch_size = _safe_int(batch_faculty_size, default=200, minimum=1, maximum=10_000)
    out_records: List[Dict[str, Any]] = []
    warnings: List[str] = []
    additional_info_count = 0
    publication_abstract_count = 0

    for i in range(0, len(faculty_ids), safe_batch_size):
        batch_ids = faculty_ids[i : i + safe_batch_size]
        additional_rows = _load_faculty_chunk_rows_for_ids(faculty_ids=batch_ids)
        publication_rows = _load_faculty_publication_rows_for_ids(faculty_ids=batch_ids)
        if (not additional_rows) and (not publication_rows):
            continue

        content_by_row_id: Dict[int, str] = {}
        if additional_rows:
            try:
                loaded = load_extracted_content(
                    additional_rows,
                    url_attr="additional_info_url",
                    group_chunks=False,
                    include_row_meta=True,
                )
                for item in list(loaded or []):
                    row_id = _safe_int(item.get("row_id"), default=0, minimum=0, maximum=2_147_483_647)
                    if row_id <= 0:
                        continue
                    content_by_row_id[row_id] = _truncate(
                        _clean_text(item.get("content")),
                        max_chars=max_chars_per_chunk,
                    )
            except Exception as e:
                warnings.append(f"faculty_content_load_failed_batch_start_{i}: {type(e).__name__}: {e}")

        for row in additional_rows:
            chunk_id = _safe_int(getattr(row, "id", 0), default=0, minimum=0, maximum=2_147_483_647)
            chunk_text = _clean_text(content_by_row_id.get(chunk_id))
            if not chunk_text:
                continue

            out_records.append(
                {
                    "fac_id": _safe_int(getattr(row, "faculty_id", 0), default=0, minimum=0, maximum=2_147_483_647),
                    "source_type": "additional_info",
                    "chunk_id": chunk_id,
                    "chunk_index": _safe_int(
                        getattr(row, "chunk_index", 0),
                        default=0,
                        minimum=0,
                        maximum=1_000_000,
                    ),
                    "text": chunk_text,
                }
            )
            additional_info_count += 1

        for row in publication_rows:
            pub_id = _safe_int(getattr(row, "id", 0), default=0, minimum=0, maximum=2_147_483_647)
            abstract_text = _truncate(_clean_text(getattr(row, "abstract", "")), max_chars=max_chars_per_chunk)
            if not abstract_text:
                continue
            out_records.append(
                {
                    "fac_id": _safe_int(getattr(row, "faculty_id", 0), default=0, minimum=0, maximum=2_147_483_647),
                    "source_type": "publication_abstract",
                    "chunk_id": pub_id,
                    "chunk_index": 0,
                    "text": abstract_text,
                }
            )
            publication_abstract_count += 1

    out_records.sort(
        key=lambda x: (
            int(x.get("fac_id") or 0),
            str(x.get("source_type") or ""),
            int(x.get("chunk_index") or 0),
            int(x.get("chunk_id") or 0),
        )
    )

    return {
        "fac_chunks": out_records,
        "meta": {
            "faculty_count_scanned": len(faculty_ids),
            "fac_chunk_count": len(out_records),
            "additional_info_chunk_count": int(additional_info_count),
            "publication_abstract_count": int(publication_abstract_count),
            "limit_faculties": int(limit_faculties),
            "batch_faculty_size": int(safe_batch_size),
            "max_chars_per_chunk": int(max_chars_per_chunk),
            "warnings": warnings,
        },
    }


def build_exports(
    *,
    grant_output: Path,
    fac_output: Path,
    limit_grants: int,
    limit_faculties: int,
    min_spec_weight: float,
    batch_faculty_size: int,
    max_chars_per_chunk: int,
    indent: int,
) -> Dict[str, Any]:
    grant_payload = _build_grant_keyword_records(
        limit_grants=limit_grants,
        min_spec_weight=min_spec_weight,
    )
    fac_payload = _build_faculty_chunk_records(
        limit_faculties=limit_faculties,
        batch_faculty_size=batch_faculty_size,
        max_chars_per_chunk=max_chars_per_chunk,
    )

    created_at_utc = datetime.now(timezone.utc).isoformat()
    grant_payload["meta"]["created_at_utc"] = created_at_utc
    grant_payload["meta"]["source"] = "postgresql"
    fac_payload["meta"]["created_at_utc"] = created_at_utc
    fac_payload["meta"]["source"] = "postgresql(+s3_content)"

    grant_output.parent.mkdir(parents=True, exist_ok=True)
    fac_output.parent.mkdir(parents=True, exist_ok=True)
    grant_output.write_text(
        json.dumps(grant_payload, ensure_ascii=False, indent=max(0, int(indent))),
        encoding="utf-8",
    )
    fac_output.write_text(
        json.dumps(fac_payload, ensure_ascii=False, indent=max(0, int(indent))),
        encoding="utf-8",
    )
    return {"grant_payload": grant_payload, "fac_payload": fac_payload}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export two local JSON DB snapshots: "
            "(1) grant keywords/spec keywords, (2) faculty additional-info chunks + publication abstracts."
        )
    )
    p.add_argument(
        "--grant-output",
        type=str,
        default="cross_encoder/spec_to_chunk/dataset/grant_keywords_spec_keywords_db.json",
        help="Output JSON path for grant keywords/spec keywords.",
    )
    p.add_argument(
        "--fac-output",
        type=str,
        default="cross_encoder/spec_to_chunk/dataset/fac_chunks_db.json",
        help="Output JSON path for faculty chunks (additional-info + publication abstracts).",
    )
    p.add_argument(
        "--limit-grants",
        type=int,
        default=0,
        help="Max number of grants to export (0 = all matched grants).",
    )
    p.add_argument(
        "--limit-faculties",
        type=int,
        default=0,
        help="Max number of faculties to scan (0 = all faculties).",
    )
    p.add_argument(
        "--min-spec-weight",
        type=float,
        default=0.0,
        help="Minimum specialization weight when reading opportunity_specialization_embedding.",
    )
    p.add_argument(
        "--batch-faculty-size",
        type=int,
        default=200,
        help="How many faculty IDs to process per content-loading batch.",
    )
    p.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=6000,
        help="Trim chunk text to this many characters (0 = no trim).",
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
    grant_output = _resolve_path(args.grant_output)
    fac_output = _resolve_path(args.fac_output)
    limit_grants = _safe_int(args.limit_grants, default=0, minimum=0, maximum=5_000_000)
    limit_faculties = _safe_int(args.limit_faculties, default=0, minimum=0, maximum=5_000_000)
    min_spec_weight = _safe_float(args.min_spec_weight, default=0.0, minimum=0.0, maximum=1.0)
    batch_faculty_size = _safe_int(args.batch_faculty_size, default=200, minimum=1, maximum=10_000)
    max_chars_per_chunk = _safe_int(args.max_chars_per_chunk, default=6000, minimum=0, maximum=500_000)
    indent = _safe_int(args.indent, default=2, minimum=0, maximum=16)

    result = build_exports(
        grant_output=grant_output,
        fac_output=fac_output,
        limit_grants=limit_grants,
        limit_faculties=limit_faculties,
        min_spec_weight=min_spec_weight,
        batch_faculty_size=batch_faculty_size,
        max_chars_per_chunk=max_chars_per_chunk,
        indent=indent,
    )

    grant_meta = dict(result["grant_payload"].get("meta") or {})
    fac_meta = dict(result["fac_payload"].get("meta") or {})
    print(f"grant_output={grant_output}")
    print(f"grant_count={grant_meta.get('grant_count')}")
    print(f"grant_keyword_count={grant_meta.get('grant_keyword_count')}")
    print(f"grant_spec_keyword_count={grant_meta.get('grant_spec_keyword_count')}")
    print(f"fac_output={fac_output}")
    print(f"faculty_count_scanned={fac_meta.get('faculty_count_scanned')}")
    print(f"fac_chunk_count={fac_meta.get('fac_chunk_count')}")
    print(f"additional_info_chunk_count={fac_meta.get('additional_info_chunk_count')}")
    print(f"publication_abstract_count={fac_meta.get('publication_abstract_count')}")
    print(f"warning_count={len(list(fac_meta.get('warnings') or []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
