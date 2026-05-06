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

GRANT_OUTPUT_DEFAULT = "cross_encoder/dataset/source/grant_keywords_spec_keywords_db.json"
FAC_OUTPUT_DEFAULT = "cross_encoder/dataset/source/fac_specs_db.json"


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
    model_id: str,
) -> Dict[str, List[str]]:
    if not grant_ids:
        return {}

    sql_body = """
        SELECT
            ose.opportunity_id AS grant_id,
            COALESCE(ose.spec_text, '') AS spec_text
        FROM opportunity_specialization_embedding ose
        WHERE ose.opportunity_id IN :grant_ids
          AND COALESCE(ose.spec_text, '') <> ''
          AND COALESCE(ose.spec_weight, 1.0) >= :min_spec_weight
    """
    if _clean_text(model_id):
        sql_body += "\n  AND COALESCE(ose.model, '') = :model_id"
    sql_body += """
        ORDER BY
            ose.opportunity_id ASC,
            COALESCE(ose.spec_weight, 1.0) DESC,
            ose.id ASC
    """

    sql = text(sql_body).bindparams(bindparam("grant_ids", expanding=True))
    params: Dict[str, Any] = {
        "grant_ids": list(grant_ids),
        "min_spec_weight": float(max(0.0, min_spec_weight)),
    }
    if _clean_text(model_id):
        params["model_id"] = _clean_text(model_id)

    with SessionLocal() as sess:
        rows = sess.execute(sql, params).mappings().all()

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
    model_id: str,
) -> Dict[str, Any]:
    base_rows = _load_grant_base_rows(limit_grants=limit_grants)
    grant_ids = [_clean_text(r.get("grant_id")) for r in base_rows if _clean_text(r.get("grant_id"))]
    spec_by_grant = _load_grant_spec_keywords_from_embedding(
        grant_ids=grant_ids,
        min_spec_weight=min_spec_weight,
        model_id=model_id,
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
            "model_filter": _clean_text(model_id),
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


def _load_faculty_specs_from_embedding(
    *,
    faculty_ids: List[int],
    min_spec_weight: float,
    model_id: str,
) -> List[Dict[str, Any]]:
    if not faculty_ids:
        return []

    sql_body = """
        SELECT
            fse.id AS fac_spec_id,
            fse.faculty_id AS fac_id,
            COALESCE(fse.section, '') AS section,
            COALESCE(fse.spec_text, '') AS spec_text,
            COALESCE(fse.spec_weight, 1.0) AS spec_weight
        FROM faculty_specialization_embedding fse
        WHERE fse.faculty_id IN :faculty_ids
          AND COALESCE(fse.spec_text, '') <> ''
          AND COALESCE(fse.spec_weight, 1.0) >= :min_spec_weight
    """
    if _clean_text(model_id):
        sql_body += "\n  AND COALESCE(fse.model, '') = :model_id"
    sql_body += """
        ORDER BY
            fse.faculty_id ASC,
            CASE COALESCE(fse.section, '') WHEN 'research' THEN 0 WHEN 'application' THEN 1 ELSE 2 END ASC,
            COALESCE(fse.spec_weight, 1.0) DESC,
            fse.id ASC
    """

    sql = text(sql_body).bindparams(bindparam("faculty_ids", expanding=True))
    params: Dict[str, Any] = {
        "faculty_ids": list(faculty_ids),
        "min_spec_weight": float(max(0.0, min_spec_weight)),
    }
    if _clean_text(model_id):
        params["model_id"] = _clean_text(model_id)

    with SessionLocal() as sess:
        rows = sess.execute(sql, params).mappings().all()

    out: List[Dict[str, Any]] = []
    for row in rows:
        fac_id = _safe_int(row.get("fac_id"), default=0, minimum=0, maximum=2_147_483_647)
        fac_spec_id = _safe_int(row.get("fac_spec_id"), default=0, minimum=0, maximum=9_223_372_036_854_775_807)
        section = _clean_text(row.get("section")) or "unknown"
        spec_text = _clean_text(row.get("spec_text"))
        spec_weight = _safe_float(row.get("spec_weight"), default=1.0, minimum=0.0, maximum=1_000_000.0)
        if fac_id <= 0 or fac_spec_id <= 0 or not spec_text:
            continue
        out.append(
            {
                "fac_id": fac_id,
                "fac_spec_id": fac_spec_id,
                "section": section,
                "text": spec_text,
                "weight": float(spec_weight),
            }
        )
    return out


def _build_faculty_spec_records(
    *,
    limit_faculties: int,
    min_spec_weight: float,
    model_id: str,
) -> Dict[str, Any]:
    faculty_ids = _load_faculty_ids(limit_faculties=limit_faculties)
    rows = _load_faculty_specs_from_embedding(
        faculty_ids=faculty_ids,
        min_spec_weight=min_spec_weight,
        model_id=model_id,
    )

    per_fac_idx: Dict[int, int] = {}
    out_records: List[Dict[str, Any]] = []
    faculty_with_specs = set()
    section_counts: Dict[str, int] = {}

    for row in rows:
        fac_id = int(row["fac_id"])
        fac_spec_id = int(row["fac_spec_id"])
        section = _clean_text(row.get("section")) or "unknown"
        spec_text = _clean_text(row.get("text"))
        if not spec_text:
            continue

        idx = int(per_fac_idx.get(fac_id, 0))
        per_fac_idx[fac_id] = idx + 1
        faculty_with_specs.add(fac_id)
        section_counts[section] = int(section_counts.get(section, 0) + 1)

        out_records.append(
            {
                "fac_id": fac_id,
                "fac_spec_id": fac_spec_id,
                "fac_spec_idx": idx,
                "section": section,
                "text": spec_text,
            }
        )

    return {
        "fac_specs": out_records,
        "meta": {
            "faculty_count_scanned": len(faculty_ids),
            "faculty_with_spec_count": int(len(faculty_with_specs)),
            "fac_spec_count": int(len(out_records)),
            "section_counts": section_counts,
            "min_spec_weight": float(min_spec_weight),
            "model_filter": _clean_text(model_id),
            "limit_faculties": int(limit_faculties),
        },
    }


def build_exports(
    *,
    grant_output: Path,
    fac_output: Path,
    limit_grants: int,
    limit_faculties: int,
    min_spec_weight: float,
    model_id: str,
    indent: int,
) -> Dict[str, Any]:
    grant_payload = _build_grant_keyword_records(
        limit_grants=limit_grants,
        min_spec_weight=min_spec_weight,
        model_id=model_id,
    )
    fac_payload = _build_faculty_spec_records(
        limit_faculties=limit_faculties,
        min_spec_weight=min_spec_weight,
        model_id=model_id,
    )

    created_at_utc = datetime.now(timezone.utc).isoformat()
    grant_payload["meta"]["created_at_utc"] = created_at_utc
    grant_payload["meta"]["source"] = "postgresql"
    fac_payload["meta"]["created_at_utc"] = created_at_utc
    fac_payload["meta"]["source"] = "postgresql"

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
            "Export two local JSON DB snapshots for spec->facspec: "
            "(1) grant keywords/spec keywords, (2) faculty specialization texts."
        )
    )
    p.add_argument(
        "--grant-output",
        type=str,
        default=GRANT_OUTPUT_DEFAULT,
        help="Output JSON path for grant keywords/spec keywords.",
    )
    p.add_argument(
        "--fac-output",
        type=str,
        default=FAC_OUTPUT_DEFAULT,
        help="Output JSON path for faculty specialization texts.",
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
        help="Minimum specialization weight when reading *_specialization_embedding tables.",
    )
    p.add_argument(
        "--model-id",
        type=str,
        default="",
        help="Optional embedding model filter for both opportunity/faculty specialization tables.",
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
    model_id = _clean_text(args.model_id)
    indent = _safe_int(args.indent, default=2, minimum=0, maximum=16)

    result = build_exports(
        grant_output=grant_output,
        fac_output=fac_output,
        limit_grants=limit_grants,
        limit_faculties=limit_faculties,
        min_spec_weight=min_spec_weight,
        model_id=model_id,
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
    print(f"faculty_with_spec_count={fac_meta.get('faculty_with_spec_count')}")
    print(f"fac_spec_count={fac_meta.get('fac_spec_count')}")
    print(f"model_filter={fac_meta.get('model_filter')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
