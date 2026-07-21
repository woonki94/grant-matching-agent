from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dao.faculty_dao import FacultyDAO
from db.db_conn import SessionLocal
from dto.faculty_dto import FacultyDTO
from logging_setup import setup_logging
from services.faculty.faculty_profile_service import FacultyProfileService

logger = logging.getLogger("manual_upsert_faculty")
setup_logging()


def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _safe_text(v: Any) -> Optional[str]:
    if v is None:
        return None
    out = str(v).strip()
    return out or None


def _normalize_basic_info(payload: Dict[str, Any]) -> Dict[str, Any]:
    basic_info = dict(payload.get("basic_info") or {})
    if not basic_info:
        basic_info = {}

    # Convenience top-level aliases.
    field_map = {
        "faculty_name": "faculty_name",
        "name": "faculty_name",
        "position": "position",
        "phone": "phone",
        "address": "address",
        "biography": "biography",
        "degrees": "degrees",
        "expertise": "expertise",
        "organizations": "organizations",
    }
    for src_key, dst_key in field_map.items():
        if dst_key in basic_info:
            continue
        if src_key in payload:
            basic_info[dst_key] = payload.get(src_key)
    return basic_info


def _normalize_attached_file_add(rows: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in _as_list(rows):
        if isinstance(row, str):
            url = _safe_text(row)
            if url:
                out.append({"source_url": url})
            continue
        if isinstance(row, dict):
            url = _safe_text(row.get("source_url") or row.get("additional_info_url"))
            if not url:
                continue
            out.append({"source_url": url})
    return out


def _normalize_data_from(payload: Dict[str, Any]) -> Dict[str, Any]:
    data_from = dict(payload.get("data_from") or {})
    if not data_from:
        data_from = {}

    if "info_source_url" not in data_from and "info_source_url" in payload:
        data_from["info_source_url"] = payload.get("info_source_url")

    # Convenience: top-level publications treated as publications.add
    if "publications" in payload and "publications" not in data_from:
        data_from["publications"] = {"add": _as_list(payload.get("publications"))}

    # Convenience: top-level attached_files/additional_info_urls treated as attached_files.add
    if "attached_files" in payload and "attached_files" not in data_from:
        data_from["attached_files"] = {
            "add": _normalize_attached_file_add(payload.get("attached_files"))
        }
    if "additional_info_urls" in payload and "attached_files" not in data_from:
        data_from["attached_files"] = {
            "add": _normalize_attached_file_add(payload.get("additional_info_urls"))
        }

    if isinstance(data_from.get("attached_files"), dict):
        files_ops = dict(data_from.get("attached_files") or {})
        if "add" in files_ops:
            files_ops["add"] = _normalize_attached_file_add(files_ops.get("add"))
        data_from["attached_files"] = files_ops

    return data_from


def _resolve_create_source_url(payload: Dict[str, Any], data_from: Dict[str, Any]) -> Optional[str]:
    candidates = [
        _safe_text(payload.get("source_url")),
        _safe_text(payload.get("info_source_url")),
        _safe_text(data_from.get("info_source_url")),
    ]
    for x in candidates:
        if x:
            return x
    return None


def _ensure_faculty_exists(
    *,
    email: str,
    payload: Dict[str, Any],
    basic_info: Dict[str, Any],
    data_from: Dict[str, Any],
) -> Dict[str, Any]:
    with SessionLocal() as sess:
        dao = FacultyDAO(sess)
        existing = dao.get_by_email(email)
        if existing is not None:
            return {
                "created": False,
                "faculty_id": int(existing.faculty_id),
            }

        source_url = _resolve_create_source_url(payload, data_from)
        if not source_url:
            raise ValueError(
                "source_url is required to create a new faculty. "
                "Provide top-level source_url/info_source_url or data_from.info_source_url."
            )

        organizations = basic_info.get("organizations")
        organization_one_line = None
        if isinstance(organizations, list):
            cleaned_orgs = [str(x).strip() for x in organizations if str(x).strip()]
            if cleaned_orgs:
                organization_one_line = " | ".join(cleaned_orgs)
        elif _safe_text(payload.get("organization")):
            organization_one_line = _safe_text(payload.get("organization"))

        dto = FacultyDTO(
            source_url=str(source_url),
            name=_safe_text(basic_info.get("faculty_name") or basic_info.get("name")),
            email=str(email),
            phone=_safe_text(basic_info.get("phone")),
            position=_safe_text(basic_info.get("position")),
            organization=organization_one_line,
            organizations=(basic_info.get("organizations") if isinstance(basic_info.get("organizations"), list) else None),
            address=_safe_text(basic_info.get("address")),
            biography=_safe_text(basic_info.get("biography")),
            degrees=(basic_info.get("degrees") if isinstance(basic_info.get("degrees"), list) else None),
            expertise=(basic_info.get("expertise") if isinstance(basic_info.get("expertise"), list) else None),
        )
        created = dao.upsert_faculty(dto)
        sess.commit()
        return {
            "created": True,
            "faculty_id": int(created.faculty_id),
        }


def run_manual_upsert(
    *,
    payload: Dict[str, Any],
    run_postprocess: bool = True,
) -> Dict[str, Any]:
    email = _safe_text(payload.get("email"))
    if not email:
        raise ValueError("payload.email is required.")
    normalized_email = str(email).lower()

    basic_info = _normalize_basic_info(payload)
    data_from = _normalize_data_from(payload)
    create_result = _ensure_faculty_exists(
        email=normalized_email,
        payload=payload,
        basic_info=basic_info,
        data_from=data_from,
    )

    service = FacultyProfileService()
    out = service.edit_faculty_profile(
        email=normalized_email,
        basic_info=basic_info,
        data_from=data_from,
        all_keywords=(payload.get("all_keywords") if "all_keywords" in payload else None),
        keyword_source=payload.get("keyword_source"),
        force_regenerate_keywords=payload.get("force_regenerate_keywords"),
        run_postprocess=bool(run_postprocess),
    )

    return {
        "created_faculty": bool(create_result.get("created")),
        "created_faculty_id": int(create_result.get("faculty_id") or 0),
        **dict(out or {}),
    }


def _normalize_payload_entries(payload_doc: Any) -> List[Dict[str, Any]]:
    """
    Accept payload file shapes:
      1) { ...single faculty payload... }
      2) [ { ... }, { ... } ]
      3) { "faculties": [ { ... }, { ... } ] }
    Returns a non-empty list of faculty payload objects.
    """
    entries_raw: Any
    if isinstance(payload_doc, list):
        entries_raw = payload_doc
    elif isinstance(payload_doc, dict):
        if isinstance(payload_doc.get("faculties"), list):
            entries_raw = payload_doc.get("faculties")
        else:
            entries_raw = [payload_doc]
    else:
        raise ValueError("Payload file must be a JSON object or array.")

    entries: List[Dict[str, Any]] = []
    for item in list(entries_raw or []):
        if not isinstance(item, dict):
            raise ValueError("Each faculty payload must be a JSON object.")
        entries.append(dict(item))
    if not entries:
        raise ValueError("No faculty payloads found in payload file.")
    return entries


def run_manual_upsert_batch(
    *,
    payload_doc: Any,
    run_postprocess: bool = True,
) -> Dict[str, Any]:
    entries = _normalize_payload_entries(payload_doc)
    results: List[Dict[str, Any]] = []
    created_count = 0
    succeeded = 0
    failed = 0
    total_match_rows = 0

    for idx, payload in enumerate(entries, start=1):
        email = str(_safe_text(payload.get("email")) or "").strip().lower()
        try:
            out = run_manual_upsert(
                payload=dict(payload),
                run_postprocess=bool(run_postprocess),
            )
            results.append(
                {
                    "index": int(idx),
                    "email": email or None,
                    "ok": True,
                    **dict(out or {}),
                }
            )
            succeeded += 1
            if bool(out.get("created_faculty")):
                created_count += 1
            total_match_rows += int(out.get("match_rows_upserted") or 0)
        except Exception as exc:
            failed += 1
            err = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "manual_upsert_faculty batch item failed index=%s email=%s error=%s",
                int(idx),
                email,
                err,
            )
            results.append(
                {
                    "index": int(idx),
                    "email": email or None,
                    "ok": False,
                    "error": err,
                }
            )

    return {
        "ok": bool(failed == 0),
        "mode": "batch",
        "total": int(len(entries)),
        "succeeded": int(succeeded),
        "failed": int(failed),
        "created_faculty_count": int(created_count),
        "match_rows_upserted_total": int(total_match_rows),
        "run_postprocess": bool(run_postprocess),
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Temporary helper: manually upsert faculty payload(s) from JSON file. "
            "Supports single or batch payload shapes."
        )
    )
    parser.add_argument(
        "--payload-file",
        type=str,
        required=True,
        help="Path to JSON payload file.",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Skip keyword/match postprocess after source updates.",
    )
    args = parser.parse_args()

    payload_path = Path(str(args.payload_file)).expanduser().resolve()
    if not payload_path.exists():
        raise FileNotFoundError(f"Payload file not found: {payload_path}")

    with payload_path.open("r", encoding="utf-8") as f:
        payload_doc = json.load(f)

    summary = run_manual_upsert_batch(
        payload_doc=payload_doc,
        run_postprocess=(not bool(args.no_postprocess)),
    )
    print(json.dumps(summary, ensure_ascii=False))
