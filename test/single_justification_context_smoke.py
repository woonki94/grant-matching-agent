from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dao.faculty_dao import FacultyDAO
from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from logging_setup import setup_logging
from services.context_retrieval.context_generator import ContextGenerator
from utils.content_extractor import load_extracted_content


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _short(text: Any, max_chars: int = 260) -> str:
    s = _norm(text)
    if len(s) <= int(max_chars):
        return s
    return s[: int(max_chars)].rstrip()


def _normalize_source_type(value: Any) -> str:
    token = _norm(value).lower().replace("-", "_").replace(" ", "_")
    if token in {"additional_info_chunk", "faculty_additional_info_chunk", "grant_additional_info_chunk"}:
        return "additional_info_chunk"
    if token in {"attachment_chunk", "attachments_chunk"}:
        return "attachment_chunk"
    if token in {"publication", "faculty_publication"}:
        return "publication"
    return token


def _extract_sources(sources: Any, *, allowed_types: set[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for src in list(sources or []):
        if not isinstance(src, dict):
            continue
        stype = _normalize_source_type(src.get("type"))
        if stype not in allowed_types:
            continue
        try:
            sid = int(src.get("id"))
        except Exception:
            continue
        key = (stype, sid)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "type": stype,
                "id": sid,
                "score": _safe_float(src.get("score")),
            }
        )
    out.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return out


def _index_by_id(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in list(rows or []):
        rid = row.get("id", row.get("row_id"))
        try:
            out[int(rid)] = dict(row)
        except Exception:
            continue
    return out


def _build_top_rows(
    *,
    sess,
    faculty_id: int,
    k: int,
    opportunity_id: str | None,
) -> List[Tuple[str, float, float]]:
    mdao = MatchDAO(sess)

    if opportunity_id:
        row = mdao.get_match_for_faculty_opportunity(
            faculty_id=int(faculty_id),
            opportunity_id=str(opportunity_id),
        )
        if not row:
            raise ValueError(
                f"No match row found for faculty_id={faculty_id}, opportunity_id={opportunity_id}"
            )
        return [
            (
                str(opportunity_id),
                _safe_float(row.get("domain_score")),
                _safe_float(row.get("llm_score")),
            )
        ]

    rows = mdao.top_matches_for_faculty(faculty_id=int(faculty_id), k=max(1, int(k)))
    if not rows:
        raise ValueError(f"No matches found for faculty_id={faculty_id}.")

    out: List[Tuple[str, float, float]] = []
    for opp_id, domain_score, llm_score in list(rows or [])[: max(1, int(k))]:
        oid = _norm(opp_id)
        if not oid:
            continue
        out.append((oid, _safe_float(domain_score), _safe_float(llm_score)))

    if not out:
        raise ValueError("No valid opportunity IDs were found in top matches.")
    return out


def _build_opportunity_contexts(*, sess, opportunity_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    ids = [_norm(x) for x in list(opportunity_ids or []) if _norm(x)]
    if not ids:
        return {}
    odao = OpportunityDAO(sess)
    opps = odao.read_opportunities_by_ids_with_relations(ids)
    out: Dict[str, Dict[str, Any]] = {}
    for opp in list(opps or []):
        oid = _norm(getattr(opp, "opportunity_id", None))
        if not oid:
            continue
        out[oid] = {
            "opportunity_id": oid,
            "summary_description": _norm(getattr(opp, "summary_description", None)),
            "additional_info_extracted": load_extracted_content(
                list(getattr(opp, "additional_info", None) or []),
                url_attr="additional_info_url",
                group_chunks=False,
                include_row_meta=True,
            ),
            "attachments_extracted": load_extracted_content(
                list(getattr(opp, "attachments", None) or []),
                url_attr="file_download_path",
                title_attr="file_name",
                group_chunks=False,
                include_row_meta=True,
            ),
        }
    return out


def _build_faculty_spec_index(fac_keywords: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    out: Dict[str, Dict[int, Dict[str, Any]]] = {"research": {}, "application": {}}
    for sec in ("research", "application"):
        specs = ((fac_keywords.get(sec) or {}).get("specialization") or [])
        for idx, item in enumerate(specs):
            if isinstance(item, dict):
                text = _norm(item.get("t") if item.get("t") is not None else item.get("text"))
                sources = list(item.get("sources") or [])
            else:
                text = _norm(item)
                sources = []
            if text:
                out[sec][int(idx)] = {"text": text, "sources": sources}
    return out


def _build_requirement_rows(
    *,
    opp_payload: Dict[str, Any],
    match_row: Dict[str, Any],
    max_requirements: int,
) -> List[Dict[str, Any]]:
    opp_keywords = dict(opp_payload.get("keywords") or {})
    evidence = dict(match_row.get("evidence") or {})
    sections = dict(evidence.get("sections") or {})

    rows: List[Dict[str, Any]] = []
    for sec in ("research", "application"):
        sec_rows = dict(sections.get(sec) or {})
        specs = list(((opp_keywords.get(sec) or {}).get("specialization") or []))
        for idx_key, row in sec_rows.items():
            try:
                req_idx = int(idx_key)
            except Exception:
                continue
            data = dict(row or {})
            req_text = _norm(data.get("text"))
            req_sources: List[Dict[str, Any]] = []
            if 0 <= req_idx < len(specs):
                spec_item = specs[req_idx]
                if isinstance(spec_item, dict):
                    if not req_text:
                        req_text = _norm(spec_item.get("t") if spec_item.get("t") is not None else spec_item.get("text"))
                    req_sources = list(spec_item.get("sources") or [])
            if not req_text:
                continue

            rows.append(
                {
                    "section": sec,
                    "idx": int(req_idx),
                    "text": req_text,
                    "score": _safe_float(data.get("score")),
                    "pair_scores": list(data.get("pair_scores") or []),
                    "sources": req_sources,
                    "has_sources": bool(req_sources),
                }
            )

    rows.sort(
        key=lambda x: (
            1 if bool(x.get("has_sources")) else 0,
            float(x.get("score") or 0.0),
        ),
        reverse=True,
    )
    return rows[: max(1, int(max_requirements))]


def _format_text(
    *,
    fac_ctx: Dict[str, Any],
    fac_keywords: Dict[str, Any],
    fac_publication_by_id: Dict[int, Any],
    fac_profile_chunk_by_id: Dict[int, Dict[str, Any]],
    opp_payloads: List[Dict[str, Any]],
    opp_contexts: Dict[str, Dict[str, Any]],
    match_rows_by_opp: Dict[str, Dict[str, Any]],
    max_requirements: int,
    grant_evidence_per_requirement: int,
    faculty_evidence_per_requirement: int,
) -> str:
    fac_specs = _build_faculty_spec_index(fac_keywords)
    lines: List[str] = [
        "FACULTY",
        f"- ID: {fac_ctx.get('faculty_id')}",
        f"- Name: {_norm(fac_ctx.get('name'))}",
        f"- Email: {_norm(fac_ctx.get('email'))}",
    ]

    for i, opp in enumerate(list(opp_payloads or []), start=1):
        oid = _norm(opp.get("opportunity_id"))
        opp_ctx = dict(opp_contexts.get(oid) or {})
        add_by_id = _index_by_id(list(opp_ctx.get("additional_info_extracted") or []))
        att_by_id = _index_by_id(list(opp_ctx.get("attachments_extracted") or []))
        match_row = dict(match_rows_by_opp.get(oid) or {})

        lines.extend(
            [
                "",
                f"GRANT #{i}",
                f"- ID: {oid}",
                f"- Title: {_norm(opp.get('opportunity_title'))}",
                f"- Agency: {_norm(opp.get('agency_name'))}",
                f"- Summary: {_short(opp_ctx.get('summary_description') or opp.get('summary_description'), 700)}",
            ]
        )

        req_rows = _build_requirement_rows(
            opp_payload=opp,
            match_row=match_row,
            max_requirements=max_requirements,
        )
        if not req_rows:
            lines.append("- Requirements from match evidence: (none)")
            continue

        lines.append("EVIDENCE-ALIGNED REQUIREMENTS")
        for ridx, req in enumerate(req_rows, start=1):
            lines.append(f"{ridx}) Requirement: {req['text']} (score={_safe_float(req.get('score')):.2f})")

            lines.append("   Grant evidence (source-linked chunks):")
            grant_refs = _extract_sources(
                req.get("sources"),
                allowed_types={"additional_info_chunk", "attachment_chunk"},
            )
            grant_lines: List[str] = []
            for ref in grant_refs:
                row = add_by_id.get(ref["id"]) if ref["type"] == "additional_info_chunk" else att_by_id.get(ref["id"])
                if not row:
                    continue
                if ref["type"] == "additional_info_chunk":
                    label = f"Additional link ({_norm(row.get('url')) or 'N/A'})"
                else:
                    label = f"Attachment ({_norm(row.get('title')) or _norm(row.get('url')) or 'N/A'})"
                grant_lines.append(f"{label}, src={ref['score']:.2f}: {_short(row.get('content'))}")
                if len(grant_lines) >= int(grant_evidence_per_requirement):
                    break
            if not grant_lines:
                lines.append("   - (none)")
            else:
                lines.extend([f"   - {x}" for x in grant_lines])

            lines.append("   Faculty evidence (source-linked):")
            pair_scores = sorted(
                list(req.get("pair_scores") or []),
                key=lambda x: _safe_float((x or {}).get("score")),
                reverse=True,
            )
            fac_refs: List[Dict[str, Any]] = []
            for pair in pair_scores:
                try:
                    fac_spec_idx = int(pair.get("fac_spec_idx"))
                except Exception:
                    continue
                fac_spec = fac_specs.get(req["section"], {}).get(fac_spec_idx)
                if not fac_spec:
                    continue
                fac_refs.extend(
                    _extract_sources(
                        fac_spec.get("sources"),
                        allowed_types={"publication", "additional_info_chunk"},
                    )
                )

            fac_lines: List[str] = []
            seen = set()
            for ref in fac_refs:
                key = (ref["type"], ref["id"])
                if key in seen:
                    continue
                seen.add(key)

                if ref["type"] == "publication":
                    pub = fac_publication_by_id.get(ref["id"])
                    if not pub:
                        continue
                    title = _norm(getattr(pub, "title", None))
                    abstract = _norm(getattr(pub, "abstract", None))
                    year = getattr(pub, "year", None)
                    fac_lines.append(
                        f"Publication ({year}) {title}, src={ref['score']:.2f}: {_short(abstract or title)}"
                    )
                elif ref["type"] == "additional_info_chunk":
                    row = fac_profile_chunk_by_id.get(ref["id"])
                    if not row:
                        continue
                    fac_lines.append(
                        f"Profile chunk ({_norm(row.get('url')) or 'N/A'}), src={ref['score']:.2f}: {_short(row.get('content'))}"
                    )

                if len(fac_lines) >= int(faculty_evidence_per_requirement):
                    break

            if not fac_lines:
                lines.append("   - (none)")
            else:
                lines.extend([f"   - {x}" for x in fac_lines])

    return "\n".join(lines).rstrip() + "\n"


def main(
    email: str,
    k: int,
    opportunity_id: str | None,
    output_format: str,
    max_requirements: int,
    grant_evidence_per_requirement: int,
    faculty_evidence_per_requirement: int,
) -> int:
    with SessionLocal() as sess:
        fdao = FacultyDAO(sess)
        fac = fdao.get_with_relations_by_email(email)
        if not fac:
            raise ValueError(f"No faculty found with email: {email}")

        top_rows = _build_top_rows(
            sess=sess,
            faculty_id=int(fac.faculty_id),
            k=k,
            opportunity_id=(_norm(opportunity_id) if opportunity_id else None),
        )

        mdao = MatchDAO(sess)
        match_rows_by_opp: Dict[str, Dict[str, Any]] = {}
        for oid, _, _ in list(top_rows or []):
            row = mdao.get_match_for_faculty_opportunity(
                faculty_id=int(fac.faculty_id),
                opportunity_id=str(oid),
            )
            if row:
                match_rows_by_opp[str(oid)] = dict(row)

        cgen = ContextGenerator()
        fac_ctx, opp_payloads = cgen.build_faculty_recommendation_payloads(
            sess=sess,
            fac=fac,
            top_rows=top_rows,
        )

        opp_ids = [str(oid) for oid, _, _ in top_rows]
        opp_contexts = _build_opportunity_contexts(sess=sess, opportunity_ids=opp_ids)

        fac_profile_chunks = load_extracted_content(
            list(getattr(fac, "additional_info", None) or []),
            url_attr="additional_info_url",
            group_chunks=False,
            include_row_meta=True,
        )
        fac_profile_chunk_by_id = _index_by_id(fac_profile_chunks)
        fac_publication_by_id = {
            int(getattr(p, "id")): p
            for p in list(getattr(fac, "publications", None) or [])
            if getattr(p, "id", None) is not None
        }

        if _norm(output_format).lower() == "json":
            print(
                json.dumps(
                    {
                        "faculty_context": fac_ctx,
                        "opportunity_payloads": opp_payloads,
                        "opportunity_contexts": opp_contexts,
                        "match_rows_by_opp": match_rows_by_opp,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
            return 0

        text = _format_text(
            fac_ctx=dict(fac_ctx or {}),
            fac_keywords=dict((fac_ctx or {}).get("keywords") or {}),
            fac_publication_by_id=fac_publication_by_id,
            fac_profile_chunk_by_id=fac_profile_chunk_by_id,
            opp_payloads=[dict(x or {}) for x in list(opp_payloads or [])],
            opp_contexts=dict(opp_contexts or {}),
            match_rows_by_opp=dict(match_rows_by_opp or {}),
            max_requirements=max_requirements,
            grant_evidence_per_requirement=grant_evidence_per_requirement,
            faculty_evidence_per_requirement=faculty_evidence_per_requirement,
        )
        print(text)
        return 0


if __name__ == "__main__":
    setup_logging("justification")
    parser = argparse.ArgumentParser(description="Smoke test: source-linked justification context retrieval.")
    parser.add_argument("--email", required=True, help="Faculty email in DB")
    parser.add_argument("--k", type=int, default=10, help="Top-K matched opportunities")
    parser.add_argument("--opportunity-id", default=None, help="Optional single opportunity_id")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--max-requirements", type=int, default=5, help="Max requirements from match evidence")
    parser.add_argument(
        "--grant-evidence-per-requirement",
        type=int,
        default=5,
        help="Max source-linked grant chunks per requirement",
    )
    parser.add_argument(
        "--faculty-evidence-per-requirement",
        type=int,
        default=5,
        help="Max source-linked faculty evidence items per requirement",
    )
    args = parser.parse_args()
    raise SystemExit(
        main(
            email=args.email.strip(),
            k=int(args.k),
            opportunity_id=args.opportunity_id,
            output_format=args.format,
            max_requirements=max(1, int(args.max_requirements)),
            grant_evidence_per_requirement=max(1, int(args.grant_evidence_per_requirement)),
            faculty_evidence_per_requirement=max(1, int(args.faculty_evidence_per_requirement)),
        )
    )
