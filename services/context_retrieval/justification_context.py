from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from dao.faculty_dao import FacultyDAO
from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.models import Faculty
from services.context_retrieval.faculty_context import FacultyContextBuilder
from services.context_retrieval.opportunity_context import OpportunityContextBuilder
from utils.content_extractor import load_extracted_content
from utils.keyword_utils import keyword_inventory_for_rerank

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class JustificationContextBuilder:
    PROFILE_FIELDS: Dict[str, Tuple[str, ...]] = {
        "faculty_recommendation_faculty": (
            "faculty_id",
            "name",
            "email",
            "profile_url",
            "keywords",
        ),
        "faculty_recommendation_opportunity": (
            "opportunity_id",
            "opportunity_title",
            "agency_name",
            "opportunity_link",
            "keywords",
            "domain_score",
            "llm_score",
        ),
    }

    def __init__(
        self,
        *,
        faculty_builder: FacultyContextBuilder | None = None,
        opportunity_builder: OpportunityContextBuilder | None = None,
    ):
        self.faculty = faculty_builder or FacultyContextBuilder()
        self.opportunity = opportunity_builder or OpportunityContextBuilder()

    @staticmethod
    def _select_fields(payload: Dict[str, Any], fields: Tuple[str, ...]) -> Dict[str, Any]:
        return {k: payload.get(k) for k in fields}

    @staticmethod
    def _norm(text: Any) -> str:
        return " ".join(str(text or "").split()).strip()

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @classmethod
    def _short(cls, text: Any, max_chars: int = 260) -> str:
        s = cls._norm(text)
        if len(s) <= int(max_chars):
            return s
        return s[: int(max_chars)].rstrip()

    @classmethod
    def _normalize_source_type(cls, value: Any) -> str:
        token = cls._norm(value).lower().replace("-", "_").replace(" ", "_")
        if token in {"additional_info_chunk", "faculty_additional_info_chunk", "grant_additional_info_chunk"}:
            return "additional_info_chunk"
        if token in {"attachment_chunk", "attachments_chunk"}:
            return "attachment_chunk"
        if token in {"publication", "faculty_publication"}:
            return "publication"
        return token

    @classmethod
    def _extract_sources(cls, sources: Any, *, allowed_types: set[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for src in list(sources or []):
            if not isinstance(src, dict):
                continue
            stype = cls._normalize_source_type(src.get("type"))
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
                    "score": cls._safe_float(src.get("score")),
                }
            )
        out.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        return out

    @staticmethod
    def _index_by_id(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        for row in list(rows or []):
            rid = row.get("id", row.get("row_id"))
            try:
                out[int(rid)] = dict(row)
            except Exception:
                continue
        return out

    @classmethod
    def _build_opportunity_contexts_source_linked(
        cls,
        *,
        sess,
        opportunity_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        ids = [cls._norm(x) for x in list(opportunity_ids or []) if cls._norm(x)]
        if not ids:
            return {}
        odao = OpportunityDAO(sess)
        opps = odao.read_opportunities_by_ids_with_relations(ids)
        out: Dict[str, Dict[str, Any]] = {}
        for opp in list(opps or []):
            oid = cls._norm(getattr(opp, "opportunity_id", None))
            if not oid:
                continue
            out[oid] = {
                "opportunity_id": oid,
                "summary_description": cls._norm(getattr(opp, "summary_description", None)),
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

    def build_grant_context_only(
        self,
        *,
        sess,
        opportunity_id: str,
        preview_chars: int = 700,
    ) -> Dict[str, Any]:
        """Fetch one grant context with extracted additional-link and attachment chunks.

        This is grant-only context (no faculty dependency), intended for grant explanation.
        """
        oid = self._norm(opportunity_id)
        if not oid:
            raise ValueError("opportunity_id is required")

        odao = OpportunityDAO(sess)
        opps = odao.read_opportunities_by_ids_with_relations([oid])
        opp = opps[0] if opps else None
        if not opp:
            raise ValueError(f"Opportunity not found: {oid}")

        add_rows = load_extracted_content(
            list(getattr(opp, "additional_info", None) or []),
            url_attr="additional_info_url",
            group_chunks=False,
            include_row_meta=True,
        )
        att_rows = load_extracted_content(
            list(getattr(opp, "attachments", None) or []),
            url_attr="file_download_path",
            title_attr="file_name",
            group_chunks=False,
            include_row_meta=True,
        )

        def _trim_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for row in list(rows or []):
                item = dict(row or {})
                if "content" in item:
                    item["content"] = self._short(item.get("content"), max_chars=int(preview_chars))
                out.append(item)
            return out

        return {
            "opportunity_id": getattr(opp, "opportunity_id", None),
            "title": getattr(opp, "opportunity_title", None),
            "agency": getattr(opp, "agency_name", None),
            "status": getattr(opp, "opportunity_status", None),
            "opportunity_link": None,
            "summary": self._norm(getattr(opp, "summary_description", None)),
            "additional_info_count": len(add_rows),
            "attachment_count": len(att_rows),
            "additional_info_extracted": _trim_rows(add_rows),
            "attachments_extracted": _trim_rows(att_rows),
        }

    def build_rerank_keyword_inventory_for_opportunity(
        self,
        *,
        sess,
        opportunity_id: str,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Keyword-only inventory payload for one grant against top-k matched faculty."""
        oid = self._norm(opportunity_id)
        if not oid:
            raise ValueError("opportunity_id is required")
        top_k = max(1, int(k))

        odao = OpportunityDAO(sess)
        mdao = MatchDAO(sess)
        fdao = FacultyDAO(sess)

        grant_ctx = odao.read_opportunity_context(oid)
        if not grant_ctx:
            raise ValueError(f"Opportunity not found: {oid}")

        gkw = keyword_inventory_for_rerank(dict(grant_ctx.get("keywords") or {}))
        rows = mdao.list_matches_for_opportunity(oid, limit=top_k)

        matches: List[Dict[str, Any]] = []
        for row in list(rows or []):
            fid = int(row.get("faculty_id"))
            fac_ctx = fdao.get_faculty_keyword_context(fid) or {}
            fkw = keyword_inventory_for_rerank(dict((fac_ctx or {}).get("keywords") or {}))
            matches.append(
                {
                    "domain_score": float(row.get("domain_score") or 0.0),
                    "llm_score": float(row.get("llm_score") or 0.0),
                    "domain_keywords": fkw.get("domain") or [],
                    "specialization_keywords": fkw.get("specialization") or {},
                }
            )

        return {
            "grant": {
                "opportunity_id": grant_ctx.get("opportunity_id"),
                "title": grant_ctx.get("title"),
                "grant_domain_keywords": gkw.get("domain") or [],
                "grant_specialization_keywords": gkw.get("specialization") or {},
            },
            "matches": matches,
        }

    @classmethod
    def _build_faculty_spec_index(cls, fac_keywords: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, Any]]]:
        out: Dict[str, Dict[int, Dict[str, Any]]] = {"research": {}, "application": {}}
        for sec in ("research", "application"):
            specs = ((fac_keywords.get(sec) or {}).get("specialization") or [])
            for idx, item in enumerate(specs):
                if isinstance(item, dict):
                    text = cls._norm(item.get("t") if item.get("t") is not None else item.get("text"))
                    sources = list(item.get("sources") or [])
                else:
                    text = cls._norm(item)
                    sources = []
                if text:
                    out[sec][int(idx)] = {"text": text, "sources": sources}
        return out

    @classmethod
    def _build_requirement_rows(
        cls,
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
                req_text = cls._norm(data.get("text"))
                req_sources: List[Dict[str, Any]] = []
                if 0 <= req_idx < len(specs):
                    spec_item = specs[req_idx]
                    if isinstance(spec_item, dict):
                        if not req_text:
                            req_text = cls._norm(spec_item.get("t") if spec_item.get("t") is not None else spec_item.get("text"))
                        req_sources = list(spec_item.get("sources") or [])
                if not req_text:
                    continue
                rows.append(
                    {
                        "section": sec,
                        "idx": int(req_idx),
                        "text": req_text,
                        "score": cls._safe_float(data.get("score")),
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


    def build_justification_retrievable_context(
        self,
        *,
        sess,
        fac: Faculty,
        top_rows: List[Tuple[str, float, float]],
    ) -> Dict[str, Any]:
        fac_ctx = self.faculty.build_faculty_context(fac, profile="keyword")

        opp_ids = [gid for (gid, _, _) in (top_rows or [])]
        score_map = {
            gid: {"domain_score": domain_score, "llm_score": llm_score}
            for (gid, domain_score, llm_score) in (top_rows or [])
        }
        if not opp_ids:
            return {
                "faculty": fac_ctx,
                "opportunities": [],
            }

        opp_dao = OpportunityDAO(sess)
        opps = opp_dao.read_opportunities_by_ids_for_keyword_context(opp_ids)
        opp_map = {o.opportunity_id: o for o in opps}

        opp_payloads: List[Dict[str, object]] = []
        for oid in opp_ids:
            opp = opp_map.get(oid)
            if not opp:
                continue
            opp_ctx = self.opportunity.build_opportunity_keyword_context(opp)
            scores = score_map.get(oid, {"domain_score": None, "llm_score": None})
            payload: Dict[str, object] = {
                "opportunity_id": opp_ctx.get("opportunity_id") or oid,
                "opportunity_title": opp_ctx.get("opportunity_title"),
                "agency_name": opp_ctx.get("agency_name"),
                "opportunity_link": opp_ctx.get("opportunity_link"),
                "keywords": opp_ctx.get("keywords") or {},
                "domain_score": float(scores["domain_score"] or 0.0),
                "llm_score": float(scores["llm_score"] or 0.0),
            }
            opp_payloads.append(payload)

        return {
            "faculty": fac_ctx,
            "opportunities": opp_payloads,
        }

    def build_justification_context(
        self,
        *,
        profile: str,
        sess,
        fac: Faculty,
        top_rows: List[Tuple[str, float, float]],
    ) -> Dict[str, Any]:
        normalized = str(profile or "").strip().lower()
        faculty_fields = self.PROFILE_FIELDS.get(f"{normalized}_faculty")
        opportunity_fields = self.PROFILE_FIELDS.get(f"{normalized}_opportunity")
        if not faculty_fields or not opportunity_fields:
            raise ValueError(f"Unsupported justification context profile: {profile}")
        full = self.build_justification_retrievable_context(
            sess=sess,
            fac=fac,
            top_rows=top_rows,
        )
        return {
            "faculty": self._select_fields(dict(full.get("faculty") or {}), faculty_fields),
            "opportunities": [
                self._select_fields(dict(payload or {}), opportunity_fields)
                for payload in list(full.get("opportunities") or [])
            ],
        }

    def build_faculty_recommendation_payloads(
        self,
        *,
        sess,
        fac: Faculty,
        top_rows: List[Tuple[str, float, float]],
    ) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
        ctx = self.build_justification_context(
            profile="faculty_recommendation",
            sess=sess,
            fac=fac,
            top_rows=top_rows,
        )
        return (
            dict(ctx.get("faculty") or {}),
            list(ctx.get("opportunities") or []),
        )

    def build_faculty_recommendation_source_linked_payload(
        self,
        *,
        sess,
        fac: Faculty,
        top_rows: List[Tuple[str, float, float]],
    ) -> Dict[str, Any]:
        fac_ctx, opp_payloads = self.build_faculty_recommendation_payloads(
            sess=sess,
            fac=fac,
            top_rows=top_rows,
        )

        opp_ids = [str(oid) for oid, _, _ in list(top_rows or [])]
        opp_contexts = self._build_opportunity_contexts_source_linked(
            sess=sess,
            opportunity_ids=opp_ids,
        )

        mdao = MatchDAO(sess)
        match_rows_by_opp: Dict[str, Dict[str, Any]] = {}
        for oid in list(opp_ids or []):
            row = mdao.get_match_for_faculty_opportunity(
                faculty_id=int(getattr(fac, "faculty_id")),
                opportunity_id=str(oid),
            )
            if row:
                match_rows_by_opp[str(oid)] = dict(row)

        fac_profile_chunks = load_extracted_content(
            list(getattr(fac, "additional_info", None) or []),
            url_attr="additional_info_url",
            group_chunks=False,
            include_row_meta=True,
        )
        fac_profile_chunk_by_id = self._index_by_id(fac_profile_chunks)
        fac_publication_by_id = {
            int(getattr(p, "id")): p
            for p in list(getattr(fac, "publications", None) or [])
            if getattr(p, "id", None) is not None
        }

        return {
            "faculty_context": dict(fac_ctx or {}),
            "opportunity_payloads": [dict(x or {}) for x in list(opp_payloads or [])],
            "opportunity_contexts": dict(opp_contexts or {}),
            "match_rows_by_opp": dict(match_rows_by_opp or {}),
            "faculty_profile_chunks_by_id": fac_profile_chunk_by_id,
            "faculty_publications_by_id": fac_publication_by_id,
        }

    def build_faculty_recommendation_source_linked_text(
        self,
        *,
        sess,
        fac: Faculty,
        top_rows: List[Tuple[str, float, float]],
        max_requirements: int = 5,
        grant_evidence_per_requirement: int = 3,
        faculty_evidence_per_requirement: int = 3,
    ) -> str:
        payload = self.build_faculty_recommendation_source_linked_payload(
            sess=sess,
            fac=fac,
            top_rows=top_rows,
        )
        fac_ctx = dict(payload.get("faculty_context") or {})
        fac_keywords = dict((fac_ctx or {}).get("keywords") or {})
        fac_specs = self._build_faculty_spec_index(fac_keywords)
        fac_publications = dict(payload.get("faculty_publications_by_id") or {})
        fac_profile_chunk_by_id = dict(payload.get("faculty_profile_chunks_by_id") or {})
        opp_payloads = list(payload.get("opportunity_payloads") or [])
        opp_contexts = dict(payload.get("opportunity_contexts") or {})
        match_rows_by_opp = dict(payload.get("match_rows_by_opp") or {})

        lines: List[str] = [
            "FACULTY",
            f"- ID: {fac_ctx.get('faculty_id')}",
            f"- Name: {self._norm(fac_ctx.get('name'))}",
            f"- Email: {self._norm(fac_ctx.get('email'))}",
        ]

        for i, opp in enumerate(list(opp_payloads or []), start=1):
            oid = self._norm(opp.get("opportunity_id"))
            opp_ctx = dict(opp_contexts.get(oid) or {})
            add_by_id = self._index_by_id(list(opp_ctx.get("additional_info_extracted") or []))
            att_by_id = self._index_by_id(list(opp_ctx.get("attachments_extracted") or []))
            match_row = dict(match_rows_by_opp.get(oid) or {})

            lines.extend(
                [
                    "",
                    f"GRANT #{i}",
                    f"- ID: {oid}",
                    f"- Title: {self._norm(opp.get('opportunity_title'))}",
                    f"- Agency: {self._norm(opp.get('agency_name'))}",
                    f"- Summary: {self._short(opp_ctx.get('summary_description') or opp.get('summary_description'), 700)}",
                ]
            )

            req_rows = self._build_requirement_rows(
                opp_payload=dict(opp or {}),
                match_row=match_row,
                max_requirements=max_requirements,
            )
            if not req_rows:
                lines.append("- Requirements from match evidence: (none)")
                continue

            lines.append("EVIDENCE-ALIGNED REQUIREMENTS")
            for ridx, req in enumerate(req_rows, start=1):
                lines.append(f"{ridx}) Requirement: {req['text']} (score={self._safe_float(req.get('score')):.2f})")

                lines.append("   Grant evidence (source-linked chunks):")
                grant_refs = self._extract_sources(
                    req.get("sources"),
                    allowed_types={"additional_info_chunk", "attachment_chunk"},
                )
                grant_lines: List[str] = []
                for ref in grant_refs:
                    row = add_by_id.get(ref["id"]) if ref["type"] == "additional_info_chunk" else att_by_id.get(ref["id"])
                    if not row:
                        continue
                    if ref["type"] == "additional_info_chunk":
                        label = f"Additional link ({self._norm(row.get('url')) or 'N/A'})"
                    else:
                        label = f"Attachment ({self._norm(row.get('title')) or self._norm(row.get('url')) or 'N/A'})"
                    grant_lines.append(f"{label}, src={ref['score']:.2f}: {self._short(row.get('content'))}")
                    if len(grant_lines) >= int(grant_evidence_per_requirement):
                        break
                if not grant_lines:
                    lines.append("   - (none)")
                else:
                    lines.extend([f"   - {x}" for x in grant_lines])

                lines.append("   Faculty evidence (source-linked):")
                pair_scores = sorted(
                    list(req.get("pair_scores") or []),
                    key=lambda x: self._safe_float((x or {}).get("score")),
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
                        self._extract_sources(
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
                        pub = fac_publications.get(ref["id"])
                        if not pub:
                            continue
                        title = self._norm(getattr(pub, "title", None))
                        abstract = self._norm(getattr(pub, "abstract", None))
                        year = getattr(pub, "year", None)
                        fac_lines.append(
                            f"Publication ({year}) {title}, src={ref['score']:.2f}: {self._short(abstract or title)}"
                        )
                    elif ref["type"] == "additional_info_chunk":
                        row = fac_profile_chunk_by_id.get(ref["id"])
                        if not row:
                            continue
                        fac_lines.append(
                            f"Profile chunk ({self._norm(row.get('url')) or 'N/A'}), src={ref['score']:.2f}: "
                            f"{self._short(row.get('content'))}"
                        )

                    if len(fac_lines) >= int(faculty_evidence_per_requirement):
                        break

                if not fac_lines:
                    lines.append("   - (none)")
                else:
                    lines.extend([f"   - {x}" for x in fac_lines])

        return "\n".join(lines).rstrip() + "\n"
