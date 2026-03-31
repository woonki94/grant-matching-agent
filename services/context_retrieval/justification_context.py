from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple



class JustificationContextBuilder:
    """Pure justification context builder.

    This builder shapes payloads from already-fetched entities/rows. It does not
    instantiate DAOs or call other context generators directly.
    """

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

    @staticmethod
    def _ordered_unique(values: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in list(values or []):
            v = str(value or "").strip()
            if not v or v in seen:
                continue
            seen.add(v)
            out.append(v)
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
                            req_text = cls._norm(
                                spec_item.get("t") if spec_item.get("t") is not None else spec_item.get("text")
                            )
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


    @classmethod
    def build_faculty_recommendation_payloads_from_entities(
        cls,
        *,
        fac: Any,
        opportunities: List[Any],
        top_rows: List[Tuple[str, float, float]],
        build_faculty_keyword_context: Callable[[Any], Dict[str, Any]],
        build_opportunity_keyword_context: Callable[[Any], Dict[str, Any]],
    ) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
        """Build faculty + top-opportunity payloads for recommendation prompting."""
        fac_ctx = dict(build_faculty_keyword_context(fac) or {})
        opp_by_id: Dict[str, Any] = {}
        for opp in list(opportunities or []):
            oid = str(getattr(opp, "opportunity_id", "") or "").strip()
            if not oid:
                continue
            opp_by_id[oid] = opp

        score_map = {
            str(oid): {"domain_score": float(domain_score), "llm_score": float(llm_score)}
            for oid, domain_score, llm_score in list(top_rows or [])
        }

        payloads: List[Dict[str, object]] = []
        ordered_ids = cls._ordered_unique([str(oid) for oid, _, _ in list(top_rows or [])])
        for oid in ordered_ids:
            opp = opp_by_id.get(oid)
            if not opp:
                continue
            opp_ctx = dict(build_opportunity_keyword_context(opp) or {})
            scores = score_map.get(oid, {"domain_score": 0.0, "llm_score": 0.0})
            payloads.append(
                {
                    "opportunity_id": opp_ctx.get("opportunity_id") or oid,
                    "opportunity_title": opp_ctx.get("opportunity_title"),
                    "agency_name": opp_ctx.get("agency_name"),
                    "opportunity_link": opp_ctx.get("opportunity_link"),
                    "keywords": opp_ctx.get("keywords") or {},
                    "domain_score": cls._safe_float(scores.get("domain_score"), 0.0),
                    "llm_score": cls._safe_float(scores.get("llm_score"), 0.0),
                }
            )

        return fac_ctx, payloads

    @classmethod
    def build_opportunity_source_contexts_from_entities(
        cls,
        *,
        opportunities: List[Any],
        build_opportunity_source_linked_context: Callable[[Any], Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Build source-linked opportunity contexts with extracted chunk blocks."""
        out: Dict[str, Dict[str, Any]] = {}
        for opp in list(opportunities or []):
            oid = cls._norm(getattr(opp, "opportunity_id", None))
            if not oid:
                continue
            full = dict(build_opportunity_source_linked_context(opp) or {})
            out[oid] = {
                "opportunity_id": oid,
                "summary_description": cls._norm(full.get("summary_description")),
                "additional_info_extracted": list(full.get("additional_info_extracted") or []),
                "attachments_extracted": list(full.get("attachments_extracted") or []),
            }
        return out

    @classmethod
    def build_match_rows_by_opp_from_rows(
        cls,
        *,
        rows: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Index match rows by opportunity/grant id."""
        out: Dict[str, Dict[str, Any]] = {}
        for row in list(rows or []):
            oid = cls._norm((row or {}).get("grant_id") or (row or {}).get("opportunity_id"))
            if not oid:
                continue
            out[oid] = dict(row or {})
        return out

    @classmethod
    def build_faculty_recommendation_source_linked_payload_from_entities(
        cls,
        *,
        fac: Any,
        opportunities: List[Any],
        top_rows: List[Tuple[str, float, float]],
        match_rows: List[Dict[str, Any]],
        build_faculty_keyword_context: Callable[[Any], Dict[str, Any]],
        build_opportunity_keyword_context: Callable[[Any], Dict[str, Any]],
        build_faculty_source_linked_context: Callable[[Any], Dict[str, Any]],
        build_opportunity_source_linked_context: Callable[[Any], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build source-linked recommendation payload from entities + match rows."""
        fac_ctx, opp_payloads = cls.build_faculty_recommendation_payloads_from_entities(
            fac=fac,
            opportunities=opportunities,
            top_rows=top_rows,
            build_faculty_keyword_context=build_faculty_keyword_context,
            build_opportunity_keyword_context=build_opportunity_keyword_context,
        )

        opp_contexts = cls.build_opportunity_source_contexts_from_entities(
            opportunities=opportunities,
            build_opportunity_source_linked_context=build_opportunity_source_linked_context,
        )
        match_rows_by_opp = cls.build_match_rows_by_opp_from_rows(rows=match_rows)

        fac_source_ctx = dict(build_faculty_source_linked_context(fac) or {})
        fac_profile_chunks = list(fac_source_ctx.get("additional_info_extracted") or [])
        fac_publication_rows = list(fac_source_ctx.get("publications") or [])
        fac_profile_chunk_by_id = cls._index_by_id(fac_profile_chunks)
        fac_publication_by_id: Dict[int, Dict[str, Any]] = {}
        for p in list(fac_publication_rows or []):
            try:
                pid = int((p or {}).get("id"))
            except Exception:
                continue
            fac_publication_by_id[pid] = dict(p or {})

        return {
            "faculty_context": dict(fac_ctx or {}),
            "opportunity_payloads": [dict(x or {}) for x in list(opp_payloads or [])],
            "opportunity_contexts": dict(opp_contexts or {}),
            "match_rows_by_opp": dict(match_rows_by_opp or {}),
            "faculty_profile_chunks_by_id": fac_profile_chunk_by_id,
            "faculty_publications_by_id": fac_publication_by_id,
        }

    @classmethod
    def build_faculty_recommendation_source_linked_text_from_payload(
        cls,
        *,
        payload: Dict[str, Any],
        max_requirements: int = 5,
        grant_evidence_per_requirement: int = 3,
        faculty_evidence_per_requirement: int = 3,
    ) -> str:
        """Render source-linked recommendation payload into compact evidence text."""
        fac_ctx = dict(payload.get("faculty_context") or {})
        fac_keywords = dict((fac_ctx or {}).get("keywords") or {})
        fac_specs = cls._build_faculty_spec_index(fac_keywords)
        fac_publications = dict(payload.get("faculty_publications_by_id") or {})
        fac_profile_chunk_by_id = dict(payload.get("faculty_profile_chunks_by_id") or {})
        opp_payloads = list(payload.get("opportunity_payloads") or [])
        opp_contexts = dict(payload.get("opportunity_contexts") or {})
        match_rows_by_opp = dict(payload.get("match_rows_by_opp") or {})

        lines: List[str] = [
            "FACULTY",
            f"- ID: {fac_ctx.get('faculty_id')}",
            f"- Name: {cls._norm(fac_ctx.get('name'))}",
            f"- Email: {cls._norm(fac_ctx.get('email'))}",
        ]

        for i, opp in enumerate(list(opp_payloads or []), start=1):
            oid = cls._norm(opp.get("opportunity_id"))
            opp_ctx = dict(opp_contexts.get(oid) or {})
            add_by_id = cls._index_by_id(list(opp_ctx.get("additional_info_extracted") or []))
            att_by_id = cls._index_by_id(list(opp_ctx.get("attachments_extracted") or []))
            match_row = dict(match_rows_by_opp.get(oid) or {})

            lines.extend(
                [
                    "",
                    f"GRANT #{i}",
                    f"- ID: {oid}",
                    f"- Title: {cls._norm(opp.get('opportunity_title'))}",
                    f"- Agency: {cls._norm(opp.get('agency_name'))}",
                    f"- Summary: {cls._short(opp_ctx.get('summary_description') or opp.get('summary_description'), 700)}",
                ]
            )

            req_rows = cls._build_requirement_rows(
                opp_payload=dict(opp or {}),
                match_row=match_row,
                max_requirements=max_requirements,
            )
            if not req_rows:
                lines.append("- Requirements from match evidence: (none)")
                continue

            lines.append("EVIDENCE-ALIGNED REQUIREMENTS")
            for ridx, req in enumerate(req_rows, start=1):
                lines.append(f"{ridx}) Requirement: {req['text']} (score={cls._safe_float(req.get('score')):.2f})")

                lines.append("   Grant evidence (source-linked chunks):")
                grant_refs = cls._extract_sources(
                    req.get("sources"),
                    allowed_types={"additional_info_chunk", "attachment_chunk"},
                )
                grant_lines: List[str] = []
                for ref in grant_refs:
                    row = add_by_id.get(ref["id"]) if ref["type"] == "additional_info_chunk" else att_by_id.get(ref["id"])
                    if not row:
                        continue
                    if ref["type"] == "additional_info_chunk":
                        label = f"Additional link ({cls._norm(row.get('url')) or 'N/A'})"
                    else:
                        label = f"Attachment ({cls._norm(row.get('title')) or cls._norm(row.get('url')) or 'N/A'})"
                    grant_lines.append(f"{label}, src={ref['score']:.2f}: {cls._short(row.get('content'))}")
                    if len(grant_lines) >= int(grant_evidence_per_requirement):
                        break
                if not grant_lines:
                    lines.append("   - (none)")
                else:
                    lines.extend([f"   - {x}" for x in grant_lines])

                lines.append("   Faculty evidence (source-linked):")
                pair_scores = sorted(
                    list(req.get("pair_scores") or []),
                    key=lambda x: cls._safe_float((x or {}).get("score")),
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
                        cls._extract_sources(
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
                        if isinstance(pub, dict):
                            title = cls._norm(pub.get("title"))
                            abstract = cls._norm(pub.get("abstract"))
                            year = pub.get("year")
                        else:
                            title = cls._norm(getattr(pub, "title", None))
                            abstract = cls._norm(getattr(pub, "abstract", None))
                            year = getattr(pub, "year", None)
                        fac_lines.append(
                            f"Publication ({year}) {title}, src={ref['score']:.2f}: {cls._short(abstract or title)}"
                        )
                    elif ref["type"] == "additional_info_chunk":
                        row = fac_profile_chunk_by_id.get(ref["id"])
                        if not row:
                            continue
                        fac_lines.append(
                            f"Profile chunk ({cls._norm(row.get('url')) or 'N/A'}), src={ref['score']:.2f}: "
                            f"{cls._short(row.get('content'))}"
                        )

                    if len(fac_lines) >= int(faculty_evidence_per_requirement):
                        break

                if not fac_lines:
                    lines.append("   - (none)")
                else:
                    lines.extend([f"   - {x}" for x in fac_lines])

        return "\n".join(lines).rstrip() + "\n"
