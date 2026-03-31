from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.keyword_utils import extract_specializations


class MatchingContextBuilder:
    """Pure matching context builder.

    This builder only shapes payloads from already-prepared contexts/data.
    It does not fetch DB rows or call faculty/opportunity context builders.
    """

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

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
    def _opportunity_map(opportunities: List[Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for opp in list(opportunities or []):
            oid = str(getattr(opp, "opportunity_id", "") or "").strip()
            if not oid:
                continue
            out[oid] = opp
        return out

    @staticmethod
    def _faculty_map(faculties: List[Any]) -> Dict[int, Any]:
        out: Dict[int, Any] = {}
        for fac in list(faculties or []):
            try:
                fid = int(getattr(fac, "faculty_id"))
            except Exception:
                continue
            out[fid] = fac
        return out

    @classmethod
    def build_group_matching_context(
        cls,
        *,
        opp_ctx: Dict[str, Any],
        fac_ctxs: List[Dict[str, Any]],
        coverage: Any,
        member_coverages: Optional[Dict[int, Dict[str, Dict[int, float]]]] = None,
        group_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build group matching payload from prepared grant/faculty contexts."""
        source_opp = dict(opp_ctx or {})
        source_member_coverages = dict(member_coverages or {})

        grant_payload = {
            "opportunity_id": source_opp.get("opportunity_id"),
            "opportunity_title": source_opp.get("opportunity_title") or source_opp.get("title"),
            "agency_name": source_opp.get("agency_name") or source_opp.get("agency"),
            "summary_description": source_opp.get("summary_description") or source_opp.get("summary"),
            "keywords": source_opp.get("keywords") or {},
        }

        team_payload: List[Dict[str, Any]] = []
        for fctx in list(fac_ctxs or []):
            fid_raw = fctx.get("faculty_id") if isinstance(fctx, dict) else None
            try:
                fid = int(fid_raw)
            except Exception:
                fid = None
            covered = {"application": {}, "research": {}}
            if fid is not None:
                covered = dict(source_member_coverages.get(fid) or covered)
            team_payload.append(
                {
                    "faculty_id": fid,
                    "name": (fctx or {}).get("name"),
                    "email": (fctx or {}).get("email"),
                    "covered": {
                        "application": dict((covered or {}).get("application") or {}),
                        "research": dict((covered or {}).get("research") or {}),
                    },
                }
            )

        return {
            "grant": grant_payload,
            "team": team_payload,
            "coverage": coverage,
            "group_match": dict(group_meta or {}),
        }


    @classmethod
    def build_top_match_payload_row(
        cls,
        *,
        opp_ctx: Dict[str, Any],
        top_row: Tuple[str, float, float],
    ) -> Dict[str, Any]:
        """Build one top-match row for ranking/justification views."""
        source_opp = dict(opp_ctx or {})
        opportunity_id, domain_score, llm_score = top_row
        return {
            "opportunity_id": str(opportunity_id),
            "opportunity_title": source_opp.get("opportunity_title") or source_opp.get("title"),
            "agency_name": source_opp.get("agency_name") or source_opp.get("agency"),
            "domain_score": cls._to_float(domain_score),
            "llm_score": cls._to_float(llm_score),
        }

    @classmethod
    def build_requirements_from_opportunity_keywords(
        cls,
        *,
        opportunity_keywords: Dict[str, Any],
    ) -> Dict[str, Dict[int, float]]:
        """Convert opportunity specialization keywords to weighted requirement map."""
        spec_items = extract_specializations(dict(opportunity_keywords or {}))
        requirements: Dict[str, Dict[int, float]] = {"application": {}, "research": {}}
        for sec in ("application", "research"):
            for idx, item in enumerate(list(spec_items.get(sec) or [])):
                requirements[sec][int(idx)] = cls._to_float((item or {}).get("w"), default=1.0)
        return requirements

    @classmethod
    def build_member_coverages_from_match_rows(
        cls,
        *,
        match_rows: List[Dict[str, Any]],
    ) -> Dict[int, Dict[str, Dict[int, float]]]:
        """Aggregate per-faculty coverage from stored one-to-one match rows."""
        coverage: Dict[int, Dict[str, Dict[int, float]]] = {}
        for row in list(match_rows or []):
            try:
                fid = int((row or {}).get("faculty_id"))
            except Exception:
                continue

            if fid not in coverage:
                coverage[fid] = {"application": {}, "research": {}}

            covered = dict((row or {}).get("covered") or {})
            for sec in ("application", "research"):
                sec_map = dict(covered.get(sec) or {})
                for k, v in sec_map.items():
                    try:
                        idx = int(k)
                    except Exception:
                        continue
                    prev = cls._to_float(coverage[fid][sec].get(idx), default=0.0)
                    cur = cls._to_float(v, default=0.0)
                    coverage[fid][sec][idx] = max(prev, cur)

        return coverage

    @staticmethod
    def build_matching_inputs_payload(
        *,
        faculty_ids: List[int],
        requirements: Dict[str, Dict[int, float]],
        coverage: Dict[int, Dict[str, Dict[int, float]]],
    ) -> Dict[str, Any]:
        """Build compact input payload used by team matching optimization."""
        return {
            "faculty_ids": [int(x) for x in list(faculty_ids or [])],
            "requirements": {
                "application": dict((requirements or {}).get("application") or {}),
                "research": dict((requirements or {}).get("research") or {}),
            },
            "coverage": dict(coverage or {}),
        }

    @classmethod
    def build_matching_inputs_payload_from_opportunity_and_match_rows(
        cls,
        *,
        opp: Any,
        match_rows: List[Dict[str, Any]],
        build_opportunity_keyword_context: Callable[[Any], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build matching inputs payload from one opportunity entity and stored match rows."""
        opp_kw_ctx = dict(build_opportunity_keyword_context(opp) or {})
        requirements = cls.build_requirements_from_opportunity_keywords(
            opportunity_keywords=dict(opp_kw_ctx.get("keywords") or {}),
        )
        coverage = cls.build_member_coverages_from_match_rows(match_rows=match_rows)
        return cls.build_matching_inputs_payload(
            faculty_ids=sorted(list(coverage.keys())),
            requirements=requirements,
            coverage=coverage,
        )

    @classmethod
    def build_rerank_keyword_inventory(
        cls,
        *,
        faculty_keyword_inventory: Dict[str, Any],
        grant_keyword_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build rerank inventory from prepared faculty/grant keyword inventories."""
        fac_inv = dict(faculty_keyword_inventory or {})

        grants: List[Dict[str, Any]] = []
        for row in list(grant_keyword_rows or []):
            opp_inv = dict((row or {}).get("opportunity_keyword_inventory") or {})
            grants.append(
                {
                    "opportunity_id": opp_inv.get("opportunity_id"),
                    "opportunity_title": opp_inv.get("opportunity_title") or opp_inv.get("title"),
                    "domain_score": cls._to_float((row or {}).get("domain_score"), default=0.0),
                    "llm_score": cls._to_float((row or {}).get("llm_score"), default=0.0),
                    "grant_domain_keywords": list(opp_inv.get("grant_domain_keywords") or []),
                    "grant_specialization_keywords": dict(opp_inv.get("grant_specialization_keywords") or {}),
                }
            )

        return {
            "faculty": {
                "faculty_id": fac_inv.get("faculty_id"),
                "name": fac_inv.get("name"),
                "domain_keywords": list(fac_inv.get("domain_keywords") or []),
                "specialization_keywords": dict(fac_inv.get("specialization_keywords") or {}),
            },
            "grants": grants,
        }

    @classmethod
    def build_top_match_payload_from_entities(
        cls,
        *,
        top_rows: List[Tuple[str, float, float]],
        opportunities: List[Any],
        build_opportunity_matching_context: Callable[[Any], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build top-match payload list from top rows and opportunity entities."""
        by_id = cls._opportunity_map(opportunities)
        out: List[Dict[str, Any]] = []
        for oid, domain_score, llm_score in list(top_rows or []):
            oid_norm = str(oid or "").strip()
            opp = by_id.get(oid_norm)
            if not opp:
                continue
            opp_ctx = dict(build_opportunity_matching_context(opp) or {})
            out.append(
                cls.build_top_match_payload_row(
                    opp_ctx=opp_ctx,
                    top_row=(oid_norm, float(domain_score or 0.0), float(llm_score or 0.0)),
                )
            )
        return out

    @classmethod
    def build_rerank_keyword_inventory_from_entities(
        cls,
        *,
        faculty: Any,
        top_rows: List[Tuple[str, float, float]],
        opportunities: List[Any],
        build_faculty_keyword_inventory: Callable[[Any], Dict[str, Any]],
        build_opportunity_keyword_inventory: Callable[[Any], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build rerank inventory from faculty + top-row opportunities using side inventories."""
        fac_kw_inv = dict(build_faculty_keyword_inventory(faculty) or {})
        by_id = cls._opportunity_map(opportunities)

        grant_keyword_rows: List[Dict[str, Any]] = []
        for oid, domain_score, llm_score in list(top_rows or []):
            oid_norm = str(oid or "").strip()
            opp = by_id.get(oid_norm)
            if not opp:
                continue
            grant_keyword_rows.append(
                {
                    "domain_score": float(domain_score or 0.0),
                    "llm_score": float(llm_score or 0.0),
                    "opportunity_keyword_inventory": dict(build_opportunity_keyword_inventory(opp) or {}),
                }
            )

        return cls.build_rerank_keyword_inventory(
            faculty_keyword_inventory=fac_kw_inv,
            grant_keyword_rows=grant_keyword_rows,
        )

    @classmethod
    def build_rerank_keyword_inventory_for_opportunity_from_entities(
        cls,
        *,
        opp: Any,
        match_rows: List[Dict[str, Any]],
        faculties: List[Any],
        build_opportunity_keyword_inventory: Callable[[Any], Dict[str, Any]],
        build_faculty_keyword_inventory: Callable[[Any], Dict[str, Any]],
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build grant-vs-faculty keyword inventory from entities using side inventories."""
        grant_kw_inv = dict(build_opportunity_keyword_inventory(opp) or {})
        fac_by_id = cls._faculty_map(faculties)

        limit = int(k) if k is not None and int(k) > 0 else None
        source_rows = list(match_rows or [])
        if limit is not None:
            source_rows = source_rows[:limit]

        faculty_keyword_rows: List[Dict[str, Any]] = []
        for row in source_rows:
            try:
                fid = int((row or {}).get("faculty_id"))
            except Exception:
                continue
            fac = fac_by_id.get(fid)
            if fac is None:
                continue
            faculty_keyword_rows.append(
                {
                    "domain_score": cls._to_float((row or {}).get("domain_score"), default=0.0),
                    "llm_score": cls._to_float((row or {}).get("llm_score"), default=0.0),
                    "faculty_keyword_inventory": dict(build_faculty_keyword_inventory(fac) or {}),
                }
            )
        matches: List[Dict[str, Any]] = []
        for row in faculty_keyword_rows:
            fac_inv = dict((row or {}).get("faculty_keyword_inventory") or {})
            matches.append(
                {
                    "domain_score": cls._to_float((row or {}).get("domain_score"), default=0.0),
                    "llm_score": cls._to_float((row or {}).get("llm_score"), default=0.0),
                    "domain_keywords": list(fac_inv.get("domain_keywords") or []),
                    "specialization_keywords": dict(fac_inv.get("specialization_keywords") or {}),
                }
            )

        return {
            "grant": {
                "opportunity_id": grant_kw_inv.get("opportunity_id"),
                "title": grant_kw_inv.get("opportunity_title") or grant_kw_inv.get("title"),
                "grant_domain_keywords": list(grant_kw_inv.get("grant_domain_keywords") or []),
                "grant_specialization_keywords": dict(grant_kw_inv.get("grant_specialization_keywords") or {}),
            },
            "matches": matches,
        }
