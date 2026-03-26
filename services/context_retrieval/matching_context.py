from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from utils.keyword_utils import extract_specializations


class MatchingContextBuilder:
    PROFILE_FIELDS: Dict[str, Tuple[str, ...]] = {
        "group": (
            "grant",
            "team",
            "coverage",
            "group_match",
        ),
        "group_grant": (
            "id",
            "title",
            "agency",
            "summary",
            "keywords",
        ),
        "group_team_member": (
            "faculty_id",
            "name",
            "email",
            "covered",
        ),
        "group_team_member_covered": (
            "application",
            "research",
        ),
        "top_match": (
            "opportunity_id",
            "title",
            "agency",
            "domain_score",
            "llm_score",
        ),
        "inputs": (
            "faculty_ids",
            "requirements",
            "coverage",
        ),
        "inputs_requirements": (
            "application",
            "research",
        ),
    }

    @staticmethod
    def _select_fields(payload: Dict[str, Any], fields: Tuple[str, ...]) -> Dict[str, Any]:
        return {k: payload.get(k) for k in fields}

    def build_matching_retrievable_context(
        self,
        *,
        profile: str,
        opp_ctx: Optional[Dict[str, Any]] = None,
        fac_ctxs: Optional[List[Dict[str, Any]]] = None,
        coverage: Any = None,
        member_coverages: Optional[Dict[int, Dict[str, Dict[int, float]]]] = None,
        group_meta: Optional[Dict[str, Any]] = None,
        top_row: Optional[Tuple[str, float, float]] = None,
        faculty_ids: Optional[List[int]] = None,
        requirements: Optional[Dict[str, Dict[int, float]]] = None,
    ) -> Dict[str, Any]:
        normalized = str(profile or "").strip().lower()

        if normalized == "group":
            source_opp = opp_ctx or {}
            source_member_coverages = member_coverages or {}
            grant_payload = {
                "id": source_opp.get("opportunity_id") or source_opp.get("id"),
                "title": source_opp.get("title"),
                "agency": source_opp.get("agency"),
                "summary": source_opp.get("summary"),
                "keywords": source_opp.get("keywords"),
            }

            team_payload: List[Dict[str, Any]] = []
            for f in fac_ctxs or []:
                faculty_id = f.get("faculty_id") or f.get("id")
                covered = {"application": {}, "research": {}}
                if faculty_id is not None:
                    covered = source_member_coverages.get(int(faculty_id), covered)
                team_payload.append(
                    {
                        "faculty_id": faculty_id,
                        "name": f.get("name"),
                        "email": f.get("email"),
                        "covered": covered,
                    }
                )

            return {
                "grant": grant_payload,
                "team": team_payload,
                "coverage": coverage,
                "group_match": group_meta,
            }

        if normalized == "top_match":
            source_opp = opp_ctx or {}
            opp_id, domain_score, llm_score = top_row or ("", 0.0, 0.0)
            return {
                "opportunity_id": opp_id,
                "title": source_opp.get("title"),
                "agency": source_opp.get("agency"),
                "domain_score": float(domain_score),
                "llm_score": float(llm_score),
            }

        if normalized == "inputs":
            return {
                "faculty_ids": list(faculty_ids or []),
                "requirements": requirements or {"application": {}, "research": {}},
                "coverage": coverage or {},
            }

        raise ValueError(f"Unsupported matching context profile: {profile}")

    def build_matching_context(
        self,
        *,
        profile: str,
        opp_ctx: Optional[Dict[str, Any]] = None,
        fac_ctxs: Optional[List[Dict[str, Any]]] = None,
        coverage: Any = None,
        member_coverages: Optional[Dict[int, Dict[str, Dict[int, float]]]] = None,
        group_meta: Optional[Dict[str, Any]] = None,
        top_row: Optional[Tuple[str, float, float]] = None,
        faculty_ids: Optional[List[int]] = None,
        requirements: Optional[Dict[str, Dict[int, float]]] = None,
    ) -> Dict[str, Any]:
        normalized = str(profile or "").strip().lower()
        fields = self.PROFILE_FIELDS.get(normalized)
        if not fields:
            raise ValueError(f"Unsupported matching context profile: {profile}")

        full = self.build_matching_retrievable_context(
            profile=normalized,
            opp_ctx=opp_ctx,
            fac_ctxs=fac_ctxs,
            coverage=coverage,
            member_coverages=member_coverages,
            group_meta=group_meta,
            top_row=top_row,
            faculty_ids=faculty_ids,
            requirements=requirements,
        )
        context = self._select_fields(full, fields)

        if normalized == "group":
            grant_fields = self.PROFILE_FIELDS.get("group_grant", ())
            member_fields = self.PROFILE_FIELDS.get("group_team_member", ())
            covered_fields = self.PROFILE_FIELDS.get("group_team_member_covered", ())
            context["grant"] = self._select_fields(dict(full.get("grant") or {}), grant_fields)
            filtered_team: List[Dict[str, Any]] = []
            for member in list(full.get("team") or []):
                m = self._select_fields(dict(member or {}), member_fields)
                m["covered"] = self._select_fields(dict(m.get("covered") or {}), covered_fields)
                filtered_team.append(m)
            context["team"] = filtered_team
            if not full.get("group_match"):
                context.pop("group_match", None)

        if normalized == "inputs":
            req_fields = self.PROFILE_FIELDS.get("inputs_requirements", ())
            context["requirements"] = self._select_fields(dict(full.get("requirements") or {}), req_fields)

        return context

    # ====================
    # Group Matching Context
    # ====================
    def build_group_matching_context(
        self,
        *,
        opp_ctx: Dict[str, Any],
        fac_ctxs: List[Dict[str, Any]],
        coverage: Any,
        member_coverages: Optional[Dict[int, Dict[str, Dict[int, float]]]] = None,
        group_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.build_matching_context(
            profile="group",
            opp_ctx=opp_ctx,
            fac_ctxs=fac_ctxs,
            coverage=coverage,
            member_coverages=member_coverages,
            group_meta=group_meta,
        )

    # ====================
    # Top Match Context
    # ====================
    def build_top_match_payload(
        self,
        *,
        sess,
        top_rows: List[Tuple[str, float, float]],
    ) -> List[Dict[str, Any]]:
        opp_dao = OpportunityDAO(sess)
        out: List[Dict[str, Any]] = []
        for row in top_rows or []:
            opp_id, domain_score, llm_score = row
            opp_ctx = opp_dao.read_opportunity_context(opp_id) or {}
            out.append(
                self.build_matching_context(
                    profile="top_match",
                    opp_ctx=opp_ctx,
                    top_row=(opp_id, domain_score, llm_score),
                )
            )
        return out

    # ====================
    # Input / Coverage Context
    # ====================
    def build_inputs_for_opportunity(
        self,
        *,
        sess,
        opportunity_id: str,
        limit_rows: int = 500,
    ) -> Tuple[List[int], Dict[str, Dict[int, float]], Dict[int, Dict[str, Dict[int, float]]]]:
        opp_dao = OpportunityDAO(sess)

        opps = opp_dao.read_opportunities_by_ids_with_relations([opportunity_id])
        if not opps:
            raise ValueError(f"Opportunity not found: {opportunity_id}")
        opp = opps[0]

        kw_raw = getattr(opp.keyword, "keywords", {}) or {}
        spec_items = extract_specializations(kw_raw)

        requirements: Dict[str, Dict[int, float]] = {"application": {}, "research": {}}
        for sec in ("application", "research"):
            for idx, item in enumerate(spec_items[sec]):
                requirements[sec][idx] = float(item.get("w", 1.0))

        coverage = self.build_member_coverages_for_opportunity(
            sess=sess,
            opportunity_id=opportunity_id,
            limit_rows=limit_rows,
        )
        if not coverage:
            raise ValueError("No match rows found.")

        ctx = self.build_matching_context(
            profile="inputs",
            faculty_ids=sorted(coverage.keys()),
            requirements=requirements,
            coverage=coverage,
        )
        return (
            list(ctx.get("faculty_ids") or []),
            dict(ctx.get("requirements") or {"application": {}, "research": {}}),
            dict(ctx.get("coverage") or {}),
        )

    def build_member_coverages_for_opportunity(
        self,
        *,
        sess,
        opportunity_id: str,
        limit_rows: int = 500,
    ) -> Dict[int, Dict[str, Dict[int, float]]]:
        match_dao = MatchDAO(sess)
        match_rows = match_dao.list_matches_for_opportunity(opportunity_id, limit=limit_rows) or []
        if not match_rows:
            return {}

        coverage: Dict[int, Dict[str, Dict[int, float]]] = {}
        for row in match_rows:
            fid = int(row["faculty_id"])
            if fid not in coverage:
                coverage[fid] = {"application": {}, "research": {}}
            cov = row.get("covered") or {}
            for sec in ("application", "research"):
                sec_map = cov.get(sec) or {}
                for k, v in sec_map.items():
                    try:
                        idx = int(k)
                        cval = float(v)
                    except Exception:
                        continue
                    prev = coverage[fid][sec].get(idx, 0.0)
                    coverage[fid][sec][idx] = max(prev, cval)

        return coverage
