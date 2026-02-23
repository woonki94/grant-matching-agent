from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from utils.keyword_utils import extract_specializations


class MatchingContextBuilder:
    def build_group_matching_context(
        self,
        *,
        opp_ctx: Dict[str, Any],
        fac_ctxs: List[Dict[str, Any]],
        coverage: Any,
        member_coverages: Optional[Dict[int, Dict[str, Dict[int, float]]]] = None,
        group_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        member_coverages = member_coverages or {}
        payload: Dict[str, Any] = {
            "grant": {
                "id": opp_ctx.get("opportunity_id") or opp_ctx.get("id"),
                "title": opp_ctx.get("title"),
                "agency": opp_ctx.get("agency"),
                "summary": opp_ctx.get("summary"),
                "keywords": opp_ctx.get("keywords"),
            },
            "team": [
                {
                    "faculty_id": f.get("faculty_id") or f.get("id"),
                    "name": f.get("name"),
                    "email": f.get("email"),
                    "covered": member_coverages.get(
                        int(f.get("faculty_id") or f.get("id")),
                        {"application": {}, "research": {}},
                    )
                    if (f.get("faculty_id") or f.get("id")) is not None
                    else {"application": {}, "research": {}},
                }
                for f in fac_ctxs
            ],
            "coverage": coverage,
        }
        if group_meta:
            payload["group_match"] = group_meta
        return payload

    def build_top_match_payload(
        self,
        *,
        sess,
        top_rows: List[Tuple[str, float, float]],
    ) -> List[Dict[str, Any]]:
        opp_dao = OpportunityDAO(sess)
        out: List[Dict[str, Any]] = []
        for opp_id, domain_score, llm_score in top_rows or []:
            opp_ctx = opp_dao.read_opportunity_context(opp_id) or {}
            out.append(
                {
                    "opportunity_id": opp_id,
                    "title": opp_ctx.get("title"),
                    "agency": opp_ctx.get("agency"),
                    "domain_score": float(domain_score),
                    "llm_score": float(llm_score),
                }
            )
        return out

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

        faculty_ids = sorted(coverage.keys())
        return faculty_ids, requirements, coverage

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
