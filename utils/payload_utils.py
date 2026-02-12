from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def build_base_payload(
    *,
    opp_ctx: Dict[str, Any],
    fac_ctxs: List[Dict[str, Any]],
    coverage: Any,
    member_coverages: Optional[Dict[int, Dict[str, Dict[int, float]]]] = None,
    group_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Base payload sent to split writer steps."""
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
                #"keywords": f.get("keywords"),
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


def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def extract_requirement_specs(opp_ctx: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Extract requirement text/weight by section/index from opportunity keyword payload."""
    out: Dict[str, Dict[int, Dict[str, Any]]] = {"application": {}, "research": {}}
    kw = (opp_ctx.get("keywords") or {}) if isinstance(opp_ctx, dict) else {}

    for sec in ("application", "research"):
        sec_obj = kw.get(sec) if isinstance(kw, dict) else None
        if not isinstance(sec_obj, dict):
            continue
        specs = sec_obj.get("specialization")
        if not isinstance(specs, list):
            continue
        for i, item in enumerate(specs):
            if not isinstance(item, dict):
                continue
            out[sec][i] = {
                "text": str(item.get("t") or f"{sec} requirement {i}"),
                "weight": float(item.get("w") or 0.0),
            }
    return out
