from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


def coerce_keyword_sections(kw_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure research/application sections are dicts, parsing JSON-strings when needed."""
    out = dict(kw_dict or {})
    for section in ("research", "application"):
        if isinstance(out.get(section), str):
            out[section] = json.loads(out[section])
    return out


def apply_weighted_specializations(*, keywords: Dict[str, Any], weighted: Any) -> Dict[str, Any]:
    out = dict(keywords or {})
    out["research"] = dict(out.get("research") or {})
    out["application"] = dict(out.get("application") or {})
    out["research"]["specialization"] = [x.model_dump() for x in (getattr(weighted, "research", None) or [])]
    out["application"]["specialization"] = [x.model_dump() for x in (getattr(weighted, "application", None) or [])]
    return out


def extract_domains_from_keywords(kw: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    r = (kw.get("research") or {}).get("domain") or []
    a = (kw.get("application") or {}).get("domain") or []
    return list(r), list(a)


def extract_specializations(kw: dict) -> dict:
    kw = kw or {}
    out = {"research": [], "application": []}

    for sec in ("research", "application"):
        specs = (kw.get(sec) or {}).get("specialization") or []
        for s in specs:
            if isinstance(s, dict) and "t" in s:
                out[sec].append(
                    {
                        "t": str(s["t"]),
                        "w": float(s.get("w", 1.0)),
                    }
                )
            elif isinstance(s, str):
                out[sec].append(
                    {
                        "t": str(s),
                        "w": 1.0,
                    }
                )

    return out


def keywords_for_matching(kw: dict) -> dict:
    specs = extract_specializations(kw)
    return {
        sec: {
            "domain": (kw.get(sec) or {}).get("domain") or [],
            "specialization": [s["t"] for s in specs[sec]],
        }
        for sec in ("research", "application")
    }


def requirements_indexed(kw: dict) -> dict:
    specs = extract_specializations(kw)
    out = {"application": {}, "research": {}}

    for sec in ("application", "research"):
        for i, s in enumerate(specs[sec]):
            out[sec][str(i)] = s["t"]

    return out



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
