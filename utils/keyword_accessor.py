def extract_specializations(kw: dict) -> dict:
    """
    Returns:
      {
        "research": [{"t": str, "w": float}, ...],
        "application": [{"t": str, "w": float}, ...]
      }
    """
    kw = kw or {}
    out = {"research": [], "application": []}

    for sec in ("research", "application"):
        specs = (kw.get(sec) or {}).get("specialization") or []
        for s in specs:
            if isinstance(s, dict) and "t" in s:
                out[sec].append({
                    "t": str(s["t"]),
                    "w": float(s.get("w", 1.0)),
                })
            elif isinstance(s, str):
                out[sec].append({
                    "t": str(s),
                    "w": 1.0,
                })

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

def specialization_weights(kw: dict) -> dict:
    specs = extract_specializations(kw)
    return {
        sec: {s["t"]: s["w"] for s in specs[sec]}
        for sec in ("research", "application")
    }

def requirements_indexed(kw: dict) -> dict:
    specs = extract_specializations(kw)
    out = {"application": {}, "research": {}}

    for sec in ("application", "research"):
        for i, s in enumerate(specs[sec]):
            out[sec][str(i)] = s["t"]

    return out


