"""
TODO: Could probably go to mapper folder
"""

def keywords_for_matching(kw: dict) -> dict:
    kw = kw or {}
    out = {
        "research": {"domain": [], "specialization": []},
        "application": {"domain": [], "specialization": []},
    }

    for sec in ("research", "application"):
        block = kw.get(sec) or {}
        out[sec]["domain"] = block.get("domain") or []

        specs = block.get("specialization") or []
        spec_texts = []
        for s in specs:
            if isinstance(s, str):
                spec_texts.append(s)
            elif isinstance(s, dict) and "t" in s:
                spec_texts.append(s["t"])
        out[sec]["specialization"] = spec_texts

    return out

def specialization_weights(kw: dict) -> dict:
    """
    Returns:
      {
        "research": {text: weight, ...},
        "application": {text: weight, ...}
      }
    """
    kw = kw or {}
    out = {"research": {}, "application": {}}

    for sec in ("research", "application"):
        block = kw.get(sec) or {}
        specs = block.get("specialization") or []
        for s in specs:
            if isinstance(s, dict) and "t" in s:
                out[sec][s["t"]] = float(s.get("w", 1.0))
            elif isinstance(s, str):
                out[sec][s] = 1.0
    return out


def requirements_indexed(kw: dict) -> dict:
    """
    Returns:
      {
        "application": {"0": "...", "1": "..."},
        "research": {"0": "...", "1": "..."}
      }
    """
    kw = kw or {}
    out = {"application": {}, "research": {}}

    for sec in ("application", "research"):
        specs = ((kw.get(sec) or {}).get("specialization") or [])
        for i, s in enumerate(specs):
            if isinstance(s, dict) and "t" in s:
                out[sec][str(i)] = s["t"]
            elif isinstance(s, str):
                out[sec][str(i)] = s

    return out