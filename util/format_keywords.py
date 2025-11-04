from typing import Dict, Any, List, Set

NEW_EMPTY_SCHEMA = {
    "area": "",
    "discipline": "",
    "application_domain": [],
    "research_area": [],
    "methods": [],
    "models": [],
}

LIST_FIELDS = ("application_domain", "research_area", "methods", "models")

def _normalize_to_new_schema(items: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce legacy shapes to the new structured schema.
    - Legacy: {"keywords": ["a","b",...]}  -> maps everything into research_area by default
    - New:    {"area": "...", "discipline": "...", "application_domain": [...], ...} (pass-through)
    """
    if not isinstance(items, dict):
        return NEW_EMPTY_SCHEMA.copy()

    # New schema: detect by presence of required keys
    if all(k in items for k in NEW_EMPTY_SCHEMA.keys()):
        # Shallow copy + type guards
        out = {
            "area": str(items.get("area", "") or ""),
            "discipline": str(items.get("discipline", "") or ""),
            "application_domain": list(items.get("application_domain", []) or []),
            "research_area": list(items.get("research_area", []) or []),
            "methods": list(items.get("methods", []) or []),
            "models": list(items.get("models", []) or []),
        }
        # strip/clean and dedupe
        for f in LIST_FIELDS:
            vals = [v.strip() for v in out[f] if isinstance(v, str) and v.strip()]
            seen: Set[str] = set()
            dedup: List[str] = []
            for v in vals:
                k = v.lower()
                if k not in seen:
                    seen.add(k)
                    dedup.append(v)
            out[f] = dedup
        return out

    # Legacy schema with flat array
    legacy = items.get("keywords")
    if isinstance(legacy, list):
        vals = [str(v).strip() for v in legacy if isinstance(v, (str,))]
        vals = [v for v in vals if v]
        # Map legacy keywords into a reasonable default bucket
        return {
            "area": "",
            "discipline": "",
            "application_domain": [],
            "research_area": sorted(set(vals), key=str.lower),  # default bucket
            "methods": [],
            "models": [],
        }

    # Anything else -> empty schema
    return NEW_EMPTY_SCHEMA.copy()


def _count_total_strings(struct_obj: Dict[str, Any]) -> int:
    s: Set[str] = set()
    for f in LIST_FIELDS:
        for v in struct_obj.get(f, []) or []:
            if isinstance(v, str) and v.strip():
                s.add(v.strip().lower())
    return len(s)