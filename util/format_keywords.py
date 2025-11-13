from typing import Dict, Any, List, Set, Tuple

NEW_EMPTY_SCHEMA: Dict[str, Any] = {
    "research":    {"domain": [], "specialization": []},
    "application": {"domain": [], "specialization": []},
}

# path tuples
LIST_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("research", "domain"),
    ("research", "specialization"),
    ("application", "domain"),
    ("application", "specialization"),
)

def _empty_schema() -> Dict[str, Any]:
    # deep-ish copy so inner lists aren’t shared
    return {
        "research": {"domain": [], "specialization": []},
        "application": {"domain": [], "specialization": []},
    }

def _normalize_to_new_schema(items: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce legacy shapes to the new structured schema.
    - Legacy: {"keywords": ["a","b",...]}  -> maps everything into specialization lists
    - New:    {"research": {...}, "application": {...}} (pass-through with clean/dedupe)
    """
    if not isinstance(items, dict):
        return _empty_schema()

    # New schema: detect by presence of required keys
    if all(k in items for k in NEW_EMPTY_SCHEMA.keys()):
        out = {
            "research": {
                "domain": list(items.get("research", {}).get("domain", []) or []),
                "specialization": list(items.get("research", {}).get("specialization", []) or []),
            },
            "application": {
                "domain": list(items.get("application", {}).get("domain", []) or []),
                "specialization": list(items.get("application", {}).get("specialization", []) or []),
            },
        }
        # strip/clean and dedupe
        for section, key in LIST_FIELDS:
            vals = [v.strip() for v in out[section][key] if isinstance(v, str) and v.strip()]
            seen: Set[str] = set()
            dedup: List[str] = []
            for v in vals:
                k = v.lower()
                if k not in seen:
                    seen.add(k)
                    dedup.append(v)
            out[section][key] = dedup
        return out

    # Legacy schema with flat array
    legacy = items.get("keywords")
    if isinstance(legacy, list):
        vals = [str(v).strip() for v in legacy if isinstance(v, str)]
        vals = [v for v in vals if v]
        # Map legacy keywords into a reasonable default bucket (kept as-is)
        return {
            "research": {
                "domain": [],
                "specialization": sorted(set(vals), key=str.lower),
            },
            "application": {
                "domain": [],
                "specialization": sorted(set(vals), key=str.lower),
            },
        }

    # Anything else -> empty schema
    return _empty_schema()


def _count_total_strings(struct_obj: Dict[str, Any]) -> int:
    s: Set[str] = set()
    for section, key in LIST_FIELDS:
        for v in (struct_obj.get(section, {}).get(key) or []):
            if isinstance(v, str) and v.strip():
                s.add(v.strip().lower())
    return len(s)