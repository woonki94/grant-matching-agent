from __future__ import annotations

import re
from typing import Any

_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def sanitize_for_postgres(obj: Any) -> Any:
    """Recursively strip control chars Postgres rejects in JSON/text payloads."""
    if isinstance(obj, str):
        obj = obj.replace("\x00", "")
        return _CTRL_RE.sub("", obj)
    if isinstance(obj, list):
        return [sanitize_for_postgres(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize_for_postgres(v) for k, v in obj.items()}
    return obj
