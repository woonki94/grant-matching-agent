from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional

SUPPORTED_SCHEMES = ("bolt://", "bolt+s://", "bolt+ssc://", "neo4j://", "neo4j+s://", "neo4j+ssc://")


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    username: str
    password: str
    database: str


def load_dotenv_if_present() -> None:
    """
    Lightweight .env reader.
    Existing environment variables are preserved.
    """
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def read_neo4j_settings(
    *,
    uri: str | None = None,
    username: str | None = None,
    password: str | None = None,
    database: str | None = None,
) -> Neo4jSettings:
    resolved_uri = (uri or os.getenv("NEO4J_URI") or "").strip()
    resolved_user = (username or os.getenv("NEO4J_USERNAME") or "").strip()
    resolved_password = (password or os.getenv("NEO4J_PASSWORD") or "").strip()
    resolved_db = (database or os.getenv("NEO4J_DATABASE") or "neo4j").strip()

    if not resolved_uri:
        raise ValueError("Missing Neo4j URI. Set NEO4J_URI or pass --uri.")
    if not resolved_uri.startswith(SUPPORTED_SCHEMES):
        raise ValueError(
            "Unsupported URI scheme for NEO4J_URI. "
            f"Expected one of: {', '.join(SUPPORTED_SCHEMES)}"
        )
    if not resolved_user:
        raise ValueError("Missing Neo4j username. Set NEO4J_USERNAME or pass --username.")
    if not resolved_password:
        raise ValueError("Missing Neo4j password. Set NEO4J_PASSWORD or pass --password.")
    if not resolved_db:
        raise ValueError("Missing Neo4j database. Set NEO4J_DATABASE or pass --database.")

    return Neo4jSettings(
        uri=resolved_uri,
        username=resolved_user,
        password=resolved_password,
        database=resolved_db,
    )


def safe_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def coerce_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        value = [value]
    if not isinstance(value, Iterable):
        return []

    out: List[str] = []
    seen = set()
    for item in value:
        token = safe_text(item)
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(token)
    return out


def coerce_iso_datetime(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return safe_text(value)


def coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def json_ready(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "items"):
        try:
            return {str(k): json_ready(v) for k, v in value.items()}
        except Exception:
            pass

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    return value
