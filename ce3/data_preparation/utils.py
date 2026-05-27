from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from ce3.data_preparation.llm_runtime import clean_text, normalize_ws


PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class SpecItem:
    item_id: str
    kind: str
    text: str
    meta: Dict[str, Any]


def resolve_path(value: str | Path, *, project_root: Path = PROJECT_ROOT) -> Path:
    p = Path(clean_text(value)).expanduser()
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_jsonl_by_key(path: Path, key: str) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = clean_text(raw)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                k = clean_text(obj.get(key))
                if k:
                    out[k] = obj
    return out


def load_grant_specializations(path: Path, *, max_items: int, seed: int) -> List[SpecItem]:
    db = read_json(path)
    grants = db.get("grants") if isinstance(db, dict) else []
    items: List[SpecItem] = []
    for grant in grants or []:
        if not isinstance(grant, dict):
            continue
        grant_id = clean_text(grant.get("grant_id"))
        specs = grant.get("grant_spec_keywords")
        if not isinstance(specs, list):
            continue
        for idx, text in enumerate(specs):
            norm = normalize_ws(text)
            if not norm:
                continue
            items.append(
                SpecItem(
                    item_id=f"grant:{grant_id}:{idx}",
                    kind="grant",
                    text=norm,
                    meta={
                        "grant_id": grant_id,
                        "grant_spec_idx": int(idx),
                        "grant_keywords": list(grant.get("grant_keywords") or []),
                    },
                )
            )
    rng = random.Random(int(seed))
    rng.shuffle(items)
    if int(max_items) > 0:
        items = items[: int(max_items)]
    return items


def load_faculty_specializations(path: Path, *, max_items: int, seed: int) -> List[SpecItem]:
    db = read_json(path)
    rows = db.get("fac_specs") if isinstance(db, dict) else []
    items: List[SpecItem] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        text = normalize_ws(row.get("text"))
        if not text:
            continue
        fac_id = row.get("fac_id")
        fac_spec_id = row.get("fac_spec_id")
        fac_spec_idx = row.get("fac_spec_idx")
        items.append(
            SpecItem(
                item_id=f"fac:{fac_id}:{fac_spec_id}:{fac_spec_idx}",
                kind="faculty",
                text=text,
                meta={
                    "fac_id": fac_id,
                    "fac_spec_id": fac_spec_id,
                    "fac_spec_idx": fac_spec_idx,
                    "section": row.get("section"),
                },
            )
        )
    rng = random.Random(int(seed) + 13)
    rng.shuffle(items)
    if int(max_items) > 0:
        items = items[: int(max_items)]
    return items


def row_needs_redecompose(row: Dict[str, Any], *, aspects: Sequence[str]) -> bool:
    if not isinstance(row, dict) or not bool(row.get("parse_ok")):
        return True
    decomp = row.get("decomposition")
    if not isinstance(decomp, dict):
        return True
    if any(aspect not in decomp for aspect in aspects):
        return True
    return not any(
        isinstance(decomp.get(aspect), list) and any(normalize_ws(x) for x in decomp.get(aspect, []))
        for aspect in aspects
    )


def refresh_decomposition_cache(
    existing: Dict[str, Dict[str, Any]],
    *,
    refresh_failed_only: bool,
    refresh_all: bool,
    aspects: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    if refresh_all:
        return {}
    if not refresh_failed_only:
        return existing
    return {
        item_id: row
        for item_id, row in existing.items()
        if not row_needs_redecompose(row, aspects=aspects)
    }


def cached_row_matches_item(row: Dict[str, Any], item: SpecItem) -> bool:
    return (
        isinstance(row, dict)
        and clean_text(row.get("item_id")) == clean_text(item.item_id)
        and clean_text(row.get("kind")) == clean_text(item.kind)
        and normalize_ws(row.get("text")) == normalize_ws(item.text)
    )


def drop_stale_cached_rows(
    *,
    existing: Dict[str, Dict[str, Any]],
    items: Sequence[SpecItem],
) -> tuple[Dict[str, Dict[str, Any]], int]:
    out = dict(existing)
    stale = 0
    for item in items:
        row = out.get(item.item_id)
        if row is not None and not cached_row_matches_item(row, item):
            out.pop(item.item_id, None)
            stale += 1
    return out, stale
