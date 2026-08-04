"""Export grant and faculty specialization keywords from PostgreSQL.

The CE5 prefilter scores keyword text directly with a cross encoder, so this
export intentionally contains no embedding metadata or vector files.  Research
and application specializations are merged into one deduplicated keyword list
for each grant or faculty owner.

Run from the repository root with::

    python ce5/data_preparation/export_specialization_keywords.py

By default, the two JSON files are written to ``ce5/dataset/source``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sqlalchemy import text


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db.db_conn import SessionLocal  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "dataset" / "source"
DEFAULT_GRANT_OUTPUT = OUTPUT_DIR / "grant_specialization_keyword_db.json"
DEFAULT_FACULTY_OUTPUT = OUTPUT_DIR / "faculty_specialization_keywords_db.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class KeywordRecord:
    owner_id: str | int
    keywords: tuple[str, ...]

    @property
    def keyword_count(self) -> int:
        return len(self.keywords)


def _clean_text(value: Any) -> str:
    """Return a keyword as one clean line of text."""

    return " ".join(str(value or "").split()).strip()


def _clean_owner_id(value: Any) -> str | int:
    """Keep numeric database IDs numeric and normalize textual IDs."""

    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return _clean_text(value)


def _specialization_text(item: Any) -> str:
    """Support plain strings and the project's weighted keyword objects."""

    if isinstance(item, Mapping):
        for key in ("t", "text", "value"):
            cleaned = _clean_text(item.get(key))
            if cleaned:
                return cleaned
        return ""
    return _clean_text(item)


def _extract_section(keywords: Any, section: str) -> tuple[str, ...]:
    if not isinstance(keywords, Mapping):
        return ()

    section_data = keywords.get(section)
    if not isinstance(section_data, Mapping):
        return ()

    raw_items = section_data.get("specialization") or []
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        raw_items = [raw_items]

    output: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        keyword = _specialization_text(item)
        dedupe_key = keyword.casefold()
        if not keyword or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        output.append(keyword)
    return tuple(output)


def _records_from_rows(rows: Iterable[Mapping[str, Any]]) -> list[KeywordRecord]:
    records: list[KeywordRecord] = []
    for row in rows:
        owner_id = _clean_owner_id(row.get("owner_id"))
        keywords = row.get("keywords") or {}
        combined: list[str] = []
        seen: set[str] = set()
        for section in ("research", "application"):
            for keyword in _extract_section(keywords, section):
                dedupe_key = keyword.casefold()
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                combined.append(keyword)
        record = KeywordRecord(owner_id=owner_id, keywords=tuple(combined))
        if owner_id and record.keyword_count:
            records.append(record)
    return records


def fetch_grant_specializations(limit: int | None = None) -> list[KeywordRecord]:
    query = """
        SELECT opportunity_id AS owner_id, keywords
        FROM opportunity_keywords
        WHERE keywords IS NOT NULL
        ORDER BY opportunity_id ASC
    """
    params: dict[str, int] = {}
    if limit is not None:
        query += " LIMIT :limit"
        params["limit"] = limit

    with SessionLocal() as session:
        rows = session.execute(text(query), params).mappings().all()
    return _records_from_rows(rows)


def fetch_faculty_specializations(limit: int | None = None) -> list[KeywordRecord]:
    query = """
        SELECT faculty_id AS owner_id, keywords
        FROM faculty_keywords
        WHERE keywords IS NOT NULL
        ORDER BY faculty_id ASC
    """
    params: dict[str, int] = {}
    if limit is not None:
        query += " LIMIT :limit"
        params["limit"] = limit

    with SessionLocal() as session:
        rows = session.execute(text(query), params).mappings().all()
    return _records_from_rows(rows)


def write_keyword_file(
    path: Path,
    records: Sequence[KeywordRecord],
    *,
    id_label: str,
    collection_name: str,
) -> tuple[int, int]:
    """Write the keyword-only JSON contract consumed by CE5 prefiltering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        collection_name: [
            {
                id_label: record.owner_id,
                "specialization_keywords": list(record.keywords),
            }
            for record in records
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    keyword_count = sum(record.keyword_count for record in records)
    return len(records), keyword_count


def _positive_limit(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("limit must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export specialization keyword lists from PostgreSQL."
    )
    parser.add_argument("--grant-output", type=Path, default=DEFAULT_GRANT_OUTPUT)
    parser.add_argument("--faculty-output", type=Path, default=DEFAULT_FACULTY_OUTPUT)
    parser.add_argument(
        "--limit",
        type=_positive_limit,
        default=None,
        help="Optional maximum number of owners to fetch from each keyword table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grant_records = fetch_grant_specializations(args.limit)
    faculty_records = fetch_faculty_specializations(args.limit)

    grant_owners, grant_keywords = write_keyword_file(
        args.grant_output,
        grant_records,
        id_label="grant_id",
        collection_name="grants",
    )
    faculty_owners, faculty_keywords = write_keyword_file(
        args.faculty_output,
        faculty_records,
        id_label="faculty_id",
        collection_name="faculty",
    )

    print(
        f"Wrote {grant_keywords} grant keywords for {grant_owners} grants to "
        f"{args.grant_output}"
    )
    print(
        f"Wrote {faculty_keywords} faculty keywords for {faculty_owners} faculty to "
        f"{args.faculty_output}"
    )


if __name__ == "__main__":
    main()
