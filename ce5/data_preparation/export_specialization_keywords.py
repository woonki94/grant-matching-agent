"""Export specialization keywords and their existing Postgres embeddings.

The readable keyword metadata stays in JSON, while the large float vectors are
stored in a compressed NumPy file.  Each owner's ``embedding_rows`` entries
point to that owner's available vectors in the NPZ ``vectors`` array::

    owner_vectors = vectors[owner["embedding_rows"]]

The current database stores owner-level research/application domain vectors,
not one vector per specialization.  Section labels are deliberately omitted
here; retrieval can compare the small vector sets with maximum cosine.

Run from the repository root with::

    python ce5/data_preparation/export_specialization_keywords.py

By default, all four files are written to ``ce5/dataset/source``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sqlalchemy import bindparam, text


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import settings  # noqa: E402
from db.db_conn import SessionLocal  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "dataset" / "source"
DEFAULT_GRANT_OUTPUT = OUTPUT_DIR / "grant_specialization_keyword_db.json"
DEFAULT_FACULTY_OUTPUT = OUTPUT_DIR / "faculty_specialization_keywords_db.json"
DEFAULT_GRANT_EMBEDDING_OUTPUT = OUTPUT_DIR / "grant_specialization_vectors.npz"
DEFAULT_FACULTY_EMBEDDING_OUTPUT = OUTPUT_DIR / "faculty_specialization_vectors.npz"


@dataclass(frozen=True)
class KeywordRecord:
    owner_id: str | int
    keywords: tuple[str, ...]

    @property
    def keyword_count(self) -> int:
        return len(self.keywords)


@dataclass(frozen=True)
class EmbeddingRecord:
    owner_id: str | int
    vector: tuple[float, ...]


@dataclass(frozen=True)
class EmbeddingMapping:
    rows_by_owner: tuple[tuple[int, ...], ...]
    vectors: np.ndarray
    missing_count: int


def _clean_text(value: Any) -> str:
    """Return a keyword as one clean line of text."""
    return " ".join(str(value or "").split()).strip()


def _clean_owner_id(value: Any) -> str | int:
    """Keep numeric database IDs numeric and normalize textual IDs."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return _clean_text(value)


def _owner_lookup_key(value: Any) -> str:
    return str(_clean_owner_id(value))


def _specialization_text(item: Any) -> str:
    """Support both plain strings and the project's weighted keyword objects."""
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


def _coerce_vector(value: Any) -> tuple[float, ...]:
    if isinstance(value, np.ndarray):
        array = value.astype(np.float32, copy=False).reshape(-1)
    elif isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return ()
    elif isinstance(value, str):
        raw = value.strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        array = np.fromstring(raw, sep=",", dtype=np.float32)
    else:
        return ()

    if array.size == 0 or not np.isfinite(array).all():
        return ()
    return tuple(float(item) for item in array)


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


def _fetch_embeddings(
    *,
    table_name: str,
    owner_column: str,
    owner_ids: Sequence[str | int],
    embedding_model: str,
) -> list[EmbeddingRecord]:
    if not owner_ids:
        return []

    # Table and column names are internal constants, not user input.
    query = text(
        f"""
        SELECT
            {owner_column} AS owner_id,
            research_domain_vec,
            application_domain_vec
        FROM {table_name}
        WHERE model = :embedding_model
          AND {owner_column} IN :owner_ids
        ORDER BY {owner_column} ASC
        """
    ).bindparams(bindparam("owner_ids", expanding=True))

    with SessionLocal() as session:
        rows = session.execute(
            query,
            {"embedding_model": embedding_model, "owner_ids": list(owner_ids)},
        ).mappings().all()

    output: list[EmbeddingRecord] = []
    for row in rows:
        owner_id = _clean_owner_id(row.get("owner_id"))
        if not owner_id:
            continue
        for vector_column in ("research_domain_vec", "application_domain_vec"):
            vector = _coerce_vector(row.get(vector_column))
            if not vector:
                continue
            output.append(
                EmbeddingRecord(
                    owner_id=owner_id,
                    vector=vector,
                )
            )
    return output


def fetch_grant_embeddings(
    records: Sequence[KeywordRecord], embedding_model: str
) -> list[EmbeddingRecord]:
    return _fetch_embeddings(
        table_name="opportunity_keyword_embedding",
        owner_column="opportunity_id",
        owner_ids=[record.owner_id for record in records],
        embedding_model=embedding_model,
    )


def fetch_faculty_embeddings(
    records: Sequence[KeywordRecord], embedding_model: str
) -> list[EmbeddingRecord]:
    return _fetch_embeddings(
        table_name="faculty_keyword_embedding",
        owner_column="faculty_id",
        owner_ids=[record.owner_id for record in records],
        embedding_model=embedding_model,
    )


def map_embeddings(
    records: Sequence[KeywordRecord],
    embeddings: Sequence[EmbeddingRecord],
) -> EmbeddingMapping:
    if not embeddings:
        raise ValueError("No valid specialization embeddings were found for the selected model.")

    dimensions = {len(record.vector) for record in embeddings}
    if len(dimensions) != 1:
        raise ValueError(f"Embedding rows contain inconsistent dimensions: {sorted(dimensions)}")
    dimension = dimensions.pop()

    lookup: dict[str, list[tuple[float, ...]]] = {}
    for record in embeddings:
        lookup.setdefault(_owner_lookup_key(record.owner_id), []).append(record.vector)

    vectors: list[tuple[float, ...]] = []
    rows_by_owner: list[tuple[int, ...]] = []
    missing_count = 0
    for owner in records:
        owner_rows: list[int] = []
        owner_vectors = lookup.get(_owner_lookup_key(owner.owner_id), [])
        if not owner_vectors:
            missing_count += 1
        for vector in owner_vectors:
            owner_rows.append(len(vectors))
            vectors.append(vector)
        rows_by_owner.append(tuple(owner_rows))

    if not vectors:
        raise ValueError(
            "Embedding rows were found, but none matched the exported specialization keywords."
        )

    matrix = np.asarray(vectors, dtype=np.float32).reshape(-1, dimension)
    return EmbeddingMapping(
        rows_by_owner=tuple(rows_by_owner),
        vectors=matrix,
        missing_count=missing_count,
    )


def write_vector_file(path: Path, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, vectors=vectors)


def write_keyword_file(
    path: Path,
    records: Sequence[KeywordRecord],
    mapping: EmbeddingMapping,
    *,
    embedding_path: Path,
    embedding_model: str,
    id_label: str,
    collection_name: str,
) -> tuple[int, int]:
    """Write keyword metadata whose row indices reference the NPZ matrix."""
    if len(records) != len(mapping.rows_by_owner):
        raise ValueError("Keyword records and embedding row mappings are not aligned.")

    path.parent.mkdir(parents=True, exist_ok=True)
    embedding_reference = os.path.relpath(embedding_path, start=path.parent)
    payload = {
        "schema_version": 2,
        "embedding_file": embedding_reference,
        "embedding_array": "vectors",
        "embedding_model": embedding_model,
        "embedding_dimension": int(mapping.vectors.shape[1]),
        "embedding_scope": "owner_domain_vectors",
        collection_name: [
            {
                id_label: record.owner_id,
                "specialization_keywords": list(record.keywords),
                "embedding_rows": list(owner_rows),
            }
            for record, owner_rows in zip(records, mapping.rows_by_owner)
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
        description="Export specialization keywords and existing Postgres embeddings."
    )
    parser.add_argument("--grant-output", type=Path, default=DEFAULT_GRANT_OUTPUT)
    parser.add_argument("--faculty-output", type=Path, default=DEFAULT_FACULTY_OUTPUT)
    parser.add_argument(
        "--grant-embedding-output",
        type=Path,
        default=DEFAULT_GRANT_EMBEDDING_OUTPUT,
    )
    parser.add_argument(
        "--faculty-embedding-output",
        type=Path,
        default=DEFAULT_FACULTY_EMBEDDING_OUTPUT,
    )
    parser.add_argument(
        "--embedding-model",
        default=settings.bedrock_embed_model_id,
        help="Embedding model stored in the specialization embedding tables.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_limit,
        default=None,
        help="Optional maximum number of owners to fetch from each keyword table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding_model = _clean_text(args.embedding_model)
    if not embedding_model:
        raise ValueError("An embedding model is required.")

    grant_records = fetch_grant_specializations(args.limit)
    faculty_records = fetch_faculty_specializations(args.limit)
    grant_mapping = map_embeddings(
        grant_records,
        fetch_grant_embeddings(grant_records, embedding_model),
    )
    faculty_mapping = map_embeddings(
        faculty_records,
        fetch_faculty_embeddings(faculty_records, embedding_model),
    )
    if grant_mapping.vectors.shape[1] != faculty_mapping.vectors.shape[1]:
        raise ValueError(
            "Grant and faculty embedding dimensions differ: "
            f"{grant_mapping.vectors.shape[1]} != {faculty_mapping.vectors.shape[1]}"
        )

    write_vector_file(args.grant_embedding_output, grant_mapping.vectors)
    write_vector_file(args.faculty_embedding_output, faculty_mapping.vectors)
    grant_owners, grant_keywords = write_keyword_file(
        args.grant_output,
        grant_records,
        grant_mapping,
        embedding_path=args.grant_embedding_output,
        embedding_model=embedding_model,
        id_label="grant_id",
        collection_name="grants",
    )
    faculty_owners, faculty_keywords = write_keyword_file(
        args.faculty_output,
        faculty_records,
        faculty_mapping,
        embedding_path=args.faculty_embedding_output,
        embedding_model=embedding_model,
        id_label="faculty_id",
        collection_name="faculty",
    )

    print(
        f"Wrote {grant_keywords} grant keywords for {grant_owners} grants and "
        f"{grant_mapping.vectors.shape[0]} vectors to {args.grant_output} and "
        f"{args.grant_embedding_output}"
    )
    print(
        f"Wrote {faculty_keywords} faculty keywords for {faculty_owners} faculty and "
        f"{faculty_mapping.vectors.shape[0]} vectors to {args.faculty_output} and "
        f"{args.faculty_embedding_output}"
    )
    if grant_mapping.missing_count or faculty_mapping.missing_count:
        print(
            "Warning: owners without embedding rows: "
            f"grants={grant_mapping.missing_count}, faculty={faculty_mapping.missing_count}"
        )


if __name__ == "__main__":
    main()
