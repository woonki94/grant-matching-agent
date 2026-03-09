# Faculty GraphRAG Prototype (Neo4j)

This folder contains a faculty graph sync prototype that mirrors the grant graph flow.

Shared modules under `tmp/`:
- `tmp/neo4j_common.py`
- `tmp/neo4j_schema.py`

## Files
- `faculty_sync_neo4j.py`: sync faculty rows to Neo4j
- `faculty_sync_all_with_extraction.py`: full pipeline (extract additional-info text to S3 + sync all faculty)

## Synced Faculty Data
- Basic faculty info (`name`, `position`, `organization`, `biography`, etc.)
- Additional info rows + extracted text from `content_path`
- Publications (`title`, `abstract`, `year`) without OpenAlex config IDs
- Keywords with weighted relations:
  - `HAS_RESEARCH_DOMAIN`
  - `HAS_RESEARCH_SPECIALIZATION`
  - `HAS_APPLICATION_DOMAIN`
  - `HAS_APPLICATION_SPECIALIZATION`

## Usage
Run shared schema:

```bash
python tmp/neo4j_schema.py
```

Sync one faculty:

```bash
python tmp/facult_graph_rag/faculty_sync_neo4j.py --email "alan.fern@oregonstate.edu" --verify-email "alan.fern@oregonstate.edu"
```

Sync all faculty with existing extracted text:

```bash
python tmp/facult_graph_rag/faculty_sync_neo4j.py --all
```

Full faculty pipeline (extract + sync all):

```bash
python tmp/facult_graph_rag/faculty_sync_all_with_extraction.py --retry-failed-extraction
```

Force re-extract all additional-info content, then sync:

```bash
python tmp/facult_graph_rag/faculty_sync_all_with_extraction.py --force-reextract
```
