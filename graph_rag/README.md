# GraphRAG (Production Path)

This folder is the non-`tmp` GraphRAG path.

## Scope (current)

- Grant graph sync with extracted text.
- Faculty graph sync with extracted text and publication abstracts.
- Clean text before writing to Neo4j and before embedding.
- Domain keywords are filter-only (no edge weights).
- Specialization keywords keep edge weights and are embedded in Neo4j.
- Text chunks from summary/additional-info/attachments are embedded in Neo4j.
- Faculty publication abstracts are embedded on `FacultyPublication` nodes.

## Key relation semantics

Keyword relation mapping:

- `HAS_RESEARCH_DOMAIN` -> `("research", "domain")`
- `HAS_RESEARCH_SPECIALIZATION` -> `("research", "specialization")`
- `HAS_APPLICATION_DOMAIN` -> `("application", "domain")`
- `HAS_APPLICATION_SPECIALIZATION` -> `("application", "specialization")`

Text chunk relation mapping:

- `HAS_SUMMARY_CHUNK`
- `HAS_ADDITIONAL_INFO_CHUNK`
- `HAS_ATTACHMENT_CHUNK`

## Run

Initialize schema:

```bash
python graph_rag/schema.py --grants-only
```

Sync grants (single id):

```bash
python graph_rag/grant/sync_neo4j.py --opportunity-id "YOUR_OPP_ID"
```

End-to-end extraction + all-grant sync:

```bash
python graph_rag/grant/sync_all_with_extraction.py
```

Sync faculty (single email):

```bash
python graph_rag/faculty/sync_neo4j.py --email "alan.fern@oregonstate.edu"
```

End-to-end extraction + all-faculty sync:

```bash
python graph_rag/faculty/sync_all_with_extraction.py
```

Simple domain-cosine prefilter:

```bash
python graph_rag/agentic_architecture/run_orchestrator.py \
  --scenario faculty_one_to_one_match \
  --email "alan.fern@oregonstate.edu" \
  --threshold 0.2 \
  --top-k 100
```
