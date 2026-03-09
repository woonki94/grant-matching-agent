# Grant GraphRAG Prototype (Neo4j)

This folder contains a local prototype for syncing grant data from Postgres into Neo4j and retrieving grants with graph traversal over weighted keyword edges.

Shared modules under `tmp/`:
- `tmp/neo4j_common.py`
- `tmp/neo4j_schema.py`

## Files
- `neo4j_common.py`: shared settings/helpers
- `neo4j_schema.py`: creates constraints/indexes
- `grant_sync_neo4j.py`: syncs `Opportunity` + related data into graph
- `grant_retrieve_neo4j.py`: filter + keyword retrieval query
- `grant_sync_all_with_extraction.py`: full pipeline (extract to S3 content paths + sync all grants with actual text)

## Graph Model (high level)
- `(:Grant {opportunity_id, opportunity_title, summary_description, award_ceiling, award_floor, close_date, ...})`
- `(:Agency {name})`
- `(:GrantKeyword {value, section, bucket})`
- `(:GrantBroadCategory {name})`
- `(:GrantSpecificCategory {name})`
- `(:OpportunityCategory {name})`
- `(:ApplicantType {name})`
- `(:FundingCategory {name})`
- `(:FundingInstrument {name})`
- `(:GrantAdditionalInfo {additional_info_id, additional_info_url, content_path, extracted_text, ...})`
- `(:GrantAttachment {attachment_id, file_name, file_download_path, content_path, extracted_text, ...})`

Key relationships:
- `(Grant)-[:FUNDED_BY]->(Agency)`
- `(Grant)-[:HAS_RESEARCH_DOMAIN|HAS_RESEARCH_SPECIALIZATION|HAS_APPLICATION_DOMAIN|HAS_APPLICATION_SPECIALIZATION {weight}]->(GrantKeyword)`
- `(Grant)-[:HAS_BROAD_CATEGORY]->(GrantBroadCategory)`
- `(Grant)-[:HAS_SPECIFIC_CATEGORY]->(GrantSpecificCategory)`
- `(Grant)-[:IN_OPPORTUNITY_CATEGORY]->(OpportunityCategory)`
- `(Grant)-[:HAS_APPLICANT_TYPE]->(ApplicantType)`
- `(Grant)-[:HAS_FUNDING_CATEGORY]->(FundingCategory)`
- `(Grant)-[:HAS_FUNDING_INSTRUMENT]->(FundingInstrument)`
- `(Grant)-[:HAS_ADDITIONAL_INFO]->(GrantAdditionalInfo)`
- `(Grant)-[:HAS_ATTACHMENT]->(GrantAttachment)`

## Usage
From project root:

```bash
python tmp/grant_graph_rag/neo4j_schema.py
```

Sync one grant:

```bash
python tmp/grant_graph_rag/grant_sync_neo4j.py --opportunity-id "<ID>" --verify-opportunity-id "<ID>"
```

Sync all grants (batch):

```bash
python tmp/grant_graph_rag/grant_sync_neo4j.py --all --limit 500 --offset 0
```

If you want faster sync without S3 text fetch:

```bash
python tmp/grant_graph_rag/grant_sync_neo4j.py --all --skip-extracted-text
```

Retrieve grants by keyword/category:

```bash
python tmp/grant_graph_rag/grant_retrieve_neo4j.py \
  --query-keywords "robotics, reinforcement learning" \
  --broad-category applied_research \
  --top-k 10
```

Full pipeline (recommended for your use case):

```bash
python tmp/grant_graph_rag/grant_sync_all_with_extraction.py \
  --retry-failed-extraction
```

## Notes
- Neo4j credentials use `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` (or CLI flags).
- Extracted text is loaded from S3 using existing `content_path` records.
- Keyword edge weights come from the stored weighted specialization format (`{"t": ..., "w": ...}`).
- `--agency` retrieval filtering is normalized/fuzzy and acronym-aware (e.g., `NSF`, `NIH`, `DoD`) without a manual synonym table.
