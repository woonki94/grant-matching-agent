# GrantFetcher

GrantFetcher is a backend pipeline for collecting grant opportunities and faculty profiles, extracting supporting content, generating research keywords, and matching faculty with relevant funding opportunities.

The system combines structured PostgreSQL data, pgvector embeddings, extracted document/web content stored in S3, and AWS Bedrock models for keyword generation, reranking, and recommendation explanations.

## System Overview

The implementation is organized around five main stages:

1. **Data ingestion**: import grant opportunities from the Simpler/Grants.gov API and faculty profiles from OSU Engineering pages.
2. **Content extraction**: download and parse opportunity attachments, opportunity links, and faculty links; store extracted text in S3.
3. **Keyword generation**: generate normalized faculty and opportunity keywords with Bedrock-backed LLM calls and embeddings.
4. **Matching**: compute one-to-one faculty/opportunity matches, then optionally rerank with an LLM.
5. **Serving and explanations**: expose chat, team-building, faculty-profile, authentication, and justification endpoints through Flask.

## Repository Layout

```text
config.py                     Application settings loaded from .env
db/                           SQLAlchemy base, database connection, models, migrations, initialization scripts
dao/                          Database access objects
services/opportunity/         Grant opportunity import and Grants.gov API integration
services/faculty/             Faculty crawling, profile parsing, publication enrichment, profile APIs
services/extract_content.py   Attachment/link extraction pipeline
services/keywords/            Faculty and opportunity keyword generation
services/context_retrieval/   Context builders for keyword, matching, and justification prompts
services/matching/            One-to-one and group matching logic
services/justification/       Grant explanations and recommendation justifications
services/agent_v2/            Chat orchestration and routing
scripts/                      Shell entry points for common pipelines
chat_stream_entrypoint.py     Flask API and Server-Sent Events entry point
```

## Prerequisites

- Python 3.11 or newer.
- PostgreSQL with the `pgvector` extension available.
- AWS credentials with access to Bedrock and the S3 bucket used for extracted content.
- A Grants.gov/Simpler API key.
- `psql` available on your `PATH` if you plan to run SQL migrations.

## Installation

Create and activate a virtual environment from the project root:

```bash
python -m venv venv
source venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

If Playwright reports missing browser binaries during extraction, install them:

```bash
python -m playwright install
```

## Configuration

Create a `.env` file in the project root. The shell scripts load this file automatically.

```bash
# PostgreSQL
PGUSER=postgres
PGPASSWORD=postgres
PGHOST=localhost
PGPORT=5432
PGDATABASE=grantfetcher

# Grants.gov / Simpler
GRANT_API_KEY=your_grants_api_key
SIMPLER_SEARCH_URL=https://api.simpler.grants.gov/opportunities/v1/search
SIMPLER_DETAIL_BASE_URL=https://api.simpler.grants.gov/opportunities/v1/opportunities

# Providers
LLM_PROVIDER=bedrock
EMBEDDING_PROVIDER=bedrock

# AWS / Bedrock
AWS_REGION=us-east-2
AWS_PROFILE=your_optional_aws_profile
BEDROCK_CLAUDE_HAIKU=your_haiku_model_id
BEDROCK_CLAUDE_SONNET=your_optional_sonnet_model_id
BEDROCK_CLAUDE_OPUS=your_optional_opus_model_id
BEDROCK_EMBED_MODEL_ID=your_embedding_model_id

# Extracted content storage
EXTRACTED_CONTENT_BUCKET=your_s3_bucket
EXTRACTED_CONTENT_PREFIX_OPPORTUNITY=opportunities
EXTRACTED_CONTENT_PREFIX_FACULTY=faculty

# Optional integrations
S2_API_KEY=your_optional_semantic_scholar_key
SES_FROM_EMAIL=your_optional_sender_email
SES_REPLY_TO_EMAIL=your_optional_reply_to_email
SES_CONFIGURATION_SET=your_optional_ses_configuration_set
```

Do not commit `.env` or any real credentials.

## Database Setup

Create the PostgreSQL database before running the initializer. For example:

```bash
createdb grantfetcher
```

Initialize the schema:

```bash
python db/init_db.py
```

The initializer loads all SQLAlchemy models, enables the `vector` extension if needed, and creates missing tables.

Run SQL migrations after initialization:

```bash
./scripts/run_migrations.sh
```

The migration runner records applied migration files in `schema_migrations` and skips files that have already run.

## Data Preparation Pipeline

Run these steps in order to prepare the database for recommendation and chat workflows.

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Initialize the Database

```bash
python db/init_db.py
./scripts/run_migrations.sh
```

### 3. Import Grant Opportunities and Faculty Profiles

Import grant opportunities:

```bash
./scripts/import_opportunity.sh
```

Useful options:

```bash
./scripts/import_opportunity.sh --page-size 200
./scripts/import_opportunity.sh --page-size 200 --query "machine learning"
./scripts/import_opportunity.sh --page-size 200 --agencies "HHS-NIH11,NSF"
./scripts/import_opportunity.sh --page-size 200 --fetch-workers 10 --extract-workers 6
```

Import faculty profiles:

```bash
./scripts/import_faculty.sh
```

Useful options:

```bash
./scripts/import_faculty.sh --max-pages 3 --years-back 10 --max-faculty 50
./scripts/import_faculty.sh --max-pages 3 --years-back 10 --max-faculty 50 --workers 8 --extract-workers 6
```

For a small smoke run, use low limits first:

```bash
./scripts/import_opportunity.sh --page-size 10
./scripts/import_faculty.sh --max-pages 1 --max-faculty 10
```

### 4. Generate Keywords

Generate both faculty and opportunity keywords:

```bash
./scripts/generate_keywords.sh
```

Useful options:

```bash
./scripts/generate_keywords.sh --mode faculty --limit 50
./scripts/generate_keywords.sh --mode opp --limit 100
./scripts/generate_keywords.sh --mode all --workers 8
./scripts/generate_keywords.sh --mode all --force-regenerate
```

### 5. Generate One-to-One Matches

Generate faculty-to-opportunity matches and rerank them:

```bash
./scripts/generate_one_to_one_match.sh
```

Useful options:

```bash
./scripts/generate_one_to_one_match.sh --mode match_and_rerank --k 50 --min-domain 0.30 --limit-faculty 600 --commit-every 30
./scripts/generate_one_to_one_match.sh --mode rerank_only --limit-faculty 600 --rerank-workers 4 --max-context-chars 100000
```

At this point, the core data is ready: opportunities, faculty, extracted content, keywords, embeddings, and one-to-one match rows are available for downstream recommendation flows.

## Optional Post-Processing

Pre-generate grant explanations for opportunities that do not have one:

```bash
./scripts/generate_grant_explanations.sh
```

Generate a single faculty justification report:

```bash
./scripts/generate_justification.sh --email faculty.name@oregonstate.edu --k 5
```

Generate a group justification report:

```bash
./scripts/generate_group_justification.sh \
  --email member1@oregonstate.edu \
  --email member2@oregonstate.edu \
  --team-size 2
```

## Running the API

Start the Flask API from the project root:

```bash
flask --app chat_stream_entrypoint run --host 0.0.0.0 --port 5000
```

Primary API areas include:

- `POST /api/chat`: streaming chat orchestration over Server-Sent Events.
- `POST /api/team/find-collaborators`: suggest collaborators for a known grant and existing team members.
- `POST /api/team/form-team`: suggest a complete team for a known grant.
- `GET /api/faculty`: list or fetch faculty profiles.
- `GET|POST|PATCH /api/faculty/by-email`: profile lookup and profile updates by email.
- `POST /api/faculty/create`: admin faculty creation.
- `POST /api/auth/login` and `POST /api/auth/signup`: OSU email account flows.
- `POST /api/notifications/email-justification`: send generated justification content by email.

## Implementation Notes

- Settings are centralized in `config.py` using `pydantic-settings`; environment variable names are case-insensitive.
- Database access uses SQLAlchemy sessions from `db/db_conn.py`; model definitions live under `db/models/`.
- Import scripts are safe to run repeatedly where the underlying DAO uses upsert behavior.
- Extracted content is stored in S3, while structured metadata and embeddings remain in PostgreSQL.
- Matching depends on generated keywords and embeddings, so run imports and keyword generation before one-to-one matching.
- The shell wrappers prefer `venv2/bin/python`, then `venv/bin/python`, then system `python3`.
- Use lower limits for local smoke tests before running full imports or full reranking jobs.

## Troubleshooting

- If `python db/init_db.py` fails with a connection error, confirm the `PG*` variables and that PostgreSQL is running.
- If `CREATE EXTENSION vector` fails, install pgvector on the PostgreSQL server.
- If imports fail during extraction, confirm `EXTRACTED_CONTENT_BUCKET`, AWS credentials, and S3 permissions.
- If keyword generation or reranking fails, confirm Bedrock model IDs, AWS region, and Bedrock access.
- If migrations fail, confirm `psql` is installed and that the database user can create tables.
