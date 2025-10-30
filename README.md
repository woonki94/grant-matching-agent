
Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

requirements.txt should include:

SQLAlchemy>=2.0.0
psycopg2-binary>=2.9.9
python-dotenv>=1.0.1
Jinja2>=3.1.4
google-generativeai>=0.5.0
Flask>=3.0.0

Setup
	1.	Start PostgreSQL.
	2.	Configure the connection string in data/db_conn.py
Example:

DATABASE_URL = "postgresql+psycopg2://user:password@localhost:5432/grants"


	3.	Run the initialization script:
```bash
./scripts/init_db.sh
```
This creates the database and tables.

	4.	Fetch and save grant data:
```bash
./scripts/fetch_commit_grant.sh [page_offset] [page_size] [query]
```
Example:
```bash
./scripts/fetch_commit_grant.sh 1 10 "machine learning"
```

	5.	Generate keywords using Gemini:
```bash
./scripts/generate_keyword.sh [batch_size] [max_keywords]
```
Example:
```bash
./scripts/generate_keyword.sh 50 25
```


API Keys

Create a file named api.env in the project root:

API_KEY=
GEMINI_API_KEY=

Put your actual keys in this file.

⸻
