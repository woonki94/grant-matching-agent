from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from tmp.agentic_arch.neo4j_tools import Neo4jFacultyTools, Neo4jGrantTools
from tmp.agentic_arch.orchestrator import ProfessionGrantMatchOrchestrator
from tmp.neo4j_common import json_ready, load_dotenv_if_present, read_neo4j_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run independent faculty->grant one-to-one match using Neo4j tool adapters.")
    parser.add_argument("--email", type=str, required=True, help="Faculty email to match from Faculty graph.")
    parser.add_argument("--candidate-grant-k", type=int, default=20, help="Candidate grant count before ranking.")
    parser.add_argument("--top-k", type=int, default=5, help="Final top-k one-to-one matches.")
    parser.add_argument("--uri", type=str, default="", help="Neo4j URI. Fallback: NEO4J_URI")
    parser.add_argument("--username", type=str, default="", help="Neo4j username. Fallback: NEO4J_USERNAME")
    parser.add_argument("--password", type=str, default="", help="Neo4j password. Fallback: NEO4J_PASSWORD")
    parser.add_argument("--database", type=str, default="", help="Neo4j database. Fallback: NEO4J_DATABASE or neo4j")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    load_dotenv_if_present()

    settings = read_neo4j_settings(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    faculty_tools = Neo4jFacultyTools(settings)
    grant_tools = Neo4jGrantTools(settings)
    try:
        orchestrator = ProfessionGrantMatchOrchestrator(
            faculty_tools=faculty_tools,
            grant_tools=grant_tools,
        )
        out = orchestrator.run_one_to_one_sync(
            faculty_email=args.email,
            candidate_grant_k=max(1, int(args.candidate_grant_k or 20)),
            result_top_k=max(1, int(args.top_k or 5)),
        )
        print(json.dumps(json_ready(out), indent=2))
        return 0
    finally:
        faculty_tools.close()
        grant_tools.close()


if __name__ == "__main__":
    raise SystemExit(main())
