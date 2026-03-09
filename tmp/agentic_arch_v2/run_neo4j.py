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
from tmp.agentic_arch_v2.orchestrator import ProfessionGrantConversationOrchestrator
from tmp.agentic_arch_v2.planner import LLMQueryPlanner
from tmp.neo4j_common import json_ready, load_dotenv_if_present, read_neo4j_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v2 faculty<->grant conversational orchestration on Neo4j.")
    parser.add_argument("--email", type=str, required=True, help="Faculty email")
    parser.add_argument("--candidate-grant-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-queries-per-round", type=int, default=120)
    parser.add_argument("--disable-llm-planner", action="store_true", help="Use deterministic fallback query planner.")
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
        orchestrator = ProfessionGrantConversationOrchestrator(
            faculty_tools=faculty_tools,
            grant_tools=grant_tools,
            planner=LLMQueryPlanner(enable_llm=not bool(args.disable_llm_planner)),
        )
        out = orchestrator.run_sync(
            faculty_email=args.email,
            candidate_grant_k=max(1, int(args.candidate_grant_k or 20)),
            result_top_k=max(1, int(args.top_k or 5)),
            max_rounds=max(1, int(args.max_rounds or 3)),
            max_queries_per_round=max(1, int(args.max_queries_per_round or 120)),
        )

        payload = {
            "faculty_profile": {
                "email": out.faculty_profile.email,
                "faculty_name": out.faculty_profile.basic_info.faculty_name,
                "position": out.faculty_profile.basic_info.position,
                "profession_focus": list(out.faculty_profile.profession_focus or []),
            },
            "candidate_grant_ids": list(out.candidate_grant_ids or []),
            "stop_reason": out.stop_reason,
            "rounds": [
                {
                    "round_index": r.round_index,
                    "query_count": len(r.queries),
                    "answer_count": len(r.answers),
                    "avg_confidence": (
                        sum(float(a.confidence) for a in (r.answers or [])) / max(len(r.answers), 1)
                    ),
                }
                for r in out.rounds
            ],
            "matches": [
                {
                    "grant_id": m.grant_id,
                    "grant_name": m.grant_name,
                    "agency_name": m.agency_name,
                    "close_date": m.close_date,
                    "score": m.score,
                    "matched_professions": list(m.matched_professions or []),
                    "reason": m.reason,
                }
                for m in out.matches
            ],
        }

        print(json.dumps(json_ready(payload), indent=2))
        return 0
    finally:
        faculty_tools.close()
        grant_tools.close()


if __name__ == "__main__":
    raise SystemExit(main())
