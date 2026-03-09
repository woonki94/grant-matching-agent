from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from tmp.agentic_arch.orchestrator import ProfessionGrantMatchOrchestrator
from tmp.agentic_arch.run_demo import build_demo_orchestrator
from tmp.agentic_arch.tools import InMemoryFacultyTools, InMemoryGrantTools
from tmp.agentic_arch_v2.orchestrator import ProfessionGrantConversationOrchestrator
from tmp.agentic_arch_v2.planner import LLMQueryPlanner


def _build_tools_from_v1_demo() -> tuple[InMemoryFacultyTools, InMemoryGrantTools]:
    v1 = build_demo_orchestrator()
    if not isinstance(v1, ProfessionGrantMatchOrchestrator):
        raise RuntimeError("Unexpected demo orchestrator type")
    return v1.faculty_agent.tools, v1.grant_agent.tools


def main() -> int:
    faculty_tools, grant_tools = _build_tools_from_v1_demo()
    orchestrator = ProfessionGrantConversationOrchestrator(
        faculty_tools=faculty_tools,
        grant_tools=grant_tools,
        planner=LLMQueryPlanner(enable_llm=False),
    )
    out = orchestrator.run_sync(
        faculty_email="alan.fern@oregonstate.edu",
        candidate_grant_k=10,
        result_top_k=3,
        max_rounds=5,
    )

    payload = {
        "faculty_profile": {
            "email": out.faculty_profile.email,
            "profession_focus": list(out.faculty_profile.profession_focus or []),
        },
        "candidate_grant_ids": list(out.candidate_grant_ids or []),
        "stop_reason": out.stop_reason,
        "rounds": [
            {
                "round_index": r.round_index,
                "query_count": len(r.queries),
                "answer_count": len(r.answers),
                "queries": [
                    {
                        "query_id": q.query_id,
                        "target_agent": q.target_agent,
                        "intent": q.intent,
                        "grant_id": q.grant_id,
                        "question": q.question,
                    }
                    for q in r.queries
                ],
                "answers": [
                    {
                        "query_id": a.query_id,
                        "target_agent": a.target_agent,
                        "intent": a.intent,
                        "confidence": a.confidence,
                        "answer": a.answer,
                        "followups": [f.query_id for f in (a.followup_queries or [])],
                    }
                    for a in r.answers
                ],
            }
            for r in out.rounds
        ],
        "matches": [
            {
                "grant_id": m.grant_id,
                "score": m.score,
                "grant_name": m.grant_name,
                "agency_name": m.agency_name,
                "close_date": m.close_date,
                "matched_professions": list(m.matched_professions or []),
            }
            for m in out.matches
        ],
    }

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
