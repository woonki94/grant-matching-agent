from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from flask import Flask, Response, request, stream_with_context

app = Flask(__name__)

_ORCHESTRATOR = None
_ORCHESTRATOR_ERROR: Optional[str] = None

MOCK_GRANTS: List[Dict[str, Any]] = [
    {
        "opportunity_id": "mock-opp-ai-001",
        "title": "Applied AI for Healthcare Research",
        "agency": "NIH",
        "tags": ["ai", "machine", "learning", "healthcare", "biomedical"],
    },
    {
        "opportunity_id": "mock-opp-space-002",
        "title": "Space Technology and Systems Engineering",
        "agency": "NASA",
        "tags": ["space", "satellite", "aerospace", "engineering", "systems"],
    },
    {
        "opportunity_id": "mock-opp-climate-003",
        "title": "Climate Resilience and Sustainability",
        "agency": "NSF",
        "tags": ["climate", "sustainability", "environment", "resilience", "energy"],
    },
    {
        "opportunity_id": "mock-opp-data-004",
        "title": "Data Infrastructure for Science",
        "agency": "DOE",
        "tags": ["data", "infrastructure", "hpc", "modeling", "science"],
    },
]


class MockGrantMatcherAgent:
    @staticmethod
    def _call(name: str) -> None:
        print(name)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t}

    def match(self, user_message: str, k: int = 3) -> List[Dict[str, Any]]:
        self._call("MockGrantMatcherAgent.match")
        query_tokens = self._tokenize(user_message)
        ranked: List[Dict[str, Any]] = []

        for item in MOCK_GRANTS:
            tag_set = set(item.get("tags") or [])
            overlap = sorted(tag_set & query_tokens)
            score = len(overlap) / max(len(tag_set), 1)
            ranked.append(
                {
                    "opportunity_id": item["opportunity_id"],
                    "title": item["title"],
                    "agency": item["agency"],
                    "score": round(score, 2),
                    "matched_terms": overlap,
                }
            )

        ranked.sort(key=lambda x: (x["score"], len(x["matched_terms"])), reverse=True)
        top = ranked[:k]

        if top and top[0]["score"] > 0:
            return top

        # Fallback when no tokens match: return first k with low confidence.
        return [
            {
                **x,
                "score": 0.1,
                "matched_terms": [],
            }
            for x in ranked[:k]
        ]


def _get_orchestrator():
    global _ORCHESTRATOR, _ORCHESTRATOR_ERROR
    if _ORCHESTRATOR is not None or _ORCHESTRATOR_ERROR is not None:
        return _ORCHESTRATOR, _ORCHESTRATOR_ERROR
    try:
        from services.agent_v2 import GrantMatchOrchestrator, build_memory_checkpointer

        _ORCHESTRATOR = GrantMatchOrchestrator(checkpointer=build_memory_checkpointer())
        return _ORCHESTRATOR, None
    except Exception as e:
        _ORCHESTRATOR_ERROR = f"{type(e).__name__}: {e}"
        return None, _ORCHESTRATOR_ERROR


def _to_optional_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def _to_email_list(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        x = v.strip()
        return [x] if x else []
    return []


@app.post("/api/chat")
def chat():
    body = request.get_json(silent=True) or {}
    user_message = str(body.get("message") or "").strip()
    thread_id = str(body.get("thread_id") or "default-thread")
    matcher = MockGrantMatcherAgent()

    def emit(event_name: str, payload: dict) -> str:
        return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @stream_with_context
    def generate():
        print("chat.start")

        if not user_message:
            print("chat.request_info.empty_message")
            yield emit(
                "request_info",
                {
                    "type": "missing_message",
                    "message": "Please enter a message.",
                },
            )
            return

        orchestrator, orchestrator_error = _get_orchestrator()
        if orchestrator is None:
            print("chat.error.agent_v2_unavailable")
            yield emit(
                "message",
                {
                    "type": "error",
                    "message": "agent_v2 orchestrator is unavailable.",
                    "detail": orchestrator_error,
                },
            )
            return

        print("chat.step.run_orchestrator")
        yield emit("step_update", {"message": "Running orchestration..."})
        time.sleep(1)

        from services.agent_v2 import GrantMatchRequest

        decision_out = orchestrator.run(
            GrantMatchRequest(
                user_input=user_message,
                email=body.get("email"),
                emails=_to_email_list(body.get("emails")),
                faculty_in_db=_to_optional_bool(body.get("faculty_in_db")),
                email_in_db=_to_optional_bool(body.get("email_in_db")),
                grant_link=body.get("grant_link"),
                grant_title=body.get("grant_title"),
                grant_identifier_type=body.get("grant_identifier_type"),
                grant_in_db=_to_optional_bool(body.get("grant_in_db")),
                grant_link_valid=_to_optional_bool(body.get("grant_link_valid")),
                grant_title_confirmed=_to_optional_bool(body.get("grant_title_confirmed")),
            ),
            thread_id=thread_id,
        )

        next_action = (decision_out.get("result") or {}).get("next_action")
        print(f"chat.decision.{next_action}")
        if next_action == "ask_email":
            yield emit(
                "request_info",
                {
                    "type": "missing_email",
                    "message": "Please provide your faculty email.",
                    "orchestrator": decision_out,
                },
            )
            return
        if next_action == "ask_group_emails":
            yield emit(
                "request_info",
                {
                    "type": "missing_group_emails",
                    "message": "Please provide at least two faculty emails for group matching.",
                    "orchestrator": decision_out,
                },
            )
            return
        if next_action == "ask_user_reference_data":
            yield emit(
                "request_info",
                {
                    "type": "faculty_missing",
                    "message": "Faculty not found in DB. Please upload reference profile data.",
                    "orchestrator": decision_out,
                },
            )
            return
        if next_action == "ask_grant_identifier":
            yield emit(
                "request_info",
                {
                    "type": "missing_grant_identifier",
                    "message": "Please provide a specific grant link or grant title.",
                    "orchestrator": decision_out,
                },
            )
            return

        print("chat.step.fetching_grants")
        yield emit("step_update", {"message": "Fetching grants..."})
        time.sleep(1)

        print("chat.step.generating_keywords")
        yield emit("step_update", {"message": "Generating keywords..."})
        time.sleep(1)

        print("chat.step.mock_matching")
        yield emit("step_update", {"message": "Matching opportunities to your prompt..."})
        results = matcher.match(user_message, k=3)

        print("chat.final.message")
        yield emit(
            "message",
            {
                "message": "Here are the prompt-matched mock results.",
                "query": user_message,
                "orchestrator": decision_out,
                "results": results,
            },
        )

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return Response(generate(), mimetype="text/event-stream", headers=headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
