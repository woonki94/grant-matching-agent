from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from flask import Flask, Response, request, stream_with_context

logger = logging.getLogger(__name__)

app = Flask(__name__)

_ORCHESTRATOR = None

NODE_STEP_MESSAGES = {
    "route": "Parsing your request...",
    "run_general_response": "Handling general question briefly...",
    "decide_one_to_one": "Checking one-to-one prerequisites...",
    "decide_group": "Checking group-matching prerequisites...",
    "decide_group_specific_grant": "Checking group + specific grant prerequisites...",
    "search_grant_by_link_in_db": "Searching grant by link in DB...",
    "search_grant_by_title_in_db": "Searching grant by title in DB...",
    "fetch_grant_from_source": "Grant not in DB. Fetching from source...",
    "generate_keywords_group": "Generating group keywords...",
    "generate_keywords_group_specific_grant": "Generating group+grant keywords and specific-grant match rows...",
    "generate_keywords_one_to_one_specific_grant": "Generating faculty/grant keywords and specific-grant match rows...",
    "run_one_to_one_matching": "Checking stored matches; generating keywords/matches if needed; building recommendations...",
    "run_one_to_one_matching_with_specific_grant": "Checking one-to-one match for the specific grant...",
    "run_one_to_one_specific_grant_justification": "Generating one-to-one justification for the specific grant...",
    "run_group_matching": "Running group matching...",
    "run_group_matching_with_specific_grant": "Running group matching with specific grant...",
}

def _get_orchestrator():
    global _ORCHESTRATOR
    if _ORCHESTRATOR is not None:
        return _ORCHESTRATOR, None
    try:
        from services.agent_v2 import GrantMatchOrchestrator, build_memory_checkpointer

        _ORCHESTRATOR = GrantMatchOrchestrator(checkpointer=build_memory_checkpointer())
        return _ORCHESTRATOR, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


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


def _to_optional_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _to_email_list(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        x = v.strip()
        return [x] if x else []
    return []


def _sanitize_step_update(update: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(update, dict):
        return {}
    if "result" not in update:
        return update
    result = update.get("result")
    if isinstance(result, dict):
        return {"result": {"next_action": result.get("next_action"), "source": result.get("source")}}
    return {"result": {}}


def _parse_request() -> tuple[dict, Optional[dict]]:
    """
    Parse the incoming request into (fields_dict, cv_pdf_map_or_none).

    cv_pdf_map is a dict of {email_lowercase: pdf_bytes} so multiple CVs can be
    attached in a single request (one per faculty member).

    Supported content types
    ───────────────────────
    application/json
        Standard JSON body, no file upload.  cv_pdf_map will be None.

    multipart/form-data — two attachment styles:

      Style A — single CV (most common, backward-compatible):
        Field  "cv"       → the PDF file
        The CV is mapped to the value of the "email" form field.

      Style B — multiple CVs (group matching):
        Fields "cv_email" → repeated, one per faculty (e.g. A@osu.edu, B@osu.edu)
        Fields "cv_file"  → repeated, PDF file for each cv_email (same order)
        Both lists are zipped together to build the map.

    Style B takes precedence when cv_email/cv_file fields are present.
    """
    content_type = (request.content_type or "").lower()

    if "multipart/form-data" not in content_type:
        body = request.get_json(silent=True) or {}
        return body, None

    # ── Collect scalar/list form fields ──────────────────────────────────────
    body: dict = {}
    for key in request.form:
        values = request.form.getlist(key)
        body[key] = values[0] if len(values) == 1 else values

    # JSON-decode fields that may be sent as JSON strings (e.g. emails array).
    for key in ("emails",):
        raw = body.get(key)
        if isinstance(raw, str):
            try:
                body[key] = json.loads(raw)
            except Exception:
                pass

    # ── Build cv_pdf_map ─────────────────────────────────────────────────────
    cv_pdf_map: dict = {}

    # Style B: explicit per-faculty email+file pairs (group use case).
    # Each cv_email field is the "label/comment" identifying whose CV the
    # corresponding cv_file belongs to.  File names are ignored entirely.
    cv_emails = request.form.getlist("cv_email")
    cv_files = request.files.getlist("cv_file")
    if cv_emails and cv_files:
        if len(cv_emails) != len(cv_files):
            logger.warning(
                "_parse_request: cv_email count (%d) != cv_file count (%d); "
                "extra entries will be dropped (zip stops at the shorter list)",
                len(cv_emails),
                len(cv_files),
            )
        for email, fobj in zip(cv_emails, cv_files):
            email = email.strip().lower()
            if email:
                cv_pdf_map[email] = fobj.read()

    # Style A: single "cv" file → associate with the primary "email" field.
    if not cv_pdf_map:
        cv_file = request.files.get("cv")
        if cv_file:
            primary_email = str(body.get("email") or "").strip().lower()
            if primary_email:
                cv_pdf_map[primary_email] = cv_file.read()

    return body, cv_pdf_map or None


@app.post("/api/chat")
def chat():
    body, cv_pdf_map = _parse_request()
    user_message = str(body.get("message") or "").strip()
    thread_id = str(body.get("thread_id") or "default-thread")

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
            print(f"chat.error.agent_v2_unavailable: {orchestrator_error}")
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

        from services.agent_v2 import GrantMatchRequest

        req = GrantMatchRequest(
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
            desired_broad_category=(
                body["desired_broad_category"]
                if "desired_broad_category" in body
                else body.get("broad_category_filter")
            ),
            topic_query=body.get("topic_query"),
            requested_team_size=_to_optional_int(body.get("requested_team_size") or body.get("team_size")),
            requested_top_k_grants=_to_optional_int(
                body.get("requested_top_k_grants")
                or body.get("top_k_grants")
                or body.get("top_k")
                or body.get("k")
            ),
            cv_pdf_map=cv_pdf_map,
        )

        decision_out: Dict[str, Any] = {}
        for evt in orchestrator.stream(req, thread_id=thread_id):
            etype = evt.get("type")
            if etype == "step":
                node = str(evt.get("node") or "")
                print(f"chat.step.{node}")
                yield emit(
                    "step_update",
                    {
                        "message": NODE_STEP_MESSAGES.get(node, f"Running step: {node}"),
                        "node": node,
                        "update": _sanitize_step_update(evt.get("update") or {}),
                    },
                )
                continue

            if etype == "final":
                decision_out = evt.get("output") or {}

        if not decision_out:
            yield emit(
                "message",
                {
                    "type": "error",
                    "message": "Orchestrator did not return a final output.",
                },
            )
            return

        next_action = (decision_out.get("result") or {}).get("next_action")
        print(f"chat.decision.{next_action}")
        if next_action == "general_reply":
            result = decision_out.get("result") or {}
            yield emit(
                "message",
                {
                    "message": result.get("message") or "I focus on grant matching.",
                    "query": user_message,
                    "orchestrator": decision_out,
                    "result": result,
                },
            )
            return
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

        print("chat.final.message")
        print(decision_out.get("result"))
        yield emit(
            "message",
            {
                "message": "Here are the matching results.",
                "query": user_message,
                "orchestrator": decision_out,
                "result": decision_out.get("result") or {},
            },
        )

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return Response(generate(), mimetype="text/event-stream", headers=headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
