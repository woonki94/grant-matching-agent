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


def _parse_request() -> tuple[dict, Optional[dict], Optional[dict]]:
    """
    Parse the incoming request into (fields_dict, cv_pdf_map, osu_url_map).

    cv_pdf_map  — {email: pdf_bytes}  — per-faculty CV upload
    osu_url_map — {email: osu_url}    — explicit OSU profile URL when the slug
                                        cannot be derived from the email address
                                        (e.g. houssam.abbas → /people/houssam-abbas)

    Supported content types
    ───────────────────────
    application/json
        Standard JSON body, no file upload.  Both maps will be None.

    multipart/form-data — two attachment styles for each map:

      CV  — Style A (single faculty):
        Field "cv"       → the PDF file  (mapped to the "email" form field)
      CV  — Style B (group):
        Fields "cv_email" + "cv_file"   → repeated pairs, one per faculty

      OSU — Style A (single faculty):
        Field "osu_url"  → explicit URL  (mapped to the "email" form field)
      OSU — Style B (group):
        Fields "osu_url_email" + "osu_url_value" → repeated pairs, one per faculty

    Style B takes precedence over Style A for both maps.
    """
    content_type = (request.content_type or "").lower()

    if "multipart/form-data" not in content_type:
        body = request.get_json(silent=True) or {}
        return body, None, None

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

    primary_email = str(body.get("email") or "").strip().lower()

    # ── Build cv_pdf_map ─────────────────────────────────────────────────────
    cv_pdf_map: dict = {}

    cv_emails = request.form.getlist("cv_email")
    cv_files = request.files.getlist("cv_file")
    if cv_emails and cv_files:
        if len(cv_emails) != len(cv_files):
            logger.warning(
                "_parse_request: cv_email count (%d) != cv_file count (%d); "
                "extra entries will be dropped",
                len(cv_emails), len(cv_files),
            )
        for em, fobj in zip(cv_emails, cv_files):
            em = em.strip().lower()
            if em:
                cv_pdf_map[em] = fobj.read()

    if not cv_pdf_map:
        cv_file = request.files.get("cv")
        if cv_file and primary_email:
            cv_pdf_map[primary_email] = cv_file.read()

    # ── Build osu_url_map ─────────────────────────────────────────────────────
    osu_url_map: dict = {}

    osu_url_emails = request.form.getlist("osu_url_email")
    osu_url_values = request.form.getlist("osu_url_value")
    if osu_url_emails and osu_url_values:
        if len(osu_url_emails) != len(osu_url_values):
            logger.warning(
                "_parse_request: osu_url_email count (%d) != osu_url_value count (%d); "
                "extra entries will be dropped",
                len(osu_url_emails), len(osu_url_values),
            )
        for em, url in zip(osu_url_emails, osu_url_values):
            em = em.strip().lower()
            url = url.strip()
            if em and url:
                osu_url_map[em] = url

    if not osu_url_map:
        single_osu_url = str(body.get("osu_url") or "").strip()
        if single_osu_url and primary_email:
            osu_url_map[primary_email] = single_osu_url

    return body, cv_pdf_map or None, osu_url_map or None


@app.post("/api/chat")
def chat():
    body, cv_pdf_map, osu_url_map = _parse_request()
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

        # Validate: every explicitly provided email must have a matching osu_url entry.
        # This check only applies when emails are given in the request body; the
        # orchestrator may still ask for emails later (via ask_email / ask_group_emails).
        explicitly_provided_emails: List[str] = []
        raw_single = str(body.get("email") or "").strip().lower()
        if raw_single:
            explicitly_provided_emails.append(raw_single)
        for e in _to_email_list(body.get("emails")):
            if e not in explicitly_provided_emails:
                explicitly_provided_emails.append(e)

        if explicitly_provided_emails:
            missing_osu = [e for e in explicitly_provided_emails if e not in (osu_url_map or {})]
            if missing_osu:
                print(f"chat.request_info.missing_osu_url: {missing_osu}")
                yield emit(
                    "request_info",
                    {
                        "type": "missing_osu_url",
                        "message": (
                            "Please provide an OSU engineering profile URL for each faculty member. "
                            f"Missing for: {', '.join(missing_osu)}"
                        ),
                        "emails_missing_osu_url": missing_osu,
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
            osu_url_map=osu_url_map,
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


# ─────────────────────────────────────────────────────────────────────────────
# Team endpoints — shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sse_headers() -> dict:
    return {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _parse_team_request():
    """
    Parse multipart/form-data (or JSON) for team sub-endpoints.
    Returns:
        (grant_link, grant_title, emails, osu_url_map, cv_pdf_map,
         additional_count, team_size, message)
    """
    content_type = (request.content_type or "").lower()

    if "multipart/form-data" in content_type:
        body: dict = {}
        for key in request.form:
            values = request.form.getlist(key)
            body[key] = values[0] if len(values) == 1 else values

        for key in ("emails",):
            raw = body.get(key)
            if isinstance(raw, str):
                try:
                    body[key] = json.loads(raw)
                except Exception:
                    pass

        primary_email = str(body.get("email") or "").strip().lower()

        # OSU URL map
        osu_url_map: dict = {}
        for em, url in zip(request.form.getlist("osu_url_email"), request.form.getlist("osu_url_value")):
            em, url = em.strip().lower(), url.strip()
            if em and url:
                osu_url_map[em] = url
        if not osu_url_map:
            single_osu = str(body.get("osu_url") or "").strip()
            if single_osu and primary_email:
                osu_url_map[primary_email] = single_osu

        # CV map
        cv_pdf_map: dict = {}
        for em, fobj in zip(request.form.getlist("cv_email"), request.files.getlist("cv_file")):
            em = em.strip().lower()
            if em:
                cv_pdf_map[em] = fobj.read()
        if not cv_pdf_map:
            cv_file = request.files.get("cv")
            if cv_file and primary_email:
                cv_pdf_map[primary_email] = cv_file.read()
    else:
        body = request.get_json(silent=True) or {}
        osu_url_map = {}
        cv_pdf_map = {}

    grant_link  = str(body.get("grant_link")  or "").strip() or None
    grant_title = str(body.get("grant_title") or "").strip() or None
    emails      = _to_email_list(body.get("emails") or body.get("email"))
    additional_count = _to_optional_int(body.get("additional_count")) or 3
    team_size        = _to_optional_int(body.get("team_size"))        or 3
    message = str(body.get("message") or "").strip() or None

    return (
        grant_link, grant_title, emails,
        osu_url_map or None, cv_pdf_map or None,
        additional_count, team_size, message,
    )


def _resolve_opportunity_id(grant_link=None, grant_title=None):
    """
    Find opportunity_id from a simpler.grants.gov URL, a bare opportunity ID,
    or a title/keyword search.  Returns (opportunity_id, opportunity_title) or (None, None).

    Delegates to OpportunityContextAgent so the extraction logic stays in one place.
    """
    from services.agent_v2.agents.opportunity_context_agent import OpportunityContextAgent
    agent = OpportunityContextAgent()

    logger.info("_resolve_opportunity_id: grant_link=%r  grant_title=%r", grant_link, grant_title)

    # 1. Try link/URL/bare-ID lookup first
    if grant_link:
        result = agent.search_grant_by_link_in_db(grant_link=str(grant_link).strip())
        logger.info("_resolve_opportunity_id: link search result=%s", result)
        if result.get("found"):
            return result["opportunity_id"], result.get("opportunity_title")

    # 2. Try title search
    if grant_title:
        result = agent.search_grant_by_title_in_db(grant_title=str(grant_title).strip())
        logger.info("_resolve_opportunity_id: title search result=%s", result)
        if result.get("found"):
            return result["opportunity_id"], result.get("opportunity_title")

    # 3. If user put title text in the link field (no URL detected), try that too
    if grant_link and not grant_title:
        result = agent.search_grant_by_title_in_db(grant_title=str(grant_link).strip())
        logger.info("_resolve_opportunity_id: link-as-title search result=%s", result)
        if result.get("found"):
            return result["opportunity_id"], result.get("opportunity_title")

    logger.warning("_resolve_opportunity_id: grant not found for link=%r title=%r", grant_link, grant_title)
    return None, None


def _resolve_faculty_ids_for_team(emails, osu_url_map, cv_pdf_map):
    """
    Resolve + optionally ingest faculty, return {email: faculty_id}.
    Skips gracefully if emails is empty.
    """
    if not emails:
        return {}
    try:
        from services.agent_v2.agents.faculty_context_agent import FacultyContextAgent
        agent = FacultyContextAgent()
        result = agent.resolve_and_ingest_faculties(
            emails=emails,
            osu_url_map=osu_url_map,
            cv_pdf_map=cv_pdf_map,
        )
        # Build email → faculty_id from resolved ids
        from db.db_conn import SessionLocal
        from dao.faculty_dao import FacultyDAO
        email_to_fid = {}
        with SessionLocal() as sess:
            dao = FacultyDAO(sess)
            for email in emails:
                fid = dao.get_faculty_id_by_email(email)
                if fid:
                    email_to_fid[email] = int(fid)
        return email_to_fid
    except Exception:
        logger.exception("_resolve_faculty_ids_for_team failed")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/team/find-collaborators  — scenario b
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/team/find-collaborators")
def find_collaborators():
    (grant_link, grant_title, emails,
     osu_url_map, cv_pdf_map,
     additional_count, _, message) = _parse_team_request()

    @stream_with_context
    def generate():
        if not grant_link and not grant_title:
            yield _sse("request_info", {"type": "missing_grant", "message": "Please provide a grant link or title."})
            return
        if not emails:
            yield _sse("request_info", {"type": "missing_emails", "message": "Please provide at least one existing team member email."})
            return

        # Validate osu_url for provided emails
        missing_osu = [e for e in emails if e not in (osu_url_map or {})]
        if missing_osu:
            yield _sse("request_info", {
                "type": "missing_osu_url",
                "message": f"Please provide an OSU profile URL for: {', '.join(missing_osu)}",
                "emails_missing_osu_url": missing_osu,
            })
            return

        searched = grant_title or grant_link or "(none)"
        yield _sse("step_update", {"message": f"Searching for grant: {searched!r}..."})
        opp_id, opp_title = _resolve_opportunity_id(grant_link, grant_title)
        if not opp_id:
            yield _sse("message", {
                "type": "error",
                "message": (
                    f"Grant not found in the database for: {searched!r}. "
                    "Try the exact grant title or paste the simpler.grants.gov URL."
                ),
            })
            return

        yield _sse("step_update", {"message": f"Grant resolved: {opp_title or opp_id}. Resolving existing team members..."})
        email_to_fid = _resolve_faculty_ids_for_team(emails, osu_url_map, cv_pdf_map)
        existing_ids = list(email_to_fid.values())

        yield _sse("step_update", {"message": f"Searching for {additional_count} collaborator(s)..."})
        try:
            from services.agent_v2.agents.matching_execution_agent import MatchingExecutionAgent
            agent = MatchingExecutionAgent()
            result = agent.find_collaborators_for_grant(
                opportunity_id=opp_id,
                existing_faculty_ids=existing_ids,
                additional_count=additional_count,
            )
        except Exception as e:
            yield _sse("message", {"type": "error", "message": f"Matching error: {e}"})
            return

        if result.get("next_action", "").startswith("error"):
            yield _sse("message", {"type": "error", "message": result.get("error", "Unknown error.")})
            return

        yield _sse("message", {
            "message": "Here are the suggested collaborators.",
            "result": result,
        })

    return Response(generate(), mimetype="text/event-stream", headers=_sse_headers())


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/team/form-team  — scenario c
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/team/form-team")
def form_team():
    (grant_link, grant_title, emails,
     osu_url_map, cv_pdf_map,
     _, team_size, message) = _parse_team_request()

    @stream_with_context
    def generate():
        if not grant_link and not grant_title:
            yield _sse("request_info", {"type": "missing_grant", "message": "Please provide a grant link or title."})
            return

        # If emails given, osu_url is required for each
        if emails:
            missing_osu = [e for e in emails if e not in (osu_url_map or {})]
            if missing_osu:
                yield _sse("request_info", {
                    "type": "missing_osu_url",
                    "message": f"Please provide an OSU profile URL for: {', '.join(missing_osu)}",
                    "emails_missing_osu_url": missing_osu,
                })
                return

        searched = grant_title or grant_link or "(none)"
        yield _sse("step_update", {"message": f"Searching for grant: {searched!r}..."})
        opp_id, opp_title = _resolve_opportunity_id(grant_link, grant_title)
        if not opp_id:
            yield _sse("message", {
                "type": "error",
                "message": (
                    f"Grant not found in the database for: {searched!r}. "
                    "Try the exact grant title or paste the simpler.grants.gov URL."
                ),
            })
            return

        existing_ids: List[int] = []
        if emails:
            yield _sse("step_update", {"message": "Resolving existing team members..."})
            email_to_fid = _resolve_faculty_ids_for_team(emails, osu_url_map, cv_pdf_map)
            existing_ids = list(email_to_fid.values())

        yield _sse("step_update", {"message": f"Finding best team of {team_size} for this grant..."})
        try:
            from services.agent_v2.agents.matching_execution_agent import MatchingExecutionAgent
            agent = MatchingExecutionAgent()
            result = agent.find_team_for_grant(
                opportunity_id=opp_id,
                team_size=team_size,
                existing_faculty_ids=existing_ids or None,
            )
        except Exception as e:
            yield _sse("message", {"type": "error", "message": f"Matching error: {e}"})
            return

        if result.get("next_action", "").startswith("error"):
            yield _sse("message", {"type": "error", "message": result.get("error", "Unknown error.")})
            return

        yield _sse("message", {
            "message": "Here is the suggested team.",
            "result": result,
        })

    return Response(generate(), mimetype="text/event-stream", headers=_sse_headers())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
