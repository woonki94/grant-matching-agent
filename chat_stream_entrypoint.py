from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from flask import Flask, Response, request, stream_with_context

import bcrypt
from db.db_conn import SessionLocal
from dao.faculty_dao import FacultyDAO
from dao.user_dao import UserDAO
from db.models.faculty import Faculty

logger = logging.getLogger(__name__)

app = Flask(__name__)


_ORCHESTRATOR = None
_FACULTY_PROFILE_JOB_MANAGER = None

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

# Temporary safety override for email delivery during testing.
FORCED_JUSTIFICATION_RECIPIENT = "kimwoon@oregonstate.edu"

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


def _get_faculty_profile_job_manager():
    global _FACULTY_PROFILE_JOB_MANAGER
    if _FACULTY_PROFILE_JOB_MANAGER is not None:
        return _FACULTY_PROFILE_JOB_MANAGER
    from services.faculty.faculty_profile_job_manager import FacultyProfileUpdateJobManager

    _FACULTY_PROFILE_JOB_MANAGER = FacultyProfileUpdateJobManager()
    return _FACULTY_PROFILE_JOB_MANAGER


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


def _normalize_email(v: Any) -> Optional[str]:
    s = str(v or "").strip().lower()
    return s or None


def _normalize_email_list(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in list(values or []):
        email = _normalize_email(raw)
        if not email or email in seen:
            continue
        seen.add(email)
        out.append(email)
    return out


def _resolve_chat_thread_id(
    *,
    body: Dict[str, Any],
    primary_email: Optional[str],
    emails: List[str],
) -> str:
    raw = str(body.get("thread_id") or "").strip()
    if raw:
        return raw

    participants = _normalize_email_list([primary_email, *list(emails or [])])
    if participants:
        if len(participants) == 1:
            return f"chat:faculty:{participants[0]}"
        return f"chat:group:{'|'.join(sorted(participants))}"

    # Avoid shared-state leakage when frontend omits thread_id.
    return f"chat:ephemeral:{uuid.uuid4().hex}"


def _parse_json_value(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return v
    return v


def _collect_recipient_emails(body: Dict[str, Any]) -> List[str]:
    keys = (
        "justification_email",
        "justification_emails",
        "recipient_email",
        "recipient_emails",
        "notify_email",
        "notify_emails",
        "email_to",
        "email_to_list",
    )
    raw: List[str] = []
    for key in keys:
        raw.extend(_to_email_list(body.get(key)))

    send_email = _parse_json_value(body.get("send_email"))
    if isinstance(send_email, dict):
        for key in ("to", "email", "emails", "recipient", "recipients"):
            raw.extend(_to_email_list(send_email.get(key)))

    out: List[str] = []
    seen = set()
    for x in raw:
        parts = str(x).replace(";", ",").split(",")
        for part in parts:
            email = part.strip().lower()
            if "@" not in email:
                continue
            if email in seen:
                continue
            seen.add(email)
            out.append(email)
    return out


def _extract_frontend_justification_payload(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates: List[Any] = []
    for key in ("justification_result", "justification_payload", "result_payload", "result", "orchestrator"):
        candidates.append(_parse_json_value(body.get(key)))
    # Compact FE shape:
    # { "title": "...", "content": "..." }
    compact_title = str(body.get("title") or "").strip()
    compact_content = str(body.get("content") or "").strip()
    if compact_title or compact_content:
        compact_payload: Dict[str, Any] = {}
        if compact_title:
            compact_payload["title"] = compact_title
        if compact_content:
            compact_payload["content"] = compact_content
        candidates.append(compact_payload)

    send_email = _parse_json_value(body.get("send_email"))
    if isinstance(send_email, dict):
        for key in ("result", "payload", "justification_result", "justification_payload"):
            candidates.append(_parse_json_value(send_email.get(key)))
        send_title = str(send_email.get("title") or "").strip()
        send_content = str(send_email.get("content") or "").strip()
        if send_title or send_content:
            compact_send_payload: Dict[str, Any] = {}
            if send_title:
                compact_send_payload["title"] = send_title
            if send_content:
                compact_send_payload["content"] = send_content
            candidates.append(compact_send_payload)

    for raw in candidates:
        if not isinstance(raw, dict) or not raw:
            continue

        if isinstance(raw.get("result"), dict) and raw.get("result"):
            return dict(raw["result"])
        if isinstance(raw.get("orchestrator"), dict):
            inner = raw["orchestrator"].get("result")
            if isinstance(inner, dict) and inner:
                return dict(inner)
        return dict(raw)
    return None


def _split_existing_vs_missing_db_emails(emails: List[str]) -> Dict[str, List[str]]:
    normalized: List[str] = []
    for x in emails or []:
        e = str(x or "").strip().lower()
        if e and e not in normalized:
            normalized.append(e)
    if not normalized:
        return {"existing": [], "missing": []}

    existing: List[str] = []
    missing: List[str] = []
    try:
        from db.db_conn import SessionLocal
        from dao.faculty_dao import FacultyDAO

        with SessionLocal() as sess:
            dao = FacultyDAO(sess)
            for email in normalized:
                if dao.get_faculty_id_by_email(email) is None:
                    missing.append(email)
                else:
                    existing.append(email)
    except Exception:
        logger.exception("Failed to classify emails by DB existence; treating as missing")
        return {"existing": [], "missing": normalized}

    return {"existing": existing, "missing": missing}


def _run_faculty_profile_postprocess_job(
    *,
    email: str,
    faculty_id: int,
    postprocess_plan: Dict[str, Any],
) -> Dict[str, Any]:
    from services.faculty.faculty_profile_service import FacultyProfileService

    service = FacultyProfileService()
    post_out = service.run_profile_postprocess(
        faculty_id=int(faculty_id),
        postprocess_plan=dict(postprocess_plan or {}),
        request_email=str(email or "").strip().lower(),
    )
    updated = service.get_faculty_profile(faculty_id=int(faculty_id))
    return {
        "email": str(email or "").strip().lower(),
        "faculty_id": int(faculty_id),
        "faculty": updated,
        "updated_keywords": ((updated or {}).get("all_keywords") or {}),
        **dict(post_out or {}),
    }


def _maybe_send_result_email(
    *,
    recipient_emails: List[str],
    result: Dict[str, Any],
    query: Optional[str] = None,
) -> Dict[str, Any]:
    if not recipient_emails:
        return {"attempted": False, "status": "skipped", "reason": "no_recipients"}
    if not isinstance(result, dict) or not result:
        return {"attempted": False, "status": "skipped", "reason": "empty_result"}

    try:
        from services.notifications import SesEmailService
        from services.notifications.ses_email_service import SesEmailAttachment
        from services.notifications.pdf_builder import (
            build_pdf_filename,
            build_styled_text_pdf_bytes,
        )
        from services.notifications.justification_email_builder import build_justification_email

        content = build_justification_email(result=result, query=query)
        if content is None:
            return {
                "attempted": False,
                "status": "skipped",
                "reason": "no_justification_content",
                "to": recipient_emails,
            }

        attachments: List[SesEmailAttachment] = []
        attachment_errors: List[str] = []
        attachment_text = str(content.attachment_text_body or content.text_body or "").strip()

        if attachment_text:
            try:
                pdf_bytes = build_styled_text_pdf_bytes(attachment_text)
                attachments.append(
                    SesEmailAttachment(
                        filename=build_pdf_filename(content.subject),
                        content_bytes=pdf_bytes,
                        content_type="application/pdf",
                    )
                )
            except Exception as e:
                attachment_errors.append(f"pdf:{type(e).__name__}: {e}")

        send_out = SesEmailService().send_email(
            to_addresses=recipient_emails,
            subject=content.subject,
            text_body=content.text_body,
            html_body=content.html_body,
            attachments=attachments,
        )
        out = {
            "attempted": True,
            "status": "sent",
            "to": recipient_emails,
            "subject": content.subject,
            "message_id": send_out.get("message_id"),
        }
        if attachments:
            out["attachments"] = [
                {"filename": a.filename, "content_type": a.content_type}
                for a in attachments
            ]
            pdfs = [a for a in attachments if str(a.content_type).lower() == "application/pdf"]
            if pdfs:
                out["pdf_attachment"] = {
                    "included": True,
                    "filename": pdfs[0].filename,
                }
        elif attachment_errors:
            out["pdf_attachment"] = {
                "included": False,
                "error": "; ".join(attachment_errors),
            }
        if attachment_errors and attachments:
            out["attachment_errors"] = list(attachment_errors)
        return out
    except RuntimeError as e:
        logger.warning("SES justification email send failed: %s", e)
        return {
            "attempted": True,
            "status": "error",
            "to": recipient_emails,
            "error": f"{type(e).__name__}: {e}",
        }
    except Exception as e:
        logger.exception("SES justification email send failed")
        return {
            "attempted": True,
            "status": "error",
            "to": recipient_emails,
            "error": f"{type(e).__name__}: {e}",
        }


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
    actor_email, actor_role = _get_request_actor()
    primary_email = _normalize_email(body.get("email"))
    request_emails = _normalize_email_list(_to_email_list(body.get("emails")))
    if not primary_email and len(request_emails) == 1:
        primary_email = request_emails[0]
    if not primary_email and not request_emails and actor_email and actor_role != "admin":
        primary_email = _normalize_email(actor_email)
    thread_id = _resolve_chat_thread_id(
        body=body,
        primary_email=primary_email,
        emails=request_emails,
    )

    def emit(event_name: str, payload: dict) -> str:
        return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @stream_with_context
    def generate():
        print("chat.start")
        start_time = time.time()

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
            email=primary_email,
            emails=request_emails,
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
            agency_filter=body.get("agency_filter") or body.get("agency"),
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
                    "elapsed_seconds": round(time.time() - start_time, 2),
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
            result = decision_out.get("result") or {}
            reported_missing = _to_email_list(result.get("missing_emails"))
            db_missing = _split_existing_vs_missing_db_emails(reported_missing).get("missing") if reported_missing else []
            if db_missing:
                info_type = "email_not_in_db" if len(db_missing) == 1 else "emails_not_in_db"
                msg = (
                    "This faculty email was not found in DB. Please upload reference profile data."
                    if len(db_missing) == 1
                    else "Some faculty emails were not found in DB. Please upload reference profile data."
                )
                yield emit(
                    "request_info",
                    {
                        "type": info_type,
                        "message": msg,
                        "received_emails": reported_missing,
                        "emails_missing_in_db": db_missing,
                        "orchestrator": decision_out,
                    },
                )
                return
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
                "elapsed_seconds": round(time.time() - start_time, 2),
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
    Resolve + optionally ingest faculty.
    Returns:
      {
        "email_to_fid": {email: faculty_id},
        "failed_emails": [email, ...],
      }
    Skips gracefully if emails is empty.
    """
    if not emails:
        return {"email_to_fid": {}, "failed_emails": []}
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
        failed = _to_email_list(result.get("failed"))
        unresolved = [e for e in _to_email_list(emails) if e not in email_to_fid]
        for e in unresolved:
            if e not in failed:
                failed.append(e)
        return {"email_to_fid": email_to_fid, "failed_emails": failed}
    except Exception:
        logger.exception("_resolve_faculty_ids_for_team failed")
        return {"email_to_fid": {}, "failed_emails": _to_email_list(emails)}


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
        start_time = time.time()
        if not grant_link and not grant_title:
            yield _sse("request_info", {"type": "missing_grant", "message": "Please provide a grant link or title."})
            return
        if not emails:
            yield _sse("request_info", {"type": "missing_emails", "message": "Please provide at least one existing team member email."})
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
        resolve_out = _resolve_faculty_ids_for_team(emails, osu_url_map, cv_pdf_map)
        email_to_fid = dict(resolve_out.get("email_to_fid") or {})
        failed_emails = _to_email_list(resolve_out.get("failed_emails"))
        if failed_emails:
            info_type = "email_not_in_db" if len(failed_emails) == 1 else "emails_not_in_db"
            msg = (
                "This faculty email was not found in DB. Please add reference profile data first."
                if len(failed_emails) == 1
                else "Some faculty emails were not found in DB. Please add reference profile data first."
            )
            yield _sse("request_info", {
                "type": info_type,
                "message": msg,
                "received_emails": _to_email_list(emails),
                "emails_missing_in_db": failed_emails,
            })
            return
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
            "elapsed_seconds": round(time.time() - start_time, 2),
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
        start_time = time.time()
        if not grant_link and not grant_title:
            yield _sse("request_info", {"type": "missing_grant", "message": "Please provide a grant link or title."})
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
            resolve_out = _resolve_faculty_ids_for_team(emails, osu_url_map, cv_pdf_map)
            email_to_fid = dict(resolve_out.get("email_to_fid") or {})
            failed_emails = _to_email_list(resolve_out.get("failed_emails"))
            if failed_emails:
                info_type = "email_not_in_db" if len(failed_emails) == 1 else "emails_not_in_db"
                msg = (
                    "This faculty email was not found in DB. Please add reference profile data first."
                    if len(failed_emails) == 1
                    else "Some faculty emails were not found in DB. Please add reference profile data first."
                )
                yield _sse("request_info", {
                    "type": info_type,
                    "message": msg,
                    "received_emails": _to_email_list(emails),
                    "emails_missing_in_db": failed_emails,
                })
                return
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
            "elapsed_seconds": round(time.time() - start_time, 2),
        })

    return Response(generate(), mimetype="text/event-stream", headers=_sse_headers())


@app.post("/api/notifications/email-justification")
def email_justification():
    body = request.get_json(silent=True) or {}
    print("body")
    print(body)
    requested_recipients = _collect_recipient_emails(body)
    payload = _extract_frontend_justification_payload(body)
    query = str(body.get("query") or body.get("message") or "").strip() or None

    if not isinstance(payload, dict) or not payload:
        return {
            "ok": False,
            "error": "No justification result payload provided.",
            "email_delivery": {"attempted": False, "status": "skipped", "reason": "empty_result"},
        }, 400

    print(f"email_justification.requested_recipients={requested_recipients}")
    print(f"email_justification.delivery_recipient={FORCED_JUSTIFICATION_RECIPIENT}")

    delivery_recipients = [FORCED_JUSTIFICATION_RECIPIENT]
    delivery = _maybe_send_result_email(
        recipient_emails=delivery_recipients,
        result=payload,
        query=query,
    )
    print(
        "email_justification.status="
        f"{delivery.get('status')} "
        f"message_id={delivery.get('message_id')} "
        f"error={delivery.get('error')}"
    )
    return {
        "ok": delivery.get("status") == "sent",
        "requested_recipients": requested_recipients,
        "delivery_recipients": delivery_recipients,
        "email_delivery": delivery,
    }, (200 if delivery.get("status") in {"sent", "skipped"} else 502)


@app.get("/api/faculty")
def get_faculty_profiles():
    faculty_id = _to_optional_int(request.args.get("faculty_id"))
    email = str(request.args.get("email") or "").strip().lower() or None
    publication_year_from = _to_optional_int(
        request.args.get("publication_year_from") or request.args.get("year_from")
    )
    publication_year_to = _to_optional_int(
        request.args.get("publication_year_to") or request.args.get("year_to")
    )

    limit_raw = _to_optional_int(request.args.get("limit"))
    offset_raw = _to_optional_int(request.args.get("offset"))
    limit = max(1, min(int(limit_raw or 50), 200))
    offset = max(0, int(offset_raw or 0))

    try:
        from services.faculty.faculty_profile_service import FacultyProfileService

        service = FacultyProfileService()
        if faculty_id or email:
            faculty = service.get_faculty_profile(
                faculty_id=faculty_id,
                email=email,
                publication_year_from=publication_year_from,
                publication_year_to=publication_year_to,
            )
            if not faculty:
                return {
                    "ok": False,
                    "error": "Faculty not found.",
                    "faculty_id": faculty_id,
                    "email": email,
                }, 404
            return {"ok": True, "faculty": faculty}, 200

        rows = service.list_faculty_profiles(
            limit=limit,
            offset=offset,
            publication_year_from=publication_year_from,
            publication_year_to=publication_year_to,
        )
        return {
            "ok": True,
            "count": len(rows),
            "limit": limit,
            "offset": offset,
            "faculty": rows,
        }, 200
    except ValueError as e:
        return {"ok": False, "error": str(e)}, 400
    except Exception as e:
        logger.exception("GET /api/faculty failed")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}, 500


@app.get("/api/faculty/<int:faculty_id>")
def get_faculty_profile_by_id(faculty_id: int):
    publication_year_from = _to_optional_int(
        request.args.get("publication_year_from") or request.args.get("year_from")
    )
    publication_year_to = _to_optional_int(
        request.args.get("publication_year_to") or request.args.get("year_to")
    )
    try:
        from services.faculty.faculty_profile_service import FacultyProfileService

        service = FacultyProfileService()
        faculty = service.get_faculty_profile(
            faculty_id=faculty_id,
            publication_year_from=publication_year_from,
            publication_year_to=publication_year_to,
        )
        if not faculty:
            return {
                "ok": False,
                "error": "Faculty not found.",
                "faculty_id": int(faculty_id),
            }, 404
        return {"ok": True, "faculty": faculty}, 200
    except ValueError as e:
        return {"ok": False, "error": str(e)}, 400
    except Exception as e:
        logger.exception("GET /api/faculty/<faculty_id> failed")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}, 500

def _get_request_actor() -> tuple[str, str]:
    actor_email = str(
        request.headers.get("X-User-Email")
        or request.headers.get("x-user-email")
        or ""
    ).strip().lower()

    actor_role = str(
        request.headers.get("X-User-Role")
        or request.headers.get("x-user-role")
        or "normal_user"
    ).strip().lower()

    return actor_email, actor_role


def _can_access_target_email(actor_email: str, actor_role: str, target_email: str) -> bool:
    if actor_role == "admin":
        return True
    return actor_email == str(target_email or "").strip().lower()

@app.get("/api/faculty/by-email")
@app.post("/api/faculty/by-email")
def get_faculty_profile_by_email():
    body = request.get_json(silent=True) or {}
    email = str(
        body.get("email")
        or request.args.get("email")
        or ""
    ).strip().lower()
    publication_year_from = _to_optional_int(
        body.get("publication_year_from")
        or body.get("year_from")
        or request.args.get("publication_year_from")
        or request.args.get("year_from")
    )
    publication_year_to = _to_optional_int(
        body.get("publication_year_to")
        or body.get("year_to")
        or request.args.get("publication_year_to")
        or request.args.get("year_to")
    )

    if not email:
        return {
            "ok": False,
            "error": "email is required.",
        }, 400

    actor_email, actor_role = _get_request_actor()
    if not _can_access_target_email(actor_email, actor_role, email):
        return {
            "ok": False,
            "error": "You are not allowed to view this profile.",
        }, 403

    try:
        from services.faculty.faculty_profile_service import FacultyProfileService

        service = FacultyProfileService()
        faculty = service.get_faculty_profile(
            email=email,
            publication_year_from=publication_year_from,
            publication_year_to=publication_year_to,
        )
        if not faculty:
            return {
                "ok": False,
                "error": "Faculty not found.",
                "email": email,
            }, 404
        return {"ok": True, "faculty": faculty}, 200
    except ValueError as e:
        return {"ok": False, "error": str(e)}, 400
    except Exception as e:
        logger.exception("POST /api/faculty/by-email failed")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}, 500


@app.patch("/api/faculty/by-email")
def edit_faculty_profile_by_email():
    body = request.get_json(silent=True) or {}
    email = str(body.get("email") or "").strip().lower()
    if not email:
        return {
            "ok": False,
            "error": "email is required.",
        }, 400
    
    actor_email, actor_role = _get_request_actor()
    if not _can_access_target_email(actor_email, actor_role, email):
        return {
            "ok": False,
            "error": "You are not allowed to edit this profile.",
        }, 403

    #print(body)
    basic_info = body.get("basic_info") or {}
    data_from = body.get("data_from") or {}
    all_keywords = body.get("all_keywords") if "all_keywords" in body else None
    keyword_source = body.get("keyword_source")
    force_regenerate_keywords = _to_optional_bool(body.get("force_regenerate_keywords"))
    if "force_regenerate_keywords" in body and force_regenerate_keywords is None:
        return {
            "ok": False,
            "error": "force_regenerate_keywords must be a boolean.",
        }, 400
    async_requested = _to_optional_bool(body.get("async"))
    if "async" in body and async_requested is None:
        return {
            "ok": False,
            "error": "async must be a boolean.",
        }, 400
    run_async = True if async_requested is None else bool(async_requested)

    try:
        from services.faculty.faculty_profile_service import FacultyProfileService

        service = FacultyProfileService()
        if not run_async:
            out = service.edit_faculty_profile(
                email=email,
                basic_info=basic_info,
                data_from=data_from,
                all_keywords=all_keywords,
                keyword_source=keyword_source,
                force_regenerate_keywords=force_regenerate_keywords,
                run_postprocess=True,
            )
            return {"ok": True, "async": False, **out}, 200

        out = service.edit_faculty_profile(
            email=email,
            basic_info=basic_info,
            data_from=data_from,
            all_keywords=all_keywords,
            keyword_source=keyword_source,
            force_regenerate_keywords=force_regenerate_keywords,
            run_postprocess=False,
        )
        faculty_id = int(out.get("faculty_id") or 0)
        postprocess_plan = dict(out.get("postprocess_plan") or {})

        manager = _get_faculty_profile_job_manager()
        job_id = manager.submit(
            job_type="faculty_profile_postprocess",
            payload={
                "email": email,
                "faculty_id": int(faculty_id),
                "postprocess_plan": postprocess_plan,
            },
            run_fn=lambda: _run_faculty_profile_postprocess_job(
                email=email,
                faculty_id=int(faculty_id),
                postprocess_plan=postprocess_plan,
            ),
        )
        return {
            "ok": True,
            "async": True,
            "job_id": str(job_id),
            "job_status": "queued",
            **out,
        }, 202
    except LookupError as e:
        return {"ok": False, "error": str(e), "email": email}, 404
    except ValueError as e:
        return {"ok": False, "error": str(e), "email": email}, 400
    except Exception as e:
        logger.exception("PATCH /api/faculty/by-email failed")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}, 500


@app.get("/api/faculty/by-email/jobs/<job_id>")
def get_faculty_profile_update_job(job_id: str):
    key = str(job_id or "").strip()
    if not key:
        return {"ok": False, "error": "job_id is required."}, 400

    manager = _get_faculty_profile_job_manager()
    row = manager.get(key)
    if not row:
        return {"ok": False, "error": "Job not found.", "job_id": key}, 404
    return {"ok": True, "job": row}, 200




@app.post("/api/faculty/create")
def create_faculty_profiles():
    # Parse multipart/form-data for faculty creation
    actor_email, actor_role = _get_request_actor()
    if actor_role != "admin":
        return {
            "ok": False,
            "error": "Only admins can create faculty.",
        }, 403

    content_type = (request.content_type or "").lower()
    if "multipart/form-data" not in content_type:
        return {
            "ok": False,
            "error": "Expected multipart/form-data.",
        }, 400

    body: dict = {}
    for key in request.form:
        values = request.form.getlist(key)
        body[key] = values[0] if len(values) == 1 else values

    # JSON-decode emails if needed
    for key in ("emails",):
        raw = body.get(key)
        if isinstance(raw, str):
            try:
                body[key] = json.loads(raw)
            except Exception:
                pass

    emails = _to_email_list(body.get("emails") or body.get("email"))
    if not emails:
        return {
            "ok": False,
            "error": "At least one email is required.",
        }, 400


    # Check if any faculty already exist
    from db.db_conn import SessionLocal
    with SessionLocal() as session:
        from dao.faculty_dao import FacultyDAO
        dao = FacultyDAO(session)
        existing_faculty = []
        for email in emails:
            if dao.get_by_email(email):
                existing_faculty.append(email)
        
        if existing_faculty:
            return {
                "ok": False,
                "error": f"Faculty already exists in the database: {', '.join(existing_faculty)}",
            }, 400

    # Build OSU URL map
    osu_url_map: dict = {}
    for em, url in zip(request.form.getlist("osu_url_email"), request.form.getlist("osu_url_value")):
        em, url = em.strip().lower(), url.strip()
        if em and url:
            osu_url_map[em] = url

    # Validate required OSU URLs
    missing_osu = [e for e in emails if e not in osu_url_map]
    if missing_osu:
        return {
            "ok": False,
            "error": f"OSU profile URL is required for: {', '.join(missing_osu)}",
        }, 400

    # Build CV PDF map (optional)
    cv_pdf_map: dict = {}
    for em, fobj in zip(request.form.getlist("cv_email"), request.files.getlist("cv_file")):
        em = em.strip().lower()
        if em:
            cv_pdf_map[em] = fobj.read()

    try:
        from services.agent_v2.agents.faculty_context_agent import FacultyContextAgent
        from services.keywords.keyword_generator import FacultyKeywordGenerator
        from services.context_retrieval.context_generator import ContextGenerator

        logger.info(f"Creating faculty for emails: {emails}, osu_url_map: {osu_url_map}")
        agent = FacultyContextAgent()
        keyword_generator = FacultyKeywordGenerator(context_generator=ContextGenerator())
        result = agent.resolve_and_ingest_faculties(
            emails=emails,
            osu_url_map=osu_url_map,
            cv_pdf_map=cv_pdf_map,
            keyword_generator=keyword_generator,
            run_matching_for_new=True,
        )
        logger.info(f"Result: {result}")
        # Count created faculty (those with faculty_id)
        created_count = len(result.get("newly_added", []))
        return {
            "ok": True,
            "pipeline_status": "done",
            "message": f"Created {created_count} faculty profile(s).",
            "created": created_count,
            "resolved": result.get("resolved", []),
            "newly_added": result.get("newly_added", []),
            "failed": result.get("failed", []),
            "match_rows_upserted": int(result.get("match_rows_upserted") or 0),
            "matching_failed": result.get("matching_failed_emails", []),
            "matching_ran": bool(result.get("matching_ran")),
        }, 200
    except Exception as e:
        logger.exception("POST /api/faculty/create failed")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}, 500


@app.post("/api/auth/login")
def login():
    body = request.get_json(silent=True) or {}
    email = str(body.get("email") or "").strip().lower()
    password = str(body.get("password") or "")

    print("login.hit", email)

    if not email or not password:
        return {
            "ok": False,
            "error": "Email and password are required."
        }, 400

    if not email.endswith("@oregonstate.edu"):
        return {
            "ok": False,
            "error": "Only @oregonstate.edu email addresses are allowed."
        }, 403

    try:
        with SessionLocal() as sess:
            print("login.db_session_open")

            faculty_dao = FacultyDAO(sess)
            user_dao = UserDAO(sess)

            print("login.before_faculty_lookup")
            faculty = faculty_dao.get_by_email(email)
            print("login.after_faculty_lookup", faculty is not None)

            if not faculty:
                return {
                    "ok": False,
                    "error": "Email is not authorized."
                }, 403

            print("login.before_user_lookup")
            user = user_dao.get_by_email(email)
            print("login.after_user_lookup", user is not None)

            if user is None or not user.password_hash:
                return {
                    "ok": False,
                    "error": "No password is set for this account yet."
                }, 401

            print("login.before_password_check")
            ok = bcrypt.checkpw(
                password.encode("utf-8"),
                user.password_hash.encode("utf-8")
            )
            print("login.after_password_check", ok)

            if not ok:
                return {
                    "ok": False,
                    "error": "Invalid password."
                }, 401

            print("login.success_return")
            return {
                "ok": True,
                "message": "Login successful.",
                "user": {
                    "email": user.email,
                    "role": user.role,
                    "faculty_id": user.faculty_id,
                }
            }, 200

    except Exception as e:
        logger.exception("POST /api/auth/login failed")
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}"
        }, 500


@app.post("/api/auth/signup")
def signup():
    body = request.get_json(silent=True) or {}
    email = str(body.get("email") or "").strip().lower()
    password = str(body.get("password") or "")
    confirm_password = str(body.get("confirm_password") or "")

    if not email or not password or not confirm_password:
        return {"ok": False, "error": "Email, password, and confirm password are required."}, 400

    if not email.endswith("@oregonstate.edu"):
        return {"ok": False, "error": "Use your Oregon State email."}, 403

    if password != confirm_password:
        return {"ok": False, "error": "Passwords do not match."}, 400

    if len(password) < 6:
        return {"ok": False, "error": "Password must be at least 6 characters."}, 400

    try:
        with SessionLocal() as sess:
            faculty_dao = FacultyDAO(sess)
            user_dao = UserDAO(sess)

            existing_faculty = faculty_dao.get_by_email(email)
            existing_user = user_dao.get_by_email(email)

            if existing_faculty or existing_user:
                return {"ok": False, "error": "An account with this email already exists. Please log in."}, 400

            faculty = Faculty(
                name="",
                email=email,
                source_url="pending"
            )
            sess.add(faculty)
            sess.flush()

            password_hash = bcrypt.hashpw(
                password.encode("utf-8"),
                bcrypt.gensalt()
            ).decode("utf-8")

            user = user_dao.create_user(
                faculty_id=faculty.faculty_id,
                email=email,
                password_hash=password_hash,
                role="normal_user"
            )

            sess.commit()

            return {
                "ok": True,
                "message": "Signup successful.",
                "user": {
                    "email": user.email,
                    "role": user.role,
                    "faculty_id": user.faculty_id,
                }
            }, 200
    except Exception as e:
        logger.exception("POST /api/auth/signup failed")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
