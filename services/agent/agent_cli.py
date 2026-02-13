from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, List

from dao.faculty_dao import FacultyDAO
from db.db_conn import SessionLocal
from services.faculty.enrich_profile import enrich_new_faculty
from services.keywords.generate_keywords import (
    generate_faculty_keywords_for_id,
    _extract_domains_from_keywords,
    _embed_domain_bucket,
)
from services.search.search_grants import generate_query_keywords
from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from config import get_llm_client
from services.prompts.matching_prompt import MATCH_PROMPT
from dto.llm_response_dto import LLMMatchOut
from utils.keyword_accessor import keywords_for_matching, requirements_indexed
from services.matching.hybrid_matcher import covered_to_grouped, missing_to_grouped
from services.justification.generate_justification import generate_faculty_recs, print_faculty_recs
from services.agent.agent import run_agent
from services.agent.tools import call_tool


logger = logging.getLogger(__name__)


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def parse_structured_prompt(prompt_text: str) -> Dict[str, Optional[str]]:
    fields = {
        "name": None,
        "email": None,
        "osu_webpage": None,
        "google_scholar": None,
        "personal_website": None,
        "query_text": "",
    }
    if not prompt_text:
        return fields

    patterns = {
        "name": re.compile(r"^\s*name\s*:\s*(.+?)\s*$", re.IGNORECASE),
        "email": re.compile(r"^\s*email\s*:\s*(.+?)\s*$", re.IGNORECASE),
        "osu_webpage": re.compile(r"^\s*osu\s*webpage\s*:\s*(.+?)\s*$", re.IGNORECASE),
        "google_scholar": re.compile(r"^\s*google\s*scholar\s*:\s*(.+?)\s*$", re.IGNORECASE),
        "personal_website": re.compile(r"^\s*personal\s*website\s*:\s*(.+?)\s*$", re.IGNORECASE),
    }

    query_lines: List[str] = []
    for line in prompt_text.splitlines():
        matched = False
        for key, pat in patterns.items():
            m = pat.match(line)
            if not m:
                continue
            val = (m.group(1) or "").strip()
            if val:
                fields[key] = val
            matched = True
            break
        if not matched and line.strip():
            query_lines.append(line)

    fields["query_text"] = "\n".join(query_lines).strip()
    return fields


def _should_use_agent_planner(prompt_text: str, parsed: Dict[str, Optional[str]]) -> bool:
    # If there's no structured section and the prompt looks like a collaborator request,
    # use the planner-based agent flow.
    has_structured = any(
        parsed.get(k)
        for k in ("name", "email", "osu_webpage", "google_scholar", "personal_website")
    )
    if has_structured:
        return False
    text = (prompt_text or "").lower()
    keywords = (
        "collaborator",
        "collaborators",
        "team",
        "add",
        "additional",
        "add people",
        "co-pi",
        "co pi",
        "co-investigator",
        "co investigator",
        "partner",
        "coauthor",
        "co-author",
    )
    return any(k in text for k in keywords)


def _inject_opportunity_hint(prompt_text: str) -> str:
    m = re.search(r"\bopportunity\s+([^\n\.\,]+)", prompt_text, flags=re.IGNORECASE)
    if not m:
        return prompt_text
    title = m.group(1).strip()
    if not title:
        return prompt_text
    # Add a structured hint line to help the planner/tooling
    return f"{prompt_text.rstrip()}\nOpportunity ID or title: {title}\n"


def _parse_collab_intent_inputs(prompt_text: str) -> Dict[str, Any]:
    text = prompt_text or ""
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    emails = list(dict.fromkeys([e.strip().lower() for e in emails if e.strip()]))

    need_y = None
    for pat in (
        r"\bneed\s+(\d+)\s+more",
        r"\bneed\s+(\d+)\s+additional",
        r"\badd\s+(\d+)\b",
    ):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                need_y = int(m.group(1))
                break
            except Exception:
                pass

    opp_title = None
    m = re.search(r"\bopportunity\s+([^\n\.\,]+)", text, flags=re.IGNORECASE)
    if m:
        opp_title = m.group(1).strip()

    return {
        "faculty_emails": emails,
        "need_y": need_y,
        "opp_ids": [opp_title] if opp_title else None,
    }


def prompt_for_email() -> str:
    while True:
        raw = input("Email is required. Please enter your OSU email (e.g., name@oregonstate.edu): ").strip()
        if not raw:
            continue
        if not _EMAIL_RE.match(raw):
            continue
        return raw.strip().lower()


def prompt_for_url() -> str:
    while True:
        raw = input(
            "No profile URL provided. Enter at least one URL (OSU webpage, Google Scholar, or Personal website): "
        ).strip()
        if not raw:
            continue
        if raw.startswith("http://") or raw.startswith("https://"):
            return raw


def _faculty_exists(email: str) -> bool:
    with SessionLocal() as sess:
        dao = FacultyDAO(sess)
        return dao.get_faculty_id_by_email(email) is not None


def _classify_url(url: str) -> str:
    u = (url or "").lower()
    if "engineering.oregonstate.edu/people/" in u:
        return "osu"
    if "scholar.google" in u:
        return "scholar"
    return "personal"


def _ensure_faculty_record(
    *,
    email: str,
    name: Optional[str],
    osu_webpage: Optional[str],
    personal_website: Optional[str],
    google_scholar: Optional[str],
) -> tuple[Optional[int], bool]:
    with SessionLocal() as sess:
        dao = FacultyDAO(sess)
        fac = dao.get_by_email(email)
        if fac:
            logger.info(
                "Faculty already present in DB",
                extra={"email": email, "faculty_id": fac.faculty_id},
            )
            if osu_webpage and osu_webpage != fac.source_url:
                fac.source_url = osu_webpage
                sess.commit()
            return fac.faculty_id, False

        logger.info(
            "Faculty not found; creating new faculty record",
            extra={"email": email},
        )
        from dto.faculty_dto import FacultyDTO  # local import to avoid cycles

        dto = FacultyDTO(
            email=email,
            name=name,
            source_url=osu_webpage or "",
            biography=None,
            expertise=None,
        )
        fac = dao.upsert_faculty(dto)
        sess.commit()
        return (fac.faculty_id if fac else None), True


def _merge_keywords(fac_kw: dict, query_kw: dict) -> dict:
    out = {"research": {"domain": [], "specialization": []}, "application": {"domain": [], "specialization": []}}

    def _merge_domains(sec: str):
        seen = set()
        merged = []
        for src in (fac_kw, query_kw):
            for d in (src.get(sec) or {}).get("domain", []) or []:
                t = str(d).strip()
                if not t or t in seen:
                    continue
                seen.add(t)
                merged.append(t)
        return merged

    def _merge_specs(sec: str):
        merged = {}
        for src in (fac_kw, query_kw):
            specs = (src.get(sec) or {}).get("specialization") or []
            for s in specs:
                if isinstance(s, dict) and "t" in s:
                    t = str(s.get("t")).strip()
                    w = float(s.get("w", 1.0))
                else:
                    t = str(s).strip()
                    w = 1.0
                if not t:
                    continue
                prev = merged.get(t)
                merged[t] = w if prev is None else max(prev, w)
        return [{"t": t, "w": w} for t, w in merged.items()]

    for sec in ("research", "application"):
        out[sec]["domain"] = _merge_domains(sec)
        out[sec]["specialization"] = _merge_specs(sec)

    return out


def _run_matching_for_faculty_query(
    *,
    faculty_id: int,
    combined_kw: dict,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    r_domains, a_domains = _extract_domains_from_keywords(combined_kw)
    r_vec = _embed_domain_bucket(r_domains)
    a_vec = _embed_domain_bucket(a_domains)

    if r_vec is None and a_vec is None:
        return []

    llm = get_llm_client().build()
    chain = MATCH_PROMPT | llm.with_structured_output(LLMMatchOut)

    fac_kw_simple = keywords_for_matching(combined_kw)
    fac_json = json.dumps(fac_kw_simple, ensure_ascii=False)

    with SessionLocal() as sess:
        match_dao = MatchDAO(sess)
        opp_dao = OpportunityDAO(sess)

        candidates = match_dao.topk_opps_for_query(
            research_vec=r_vec,
            application_vec=a_vec,
            k=max(top_k * 5, top_k),
        )
        if not candidates:
            return []

        opp_ids = [oid for (oid, _) in candidates]
        opps = opp_dao.read_opportunities_by_ids_with_relations(opp_ids)
        opp_map = {o.opportunity_id: o for o in opps}

        out_rows = []
        results = []
        for opp_id, domain_sim in candidates:
            opp = opp_map.get(opp_id)
            if not opp or not opp.keyword:
                continue

            opp_kw = keywords_for_matching(getattr(opp.keyword, "keywords", {}) or {})
            req_idx = requirements_indexed(opp_kw)
            opp_req_idx_json = json.dumps(req_idx, ensure_ascii=False)

            scored: LLMMatchOut = chain.invoke(
                {
                    "faculty_kw_json": fac_json,
                    "requirements_indexed": opp_req_idx_json,
                }
            )

            out_rows.append(
                {
                    "grant_id": opp_id,
                    "faculty_id": faculty_id,
                    "domain_score": float(domain_sim),
                    "llm_score": float(scored.llm_score),
                    "reason": (scored.reason or "").strip(),
                    "covered": covered_to_grouped(scored.covered),
                    "missing": missing_to_grouped(scored.missing),
                }
            )

            results.append(
                {
                    "opportunity_id": opp_id,
                    "title": opp.opportunity_title,
                    "agency": opp.agency_name,
                    "domain_score": float(domain_sim),
                    "llm_score": float(scored.llm_score),
                    "reason": (scored.reason or "").strip(),
                }
            )

        if out_rows:
            match_dao.upsert_matches(out_rows)
            sess.commit()

        results.sort(key=lambda r: (r.get("llm_score", 0.0), r.get("domain_score", 0.0)), reverse=True)
        return results[:top_k]


def _print_fallback_matches(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No matching opportunities found.")
        return

    def score_val(r: Dict[str, Any]) -> float:
        v = r.get("llm_score")
        return float(v) if isinstance(v, (int, float)) else 0.0

    results = list(results)
    results.sort(key=score_val, reverse=True)

    top_n = min(7, len(results))
    top = results[:top_n]
    rest = results[top_n:]

    print("\nTop Matches:\n")
    for i, r in enumerate(top, start=1):
        title = r.get("title") or "Untitled"
        agency = r.get("agency") or "Unknown agency"
        score = r.get("llm_score")
        score_txt = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
        reason = r.get("reason") or ""
        print(f"{i}. {title} ({agency}) - Score: {score_txt}")
        if reason:
            print(f"   - {reason}")
        print("")

    if rest:
        print("Additional Opportunities:\n")
        for j, r in enumerate(rest, start=top_n + 1):
            title = r.get("title") or "Untitled"
            agency = r.get("agency") or "Unknown agency"
            score = r.get("llm_score")
            score_txt = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
            print(f"{j}. {title} ({agency}) - Score: {score_txt}")

    print("\nTop results:")
    for idx, r in enumerate(results, start=1):
        title = r.get("title") or "Untitled"
        score = r.get("llm_score")
        score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        print(f"{idx}. {title} (score: {score_txt})")


def _save_state(state_path: str, state: Dict[str, Any]) -> None:
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_state(state_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not state_path:
        return None
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the grant-matching agent")
    parser.add_argument("--prompt", required=True, help="User prompt")
    parser.add_argument("--refresh_profile", action="store_true", help="Force refresh of faculty profile scrape")
    parser.add_argument(
        "--state-json",
        default=None,
        help="Optional path to JSON state file to continue a prior interaction",
    )
    args = parser.parse_args()

    parsed = parse_structured_prompt(args.prompt)

    if _should_use_agent_planner(args.prompt, parsed):
        # Strong direct path for collaborator requests
        collab = _parse_collab_intent_inputs(args.prompt)
        # Ensure required collaborator inputs; ask interactively if missing
        tool_input = {
            "faculty_emails": collab.get("faculty_emails") or [],
            "need_y": collab.get("need_y"),
            "opp_ids": collab.get("opp_ids"),
        }
        while True:
            if not tool_input["faculty_emails"]:
                resp = input("Enter current team emails (comma-separated): ").strip()
                if resp:
                    tool_input["faculty_emails"] = [e.strip() for e in resp.split(",") if e.strip()]
                    continue
            if tool_input.get("need_y") is None:
                resp = input("How many additional collaborators do you need to add? ").strip()
                if resp.isdigit():
                    tool_input["need_y"] = int(resp)
                    continue

            result = call_tool("find_additional_collaborators", tool_input)
            state_path = args.state_json or ".agent_state.json"
            _save_state(state_path, {"tool_result": result})
            if isinstance(result, dict) and result.get("error"):
                err = result["error"]
                msg = err.get("message") or "Missing information."
                missing = err.get("missing_fields") or []
                print(msg)
                if "faculty_emails" in missing:
                    tool_input["faculty_emails"] = []
                    continue
                if "need_y" in missing:
                    tool_input["need_y"] = None
                    continue
                return
            elif isinstance(result, str):
                print(result)
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        # Planner path with interactive clarification loop (non-collaborator)
        prompt_text = _inject_opportunity_hint(args.prompt)
        state = _load_state(args.state_json)
        while True:
            result = run_agent(prompt_text, state=state)
            result_state = result.get("state") or {}

            state_path = args.state_json or ".agent_state.json"
            _save_state(state_path, result_state)

            if result.get("type") == "clarification":
                print(result.get("question") or "")
                user_reply = input("> ").strip()
                if not user_reply:
                    continue
                prompt_text = user_reply
                state = result_state
                continue

            if result.get("type") == "final":
                print(result.get("answer") or "")
                return

            print("Unexpected response type.")
            return

    email = (parsed.get("email") or "").strip().lower()
    used_email_fallback = False
    if not email:
        logger.warning("Email missing in prompt; requested interactively")
        email = prompt_for_email()
        used_email_fallback = True

    osu_url = parsed.get("osu_webpage")
    scholar_url = parsed.get("google_scholar")
    personal_url = parsed.get("personal_website")

    if not _faculty_exists(email) and not (osu_url or scholar_url or personal_url):
        logger.warning("No URL provided for new faculty; requested interactively")
        url = prompt_for_url()
        bucket = _classify_url(url)
        if bucket == "osu":
            osu_url = url
        elif bucket == "scholar":
            scholar_url = url
        else:
            personal_url = url

    logger.info(
        "Parsed structured prompt",
        extra={
            "email": email,
            "has_osu": bool(osu_url),
            "has_scholar": bool(scholar_url),
            "has_personal": bool(personal_url),
        },
    )

    faculty_id, created = _ensure_faculty_record(
        email=email,
        name=parsed.get("name"),
        osu_webpage=osu_url,
        personal_website=personal_url,
        google_scholar=scholar_url,
    )

    if created and faculty_id:
        enrich_new_faculty(
            email=email,
            faculty_id=faculty_id,
            osu_webpage=osu_url,
            personal_website=personal_url,
            google_scholar=scholar_url,
        )
    elif faculty_id and (osu_url or personal_url or scholar_url):
        # Refresh profile if stale or explicitly requested
        with SessionLocal() as sess:
            fac = FacultyDAO(sess).get_by_email(email)
            last = fac.profile_last_refreshed_at if fac else None
            if last and last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)

        fresh_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        if last and last >= fresh_cutoff and not args.refresh_profile:
            logger.info("Skipping scrape; profile fresh", extra={"email": email})
        else:
            if args.refresh_profile:
                logger.info("Refreshing profile due to flag", extra={"email": email})
            enrich_new_faculty(
                email=email,
                faculty_id=faculty_id,
                osu_webpage=osu_url,
                personal_website=personal_url,
                google_scholar=scholar_url,
            )

    query_text = (parsed.get("query_text") or "").strip()
    if not query_text:
        print("Missing query text. Provide free text after the structured section.")
        return

    # 1) Faculty keywords + embeddings
    fac_kw = generate_faculty_keywords_for_id(faculty_id) if faculty_id else None
    if not fac_kw:
        print("Failed to generate faculty keywords.")
        return

    # 2) Query keywords
    query_kw = generate_query_keywords(query_text, user_urls=None)

    # 3) Combine keywords and match
    combined_kw = _merge_keywords(fac_kw, query_kw)
    results = _run_matching_for_faculty_query(
        faculty_id=faculty_id,
        combined_kw=combined_kw,
        top_k=10,
    )

    # 4) Justification + output
    if results:
        try:
            out = generate_faculty_recs(email=email, k=10)
            if not getattr(out, "recommendations", None):
                print("No justifications generated by LLM; showing ranked matches instead.")
                _print_fallback_matches(results)
            else:
                print_faculty_recs(out, email, show_full_id=True)
        except Exception as exc:
            print(f"Failed to generate justifications: {exc}")
            _print_fallback_matches(results)
    else:
        print("No matching opportunities found.")

    # Save minimal state for inspection
    state_path = args.state_json or ".agent_state.json"
    _save_state(
        state_path,
        {
            "email": email,
            "faculty_id": faculty_id,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
