from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EmailContent:
    subject: str
    text_body: str
    html_body: str


def _safe_lines(raw: Any) -> List[str]:
    out: List[str] = []
    if isinstance(raw, list):
        for x in raw:
            s = str(x).strip()
            if s:
                out.append(s)
    return out


def _make_html(text: str) -> str:
    escaped = html.escape(text)
    return f"<html><body><pre style='font-family:Arial,sans-serif;white-space:pre-wrap'>{escaped}</pre></body></html>"


def _group_justification_text(
    *,
    result: Dict[str, Any],
    query: Optional[str],
) -> Optional[EmailContent]:
    just = result.get("group_justification") or {}
    if not isinstance(just, dict) or not just:
        return None

    title = str(result.get("opportunity_title") or result.get("opportunity_id") or "Grant Team Justification")
    subject = f"[GrantFetcher] Team Justification - {title}"
    lines: List[str] = []
    lines.append(f"Grant: {title}")
    if query:
        lines.append(f"Query: {query}")
    lines.append("")

    one_paragraph = str(just.get("one_paragraph") or "").strip()
    if one_paragraph:
        lines.append("Summary")
        lines.append(one_paragraph)
        lines.append("")

    members = result.get("suggested_team") or result.get("suggested_collaborators") or []
    existing_members = result.get("existing_team_details") or []
    if isinstance(members, list) and members:
        lines.append("Suggested Members")
        for m in members:
            if not isinstance(m, dict):
                continue
            name = str(m.get("name") or f"Faculty {m.get('faculty_id')}")
            email = str(m.get("email") or "").strip()
            llm_score = m.get("llm_score")
            domain_score = m.get("domain_score")
            if email:
                lines.append(f"- {name} ({email}) | llm={llm_score} domain={domain_score}")
            else:
                lines.append(f"- {name} | llm={llm_score} domain={domain_score}")
        lines.append("")

    if isinstance(existing_members, list) and existing_members:
        lines.append("Existing Team")
        for m in existing_members:
            if not isinstance(m, dict):
                continue
            name = str(m.get("name") or f"Faculty {m.get('faculty_id')}")
            email = str(m.get("email") or "").strip()
            if email:
                lines.append(f"- {name} ({email})")
            else:
                lines.append(f"- {name}")
        lines.append("")

    why_not = _safe_lines(just.get("why_not_working"))
    missing = _safe_lines((just.get("coverage") or {}).get("missing") if isinstance(just.get("coverage"), dict) else [])
    gaps = why_not + [x for x in missing if x not in why_not]
    if gaps:
        lines.append("Gaps")
        for g in gaps:
            lines.append(f"- {g}")
        lines.append("")

    recommendation = str(just.get("recommendation") or "").strip()
    if recommendation:
        lines.append("Recommendation")
        lines.append(recommendation)
        lines.append("")

    body = "\n".join(lines).strip()
    if not body:
        return None
    return EmailContent(subject=subject, text_body=body, html_body=_make_html(body))


def _one_to_one_recommendation_text(
    *,
    result: Dict[str, Any],
    query: Optional[str],
) -> Optional[EmailContent]:
    recommendation = result.get("recommendation") or {}
    recs = recommendation.get("recommendations") if isinstance(recommendation, dict) else None
    if not isinstance(recs, list) or not recs:
        return None

    faculty_name = str(recommendation.get("faculty_name") or result.get("faculty_email") or "Faculty")
    subject = f"[GrantFetcher] Grant Justification - {faculty_name}"
    lines: List[str] = []
    lines.append(f"Faculty: {faculty_name}")
    if query:
        lines.append(f"Query: {query}")
    lines.append("")

    for i, rec in enumerate(recs, start=1):
        if not isinstance(rec, dict):
            continue
        title = str(rec.get("title") or rec.get("opportunity_id") or "Untitled opportunity")
        lines.append(f"{i}. {title}")
        lines.append(f"   opportunity_id: {rec.get('opportunity_id')}")
        lines.append(f"   agency: {rec.get('agency')}")
        lines.append(f"   scores: llm={rec.get('llm_score')} domain={rec.get('domain_score')} fit={rec.get('fit_label')}")
        why = rec.get("why_match") if isinstance(rec.get("why_match"), dict) else {}
        summary = str((why or {}).get("summary") or "").strip()
        if summary:
            lines.append(f"   why: {summary}")
        pitch = str(rec.get("suggested_pitch") or "").strip()
        if pitch:
            lines.append(f"   suggested_pitch: {pitch}")
        lines.append("")

    body = "\n".join(lines).strip()
    if not body:
        return None
    return EmailContent(subject=subject, text_body=body, html_body=_make_html(body))


def _single_recommendation_row_text(
    *,
    result: Dict[str, Any],
    query: Optional[str],
) -> Optional[EmailContent]:
    # Supports FE payloads where justification_result is a single recommendation row.
    if not any(result.get(k) for k in ("opportunity_id", "title")):
        return None

    faculty_name = str(result.get("faculty_name") or result.get("faculty_email") or "Faculty")
    title = str(result.get("title") or result.get("opportunity_id") or "Untitled opportunity")
    subject = f"[GrantFetcher] Grant Justification - {faculty_name}"

    lines: List[str] = []
    lines.append(f"Faculty: {faculty_name}")
    if query:
        lines.append(f"Query: {query}")
    lines.append("")

    lines.append(f"1. {title}")
    lines.append(f"   opportunity_id: {result.get('opportunity_id')}")
    lines.append(f"   agency: {result.get('agency')}")
    lines.append(
        "   scores: "
        f"llm={result.get('llm_score')} "
        f"domain={result.get('domain_score')} "
        f"fit={result.get('fit_label')} "
        f"score={result.get('score')}"
    )

    why = result.get("why_match") if isinstance(result.get("why_match"), dict) else {}
    summary = str((why or {}).get("summary") or "").strip()
    if summary:
        lines.append(f"   why: {summary}")

    pitch = str(result.get("suggested_pitch") or "").strip()
    if pitch:
        lines.append(f"   suggested_pitch: {pitch}")

    body = "\n".join(lines).strip()
    if not body:
        return None
    return EmailContent(subject=subject, text_body=body, html_body=_make_html(body))


def _title_content_text(
    *,
    result: Dict[str, Any],
    query: Optional[str],
) -> Optional[EmailContent]:
    # Supports FE payloads in compact form:
    # { "title": "...", "content": "..." }
    content = str(result.get("content") or "").strip()
    if not content:
        return None

    title = str(result.get("title") or "Grant Justification").strip()
    subject = f"[GrantFetcher] {title}"

    lines: List[str] = []
    lines.append(f"Title: {title}")
    if query:
        lines.append(f"Query: {query}")
    lines.append("")
    lines.append(content)

    body = "\n".join(lines).strip()
    if not body:
        return None
    return EmailContent(subject=subject, text_body=body, html_body=_make_html(body))


def _group_matches_text(
    *,
    result: Dict[str, Any],
    query: Optional[str],
) -> Optional[EmailContent]:
    matches = result.get("matches")
    if not isinstance(matches, list) or not matches:
        return None

    rows = [m for m in matches if isinstance(m, dict) and isinstance(m.get("justification"), dict)]
    if not rows:
        return None

    subject = "[GrantFetcher] Group Matching Justifications"
    lines: List[str] = []
    if query:
        lines.append(f"Query: {query}")
        lines.append("")

    for i, row in enumerate(rows[:5], start=1):
        title = str(row.get("grant_title") or row.get("grant_id") or "Untitled grant")
        lines.append(f"{i}. {title}")
        lines.append(f"   grant_id: {row.get('grant_id')}")
        lines.append(f"   team_score: {row.get('team_score')}")

        members = row.get("team_members") or []
        if isinstance(members, list) and members:
            lines.append("   team:")
            for m in members:
                if not isinstance(m, dict):
                    continue
                name = str(m.get("faculty_name") or f"Faculty {m.get('faculty_id')}")
                email = str(m.get("faculty_email") or "").strip()
                if email:
                    lines.append(f"   - {name} ({email})")
                else:
                    lines.append(f"   - {name}")

        just = row.get("justification") or {}
        one_paragraph = str(just.get("one_paragraph") or "").strip()
        if one_paragraph:
            lines.append(f"   summary: {one_paragraph}")
        recommendation = str(just.get("recommendation") or "").strip()
        if recommendation:
            lines.append(f"   recommendation: {recommendation}")
        lines.append("")

    body = "\n".join(lines).strip()
    if not body:
        return None
    return EmailContent(subject=subject, text_body=body, html_body=_make_html(body))


def build_justification_email(
    *,
    result: Dict[str, Any],
    query: Optional[str] = None,
) -> Optional[EmailContent]:
    if not isinstance(result, dict) or not result:
        return None

    for builder in (
        _group_justification_text,
        _one_to_one_recommendation_text,
        _title_content_text,
        _single_recommendation_row_text,
        _group_matches_text,
    ):
        content = builder(result=result, query=query)
        if content is not None:
            return content

    return None
