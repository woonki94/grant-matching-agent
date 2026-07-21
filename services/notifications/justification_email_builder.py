from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EmailContent:
    subject: str
    text_body: str
    html_body: str
    attachment_text_body: Optional[str] = None
    attachment_html_body: Optional[str] = None


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


def _make_markdown_html(text: str) -> str:
    """
    Lightweight safe markdown rendering for email/attachment bodies.
    Supports:
      - # / ## / ### headings
      - **bold**
      - bullet lists with '-', '*', or '•'
      - paragraphs
    """
    src = _normalize_newlines(text)
    lines = src.split("\n")

    body_parts: List[str] = []
    para_buf: List[str] = []
    in_list = False

    heading_styles = {
        1: "margin:18px 0 8px;font-size:20px;line-height:1.25;font-weight:700;color:#1f2a44;",
        2: "margin:16px 0 8px;font-size:17px;line-height:1.3;font-weight:700;color:#24314f;",
        3: "margin:14px 0 6px;font-size:15px;line-height:1.3;font-weight:700;color:#2b3550;",
    }

    def _inline_markdown(value: str) -> str:
        safe = html.escape(str(value or ""))
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe, flags=re.DOTALL)

    def _flush_paragraph() -> None:
        nonlocal para_buf
        if not para_buf:
            return
        joined = "<br>".join(para_buf)
        body_parts.append(f"<p>{joined}</p>")
        para_buf = []

    for raw in lines:
        line = str(raw or "").rstrip()
        stripped = line.strip()
        if not stripped:
            _flush_paragraph()
            if in_list:
                body_parts.append("</ul>")
                in_list = False
            continue

        heading_level = None
        heading_text = ""
        if stripped.startswith("### "):
            heading_level, heading_text = 3, stripped[4:].strip()
        elif stripped.startswith("## "):
            heading_level, heading_text = 2, stripped[3:].strip()
        elif stripped.startswith("# "):
            heading_level, heading_text = 1, stripped[2:].strip()
        if heading_level and heading_text:
            _flush_paragraph()
            if in_list:
                body_parts.append("</ul>")
                in_list = False
            style = heading_styles.get(int(heading_level), heading_styles[2])
            body_parts.append(
                f"<h{heading_level} style='{style}'>{_inline_markdown(heading_text)}</h{heading_level}>"
            )
            continue

        bullet_match = re.match(r"^\s*[-*•]\s+(.+)$", stripped)
        if bullet_match:
            _flush_paragraph()
            if not in_list:
                body_parts.append("<ul>")
                in_list = True
            body_parts.append(f"<li>{_inline_markdown(bullet_match.group(1).strip())}</li>")
            continue

        if in_list:
            body_parts.append("</ul>")
            in_list = False
        para_buf.append(_inline_markdown(stripped))

    _flush_paragraph()
    if in_list:
        body_parts.append("</ul>")

    body = "".join(body_parts).strip()
    if not body:
        return "<html><body></body></html>"
    return (
        "<html><body style='font-family:Arial,sans-serif;font-size:16px;line-height:1.55;"
        "white-space:normal'>"
        f"{body}</body></html>"
    )


def _normalize_newlines(text: Any) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def _normalize_label_key(label: str) -> str:
    return re.sub(r"[^a-z]+", "", str(label or "").lower())


def _section_to_items(text: str) -> List[str]:
    src = _normalize_newlines(text).strip()
    if not src:
        return []

    lines = [x.strip() for x in src.split("\n") if x.strip()]
    bullet_lines = []
    for line in lines:
        m = re.match(r"^\s*[-*•]\s+(.+)$", line)
        if m:
            bullet_lines.append(m.group(1).strip())
        else:
            bullet_lines.append(line)
    if len(bullet_lines) > 1:
        return [x for x in bullet_lines if x]

    flat = " ".join(src.split()).strip()
    flat = re.sub(r"^[\-*•]\s+", "", flat)
    if not flat:
        return []
    parts = [p.strip(" ;") for p in re.split(r"\s+[•]\s+|\s+-\s+", flat) if p.strip(" ;")]
    if len(parts) > 1:
        return parts
    return [flat]


def _extract_compact_email_fields(content: str) -> Dict[str, str]:
    src = _normalize_newlines(content)

    agency = ""
    score = ""
    m_agency = re.search(r"(?im)^\s*Agency\s*:\s*(.+?)\s*$", src)
    if m_agency:
        agency = str(m_agency.group(1) or "").strip()
    m_score = re.search(r"(?im)^\s*Score\s*:\s*(.+?)\s*$", src)
    if m_score:
        score = str(m_score.group(1) or "").strip()

    source_wo_meta = re.sub(r"(?im)^\s*(Agency|Score)\s*:\s*.+?$", "", src)
    source_wo_meta = re.sub(r"\n{3,}", "\n\n", source_wo_meta).strip()

    label_alias = {
        "grantexplanation": "grant_explanation",
        "whatthegrantemphasizes": "what_grant_emphasizes",
        "whatcapabilitiesitexpects": "what_capabilities_expects",
        "whyitfits": "why_it_fits",
        "gapstoaddress": "gaps_to_address",
    }
    label_re = re.compile(
        r"(?is)(?:\*\*\s*)?"
        r"(Grant\s*Explanation|What\s+the\s+Grant\s+Emphasizes|What\s+Capabilities\s+It\s+Expects|Why\s+it\s+fits|Gaps\s+to\s+address)"
        r"(?:\s*\*\*)?\s*:\s*(?:\*\*)?"
    )
    matches = list(label_re.finditer(source_wo_meta))

    sections: Dict[str, str] = {
        "grant_explanation": "",
        "what_grant_emphasizes": "",
        "what_capabilities_expects": "",
        "why_it_fits": "",
        "gaps_to_address": "",
        "faculty_fit_summary": "",
    }

    if matches:
        for idx, m in enumerate(matches):
            raw_label = str(m.group(1) or "").strip()
            key = label_alias.get(_normalize_label_key(raw_label), "")
            if not key:
                continue
            start = int(m.end())
            end = int(matches[idx + 1].start()) if idx + 1 < len(matches) else len(source_wo_meta)
            chunk = source_wo_meta[start:end].strip()
            chunk = re.sub(r"^\*+\s*", "", chunk)
            chunk = re.sub(r"\s*\*+$", "", chunk)
            chunk = chunk.strip().strip("- ").strip()
            sections[key] = chunk
    else:
        sections["grant_explanation"] = source_wo_meta.strip()

    # Capture the narrative paragraph right before "Why it fits:" when present.
    why_match = re.search(r"(?is)\bWhy\s+it\s+fits\s*:", source_wo_meta)
    if why_match:
        before_why = source_wo_meta[: int(why_match.start())].strip()
        paras = [p.strip() for p in re.split(r"\n\s*\n", before_why) if p.strip()]
        candidate = ""
        for p in reversed(paras):
            if re.search(r"(?is)\bexpertise\b", p):
                candidate = p
                break
        if not candidate and paras:
            candidate = paras[-1]
        if candidate and not re.match(r"(?is)^\*{0,2}\s*What\s+Capabilities\s+It\s+Expects", candidate):
            sections["faculty_fit_summary"] = candidate

    out = {
        "agency": agency,
        "score": score,
        **sections,
    }
    return out


def _build_full_attachment_markdown(
    *,
    title: str,
    query: Optional[str],
    content: str,
) -> str:
    fields = _extract_compact_email_fields(content)
    agency = str(fields.get("agency") or "").strip()
    score = str(fields.get("score") or "").strip()
    raw_full = _normalize_newlines(content).strip()
    full_narrative = raw_full
    if full_narrative:
        # Keep every original word while making the long blob easier to scan.
        full_narrative = re.sub(
            r"\s+\*\*(What\s+the\s+Grant\s+Emphasizes|What\s+Capabilities\s+It\s+Expects)\s*:\s*",
            r"\n\n**\1:** ",
            full_narrative,
            flags=re.IGNORECASE,
        )
        full_narrative = re.sub(
            r"\s+(Why\s+it\s+fits|Gaps\s+to\s+address)\s*:\s*",
            r"\n\n\1:\n",
            full_narrative,
            flags=re.IGNORECASE,
        )
        full_narrative = full_narrative.replace(" - ", "\n- ")
        full_narrative = re.sub(r"\n{3,}", "\n\n", full_narrative).strip()

    lines: List[str] = []
    lines.append("# Grant Match Details")
    lines.append("")
    lines.append("## Grant")
    lines.append(f"- **Title:** {title}")
    if agency:
        lines.append(f"- **Agency:** {agency}")
    if score:
        lines.append(f"- **Matching Score:** {score}")
    if query:
        lines.append(f"- **Prompt:** {str(query).strip()}")
    if full_narrative:
        lines.append("")
        lines.append("## Full Narrative")
        lines.append(full_narrative)

    lines.append("")
    return "\n".join(lines).strip()


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
    return EmailContent(subject=subject, text_body=body, html_body=_make_markdown_html(body))


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
    global_grant_explanation = str(
        result.get("grant_explanation")
        or (recommendation.get("grant_explanation") if isinstance(recommendation, dict) else "")
        or ""
    ).strip()
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
        grant_explanation = str(rec.get("grant_explanation") or "").strip()
        if not grant_explanation and i == 1 and global_grant_explanation:
            grant_explanation = global_grant_explanation
        if grant_explanation:
            lines.append("   grant_explanation:")
            lines.append(grant_explanation)
        lines.append("")

    body = "\n".join(lines).strip()
    if not body:
        return None
    return EmailContent(subject=subject, text_body=body, html_body=_make_markdown_html(body))


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
    grant_explanation = str(result.get("grant_explanation") or "").strip()
    if grant_explanation:
        lines.append("   grant_explanation:")
        lines.append(grant_explanation)

    body = "\n".join(lines).strip()
    if not body:
        return None
    return EmailContent(subject=subject, text_body=body, html_body=_make_markdown_html(body))


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
    fields = _extract_compact_email_fields(content)
    agency = str(fields.get("agency") or "").strip()
    score = str(fields.get("score") or "").strip()
    grant_explanation = str(fields.get("grant_explanation") or "").strip()
    emphasizes = str(fields.get("what_grant_emphasizes") or "").strip()
    capabilities = str(fields.get("what_capabilities_expects") or "").strip()
    why_fit = str(fields.get("why_it_fits") or "").strip()
    gaps = str(fields.get("gaps_to_address") or "").strip()
    fit_summary = str(fields.get("faculty_fit_summary") or "").strip()

    lines: List[str] = []
    lines.append("## What This Email Is")
    lines.append("This is a quick summary of one grant match from GrantFetcher.")
    lines.append("")
    lines.append("## Quick Summary")
    lines.append(f"- **Grant:** {title}")
    if agency:
        lines.append(f"- **Agency:** {agency}")
    if score:
        lines.append(f"- **Matching Score:** {score}")

    if grant_explanation:
        lines.append("")
        lines.append("## Grant Explanation")
        lines.append(grant_explanation)

    if fit_summary:
        lines.append("")
        lines.append("## Faculty Fit Snapshot")
        lines.append(fit_summary)

    # Keep the email short but useful when explanation is missing.
    if not grant_explanation and not fit_summary:
        concise_bits: List[str] = []
        for field_text in (emphasizes, capabilities, why_fit, gaps):
            items = _section_to_items(field_text)
            if items:
                concise_bits.append(items[0])
            if len(concise_bits) >= 2:
                break
        if concise_bits:
            lines.append("")
            lines.append("## Key Points")
            for item in concise_bits:
                lines.append(f"- {item}")

    lines.append("")
    lines.append("## Full Details")
    lines.append("For full analysis and detailed rationale, refer to the attachments.")

    body = "\n".join(lines).strip()
    if not body:
        return None

    full_attachment_text = _build_full_attachment_markdown(
        title=title,
        query=query,
        content=content,
    )
    return EmailContent(
        subject=subject,
        text_body=body,
        html_body=_make_markdown_html(body),
        attachment_text_body=full_attachment_text,
        attachment_html_body=_make_markdown_html(full_attachment_text),
    )


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
    return EmailContent(subject=subject, text_body=body, html_body=_make_markdown_html(body))


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
