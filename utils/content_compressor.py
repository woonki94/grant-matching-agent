"""
To compress attachment content size ( due to limited input token in gpt5)
"""
import re
from typing import Any, Dict, List

def _compress_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        if len(ln) < 4:
            continue
        if re.fullmatch(r"\d+", ln):
            continue
        low = ln.lower()
        if low.startswith(("references", "copyright", "table of contents")):
            continue
        cleaned.append(ln)
    return "\n".join(cleaned)[:max_chars]


def cap_extracted_blocks(
    blocks: List[Dict[str, Any]],
    *,
    max_total_chars: int,
    max_per_doc_chars: int,
    content_key: str = "content",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    total = 0

    for b in blocks or []:
        raw = b.get(content_key) if isinstance(b, dict) else None
        text = _compress_text(raw or "", max_per_doc_chars)
        if not text:
            continue

        remaining = max_total_chars - total
        if remaining <= 0:
            break

        text = text[:remaining]
        total += len(text)

        bb = dict(b)
        bb[content_key] = text
        out.append(bb)

    return out


# ----------------------------
# Context capping (token safety)
# ----------------------------

def cap_fac(ctx: dict) -> dict:
    if "additional_infos" in ctx:
        ctx["additional_infos"] = cap_extracted_blocks(
            ctx["additional_infos"], max_total_chars=8_000, max_per_doc_chars=1_200
        )

    pubs = ctx.get("publications") or []
    for p in pubs:
        if isinstance(p, dict) and p.get("abstract"):
            p["abstract"] = p["abstract"][:600]
    ctx["publications"] = pubs[:8]
    return ctx

def cap_opp(ctx: dict) -> dict:
    if "attachments_extracted" in ctx:
        ctx["attachments_extracted"] = cap_extracted_blocks(
            ctx["attachments_extracted"], max_total_chars=8_000, max_per_doc_chars=1_500
        )
    if "additional_info_extracted" in ctx:
        ctx["additional_info_extracted"] = cap_extracted_blocks(
            ctx["additional_info_extracted"], max_total_chars=6_000, max_per_doc_chars=1_200
        )
    if ctx.get("summary_description"):
        ctx["summary_description"] = str(ctx["summary_description"])[:3_500]
    return ctx