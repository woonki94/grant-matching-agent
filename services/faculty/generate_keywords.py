from __future__ import annotations

from config import OPENAI_MODEL, OPENAI_API_KEY
from services.justification.generate_justification_rag import get_faculty_context
from util.keyword_cleanup import enforce_caps_and_limits 

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import os
import math
import re
import time

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from google import genai
from db.db_conn import SessionLocal
from dao.faculty import FacultyDAO
from dao.keywords_faculty import FacultyKeywordDAO

import db.models.faculty as mf
import db.models.keywords_faculty as mfk
from util.build_prompt import build_prompt
from util.format_keywords import _normalize_to_new_schema, _count_total_strings
from util.content_from_url import fetch_content
FACULTY_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "faculty_keyword_prompt.txt"

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _strip_html(s: Optional[str]) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&nbsp;", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _build_corpus_for_faculty(fac: mf.Faculty, *, max_chars: int = 40_000) -> str:
    parts: List[str] = []

    hdr_bits = [x for x in [fac.name, fac.position, fac.organization] if x]
    if hdr_bits:
        parts.append(" | ".join(hdr_bits))

    if fac.biography:
        parts.append(_strip_html(fac.biography))

    # expertise
    ex_terms = sorted({(e.term or "").strip() for e in (fac.expertise or []) if e.term and e.term.strip()})
    if ex_terms:
        parts.append("Expertise: " + ", ".join(ex_terms))

    # research groups
    rg_names = sorted({(g.name or "").strip() for g in (fac.research_groups or []) if g.name and g.name.strip()})
    if rg_names:
        parts.append("Research Groups: " + ", ".join(rg_names))

    #additional links
    links = []
    for lk in (fac.links or []):
        name = (lk.name or "").strip()
        url = (lk.url or "").strip()
        if not url:
            continue
        links.append(f"{name} ({fetch_content(url)})" if name else url)
    links = sorted(set(links))
    if links:
        parts.append("Additional Links: " + ", ".join(links))

    # publications
    pubs = sorted(
        fac.publications or [],
        key=lambda p: (p.year or 0),
        reverse=True
    )
    if pubs:
        pub_lines = ["Research:\n"]
        for p in pubs:
            if not p.title:
                continue
            line = [f"{p.year} — {p.title}" if p.year else p.title]
            if p.abstract:
                line.append(p.abstract.strip())
            pub_lines.append("\n".join(line) + "\n")
        parts.append("\n".join(pub_lines).strip())

    corpus = "\n\n".join([p for p in parts if p]).strip()
    return corpus[:max_chars]


def extract_keywords_via_llm(corpus: str, max_keywords: int = 30) -> Dict[str, Any]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = build_prompt(FACULTY_PROMPT_PATH, corpus, max_keywords)
    # print(prompt)

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt
    )
    text = response.output_text or "{}"
    data = json.loads(text)

    # NEW: normalize + semantic de-dup (research vs application)
    data = enforce_caps_and_limits(data, max_items_per_list=10)
    return data


# ─────────────────────────────────────────────────────────────
# Public runners (single + batch)
# ─────────────────────────────────────────────────────────────
def mine_keywords_for_one_faculty(db: Session, faculty_id: int, *, max_keywords: int = 30, source_tag: str ) -> int:
    fac = FacultyDAO.get_faculty_info(db, faculty_id=faculty_id)
    if not fac:
        return 0

    corpus = _build_corpus_for_faculty(fac)

    if not corpus:
        return 0
    #print(corpus)
    items = extract_keywords_via_llm(corpus, max_keywords=max_keywords)
    #print(items)
    structured_keywords = _normalize_to_new_schema(items)
    #print(structured_keywords)

    if (_count_total_strings(structured_keywords) == 0
            and not structured_keywords.get("area")
            and not structured_keywords.get("discipline")):
        return 0

    FacultyKeywordDAO.upsert_keywords_json(
        db,
        [{
            "faculty_id": fac.id,
            "keywords": structured_keywords,
            "raw_json": items,
            "source": source_tag,
        }],
    )
    return len(structured_keywords)


def mine_keywords_for_all_faculty(
    db: Session,
    *,
    batch_size: int = 100,
    max_keywords: int = 50,
    only_missing: bool = False,
    sleep_s: float = 0.25,
) -> Dict[str, Any]:
    """
    Batch-process faculty. If only_missing=True, process those with no faculty_keywords row.
    Returns stats dict.
    """
    if only_missing:
        q = (
            select(mf.Faculty.id)
            .outerjoin(mfk.FacultyKeyword, mf.Faculty.id == mfk.FacultyKeyword.faculty_id)
            .where(mfk.FacultyKeyword.id.is_(None))
            .order_by(mf.Faculty.id.asc())
        )
    else:
        q = select(mf.Faculty.id).order_by(mf.Faculty.id.asc())

    ids = [rid for rid in db.execute(q).scalars().all()]
    total = len(ids)
    pages = math.ceil(total / batch_size) if total else 0

    done = 0
    written = 0

    for page in range(pages):
        chunk = ids[page * batch_size : (page + 1) * batch_size]
        for fid in chunk:
            try:
                c = mine_keywords_for_one_faculty(db, fid, max_keywords=max_keywords, source_tag=OPENAI_MODEL)
                written += c
            except Exception as e:
                print(f"[faculty_keyword_mining] faculty_id={fid}: ERROR {e}")
            done += 1
            if sleep_s:
                time.sleep(sleep_s)
        db.commit()
        print(f"[faculty_keyword_mining] progress {done}/{total} faculty")

    return {"faculty_processed": done, "keywords_written": written, "total_faculty": total}


if __name__ == "__main__":
    with SessionLocal() as sess:
        report = mine_keywords_for_all_faculty(sess, batch_size=50, max_keywords=50)
        print(report)