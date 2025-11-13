# services/generate_faculty_keywords.py
from __future__ import annotations
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
from sqlalchemy.orm import Session


from google import genai
from db.db_conn import SessionLocal
from db.dao.keywords_faculty import FacultyKeywordDAO

import db.models.faculty as mf
import db.models.keywords_faculty as mfk
from util.build_prompt import build_prompt
from util.format_keywords import _normalize_to_new_schema, _count_total_strings

# ─────────────────────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────────────────────
env_path = Path(__file__).resolve().parents[2] / "api.env"
load_dotenv(dotenv_path=env_path, override=True)
gemini_key = os.getenv("GEMINI_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

FACULTY_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "faculty_keyword_prompt_v3.txt"


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


def _build_corpus_for_faculty(db: Session, fac: mf.Faculty, *, max_chars: int = 40_000) -> str:
    """
    Compose a concise text for keyword extraction.
    Parts:
      - header: name, position, organization
      - biography (text-only)
      - expertise terms (comma-joined)
      - research group names (comma-joined)
    """
    parts: List[str] = []
    hdr_bits = [x for x in [fac.name, fac.position, fac.organization] if x]
    if hdr_bits:
        parts.append(" | ".join(hdr_bits))

    if fac.biography:
        parts.append(_strip_html(fac.biography))

    # expertise
    ex_terms = db.execute(
        select(mf.FacultyExpertise.term).where(mf.FacultyExpertise.faculty_id == fac.id)
    ).scalars().all()
    if ex_terms:
        parts.append("Expertise: " + ", ".join(sorted(set(t.strip() for t in ex_terms if t))))

    # research groups
    rg_names = db.execute(
        select(mf.FacultyResearchGroup.name).where(mf.FacultyResearchGroup.faculty_id == fac.id)
    ).scalars().all()
    if rg_names:
        parts.append("Research Groups: " + ", ".join(sorted(set(n.strip() for n in rg_names if n))))

    # ---------------------------------------------------------
    # Publications (
    # ---------------------------------------------------------
    pubs = db.execute(
        select(
            mf.FacultyPublication.year,
            mf.FacultyPublication.title,
            mf.FacultyPublication.abstract
        ).where(mf.FacultyPublication.faculty_id == fac.id)
        .order_by(mf.FacultyPublication.year.desc())
    ).all()

    if pubs:
        pub_lines = ["Research:\n"]
        for year, title, abstract in pubs:
            if not title:
                continue

            line = []

            if year:
                line.append(f"{year} — {title}")
            else:
                line.append(title)

            if abstract:
                line.append(abstract.strip())

            pub_lines.append("\n".join(line) + "\n")

        parts.append("\n".join(pub_lines).strip())


    corpus = "\n\n".join([p for p in parts if p]).strip()
    return corpus[:max_chars]

def extract_keywords_via_llm(corpus: str, max_keywords: int = 30) -> Dict[str, Any]:
    #provider = (os.getenv("KEYWORD_LLM_PROVIDER") or "").lower()  # "openai" | "gemini" | ""
    provider = "openai"

    if provider == "openai":
        return _extract_with_openai(corpus, max_keywords)

    if provider == "gemini":
        return _extract_with_gemini(corpus, max_keywords)


# def _extract_with_openai(corpus: str, max_keywords: int) -> Dict[str, Any]:
#     client = OpenAI(api_key=openai_key)
#     prompt = build_prompt(FACULTY_PROMPT_PATH, corpus, max_keywords)

#     response = client.responses.create(
#         model="gpt-5",
#         input=prompt
#     )
#     text = response.output_text or "{}"
#     #print(text)
#     data = json.loads(text)
#     return data

# Added NEW: normalize + Qwen-embedding-based cross-domain de-dup
def _extract_with_openai(corpus: str, max_keywords: int) -> Dict[str, Any]:
    client = OpenAI(api_key=openai_key)
    prompt = build_prompt(FACULTY_PROMPT_PATH, corpus, max_keywords)
    #print(prompt)

    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )
    text = response.output_text or "{}"
    data = json.loads(text)

    # NEW: normalize + semantic de-dup (research vs application)
    data = enforce_caps_and_limits(data, max_items_per_list=10)
    return data


def _extract_with_gemini(corpus: str, max_keywords: int) -> Dict[str, Any]:

    client = genai.Client(api_key=gemini_key)
    prompt = build_prompt(FACULTY_PROMPT_PATH, corpus, max_keywords)
    response = client.models.generate_content(
        model="gemini-2.5-pro", contents=prompt
    )

    #print(response)
    text = response.text or "{}"
    s = text.strip()

    if s.startswith("```"):
        # remove leading ```json or ``` and trailing ```
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()

    # If still not a pure JSON object, try to grab the first {...} block
    if not s.startswith("{"):
        m = re.search(r"\{(?:[^{}]|(?R))*\}", s, re.S)  # recursive-ish fallback
        s = m.group(0) if m else "{}"

    data = json.loads(s)
    return data

# ─────────────────────────────────────────────────────────────
# Public runners (single + batch)
# ─────────────────────────────────────────────────────────────
def mine_keywords_for_one_faculty(db: Session, faculty_id: int, *, max_keywords: int = 30, source_tag: str = "gpt-5") -> int:
    fac = db.get(mf.Faculty, faculty_id)
    if not fac:
        return 0

    corpus = _build_corpus_for_faculty(db, fac)
    if not corpus:
        return 0

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
            "source": source_tag if gemini_key else "heuristic",
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
    # build base query
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
                c = mine_keywords_for_one_faculty(db, fid, max_keywords=max_keywords)
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