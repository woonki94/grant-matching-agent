# services/generate_faculty_keywords.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import os
import math
import re
import time

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.orm import Session


from google import genai
from db.db_conn import SessionLocal
from db.dao.keywords_faculty import FacultyKeywordDAO

import db.models.faculty as mf
import db.models.keywords_faculty as mfk

# ─────────────────────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────────────────────
env_path = Path(__file__).resolve().parents[2] / "api.env"
load_dotenv(dotenv_path=env_path, override=True)
gemini_key = os.getenv("GEMINI_API_KEY")


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

    corpus = "\n\n".join([p for p in parts if p]).strip()
    return corpus[:max_chars]


# ─────────────────────────────────────────────────────────────
# LLM adapter (Gemini), JSON-only output
# ─────────────────────────────────────────────────────────────
def _extract_with_gemini(corpus: str, max_keywords: int) -> Dict[str, Any]:
    """
    Return {"keywords": [...]} and keep raw response in "raw".
    If key is missing, returns a fallback list from comma splits.
    """
    if not gemini_key:
        naive = [t.strip() for t in re.split(r"[,\n;]", corpus) if t.strip()]
        naive = list(dict.fromkeys(naive))[:max_keywords]
        return {"keywords": naive, "raw": {"fallback": True, "items": naive}}

    client = genai.Client(api_key=gemini_key)
    prompt = f"""
        Extract up to {max_keywords} concise, meaningful research keywords or short phrases
        that summarize this faculty member's research areas. 
        
        Visit research_website_url and links then also see the information there.
        
        Avoid generic/administrative terms (e.g., "project", "research", "work").
        Prefer domain-specific terms (e.g., "reinforcement learning", "humanoid robotics",
        "power systems stability", "bioprinting", "estimation theory").
        
        Try to keep the keywords in one word, unless it does not deliver clear meaning when separating the words
        (e.g. reinforcement learning -> reinforcement, learning does not give original meaning)  
        
        Return ONLY valid JSON in exactly this format:
        {{
          "keywords": ["keyword1", "keyword2", "keyword3", ...]
        }}

        Do not include any explanations outside the JSON.

        Faculty profile content:
        {corpus}
    """

    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    text = (resp.text or "{}").strip()

    # strip fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    # if not a pure JSON, try first {...}
    if not text.startswith("{"):
        m = re.search(r"\{(?:[^{}]|(?R))*\}", text, re.S)
        text = m.group(0) if m else "{}"

    data = json.loads(text)
    return {"keywords": data.get("keywords", []) or [], "raw": data}


# ─────────────────────────────────────────────────────────────
# Public runners (single + batch)
# ─────────────────────────────────────────────────────────────
def mine_keywords_for_one_faculty(db: Session, faculty_id: int, *, max_keywords: int = 30, source_tag: str = "gemini") -> int:
    fac = db.get(mf.Faculty, faculty_id)
    if not fac:
        return 0

    corpus = _build_corpus_for_faculty(db, fac)
    if not corpus:
        return 0

    out = _extract_with_gemini(corpus, max_keywords=max_keywords)
    keywords = out.get("keywords") or []
    raw_json = out.get("raw")

    if not keywords:
        return 0

    FacultyKeywordDAO.upsert_keywords_json(
        db,
        [{
            "faculty_id": fac.id,
            "keywords": keywords,
            "raw_json": raw_json,
            "source": source_tag if gemini_key else "heuristic",
        }],
    )
    return len(keywords)


def mine_keywords_for_all_faculty(
    db: Session,
    *,
    batch_size: int = 100,
    max_keywords: int = 50,
    only_missing: bool = True,
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