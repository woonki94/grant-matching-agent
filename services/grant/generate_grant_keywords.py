# services/generate_grant_keywords.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import os
import math
import re

from dotenv import load_dotenv
from sqlalchemy.orm import Session
from google import genai
from db.db_conn import SessionLocal

from db.dao.keywords_grant import KeywordDAO
from db.dao.grant import OpportunityReadDAO

env_path = Path(__file__).resolve().parents[2] / "api.env"
loaded = load_dotenv(dotenv_path=env_path, override=True)
API_KEY = os.getenv("API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

def _strip_html(s: Optional[str]) -> str:
    if not s:
        return ""
    # very light HTML remover; you already store stripped, but just in case
    return re.sub(r"<[^>]+>", " ", s).replace("&nbsp;", " ").strip()

def _build_corpus(summary: Dict[str, Any], files: List[Dict[str, str]], max_chars: int = 40_000) -> str:
    parts: List[str] = []

    # pick high-signal summary fields
    keys = [
        "opportunity_title",
        "summary_description", "additional_info_url",
    ]
    for k in keys:
        v = summary.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            v = ", ".join(map(str, v))
        parts.append(f"{k}: {_strip_html(str(v))}")

    # append attachments (cap per-file to keep size sane)
    for f in files:
        fname = f.get("file_name") or ""
        ftxt = (f.get("file_content") or "").strip()
        if not ftxt:
            continue
        # take first ~6k chars per file to avoid token blowups
        parts.append(f"\n[ATTACHMENT] {fname}\n{ftxt[:6000]}")

    corpus = "\n\n".join(parts)
    return corpus[:max_chars]

# ---------- LLM adapter (plug your provider here) ----------
class KeywordCandidate(Dict[str, Any]):
    # shape: {"keyword": str, "weight": int|None, "category": str|None}
    pass

def extract_keywords_via_llm(corpus: str, max_keywords: int = 30) -> List[KeywordCandidate]:
    #provider = (os.getenv("KEYWORD_LLM_PROVIDER") or "").lower()  # "openai" | "gemini" | ""
    provider = "gemini"

    #TODO: Try openAI aswell
    #if provider == "openai":
    #    return _extract_with_openai(corpus, max_keywords)
    if provider == "gemini":
        return _extract_with_gemini(corpus, max_keywords)

def _extract_with_openai(corpus: str, max_keywords: int) -> List[KeywordCandidate]:
    return []

def _extract_with_gemini(corpus: str, max_keywords: int) -> List[KeywordCandidate]:

    client = genai.Client(api_key=gemini_key)
    prompt = f"""
        Extract up to {max_keywords} concise, meaningful keywords or short phrases
        that summarize the core research topics of the following grant opportunity.
        
        The section labeled "summary description" provides the grant’s main project summary.
        The field "additional_info_url" may point to the full grant announcement. If available,
        imagine you can visit it to understand what the grant is doing.
        Under the section [ATTACHMENT], you may find supplementary materials or files
        attached to the opportunity (such as project descriptions, calls, or requirements).
        
        Use all this information to identify the central research areas or topics that best
        represent what the grant is about.
        
        Each keyword should be a single word or very short phrase (e.g., "genomics", "renewable energy",
        "machine learning", "public health", "data privacy"). Avoid generic or administrative terms
        (e.g., "project", "proposal", "research", "development", "support").
        
        Try to keep the keywords in one word, unless it does not deliver clear meaning when separating the words
        (e.g. reinforcement learning -> reinforcement, learning does not give original meaning)  
        
        Return ONLY valid JSON in exactly this format:
        {{
          "keywords": ["keyword1", "keyword2", "keyword3", ...]
        }}
        
        Do not include any explanations, comments, or text outside the JSON object.
        
        Grant Summary and Materials:
        {corpus}
        """
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
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


def mine_keywords_for_one(db: Session, opportunity_id: str, *, max_keywords: int = 30, source_tag: str = "gemini") -> int:
    blob = OpportunityReadDAO.get_summary_and_files(db, opportunity_id)
    summary = blob.get("Summary")
    files = blob.get("additional_files") or []
    if not summary:
        return 0

    corpus = _build_corpus(summary, files)
    items = extract_keywords_via_llm(corpus, max_keywords=max_keywords)

    keywords = items.get("keywords", [])
    if not keywords:
        return 0

    # Store as one row — entire list + raw_json
    row = {
        "opportunity_id": opportunity_id,
        "keywords": keywords,  # store as JSON array
        "raw_json": items,     # keep raw LLM output
        "source": source_tag,
    }

    KeywordDAO.upsert_keywords_json(db, [row])
    return len(keywords)

def mine_keywords_for_all(db: Session, *, batch_size: int = 100, max_keywords: int = 30) -> Dict[str, Any]:

    from db.models.grant import Opportunity
    total = db.query(Opportunity).count()
    done = 0
    written = 0

    pages = math.ceil(total / batch_size) if total else 0
    for page in range(pages):
        q = (
            db.query(Opportunity.opportunity_id)
              .order_by(Opportunity.post_date.desc().nullslast())
              .limit(batch_size)
              .offset(page * batch_size)
        )
        ids = [row[0] for row in q.all()]
        for oid in ids:
            try:
                n = mine_keywords_for_one(db, oid, max_keywords=max_keywords)
                written += n
            except Exception as e:
                # log and continue
                print(f"[keyword_mining] {oid}: ERROR {e}")
            done += 1
        db.commit()  # commit per batch
        print(f"[keyword_mining] progress {done}/{total} opportunities")

    return {"opportunities_processed": done, "keywords_written": written, "total_opportunities": total}


if __name__ == "__main__":
    with SessionLocal() as sess:
        report = mine_keywords_for_all(sess, batch_size=50, max_keywords=50)
        print(report)