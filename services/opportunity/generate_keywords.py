from __future__ import annotations

from config import OPENAI_API_KEY, OPENAI_MODEL
from util.keyword_cleanup import enforce_caps_and_limits

from typing import List, Dict, Any, Optional
import json
import os
import math
import re
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.orm import Session
from db.db_conn import SessionLocal

from dao.keywords_opportunity import KeywordDAO
from dao.opportunity import OpportunityReadDAO
from util.build_prompt import build_prompt
from util.format_keywords import _normalize_to_new_schema, _count_total_strings

from google import genai
from openai import OpenAI

GRANT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "opportunity_keyword_prompt.txt"

def _strip_html(s: Optional[str]) -> str:
    if not s:
        return ""
    # very light HTML remover; you already store stripped, but just in case
    return re.sub(r"<[^>]+>", " ", s).replace("&nbsp;", " ").strip()

def _build_corpus_for_opportunity(summary: Dict[str, Any], files: List[Dict[str, str]], max_chars: int = 400_000) -> str:
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
        parts.append(f"\n[ATTACHMENT] {fname}\n{ftxt}")

    corpus = "\n\n".join(parts)

    print(corpus)
    return corpus[:max_chars]


def extract_keywords_via_llm(corpus: str, max_keywords: int = 30) -> Dict[str, Any]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = build_prompt(GRANT_PROMPT_PATH, corpus, max_keywords)
    print(corpus)

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt
    )
    text = response.output_text or "{}"
    data = json.loads(text)

    # NEW: normalize + Qwen-embedding-based cross-domain de-dup
    data = enforce_caps_and_limits(data, max_items_per_list=10)
    return data


def mine_keywords_for_one(db: Session, opportunity_id: str, *, max_keywords: int = 30, source_tag: str) -> int:
    existing = KeywordDAO.get_by_opportunity_id(db, opportunity_id)
    if existing:
        print(f"[skip] Keywords already exist for {opportunity_id}, skipping.")
        return 0

    blob = OpportunityReadDAO.get_summary_and_files(db, opportunity_id)
    summary = blob.get("Summary")
    files = blob.get("additional_files") or []
    if not summary:
        return 0

    corpus = _build_corpus_for_opportunity(summary, files)
    items = extract_keywords_via_llm(corpus, max_keywords=max_keywords)
    #print(items)
    structured_keywords = _normalize_to_new_schema(items)
    #print(structured_keywords)
    if (_count_total_strings(structured_keywords) == 0
            and not structured_keywords.get("area")
            and not structured_keywords.get("discipline")):
        return 0

    # Store as one row — entire list + raw_json
    row = {
        "opportunity_id": opportunity_id,
        "keywords": structured_keywords,  # store as JSON array
        "raw_json": items,     # keep raw LLM output
        "source": source_tag,
    }

    KeywordDAO.upsert_keywords_json(db, [row])
    return len(structured_keywords)

def mine_keywords_for_all(db: Session, *, batch_size: int = 100, max_keywords: int = 30) -> Dict[str, Any]:

    from db.models.opportunity import Opportunity
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
                n = mine_keywords_for_one(db, oid, max_keywords=max_keywords, source_tag=OPENAI_MODEL)
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