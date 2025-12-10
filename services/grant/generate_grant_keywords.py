# services/generate_grant_keywords.py
from __future__ import annotations
from util.keyword_cleanup import enforce_caps_and_limits


from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import os
import math
import re
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.orm import Session
from db.db_conn import SessionLocal

from db.dao.keywords_grant import KeywordDAO
from db.dao.grant import OpportunityReadDAO
from util.build_prompt import build_prompt
from util.format_keywords import _normalize_to_new_schema, _count_total_strings

from google import genai
from openai import OpenAI


env_path = Path(__file__).resolve().parents[2] / "api.env"
loaded = load_dotenv(dotenv_path=env_path, override=True)
API_KEY = os.getenv("API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

GRANT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "grant_keyword_prompt_v3.txt"

def _strip_html(s: Optional[str]) -> str:
    if not s:
        return ""
    # very light HTML remover; you already store stripped, but just in case
    return re.sub(r"<[^>]+>", " ", s).replace("&nbsp;", " ").strip()

def _build_corpus(summary: Dict[str, Any], files: List[Dict[str, str]], max_chars: int = 400_000) -> str:
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
        parts.append(f"\n[ATTACHMENT] {fname}\n{ftxt[:100_000]}")

    corpus = "\n\n".join(parts)
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
#     prompt = build_prompt(GRANT_PROMPT_PATH, corpus, max_keywords)

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
    prompt = build_prompt(GRANT_PROMPT_PATH, corpus, max_keywords)

    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )
    text = response.output_text or "{}"
    data = json.loads(text)

    # NEW: normalize + Qwen-embedding-based cross-domain de-dup
    data = enforce_caps_and_limits(data, max_items_per_list=10)
    return data


def _extract_with_gemini(corpus: str, max_keywords: int) -> Dict[str, Any]:

    client = genai.Client(api_key=gemini_key)
    prompt = build_prompt(GRANT_PROMPT_PATH, corpus, max_keywords)
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


def mine_keywords_for_one(db: Session, opportunity_id: str, *, max_keywords: int = 30, source_tag: str = "gpt-5") -> int:
    blob = OpportunityReadDAO.get_summary_and_files(db, opportunity_id)
    summary = blob.get("Summary")
    files = blob.get("additional_files") or []
    if not summary:
        return 0

    corpus = _build_corpus(summary, files)
    items = extract_keywords_via_llm(corpus, max_keywords=max_keywords)
    print(items)
    structured_keywords = _normalize_to_new_schema(items)
    print(structured_keywords)
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