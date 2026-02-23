"""
Publication extraction pipeline — CV-based, no external author-search APIs.

Pipeline:
  1. extract_pdf_bytes()              — raw text from uploaded CV PDF
  2. extract_publications_from_cv_text() — LLM parses {title, url, year} list
  3. enrich_with_abstracts()          — for each entry, tries in order:
       a. arXiv title search  (fast, free, great for CS/ML)
       b. Semantic Scholar title search  (broader coverage, free)
       c. LLM extraction from the URL provided in the CV
  4. Returns List[FacultyPublicationDTO] ready to be saved via
     FacultyDAO.upsert_publications_by_title()
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import requests
from langchain_core.messages import HumanMessage

from dto.faculty_dto import FacultyPublicationDTO
from utils.content_extractor import extract_pdf_bytes, fetch_and_extract_one

logger = logging.getLogger(__name__)

# ─── External API endpoints ───────────────────────────────────────────────────
_ARXIV_API = "https://export.arxiv.org/api/query"
_S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

# ─── Similarity helper ────────────────────────────────────────────────────────

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


# ─── Step 1: LLM extracts publication list from CV text ──────────────────────

_CV_PARSE_PROMPT = """\
You are an academic CV parser. Given the text of a faculty member's CV, \
extract every publication listed (journal articles, conference papers, book chapters, \
preprints, technical reports, etc.).

For each publication return a JSON object with:
  "title" : full title string  (required)
  "url"   : direct URL to the paper if explicitly listed in the CV, else null
  "year"  : integer publication year if mentioned, else null

Return ONLY a JSON array — no markdown fences, no explanation.
If no publications are found return an empty array: []

CV TEXT:
{cv_text}"""


def extract_publications_from_cv_text(
    cv_text: str,
    llm,  # ChatBedrock (or any LangChain chat model)
    max_cv_chars: int = 80_000,
) -> List[Dict[str, Any]]:
    """
    Use the LLM to parse a list of {title, url, year} dicts from raw CV text.
    Returns an empty list on any failure.
    """
    prompt = _CV_PARSE_PROMPT.format(cv_text=cv_text[:max_cv_chars])
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = (response.content if hasattr(response, "content") else str(response)).strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            parts = raw.split("```", 2)
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [p for p in parsed if isinstance(p, dict) and p.get("title")]
        return []
    except Exception:
        logger.exception("LLM failed to parse publications from CV text")
        return []


# ─── Step 2a: arXiv title search ─────────────────────────────────────────────

def _abstract_from_arxiv(title: str, threshold: float = 0.85) -> Optional[str]:
    try:
        resp = requests.get(
            _ARXIV_API,
            params={
                "search_query": f'ti:"{title}"',
                "max_results": 3,
                "sortBy": "relevance",
            },
            timeout=15,
            headers={"User-Agent": "GrantMatcher/1.0 (educational research)"},
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        for entry in root.findall("atom:entry", _ARXIV_NS):
            found = (entry.findtext("atom:title", "", _ARXIV_NS) or "").replace("\n", " ").strip()
            abstract = (entry.findtext("atom:summary", "", _ARXIV_NS) or "").replace("\n", " ").strip()
            if _sim(title, found) >= threshold and abstract:
                logger.debug("arXiv match (sim=%.2f): %s", _sim(title, found), found)
                return abstract
    except Exception:
        logger.debug("arXiv search failed for: %s", title)
    return None


# ─── Step 2b: Semantic Scholar fallback ──────────────────────────────────────

def _abstract_from_semantic_scholar(title: str, threshold: float = 0.85) -> Optional[str]:
    try:
        resp = requests.get(
            _S2_API,
            params={"query": title, "fields": "title,abstract", "limit": 3},
            timeout=15,
            headers={"User-Agent": "GrantMatcher/1.0 (educational research)"},
        )
        resp.raise_for_status()
        for paper in (resp.json().get("data") or []):
            found = (paper.get("title") or "").strip()
            abstract = (paper.get("abstract") or "").strip()
            if _sim(title, found) >= threshold and abstract:
                logger.debug("S2 match (sim=%.2f): %s", _sim(title, found), found)
                return abstract
    except Exception:
        logger.debug("Semantic Scholar search failed for: %s", title)
    return None


# ─── Step 2c: LLM extraction from a provided URL ─────────────────────────────

_URL_ABSTRACT_PROMPT = """\
You are an academic paper parser.
Given the text content of a web page or PDF for a paper titled:
"{title}"

Extract the abstract of the paper and return it as plain text.
If no abstract can be found, return exactly: NOT_FOUND

PAGE CONTENT:
{content}"""


def _abstract_from_url(title: str, url: str, llm) -> Optional[str]:
    try:
        result = fetch_and_extract_one(url, timeout=30)
        content = (result.get("text") or "").strip()
        if not content or result.get("error"):
            return None
        prompt = _URL_ABSTRACT_PROMPT.format(title=title, content=content[:12_000])
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = (response.content if hasattr(response, "content") else str(response)).strip()
        if raw and raw.upper() != "NOT_FOUND":
            return raw
    except Exception:
        logger.debug("URL-based abstract extraction failed for url=%s", url)
    return None


# ─── Step 3: Enrich each extracted publication with an abstract ───────────────

def enrich_with_abstracts(
    raw_pubs: List[Dict[str, Any]],
    llm,
    inter_request_sleep: float = 0.5,
) -> List[FacultyPublicationDTO]:
    """
    For each raw pub dict {title, url?, year?} try to fetch an abstract via:
      1. arXiv title search
      2. Semantic Scholar title search
      3. LLM extraction from the URL listed in the CV (if any)

    Pubs with no title are silently skipped.
    Pubs where no abstract can be found are included with abstract=None.
    """
    dtos: List[FacultyPublicationDTO] = []

    for pub in raw_pubs:
        title = (pub.get("title") or "").strip()
        if not title:
            continue

        url: Optional[str] = pub.get("url") or None
        year: Optional[int] = pub.get("year") or None
        abstract: Optional[str] = None

        abstract = _abstract_from_arxiv(title)
        if not abstract:
            time.sleep(inter_request_sleep)
            abstract = _abstract_from_semantic_scholar(title)
        if not abstract and url:
            abstract = _abstract_from_url(title, url, llm)

        dtos.append(FacultyPublicationDTO(title=title, abstract=abstract, year=year))
        time.sleep(inter_request_sleep)

    logger.info(
        "Publication enrichment done: %d total, %d with abstract",
        len(dtos),
        sum(1 for d in dtos if d.abstract),
    )
    return dtos


# ─── Entry point: bytes → FacultyPublicationDTO list ─────────────────────────

def extract_publications_from_cv_bytes(
    cv_bytes: bytes,
    llm,
) -> List[FacultyPublicationDTO]:
    """
    Full pipeline entry point.

    1. Extract text from the uploaded CV PDF bytes.
    2. Use LLM to parse a publication list.
    3. Enrich each entry with an abstract (arXiv → S2 → URL).

    Returns a list of FacultyPublicationDTO ready for
    FacultyDAO.upsert_publications_by_title().
    """
    try:
        cv_text = extract_pdf_bytes(cv_bytes)
    except Exception:
        logger.exception("Failed to extract text from CV PDF bytes")
        return []

    if not cv_text.strip():
        logger.warning("CV PDF yielded no extractable text")
        return []

    raw_pubs = extract_publications_from_cv_text(cv_text, llm)
    if not raw_pubs:
        logger.info("No publications found in CV text")
        return []

    logger.info("LLM extracted %d publication entries from CV", len(raw_pubs))
    return enrich_with_abstracts(raw_pubs, llm)
