from __future__ import annotations

from typing import List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from dto.faculty_dto import FacultyPublicationDTO


def _normalize_scholar_url(url: str, *, pagesize: int = 100, cstart: int = 0) -> str:
    parsed = urlparse(url)
    q = parse_qs(parsed.query)
    q["pagesize"] = [str(pagesize)]
    q["cstart"] = [str(cstart)]
    query = urlencode(q, doseq=True)
    return urlunparse(parsed._replace(query=query))


def _extract_scholar_author_id(url: str) -> Optional[str]:
    parsed = urlparse(url)
    q = parse_qs(parsed.query)
    vals = q.get("user") or []
    return vals[0] if vals else None


def scrape_scholar_publications(
    url: str,
    *,
    max_pubs: int = 100,
    timeout: int = 20,
    user_agent: str = "Mozilla/5.0 (+faculty-scraper; OSU project use)",
) -> Tuple[Optional[str], List[FacultyPublicationDTO]]:
    """
    Best-effort scrape of Google Scholar profile page.
    Returns (scholar_author_id, list[FacultyPublicationDTO]).
    """
    if not url or "scholar.google" not in url:
        return None, []

    scholar_author_id = _extract_scholar_author_id(url)
    fetch_url = _normalize_scholar_url(url, pagesize=max_pubs, cstart=0)

    resp = requests.get(fetch_url, headers={"User-Agent": user_agent}, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    pubs: List[FacultyPublicationDTO] = []
    for row in soup.select("tr.gsc_a_tr"):
        title_el = row.select_one("a.gsc_a_at")
        if not title_el:
            continue
        title = title_el.get_text(" ", strip=True)
        if not title:
            continue
        year_el = row.select_one("span.gsc_a_y span")
        year_txt = year_el.get_text(strip=True) if year_el else ""
        year = int(year_txt) if year_txt.isdigit() else None

        pubs.append(
            FacultyPublicationDTO(
                openalex_work_id=None,
                scholar_author_id=scholar_author_id,
                title=title,
                abstract=None,
                year=year,
            )
        )

        if len(pubs) >= max_pubs:
            break

    return scholar_author_id, pubs
