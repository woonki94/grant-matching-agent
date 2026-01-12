from __future__ import annotations

import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from config import settings
from services.faculty.profile_parser import parse_profile

from typing import Dict, Any, List
from sqlalchemy.orm import Session, sessionmaker

from db.db_conn import engine



SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

BASE = settings.osu_eng_base_url
LIST_PATH = settings.osu_eng_list_path
HEADERS = {"User-Agent": settings.scraper_user_agent}

def fetch(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")

def extract_faculty_links(soup: BeautifulSoup) -> list[str]:
    out = []
    for a in soup.select("div.coe-brand-people-card a[href^='/people/']"):
        href = a.get("href")
        if href and href.startswith("/people/"):
            out.append(urljoin(BASE, href))
    # De-dupe while preserving order
    seen = set()
    unique = []
    for u in out:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique

def crawl(max_pages: int = 50) -> list[str]:
    all_links = []

    # Page 0: base without page number
    url = f"{BASE}{LIST_PATH}"
    soup = fetch(url)
    links = extract_faculty_links(soup)
    all_links.extend(links)

    # Pages 1..max_pages
    for p in range(1, max_pages + 1):
        url = f"{BASE}{LIST_PATH}?page={p}"
        soup = fetch(url)
        links = extract_faculty_links(soup)
        if not links:
            break
        all_links.extend(links)

    # Final de-dupe
    all_links = list(dict.fromkeys(all_links))
    return all_links

