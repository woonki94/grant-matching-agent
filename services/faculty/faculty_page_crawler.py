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

def crawl(max_pages: int = 50, max_links: int = 0) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def add_batch(links: list[str]) -> None:
        nonlocal out
        for link in links:
            if link in seen:
                continue
            seen.add(link)
            if 0 < max_links <= len(out):
                continue  # keep crawling pages, but don't store more
            out.append(link)

    # Page 0
    soup = fetch(f"{BASE}{LIST_PATH}")
    add_batch(extract_faculty_links(soup))

    # Pages 1..max_pages
    for p in range(1, max_pages + 1):
        soup = fetch(f"{BASE}{LIST_PATH}?page={p}")
        links = extract_faculty_links(soup)
        if not links:
            break
        add_batch(links)

    return out

