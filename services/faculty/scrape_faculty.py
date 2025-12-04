import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import db.models.keywords_faculty  # defines FacultyKeyword
import db.models.faculty           # defines Faculty
from services.faculty.scrape_individual_faculty import parse_profile
from services.faculty.save_faculty import save_profile_dict, enrich_faculty_publications

#TODO: PUT links on the separate config file
BASE = "https://engineering.oregonstate.edu"
LIST_PATH = "/people"
HEADERS = {"User-Agent": "Mozilla/5.0 (+faculty-link-scraper; OSU project)"}

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
    print(f"[+] Fetching: {url}")
    soup = fetch(url)
    links = extract_faculty_links(soup)
    print(f"    Found {len(links)} links")
    all_links.extend(links)

    # Pages 1..max_pages
    for p in range(1, max_pages + 1):
        url = f"{BASE}{LIST_PATH}?page={p}"
        print(f"[+] Fetching: {url}")
        soup = fetch(url)
        links = extract_faculty_links(soup)
        print(f"    Found {len(links)} links")
        if not links:
            print(f"    No links on page {p}. Stopping.")
            break
        all_links.extend(links)

    # Final de-dupe
    all_links = list(dict.fromkeys(all_links))
    print(f"\nTotal unique faculty profiles: {len(all_links)}")
    return all_links

if __name__ == "__main__":
    links = crawl(max_pages=15)

    for link in links:
        try:
            profile = parse_profile(link)
            faculty_id = save_profile_dict(profile)

            # skip publications if name is missing
            if not profile.get("name"):
                print(f"[SKIP PUBS] No name for {link}")
                continue

            num_pubs = enrich_faculty_publications(faculty_id)
            print(f"[OK] {profile.get('name')} (id={faculty_id}) -> {num_pubs} pubs")

        except Exception as e:
            # Don't let one bad profile kill the whole run
            print(f"[ERROR] link={link} error={e}")
