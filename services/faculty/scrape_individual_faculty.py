import re
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup, Tag

LABELS = {
    "organizations": ["Organizations"],
    "email": ["Email"],
    "office_phone": ["Office Phone", "Phone"],
    "fax": ["Fax"],
    "address": ["Address"],
    "research_website": ["Research Website", "Research Websites"],
    "degrees": ["Degrees"],
    "research_expertise": ["Research Expertise", "Expertise"],
    "research_groups": ["Research Groups", "Groups"],
    "biography": ["Biography", "Bio"],
    "awards": ["Awards/Accolades", "Awards", "Accolades"],
    "additional_links": ["Additional Links", "Links"],
}

def fetch_html(url: str) -> BeautifulSoup:
    headers = {
        "User-Agent": "Mozilla/5.0 (+faculty-scraper; OSU project use)"
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")

def text_clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def find_label_nodes(soup: BeautifulSoup) -> Dict[str, List[Tag]]:
    """
    Build an index from normalized label text -> list of elements that render that label.
    We match by exact visible text (ignoring whitespace).
    """
    index = {}
    # Scan all tags that might present labels (divs, headings, strong, etc.)
    for tag in soup.find_all(True):
        txt = text_clean(tag.get_text(separator=" ", strip=True))
        if not txt:
            continue
        for key, variants in LABELS.items():
            for v in variants:
                if txt == v:
                    index.setdefault(key, []).append(tag)
    return index

# Add near top
POSITION_SELECTORS = [
    ".field--name-field-c-engr-title.field__item",       # exact class you observed
    '[class*="field--name-field-c-engr-title"] .field__item',  # safety
    '[class*="field--name-field-c-engr-title"]',         # safety
]

def extract_position(soup):
    # 1) Try explicit field block(s)
    for sel in POSITION_SELECTORS:
        el = soup.select_one(sel)
        if el:
            txt = re.sub(r"\s+", " ", el.get_text(strip=True))
            if txt:
                return txt
    # 2) Fallback: short line under H1 (previous heuristic)
    h1 = soup.find("h1")
    if not h1:
        return None
    STOP = {
        "Organizations","Email","Office Phone","Phone","Address",
        "Research Website","Research Websites","Degrees",
        "Research Expertise","Expertise","Research Groups","Groups",
        "Biography","Bio","Awards/Accolades","Awards","Accolades",
        "Additional Links","Links","Related Stories"
    }
    def looks_like_title(s):
        if not s: return False
        if s in STOP: return False
        if "@" in s or re.search(r"\d{3}[-)\s]\d{3}-\d{4}", s): return False
        return len(s.split()) <= 6
    sib = h1.next_sibling
    hops = 0
    while sib and hops < 12:
        if getattr(sib, "get_text", None):
            t = re.sub(r"\s+", " ", sib.get_text(strip=True))
            if looks_like_title(t) and t[0].isupper():
                return t
        sib = sib.next_sibling
        hops += 1
    return None

def get_name(soup: BeautifulSoup) -> (Optional[str], Optional[str]):
    """
    The page shows the name in a prominent heading and the position (e.g., 'Professor') near it.
    Strategy:
      - grab the first big heading that repeats the name
      - look just below for a short role word/line (e.g., 'Professor')
    """
    # Name: try h1 first
    name = None
    for h in soup.find_all(["h1", "h2"]):
        t = text_clean(h.get_text())
        if t and len(t.split()) >= 2:
            # Heuristic: page title duplicates name; prefer the one near top
            name = t
            break


    return name

def next_list_items(label_node: Tag) -> List[str]:
    """
    From a label node, collect contiguous list-like items (bulleted <li> or plain lines)
    until the next known label or a big structural break.
    """
    items = []
    stop_texts = sum(LABELS.values(), [])
    # prefer following links/list items first
    cur = label_node
    # Look for explicit lists
    following_list = label_node.find_next(["ul", "ol"])
    if following_list and following_list.find_previous(lambda t: t == label_node or (isinstance(t, Tag) and t == label_node)):
        for li in following_list.find_all("li", recursive=False):
            s = text_clean(li.get_text(" "))
            if s:
                items.append(s)
        if items:
            return items

    # Fallback: collect successive block-level tags as separate items until a stop label appears
    cur = label_node
    for _ in range(50):
        cur = cur.find_next_sibling()
        if not cur:
            break
        s = text_clean(cur.get_text(" "))
        if not s:
            continue
        if s in stop_texts:
            break
        # Split lines if it looks like multiple entries separated by newlines
        if "\n" in cur.text:
            for line in [text_clean(x) for x in cur.text.split("\n")]:
                if line and line not in stop_texts:
                    items.append(line)
        else:
            items.append(s)
        # If we hit a long paragraph, keep it as one item (caller may join)
        # Stop if we collected something and next sibling is another labeled section
        nxt = cur.find_next_sibling()
        if nxt:
            ns = text_clean(nxt.get_text(" "))
            if ns in stop_texts:
                break
    return items

def next_text_block(label_node: Tag) -> str:
    """Grab the next meaningful paragraph/text after a label."""
    stop_texts = sum(LABELS.values(), [])
    cur = label_node
    chunks = []
    for _ in range(50):
        cur = cur.find_next()
        if not cur:
            break
        if isinstance(cur, Tag):
            s = text_clean(cur.get_text(" "))
        else:
            s = text_clean(str(cur))
        if not s:
            continue
        if s in stop_texts:
            break
        # Avoid re-capturing the label itself
        if s in stop_texts:
            continue
        # Prefer paragraphs first
        if isinstance(cur, Tag) and cur.name in ("p", "div", "section"):
            # stop once we got a clean paragraph
            if len(s) > 40:  # heuristic: likely the bio
                chunks.append(s)
                break
            else:
                chunks.append(s)
                # continue in case it spans multiple short nodes
                if len(" ".join(chunks)) > 200:
                    break
    return text_clean(" ".join(chunks))

def next_anchor_after(label_node: Tag) -> Optional[Tag]:
    """Return the first <a> after a label node."""
    return label_node.find_next("a")

def next_anchors_block(label_node: Tag) -> List[Tag]:
    """Return all anchors immediately after the label node before the next label."""
    stop_texts = sum(LABELS.values(), [])
    anchors = []
    cur = label_node
    for _ in range(80):
        cur = cur.find_next()
        if not cur:
            break
        s = text_clean(cur.get_text(" "))
        if s in stop_texts and isinstance(cur, Tag):
            break
        if isinstance(cur, Tag) and cur.name == "a":
            anchors.append(cur)
    # Deduplicate by href
    out = []
    seen = set()
    for a in anchors:
        href = a.get("href", "") or ""
        key = (a.get_text(strip=True), href)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out

def parse_profile(url: str) -> Dict:
    soup = fetch_html(url)

    name = get_name(soup)
    position = extract_position(soup)


    labels = find_label_nodes(soup)

    def one_of(keys: List[str]) -> Optional[Tag]:
        for k in keys:
            nodes = labels.get(k, [])
            if nodes:
                return nodes[0]
        return None

    # Organizations
    orgs = []
    org_node = one_of(["organizations"])
    if org_node:
        # Usually two simple lines (CRIS + EECS)
        orgs = [x for x in next_list_items(org_node) if x]

    # Email
    email = None
    email_node = one_of(["email"])
    if email_node:
        # email might be plain text or mailto link
        a = email_node.find_next("a")
        if a and ("mailto:" in (a.get("href") or "") or "@" in a.get_text(" ")):
            email = text_clean(a.get_text(" "))
        else:
            # fallback to the next visible text
            nxt = email_node.find_next(string=True)
            if nxt:
                email = text_clean(nxt)

    # Phone
    phone = None
    phone_node = one_of(["office_phone"])
    if phone_node:
        # Find nearest phone-like text
        phone_texts = []
        for candidate in [phone_node.find_next("a"), phone_node.find_next(string=True)]:
            if not candidate:
                continue
            s = text_clean(candidate.get_text(" ") if hasattr(candidate, "get_text") else str(candidate))
            if re.search(r"\(?\d{3}\)?[-\s]\d{3}[-]\d{4}", s):
                phone_texts.append(s)
        phone = phone_texts[0] if phone_texts else None

    # Address
    address = None
    addr_node = one_of(["address"])
    if addr_node:
        addr_lines = next_list_items(addr_node)
        if not addr_lines:
            # fallback: collect next few short lines
            addr_lines = []
            cur = addr_node
            for _ in range(6):
                cur = cur.find_next(string=True)
                if not cur:
                    break
                s = text_clean(str(cur))
                if not s or s in sum(LABELS.values(), []):
                    break
                addr_lines.append(s)
        # coalesce to multi-line address
        address = "\n".join([l for l in addr_lines if l])

    # Research website (name + link)
    research_website = {"name": None, "url": None}
    rw_node = one_of(["research_website"])
    if rw_node:
        a = next_anchor_after(rw_node)
        if a:
            research_website["name"] = text_clean(a.get_text(" "))
            research_website["url"] = a.get("href")

    # Degrees
    degrees = []
    deg_node = one_of(["degrees"])
    if deg_node:
        degrees = next_list_items(deg_node)
        # Sometimes degrees are stacked as separate blocks; ensure distinct
        degrees = [d for d in degrees if d and any(x in d for x in [",", "University"])]

    # Research expertise
    expertise = []
    exp_node = one_of(["research_expertise"])
    if exp_node:
        # Often comma-separated on one line
        raw = next_text_block(exp_node)
        if raw:
            expertise = [text_clean(x) for x in raw.split(",") if text_clean(x)]

    # Research groups (names + links)
    research_groups = []
    rg_node = one_of(["research_groups"])
    if rg_node:
        for a in next_anchors_block(rg_node):
            nm = text_clean(a.get_text(" "))
            href = a.get("href")
            if nm:
                research_groups.append({"name": nm, "url": href})

    # Biography
    biography = ""
    bio_node = one_of(["biography"])
    if bio_node:
        biography = next_text_block(bio_node)

    # Awards
    awards = []
    aw_node = one_of(["awards"])
    if aw_node:
        awards = next_list_items(aw_node)

    # Additional links
    additional_links = []
    al_node = one_of(["additional_links"])
    if al_node:
        for a in next_anchors_block(al_node):
            nm = text_clean(a.get_text(" "))
            href = a.get("href")
            if nm and href:
                additional_links.append({"name": nm, "url": href})

    # Organization (singular) – return as best-effort combined line if present
    organization = " | ".join(orgs) if orgs else None

    return {
        "name": name,
        "organization": organization,
        "organizations": orgs,
        "position": position,
        "email": email,
        "phone": phone,
        "address": address,
        "research_website": research_website,
        "degrees": degrees,
        "research_expertise": expertise,
        "research_groups": research_groups,
        "biography": biography,
        "awards": awards,
        "additional_links": additional_links,
        "source_url": url,
    }

