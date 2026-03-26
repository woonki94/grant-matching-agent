from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import re
import tempfile
from pathlib import Path

import boto3
import requests
from botocore.exceptions import ClientError

from config import settings


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
CHUNK_SUFFIX_RE = re.compile(r"__chunk_(\d{4})\.txt$", re.IGNORECASE)
KEEP_ELEMENT_TYPES = {
    "Title",
    "NarrativeText",
    "ListItem",
    "Table",
    "TableChunk",
    "Text",
}
NOISE_PATTERNS = [
    re.compile(r"\bcookie(s)?\b", re.IGNORECASE),
    re.compile(r"\bprivacy policy\b", re.IGNORECASE),
    re.compile(r"\bterms of use\b", re.IGNORECASE),
    re.compile(r"\ball rights reserved\b", re.IGNORECASE),
    re.compile(r"\bcopyright\b", re.IGNORECASE),
    re.compile(r"\bskip to (main )?content\b", re.IGNORECASE),
    re.compile(r"\bsign in\b|\blogin\b|\blog out\b", re.IGNORECASE),
    re.compile(r"\bcontact us\b|\bsitemap\b|\bmenu\b|\bsearch\b", re.IGNORECASE),
    re.compile(r"\bsubscribe\b|\bnewsletter\b|\bfollow us\b", re.IGNORECASE),
    re.compile(r"\bfacebook\b|\btwitter\b|\blinkedin\b|\byoutube\b", re.IGNORECASE),
    re.compile(r"\baccessibility\b|\bnon-discrimination\b", re.IGNORECASE),
    re.compile(r"\bpowered by\b|\bback to top\b", re.IGNORECASE),
]
MIN_TEXT_CHARS = 40
MAX_URL_RATIO = 0.20
DEFAULT_CHUNK_CHARS = 3000


def safe_filename(name: str) -> str:
    """Normalize untrusted file names into a safe, short ASCII-ish token."""
    name = (name or "downloaded_file").strip().replace(" ", "_")
    name = SAFE_NAME_RE.sub("_", name)
    return name[:200]


def guess_filename(url: str, headers: Dict[str, str]) -> str:
    """Infer a download filename from Content-Disposition, else URL path."""
    cd = headers.get("Content-Disposition") or headers.get("content-disposition")
    if cd:
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', cd)
        if m:
            return safe_filename(m.group(1))
    return safe_filename(Path(url).name or "downloaded_file")


def infer_ext(filename: str, content_type: Optional[str]) -> str:
    """Resolve a best-effort file extension from filename or MIME type."""
    ext = Path(filename).suffix.lower()
    if ext:
        return ext
    ctype = (content_type or "").lower()
    if "pdf" in ctype:
        return ".pdf"
    if "word" in ctype or "docx" in ctype:
        return ".docx"
    if "html" in ctype or "htm" in ctype:
        return ".html"
    if "text/plain" in ctype:
        return ".txt"
    return ext or ""


def _normalize_text(text: str) -> str:
    """Canonicalize whitespace/newlines to reduce noisy formatting variance."""
    t = str(text or "")
    t = t.replace("\u00A0", " ")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in t.split("\n")]
    lines = [ln for ln in lines if ln]
    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _looks_noisy(text: str, *, min_text_chars: int, max_url_ratio: float) -> bool:
    """Heuristically detect low-value boilerplate/navigation fragments."""
    t = str(text or "").strip()
    if len(t) < int(min_text_chars):
        return True

    alpha_chars = sum(1 for ch in t if ch.isalpha())
    if alpha_chars < 8:
        return True

    digit_chars = sum(1 for ch in t if ch.isdigit())
    if len(t) > 0 and (digit_chars / float(len(t))) > 0.35:
        return True

    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) < 6:
        return True

    upper_letters = sum(1 for ch in t if ch.isupper())
    if alpha_chars > 0 and (upper_letters / float(alpha_chars)) > 0.75 and len(words) <= 20:
        return True

    url_tokens = re.findall(r"https?://\S+|www\.\S+", t)
    url_chars = sum(len(x) for x in url_tokens)
    if len(t) > 0 and (url_chars / float(len(t))) > float(max_url_ratio):
        return True

    for pat in NOISE_PATTERNS:
        if pat.search(t):
            return True
    return False


def _extract_text_from_unstructured_elements(elements: List[Any]) -> str:
    """Filter/clean/deduplicate unstructured elements and join into plain text."""
    lines: List[str] = []
    seen = set()
    for el in list(elements or []):
        if type(el).__name__ not in KEEP_ELEMENT_TYPES:
            continue
        txt = str(getattr(el, "text", "") or "").strip()
        if _looks_noisy(txt, min_text_chars=MIN_TEXT_CHARS, max_url_ratio=MAX_URL_RATIO):
            continue
        txt = _normalize_text(txt)
        if not txt:
            continue
        dedup_key = re.sub(r"\s+", " ", txt.lower()).strip()
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        lines.append(txt)
    return "\n\n".join(lines).strip()


def _split_long_text_preserving_words(text: str, *, max_chars: int) -> List[str]:
    """Split long text into <= max_chars pieces, preferring sentence boundaries."""
    t = str(text or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]

    out: List[str] = []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    cur = ""
    for sent in sentences:
        candidate = sent if not cur else f"{cur} {sent}"
        if len(candidate) <= max_chars:
            cur = candidate
            continue
        if cur:
            out.append(cur.strip())
            cur = ""
        if len(sent) <= max_chars:
            cur = sent
            continue

        # Fallback for very long sentence: split by words.
        words = [w for w in sent.split() if w]
        wcur = ""
        for w in words:
            wcand = w if not wcur else f"{wcur} {w}"
            if len(wcand) <= max_chars:
                wcur = wcand
            else:
                if wcur:
                    out.append(wcur.strip())
                wcur = w
        if wcur:
            cur = wcur.strip()

    if cur:
        out.append(cur.strip())
    return [x for x in out if x]


def chunk_text_for_embedding(
    text: str,
    *,
    max_chars: int = DEFAULT_CHUNK_CHARS,
) -> List[str]:
    """
    Build stable ~max_chars chunks from cleaned text.

    Strategy:
      1) split by paragraph blocks
      2) pack paragraphs up to max_chars
      3) split oversized blocks by sentence/word boundaries
    """
    safe_max = max(200, int(max_chars or DEFAULT_CHUNK_CHARS))
    normalized = _normalize_text(text)
    if not normalized:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", normalized) if p.strip()]
    chunks: List[str] = []
    cur = ""

    for para in paragraphs:
        if len(para) > safe_max:
            if cur:
                chunks.append(cur.strip())
                cur = ""
            chunks.extend(_split_long_text_preserving_words(para, max_chars=safe_max))
            continue

        candidate = para if not cur else f"{cur}\n\n{para}"
        if len(candidate) <= safe_max:
            cur = candidate
        else:
            if cur:
                chunks.append(cur.strip())
            cur = para

    if cur:
        chunks.append(cur.strip())

    out: List[str] = []
    seen = set()
    for ch in chunks:
        c = _normalize_text(ch)
        if not c:
            continue
        key = re.sub(r"\s+", " ", c.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _extract_text_with_unstructured_file(filename: str) -> str:
    """Partition a local file with Unstructured and return cleaned text."""
    from unstructured.partition.auto import partition

    elements = partition(filename=filename)
    return _extract_text_from_unstructured_elements(elements)


def _extract_text_with_unstructured_bytes(
    data: bytes,
    *,
    filename: str,
    content_type: Optional[str],
) -> Tuple[str, str]:
    """Persist bytes to a temp file, run Unstructured, and report detected type."""
    ext = infer_ext(filename, content_type)
    suffix = ext if ext else ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        text = _extract_text_with_unstructured_file(tmp.name)
    detected = ext.lstrip(".") if ext else "unknown"
    return text, detected


def _extract_html_with_unstructured(html: str) -> str:
    """Run Unstructured over rendered HTML content provided as a string."""
    with tempfile.NamedTemporaryFile("w", suffix=".html", encoding="utf-8", delete=True) as tmp:
        tmp.write(html)
        tmp.flush()
        return _extract_text_with_unstructured_file(tmp.name)


def _fetch_html_with_playwright(
    url: str,
    *,
    timeout: int,
    user_agent: str,
) -> Tuple[str, Optional[int], Optional[str]]:
    """Fetch fully rendered HTML with Playwright (headless Chromium)."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return "", None, f"playwright_import_error: {type(e).__name__}: {e}"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(user_agent=user_agent)
                response = page.goto(url, timeout=int(timeout * 1000))
                try:
                    page.wait_for_selector("body", timeout=max(1000, int(timeout * 500)))
                except Exception:
                    pass
                page.wait_for_timeout(500)
                html = str(page.content() or "")
                if not html.strip():
                    _ = page.inner_text("body")
                    html = str(page.content() or "")
                status = int(response.status) if response is not None else None
                return html, status, None
            finally:
                browser.close()
    except Exception as e:
        return "", None, f"playwright_fetch_error: {type(e).__name__}: {e}"


def extract_text_from_file_bytes(
    data: bytes,
    *,
    filename: str,
    content_type: Optional[str] = None,
) -> str:
    """Public helper: extract cleaned text from in-memory file bytes."""
    text, _ = _extract_text_with_unstructured_bytes(
        data,
        filename=filename,
        content_type=content_type,
    )
    return text


def fetch_and_extract_one(
    url: str,
    session: Optional[requests.Session] = None,
    timeout: int = 60,
    user_agent: str = "GrantFetcher/1.0 (+https://example.org)",
) -> dict:
    """
    URL extraction strategy (no legacy fallback):
      1) playwright -> rendered html -> unstructured
      2) requests download -> unstructured-bytes
    """
    s = session or requests.Session()
    headers = {"User-Agent": user_agent}
    playwright_html, playwright_status, playwright_error = _fetch_html_with_playwright(
        url,
        timeout=timeout,
        user_agent=user_agent,
    )
    if playwright_html.strip():
        try:
            text = _extract_html_with_unstructured(playwright_html)
            if text.strip():
                out = {
                    "url": url,
                    "filename": safe_filename(Path(url).name or "downloaded_file.html"),
                    "content_type": "text/html; charset=utf-8",
                    "content_length": len(playwright_html.encode("utf-8", errors="ignore")),
                    "detected_type": "html",
                    "text": text,
                    "status_code": playwright_status or 200,
                    "error": None,
                }
                if playwright_error:
                    out["playwright_warning"] = playwright_error
                return out
        except Exception as ex:
            # Continue to direct HTTP fetch path below.
            playwright_error = f"playwright_extract_error: {type(ex).__name__}: {ex}"

    try:
        r = s.get(url, headers=headers, timeout=timeout)
        status = r.status_code
        if status != 200:
            out = {
                "url": url,
                "filename": guess_filename(url, r.headers) if r.headers else None,
                "content_type": (r.headers.get("Content-Type") if r.headers else None),
                "content_length": (
                    int(r.headers.get("Content-Length"))
                    if r.headers and (r.headers.get("Content-Length") or "").isdigit()
                    else None
                ),
                "detected_type": None,
                "text": None,
                "status_code": status,
                "error": f"HTTP {status}",
            }
            if playwright_error:
                out["playwright_warning"] = playwright_error
            return out

        filename = guess_filename(url, r.headers)
        ctype = r.headers.get("Content-Type")
        clen = r.headers.get("Content-Length")
        clen_int = int(clen) if clen and clen.isdigit() else None

        try:
            text, detected = _extract_text_with_unstructured_bytes(
                r.content,
                filename=filename,
                content_type=ctype,
            )
        except Exception as ex:
            out = {
                "url": url,
                "filename": filename,
                "content_type": ctype,
                "content_length": clen_int,
                "detected_type": None,
                "text": None,
                "status_code": 200,
                "error": f"unstructured_extraction_error: {type(ex).__name__}: {ex}",
            }
            if playwright_error:
                out["playwright_warning"] = playwright_error
            return out

        out = {
            "url": url,
            "filename": filename,
            "content_type": ctype,
            "content_length": clen_int,
            "detected_type": detected or "unknown",
            "text": text,
            "status_code": 200,
            "error": None,
        }
        if playwright_error:
            out["playwright_warning"] = playwright_error
        return out

    except requests.RequestException as ex:
        out = {
            "url": url,
            "filename": None,
            "content_type": None,
            "content_length": None,
            "detected_type": None,
            "text": None,
            "status_code": None,
            "error": f"request_error: {ex}",
        }
        if playwright_error:
            out["playwright_warning"] = playwright_error
        return out


def load_extracted_content(
    rows: List[Any],
    url_attr: str,
    title_attr: Optional[str] = None,
    *,
    group_chunks: bool = True,
    include_row_meta: bool = False,
) -> List[Dict[str, Any]]:
    """Load extracted content from S3, optionally stitching chunk rows by source."""
    out: List[Dict[str, Any]] = []

    bucket_default = (settings.extracted_content_bucket or "").strip()
    if not bucket_default:
        return out

    region = (settings.aws_region or "").strip()
    profile = settings.aws_profile

    session = (
        boto3.Session(profile_name=profile, region_name=region)
        if profile
        else boto3.Session(region_name=region)
    )
    s3 = session.client("s3")

    def _parse_bucket_key(content_path: str) -> Optional[Tuple[str, str]]:
        """Normalize either s3:// URI or plain key into (bucket, key)."""
        cp = (content_path or "").strip()
        if not cp:
            return None

        if cp.startswith("s3://"):
            rest = cp[5:]
            parts = rest.split("/", 1)
            if len(parts) != 2:
                return None
            b, k = parts[0].strip(), parts[1].lstrip("/")
            if not b or not k:
                return None
            return b, k

        return bucket_default, cp.lstrip("/")

    def _row_chunk_index(row: Any) -> int:
        """Best-effort chunk index from row field, else from content-path suffix."""
        try:
            if hasattr(row, "chunk_index"):
                return int(getattr(row, "chunk_index") or 0)
        except Exception:
            pass
        cp = str(getattr(row, "content_path", "") or "")
        m = CHUNK_SUFFIX_RE.search(cp)
        if m:
            try:
                return max(0, int(m.group(1)) - 1)
            except Exception:
                return 0
        return 0

    rows_sorted = sorted(
        list(rows or []),
        key=lambda r: (
            str(getattr(r, url_attr, "") or ""),
            str(getattr(r, title_attr, "") or "") if title_attr else "",
            _row_chunk_index(r),
            int(getattr(r, "id", 0) or 0),
        ),
    )
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for r in rows_sorted:
        if getattr(r, "extract_status", None) not in ("done", "success"):
            continue
        if getattr(r, "extract_error", None):
            continue

        content_path = getattr(r, "content_path", None)
        if not content_path:
            continue

        parsed = _parse_bucket_key(str(content_path))
        if not parsed:
            continue
        use_bucket, key = parsed

        try:
            resp = s3.get_object(Bucket=use_bucket, Key=key)
            text = resp["Body"].read().decode("utf-8", errors="ignore").strip()
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404"):
                continue
            raise
        except Exception:
            continue

        if not text:
            continue

        if not group_chunks:
            item: Dict[str, Any] = {
                "url": getattr(r, url_attr, None),
                "content": text,
            }
            if title_attr:
                item["title"] = getattr(r, title_attr, None)
            if include_row_meta:
                item["row_id"] = int(getattr(r, "id", 0) or 0)
                item["chunk_index"] = int(_row_chunk_index(r))
            out.append(item)
            continue

        url_value = str(getattr(r, url_attr, "") or "")
        title_value = str(getattr(r, title_attr, "") or "") if title_attr else ""
        group_key = (url_value, title_value)
        if group_key not in grouped:
            item = {"url": getattr(r, url_attr, None), "parts": []}
            if title_attr:
                item["title"] = getattr(r, title_attr, None)
            grouped[group_key] = item
        grouped[group_key]["parts"].append(text)

    if group_chunks:
        for item in grouped.values():
            parts = [str(x).strip() for x in list(item.pop("parts", []) or []) if str(x).strip()]
            merged = "\n\n".join(parts).strip()
            if not merged:
                continue
            item["content"] = merged
            out.append(item)

    return out


def fetch_and_extract_batch(urls: List[str]) -> List[dict]:
    """Batch helper around fetch_and_extract_one() using a shared HTTP session."""
    s = requests.Session()
    return [fetch_and_extract_one(u, session=s) for u in urls]
