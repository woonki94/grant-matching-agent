from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any

import io
import re
import csv
import tempfile
import requests
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# extraction libs (no local storage; docx uses temp only)
from pdfminer.high_level import extract_text as pdf_extract_text
import docx2txt
import openpyxl

from config import settings
from utils.html_to_text import _HTMLToText

try:
    import xlrd  # for legacy .xls (use xlrd==1.2.0)
    HAS_XLRD = True
except Exception:
    HAS_XLRD = False


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    name = (name or "downloaded_file").strip().replace(" ", "_")
    name = SAFE_NAME_RE.sub("_", name)
    return name[:200]


def guess_filename(url: str, headers: Dict[str, str]) -> str:
    cd = headers.get("Content-Disposition") or headers.get("content-disposition")
    if cd:
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', cd)
        if m:
            return safe_filename(m.group(1))
    return safe_filename(Path(url).name or "downloaded_file")


def infer_ext(filename: str, content_type: Optional[str]) -> str:
    ext = Path(filename).suffix.lower()
    if ext:
        return ext
    ctype = (content_type or "").lower()
    if "pdf" in ctype:
        return ".pdf"
    if "word" in ctype or "docx" in ctype:
        return ".docx"
    if "spreadsheetml" in ctype or "xlsx" in ctype or "excel" in ctype:
        return ".xlsx"
    if "ms-excel" in ctype or "xls" in ctype:
        return ".xls"
    if "csv" in ctype:
        return ".csv"
    if "html" in ctype or "htm" in ctype:
        return ".html"
    return ext or ""


def extract_html_bytes(data: bytes) -> str:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            text = data.decode(enc, errors="strict")
            break
        except Exception:
            continue
    else:
        text = data.decode("utf-8", errors="ignore")

    parser = _HTMLToText()
    parser.feed(text)
    return parser.get_text()


def extract_pdf_bytes(data: bytes) -> str:
    return pdf_extract_text(io.BytesIO(data)) or ""


def extract_docx_bytes_via_tempfile(data: bytes) -> str:
    # docx2txt expects a file path; use a NamedTemporaryFile and cleanup
    with tempfile.NamedTemporaryFile(suffix=".docx") as tmp:
        tmp.write(data)
        tmp.flush()
        return docx2txt.process(tmp.name) or ""


def extract_xlsx_bytes(data: bytes) -> str:
    wb = openpyxl.load_workbook(filename=io.BytesIO(data), data_only=True, read_only=True)
    out: List[str] = []
    for ws in wb.worksheets:
        out.append(f"# Sheet: {ws.title}")
        for row in ws.iter_rows(values_only=True):
            cells = [(str(c) if c is not None else "").strip() for c in row]
            if any(cells):
                out.append("\t".join(cells))
    return "\n".join(out)


def extract_xls_bytes(data: bytes) -> str:
    if not HAS_XLRD:
        return "[[xlrd not installed; cannot parse .xls files]]"
    book = xlrd.open_workbook(file_contents=data)
    out: List[str] = []
    for si in range(book.nsheets):
        sh = book.sheet_by_index(si)
        out.append(f"# Sheet: {sh.name}")
        for r in range(sh.nrows):
            row_vals = [str(sh.cell_value(r, c)).strip() for c in range(sh.ncols)]
            if any(row_vals):
                out.append("\t".join(row_vals))
    return "\n".join(out)


def extract_csv_bytes(data: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            text = data.decode(enc, errors="strict")
            break
        except Exception:
            continue
    else:
        text = data.decode("utf-8", errors="ignore")

    parts: List[str] = []
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        parts.append("\t".join(cell.strip() for cell in row))
    return "\n".join(parts)


def extract_text_from_bytes(data: bytes, filename: str, content_type: Optional[str]) -> Tuple[str, str]:
    """
    Returns (text, detected_type)
    detected_type ∈ {"pdf","docx","xlsx","xls","csv","html","unknown","error"}
    """
    ext = infer_ext(filename, content_type)
    try:
        if ext == ".pdf":
            return extract_pdf_bytes(data), "pdf"
        if ext == ".docx":
            return extract_docx_bytes_via_tempfile(data), "docx"
        if ext == ".xlsx":
            return extract_xlsx_bytes(data), "xlsx"
        if ext == ".xls":
            return extract_xls_bytes(data), "xls"
        if ext == ".csv":
            return extract_csv_bytes(data), "csv"
        if ext in (".html", ".htm"):
            return extract_html_bytes(data), "html"

        # Fallback by MIME
        ctype = (content_type or "").lower()
        if "pdf" in ctype:
            return extract_pdf_bytes(data), "pdf"
        if "word" in ctype or "docx" in ctype:
            return extract_docx_bytes_via_tempfile(data), "docx"
        if "spreadsheetml" in ctype or "xlsx" in ctype or "excel" in ctype:
            return extract_xlsx_bytes(data), "xlsx"
        if "ms-excel" in ctype or "xls" in ctype:
            return extract_xlsx_bytes(data), "xls"
        if "csv" in ctype:
            return extract_csv_bytes(data), "csv"
        if "html" in ctype or "htm" in ctype or "text/html" in ctype:
            return extract_html_bytes(data), "html"

        # Last-ditch attempts
        try:
            txt = extract_pdf_bytes(data)
            if txt.strip():
                return txt, "pdf?"
        except Exception:
            pass
        try:
            txt = extract_docx_bytes_via_tempfile(data)
            if txt.strip():
                return txt, "docx?"
        except Exception:
            pass
        return "", "unknown"

    except Exception as e:
        return f"[[Extraction error: {e}]]", "error"


# --------- public API (no disk writes; docx uses temp only) ----------

def fetch_and_extract_one(
    url: str,
    session: Optional[requests.Session] = None,
    timeout: int = 60,
    user_agent: str = "GrantFetcher/1.0 (+https://example.org)",
) -> dict:
    """
    Always returns a dict. On HTTP errors or extraction errors, returns null-ish content.

    Return shape:
      {
        "url": str,
        "filename": str | None,
        "content_type": str | None,
        "content_length": int | None,
        "detected_type": str | None,
        "text": str | None,
        "status_code": int | None,
        "error": str | None
      }
    """
    s = session or requests.Session()
    headers = {"User-Agent": user_agent}

    try:
        r = s.get(url, headers=headers, timeout=timeout)
        status = r.status_code
        if status != 200:
            return {
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

        filename = guess_filename(url, r.headers)
        ctype = r.headers.get("Content-Type")
        clen = r.headers.get("Content-Length")
        clen_int = int(clen) if clen and clen.isdigit() else None

        try:
            text, detected = extract_text_from_bytes(r.content, filename, ctype)
        except Exception as ex:
            return {
                "url": url,
                "filename": filename,
                "content_type": ctype,
                "content_length": clen_int,
                "detected_type": None,
                "text": None,
                "status_code": 200,
                "error": f"extraction_error: {ex}",
            }

        return {
            "url": url,
            "filename": filename,
            "content_type": ctype,
            "content_length": clen_int,
            "detected_type": detected,
            "text": text,
            "status_code": 200,
            "error": None,
        }

    except requests.RequestException as ex:
        return {
            "url": url,
            "filename": None,
            "content_type": None,
            "content_length": None,
            "detected_type": None,
            "text": None,
            "status_code": None,
            "error": f"request_error: {ex}",
        }


def load_extracted_content(
    rows: List[Any],
    url_attr: str,
    title_attr: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Loader for extracted content from DB rows.

    Expects r.content_path to be either:
      - S3 key: "<prefix>/<subdir>/<file>.txt"
      - or full S3 URL: "s3://bucket/<prefix>/<subdir>/<file>.txt"

    This function does NOT try to attach prefixes automatically.
    Prefix selection is handled when writing (import/extraction pipeline),
    and the DB should store the full key (or full s3:// URI).
    """
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

    for r in rows or []:
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

        item: Dict[str, Any] = {
            "url": getattr(r, url_attr, None),
            "content": text,
        }
        if title_attr:
            item["title"] = getattr(r, title_attr, None)

        out.append(item)

    return out


def fetch_and_extract_batch(urls: List[str]) -> List[dict]:
    s = requests.Session()
    return [fetch_and_extract_one(u, session=s) for u in urls]
