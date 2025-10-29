#!/usr/bin/env python3
from __future__ import annotations
import os
import re
import csv
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import requests

# --- Extractors (install deps in requirements below) ---
from pdfminer.high_level import extract_text as pdf_extract_text
import docx2txt
import openpyxl  # for .xlsx
try:
    import xlrd   # for legacy .xls
    HAS_XLRD = True
except ImportError:
    HAS_XLRD = False


# ---------- Utilities ----------
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

def safe_filename(name: str) -> str:
    # normalize spaces and strip unsafe chars
    name = name.strip().replace(" ", "_")
    name = SAFE_NAME_RE.sub("_", name)
    return name[:200]  # avoid absurdly long filenames

def guess_filename(url: str, headers: dict) -> str:
    # Try Content-Disposition
    cd = headers.get("Content-Disposition") or headers.get("content-disposition")
    if cd:
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', cd)
        if m:
            return safe_filename(m.group(1))
    # Fallback to URL path
    return safe_filename(Path(url).name or "downloaded_file")

def infer_ext(name: str, headers: dict) -> str:
    # If name already has an extension, keep it
    ext = Path(name).suffix.lower()
    if ext:
        return ext
    # Try from MIME type
    ctype = (headers.get("Content-Type") or "").lower()
    if "pdf" in ctype:
        return ".pdf"
    if "word" in ctype or "docx" in ctype:
        return ".docx"
    if "excel" in ctype or "spreadsheetml" in ctype or "xlsx" in ctype:
        return ".xlsx"
    if "ms-excel" in ctype or "xls" in ctype:
        return ".xls"
    if "csv" in ctype:
        return ".csv"
    return ext or ""


# ---------- Download ----------
def download_file(url: str, outdir: Path, session: Optional[requests.Session] = None) -> Path:
    s = session or requests.Session()
    r = s.get(url, stream=True, timeout=60)
    r.raise_for_status()

    fname = guess_filename(url, r.headers)
    ext = infer_ext(fname, r.headers)
    if not fname.lower().endswith(ext) and ext:
        fname += ext
    outpath = outdir / fname

    with open(outpath, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)
    return outpath


# ---------- Extractors ----------
def extract_pdf(path: Path) -> str:
    return pdf_extract_text(str(path)) or ""

def extract_docx(path: Path) -> str:
    # docx2txt can also extract images if given an image dir; we only need text
    return docx2txt.process(str(path)) or ""

def extract_xlsx(path: Path) -> str:
    wb = openpyxl.load_workbook(str(path), data_only=True)
    parts: List[str] = []
    for ws in wb.worksheets:
        parts.append(f"# Sheet: {ws.title}")
        for row in ws.iter_rows(values_only=True):
            cells = [(str(c) if c is not None else "").strip() for c in row]
            if any(cells):
                parts.append("\t".join(cells))
    return "\n".join(parts)

def extract_xls(path: Path) -> str:
    if not HAS_XLRD:
        return "[[xlrd not installed; cannot parse .xls files]]"
    book = xlrd.open_workbook(str(path))
    parts: List[str] = []
    for si in range(book.nsheets):
        sh = book.sheet_by_index(si)
        parts.append(f"# Sheet: {sh.name}")
        for r in range(sh.nrows):
            row_vals = [str(sh.cell_value(r, c)).strip() for c in range(sh.ncols)]
            if any(v for v in row_vals):
                parts.append("\t".join(row_vals))
    return "\n".join(parts)

def extract_csv(path: Path) -> str:
    parts: List[str] = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            parts.append("\t".join(cell.strip() for cell in row))
    return "\n".join(parts)

def extract_text_from_file(path: Path) -> Tuple[str, str]:
    """
    Returns (text, detected_type)
    """
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            return extract_pdf(path), "pdf"
        if ext == ".docx":
            return extract_docx(path), "docx"
        if ext == ".xlsx":
            return extract_xlsx(path), "xlsx"
        if ext == ".xls":
            return extract_xls(path), "xls"
        if ext == ".csv":
            return extract_csv(path), "csv"
        # Fallback: try PDF, then docx
        try:
            txt = extract_pdf(path)
            if txt.strip():
                return txt, "pdf?"
        except Exception:
            pass
        try:
            txt = extract_docx(path)
            if txt.strip():
                return txt, "docx?"
        except Exception:
            pass
        return "", "unknown"
    except Exception as e:
        return f"[[Extraction error: {e}]]", "error"


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(
        description="Download grant attachments (pdf/docx/xlsx/xls/csv) and extract text."
    )
    p.add_argument("--out", default="downloads", help="Download directory (default: downloads)")
    p.add_argument("--txtout", default="extracted", help="Extracted text output dir (default: extracted)")
    p.add_argument("urls", nargs="+", help="One or more attachment URLs")
    args = p.parse_args()

    dl_dir = Path(args.out).resolve()
    txt_dir = Path(args.txtout).resolve()
    dl_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    for url in args.urls:
        print(f"↓ Downloading: {url}")
        fpath = download_file(url, dl_dir, session=session)
        print(f"   Saved to: {fpath.name}")

        text, kind = extract_text_from_file(fpath)
        outname = safe_filename(fpath.stem) + ".txt"
        outpath = txt_dir / outname
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"   Extracted as ({kind}) → {outpath.name}\n")

    print(f"✅ Done. Downloads in {dl_dir}, extracted text in {txt_dir}")

if __name__ == "__main__":
    main()