from __future__ import annotations

import argparse
import inspect
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

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
    re.compile(r"\bskip to (main )?content\b", re.IGNORECASE),
    re.compile(r"\bsign in\b|\blogin\b|\blog out\b", re.IGNORECASE),
    re.compile(r"\bcontact us\b|\bsitemap\b", re.IGNORECASE),
]


def _is_url(value: str) -> bool:
    try:
        p = urlparse(str(value or "").strip())
    except Exception:
        return False
    return p.scheme in {"http", "https"} and bool(p.netloc)


def _fetch_html_with_playwright(url: str, *, timeout_ms: int) -> str:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, timeout=int(timeout_ms))
            try:
                page.wait_for_selector("body", timeout=max(1000, int(timeout_ms // 2)))
            except Exception:
                pass
            page.wait_for_timeout(500)
            html = str(page.content() or "")
            if html.strip():
                return html
            _ = page.inner_text("body")
            return str(page.content() or "")
        finally:
            browser.close()


def _normalize_text(text: str) -> str:
    t = str(text or "")
    t = t.replace("\u00A0", " ")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in t.split("\n")]
    lines = [ln for ln in lines if ln]
    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _looks_noisy(text: str, *, min_text_chars: int, max_url_ratio: float) -> bool:
    t = str(text or "").strip()
    if len(t) < int(min_text_chars):
        return True

    alpha_chars = sum(1 for ch in t if ch.isalpha())
    if alpha_chars < 8:
        return True

    url_tokens = re.findall(r"https?://\S+|www\.\S+", t)
    url_chars = sum(len(x) for x in url_tokens)
    if len(t) > 0 and (url_chars / float(len(t))) > float(max_url_ratio):
        return True

    for pat in NOISE_PATTERNS:
        if pat.search(t):
            return True
    return False


def _preprocess_elements(
    elements: List[Any],
    *,
    min_text_chars: int,
    max_url_ratio: float,
) -> tuple[List[Any], Dict[str, Any], List[str]]:
    raw_count = len(list(elements or []))
    cleaned_elements: List[Any] = []
    seen = set()

    dropped_type = 0
    dropped_noise = 0
    dropped_empty_after_clean = 0
    dropped_duplicate = 0

    cleaned_texts: List[str] = []

    for el in list(elements or []):
        el_type = type(el).__name__
        if el_type not in KEEP_ELEMENT_TYPES:
            dropped_type += 1
            continue

        raw_text = str(getattr(el, "text", "") or "").strip()
        if _looks_noisy(raw_text, min_text_chars=min_text_chars, max_url_ratio=max_url_ratio):
            dropped_noise += 1
            continue

        txt = _normalize_text(raw_text)
        if not txt:
            dropped_empty_after_clean += 1
            continue

        dedup_key = re.sub(r"\s+", " ", txt.lower()).strip()
        if dedup_key in seen:
            dropped_duplicate += 1
            continue
        seen.add(dedup_key)

        # Keep element metadata/structure, update text in-place.
        try:
            setattr(el, "text", txt)
        except Exception:
            pass

        cleaned_elements.append(el)
        cleaned_texts.append(txt)

    stats = {
        "raw_element_count": raw_count,
        "kept_element_count": len(cleaned_elements),
        "dropped_type": dropped_type,
        "dropped_noise": dropped_noise,
        "dropped_empty_after_clean": dropped_empty_after_clean,
        "dropped_duplicate": dropped_duplicate,
        "keep_types": sorted(list(KEEP_ELEMENT_TYPES)),
    }
    return cleaned_elements, stats, cleaned_texts


def _parse_one(
    target: str,
    *,
    use_playwright_for_links: bool,
    playwright_timeout_ms: int,
    use_basic_chunking: bool,
    max_characters: int,
    new_after_n_chars: int,
    combine_text_under_n_chars: int,
    overlap: int,
    overlap_all: bool,
    apply_cleanup: bool,
    min_text_chars: int,
    max_url_ratio: float,
) -> Dict[str, Any]:
    try:
        from unstructured.partition.auto import partition
        from unstructured.partition.html import partition_html
        from unstructured.partition.pdf import partition_pdf
    except Exception as e:
        raise RuntimeError(
            "Unstructured is not available. Install it first, for example:\n"
            "  pip install 'unstructured[all-docs]'\n"
            f"Import error: {type(e).__name__}: {e}"
        ) from e

    value = str(target or "").strip()
    if not value:
        return {"source": target, "ok": False, "error": "Empty input"}

    try:
        extraction_method = ""
        playwright_error = ""

        if _is_url(value):
            html = ""
            if use_playwright_for_links:
                try:
                    html = _fetch_html_with_playwright(
                        value,
                        timeout_ms=int(playwright_timeout_ms),
                    )
                    extraction_method = "playwright_html_then_unstructured"
                except Exception as e:
                    playwright_error = f"{type(e).__name__}: {e}"

            if html:
                with tempfile.NamedTemporaryFile("w", suffix=".html", encoding="utf-8", delete=True) as tf:
                    tf.write(html)
                    tf.flush()
                    elements = partition_html(filename=tf.name)
            else:
                elements = partition_html(url=value)
                extraction_method = "unstructured_url"
        else:
            p = Path(value).expanduser().resolve()
            if not p.exists() or not p.is_file():
                return {"source": value, "ok": False, "error": f"File not found: {p}"}

            suffix = p.suffix.lower()
            if suffix == ".pdf":
                elements = partition_pdf(filename=str(p))
                extraction_method = "pdf"
            elif suffix == ".docx":
                elements = partition(filename=str(p))
                extraction_method = "docx:auto"
            elif suffix == ".doc":
                elements = partition(filename=str(p))
                extraction_method = "doc:auto"
            else:
                elements = partition(filename=str(p))
                extraction_method = f"auto:{suffix or 'noext'}"

        raw_texts: List[str] = []
        for el in elements:
            txt = str(getattr(el, "text", "") or "").strip()
            if txt:
                raw_texts.append(txt)

        cleanup_stats: Dict[str, Any] = {
            "raw_element_count": len(elements),
            "kept_element_count": len(elements),
            "dropped_type": 0,
            "dropped_noise": 0,
            "dropped_empty_after_clean": 0,
            "dropped_duplicate": 0,
            "keep_types": sorted(list(KEEP_ELEMENT_TYPES)),
        }
        cleaned_elements = list(elements)
        texts = list(raw_texts)

        if apply_cleanup:
            cleaned_elements, cleanup_stats, texts = _preprocess_elements(
                list(elements),
                min_text_chars=int(min_text_chars),
                max_url_ratio=float(max_url_ratio),
            )

        payload: Dict[str, Any] = {
            "source": value,
            "ok": True,
            "extraction_method": extraction_method,
            "element_count": len(cleaned_elements),
            "element_count_raw": len(elements),
            "non_empty_text_count": len(texts),
            "preview": texts[:5],
            "text": "\n\n".join(texts),
            "cleanup": {
                "enabled": bool(apply_cleanup),
                "min_text_chars": int(min_text_chars),
                "max_url_ratio": float(max_url_ratio),
                **cleanup_stats,
            },
        }

        if use_basic_chunking:
            try:
                from unstructured.chunking.basic import chunk_elements

                sig = inspect.signature(chunk_elements)
                supported = set(sig.parameters.keys())
                kwargs: Dict[str, Any] = {}
                if "max_characters" in supported:
                    kwargs["max_characters"] = int(max_characters)
                if "new_after_n_chars" in supported:
                    kwargs["new_after_n_chars"] = int(new_after_n_chars)
                if "combine_text_under_n_chars" in supported:
                    kwargs["combine_text_under_n_chars"] = int(combine_text_under_n_chars)
                if "overlap" in supported:
                    kwargs["overlap"] = int(overlap)
                if "overlap_all" in supported:
                    kwargs["overlap_all"] = bool(overlap_all)

                chunked = chunk_elements(cleaned_elements, **kwargs)
                chunk_texts: List[str] = []
                for ch in list(chunked or []):
                    txt = str(getattr(ch, "text", "") or "").strip()
                    if txt:
                        chunk_texts.append(txt)

                payload["chunking"] = {
                    "strategy": "basic",
                    "max_characters": int(max_characters),
                    "new_after_n_chars": int(new_after_n_chars),
                    "combine_text_under_n_chars": int(combine_text_under_n_chars),
                    "overlap": int(overlap),
                    "overlap_all": bool(overlap_all),
                    "supported_chunk_kwargs": sorted(list(kwargs.keys())),
                    "chunk_count": len(chunk_texts),
                    "chunk_preview": chunk_texts[:5],
                    "chunk_text": "\n\n".join(chunk_texts),
                }
            except Exception as e:
                payload["chunking"] = {
                    "strategy": "basic",
                    "error": f"{type(e).__name__}: {e}",
                }

        if _is_url(value) and playwright_error:
            payload["playwright_warning"] = playwright_error
        return payload
    except Exception as e:
        return {
            "source": value,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test Unstructured extraction for links, PDF, and Word files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more inputs: URL(s) and/or local file path(s) (.pdf, .doc, .docx).",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON results.")
    parser.add_argument(
        "--no-playwright-links",
        action="store_true",
        help="Disable Playwright for URL inputs and use direct Unstructured URL parsing only.",
    )
    parser.add_argument(
        "--playwright-timeout-ms",
        type=int,
        default=30000,
        help="Playwright page load timeout in milliseconds for URL inputs.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable 5-step cleanup pipeline before chunking.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=25,
        help="Drop very short elements below this length.",
    )
    parser.add_argument(
        "--max-url-ratio",
        type=float,
        default=0.35,
        help="Drop elements when URL character ratio exceeds this value.",
    )
    parser.add_argument("--basic-chunk", action="store_true", help="Apply Unstructured basic chunking.")
    parser.add_argument("--max-characters", type=int, default=1200, help="basic chunk hard max chars.")
    parser.add_argument("--new-after-n-chars", type=int, default=1000, help="basic chunk soft max chars.")
    parser.add_argument(
        "--combine-text-under-n-chars",
        type=int,
        default=200,
        help="Combine short fragments under this length (if supported by installed unstructured).",
    )
    parser.add_argument("--overlap", type=int, default=120, help="Chunk overlap chars.")
    parser.add_argument("--overlap-all", action="store_true", help="Apply overlap across all chunks.")
    args = parser.parse_args()

    results: List[Dict[str, Any]] = []
    for target in args.inputs:
        results.append(
            _parse_one(
                target,
                use_playwright_for_links=not bool(args.no_playwright_links),
                playwright_timeout_ms=int(args.playwright_timeout_ms),
                use_basic_chunking=bool(args.basic_chunk),
                max_characters=int(args.max_characters),
                new_after_n_chars=int(args.new_after_n_chars),
                combine_text_under_n_chars=int(args.combine_text_under_n_chars),
                overlap=int(args.overlap),
                overlap_all=bool(args.overlap_all),
                apply_cleanup=not bool(args.no_cleanup),
                min_text_chars=int(args.min_text_chars),
                max_url_ratio=float(args.max_url_ratio),
            )
        )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    for item in results:
        print("=" * 80)
        print(f"source: {item.get('source')}")
        print(f"ok: {item.get('ok')}")
        if not item.get("ok"):
            print(f"error: {item.get('error')}")
            continue
        print(f"method: {item.get('extraction_method')}")
        print(f"elements: {item.get('element_count')}")
        if item.get("element_count_raw") is not None:
            print(f"elements_raw: {item.get('element_count_raw')}")
        print(f"non_empty_text: {item.get('non_empty_text_count')}")
        if item.get("playwright_warning"):
            print(f"playwright_warning: {item.get('playwright_warning')}")
        cleanup = dict(item.get("cleanup") or {})
        if cleanup:
            print(
                "cleanup: "
                f"enabled={cleanup.get('enabled')} "
                f"kept={cleanup.get('kept_element_count')}/{cleanup.get('raw_element_count')} "
                f"dropped_type={cleanup.get('dropped_type')} "
                f"dropped_noise={cleanup.get('dropped_noise')} "
                f"dropped_dupe={cleanup.get('dropped_duplicate')}"
            )

        chunking = dict(item.get("chunking") or {})
        if chunking:
            print(f"chunking_strategy: {chunking.get('strategy')}")
            if chunking.get("error"):
                print(f"chunking_error: {chunking.get('error')}")
            else:
                print(f"chunk_count: {chunking.get('chunk_count')}")
                print(f"supported_chunk_kwargs: {chunking.get('supported_chunk_kwargs')}")
                chunk_preview = list(chunking.get("chunk_preview") or [])
                if chunk_preview:
                    print("chunk_preview:")
                    for i, line in enumerate(chunk_preview, start=1):
                        print(f"  {i}. {line[:180]}")

        preview = list(item.get("preview") or [])
        if preview:
            print("preview:")
            for i, line in enumerate(preview, start=1):
                print(f"  {i}. {line[:180]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
