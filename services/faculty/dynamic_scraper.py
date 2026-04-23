from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment
from langchain_core.messages import HumanMessage

from config import get_llm_client

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Mozilla/5.0 (+faculty-dynamic-scraper; AI pipeline)"
MAX_HTML_CHARS = 10_000

PROMPT_TEMPLATE = """You are a Python engineer who must generate a BeautifulSoup-based faculty scraper.

The website is:
  base_url: {base_url}
  list_path: {list_path}

Your generated code must define a function build_scraper() that returns an object with two methods:
  1) extract_profile_links(list_html: str, base_url: str) -> List[str]
  2) parse_profile(profile_html: str, profile_url: str) -> Dict[str, Any]


The parser must return a dictionary with these keys:
  name, position, email, organization, expertise, biography, additional_links, source_url
  AND any other fields you think are relevant based on the sample profiles.

Rules:
  - Use BeautifulSoup and urljoin only.
  - Return None for missing scalar fields.
  - Return [] for missing list fields.
  - Generate VALID Python code only. No syntax errors, incomplete lines, or extra text.
  - Do not include code comments, markdown fences, or conversational text.
  - Do not call external services or use dangerous imports.

Sample profile pages:
{sample_text}
"""

DANGEROUS_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "from os",
    "from sys",
    "from subprocess",
    "eval(",
    "exec(",
    "__import__",
    "open(",
    "socket",
    "Popen",
    "subprocess",
    "pickle",
    "shutil",
]


def _truncate_html(html: str, max_chars: int = MAX_HTML_CHARS) -> str:
    html = str(html or "")
    if len(html) <= max_chars:
        return html
    return html[:max_chars] + "\n<!-- TRUNCATED -->"


def _extract_main_body_html(html: str) -> str:
    soup = BeautifulSoup(str(html or ""), "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    body = soup.body or soup
    for block in body.find_all(["header", "footer", "nav", "aside"]):
        block.decompose()
    main_tag = body.find("main")
    if main_tag:
        return str(main_tag)
    return str(body)


def _prepare_html_for_prompt(html: str, max_chars: int = MAX_HTML_CHARS) -> str:
    cleaned = _extract_main_body_html(html)
    return _truncate_html(cleaned, max_chars)


def _clean_llm_code(raw: str) -> str:
    code = str(raw or "").strip()
    if code.startswith("```"):
        parts = code.split("```", 2)
        if len(parts) > 1:
            code = parts[1].lstrip("python\n").strip()
    return code


def _validate_generated_code(code: str) -> None:
    normalized = code.lower()
    print(code)  # for debugging
    for pattern in DANGEROUS_PATTERNS:
        if pattern in normalized:
            raise ValueError(f"Generated code contains a forbidden pattern: {pattern}")


def _extract_candidate_profile_urls(base_url: str, html: str, max_links: int = 20) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    body = soup.body or soup
    for block in body.find_all(["header", "footer", "nav", "aside"]):
        block.decompose()

    urls: List[str] = []
    allowed_domain = urlparse(base_url).netloc.lower()

    for anchor in body.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.startswith("#") or href.lower().startswith("mailto:"):
            continue
        link = urljoin(base_url, href)
        if urlparse(link).netloc.lower() != allowed_domain:
            continue
        if link not in urls:
            urls.append(link)
        if len(urls) >= max_links:
            break
    return urls


def _build_prompt(
    base_url: str,
    list_path: str,
    list_html: str,
    profile_samples: List[Tuple[str, str]],
) -> str:
    sample_parts = []
    for idx, (url, html) in enumerate(profile_samples, start=1):
        sample_parts.append(f"SAMPLE PROFILE #{idx} URL: {url}\n{html}")
    sample_text = "\n\n".join(sample_parts)
    return PROMPT_TEMPLATE.format(
        base_url=base_url,
        list_path=list_path,
        list_html=_prepare_html_for_prompt(list_html, MAX_HTML_CHARS),
        sample_text=sample_text,
    )


def _invoke_llm(prompt: str, model_id: Optional[str]) -> str:
    llm = get_llm_client(model_id=model_id or None).build()
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content if hasattr(response, "content") else str(response)
    code = _clean_llm_code(raw)
    if not code:
        raise RuntimeError("LLM returned no code.")
    _validate_generated_code(code)
    return code


def _load_generated_scraper(code: str) -> Any:
    wrapper = (
        "from bs4 import BeautifulSoup\n"
        "from urllib.parse import urljoin\n"
        "import requests\n"
        "import re\n\n"
        + code
    )
    namespace: Dict[str, Any] = {}
    try:
        exec(wrapper, namespace)
    except SyntaxError as e:
        raise RuntimeError(f"Generated scraper code has a syntax error: {e}. Generated code:\n{code}") from e
    if "build_scraper" not in namespace:
        raise RuntimeError("Generated code must define build_scraper().")
    scraper = namespace["build_scraper"]()
    if not hasattr(scraper, "extract_profile_links") or not hasattr(scraper, "parse_profile"):
        raise RuntimeError(
            "build_scraper() must return an object with extract_profile_links and parse_profile methods."
        )
    return scraper


def fetch_html(url: str, timeout: int = 20) -> str:
    resp = requests.get(url, headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


class DynamicFacultyScraperPipeline:
    def __init__(
        self,
        *,
        llm_model_id: Optional[str] = None,
        sample_profile_count: int = 2,
        timeout: int = 20,
    ):
        self.llm_model_id = llm_model_id
        self.sample_profile_count = max(1, int(sample_profile_count))
        self.timeout = int(timeout)
        self.last_generated_code: Optional[str] = None

    def generate_scraper(
        self,
        base_url: str,
        list_path: str,
        sample_profile_urls: Optional[List[str]] = None,
    ) -> Any:
        list_url = urljoin(base_url, list_path)
        list_html = fetch_html(list_url, timeout=self.timeout)
        candidates = sample_profile_urls or _extract_candidate_profile_urls(base_url, list_html, max_links=25)
        if not candidates:
            raise RuntimeError("Could not infer any candidate profile URLs from the list page.")

        profile_samples: List[Tuple[str, str]] = []
        for profile_url in candidates[: self.sample_profile_count]:
            try:
                html = fetch_html(profile_url, timeout=self.timeout)
            except Exception as exc:
                logger.warning("Failed to fetch sample profile %s: %s", profile_url, exc)
                continue
            profile_samples.append((profile_url, _prepare_html_for_prompt(html, MAX_HTML_CHARS)))

        if not profile_samples:
            raise RuntimeError("Could not fetch any sample profile pages to infer scraping logic.")
        
        # print(f"LIST PAGE HTML: {list_html}\n\n\n")  # for debugging
        # print(f"CANDIDATE PROFILE URLS: {candidates}\n\n\n")  # for debugging
        # print(f"PROFILE SAMPLES: {profile_samples}\n\n\n")  # for debugging
        #return

        prompt = _build_prompt(base_url, list_path, list_html, profile_samples)
        code = _invoke_llm(prompt, model_id=self.llm_model_id)
        self.last_generated_code = code
        return _load_generated_scraper(code)

    def run(
        self,
        base_url: str,
        list_path: str,
        profile_limit: int = 10,
        sample_profile_urls: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        scraper = self.generate_scraper(
            base_url=base_url,
            list_path=list_path,
            sample_profile_urls=sample_profile_urls,
        )
        list_url = urljoin(base_url, list_path)
        list_html = fetch_html(list_url, timeout=self.timeout)
        profile_urls = scraper.extract_profile_links(list_html, base_url)
        if not profile_urls:
            raise RuntimeError("Generated scraper did not return any profile URLs.")

        results: List[Dict[str, Any]] = []
        for profile_url in profile_urls[:profile_limit]:
            try:
                profile_html = fetch_html(profile_url, timeout=self.timeout)
                parsed = scraper.parse_profile(profile_html, profile_url)
                if isinstance(parsed, dict):
                    results.append(parsed)
            except Exception as exc:
                logger.warning("Failed to parse profile %s: %s", profile_url, exc)
        return results
