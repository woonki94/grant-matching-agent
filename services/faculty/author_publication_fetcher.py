from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import time
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote

import requests

from config import settings
from dto.faculty_dto import FacultyPublicationDTO

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 20
_USER_AGENT = "GrantFetcher/1.0 (faculty-publication-sync)"

def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def _norm(s: str) -> str:
    return _clean_text(s).lower()

class AuthorPublicationFetcher:
    def __init__(self, *, timeout: int = _DEFAULT_TIMEOUT):
        self.timeout = int(timeout)
        self._openalex_base = _clean_text(settings.openalex_base_url).rstrip("/")
        self._semantic_scholar_paper_search_api = _clean_text(
            settings.semantic_scholar_paper_search_api
        ).rstrip("/")
        self._semantic_scholar_paper_api = self._semantic_scholar_paper_search_api.replace(
            "/search",
            "",
        )
        self._default_required_org = _clean_text(settings.university_name)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})
        self._s2_abstract_cache: Dict[str, Optional[str]] = {}
        self._s2_last_request_at: float = 0.0
        self._s2_rate_limited_until: float = 0.0
        self._s2_consecutive_429: int = 0

    @staticmethod
    def normalize_doi(raw: str) -> str:
        txt = _clean_text(raw)
        if not txt:
            return ""
        txt = re.sub(r"^doi:\s*", "", txt, flags=re.IGNORECASE).strip()
        txt = re.sub(r"^https?://(dx\.)?doi\.org/", "", txt, flags=re.IGNORECASE).strip()
        return txt

    def _extract_openalex_id(self, raw: str) -> Optional[str]:
        txt = _clean_text(raw)
        if not txt:
            return None
        parts = txt.rstrip("/").split("/")
        token = parts[-1] if parts else txt
        token = token.strip()
        if re.fullmatch(r"[AW]\d+", token):
            return token
        return None

    def _split_author_ids(self, raw: str) -> List[str]:
        txt = _clean_text(raw)
        if not txt:
            return []
        out: List[str] = []
        for chunk in re.split(r"[,\s]+", txt):
            aid = self._extract_openalex_id(chunk)
            if aid and aid not in out:
                out.append(aid)
        return out

    def _extract_last_known_institutions(self, author_row: dict) -> List[str]:
        out: List[str] = []

        def _add(name: str) -> None:
            txt = _clean_text(name)
            if txt and txt not in out:
                out.append(txt)

        for inst in (author_row.get("last_known_institutions") or []):
            if isinstance(inst, dict):
                _add(inst.get("display_name") or "")

        lki = author_row.get("last_known_institution") or {}
        if isinstance(lki, dict):
            _add(lki.get("display_name") or "")

        return out

    def _is_exact_author_match(
        self,
        *,
        author_row: dict,
        input_name: str,
        input_institution: str,
    ) -> bool:
        display_name = _clean_text(author_row.get("display_name") or "")
        if _norm(display_name) != _norm(input_name):
            return False

        required_inst = _norm(input_institution)
        if not required_inst:
            return False
        last_insts = self._extract_last_known_institutions(author_row)
        if not last_insts:
            return False
        return any(_norm(inst) == required_inst for inst in last_insts)

    def _abstract_from_inverted_index(self, inv: Optional[Dict[str, List[int]]]) -> Optional[str]:
        if not isinstance(inv, dict) or not inv:
            return None
        try:
            max_pos = -1
            for positions in inv.values():
                if isinstance(positions, list) and positions:
                    max_pos = max(max_pos, max(int(x) for x in positions))
            if max_pos < 0:
                return None
            words = [""] * (max_pos + 1)
            for word, positions in inv.items():
                if not isinstance(positions, list):
                    continue
                for p in positions:
                    idx = int(p)
                    if 0 <= idx < len(words):
                        words[idx] = str(word)
            text = " ".join(w for w in words if w).strip()
            return text or None
        except Exception:
            return None

    def _fetch_author_profile(self, author_id: str) -> Optional[dict]:
        aid = self._extract_openalex_id(author_id or "")
        if not aid:
            return None
        try:
            resp = self._session.get(
                f"{self._openalex_base}/authors/{aid}",
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json() or {}
        except Exception as e:
            logger.warning("OpenAlex author profile fetch failed for %s: %s", aid, e)
            return None

    def _get_s2_api_key(self) -> str:
        env_key = _clean_text(os.getenv("S2_API_KEY", ""))
        if env_key:
            return env_key

        env_path = Path(__file__).resolve().parents[2] / ".env"
        if not env_path.exists():
            return ""
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                raw = line.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                k, v = raw.split("=", 1)
                key = _clean_text(k)
                if key.lower().startswith("export "):
                    key = _clean_text(key[7:])
                if key == "S2_API_KEY":
                    return _clean_text(v).strip("\"'")
        except Exception:
            return ""
        return ""

    def _parse_retry_after(self, raw: str) -> Optional[float]:
        txt = _clean_text(raw)
        if not txt:
            return None
        try:
            return max(0.0, float(txt))
        except Exception:
            return None

    def _request_with_backoff(
        self,
        *,
        url: str,
        params: Dict[str, str],
        headers: Dict[str, str],
        max_retries: int,
        label: str,
    ) -> Optional[requests.Response]:
        retries = max(0, int(max_retries))
        for attempt in range(retries + 1):
            try:
                resp = self._session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
            except Exception as e:
                if attempt >= retries:
                    logger.warning("%s request failed: %s", label, e)
                    return None
                time.sleep(min(2 ** attempt, 5))
                continue

            if resp.status_code == 429:
                wait = self._parse_retry_after(resp.headers.get("Retry-After", "")) or min(2 ** attempt, 8)
                if attempt < retries:
                    time.sleep(max(0.25, wait))
                    continue

                # Final-rate-limit handling: apply cooldown to avoid repeated 429 spam.
                if label == "Semantic Scholar":
                    self._s2_consecutive_429 += 1
                    extra = min(60.0, 5.0 * float(self._s2_consecutive_429))
                    cooldown = max(1.0, float(wait)) + extra
                    self._s2_rate_limited_until = max(
                        self._s2_rate_limited_until,
                        time.monotonic() + cooldown,
                    )
                    logger.warning(
                        "%s rate-limited (429). Pausing S2 fallback for %.1fs",
                        label,
                        cooldown,
                    )
                else:
                    logger.warning("%s request rate-limited (429).", label)
                return None

            try:
                resp.raise_for_status()
                if label == "Semantic Scholar":
                    self._s2_consecutive_429 = 0
                return resp
            except Exception as e:
                logger.warning("%s request failed: %s", label, e)
                return None
        return None

    def fetch_semantic_scholar_abstract_by_title(self, *, title: str) -> Optional[str]:
        """
        Fetch abstract from Semantic Scholar using title query.
        Fallback order:
        1) Semantic Scholar abstract
        2) Semantic Scholar tldr.text
        """
        query = _clean_text(title)
        if not query:
            return None
        key = _norm(query)
        if key in self._s2_abstract_cache:
            return self._s2_abstract_cache[key]

        # If we were recently rate-limited by S2, skip calls until cooldown expires.
        now = time.monotonic()
        if now < self._s2_rate_limited_until:
            self._s2_abstract_cache[key] = None
            return None

        headers = {"User-Agent": _USER_AGENT}
        api_key = self._get_s2_api_key()
        if api_key:
            headers["x-api-key"] = api_key

        # Lightweight pacing to reduce 429 risk.
        min_interval = 0.20 if api_key else 1.00
        wait_for_slot = (self._s2_last_request_at + min_interval) - now
        if wait_for_slot > 0:
            time.sleep(wait_for_slot)
        self._s2_last_request_at = time.monotonic()

        resp = self._request_with_backoff(
            url=self._semantic_scholar_paper_search_api,
            params={
                "query": query,
                "fields": "title,abstract,tldr",
                "limit": "5",
            },
            headers=headers,
            max_retries=2,
            label="Semantic Scholar",
        )
        if resp is None:
            self._s2_abstract_cache[key] = None
            return None

        try:
            rows = (resp.json() or {}).get("data") or []
        except Exception:
            rows = []
        if not isinstance(rows, list):
            rows = []

        query_n = _norm(query)

        # Prefer exact-title hits first.
        exact_rows = []
        non_exact_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            r_title = _clean_text(row.get("title") or "")
            if _norm(r_title) == query_n:
                exact_rows.append(row)
            else:
                non_exact_rows.append(row)

        ordered_rows = exact_rows + non_exact_rows

        # 1) Try full abstract first.
        for row in ordered_rows:
            abs_txt = _clean_text(row.get("abstract") or "")
            if abs_txt:
                self._s2_abstract_cache[key] = abs_txt
                return abs_txt

        # 2) Fallback to TLDR.
        for row in ordered_rows:
            tldr = row.get("tldr") or {}
            if not isinstance(tldr, dict):
                continue
            tldr_txt = _clean_text(tldr.get("text") or "")
            if tldr_txt:
                self._s2_abstract_cache[key] = tldr_txt
                return tldr_txt

        self._s2_abstract_cache[key] = None
        return None

    def fetch_semantic_scholar_paper_by_doi(self, *, doi: str) -> Optional[dict]:
        """
        Fetch one paper from Semantic Scholar using DOI.

        Returns a dict with optional keys: title, abstract, year.
        """
        normalized_doi = self.normalize_doi(doi or "")
        if not normalized_doi:
            return None

        headers = {"User-Agent": _USER_AGENT}
        api_key = self._get_s2_api_key()
        if api_key:
            headers["x-api-key"] = api_key

        paper_id = quote(f"DOI:{normalized_doi}", safe="")
        url = f"{self._semantic_scholar_paper_api}/{paper_id}"
        resp = self._request_with_backoff(
            url=url,
            params={"fields": "title,abstract,year,tldr"},
            headers=headers,
            max_retries=2,
            label="Semantic Scholar",
        )
        if resp is None:
            return None

        try:
            payload = resp.json() or {}
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None

        title = _clean_text(payload.get("title") or "")
        abstract = _clean_text(payload.get("abstract") or "")
        if not abstract:
            tldr = payload.get("tldr") or {}
            if isinstance(tldr, dict):
                abstract = _clean_text(tldr.get("text") or "")

        year_raw = payload.get("year")
        try:
            year = int(year_raw) if year_raw is not None else None
        except Exception:
            year = None

        return {
            "title": title or None,
            "abstract": abstract or None,
            "year": year,
        }

    def resolve_author_id(
        self,
        *,
        faculty_name: str,
        org_hint: Optional[str] = None,
    ) -> Optional[str]:
        ids = self.resolve_author_ids(
            faculty_name=faculty_name,
            org_hint=org_hint,
        )
        if not ids:
            return None
        return ids[0]

    def resolve_author_ids(
        self,
        *,
        faculty_name: str,
        org_hint: Optional[str] = None,
    ) -> List[str]:
        input_name = _clean_text(faculty_name)
        if not input_name:
            return []

        required_org = _clean_text(org_hint) or self._default_required_org
        if not required_org:
            return []

        try:
            items: List[dict] = []
            seen_ids = set()
            cursor = "*"
            for _ in range(5):
                resp = self._session.get(
                    f"{self._openalex_base}/authors",
                    params={
                        "search": input_name,
                        "per-page": 200,
                        "cursor": cursor,
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                payload = resp.json() or {}
                rows = payload.get("results") or []
                for row in rows:
                    aid = self._extract_openalex_id(row.get("id") or "")
                    if not aid or aid in seen_ids:
                        continue
                    seen_ids.add(aid)
                    items.append(row)
                next_cursor = ((payload.get("meta") or {}).get("next_cursor"))
                if not next_cursor or next_cursor == cursor:
                    break
                cursor = str(next_cursor)
        except Exception as e:
            logger.warning("OpenAlex author search failed: %s", e)
            return []

        matched_candidates: List[tuple] = []
        for idx, it in enumerate(items):
            aid = self._extract_openalex_id(it.get("id") or "")
            if not aid:
                continue
            profile_row = self._fetch_author_profile(aid) or it
            if self._is_exact_author_match(
                author_row=profile_row,
                input_name=input_name,
                input_institution=required_org,
            ):
                wc_raw = profile_row.get("works_count")
                if wc_raw is None:
                    wc_raw = it.get("works_count")
                works_count = int(wc_raw) if isinstance(wc_raw, int) else 0
                matched_candidates.append((works_count, idx, aid))

        if not matched_candidates:
            return []
        # Choose one author: highest works_count; tie -> earlier search order.
        matched_candidates.sort(key=lambda x: (-x[0], x[1]))
        return [matched_candidates[0][2]]

    def fetch_publications_for_author_ids_year_range(
        self,
        *,
        author_ids: Iterable[str],
        year_from: int,
        year_to: int,
    ) -> List[FacultyPublicationDTO]:
        results: List[FacultyPublicationDTO] = []
        seen_work_ids = set()
        for raw_author_id in author_ids or []:
            aid = self._extract_openalex_id(raw_author_id or "")
            if not aid:
                continue
            rows = self.fetch_publications_for_author_year_range(
                author_id=aid,
                year_from=year_from,
                year_to=year_to,
            )
            for row in rows:
                wid = self._extract_openalex_id(row.openalex_work_id or "")
                if wid and wid in seen_work_ids:
                    continue
                if wid:
                    seen_work_ids.add(wid)
                results.append(row)
        results.sort(key=lambda r: ((r.year or 0), _norm(r.title)), reverse=True)
        return results

    def fetch_publications_for_name_year_range(
        self,
        *,
        faculty_name: str,
        year_from: int,
        year_to: int,
        org_hint: Optional[str] = None,
    ) -> List[FacultyPublicationDTO]:
        author_ids = self.resolve_author_ids(
            faculty_name=faculty_name,
            org_hint=org_hint,
        )
        if not author_ids:
            return []
        return self.fetch_publications_for_author_ids_year_range(
            author_ids=author_ids,
            year_from=year_from,
            year_to=year_to,
        )

    def fetch_publications_for_author_year_range(
        self,
        *,
        author_id: str,
        year_from: int,
        year_to: int,
    ) -> List[FacultyPublicationDTO]:
        author_ids = self._split_author_ids(author_id or "")
        if not author_ids:
            return []
        if len(author_ids) > 1:
            return self.fetch_publications_for_author_ids_year_range(
                author_ids=author_ids,
                year_from=year_from,
                year_to=year_to,
            )
        aid = author_ids[0]

        from_year = int(year_from)
        to_year = int(year_to)
        if from_year > to_year:
            return []

        results: List[FacultyPublicationDTO] = []
        cursor = "*"
        seen_work_ids = set()

        # Cursor paging; cap to avoid runaway requests.
        for _ in range(25):
            try:
                resp = self._session.get(
                    f"{self._openalex_base}/works",
                    params={
                        "filter": (
                            f"authorships.author.id:https://openalex.org/{aid},"
                            f"from_publication_date:{from_year}-01-01,"
                            f"to_publication_date:{to_year}-12-31"
                        ),
                        "per-page": 200,
                        "cursor": cursor,
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                payload = resp.json() or {}
            except Exception as e:
                logger.warning("OpenAlex works fetch failed: %s", e)
                break

            items = payload.get("results") or []
            for w in items:
                work_id = self._extract_openalex_id(w.get("id") or "")
                if not work_id or work_id in seen_work_ids:
                    continue
                seen_work_ids.add(work_id)

                title = _clean_text(w.get("display_name") or w.get("title") or "")
                if not title:
                    continue
                year = w.get("publication_year")
                try:
                    year_i = int(year) if year is not None else None
                except Exception:
                    year_i = None
                abstract = self._abstract_from_inverted_index(w.get("abstract_inverted_index"))
                if not abstract:
                    abstract = self.fetch_semantic_scholar_abstract_by_title(title=title)
                results.append(
                    FacultyPublicationDTO(
                        openalex_work_id=work_id,
                        scholar_author_id=aid,
                        title=title,
                        abstract=abstract,
                        year=year_i,
                    )
                )

            meta = payload.get("meta") or {}
            next_cursor = meta.get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = str(next_cursor)

        return results
