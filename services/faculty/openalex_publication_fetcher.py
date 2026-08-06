from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional

import requests

from dto.faculty_dto import FacultyPublicationDTO

logger = logging.getLogger(__name__)

_OPENALEX_BASE = "https://api.openalex.org"
_DEFAULT_TIMEOUT = 20
_USER_AGENT = "GrantFetcher/1.0 (faculty-publication-sync)"


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _clean_text(a).lower(), _clean_text(b).lower()).ratio()


def _extract_openalex_id(raw: str) -> Optional[str]:
    txt = _clean_text(raw)
    if not txt:
        return None
    # Accept either full URL or bare token.
    # Examples:
    # - https://openalex.org/A123456789
    # - A123456789
    parts = txt.rstrip("/").split("/")
    token = parts[-1] if parts else txt
    token = token.strip()
    if re.fullmatch(r"[AW]\d+", token):
        return token
    return None


def _abstract_from_inverted_index(inv: Optional[Dict[str, List[int]]]) -> Optional[str]:
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


class OpenAlexPublicationFetcher:
    def __init__(self, *, timeout: int = _DEFAULT_TIMEOUT):
        self.timeout = int(timeout)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    def resolve_author_id(
        self,
        *,
        faculty_name: str,
        org_hint: Optional[str] = None,
    ) -> Optional[str]:
        name = _clean_text(faculty_name)
        if not name:
            return None

        try:
            resp = self._session.get(
                f"{_OPENALEX_BASE}/authors",
                params={"search": name, "per-page": 25},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            items = (resp.json() or {}).get("results") or []
        except Exception as e:
            logger.warning("OpenAlex author search failed: %s", e)
            return None

        org_l = _clean_text(org_hint).lower()
        best_score = 0.0
        best_id = None
        for it in items:
            display = _clean_text(it.get("display_name"))
            if not display:
                continue
            score = _sim(name, display)

            last_inst = ((it.get("last_known_institution") or {}).get("display_name") or "")
            inst_l = _clean_text(last_inst).lower()
            if org_l and org_l in inst_l:
                score += 0.12
            if "oregon state" in inst_l:
                score += 0.08

            works_count = it.get("works_count")
            if isinstance(works_count, int) and works_count > 0:
                score += 0.02

            cand_id = _extract_openalex_id(it.get("id") or "")
            if score > best_score and cand_id:
                best_score = score
                best_id = cand_id

        # Conservative threshold to avoid obvious wrong matches.
        if best_score < 0.70:
            return None
        return best_id

    def fetch_publications_for_author_year_range(
        self,
        *,
        author_id: str,
        year_from: int,
        year_to: int,
    ) -> List[FacultyPublicationDTO]:
        aid = _extract_openalex_id(author_id or "")
        if not aid:
            return []

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
                    f"{_OPENALEX_BASE}/works",
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
                work_id = _extract_openalex_id(w.get("id") or "")
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
                abstract = _abstract_from_inverted_index(w.get("abstract_inverted_index"))
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

