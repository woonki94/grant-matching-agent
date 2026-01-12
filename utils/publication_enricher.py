from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import requests

from config import settings
from dto.faculty_dto import FacultyPublicationDTO
from mappers.openalex_to_publication import map_openalex_works_to_publication_dtos

OPENALEX_BASE = settings.openalex_base_url


def openalex_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resp = requests.get(f"{OPENALEX_BASE}{path}", params=params or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_institution_id(university: str) -> Optional[str]:
    data = openalex_get(
        "/institutions",
        {"filter": f"display_name.search:{university}", "per-page": 1},
    )
    results = data.get("results") or []
    return results[0].get("id") if results else None


def resolve_author_id(full_name: str, institution_id: str, max_candidates: int = 10) -> Optional[str]:
    data = openalex_get("/authors", {"search": full_name, "per-page": max_candidates})
    candidates = data.get("results") or []
    if not candidates:
        return None

    def matches_inst(a: Dict[str, Any]) -> bool:
        for li in (a.get("last_known_institutions") or []):
            if li.get("id") == institution_id:
                return True
        for aff in (a.get("affiliations") or []):
            inst = (aff.get("institution") or {})
            if inst.get("id") == institution_id:
                return True
        return False

    best = next((a for a in candidates if matches_inst(a)), candidates[0])
    return best.get("id")


def fetch_author_works(
    author_id: str,
    *,
    years_back: int = 5,
    per_page: int = 50,
    max_pages: int = 1,
    sort: str = "publication_year:desc",
) -> List[Dict[str, Any]]:
    current_year = datetime.now().year
    min_year = current_year - (years_back - 1) if years_back else None

    filters = [f"authorships.author.id:{author_id}"]
    if min_year is not None:
        filters.append(f"publication_year:{min_year}-{current_year}")

    out: List[Dict[str, Any]] = []
    cursor = "*"

    for _ in range(max_pages):
        data = openalex_get(
            "/works",
            {
                "filter": ",".join(filters),
                "per-page": per_page,
                "cursor": cursor,
                "sort": sort,
            },
        )
        out.extend(data.get("results") or [])

        cursor = (data.get("meta") or {}).get("next_cursor")
        if not cursor:
            break

    return out

def get_publication_dtos_for_faculty(
    full_name: str,
    university: str,
    years_back: int = 5,
    per_page: int = 50,
    max_pages: int = 1,
) -> Tuple[Optional[str], List[FacultyPublicationDTO]]:
    inst_id = get_institution_id(university)
    if not inst_id:
        return None, []

    author_id = resolve_author_id(full_name, inst_id)
    if not author_id:
        return None, []

    works = fetch_author_works(author_id, years_back=years_back, per_page=per_page, max_pages=max_pages)
    dtos = map_openalex_works_to_publication_dtos(works, openalex_author_id=author_id)

    return author_id, dtos