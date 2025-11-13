import requests
from typing import Optional, Dict, Any, List
from dto.faculty_publication_dto import FacultyPublicationPersistenceDTO

OPENALEX_BASE = "https://api.openalex.org"


def _get(url: str, params: dict | None = None) -> dict:
    params = params or {}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# 1) Get institution id for "Oregon State University"
def get_institution_id(university_name: str) -> Optional[str]:
    """
    Returns OpenAlex institution id like 'https://openalex.org/I123456789'
    """
    url = f"{OPENALEX_BASE}/institutions"
    # filter by display_name.search (recommended pattern)
    params = {"filter": f"display_name.search:{university_name}", "per-page": 1}
    data = _get(url, params)

    results = data.get("results") or []
    if not results:
        return None
    return results[0].get("id")


# 2) Find author by name + institution id
def find_author_by_name_and_institution(
    full_name: str,
    institution_id: str,
    max_candidates: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Returns the best-matching OpenAlex author object (dict) or None.
    """
    url = f"{OPENALEX_BASE}/authors"
    # step 1: search by name
    params = {
        "search": full_name,
        "per-page": max_candidates,
    }
    data = _get(url, params)
    candidates = data.get("results") or []
    if not candidates:
        return None

    # Filter candidates whose affiliations or last_known_institutions include this institution
    for author in candidates:
        # last_known_institutions is usually the easiest to check
        last_known = author.get("last_known_institutions") or []
        for inst in last_known:
            if inst.get("id") == institution_id:
                return author

        # fallback: check affiliations list
        affiliations = author.get("affiliations") or []
        for aff in affiliations:
            inst = (aff.get("institution") or {})
            if inst.get("id") == institution_id:
                return author

    # fallback: just return top search hit if nothing matched institution
    return candidates[0]


# 3) Turn OpenAlex abstract_inverted_index into plain text
def reconstruct_abstract(
    abstract_inverted_index: Optional[Dict[str, List[int]]]
) -> Optional[str]:
    if not abstract_inverted_index:
        return None

    pos_to_word: Dict[int, str] = {}
    for word, positions in abstract_inverted_index.items():
        for p in positions:
            pos_to_word[p] = word

    if not pos_to_word:
        return None

    max_pos = max(pos_to_word.keys())
    words = [pos_to_word.get(i, "") for i in range(max_pos + 1)]
    words = [w for w in words if w]
    return " ".join(words)


# 4) Get works for author id, including reconstructed abstracts
def get_author_works_with_abstracts(
    author_id: str,
    per_page: int = 50,
    max_pages: int = 1,
    sort_order: str = "publication_year:desc",
) -> List[Dict[str, Any]]:
    """
    Fetches works for a given OpenAlex author id ordered by date (newest first).
    """
    all_works: List[Dict[str, Any]] = []
    url = f"{OPENALEX_BASE}/works"

    page_cursor = "*"
    pages_fetched = 0

    while pages_fetched < max_pages:
        params = {
            "filter": f"authorships.author.id:{author_id}",
            "per-page": per_page,
            "cursor": page_cursor,
            "sort": sort_order,
        }

        data = _get(url, params)
        results = data.get("results") or []

        for w in results:
            title = w.get("title")
            year = w.get("publication_year")
            abstract_idx = w.get("abstract_inverted_index")
            abstract = reconstruct_abstract(abstract_idx)

            all_works.append(
                {
                    "title": title,
                    "year": year,
                    "abstract": abstract,
                    "id": w.get("id"),
                    "doi": w.get("doi"),
                }
            )

        pages_fetched += 1
        page_cursor = data.get("meta", {}).get("next_cursor")
        if not page_cursor:
            break

    return all_works


def build_publication_dtos_from_openalex_works(
    *,
    faculty_id: int,
    openalex_author_id: str,
    works: List[Dict[str, Any]],
) -> List[FacultyPublicationPersistenceDTO]:
    """
    Convert OpenAlex works (dicts from get_author_works_with_abstracts)
    into persistence DTOs ready for DB insertion.
    """
    dtos: List[FacultyPublicationPersistenceDTO] = []

    for w in works:
        title = w.get("title")
        if not title:
            continue

        openalex_work_id = w.get("id")
        year = w.get("year")
        abstract = w.get("abstract")

        dtos.append(
            FacultyPublicationPersistenceDTO(
                faculty_id=faculty_id,
                scholar_author_id=openalex_author_id,
                title=title,
                year=year,
                abstract=abstract,
                openalex_work_id=openalex_work_id,
            )
        )

    return dtos


def fetch_faculty_publications_from_openalex(
    faculty_id: int,
    full_name: str,
    university: str,
    per_page: int = 50,
    max_pages: int = 1,
) -> tuple[Optional[str], List[FacultyPublicationPersistenceDTO]]:
    """
    End-to-end helper:
      - resolve institution
      - resolve author by name + institution
      - fetch works with abstracts
      - map to FacultyPublicationPersistenceDTOs

    Returns:
        (openalex_author_id, dtos)
        If resolution fails, returns (None, []).
    """
    inst_id = get_institution_id(university)
    if not inst_id:
        return None, []

    author = find_author_by_name_and_institution(full_name, inst_id)
    if not author:
        return None, []

    author_id = author["id"]

    works = get_author_works_with_abstracts(
        author_id=author_id,
        per_page=per_page,
        max_pages=max_pages,
        sort_order="publication_year:desc",
    )

    dtos = build_publication_dtos_from_openalex_works(
        faculty_id=faculty_id,
        openalex_author_id=author_id,
        works=works,
    )

    return author_id, dtos

