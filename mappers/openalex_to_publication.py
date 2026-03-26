from __future__ import annotations

from typing import Any, Dict, List, Optional

from dto.faculty_dto import FacultyPublicationDTO


def _reconstruct_abstract(abstract_inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    if not abstract_inverted_index:
        return None
    pos_to_word: Dict[int, str] = {}
    for word, positions in abstract_inverted_index.items():
        for p in positions:
            pos_to_word[p] = word
    if not pos_to_word:
        return None
    return " ".join(pos_to_word[i] for i in range(max(pos_to_word) + 1) if i in pos_to_word)


def map_openalex_works_to_publication_dtos(
    works: List[Dict[str, Any]],
    *,
    openalex_author_id: str,
) -> List[FacultyPublicationDTO]:
    dtos: List[FacultyPublicationDTO] = []

    for w in works:
        title = (w.get("title") or "").strip()
        if not title:
            continue

        dtos.append(
            FacultyPublicationDTO.model_validate(
                {
                    "openalex_work_id": w.get("id"),
                    "scholar_author_id": openalex_author_id,  # (your DTO name)
                    "title": title,
                    "year": w.get("publication_year"),
                    "abstract": _reconstruct_abstract(w.get("abstract_inverted_index")),
                }
            )
        )

    return dtos