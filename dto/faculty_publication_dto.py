# scholar_faculty_dto.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

#dto of persistence
@dataclass
class FacultyPublicationPersistenceDTO:
    faculty_id: int
    scholar_author_id: str
    title: str
    year: Optional[int]
    abstract: Optional[str] = None
    openalex_work_id: Optional[str] = None