# scripts/save_faculty_profile.py
from __future__ import annotations
from typing import Dict, Any, List
from sqlalchemy.orm import Session, sessionmaker

from db.db_conn import engine
from db.dao.faculty import (
    FacultyDAO, FacultyPublicationDAO
)
from db.models.faculty import Faculty
from services.faculty.call_publication_faculty import fetch_faculty_publications_from_openalex

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def _as_faculty_row(d: Dict[str, Any]) -> Dict[str, Any]:
    rw = d.get("research_website") or {}
    return {
        "source_url": d.get("source_url"),
        "name": d.get("name"),
        "email": d.get("email"),
        "phone": d.get("phone"),
        "position": d.get("position"),
        "organization": d.get("organization"),
        "organizations": d.get("organizations"),  # JSON list
        "address": d.get("address"),              # keep newlines
        "biography": d.get("biography"),
        "research_website_name": rw.get("name"),
        "research_website_url": rw.get("url"),
    }

def _as_degrees(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"order_index": i, "degree_text": t}
            for i, t in enumerate(d.get("degrees") or [])]

def _as_expertise(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"term": t} for t in (d.get("research_expertise") or []) if t]

def _as_groups(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for g in d.get("research_groups") or []:
        if not g: continue
        out.append({"name": g.get("name"), "url": g.get("url")})
    return out

def _as_links(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for lk in d.get("additional_links") or []:
        if not lk: continue
        if lk.get("name") and lk.get("url"):
            out.append({"name": lk["name"], "url": lk["url"]})
    return out

def save_profile_dict(profile: Dict[str, Any]) -> int:
    """Upsert one faculty profile (parent + children). Returns faculty_id."""
    if not profile.get("source_url"):
        raise ValueError("profile must include source_url")

    fac_row = _as_faculty_row(profile)
    degrees = _as_degrees(profile)
    expertise = _as_expertise(profile)
    groups = _as_groups(profile)
    links = _as_links(profile)

    with SessionLocal() as session:  # type: Session
        faculty_id = FacultyDAO.upsert_one_bundle(
            session,
            faculty_row=fac_row,
            degrees=degrees,
            expertise=expertise,
            groups=groups,
            links=links,
            delete_then_insert_children=True,   # replaces children deterministically
        )
        session.commit()
        return faculty_id

def enrich_faculty_publications(faculty_id: int) -> int:
    """
    Fetch OpenAlex publications for a faculty and upsert into faculty_publication.
    Returns number of publications upserted.
    """
    with SessionLocal() as session:
        faculty = session.query(Faculty).filter(Faculty.id == faculty_id).first()
        if not faculty:
            raise ValueError(f"Faculty id {faculty_id} not found")

        # if all OSU:
        university = "Oregon State University"

        author_id, dtos = fetch_faculty_publications_from_openalex(
            faculty_id=faculty.id,
            full_name=faculty.name,
            university=university,
            per_page=10,
            max_pages=1,
        )

        if not author_id or not dtos:
            return 0

        rows = [
            {
                "faculty_id": dto.faculty_id,
                "title": dto.title,
                "year": dto.year,
                "abstract": dto.abstract,
                "scholar_author_id": dto.scholar_author_id,
            }
            for dto in dtos
        ]

        FacultyPublicationDAO.replace_for_faculty(session, faculty.id, rows)
        session.commit()
        return len(rows)