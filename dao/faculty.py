from __future__ import annotations
from typing import List, Dict, Any, Iterable
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from sqlalchemy import select, delete

from db.models.faculty import (
    Faculty, FacultyDegree, FacultyExpertise, FacultyResearchGroup, FacultyLink, FacultyPublication
)
# ─────────────────────────────────────────────────────────────
# FACULTY DAO
# ─────────────────────────────────────────────────────────────
class FacultyDAO:
    @staticmethod
    def upsert_many(session: Session, rows: List[dict]) -> None:
        """
        rows: dicts of Faculty columns (use the model column names)
        On conflict: source_url
        """
        if not rows:
            return
        stmt = pg_insert(Faculty).values(rows)
        set_map = {
            c.name: getattr(stmt.excluded, c.name)
            for c in Faculty.__table__.columns
            if c.name not in ("id", "source_url")   # immutable identity + PK
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["source_url"],
            set_=set_map,
        )
        session.execute(stmt)

    @staticmethod
    def id_by_source_url(session: Session, source_url: str) -> int | None:
        q = session.execute(
            select(Faculty.id).where(Faculty.source_url == source_url)
        )
        return q.scalar_one_or_none()

    @staticmethod
    def ids_by_source_urls(session: Session, source_urls: Iterable[str]) -> Dict[str, int]:
        urls = list(set([u for u in source_urls if u]))
        if not urls:
            return {}
        rows = session.execute(
            select(Faculty.source_url, Faculty.id).where(Faculty.source_url.in_(urls))
        ).all()
        return {u: i for (u, i) in rows}

    @staticmethod
    def upsert_one_bundle(
        session: Session,
        *,
        faculty_row: Dict[str, Any],
        degrees: List[Dict[str, Any]] | None = None,
        expertise: List[Dict[str, Any]] | None = None,
        groups: List[Dict[str, Any]] | None = None,
        links: List[Dict[str, Any]] | None = None,
        publications: List[Dict[str, Any]] | None = None,
        delete_then_insert_children: bool = True,
    ) -> int:
        """
        Idempotent upsert of a single Faculty row, then replace child rows.
        faculty_row must include 'source_url'. Returns faculty_id.
        Child dicts must include 'faculty_id' if delete_then_insert_children=False.
        """
        # 1) Upsert parent
        FacultyDAO.upsert_many(session, [faculty_row])
        fac_id = FacultyDAO.id_by_source_url(session, faculty_row["source_url"])
        if fac_id is None:
            raise RuntimeError("Failed to upsert Faculty (no id returned)")

        # 2) Replace children (delete then insert)
        if delete_then_insert_children:
            if degrees is not None:
                session.execute(delete(FacultyDegree).where(FacultyDegree.faculty_id == fac_id))
                if degrees:
                    rows = [
                        {"faculty_id": fac_id, **r}
                        for r in degrees
                    ]
                    FacultyDegreeDAO.bulk_insert(session, rows)

            if expertise is not None:
                session.execute(delete(FacultyExpertise).where(FacultyExpertise.faculty_id == fac_id))
                if expertise:
                    rows = [{"faculty_id": fac_id, **r} for r in expertise]
                    FacultyExpertiseDAO.bulk_upsert(session, rows)

            if groups is not None:
                session.execute(delete(FacultyResearchGroup).where(FacultyResearchGroup.faculty_id == fac_id))
                if groups:
                    rows = [{"faculty_id": fac_id, **r} for r in groups]
                    FacultyResearchGroupDAO.bulk_upsert(session, rows)

            if links is not None:
                session.execute(delete(FacultyLink).where(FacultyLink.faculty_id == fac_id))
                if links:
                    rows = [{"faculty_id": fac_id, **r} for r in links]
                    FacultyLinkDAO.bulk_upsert(session, rows)

            if publications is not None:
                session.execute(delete(FacultyPublication).where(FacultyPublication.faculty_id == fac_id))
                if publications:
                    rows = [
                        {"faculty_id": fac_id, **r}
                        for r in publications
                    ]
                    FacultyPublicationDAO.bulk_upsert(session, rows)

        return fac_id


# ─────────────────────────────────────────────────────────────
# DEGREE DAO
# ─────────────────────────────────────────────────────────────
class FacultyDegreeDAO:
    @staticmethod
    def bulk_insert(session: Session, rows: List[dict]) -> None:
        """
        Simple bulk insert (used after delete in replace strategy).
        rows: {faculty_id, order_index, degree_text}
        """
        if not rows:
            return
        session.execute(pg_insert(FacultyDegree).values(rows))

    @staticmethod
    def bulk_upsert(session: Session, rows: List[dict]) -> None:
        """
        Upsert by (faculty_id, order_index)
        """
        if not rows:
            return
        stmt = pg_insert(FacultyDegree).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["faculty_id", "order_index"],
            set_={"degree_text": stmt.excluded.degree_text},
        )
        session.execute(stmt)


# ─────────────────────────────────────────────────────────────
# EXPERTISE DAO
# ─────────────────────────────────────────────────────────────
class FacultyExpertiseDAO:
    @staticmethod
    def bulk_upsert(session: Session, rows: List[dict]) -> None:
        """
        Upsert by (faculty_id, term). Since the unique key IS the value,
        we 'do nothing' on conflict (or update term to itself).
        rows: {faculty_id, term}
        """
        if not rows:
            return
        stmt = pg_insert(FacultyExpertise).values(rows)
        # no real 'update' needed; keep existing
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["faculty_id", "term"]
        )
        session.execute(stmt)


# ─────────────────────────────────────────────────────────────
# RESEARCH GROUP DAO
# ─────────────────────────────────────────────────────────────
class FacultyResearchGroupDAO:
    @staticmethod
    def bulk_upsert(session: Session, rows: List[dict]) -> None:
        """
        Upsert by (faculty_id, name, url)
        rows: {faculty_id, name, url}
        """
        if not rows:
            return
        stmt = pg_insert(FacultyResearchGroup).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["faculty_id", "name", "url"],
            set_={
                "name": stmt.excluded.name,
                "url": stmt.excluded.url,
            },
        )
        session.execute(stmt)


# ─────────────────────────────────────────────────────────────
# LINK DAO
# ─────────────────────────────────────────────────────────────
class FacultyLinkDAO:
    @staticmethod
    def bulk_upsert(session: Session, rows: List[dict]) -> None:
        """
        Upsert by (faculty_id, url) — update name if it changes.
        rows: {faculty_id, name, url}
        """
        if not rows:
            return
        stmt = pg_insert(FacultyLink).values(rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=["faculty_id", "url"])
        session.execute(stmt)

# ─────────────────────────────────────────────────────────────
# Publication DAO
# ─────────────────────────────────────────────────────────────
class FacultyPublicationDAO:
    @staticmethod
    def bulk_upsert(session: Session, rows: List[dict]) -> None:
        """
        Upsert publications by (faculty_id, openalex_work_id).

        rows: list of dicts with keys:
            - faculty_id (int)
            - openalex_work_id (str)          # REQUIRED for conflict target
            - title (str)
            - year (int | None)
            - abstract (str | None)           [optional]
            - scholar_author_id (str | None)  [optional]
        """
        if not rows:
            return

        stmt = pg_insert(FacultyPublication).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["faculty_id", "openalex_work_id"],
            set_={
                "title": stmt.excluded.title,
                "year": stmt.excluded.year,
                "abstract": stmt.excluded.abstract,
                "scholar_author_id": stmt.excluded.scholar_author_id,
            },
        )
        session.execute(stmt)

    @staticmethod
    def replace_for_faculty(
        session: Session,
        faculty_id: int,
        rows: List[dict],
    ) -> None:
        """
        Delete all existing publications for a faculty, then insert the given rows.

        rows: dicts WITHOUT faculty_id; this method will inject it.
              e.g. {
                "openalex_work_id": "...",
                "title": "...",
                "year": 2025,
                "abstract": "...",
                "scholar_author_id": "A123..."
              }
        """
        # delete old
        session.execute(
            delete(FacultyPublication).where(FacultyPublication.faculty_id == faculty_id)
        )

        if not rows:
            return

        payload = [
            {"faculty_id": faculty_id, **row}
            for row in rows
        ]
        session.execute(pg_insert(FacultyPublication).values(payload))