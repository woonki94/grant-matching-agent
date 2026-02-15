import logging

from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import selectinload

from typing import List, Dict, Any, Iterator, Optional

from db.models import Faculty
from db.models.faculty import FacultyAdditionalInfo, FacultyPublication, FacultyKeyword, FacultyKeywordEmbedding
from dto.faculty_dto import FacultyDTO, FacultyAdditionalInfoDTO, FacultyPublicationDTO

from logging_setup import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

FACULTY_COLS = {
    "source_url",
    "name",
    "email",
    "phone",
    "position",
    "organization",
    "organizations",
    "address",
    "biography",
    "degrees",
    "expertise",
}


class FacultyDAO:
    """Data access layer for faculty and related tables."""

    def __init__(self, session: Session):
        """Initialize DAO with an active SQLAlchemy session."""
        self.session = session

    # =============== Helper Actions ===============
    def _query_by_email(self, email: str):
        """Build a base faculty query filtered by email."""
        return self.session.query(Faculty).filter(Faculty.email == email)

    @staticmethod
    def _with_common_relations(query):
        """Attach commonly used faculty relations to a query."""
        return query.options(
            selectinload(Faculty.additional_info),
            selectinload(Faculty.publications),
            selectinload(Faculty.keyword),
        )

    # =============== Read Actions ===============
    def get_by_email(self, email: str) -> Optional[Faculty]:
        """Fetch one faculty row by email."""
        if not email:
            return None
        return self._query_by_email(email).one_or_none()

    def get_faculty_id_by_email(self, email: str) -> Optional[int]:
        """Fetch only faculty_id by email."""
        if not email:
            return None

        row = self._query_by_email(email).with_entities(Faculty.faculty_id).one_or_none()
        return row.faculty_id if row else None

    def get_faculty_keyword_context(self, faculty_id: int) -> Optional[Dict[str, Any]]:
        """Fetch lightweight keyword context for matching/justification."""
        fac = (
            self.session.query(Faculty)
            .options(selectinload(Faculty.keyword))
            .filter(Faculty.faculty_id == int(faculty_id))
            .one_or_none()
        )
        if not fac:
            return None

        kw = (fac.keyword.keywords if fac.keyword else {}) or {}
        return {
            "faculty_id": fac.faculty_id,
            "name": getattr(fac, "name", None),
            "email": getattr(fac, "email", None),
            "keywords": kw,
        }

    # =============== Upsert Actions ===============
    def upsert_faculty(self, dto: FacultyDTO) -> Faculty:
        """Insert or update a faculty row keyed by email."""
        obj = (
            self.session.query(Faculty)
            .filter(Faculty.email == dto.email)
            .one_or_none()
        )

        if obj is None:
            obj = Faculty(email=dto.email)
            self.session.add(obj)

        data: Dict[str, Any] = dto.model_dump(include=FACULTY_COLS, exclude_unset=True)

        for k, v in data.items():
            setattr(obj, k, v)

        return obj

    def upsert_additional_info(self, faculty_id: int, items: List[FacultyAdditionalInfoDTO]) -> int:
        """Insert or update faculty additional-info link rows."""
        count = 0
        for info in items:
            obj = (
                self.session.query(FacultyAdditionalInfo)
                .filter(
                    FacultyAdditionalInfo.faculty_id == faculty_id,
                    FacultyAdditionalInfo.additional_info_url == info.additional_info_url,
                )
                .one_or_none()
            )

            if obj is None:
                obj = FacultyAdditionalInfo(
                    faculty_id=faculty_id,
                    additional_info_url=info.additional_info_url,
                    extract_status=info.extract_status or "pending",
                )
                self.session.add(obj)
            else:
                # Keep existing extraction status/content; only refresh URL.
                obj.additional_info_url = info.additional_info_url

            count += 1

        return count

    def upsert_publications(self, faculty_id: int, items: List[FacultyPublicationDTO]) -> int:
        """Insert or update faculty publication rows keyed by OpenAlex work id."""
        count = 0

        for pub in items:
            # Skip incomplete publication rows.
            if not pub.openalex_work_id or not pub.title:
                continue

            obj = (
                self.session.query(FacultyPublication)
                .filter(
                    FacultyPublication.faculty_id == faculty_id,
                    FacultyPublication.openalex_work_id == pub.openalex_work_id,
                )
                .one_or_none()
            )

            if obj is None:
                obj = FacultyPublication(
                    faculty_id=faculty_id,
                    openalex_work_id=pub.openalex_work_id,
                )
                self.session.add(obj)

            obj.scholar_author_id = pub.scholar_author_id
            obj.title = pub.title
            obj.abstract = pub.abstract
            obj.year = pub.year

            count += 1

        return count

    def upsert_keywords_json(self, rows: List[Dict[str, Any]]) -> int:
        """Bulk upsert faculty keyword JSON rows by faculty_id."""
        if not rows:
            return 0

        stmt = pg_insert(FacultyKeyword).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[FacultyKeyword.faculty_id],
            set_={
                "keywords": stmt.excluded.keywords,
                "raw_json": stmt.excluded.raw_json,
                "source": stmt.excluded.source,
            },
        )

        self.session.execute(stmt)
        return len(rows)

    def upsert_keyword_embedding(self, row: dict) -> None:
        """Upsert one faculty keyword-embedding row by faculty_id."""
        stmt = pg_insert(FacultyKeywordEmbedding).values([row])
        stmt = stmt.on_conflict_do_update(
            index_elements=["faculty_id"],
            set_={
                "model": stmt.excluded.model,
                "research_domain_vec": stmt.excluded.research_domain_vec,
                "application_domain_vec": stmt.excluded.application_domain_vec,
            },
        )
        self.session.execute(stmt)

    # =============== Iteration Actions ===============
    def iter_faculty_with_relations(self, batch_size: int = 200, stream: bool = True) -> Iterator[Faculty]:
        """Iterate faculty rows with common relations preloaded."""
        q = self._with_common_relations(self.session.query(Faculty))

        if stream:
            q = q.yield_per(batch_size)

        return q.all() if not stream else q

    def iter_faculty_with_keywords(self):
        """Stream faculty rows with keyword relation preloaded."""
        return (
            self.session.query(Faculty)
            .options(selectinload(Faculty.keyword))
            .yield_per(200)
        )

    def iter_faculty_missing_keywords(
        self,
        batch_size: int = 200,
        stream: bool = True,
    ) -> Iterator[Faculty]:
        """Iterate faculty rows that do not yet have keyword rows."""
        q = self._with_common_relations(
            self.session.query(Faculty)
            .outerjoin(FacultyKeyword, FacultyKeyword.faculty_id == Faculty.faculty_id)
            .filter(FacultyKeyword.faculty_id.is_(None))
        )

        if stream:
            q = q.yield_per(batch_size)

        return q if stream else q.all()
