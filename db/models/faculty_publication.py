# faculty_publication.py

from sqlalchemy import (
    Column, String, Integer, Text, ForeignKey,
    UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from db.base import Base


class FacultyPublication(Base):
    __tablename__ = "faculty_publication"
    __table_args__ = (
        UniqueConstraint(
            "faculty_id",
            "openalex_work_id",
            name="ux_faculty_publication_unique_work",
        ),
        Index("ix_faculty_publication_faculty", "faculty_id"),
        Index("ix_faculty_publication_year", "year"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    faculty_id = Column(
        Integer,
        ForeignKey("faculty.id", ondelete="CASCADE"),
        nullable=False,
    )

    openalex_work_id = Column(String(255), nullable=True)
    scholar_author_id = Column(String(255))

    title = Column(Text, nullable=False)
    abstract = Column(Text)
    year = Column(Integer)

    faculty = relationship("Faculty", back_populates="publications")