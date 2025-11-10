from __future__ import annotations
from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from db.base import Base

class FacultyKeyword(Base):
    __tablename__ = "faculty_keywords"
    __table_args__ = (
        UniqueConstraint("faculty_id", name="ux_faculty_keyword_faculty"),
        Index("ix_faculty_keyword_faculty_id", "faculty_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    faculty_id = Column(Integer, ForeignKey("faculty.id", ondelete="CASCADE"), nullable=False, unique=True)

    #keywords = Column(JSONB, nullable=False, default=list)
    keywords = Column(
        JSONB,
        nullable=False,
        default=lambda: {
            "research": {"domain": [], "specialization": []},
            "application": {"domain": [], "specialization": []},
        },
    )
    raw_json = Column(JSONB)
    source = Column(String, nullable=False, default="gpt-5")

    faculty = relationship("Faculty", back_populates="keyword")