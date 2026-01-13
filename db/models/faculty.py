from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    String,
    Integer,
    Text,
    ForeignKey,
    UniqueConstraint,
    Index,
    DateTime,
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

from db.base import Base

EMBED_DIM = 4096


class Faculty(Base):
    __tablename__ = "faculty"
    __table_args__ = (
        UniqueConstraint("email", name="ux_faculty_email"),
        Index("ix_faculty_name", "name"),
    )

    faculty_id = Column(Integer, primary_key=True, autoincrement=True)

    # identity / main fields
    source_url = Column(String(1024), nullable=False)  # canonical profile URL
    name = Column(String(255))
    email = Column(String(255))
    phone = Column(String(64))
    position = Column(String(255))

    # orgs
    organization = Column(Text)   # combined single-line
    organizations = Column(JSONB) # list[str]

    # text blocks
    address = Column(Text)        # keep newlines
    biography = Column(Text)

    # collapsed arrays (store as JSON)
    degrees = Column(JSONB)         # list[str] (ordered)
    expertise = Column(JSONB)       # list[str]

    additional_info = relationship(
        "FacultyAdditionalInfo",
        back_populates="faculty",
        uselist=True,  # one-to-many
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    publications = relationship(
        "FacultyPublication",
        back_populates="faculty",
        uselist=True,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    keyword = relationship(
        "FacultyKeyword",
        back_populates="faculty",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    keyword_embedding = relationship(
        "FacultyKeywordEmbedding",
        back_populates="faculty",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class FacultyAdditionalInfo(Base):
    __tablename__ = "faculty_additional_info"
    __table_args__ = (
        UniqueConstraint(
            "faculty_id",
            "additional_info_url",
            name="ux_faculty_additional_info_opp_url",
        ),
        Index("ix_faculty_additional_info_faculty_id", "faculty_id"),
        Index("ix_faculty_additional_info_extracted_at", "extracted_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    faculty_id = Column(
        Integer,
        ForeignKey("faculty.faculty_id", ondelete="CASCADE"),
        nullable=False,
        #unique=True,  # one row per faculty
    )

    additional_info_url = Column(String, nullable=False)

    # extracted content
    content_path = Column(String(1024))
    detected_type = Column(String(32))       # pdf, html, docx, etc
    content_char_count = Column(Integer)
    extracted_at = Column(DateTime)
    extract_status = Column(String(32), nullable=False, default="pending")
    extract_error = Column(Text)

    faculty = relationship(
        "Faculty",
        back_populates="additional_info",
    )


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
        ForeignKey("faculty.faculty_id", ondelete="CASCADE"),
        nullable=False,
    )

    openalex_work_id = Column(String(255), nullable=True)
    scholar_author_id = Column(String(255))

    title = Column(Text, nullable=False)
    abstract = Column(Text)
    year = Column(Integer)

    faculty = relationship("Faculty", back_populates="publications")


class FacultyKeyword(Base):
    __tablename__ = "faculty_keywords"
    __table_args__ = (
        # one row per faculty
        UniqueConstraint("faculty_id", name="ux_faculty_keyword_faculty"),
        Index("ix_faculty_keyword_faculty_id", "faculty_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    faculty_id = Column(
        Integer,
        ForeignKey("faculty.faculty_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # redundant with UniqueConstraint, but OK; remove one if you want
    )

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


class FacultyKeywordEmbedding(Base):
    __tablename__ = "faculty_keyword_embedding"
    __table_args__ = (
        UniqueConstraint("faculty_id", name="ux_faculty_keyword_embedding_faculty"),
        Index("ix_faculty_keyword_embedding_model", "model"),
    )

    faculty_id = Column(
        Integer,
        ForeignKey("faculty.faculty_id", ondelete="CASCADE"),
        primary_key=True,
    )

    # which embedding model produced these vectors
    model = Column(String(128), nullable=False)

    # domain-only embeddings (you decided to keep it simple)
    research_domain_vec = Column(Vector(EMBED_DIM), nullable=True)      # adjust dim
    application_domain_vec = Column(Vector(EMBED_DIM), nullable=True)   # adjust dim

    faculty = relationship("Faculty", back_populates="keyword_embedding")