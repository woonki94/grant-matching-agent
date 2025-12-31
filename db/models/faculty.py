from sqlalchemy import (
    Column, String, Integer, Text, JSON, ForeignKey,
    UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from db.base import Base


class Faculty(Base):
    __tablename__ = "faculty"
    __table_args__ = (
        UniqueConstraint("source_url", name="ux_faculty_source_url"),
        #UniqueConstraint("email", name="ux_faculty_email"),
        Index("ix_faculty_name", "name"),
        Index("ix_faculty_email", "email"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    # identity / main fields
    source_url = Column(String(1024), nullable=False)  # canonical profile URL
    name = Column(String(255))
    email = Column(String(255))
    phone = Column(String(64))
    position = Column(String(255))

    # orgs
    organization = Column(Text)        # combined single-line
    organizations = Column(JSON)       # list[str]

    # text blocks
    address = Column(Text)             # keep newlines
    biography = Column(Text)

    # research website (flattened)
    research_website_name = Column(String(255))
    research_website_url = Column(String(1024))

    # simple arrays
    degrees = relationship(
        "FacultyDegree",
        back_populates="faculty",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    expertise = relationship(
        "FacultyExpertise",
        back_populates="faculty",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    research_groups = relationship(
        "FacultyResearchGroup",
        back_populates="faculty",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    links = relationship(
        "FacultyLink",
        back_populates="faculty",
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
    publications = relationship(
        "FacultyPublication",
        back_populates="faculty",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

class FacultyDegree(Base):
    __tablename__ = "faculty_degree"
    __table_args__ = (
        UniqueConstraint("faculty_id", "order_index", name="ux_faculty_degree_order"),
        Index("ix_faculty_degree_faculty", "faculty_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    faculty_id = Column(Integer, ForeignKey("faculty.id", ondelete="CASCADE"), nullable=False)
    order_index = Column(Integer, default=0, nullable=False)
    degree_text = Column(Text, nullable=False)

    faculty = relationship("Faculty", back_populates="degrees")


class FacultyExpertise(Base):
    __tablename__ = "faculty_expertise"
    __table_args__ = (
        UniqueConstraint("faculty_id", "term", name="ux_faculty_expertise_term"),
        Index("ix_faculty_expertise_faculty", "faculty_id"),
        Index("ix_faculty_expertise_term", "term"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    faculty_id = Column(Integer, ForeignKey("faculty.id", ondelete="CASCADE"), nullable=False)
    term = Column(String(255), nullable=False)

    faculty = relationship("Faculty", back_populates="expertise")


class FacultyResearchGroup(Base):
    __tablename__ = "faculty_research_group"
    __table_args__ = (
        UniqueConstraint("faculty_id", "name", "url", name="ux_faculty_group_row"),
        Index("ix_faculty_group_faculty", "faculty_id"),
        Index("ix_faculty_group_name", "name"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    faculty_id = Column(Integer, ForeignKey("faculty.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    url = Column(String(1024))

    faculty = relationship("Faculty", back_populates="research_groups")


class FacultyLink(Base):
    __tablename__ = "faculty_link"
    __table_args__ = (
        UniqueConstraint("faculty_id", "url", name="ux_faculty_link_url"),
        Index("ix_faculty_link_faculty", "faculty_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    faculty_id = Column(Integer, ForeignKey("faculty.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    url = Column(String(1024), nullable=False)

    faculty = relationship("Faculty", back_populates="links")



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