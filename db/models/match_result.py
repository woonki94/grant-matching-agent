from sqlalchemy import Column, Float, String, Integer, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

from db.base import Base

class MatchResult(Base):
    __tablename__ = "match_results"

    id = Column(Integer, primary_key=True, autoincrement=True)

    grant_id = Column(String, ForeignKey("opportunity.opportunity_id"), nullable=False)
    faculty_id = Column(Integer, ForeignKey("faculty.id"), nullable=False)

    domain_score = Column(Float, nullable=False)
    llm_score = Column(Float, nullable=False)
    reason = Column(Text, nullable=False)

    grant = relationship("Opportunity", backref="match_results")
    faculty = relationship("Faculty", backref="match_results")

    # OPTIONAL: enforce uniqueness per (grant, faculty)
    __table_args__ = (
        # Create unique constraint to avoid duplicates
        {'sqlite_autoincrement': True},
    )