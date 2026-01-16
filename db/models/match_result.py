from sqlalchemy import (
    Column, Float, String, Integer, ForeignKey,
    UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from db.base import Base

class MatchResult(Base):
    __tablename__ = "match_results"

    id = Column(Integer, primary_key=True, autoincrement=True)

    #TODO: grant_id -> opportunity_id
    grant_id = Column(
        String,
        ForeignKey("opportunity.opportunity_id", ondelete="CASCADE"),
        nullable=False,
    )
    faculty_id = Column(
        Integer,
        ForeignKey("faculty.faculty_id", ondelete="CASCADE"),
        nullable=False,
    )

    # Stage-1 fast filter score (embedding cosine)
    domain_score = Column(Float, nullable=False)

    # Stage-2 LLM judgment score
    llm_score = Column(Float, nullable=False)

    # One-sentence, brief explanation (log-level)
    reason = Column(String(256), nullable=False)

    covered = Column(JSONB, nullable=False, default=list)  # ["topic1", "topic2", ...]
    missing = Column(JSONB, nullable=False, default=list)  # ["topicX", ...]
    evidence = Column(JSONB, nullable=False, default=dict)  # {"grant_kw": ["faculty_kw", ...]}

    grant = relationship("Opportunity", backref="match_results")
    faculty = relationship("Faculty", backref="match_results")

    __table_args__ = (
        UniqueConstraint("grant_id", "faculty_id", name="ux_match_grant_faculty"),
        Index("ix_match_grant", "grant_id"),
        Index("ix_match_faculty", "faculty_id"),
    )