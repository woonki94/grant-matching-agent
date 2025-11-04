from __future__ import annotations
from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from db.base import Base

class Keyword(Base):
    __tablename__ = "keywords"
    __table_args__ = (
        UniqueConstraint("opportunity_id", name="ux_keyword_opportunity"),  # one row per opp
        Index("ix_keyword_opportunity_id", "opportunity_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    # FK + one-to-one per opportunity
    opportunity_id = Column(
        String,
        ForeignKey("opportunity.opportunity_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,             # one row per opportunity
    )

    #keywords = Column(JSONB, nullable=False, default=list)  # ← JSON array
    keywords = Column(
        JSONB,
        nullable=False,
        default=lambda: {
            "area": "",
            "discipline": "",
            "application_domain": [],
            "research_area": [],
            "methods": [],
            "models": [],
        },
    )
    raw_json = Column(JSONB)
    source = Column(String, nullable=False, default="gemini")

    opportunity = relationship("Opportunity", back_populates="keyword")