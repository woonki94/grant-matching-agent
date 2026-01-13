from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from db.base import Base

EMBED_DIM = 4096


class Opportunity(Base):
    __tablename__ = "opportunity"

    opportunity_id = Column(String, primary_key=True)

    agency_name = Column(String)
    category = Column(String)
    opportunity_status = Column(String)
    opportunity_title = Column(String)

    agency_email_address = Column(String)
    applicant_types = Column(JSONB)          # if you're on Postgres; else JSON
    archive_date = Column(String)
    award_ceiling = Column(Float)
    award_floor = Column(Float)
    close_date = Column(String)
    created_at = Column(String)
    estimated_total_program_funding = Column(Float)
    expected_number_of_awards = Column(Integer)
    forecasted_award_date = Column(String)
    forecasted_close_date = Column(String)
    forecasted_post_date = Column(String)
    forecasted_project_start_date = Column(String)
    funding_categories = Column(JSONB)
    funding_instruments = Column(JSONB)
    is_cost_sharing = Column(Boolean)
    post_date = Column(String)
    summary_description = Column(Text)

    additional_info = relationship(
        "OpportunityAdditionalInfo",
        back_populates="opportunity",
        uselist=True,  # one-to-many
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    attachments = relationship(
        "OpportunityAttachment",
        back_populates="opportunity",
        uselist=True,  # one-to-many
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    keyword = relationship(
        "OpportunityKeyword",
        back_populates="opportunity",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    keyword_embedding = relationship(
        "OpportunityKeywordEmbedding",
        back_populates="opportunity",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class OpportunityAdditionalInfo(Base):
    __tablename__ = "opportunity_additional_info"
    __table_args__ = (
        UniqueConstraint(
            "opportunity_id",
            "additional_info_url",
            name="ux_opportunity_additional_info_opp_url",
        ),
        Index("ix_additional_info_opportunity_id", "opportunity_id"),
        Index("ix_additional_info_extracted_at", "extracted_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    opportunity_id = Column(
        String,
        ForeignKey("opportunity.opportunity_id", ondelete="CASCADE"),
        nullable=False,
        #unique=True,  # one row per opportunity
    )

    additional_info_url = Column(String, nullable=False)

    # extracted content
    content_path = Column(String(1024))
    detected_type = Column(String(32))       # pdf, html, docx, etc
    content_char_count = Column(Integer)
    extracted_at = Column(DateTime)
    extract_status = Column(String(32), nullable=False, default="pending")
    extract_error = Column(Text)

    opportunity = relationship(
        "Opportunity",
        back_populates="additional_info",
    )


class OpportunityAttachment(Base):
    __tablename__ = "opportunity_attachment"
    __table_args__ = (
        UniqueConstraint("opportunity_id", "file_name", name="ux_attachment_opportunity_file"),
        UniqueConstraint("opportunity_id", "file_download_path", name="ux_attachment_opportunity_download"),
        Index("ix_attachment_opp", "opportunity_id"),
        Index("ix_attachment_detected_type", "detected_type"),
        Index("ix_attachment_extracted_at", "extracted_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    opportunity_id = Column(
        String,
        ForeignKey("opportunity.opportunity_id", ondelete="CASCADE"),
        nullable=False,
    )

    file_name = Column(String, nullable=False)
    file_download_path = Column(Text, nullable=False)

    content_path = Column(String(1024))

    detected_type = Column(String(32))
    content_char_count = Column(Integer)
    extracted_at = Column(DateTime)
    extract_status = Column(String(32), nullable=False, default="pending")
    extract_error = Column(Text)

    opportunity = relationship("Opportunity", back_populates="attachments")


class OpportunityKeyword(Base):
    __tablename__ = "opportunity_keywords"
    __table_args__ = (
        UniqueConstraint("opportunity_id", name="ux_keyword_opportunity"),
        Index("ix_keyword_opportunity_id", "opportunity_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    opportunity_id = Column(
        String,
        ForeignKey("opportunity.opportunity_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # optional if keeping UniqueConstraint
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

    opportunity = relationship("Opportunity", back_populates="keyword")


class OpportunityKeywordEmbedding(Base):
    __tablename__ = "opportunity_keyword_embedding"
    __table_args__ = (
        UniqueConstraint("opportunity_id", name="ux_opportunity_keyword_embedding_opp"),
        Index("ix_opportunity_keyword_embedding_model", "model"),
    )

    opportunity_id = Column(
        String,
        ForeignKey("opportunity.opportunity_id", ondelete="CASCADE"),
        primary_key=True,
    )

    model = Column(String(128), nullable=False)

    research_domain_vec = Column(Vector(EMBED_DIM), nullable=True)      # adjust dim
    application_domain_vec = Column(Vector(EMBED_DIM), nullable=True)   # adjust dim

    opportunity = relationship("Opportunity", back_populates="keyword_embedding")