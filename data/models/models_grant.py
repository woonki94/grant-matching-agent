# data/models_grant.py
from sqlalchemy import Column, String, Float, Boolean, Integer, ForeignKey, JSON, Text, DateTime, UniqueConstraint, \
    Index
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class Opportunity(Base):
    __tablename__ = "opportunity"

    opportunity_id = Column(String, primary_key=True)
    agency_name = Column(String)
    category = Column(String)
    opportunity_status = Column(String)
    opportunity_title = Column(String)

    # summary fields
    additional_info_url = Column(String)
    agency_email_address = Column(String)
    applicant_types = Column(JSON)
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
    funding_categories = Column(JSON)
    funding_instruments = Column(JSON)
    is_cost_sharing = Column(Boolean)
    post_date = Column(String)
    summary_description = Column(Text)

    attachments = relationship(
        "Attachment",
        back_populates="opportunity",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    keyword = relationship(
        "Keyword",
        back_populates="opportunity",
        uselist=False,  # one-to-one
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

class Attachment(Base):
    __tablename__ = "attachment"
    __table_args__ = (
        # De-dupe by (opp_id, file_name) AND also by (opp_id, download_path).
        # Having both is fine and safest in practice.
        UniqueConstraint("opportunity_id", "file_name", name="ux_attachment_opportunity_file"),
        UniqueConstraint("opportunity_id", "download_path", name="ux_attachment_opportunity_download"),
        Index("ix_attachment_opp", "opportunity_id"),
        Index("ix_attachment_detected_type", "detected_type"),
        Index("ix_attachment_extracted_at", "extracted_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    opportunity_id = Column(String, ForeignKey("opportunity.opportunity_id", ondelete="CASCADE"), index=True, nullable=False)
    file_name = Column(String, nullable=False)
    download_path = Column(Text, nullable=False)

    # NEW fields for extracted content
    content = Column(Text)                    # extracted text
    detected_type = Column(String(32))        # "pdf", "docx", "xlsx", ...
    content_char_count = Column(Integer)      # len(content)
    extracted_at = Column(DateTime)           # when we extracted

    # optional: relationship back to Opportunity
    opportunity = relationship("Opportunity", back_populates="attachments")
