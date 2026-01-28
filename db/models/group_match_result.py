from __future__ import annotations

from sqlalchemy import (
    Column, Float, String, Integer, ForeignKey,
    UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from db.base import Base


class GroupMatchResult(Base):
    __tablename__ = "group_match_results"

    id = Column(Integer, primary_key=True, autoincrement=True)

    grant_id = Column(
        String,
        ForeignKey("opportunity.opportunity_id", ondelete="CASCADE"),
        nullable=False,
    )

    # Hyperparams / config
    lambda_ = Column("lambda", Float, nullable=False)   # "lambda" is reserved in Python
    k = Column(Integer, nullable=False)       # team size
    top_n = Column(Integer, nullable=False)   # candidate shortlist size

    # e.g. {"application": 1.0, "research": 1.0}
    alpha = Column(JSONB, nullable=False, default=dict)

    # Solve outputs / metrics
    objective = Column(Float, nullable=True)
    redundancy = Column(Float, nullable=True)     # your team-level redundancy scalar (avg overlap)
    status = Column(String(64), nullable=True)    # e.g. "Optimal", "Infeasible", etc.

    # Optional: store small debugging payload (NOT huge)
    # e.g. {"penalty_sum":..., "base_sum":..., "solver":"cbc"}
    meta = Column(JSONB, nullable=False, default=dict)

    grant = relationship("Opportunity", backref="group_match_results")
    members = relationship(
        "GroupMember",
        back_populates="group",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        # Usually you’ll have multiple results per grant_id (different lambda, different runs).
        # This uniqueness prevents accidental duplicates for same config.
        UniqueConstraint("grant_id", "lambda", "k", "top_n", name="ux_group_grant_lambda_k_topn"),
        Index("ix_group_grant", "grant_id"),
        Index("ix_group_grant_lambda", "grant_id", "lambda"),
    )


class GroupMember(Base):
    __tablename__ = "group_member"

    id = Column(Integer, primary_key=True, autoincrement=True)

    group_id = Column(
        Integer,
        ForeignKey("group_match_results.id", ondelete="CASCADE"),
        nullable=False,
    )

    faculty_id = Column(
        Integer,
        ForeignKey("faculty.faculty_id", ondelete="CASCADE"),
        nullable=False,
    )

    # Optional: ordering / role
    # rank_in_group can store the ordering you want (e.g., by base score)
    rank_in_group = Column(Integer, nullable=True)
    role = Column(String(64), nullable=True)  # "PI", "Co-PI", etc (optional)

    group = relationship("GroupMatchResult", back_populates="members")
    faculty = relationship("Faculty", backref="group_memberships")

    __table_args__ = (
        UniqueConstraint("group_id", "faculty_id", name="ux_group_member_unique"),
        Index("ix_group_member_group", "group_id"),
        Index("ix_group_member_faculty", "faculty_id"),
    )