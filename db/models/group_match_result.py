from sqlalchemy import Column, Integer, String, Index, Float, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

from db.base import Base


class GroupMatchResult(Base):
    __tablename__ = "group_match_results"

    id = Column(Integer, primary_key=True)
    grant_id = Column(String, nullable=False, index=True)

    faculty_ids = Column(ARRAY(Integer), nullable=False)
    team_size = Column(Integer, nullable=False)
    final_coverage = Column(JSONB, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "grant_id",
            "faculty_ids",
            name="ux_group_grant_faculty_ids",
        ),
        Index("ix_group_faculty_ids", faculty_ids, postgresql_using="gin"),
    )