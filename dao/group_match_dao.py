from __future__ import annotations

from typing import List

from numpy import integer
from sqlalchemy.orm import Session

from sqlalchemy.dialects.postgresql import insert as pg_insert
from db.models.group_match_result import GroupMatchResult


class GroupMatchDAO:
    def __init__(self, session: Session):
        self.session = session

    def upsert(
        self,
        *,
        grant_id: str,
        faculty_ids: List[int],
        team_size: int,
        final_coverage: float,
    ) -> GroupMatchResult:

        faculty_ids = sorted(faculty_ids)

        stmt = (
            pg_insert(GroupMatchResult)
            .values(
                grant_id=grant_id,
                faculty_ids=faculty_ids,
                team_size=team_size,
                final_coverage=final_coverage,
            )
            .on_conflict_do_update(
                constraint="ux_group_grant_faculty_ids",
                set_={
                    # fields you want to update on conflict
                    "team_size": team_size,
                    "final_coverage": final_coverage,
                },
            )
            .returning(GroupMatchResult)
        )

        result = self.session.execute(stmt)
        obj = result.scalar_one()
        return obj