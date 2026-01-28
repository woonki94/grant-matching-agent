from __future__ import annotations

from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.models.group_match_result import GroupMatchResult, GroupMember  # adjust import path to yours


class GroupMatchDAO:
    def __init__(self, session: Session):
        self.session = session

    def save_group_run(self, row: Dict[str, Any], team: List[int]) -> int:
        # --- upsert group_match_results ---
        stmt = pg_insert(GroupMatchResult).values(row)
        stmt = stmt.on_conflict_do_update(
            index_elements=["grant_id", "lambda", "k", "top_n"],
            set_={
                "alpha": stmt.excluded.alpha,
                "objective": stmt.excluded.objective,
                "redundancy": stmt.excluded.redundancy,
                "status": stmt.excluded.status,
                "meta": stmt.excluded.meta,
            },
        ).returning(GroupMatchResult.id)

        group_id = int(self.session.execute(stmt).scalar_one())

        # --- replace members (delete then insert) ---
        self.session.query(GroupMember).filter(GroupMember.group_id == group_id).delete(
            synchronize_session=False
        )

        if team:
            member_rows = [
                {"group_id": group_id, "faculty_id": int(fid), "rank_in_group": int(i)}
                for i, fid in enumerate(team)
            ]
            self.session.execute(pg_insert(GroupMember).values(member_rows))

        return group_id