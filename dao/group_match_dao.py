from __future__ import annotations

from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dao.faculty_dao import FacultyDAO
from db.models import Faculty
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

    def get_faculty_id_by_email(self, email: str) -> Optional[int]:
        fac_id = (
            self.session.query(Faculty.faculty_id)
            .filter(Faculty.email == email)
            .scalar()
        )
        return int(fac_id) if fac_id is not None else None

    def list_group_runs_for_faculty_email(
            self,
            *,
            email: str,
            grant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns group_match_results rows that contain the faculty (via group_member).

        Output rows include:
          group_id, grant_id, lambda, k, top_n, objective, redundancy, status, alpha, meta
        """
        faculty_id = self.get_faculty_id_by_email(email)
        if faculty_id is None:
            return []

        params: Dict[str, Any] = {"fid": int(faculty_id), "lim": int(limit)}
        grant_filter = ""
        if grant_id:
            grant_filter = "AND g.grant_id = :gid"
            params["gid"] = grant_id

        q = text(f"""
            SELECT
                g.id            AS group_id,
                g.grant_id      AS grant_id,
                g.lambda        AS lambda,
                g.k             AS k,
                g.top_n         AS top_n,
                g.objective     AS objective,
                g.redundancy    AS redundancy,
                g.status        AS status,
                g.alpha         AS alpha,
                g.meta          AS meta
            FROM group_match_results g
            JOIN group_member m
              ON m.group_id = g.id
            WHERE m.faculty_id = :fid
            {grant_filter}
            ORDER BY g.grant_id, g.lambda ASC, g.id DESC
        """)
        rows = self.session.execute(q, params).mappings().all()
        return [dict(r) for r in rows]

    def read_group_with_members(self, *, group_id: int) -> Dict[str, Any]:
        """
        Returns:
          {
            "group": {...},
            "members": [{"faculty_id":..,"rank_in_group":..,"role":..}, ...]
          }
        """
        g = self.session.execute(
            text("""
                SELECT
                    id AS group_id,
                    grant_id,
                    lambda,
                    k,
                    top_n,
                    objective,
                    redundancy,
                    status,
                    alpha,
                    meta
                FROM group_match_results
                WHERE id = :gid
            """),
            {"gid": int(group_id)},
        ).mappings().first()

        if not g:
            raise ValueError(f"group_match_results not found: {group_id}")

        mem = self.session.execute(
            text("""
                SELECT faculty_id, rank_in_group, role
                FROM group_member
                WHERE group_id = :gid
                ORDER BY rank_in_group ASC NULLS LAST, faculty_id ASC
            """),
            {"gid": int(group_id)},
        ).mappings().all()

        return {"group": dict(g), "members": [dict(r) for r in mem]}

    def list_groups_for_faculty_email(
            self,
            *,
            email: str,
            limit: int = 100,
            grant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns group_match_results rows where the faculty (by email) is a member.
        """
        params: Dict[str, Any] = {"email": email, "lim": int(limit)}
        grant_filter = ""
        if grant_id:
            grant_filter = "AND g.grant_id = :grant_id"
            params["grant_id"] = grant_id

        q = text(f"""
            SELECT
                g.id          AS group_id,
                g.grant_id    AS grant_id,
                g.lambda      AS lambda,
                g.k           AS k,
                g.top_n       AS top_n,
                g.objective   AS objective,
                g.redundancy  AS redundancy,
                g.status      AS status,
                g.alpha       AS alpha,
                g.meta        AS meta
            FROM group_match_results g
            JOIN group_member gm
              ON gm.group_id = g.id
            JOIN faculty f
              ON f.faculty_id = gm.faculty_id
            WHERE lower(f.email) = lower(:email)
            {grant_filter}
            ORDER BY g.grant_id ASC, g.lambda ASC, g.id DESC
            LIMIT :lim
        """)

        rows = self.session.execute(q, params).mappings().all()
        return [dict(r) for r in rows]

    def read_group_members(
            self,
            *,
            group_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Returns all members for a given group_match_results.id

        [
          {
            "faculty_id": int,
            "rank_in_group": int | None,
            "role": str | None
          },
          ...
        ]
        """
        rows = self.session.execute(
            text("""
                SELECT
                    faculty_id,
                    rank_in_group,
                    role
                FROM group_member
                WHERE group_id = :gid
                ORDER BY
                    rank_in_group ASC NULLS LAST,
                    faculty_id ASC
            """),
            {"gid": int(group_id)},
        ).mappings().all()

        return [dict(r) for r in rows]