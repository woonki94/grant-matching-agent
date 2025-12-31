from __future__ import annotations
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from db.models.keywords_opportunity import Keyword


class KeywordDAO:
    @staticmethod
    def upsert_keywords_json(db: Session, rows: list[dict]):
        if not rows:
            return
        stmt = pg_insert(Keyword).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["opportunity_id"],  # adjust as needed
            set_={"keywords": stmt.excluded.keywords,
                  "raw_json": stmt.excluded.raw_json,
                  "source": stmt.excluded.source}
        )
        db.execute(stmt)
        db.commit()

    @staticmethod
    def get_by_opportunity_id(db: Session, opportunity_id: str):
        """
        Returns the Keyword row for this opportunity_id, or None if not found.

        Example:
            row = KeywordDAO.get_by_opportunity_id(db, "DE-FOA-0003141")
            if row:
                print(row.keywords)
        """
        return (
            db.query(Keyword)
            .filter(Keyword.opportunity_id == opportunity_id)
            .one_or_none()
        )