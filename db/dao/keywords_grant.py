from __future__ import annotations
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from db.models.keywords_grant import Keyword


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