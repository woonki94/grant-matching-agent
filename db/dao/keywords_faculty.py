from __future__ import annotations
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from db.models.keywords_faculty import FacultyKeyword

class FacultyKeywordDAO:
    @staticmethod
    def upsert_keywords_json(db: Session, rows: list[dict]):

        if not rows:
            return
        stmt = pg_insert(FacultyKeyword).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["faculty_id"],
            set_={
                "keywords": stmt.excluded.keywords,
                "raw_json": stmt.excluded.raw_json,
                "source": stmt.excluded.source,
            },
        )
        db.execute(stmt)
        db.commit()