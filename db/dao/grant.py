from __future__ import annotations
from datetime import datetime
from typing import List
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session,selectinload


from db.models.grant import Opportunity, Attachment
from util.extract_content import fetch_and_extract_one
def _safe_iso(v):
    if v is None:
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


# ─────────────────────────────────────────────────────────────
# OPPORTUNITY DAO
# ─────────────────────────────────────────────────────────────
class OpportunityDAO:
    @staticmethod
    def upsert_many(session: Session, rows: List[dict]):
        if not rows:
            return
        stmt = pg_insert(Opportunity).values(rows)
        set_map = {
            c.name: getattr(stmt.excluded, c.name)
            for c in Opportunity.__table__.columns
            if c.name != "opportunity_id"
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["opportunity_id"],
            set_=set_map,
        )
        session.execute(stmt)

# ─────────────────────────────────────────────────────────────
# ATTACHMENT DAO (simple version — filename-level upsert)
# ─────────────────────────────────────────────────────────────
class AttachmentDAO:
    @staticmethod
    def bulk_upsert(session: Session, rows: List[dict]):
        """
        rows = [
          {"opportunity_id": "...", "file_name": "...", "download_path": "..."},
          ...
        ]
        """
        if not rows:
            return
        stmt = pg_insert(Attachment).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["opportunity_id", "file_name"],
            set_={"download_path": stmt.excluded.download_path},
        )
        session.execute(stmt)


    # ─────────────────────────────────────────────────────────────
    # ATTACHMENT DAO (advanced — with content fetching & storage)
    # ─────────────────────────────────────────────────────────────
    def upsert_attachment_with_content(
        db: Session,
        *,
        opportunity_id: str,
        file_name: str,
        download_path: str,
        fetch_content: bool = True,
    ) -> int:
        """
        Upsert a single attachment row.
        Fetches and stores extracted text if fetch_content=True.
        """
        content = None
        detected = None
        char_count = None
        extracted_at = None

        if fetch_content and download_path:
            res = fetch_and_extract_one(download_path)
            text = (res.get("text") or "").strip()
            content = text
            detected = res.get("detected_type") or None
            char_count = len(text)
            extracted_at = datetime.utcnow()

        ins = pg_insert(Attachment.__table__).values(
            opportunity_id=opportunity_id,
            file_name=file_name,
            download_path=download_path,
            content=content,
            detected_type=detected,
            content_char_count=char_count,
            extracted_at=extracted_at,
        )

        upd = ins.on_conflict_do_update(
            index_elements=["opportunity_id", "download_path"],  # must exist in DB unique constraint
            set_={
                "file_name": file_name,
                "content": content if content is not None else Attachment.content,
                "detected_type": detected if detected is not None else Attachment.detected_type,
                "content_char_count": char_count if char_count is not None else Attachment.content_char_count,
                "extracted_at": extracted_at if extracted_at is not None else Attachment.extracted_at,
            },
        ).returning(Attachment.id)

        result = db.execute(upd)
        att_id = result.scalar_one()
        db.commit()
        return att_id


    def upsert_attachments_batch_with_content(
        db: Session,
        rows: list[dict],
        fetch_content: bool = True,
    ) -> list[int]:

        ids: list[int] = []
        for r in rows:
            # TODO: Use self.
            att_id = AttachmentDAO.upsert_attachment_with_content(
                db,
                opportunity_id=r["opportunity_id"],
                file_name=r["file_name"],
                download_path=r["download_path"],
                fetch_content=fetch_content,
            )
            ids.append(att_id)
        return ids


class OpportunityReadDAO:
    @staticmethod
    def get_summary_and_files(db: Session, opportunity_id: str) -> dict:
        row = (
            db.query(Opportunity)
              .options(selectinload(Opportunity.attachments))
              .filter(Opportunity.opportunity_id == opportunity_id)
              .one_or_none()
        )
        if not row:
            return {}

        summary = {
            "opportunity_id": row.opportunity_id,
            "opportunity_title": row.opportunity_title,
            "additional_info_url": row.additional_info_url,
            "summary_description": row.summary_description,
        }

        files = [
            {
                "file_name": a.file_name,
                "file_content": a.content,
            }
            for a in (row.attachments or [])
        ]

        return {"opportunityID": row.opportunity_id, "Summary": summary, "additional_files": files}