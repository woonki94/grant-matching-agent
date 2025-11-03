from sqlalchemy.orm import Session, selectinload
from db.models.grant import Opportunity
def _safe_iso(v):
    if v is None:
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)

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