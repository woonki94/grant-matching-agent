from __future__ import annotations

import logging
from typing import List, Dict, Any, Iterator
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import selectinload

from db.models.opportunity import Opportunity, OpportunityAttachment, OpportunityAdditionalInfo, OpportunityKeyword, \
    OpportunityKeywordEmbedding
from dto.opportunity_dto import OpportunityDTO, OpportunityAttachmentDTO, OpportunityAdditionalInfoDTO


from logging_setup import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

OPPORTUNITY_COLS = {
    "agency_name",
    "category",
    "opportunity_status",
    "opportunity_title",
    "agency_email_address",
    "applicant_types",
    "archive_date",
    "award_ceiling",
    "award_floor",
    "close_date",
    "created_at",
    "estimated_total_program_funding",
    "expected_number_of_awards",
    "forecasted_award_date",
    "forecasted_close_date",
    "forecasted_post_date",
    "forecasted_project_start_date",
    "funding_categories",
    "funding_instruments",
    "is_cost_sharing",
    "post_date",
    "summary_description",
}


class OpportunityDAO:
    def __init__(self, session: Session):
        self.session = session

    def upsert_opportunity(self, dto: OpportunityDTO) -> Opportunity:
        if not dto.opportunity_id:
            logger.exception("Opportunity ID is required for email-based upsert")
            raise

        obj = self.session.get(Opportunity, dto.opportunity_id)
        if obj is None:
            obj = Opportunity(opportunity_id=dto.opportunity_id)
            self.session.add(obj)

        data: Dict[str, Any] = dto.model_dump(include=OPPORTUNITY_COLS, exclude_unset=True)

        for k, v in data.items():
            setattr(obj, k, v)

        return obj

    def upsert_attachments(self, opportunity_id: str, attachments: List[OpportunityAttachmentDTO]) -> int:
        count = 0
        for a in attachments:
            obj = (
                self.session.query(OpportunityAttachment)
                .filter(
                    OpportunityAttachment.opportunity_id == opportunity_id,
                    OpportunityAttachment.file_name == a.file_name,
                )
                .one_or_none()
            )

            if obj is None:
                obj = OpportunityAttachment(
                    opportunity_id=opportunity_id,
                    file_name=a.file_name,
                    file_download_path=a.file_download_path,
                    extract_status=a.extract_status or "pending",
                )
                self.session.add(obj)
            else:
                # keep download path current, but DO NOT reset status here
                obj.file_download_path = a.file_download_path

            count += 1
        return count

    def upsert_additional_info(self, opportunity_id: str, items: List[OpportunityAdditionalInfoDTO]) -> int:
        count = 0
        for info in items:
            obj = (
                self.session.query(OpportunityAdditionalInfo)
                .filter(
                    OpportunityAdditionalInfo.opportunity_id == opportunity_id,
                    OpportunityAdditionalInfo.additional_info_url == info.additional_info_url,
                )
                .one_or_none()
            )

            if obj is None:
                obj = OpportunityAdditionalInfo(
                    opportunity_id=opportunity_id,
                    additional_info_url=info.additional_info_url,
                    extract_status=info.extract_status or "pending",
                )
                self.session.add(obj)
            else:
                obj.additional_info_url = info.additional_info_url

            count += 1
        return count

    def iter_opportunities_with_relations(self, batch_size: int = 200) -> Iterator[Opportunity]:
        q = (
            self.session.query(Opportunity)
            .options(
                selectinload(Opportunity.additional_info),
                selectinload(Opportunity.attachments),
                selectinload(Opportunity.keyword),
            )
            .yield_per(batch_size)
        )

        for opp in q:
            yield opp

    def read_opportunities_by_ids_with_relations(self, ids: list[str]) -> list[Opportunity]:
        if not ids:
            return []
        return (
            self.session.query(Opportunity)
            .options(
                selectinload(Opportunity.additional_info),
                selectinload(Opportunity.attachments),
                selectinload(Opportunity.keyword),
            )
            .filter(Opportunity.opportunity_id.in_(ids))
            .all()
        )

    def upsert_keywords_json(self, rows: List[Dict[str, Any]]) -> int:
        """
        Bulk upsert OpportunityKeyword rows by opportunity_id (or opportunity_id FK).
        Does NOT commit (caller commits).
        Returns number of rows provided.
        """
        if not rows:
            return 0

        stmt = pg_insert(OpportunityKeyword).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[OpportunityKeyword.opportunity_id],  # or OpportunityKeyword.opportunity_id_fk
            set_={
                "keywords": stmt.excluded.keywords,
                "raw_json": stmt.excluded.raw_json,
                "source": stmt.excluded.source,
            },
        )

        self.session.execute(stmt)
        return len(rows)

    def iter_opportunities_with_keywords(self):
        return (
            self.session.query(Opportunity)
            .options(selectinload(Opportunity.keyword))
            .yield_per(200)
        )


    def upsert_keyword_embedding(self, row: dict) -> None:
        """
        row = {
          faculty_id: int,
          model: str,
          research_domain_vec: list[float] | None,
          application_domain_vec: list[float] | None
        }
        """
        stmt = pg_insert(OpportunityKeywordEmbedding).values([row])
        stmt = stmt.on_conflict_do_update(
            index_elements=["opportunity_id"],
            set_={
                "model": stmt.excluded.model,
                "research_domain_vec": stmt.excluded.research_domain_vec,
                "application_domain_vec": stmt.excluded.application_domain_vec,
            },
        )
        self.session.execute(stmt)