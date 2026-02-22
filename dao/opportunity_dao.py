from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from sqlalchemy import String, bindparam, func, text
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import ARRAY, insert as pg_insert
from sqlalchemy.orm import selectinload

from db.models.opportunity import (
    Opportunity,
    OpportunityAdditionalInfo,
    OpportunityAttachment,
    OpportunityKeyword,
    OpportunityKeywordEmbedding,
)
from dto.opportunity_dto import OpportunityDTO, OpportunityAttachmentDTO, OpportunityAdditionalInfoDTO

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

    # =============== Helper Actions ===============
    @staticmethod
    def _with_common_relations(query):
        return query.options(
            selectinload(Opportunity.additional_info),
            selectinload(Opportunity.attachments),
            selectinload(Opportunity.keyword),
        )

    # =============== Upsert Actions ===============
    def upsert_opportunity(self, dto: OpportunityDTO) -> Opportunity:
        if not dto.opportunity_id:
            raise ValueError("Opportunity ID is required for upsert")

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

    def upsert_keywords_json(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0

        stmt = pg_insert(OpportunityKeyword).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[OpportunityKeyword.opportunity_id],
            set_={
                "keywords": stmt.excluded.keywords,
                "raw_json": stmt.excluded.raw_json,
                "source": stmt.excluded.source,
            },
        )

        self.session.execute(stmt)
        return len(rows)

    def update_keyword_categories(
        self,
        *,
        opportunity_id: str,
        broad_category: Optional[str],
        specific_categories: Optional[List[str]],
    ) -> None:
        """Update opportunity keyword category columns without model-column coupling."""
        if not opportunity_id:
            return

        clean_specific = [str(x).strip() for x in (specific_categories or []) if str(x).strip()]
        stmt = text(
            """
            UPDATE opportunity_keywords
            SET broad_category = :broad_category,
                specific_categories = :specific_categories
            WHERE opportunity_id = :opportunity_id
            """
        ).bindparams(
            bindparam("specific_categories", type_=ARRAY(String)),
        )
        self.session.execute(
            stmt,
            {
                "opportunity_id": str(opportunity_id),
                "broad_category": (str(broad_category).strip() if broad_category else None),
                "specific_categories": clean_specific,
            },
        )

    def upsert_keyword_embedding(self, row: dict) -> None:
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

    # =============== Read Actions ===============
    def read_opportunities_by_ids_with_relations(self, ids: list[str]) -> list[Opportunity]:
        if not ids:
            return []
        return (
            self._with_common_relations(self.session.query(Opportunity))
            .filter(Opportunity.opportunity_id.in_(ids))
            .all()
        )

    def read_opportunities_by_ids_for_keyword_context(self, ids: list[str]) -> list[Opportunity]:
        if not ids:
            return []
        return (
            self.session.query(Opportunity)
            .options(
                selectinload(Opportunity.additional_info),
                selectinload(Opportunity.keyword),
            )
            .filter(Opportunity.opportunity_id.in_(ids))
            .all()
        )

    def read_opportunity_context(self, opportunity_id: str) -> Optional[Dict[str, Any]]:
        opp = (
            self.session.query(Opportunity)
            .options(selectinload(Opportunity.keyword))
            .filter(Opportunity.opportunity_id == opportunity_id)
            .one_or_none()
        )
        if not opp:
            return None

        kw = (opp.keyword.keywords if opp.keyword else {}) or {}
        cat_row = self.session.execute(
            text(
                """
                SELECT broad_category, specific_categories
                FROM opportunity_keywords
                WHERE opportunity_id = :opportunity_id
                LIMIT 1
                """
            ),
            {"opportunity_id": str(opportunity_id)},
        ).mappings().first()
        broad_category = cat_row["broad_category"] if cat_row else None
        specific_categories = list(cat_row["specific_categories"] or []) if cat_row else []

        return {
            "opportunity_id": opp.opportunity_id,
            "title": getattr(opp, "opportunity_title", None),
            "agency": getattr(opp, "agency_name", None),
            "summary": getattr(opp, "summary_description", None),
            "keywords": kw,
            "broad_category": broad_category,
            "specific_categories": specific_categories,
        }

    def has_keyword_row(self, opportunity_id: str) -> bool:
        """Return True when opportunity has a keyword row."""
        if not opportunity_id:
            return False
        row = (
            self.session.query(OpportunityKeyword.opportunity_id)
            .filter(OpportunityKeyword.opportunity_id == str(opportunity_id))
            .one_or_none()
        )
        return row is not None

    def find_opportunity_by_title(self, title: str) -> Optional[Opportunity]:
        """
        Find one opportunity by title using a multi-stage search:
          1. Exact title match (case-insensitive)
          2. Partial title ILIKE match
          3. Partial summary_description ILIKE match (handles NULL/missing title rows)

        All searches skip rows where the target column is NULL so PostgreSQL's
        ``NULL ILIKE …`` silent-no-match behaviour never causes a false negative.
        """
        q = (title or "").strip()
        if not q:
            return None

        base = self._with_common_relations(self.session.query(Opportunity))

        # 1. Exact title (non-NULL only)
        exact = (
            base
            .filter(
                Opportunity.opportunity_title.isnot(None),
                func.lower(Opportunity.opportunity_title) == q.lower(),
            )
            .order_by(Opportunity.created_at.desc().nullslast())
            .first()
        )
        if exact:
            return exact

        # 2. Partial title (non-NULL only)
        partial = (
            base
            .filter(
                Opportunity.opportunity_title.isnot(None),
                Opportunity.opportunity_title.ilike(f"%{q}%"),
            )
            .order_by(Opportunity.created_at.desc().nullslast())
            .first()
        )
        if partial:
            return partial

        # 3. Fallback: search summary_description (catches records with NULL title)
        return (
            base
            .filter(
                Opportunity.summary_description.isnot(None),
                Opportunity.summary_description.ilike(f"%{q}%"),
            )
            .order_by(Opportunity.created_at.desc().nullslast())
            .first()
        )

    # =============== Iteration Actions ===============
    def iter_opportunities_with_relations(self, batch_size: int = 200) -> Iterator[Opportunity]:
        """Iterate all opportunities with common relations preloaded."""
        q = (
            self._with_common_relations(self.session.query(Opportunity))
            .yield_per(batch_size)
        )
        yield from q

    def iter_opportunity_missing_keywords(self, batch_size: int = 200) -> Iterator[Opportunity]:
        """
        Iterate over grants that has not yet generated keywords.
        """
        q = (
            self._with_common_relations(self.session.query(Opportunity))
            .outerjoin(OpportunityKeyword, OpportunityKeyword.opportunity_id == Opportunity.opportunity_id)
            .filter(OpportunityKeyword.opportunity_id.is_(None))
            .yield_per(batch_size)
        )
        yield from q
