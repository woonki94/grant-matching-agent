from sqlalchemy.orm import Session
from db.models.match_result import MatchResult


class MatchResultDAO:
    @staticmethod
    def save_match_result(
        db: Session,
        grant_id: str,
        faculty_id: int,
        domain_score: float,
        llm_score: float,
        reason: str
    ):
        """
        Insert or update a single match result.
        """

        existing = (
            db.query(MatchResult)
              .filter(
                  MatchResult.grant_id == grant_id,
                  MatchResult.faculty_id == faculty_id
              )
              .one_or_none()
        )

        if existing:
            existing.domain_score = domain_score
            existing.llm_score = llm_score
            existing.reason = reason
            db.add(existing)
        else:
            new = MatchResult(
                grant_id=grant_id,
                faculty_id=faculty_id,
                domain_score=domain_score,
                llm_score=llm_score,
                reason=reason
            )
            db.add(new)

        db.commit()