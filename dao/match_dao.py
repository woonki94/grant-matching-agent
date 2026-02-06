from typing import Any, Dict, List, Tuple, Optional

from sqlalchemy import text, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from db.models.match_result import MatchResult  # wherever you put it

SQL_TOPK_OPPS_FOR_FACULTY = text("""
SELECT
  oemb.opportunity_id,
  GREATEST(
    CASE WHEN femb.research_domain_vec IS NOT NULL AND oemb.research_domain_vec IS NOT NULL
      THEN 1 - (femb.research_domain_vec <=> oemb.research_domain_vec) END,
    CASE WHEN femb.application_domain_vec IS NOT NULL AND oemb.application_domain_vec IS NOT NULL
      THEN 1 - (femb.application_domain_vec <=> oemb.application_domain_vec) END,
    CASE WHEN femb.research_domain_vec IS NOT NULL AND oemb.application_domain_vec IS NOT NULL
      THEN 1 - (femb.research_domain_vec <=> oemb.application_domain_vec) END,
    CASE WHEN femb.application_domain_vec IS NOT NULL AND oemb.research_domain_vec IS NOT NULL
      THEN 1 - (femb.application_domain_vec <=> oemb.research_domain_vec) END
  ) AS domain_sim
FROM faculty_keyword_embedding femb
JOIN opportunity_keyword_embedding oemb ON TRUE
WHERE femb.faculty_id = :faculty_id
ORDER BY domain_sim DESC NULLS LAST
LIMIT :k
""")


class MatchDAO:
    def __init__(self, session: Session):
        self.session = session

    def upsert_matches(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0

        stmt = pg_insert(MatchResult).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="ux_match_grant_faculty",
            set_={
                "domain_score": stmt.excluded.domain_score,
                "llm_score": stmt.excluded.llm_score,
                "reason": stmt.excluded.reason,
                "covered": stmt.excluded.covered,
                "missing": stmt.excluded.missing,
                #"evidence": stmt.excluded.evidence,
            },
        )
        self.session.execute(stmt)
        return len(rows)

    def topk_opps_for_faculty(self, faculty_id: int, k: int) -> List[Tuple[str, float]]:
        rows = self.session.execute(
            SQL_TOPK_OPPS_FOR_FACULTY,
            {"faculty_id": faculty_id, "k": k},
        ).all()

        return [(r.opportunity_id, float(r.domain_sim or 0.0)) for r in rows]

    def top_matches_for_faculty(self, faculty_id: int, k: int = 5):
        q = (
            self.session.query(MatchResult.grant_id, MatchResult.domain_score, MatchResult.llm_score)
            .filter(MatchResult.faculty_id == faculty_id)
            .order_by(MatchResult.llm_score.desc(), MatchResult.domain_score.desc())
            .limit(k)
        )
        return [(gid, float(d), float(l)) for (gid, d, l) in q.all()]

    def list_matches_for_opportunity(self, opportunity_id: str, limit: int = 200):
        q = text("""
            SELECT faculty_id, domain_score, llm_score, covered, missing
            FROM match_results
            WHERE grant_id = :oid
            ORDER BY llm_score DESC, domain_score DESC
            LIMIT :lim
        """)
        rows = self.session.execute(q, {"oid": opportunity_id, "lim": limit}).mappings().all()
        return [dict(r) for r in rows]

    def get_grant_ids_for_faculty(
            self,
            *,
            faculty_id: int,
            min_domain_score: Optional[float] = None,
            min_llm_score: Optional[float] = None,
            limit: Optional[int] = None,
            order_by: str = "llm",  # "llm" | "domain"
    ) -> List[str]:
        """
        Return a list of grant_ids matched to a faculty.

        This is intended as a Stage-1 / Stage-2 filter before
        running expensive group matching.
        """

        q = (
            self.session
            .query(MatchResult.grant_id)
            .filter(MatchResult.faculty_id == faculty_id)
        )

        if min_domain_score is not None:
            q = q.filter(MatchResult.domain_score >= min_domain_score)

        if min_llm_score is not None:
            q = q.filter(MatchResult.llm_score >= min_llm_score)

        if order_by == "llm":
            q = q.order_by(desc(MatchResult.llm_score))
        elif order_by == "domain":
            q = q.order_by(desc(MatchResult.domain_score))

        if limit is not None:
            q = q.limit(limit)

        return [row.grant_id for row in q.all()]