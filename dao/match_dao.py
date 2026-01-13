from typing import Any, Dict, List, Tuple

from sqlalchemy import text
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