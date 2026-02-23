import ast
import json
from typing import Any, Dict, List, Tuple, Optional

from sqlalchemy import text, desc, bindparam
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from db.models.match_result import MatchResult


SQL_TOPK_OPPS_FOR_QUERY = text("""
SELECT
  oemb.opportunity_id,
  GREATEST(
    CASE WHEN :q_research_vec IS NOT NULL AND oemb.research_domain_vec IS NOT NULL
      THEN 1 - (oemb.research_domain_vec <=> :q_research_vec) END,
    CASE WHEN :q_application_vec IS NOT NULL AND oemb.application_domain_vec IS NOT NULL
      THEN 1 - (oemb.application_domain_vec <=> :q_application_vec) END,
    CASE WHEN :q_research_vec IS NOT NULL AND oemb.application_domain_vec IS NOT NULL
      THEN 1 - (oemb.application_domain_vec <=> :q_research_vec) END,
    CASE WHEN :q_application_vec IS NOT NULL AND oemb.research_domain_vec IS NOT NULL
      THEN 1 - (oemb.research_domain_vec <=> :q_application_vec) END
  ) AS domain_sim
FROM opportunity_keyword_embedding oemb
ORDER BY domain_sim DESC NULLS LAST
LIMIT :k
""").bindparams(
    bindparam("q_research_vec", type_=Vector),
    bindparam("q_application_vec", type_=Vector),
)

SQL_TOPK_FACULTY_FOR_OPPORTUNITY = text("""
SELECT
  femb.faculty_id,
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
JOIN opportunity_keyword_embedding oemb
  ON oemb.opportunity_id = :opportunity_id
ORDER BY domain_sim DESC NULLS LAST
LIMIT :k
""")

SQL_DOMAIN_SIM_FOR_FACULTY_OPP = text("""
SELECT
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
JOIN opportunity_keyword_embedding oemb
  ON oemb.opportunity_id = :opportunity_id
WHERE femb.faculty_id = :faculty_id
LIMIT 1
""")


class MatchDAO:
    """Data access layer for match result read/write operations."""

    def __init__(self, session: Session):
        """Initialize DAO with an active SQLAlchemy session."""
        self.session = session

    # =============== Helper Actions ===============
    @staticmethod
    def _rows_to_scored_pairs(rows) -> List[Tuple[str, float]]:
        """Normalize SQL result rows to (opportunity_id, domain_similarity)."""
        return [(r.opportunity_id, float(r.domain_sim or 0.0)) for r in rows]

    @staticmethod
    def _coerce_vector_param(vec: Any) -> Optional[List[float]]:
        """
        Normalize vector values to List[float] for pgvector binding.

        Handles:
        - None
        - list/tuple of numeric-ish values
        - JSON / repr string, e.g. "[0.1, -0.2, ...]"
        """
        if vec is None:
            return None

        if isinstance(vec, str):
            s = vec.strip()
            if not s:
                return None
            parsed = None
            try:
                parsed = json.loads(s)
            except Exception:
                try:
                    parsed = ast.literal_eval(s)
                except Exception as exc:
                    raise ValueError(f"Invalid vector string format: {s[:120]}") from exc
            vec = parsed

        if isinstance(vec, tuple):
            vec = list(vec)

        if isinstance(vec, list):
            out: List[float] = []
            for x in vec:
                out.append(float(x))
            return out

        # Last resort: try generic iterable (e.g., numpy array)
        try:
            return [float(x) for x in vec]
        except Exception as exc:
            raise ValueError(f"Unsupported vector type: {type(vec).__name__}") from exc

    # =============== Upsert Actions ===============
    def upsert_matches(self, rows: List[Dict[str, Any]]) -> int:
        """Bulk upsert match rows by (grant_id, faculty_id)."""
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
            },
        )
        self.session.execute(stmt)
        return len(rows)

    # =============== Search/Ranking Actions ===============
    def topk_opps_for_faculty(self, faculty_id: int, k: int) -> List[Tuple[str, float]]:
        """Find top-k opportunities using stored faculty embedding vectors."""
        faculty_vecs = self.session.execute(
            text(
                """
                SELECT research_domain_vec, application_domain_vec
                FROM faculty_keyword_embedding
                WHERE faculty_id = :faculty_id
                LIMIT 1
                """
            ),
            {"faculty_id": faculty_id},
        ).first()
        if not faculty_vecs:
            return []

        return self.topk_opps_for_query(
            research_vec=faculty_vecs.research_domain_vec,
            application_vec=faculty_vecs.application_domain_vec,
            k=k,
        )

    def topk_opps_for_query(
        self,
        *,
        research_vec: Optional[List[float]],
        application_vec: Optional[List[float]],
        k: int,
    ) -> List[Tuple[str, float]]:
        """Find top-k opportunities using runtime query vectors."""
        q_research_vec = self._coerce_vector_param(research_vec)
        q_application_vec = self._coerce_vector_param(application_vec)

        if q_research_vec is None and q_application_vec is None:
            return []

        rows = self.session.execute(
            SQL_TOPK_OPPS_FOR_QUERY,
            {
                "q_research_vec": q_research_vec,
                "q_application_vec": q_application_vec,
                "k": k,
            },
        ).all()
        return self._rows_to_scored_pairs(rows)

    def topk_faculties_for_opportunity(self, *, opportunity_id: str, k: int) -> List[Tuple[int, float]]:
        """Find top-k faculty for one opportunity using stored embedding vectors."""
        if not opportunity_id:
            return []
        rows = self.session.execute(
            SQL_TOPK_FACULTY_FOR_OPPORTUNITY,
            {"opportunity_id": str(opportunity_id), "k": int(k)},
        ).all()
        return [(int(r.faculty_id), float(r.domain_sim or 0.0)) for r in rows]

    def domain_similarity_for_faculty_opportunity(
        self,
        *,
        faculty_id: int,
        opportunity_id: str,
    ) -> Optional[float]:
        """Compute embedding-domain similarity for one faculty-opportunity pair."""
        row = self.session.execute(
            SQL_DOMAIN_SIM_FOR_FACULTY_OPP,
            {
                "faculty_id": int(faculty_id),
                "opportunity_id": str(opportunity_id),
            },
        ).first()
        if not row:
            return None
        return float(row.domain_sim or 0.0)

    # =============== Read Actions ===============
    def top_matches_for_faculty(self, faculty_id: int, k: int = 5):
        """Read top stored match results for one faculty ordered by LLM/domain score."""
        q = (
            self.session.query(MatchResult.grant_id, MatchResult.domain_score, MatchResult.llm_score)
            .filter(MatchResult.faculty_id == faculty_id)
            .order_by(MatchResult.llm_score.desc(), MatchResult.domain_score.desc())
            .limit(k)
        )
        return [(gid, float(d), float(l)) for (gid, d, l) in q.all()]

    def get_match_for_faculty_opportunity(
        self,
        *,
        faculty_id: int,
        opportunity_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Read one stored match row by exact (faculty_id, opportunity_id)."""
        row = self.session.execute(
            text(
                """
                SELECT grant_id, faculty_id, domain_score, llm_score, covered, missing, reason
                FROM match_results
                WHERE faculty_id = :faculty_id AND grant_id = :opportunity_id
                LIMIT 1
                """
            ),
            {
                "faculty_id": int(faculty_id),
                "opportunity_id": str(opportunity_id),
            },
        ).mappings().first()
        return dict(row) if row else None

    def list_matches_for_opportunity(self, opportunity_id: str, limit: int = 200):
        """List stored faculty match rows for a given opportunity, ordered by llm_score DESC."""
        q = text("""
            SELECT faculty_id, domain_score, llm_score, reason, covered, missing
            FROM match_results
            WHERE grant_id = :oid
            ORDER BY llm_score DESC, domain_score DESC
            LIMIT :lim
        """)
        rows = self.session.execute(q, {"oid": opportunity_id, "lim": limit}).mappings().all()
        return [dict(r) for r in rows]

    def list_matches_for_opportunity_by_faculty_ids(
        self,
        opportunity_id: str,
        faculty_ids: List[int],
    ):
        """
        Batch-fetch match rows for a specific set of faculty against one opportunity.
        Returns {faculty_id: {llm_score, domain_score, reason, covered, missing}}.
        """
        if not faculty_ids:
            return {}
        q = text("""
            SELECT faculty_id, domain_score, llm_score, reason, covered, missing
            FROM match_results
            WHERE grant_id = :oid
              AND faculty_id = ANY(:fids)
        """)
        rows = self.session.execute(
            q, {"oid": str(opportunity_id), "fids": [int(f) for f in faculty_ids]}
        ).mappings().all()
        return {
            int(r["faculty_id"]): {
                "llm_score":    round(float(r["llm_score"]    or 0), 3),
                "domain_score": round(float(r["domain_score"] or 0), 3),
                "reason":  str(r["reason"]  or ""),
                "covered": r["covered"] or {},
                "missing": r["missing"] or {},
            }
            for r in rows
        }

    def get_grant_ids_for_faculty(
        self,
        *,
        faculty_id: int,
        min_domain_score: Optional[float] = None,
        min_llm_score: Optional[float] = None,
        limit: Optional[int] = None,
        order_by: str = "llm",  # "llm" | "domain"
    ) -> List[str]:
        """Return matched grant ids for one faculty with optional filters."""
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
