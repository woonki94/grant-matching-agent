from sqlalchemy import text
from sqlalchemy.orm import Session
from db.db_conn import engine

EMAIL = "alan.fern@oregonstate.edu"

SQL = """
WITH fk AS (
  SELECT f.id AS faculty_id, LOWER(TRIM(k)) AS kw
  FROM faculty f
  JOIN faculty_keywords fk ON fk.faculty_id = f.id
  CROSS JOIN LATERAL jsonb_array_elements_text(fk.keywords) AS k(k)
  WHERE LOWER(f.email) = LOWER(:email)
),
ok AS (
  SELECT kw.opportunity_id, LOWER(TRIM(k)) AS kw
  FROM keywords kw
  CROSS JOIN LATERAL jsonb_array_elements_text(kw.keywords) AS k(k)
)
SELECT
  o.opportunity_id,
  o.opportunity_title,
  COUNT(*)::int AS overlap_count,
  ARRAY_AGG(DISTINCT fk.kw) AS matched_keywords
FROM fk
JOIN ok USING (kw)
JOIN opportunity o ON o.opportunity_id = ok.opportunity_id
GROUP BY o.opportunity_id, o.opportunity_title
ORDER BY overlap_count DESC, o.post_date DESC NULLS LAST
LIMIT 50;
"""

with Session(engine) as s:
    rows = s.execute(text(SQL), {"email": EMAIL}).mappings().all()
    for r in rows:
        print(r["opportunity_id"],r["opportunity_title"], r["overlap_count"], r["matched_keywords"])