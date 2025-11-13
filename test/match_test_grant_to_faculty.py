from sqlalchemy import text
from sqlalchemy.orm import Session
from db.db_conn import engine

SQL = """-- name: SQL_FACULTY_MATCHES_FOR_GRANT
WITH g_src AS (
  SELECT k.keywords
  FROM keywords k
  WHERE k.opportunity_id = :opportunity_id
),
-- Flatten GRANT keywords by category (6-key object)
g_terms AS (
  SELECT 'application_domain'::text AS cat, lower(v) AS kw FROM g_src
  CROSS JOIN LATERAL jsonb_array_elements_text(keywords->'application_domain') v
  UNION ALL
  SELECT 'research_area', lower(v) FROM g_src
  CROSS JOIN LATERAL jsonb_array_elements_text(keywords->'research_area') v
  UNION ALL
  SELECT 'methods', lower(v) FROM g_src
  CROSS JOIN LATERAL jsonb_array_elements_text(keywords->'methods') v
  UNION ALL
  SELECT 'models', lower(v) FROM g_src
  CROSS JOIN LATERAL jsonb_array_elements_text(keywords->'models') v
  UNION ALL
  SELECT 'area', lower(keywords->>'area') FROM g_src
  WHERE coalesce(keywords->>'area','') <> ''
  UNION ALL
  SELECT 'discipline', lower(keywords->>'discipline') FROM g_src
  WHERE coalesce(keywords->>'discipline','') <> ''
),
-- Flatten FACULTY keywords by category (supports 6-key object and legacy flat-array → research_area)
f_terms AS (
  SELECT f.id AS faculty_id, f.name, f.email, 'application_domain'::text AS cat, lower(v) AS kw
  FROM faculty f
  JOIN faculty_keywords fk ON fk.faculty_id = f.id
  CROSS JOIN LATERAL jsonb_array_elements_text(fk.keywords->'application_domain') v

  UNION ALL
  SELECT f.id, f.name, f.email, 'research_area', lower(v)
  FROM faculty f
  JOIN faculty_keywords fk ON fk.faculty_id = f.id
  CROSS JOIN LATERAL jsonb_array_elements_text(fk.keywords->'research_area') v

  UNION ALL
  SELECT f.id, f.name, f.email, 'methods', lower(v)
  FROM faculty f
  JOIN faculty_keywords fk ON fk.faculty_id = f.id
  CROSS JOIN LATERAL jsonb_array_elements_text(fk.keywords->'methods') v

  UNION ALL
  SELECT f.id, f.name, f.email, 'models', lower(v)
  FROM faculty f
  JOIN faculty_keywords fk ON fk.faculty_id = f.id
  CROSS JOIN LATERAL jsonb_array_elements_text(fk.keywords->'models') v

  UNION ALL
  SELECT f.id, f.name, f.email, 'area', lower(fk.keywords->>'area')
  FROM faculty f
  JOIN faculty_keywords fk ON fk.faculty_id = f.id
  WHERE coalesce(fk.keywords->>'area','') <> ''

  UNION ALL
  SELECT f.id, f.name, f.email, 'discipline', lower(fk.keywords->>'discipline')
  FROM faculty f
  JOIN faculty_keywords fk ON fk.faculty_id = f.id
  WHERE coalesce(fk.keywords->>'discipline','') <> ''

  UNION ALL
  -- legacy: flat array -> treat as research_area
  SELECT f.id, f.name, f.email, 'research_area', lower(v)
  FROM faculty f
  JOIN faculty_keywords fk ON fk.faculty_id = f.id
  CROSS JOIN LATERAL jsonb_array_elements_text(fk.keywords) v
  WHERE jsonb_typeof(fk.keywords)='array'
),
-- Matches within SAME category
m AS (
  SELECT ft.faculty_id, ft.name, ft.email, ft.cat, ft.kw
  FROM f_terms ft
  JOIN g_terms gt ON gt.cat = ft.cat AND gt.kw = ft.kw
)
SELECT
  m.faculty_id,
  COALESCE(m.name,'') AS name,
  COALESCE(m.email,'') AS email,
  COUNT(*) FILTER (WHERE cat='application_domain')::int AS hits_application_domain,
  ARRAY_REMOVE(ARRAY_AGG(DISTINCT kw) FILTER (WHERE cat='application_domain'), NULL) AS match_application_domain,
  COUNT(*) FILTER (WHERE cat='research_area')::int      AS hits_research_area,
  ARRAY_REMOVE(ARRAY_AGG(DISTINCT kw) FILTER (WHERE cat='research_area'), NULL)      AS match_research_area,
  COUNT(*) FILTER (WHERE cat='methods')::int            AS hits_methods,
  ARRAY_REMOVE(ARRAY_AGG(DISTINCT kw) FILTER (WHERE cat='methods'), NULL)            AS match_methods,
  COUNT(*) FILTER (WHERE cat='models')::int             AS hits_models,
  ARRAY_REMOVE(ARRAY_AGG(DISTINCT kw) FILTER (WHERE cat='models'), NULL)             AS match_models,
  COUNT(*) FILTER (WHERE cat='area')::int               AS hits_area,
  ARRAY_REMOVE(ARRAY_AGG(DISTINCT kw) FILTER (WHERE cat='area'), NULL)               AS match_area,
  COUNT(*) FILTER (WHERE cat='discipline')::int         AS hits_discipline,
  ARRAY_REMOVE(ARRAY_AGG(DISTINCT kw) FILTER (WHERE cat='discipline'), NULL)         AS match_discipline
FROM m
GROUP BY m.faculty_id, m.name, m.email
ORDER BY
  (  COUNT(*) FILTER (WHERE cat='application_domain')
   + COUNT(*) FILTER (WHERE cat='research_area')
   + COUNT(*) FILTER (WHERE cat='methods')
   + COUNT(*) FILTER (WHERE cat='models')
   + (CASE WHEN COUNT(*) FILTER (WHERE cat='area') > 0 THEN 1 ELSE 0 END)
   + (CASE WHEN COUNT(*) FILTER (WHERE cat='discipline') > 0 THEN 1 ELSE 0 END)
  ) DESC,
  m.name ASC
LIMIT :limit;"""

def print_faculty_matches_for_all_grants(limit_per_grant: int = 10):
    with Session(engine) as s:
        grants = s.execute(text("""
            SELECT opportunity_id, opportunity_title
            FROM opportunity
            ORDER BY COALESCE(post_date, created_at) DESC NULLS LAST
        """)).mappings().all()

        for g in grants:
            gid, title = g["opportunity_id"], g["opportunity_title"]
            print("="*100)
            print(f"Grant: {title}")
            print("-"*100)

            rows = s.execute(text(SQL), {"opportunity_id": gid, "limit": limit_per_grant}).mappings().all()
            if not rows:
                print("No matching faculty.\n")
                continue

            for r in rows:
                print(f"Faculty: {r['name'] or '(no name)'}  <{r['email'] or 'no-email'}>")

                counts = {
                    "application_domain": r["hits_application_domain"],
                    "research_area": r["hits_research_area"],
                    "methods": r["hits_methods"],
                    "models": r["hits_models"],
                    "area": r["hits_area"],
                    "discipline": r["hits_discipline"],
                }
                terms = {
                    "application_domain": r["match_application_domain"] or [],
                    "research_area": r["match_research_area"] or [],
                    "methods": r["match_methods"] or [],
                    "models": r["match_models"] or [],
                    "area": r["match_area"] or [],
                    "discipline": r["match_discipline"] or [],
                }

                print("Matching count from each category:")
                for cat, c in counts.items():
                    if c:
                        print(f"  - {cat}: {c}")

                print("Matching keywords from each category:")
                for cat, arr in terms.items():
                    if arr:
                        print(f"  - {cat}: {', '.join(arr)}")
                print()  # blank line

if __name__ == "__main__":
    print_faculty_matches_for_all_grants(limit_per_grant=10)