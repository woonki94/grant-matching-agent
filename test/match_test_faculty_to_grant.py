from sqlalchemy import text
from sqlalchemy.orm import Session
from db.db_conn import engine

SQL = """WITH fk_src AS (
  SELECT fk.keywords
  FROM faculty f
  JOIN faculty_keywords fk ON fk.faculty_id = f.id
  WHERE f.id = :fid
),
-- flatten FACULTY terms by category (supports 6-key object or legacy array → research_area)
f_terms AS (
  -- object: arrays
  SELECT 'application_domain'::text AS cat, lower(v) AS kw
  FROM fk_src
  CROSS JOIN LATERAL jsonb_array_elements_text(keywords->'application_domain') AS v

  UNION ALL
  SELECT 'research_area', lower(v)
  FROM fk_src
  CROSS JOIN LATERAL jsonb_array_elements_text(keywords->'research_area') AS v

  UNION ALL
  SELECT 'methods', lower(v)
  FROM fk_src
  CROSS JOIN LATERAL jsonb_array_elements_text(keywords->'methods') AS v

  UNION ALL
  SELECT 'models', lower(v)
  FROM fk_src
  CROSS JOIN LATERAL jsonb_array_elements_text(keywords->'models') AS v

  UNION ALL
  -- object: scalars
  SELECT 'area', lower(keywords->>'area')
  FROM fk_src
  WHERE coalesce(keywords->>'area','') <> ''

  UNION ALL
  SELECT 'discipline', lower(keywords->>'discipline')
  FROM fk_src
  WHERE coalesce(keywords->>'discipline','') <> ''

  UNION ALL
  -- legacy: flat array -> treat as research_area
  SELECT 'research_area', lower(v)
  FROM fk_src
  CROSS JOIN LATERAL jsonb_array_elements_text(keywords) AS v
  WHERE jsonb_typeof(keywords)='array'
),
-- flatten GRANT terms by category (6-key object)
g_terms AS (
  SELECT o.opportunity_id, o.opportunity_title, 'application_domain'::text AS cat, lower(v) AS kw
  FROM keywords k
  JOIN opportunity o ON o.opportunity_id=k.opportunity_id
  CROSS JOIN LATERAL jsonb_array_elements_text(k.keywords->'application_domain') AS v

  UNION ALL
  SELECT o.opportunity_id, o.opportunity_title, 'research_area', lower(v)
  FROM keywords k
  JOIN opportunity o ON o.opportunity_id=k.opportunity_id
  CROSS JOIN LATERAL jsonb_array_elements_text(k.keywords->'research_area') AS v

  UNION ALL
  SELECT o.opportunity_id, o.opportunity_title, 'methods', lower(v)
  FROM keywords k
  JOIN opportunity o ON o.opportunity_id=k.opportunity_id
  CROSS JOIN LATERAL jsonb_array_elements_text(k.keywords->'methods') AS v

  UNION ALL
  SELECT o.opportunity_id, o.opportunity_title, 'models', lower(v)
  FROM keywords k
  JOIN opportunity o ON o.opportunity_id=k.opportunity_id
  CROSS JOIN LATERAL jsonb_array_elements_text(k.keywords->'models') AS v

  UNION ALL
  SELECT o.opportunity_id, o.opportunity_title, 'area', lower(k.keywords->>'area')
  FROM keywords k
  JOIN opportunity o ON o.opportunity_id=k.opportunity_id
  WHERE coalesce(k.keywords->>'area','') <> ''

  UNION ALL
  SELECT o.opportunity_id, o.opportunity_title, 'discipline', lower(k.keywords->>'discipline')
  FROM keywords k
  JOIN opportunity o ON o.opportunity_id=k.opportunity_id
  WHERE coalesce(k.keywords->>'discipline','') <> ''
),
-- match within SAME category only
m AS (
  SELECT g.opportunity_id, g.opportunity_title, g.cat, g.kw
  FROM g_terms g
  JOIN f_terms f ON f.cat = g.cat AND f.kw = g.kw
)
SELECT
  opportunity_title,
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
GROUP BY opportunity_title
ORDER BY
  (  COUNT(*) FILTER (WHERE cat='application_domain')
   + COUNT(*) FILTER (WHERE cat='research_area')
   + COUNT(*) FILTER (WHERE cat='methods')
   + COUNT(*) FILTER (WHERE cat='models')
   + (CASE WHEN COUNT(*) FILTER (WHERE cat='area') > 0 THEN 1 ELSE 0 END)
   + (CASE WHEN COUNT(*) FILTER (WHERE cat='discipline') > 0 THEN 1 ELSE 0 END)
  ) DESC,
  opportunity_title ASC
LIMIT :limit;"""

def print_matches_for_all_faculty(limit_per_faculty: int = 20):
    with Session(engine) as s:
        faculties = s.execute(text("SELECT id, COALESCE(name,'') AS name, COALESCE(email,'') AS email FROM faculty ORDER BY id")).mappings().all()
        for fac in faculties:
            fid, name, email = fac["id"], fac["name"], fac["email"]
            print("="*80)
            print(f"Faculty: {name or '(no name)'}  <{email or 'no-email'}>")
            print("-"*80)

            rows = s.execute(text(SQL), {"fid": fid, "limit": limit_per_faculty}).mappings().all()
            if not rows:
                print("No matching grants.\n")
                continue

            for r in rows:
                print(f"title: {r['opportunity_title']}")
                # Matching counts by category
                counts = {
                    "application_domain": r["hits_application_domain"],
                    "research_area": r["hits_research_area"],
                    "methods": r["hits_methods"],
                    "models": r["hits_models"],
                    "area": r["hits_area"],
                    "discipline": r["hits_discipline"],
                }
                # Matching keywords by category
                terms = {
                    "application_domain": r["match_application_domain"] or [],
                    "research_area": r["match_research_area"] or [],
                    "methods": r["match_methods"] or [],
                    "models": r["match_models"] or [],
                    "area": r["match_area"] or [],
                    "discipline": r["match_discipline"] or [],
                }

                # print only non-empty categories
                print("Matching count from each category:")
                for cat, c in counts.items():
                    if c:
                        print(f"  - {cat}: {c}")

                print("Matching keywords from each category:")
                for cat, arr in terms.items():
                    if arr:
                        print(f"  - {cat}: {', '.join(arr)}")
                print()  # blank line between grants

if __name__ == "__main__":
    print_matches_for_all_faculty(limit_per_faculty=5)