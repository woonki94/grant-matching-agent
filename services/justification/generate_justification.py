from __future__ import annotations
import os, json
from pathlib import Path

from openai import OpenAI
from sqlalchemy.orm import Session

from db.db_conn import SessionLocal
from db.models.match_result import MatchResult
from db.models.grant import Opportunity   # for grant titles, agency, etc.
from db.models.keywords_grant import Keyword as GrantKeyword
from db.models.keywords_faculty import FacultyKeyword

from services.matching.hybrid_matcher import extract_domains, extract_specializations

# Load API Key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ================================================================
#   1. Query DB: Get all matched grants for a given faculty
# ================================================================
def get_matches_for_faculty(db: Session, faculty_id: int):
    rows = (
        db.query(MatchResult)
          .filter(MatchResult.faculty_id == faculty_id)
          .order_by(MatchResult.llm_score.desc())
          .all()
    )
    return rows

def load_prompt(name: str) -> str:

    base = Path(__file__).resolve().parents[2] / "prompts"
    path = base / f"{name}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    return path.read_text(encoding="utf-8")

# ================================================================
#   2. Build GPT justification prompt
# ================================================================

def build_justification_prompt(db, grant, faculty_obj, match):
    """
    Loads justification template and fills:
    - grant title + agency
    - grant keywords (domain/specialization)
    - faculty keywords (domain/specialization)
    """

    # ------------------------------------------------------
    # 1. LOAD KEYWORDS FROM SEPARATE TABLES
    # ------------------------------------------------------

    # Grant Keyword Row
    g_kw_row = (
        db.query(GrantKeyword)
          .filter(GrantKeyword.opportunity_id == grant.opportunity_id)
          .one_or_none()
    )
    g_kw = g_kw_row.keywords if g_kw_row else {}

    # Faculty Keyword Row
    f_kw_row = (
        db.query(FacultyKeyword)
          .filter(FacultyKeyword.faculty_id == faculty_obj.id)
          .one_or_none()
    )
    f_kw = f_kw_row.keywords if f_kw_row else {}

    # ------------------------------------------------------
    # 2. Extract Domains + Specializations
    # ------------------------------------------------------
    grant_domains = extract_domains(g_kw)
    grant_specs   = extract_specializations(g_kw)

    faculty_domains = extract_domains(f_kw)
    faculty_specs   = extract_specializations(f_kw)

    # ------------------------------------------------------
    # 3. Load Template
    # ------------------------------------------------------
    template = load_prompt("justification_prompt")

    # ------------------------------------------------------
    # 4. Fill Template
    # ------------------------------------------------------
    filled = template.format(
        grant_title=grant.opportunity_title,
        grant_agency=grant.agency_name,
        grant_domains=grant_domains,
        grant_specializations=grant_specs,
        faculty_domains=faculty_domains,
        faculty_specializations=faculty_specs,
    )

    return filled.strip()


# ================================================================
#   3. Ask GPT to generate justification
# ================================================================
def generate_justification(prompt: str) -> str:
    resp = client.responses.create(
        model="gpt-5",
        input=prompt
    )
    return resp.output_text.strip()


# ================================================================
#   4. Main function: recommend grants for faculty
# ================================================================
def recommend_grants_for_faculty(faculty_id: int, top_n: int = 5):
    with SessionLocal() as db:
        matches = get_matches_for_faculty(db, faculty_id)

        if not matches:
            print(f"No matches found for faculty {faculty_id}.")
            return

        # All MatchResult rows have match.faculty loaded via relationship
        faculty_obj = matches[0].faculty
        faculty_name = getattr(faculty_obj, "name", None) or \
                       f"Faculty {faculty_id}"

        matches = matches[:top_n]

        print(f"\n===============================")
        print(f" Grant Recommendations for {faculty_name} (ID: {faculty_id})")
        print(f"===============================\n")

        for match in matches:
            grant = db.get(Opportunity, match.grant_id)

            print(f"Grant: {grant.opportunity_title}")
            print(f"Agency: {grant.agency_name}")
            print(f"Scores → LLM: {match.llm_score:.3f}, Domain: {match.domain_score:.3f}\n")

            # Create GPT justification
            prompt = build_justification_prompt(db, grant, faculty_obj, match)
            justification = generate_justification(prompt)

            print("Justification:")
            print(justification)
            print("\n" + "-"*60 + "\n")


if __name__ == "__main__":
    faculty_id = int(input("Enter faculty ID: "))
    recommend_grants_for_faculty(faculty_id, top_n=5)