from __future__ import annotations
import os, json
from pathlib import Path

from openai import OpenAI
from sqlalchemy.orm import Session

from config import OPENAI_API_KEY, OPENAI_MODEL
from db.db_conn import SessionLocal

from db.models.match_result import MatchResult
from db.models.opportunity import Opportunity
from db.models.keywords_opportunity import Keyword as GrantKeyword
from db.models.keywords_faculty import FacultyKeyword
from db.models.faculty import Faculty, FacultyPublication

from services.matching.hybrid_matcher import extract_domains, extract_specializations


# ------------------------------------------------------------
#  Utility: load external prompt template file
# ------------------------------------------------------------
def load_prompt(name: str) -> str:
    base = Path(__file__).resolve().parents[2] / "prompts"
    path = base / f"{name}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    return path.read_text(encoding="utf-8")

# ------------------------------------------------------------
#  Retrieve GRANT context for RAG
# ------------------------------------------------------------
def get_grant_context(db: Session, grant_id: str, max_chars=2000):
    grant = db.get(Opportunity, grant_id)
    if not grant:
        return ""

    parts = []

    # 1. Summary description
    if grant.summary_description:
        parts.append(f"[Grant Summary]\n{grant.summary_description.strip()}")

    # 2. First attachment text
    if grant.attachments:
        for att in grant.attachments:
            if att.content:
                parts.append(f"[Grant Attachment Excerpt]\n{att.content[:max_chars]}")
                break

    # 3. Keywords (optional but helpful)
    if grant.keyword and grant.keyword.keywords:
        parts.append(f"[Grant Keywords]\n{json.dumps(grant.keyword.keywords, indent=2)}")

    return "\n\n".join(parts).strip()

# ------------------------------------------------------------
#  Retrieve FACULTY context for RAG
# ------------------------------------------------------------
def get_faculty_context(db: Session, faculty_id: int, max_chars=1500):
    fac = db.get(Faculty, faculty_id)
    if not fac:
        return ""

    parts = []

    # 1. Biography
    if fac.biography:
        parts.append(f"[Faculty Biography]\n{fac.biography.strip()}")

    # 2. Expertise terms
    if fac.expertise:
        exp_terms = [e.term for e in fac.expertise]
        parts.append(f"[Faculty Expertise Terms]\n{exp_terms}")

    # 3. Keyword schema
    if fac.keyword and fac.keyword.keywords:
        parts.append(f"[Faculty Keywords]\n{json.dumps(fac.keyword.keywords, indent=2)}")

    # 4. Publication abstracts (top 2)
    pubs = (
        db.query(FacultyPublication)
          .filter(FacultyPublication.faculty_id == faculty_id)
          .limit(2)
          .all()
    )
    for pub in pubs:
        if pub.abstract:
            parts.append(f"[Publication Abstract]\n{pub.abstract[:max_chars]}")

    return "\n\n".join(parts).strip()

# ------------------------------------------------------------
#  RAG Justification Prompt Builder
# ------------------------------------------------------------
def build_justification_prompt(db, grant, faculty_obj, match):
    # ---- Retrieve RAG context ----
    grant_context   = get_grant_context(db, grant.opportunity_id)
    faculty_context = get_faculty_context(db, faculty_obj.id)

    # ---- Load structured keywords ----
    g_kw = (db.query(GrantKeyword)
              .filter(GrantKeyword.opportunity_id == grant.opportunity_id)
              .one_or_none())
    f_kw = (db.query(FacultyKeyword)
              .filter(FacultyKeyword.faculty_id == faculty_obj.id)
              .one_or_none())

    grant_domains = extract_domains(g_kw.keywords if g_kw else {})
    grant_specs   = extract_specializations(g_kw.keywords if g_kw else {})
    grant_link = grant.additional_info_url or "N/A",

    faculty_domains = extract_domains(f_kw.keywords if f_kw else {})
    faculty_specs   = extract_specializations(f_kw.keywords if f_kw else {})

    # ---- Load template file ----
    template = load_prompt("justification_prompt")  # << new prompt file

    # ---- Fill template ----
    filled = template.format(
        grant_title=grant.opportunity_title,
        grant_agency=grant.agency_name,
        grant_link=grant_link or "N/A",
        grant_context=grant_context,
        faculty_context=faculty_context,
        grant_domains=grant_domains,
        grant_specs=grant_specs,
        faculty_domains=faculty_domains,
        faculty_specs=faculty_specs,
        domain_score=f"{match.domain_score:.3f}",
        llm_score=f"{match.llm_score:.3f}",
    )

    return filled.strip()

# ------------------------------------------------------------
#  GPT Call
# ------------------------------------------------------------
def generate_justification(prompt: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt
    )
    return resp.output_text.strip()

# ------------------------------------------------------------
#  Main Recommendation Function
# ------------------------------------------------------------
def recommend_grants_for_faculty(faculty_id: int, top_n: int = 5):
    with SessionLocal() as db:
        matches = (
            db.query(MatchResult)
              .filter(MatchResult.faculty_id == faculty_id)
              .order_by(MatchResult.llm_score.desc())
              .all()
        )

        if not matches:
            print(f"No matches found for faculty {faculty_id}.")
            return

        faculty_obj = matches[0].faculty
        faculty_name = faculty_obj.name or f"Faculty {faculty_id}"

        matches = matches[:top_n]

        print(f"\n===============================")
        print(f" Grant Recommendations for {faculty_name} (ID: {faculty_id})")
        print(f"===============================\n")

        for match in matches:
            grant = db.get(Opportunity, match.grant_id)

            print(f"Grant: {grant.opportunity_title}")
            print(f"Agency: {grant.agency_name}")
            print(f"Scores → LLM: {match.llm_score:.3f}, Domain: {match.domain_score:.3f}\n")

            # ---- RAG Justification ----
            prompt = build_justification_prompt(db, grant, faculty_obj, match)
            justification = generate_justification(prompt)

            print("Justification:")
            print(justification)
            print("\n" + "-"*60 + "\n")


if __name__ == "__main__":
    faculty_id = int(input("Enter faculty ID: "))
    recommend_grants_for_faculty(faculty_id, top_n=5)