"""
Two-Stage Matching:
1. Filter with Cosine Similarity (Domain-level semantic filtering)
2. Rerank with GPT (Specialization-aware reasoning)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json
import os

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from config import OPENAI_MODEL, OPENAI_API_KEY
from db.models.keywords_opportunity import Keyword as GrantKeyword
from db.models.keywords_faculty import FacultyKeyword
from db.db_conn import SessionLocal
from util.qwen_embeddings import embed_texts, cosine_sim_matrix
from openai import OpenAI
from dao.match_result import MatchResultDAO



# -----------------------------------------------------------
#  Extraction Helpers
# -----------------------------------------------------------

def extract_domains(kw: Dict) -> List[str]:
    """Extract research & application domains as a flat, deduped list."""
    out = []
    for top in ("research", "application"):
        out.extend(kw.get(top, {}).get("domain", []) or [])
    return list(dict.fromkeys([t.strip() for t in out if t.strip()]))


def extract_specializations(kw: Dict) -> List[str]:
    """Extract research & application specializations as a flat, deduped list."""
    out = []
    for top in ("research", "application"):
        out.extend(kw.get(top, {}).get("specialization", []) or [])
    return list(dict.fromkeys([t.strip() for t in out if t.strip()]))


# -----------------------------------------------------------
#  Stage 1 — Fast Domain-Level Filter
# -----------------------------------------------------------

def rank_by_domains_only(
    faculty_rows: List[Tuple[int, Dict]],
    grant_kw: Dict,
    top_n: int = 20
) -> List[Tuple[int, float]]:
    """
    faculty_rows: List[(faculty_id, keyword_dict)]
    Return: Top-N faculty sorted by domain cosine similarity.
    """

    g_domains = extract_domains(grant_kw)
    if not g_domains:
        return []

    # Embed opportunity domains ONCE
    G = embed_texts(g_domains)

    faculty_scores = []

    for fid, f_kw in faculty_rows:
        f_domains = extract_domains(f_kw)

        if not f_domains:
            faculty_scores.append((fid, 0.0))
            continue

        F = embed_texts(f_domains)
        S = cosine_sim_matrix(F, G)

        score = float(S.max()) if S.size > 0 else 0.0
        faculty_scores.append((fid, score))

    faculty_scores.sort(key=lambda x: x[1], reverse=True)
    return faculty_scores[:top_n]


# -----------------------------------------------------------
#  Stage 2 — GPT Reranker
# -----------------------------------------------------------

def build_llm_prompt(data: Dict) -> str:
    """
    Creates the prompt given specialization + domain context.
    """
    return f"""
You are evaluating how well a faculty member’s research matches a grant.

Grant Domains: {data["grant_domains"]}
Grant Specializations: {data["grant_specializations"]}

Faculty Domains: {data["faculty_domains"]}
Faculty Specializations: {data["faculty_specializations"]}

Initial Domain Similarity Score: {data["domain_score"]}

Tasks:
1. Do the domains meaningfully align?
2. Do the specializations reinforce or weaken the match?
3. Provide a refined relevance score from 0 to 1.
4. Provide a 1–2 sentence explanation.

Respond ONLY in JSON using this exact schema:

{{
  "score": <float>,
  "reason": "<short explanation>"
}}
    """.strip()

def gpt_rerank_single(f_id: int, f_kw: Dict, g_kw: Dict, domain_score: float) -> Dict:

    client = OpenAI(api_key=OPENAI_API_KEY)

    fac_dom = extract_domains(f_kw)
    fac_spec = extract_specializations(f_kw)
    g_dom = extract_domains(g_kw)
    g_spec = extract_specializations(g_kw)

    payload = {
        "grant_domains": g_dom,
        "grant_specializations": g_spec,
        "faculty_domains": fac_dom,
        "faculty_specializations": fac_spec,
        "domain_score": domain_score
    }

    prompt = build_llm_prompt(payload)

    # --- FIXED: Correct OpenAI API usage ---
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )

    # Extract the text from the response
    text = resp.output_text

    # The model should return JSON, so parse it
    try:
        parsed = json.loads(text)
        score = float(parsed.get("score", 0.0))
        reason = parsed.get("reason", "").strip()
    except Exception as e:
        score = 0.0
        reason = f"LLM parsing error: {e}"

    return {
        "faculty_id": f_id,
        "llm_score": score,
        "reason": reason,
        "domain_score": domain_score
    }


def rerank_with_gpt(
    faculty_rows: List[Tuple[int, Dict]],
    grant_kw: Dict,
    domain_ranked_list: List[Tuple[int, float]]
) -> List[Dict]:
    """
    Takes top faculty from domain filter and returns GPT-reranked results.
    """
    # Convert faculty_rows to {id -> kw} for fast lookup
    kw_by_id = {fid: kw for (fid, kw) in faculty_rows}

    results = []
    for fid, domain_score in domain_ranked_list:
        f_kw = kw_by_id[fid]
        out = gpt_rerank_single(fid, f_kw, grant_kw, domain_score)
        results.append(out)

    # Sort by llm_score
    results.sort(key=lambda x: x["llm_score"], reverse=True)
    return results


# -----------------------------------------------------------
#  End-to-End Entry Point
# -----------------------------------------------------------

def match_grant_to_faculty(faculty_rows: List[Tuple[int, Dict]], grant_kw: Dict, top_k: int = 20):
    """
    Full two-stage matching:
      1. Domain cosine similarity filter
      2. GPT reranking using specializations
    """

    # Stage 1
    domain_ranked = rank_by_domains_only(faculty_rows, grant_kw, top_n=top_k)

    # Stage 2
    final_ranks = rerank_with_gpt(faculty_rows, grant_kw, domain_ranked)

    return {
        "domain_filter_results": domain_ranked,
        "final_reranked_results": final_ranks
    }


def load_all_grant_keywords(db: Session):
    """
    Returns: List[(grant_id, keyword_dict)]
    """
    rows = db.query(GrantKeyword.opportunity_id, GrantKeyword.keywords).all()
    result = []
    for gid, kw in rows:
        if kw:
            result.append((gid, kw))
    return result


def load_all_faculty_keywords(db: Session):
    """
    Returns: List[(faculty_id, keyword_dict)]
    """
    rows = db.query(FacultyKeyword.faculty_id, FacultyKeyword.keywords).all()
    result = []
    for fid, kw in rows:
        if kw:
            result.append((fid, kw))
    return result


def run_matching_for_all_grants():
    all_results = {}

    with SessionLocal() as db:
        # Load everything ONCE
        all_grants = load_all_grant_keywords(db)
        faculty_rows = load_all_faculty_keywords(db)

        print(f"Loaded {len(all_grants)} grants")
        print(f"Loaded {len(faculty_rows)} faculty profiles")

        for gid, g_kw in all_grants:
            print(f"\n=== Matching grant {gid} ===")
            result = match_grant_to_faculty(
                faculty_rows=faculty_rows,
                grant_kw=g_kw,
                top_k=10
            )
            all_results[gid] = result

            print(json.dumps(result, indent=2))

    # OPTIONAL: Save to file
    '''
    with open("all_grant_matches.json", "w") as f:
        json.dump(all_results, f, indent=2)
    '''
    print("\nDone. Results saved to all_grant_matches.json")
    return all_results



def run_matching_for_partial_grants(num_grant=5, num_faculty=10):
    all_results = {}

    with SessionLocal() as db:
        # Load everything ONCE
        all_grants = load_all_grant_keywords(db)[:num_grant]
        faculty_rows = load_all_faculty_keywords(db)[:num_faculty]

        print(f"Loaded {len(all_grants)} grants")
        print(f"Loaded {len(faculty_rows)} faculty profiles")

        for gid, g_kw in all_grants:
            print(f"\n=== Matching grant {gid} ===")

            # Run 2-stage matching
            result = match_grant_to_faculty(
                faculty_rows=faculty_rows,
                grant_kw=g_kw,
                top_k=10
            )
            all_results[gid] = result

            # -----------------------------
            # SAVE RESULTS TO DATABASE
            # -----------------------------
            final_rows = result.get("final_reranked_results", [])
            for row in final_rows:
                MatchResultDAO.save_match_result(
                    db=db,
                    grant_id=gid,
                    faculty_id=row["faculty_id"],
                    domain_score=row["domain_score"],
                    llm_score=row["llm_score"],
                    reason=row["reason"]
                )

            # Print result to console
            #print(json.dumps(result, indent=2))

    print("\nDone. Results saved to DB")
    return all_results


def run_matching_for_one_faculty(faculty_id: int, top_k_per_grant: int = 10):
    all_results = {}

    with SessionLocal() as db:
        # Load all grants
        all_grants = load_all_grant_keywords(db)
        print(f"Loaded {len(all_grants)} grants")

        # Load this specific faculty's keywords
        faculty_rows = load_all_faculty_keywords(db)
        faculty_dict = {fid: kw for fid, kw in faculty_rows}

        if faculty_id not in faculty_dict:
            raise ValueError(f"Faculty {faculty_id} has no keyword entry.")

        # Wrap it into the format match_grant_to_faculty expects
        single_faculty_row = [(faculty_id, faculty_dict[faculty_id])]

        print(f"Running match for Faculty {faculty_id}")

        for grant_id, grant_kw in all_grants:
            print(f"\n=== Matching Faculty {faculty_id} with Grant {grant_id} ===")

            # Run the matching (only 1 faculty)
            result = match_grant_to_faculty(
                faculty_rows=single_faculty_row,
                grant_kw=grant_kw,
                top_k=10
            )
            all_results[grant_id] = result

            # Extract row
            final_list = result.get("final_reranked_results", [])
            if final_list:
                row = final_list[0]   # only one faculty
                MatchResultDAO.save_match_result(
                    db=db,
                    grant_id=grant_id,
                    faculty_id=faculty_id,
                    domain_score=row["domain_score"],
                    llm_score=row["llm_score"],
                    reason=row["reason"]
                )

        print("\nDone. Results saved to DB for this faculty.")
        return all_results

if __name__ == "__main__":
    #print(run_matching_for_all_grants())

    #print(run_matching_for_partial_grants())
    print(run_matching_for_one_faculty(13))