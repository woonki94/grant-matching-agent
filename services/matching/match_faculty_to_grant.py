# services/matching/match_grants_to_faculty.py
from __future__ import annotations

import os
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer

from db.db_conn import SessionLocal
import db.models.faculty as mf
import db.models.keywords_faculty as mfk
import db.models.grant as mg
import db.models.keywords_grant as mgk

# ─────────────────────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────────────────────
env_path = Path(__file__).resolve().parents[2] / "api.env"
load_dotenv(dotenv_path=env_path, override=True)
openai_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_key)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _flatten_keywords(k: Dict[str, Any]) -> str:
    """Flatten nested keyword JSON into a single space-joined string."""
    terms = []
    for section in k.values():
        for sublist in section.values():
            terms.extend(sublist)
    return " ".join(terms)

def _get_embedding_with_openai(text: str) -> np.ndarray:
    """Get embedding for text using OpenAI embedding model."""
    if not text.strip():
        return np.zeros(1536)
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return np.array(response.data[0].embedding)

def _get_embedding_with_ST(text: str) -> np.ndarray:
    """ Returns embedding for text using Sentence Transformer. """
    if not text.strip():
        return np.zeros(768)
    
    model = SentenceTransformer("all-mpnet-base-v2")
    embedding = model.encode(text)

    return embedding

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)

# ─────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────
def match_grants_to_faculty(db: Session, *, batch_size: int = 20, top_k: int = 5) -> List[Dict[str, Any]]:
    
    #Computes top-K matching faculty for each grant in the batch.
   
    # Load grants & their keywords
    grants_query = (
        select(mg.Opportunity.opportunity_id, mg.Opportunity.opportunity_title, mgk.Keyword.keywords)
        .join(mgk.Keyword, mgk.Keyword.opportunity_id == mg.Opportunity.opportunity_id)
        #.order_by(mg.Opportunity.opportunity_id.asc())
        .limit(batch_size)
    )
    grants = db.execute(grants_query).all()

    # Load all faculty & their keywords
    faculty_query = (
        select(mf.Faculty.id, mf.Faculty.name, mfk.FacultyKeyword.keywords)
        .join(mfk.FacultyKeyword, mf.Faculty.id == mfk.FacultyKeyword.faculty_id)
    )
    faculty = db.execute(faculty_query).all()

    print(f"Loaded {len(grants)} grants and {len(faculty)} faculty")

    # compute faculty embeddings
    # Should eventually precompute these in a separate file and store them in db?
    faculty_embs = []
    for fid, name, kw in faculty:
        text = _flatten_keywords(kw or {})
        emb = _get_embedding_with_openai(text)
        faculty_embs.append((fid, name, emb))

    results: List[Dict[str, Any]] = []

    # Compute top-K matches for each grant
    for gid, title, gkw in grants:
        gtext = _flatten_keywords(gkw or {})
        gemb = _get_embedding_with_openai(gtext)

        scores = []
        for fid, name, femb in faculty_embs:
            score = _cosine_similarity(gemb, femb)
            scores.append((fid, name, score))

        top = sorted(scores, key=lambda x: x[2], reverse=True)[:top_k]

        results.append(
            {
                "opportunity_id": gid,
                "title": title,
                "matches": [
                    {"faculty_id": fid, "name": name, "score": score}
                    for fid, name, score in top
                ],
            }
        )

    return results


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m services.matching.match_grants_to_faculty <batch_size> <top_k>")
        sys.exit(1)

    batch_size = int(sys.argv[1])
    top_k = int(sys.argv[2])

    with SessionLocal() as db:
        report = match_grants_to_faculty(db, batch_size=batch_size, top_k=top_k)
        print(json.dumps(report, indent=2))
