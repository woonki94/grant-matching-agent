from __future__ import annotations
import math
import numpy as np
from sqlalchemy import select
from sentence_transformers import SentenceTransformer

from db.db_conn import SessionLocal
import db.models.opportunity as mg
import db.models.keywords_opportunity as mgk
import db.models.faculty as mf
import db.models.keywords_faculty as mfk
from db.models.match_result import MatchResult

BATCH_GRANTS = 27      # change if you want more
TOP_K = 10             # <-- this is what MILP needs
MODEL_NAME = "all-mpnet-base-v2"

def flatten_keywords(k: dict) -> str:
    # keywords are stored as {"research": {"domain":[...], "specialization":[...]}, "application": {...}}
    if not isinstance(k, dict):
        return ""
    terms = []
    for section in k.values():
        if isinstance(section, dict):
            for sub in section.values():
                if isinstance(sub, list):
                    terms.extend([str(x) for x in sub])
    return " ".join(terms)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)

def main():
    model = SentenceTransformer(MODEL_NAME)

    with SessionLocal() as db:
        grants = db.execute(
            select(mg.Opportunity.opportunity_id, mg.Opportunity.opportunity_title, mgk.Keyword.keywords)
            .join(mgk.Keyword, mgk.Keyword.opportunity_id == mg.Opportunity.opportunity_id)
            .limit(BATCH_GRANTS)
        ).all()

        faculty = db.execute(
            select(mf.Faculty.id, mf.Faculty.name, mfk.FacultyKeyword.keywords)
            .join(mfk.FacultyKeyword, mf.Faculty.id == mfk.FacultyKeyword.faculty_id)
        ).all()

        # Precompute faculty embeddings once
        f_texts = [flatten_keywords(kw or {}) for _, _, kw in faculty]
        f_embs = model.encode(f_texts, normalize_embeddings=True)
        f_ids = [fid for fid, _, _ in faculty]
        f_names = [name for _, name, _ in faculty]

        inserted = 0
        for gid, title, gkw in grants:
            gtext = flatten_keywords(gkw or {})
            gemb = model.encode([gtext], normalize_embeddings=True)[0]

            scores = [(f_ids[i], f_names[i], float(np.dot(gemb, f_embs[i]))) for i in range(len(f_ids))]
            scores.sort(key=lambda x: x[2], reverse=True)
            top = scores[:TOP_K]

            for fid, fname, sc in top:
                db.add(MatchResult(
                    grant_id=gid,
                    faculty_id=fid,
                    domain_score=sc,
                    llm_score=sc,
                    reason="baseline: sentence-transformer cosine similarity (for MILP candidate pool)"
                ))
                inserted += 1

        db.commit()
        print(f"Inserted {inserted} match_results rows ({BATCH_GRANTS} grants × TOP_K={TOP_K})")

if __name__ == "__main__":
    main()
