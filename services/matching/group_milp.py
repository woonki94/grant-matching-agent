from __future__ import annotations

import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import pulp
from sqlalchemy import select

from db.db_conn import SessionLocal
import db.models.opportunity as mg
import db.models.keywords_opportunity as mgk
import db.models.faculty as mf
import db.models.keywords_faculty as mfk


def extract_terms_with_weights(keyword_obj: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    Your keywords JSON looks like:
      {"research":{"domain":[...], "specialization":[...]},
       "application":{"domain":[...], "specialization":[...]}}
    PDF says weights wk exist, but they're not stored.
    So we assign simple heuristic weights:
      specialization=1.0, domain=0.5
    """
    out: List[Tuple[str, float]] = []
    if not isinstance(keyword_obj, dict):
        return out

    for section in keyword_obj.values():  # research/application
        if not isinstance(section, dict):
            continue
        domains = section.get("domain", []) or []
        specs = section.get("specialization", []) or []
        for t in domains:
            out.append((str(t), 0.5))
        for t in specs:
            out.append((str(t), 1.0))

    # de-dup but keep max weight if repeated
    best: Dict[str, float] = {}
    for term, w in out:
        term2 = term.strip().lower()
        if not term2:
            continue
        best[term2] = max(best.get(term2, 0.0), w)
    return [(k, v) for k, v in best.items()]


def faculty_terms(keyword_obj: Dict[str, Any]) -> List[str]:
    items = extract_terms_with_weights(keyword_obj)
    return [t for t, _ in items]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def build_aik(grant_terms: List[str], faculty_term_lists: Dict[int, List[str]]) -> Dict[Tuple[int, str], float]:
    """
    Simple relevance aik:
    aik = 1 if faculty contains term as exact substring match, else 0.
    (Fast + deterministic; you can upgrade to embeddings later.)
    """
    aik: Dict[Tuple[int, str], float] = {}
    for fid, fterms in faculty_term_lists.items():
        fblob = " ".join(fterms)
        for k in grant_terms:
            aik[(fid, k)] = 1.0 if k in fblob else 0.0
    return aik


def solve_team_for_grant(opportunity_id: str, team_size: int) -> Dict[str, Any]:
    with SessionLocal() as db:
        g = db.execute(
            select(mg.Opportunity.opportunity_id, mg.Opportunity.opportunity_title, mgk.Keyword.keywords)
            .join(mgk.Keyword, mgk.Keyword.opportunity_id == mg.Opportunity.opportunity_id)
            .where(mg.Opportunity.opportunity_id == opportunity_id)
        ).first()

        if not g:
            raise SystemExit(f"Grant {opportunity_id} not found or missing keywords row.")

        gid, title, gkw = g
        g_terms_w = extract_terms_with_weights(gkw or {})
        K = [t for t, _ in g_terms_w]
        wk = {t: w for t, w in g_terms_w}

        faculty_rows = db.execute(
            select(mf.Faculty.id, mf.Faculty.name, mfk.FacultyKeyword.keywords)
            .join(mfk.FacultyKeyword, mf.Faculty.id == mfk.FacultyKeyword.faculty_id)
        ).all()

        F = [fid for fid, _, _ in faculty_rows]
        fname = {fid: name for fid, name, _ in faculty_rows}
        f_terms = {fid: faculty_terms(kw or {}) for fid, _, kw in faculty_rows}

        aik = build_aik(K, f_terms)

        # MILP
        prob = pulp.LpProblem("GrantTeam", pulp.LpMaximize)

        x = pulp.LpVariable.dicts("x", F, lowBound=0, upBound=1, cat="Binary")
        y = pulp.LpVariable.dicts("y", K, lowBound=0, upBound=1, cat="Continuous")

        # objective: max sum_k wk * yk
        prob += pulp.lpSum([wk[k] * y[k] for k in K])

        # team size
        prob += pulp.lpSum([x[i] for i in F]) == team_size

        # coverage constraints
        for k in K:
            prob += y[k] <= pulp.lpSum([aik[(i, k)] * x[i] for i in F])
            prob += y[k] <= 1.0

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        chosen = [i for i in F if pulp.value(x[i]) >= 0.5]
        covered = [(k, wk[k], pulp.value(y[k])) for k in K if pulp.value(y[k]) and pulp.value(y[k]) > 1e-6]
        covered.sort(key=lambda z: (z[2], z[1]), reverse=True)

        return {
            "opportunity_id": gid,
            "title": title,
            "team_size": team_size,
            "selected_faculty": [{"id": i, "name": fname[i]} for i in chosen],
            "objective": float(pulp.value(prob.objective) or 0.0),
            "covered_keywords": [{"keyword": k, "weight": w, "coverage": c} for k, w, c in covered],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opportunity-id", required=True)
    ap.add_argument("--team-size", type=int, default=3)
    args = ap.parse_args()

    out = solve_team_for_grant(args.opportunity_id, args.team_size)
    import json
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
