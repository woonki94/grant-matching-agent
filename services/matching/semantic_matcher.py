# services/semantic_matcher.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from util.qwen_embeddings import embed_texts, cosine_sim_matrix

def _flatten(kw: Dict) -> List[str]:
    out = []
    for top in ("research", "application"):
        for sub in ("domain", "specialization"):
            out.extend(kw.get(top, {}).get(sub, []) or [])
    # unique preserve order
    seen, uniq = set(), []
    for t in out:
        k = t.strip().lower()
        if k not in seen:
            seen.add(k); uniq.append(t.strip())
    return uniq

def rank_pairs(faculty_kw: Dict, grant_kw: Dict, top_k_pairs: int = 20) -> List[Tuple[str, str, float]]:
    f_terms = _flatten(faculty_kw)
    g_terms = _flatten(grant_kw)
    if not f_terms or not g_terms:
        return []

    F = embed_texts(f_terms)
    G = embed_texts(g_terms)
    S = cosine_sim_matrix(F, G)  # (len(f_terms), len(g_terms))

    pairs = [(f_terms[i], g_terms[j], float(S[i, j]))
             for i in range(len(f_terms)) for j in range(len(g_terms))]
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k_pairs]

def score_faculty_vs_grant(faculty_kw: Dict, grant_kw: Dict, agg: str = "max") -> float:
    """
    Aggregate a single similarity score for (faculty, grant).
    agg = 'max' (best term-to-term similarity) or 'mean_top5' (average of top-5).
    """
    f_terms = _flatten(faculty_kw)
    g_terms = _flatten(grant_kw)
    if not f_terms or not g_terms:
        return 0.0

    F = embed_texts(f_terms)
    G = embed_texts(g_terms)
    S = cosine_sim_matrix(F, G)  # (len(f_terms), len(g_terms))

    # per-faculty-term best match against grant terms
    best_per_f_term = S.max(axis=1) if S.size else np.zeros((len(f_terms),), dtype=np.float32)

    if agg == "max":
        return float(best_per_f_term.max() if best_per_f_term.size else 0.0)
    elif agg == "mean_top5":
        k = min(5, best_per_f_term.size)
        if k == 0:
            return 0.0
        topk = np.partition(best_per_f_term, -k)[-k:]
        return float(topk.mean())
    else:
        # default conservative
        return float(best_per_f_term.max() if best_per_f_term.size else 0.0)
