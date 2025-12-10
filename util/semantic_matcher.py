# services/semantic_matcher.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from util.qwen_embeddings import embed_texts, cosine_sim_matrix

def flatten_keywords(kw: Dict[str, Dict[str, List[str]]]) -> List[str]:
    out = []
    for top in ("research", "application"):
        for sub in ("domain", "specialization"):
            out.extend(kw.get(top, {}).get(sub, []))
    # unique preserve order
    seen = set()
    uniq = []
    for t in out:
        k = t.strip().lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(t.strip())
    return uniq

def rank_faculty_to_grant(
    faculty_keywords: Dict,
    grant_keywords: Dict,
    top_k_pairs: int = 10
) -> List[Tuple[str, str, float]]:
    f_terms = flatten_keywords(faculty_keywords)
    g_terms = flatten_keywords(grant_keywords)
    if not f_terms or not g_terms:
        return []

    F = embed_texts(f_terms)
    G = embed_texts(g_terms)
    S = cosine_sim_matrix(F, G)  # (len(f_terms), len(g_terms))

    pairs = [(f_terms[i], g_terms[j], float(S[i, j]))
             for i in range(len(f_terms)) for j in range(len(g_terms))]
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k_pairs]
