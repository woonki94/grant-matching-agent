# util/keyword_cleanup.py  (PATCH)
from __future__ import annotations
from typing import Dict, List, Tuple
import re
from util.qwen_embeddings import embed_texts, cosine_sim_matrix  # will do lazy env inside

def _norm(s: str) -> str:
    s = (s or "").strip()
    return re.sub(r"\s+", " ", s).lower()

def _unique(items: List[str]) -> List[str]:
    seen, out = set(), []
    for it in items or []:
        k = _norm(it)
        if k and k not in seen:
            seen.add(k); out.append(it.strip())
    return out

def _semantic_filter(research_specs: List[str], app_specs: List[str], threshold=0.88) -> Tuple[List[str], List[str]]:
    r = _unique(research_specs)
    a = _unique(app_specs)
    if not r or not a:
        return r, a
    try:
        R = embed_texts(r)
        A = embed_texts(a)
        S = cosine_sim_matrix(A, R)
        rset = {_norm(x) for x in r}
        keep = []
        for i, t in enumerate(a):
            literal_dup = _norm(t) in rset
            sem_dup = (S[i].max() if S.size else 0.0) >= threshold
            if not (literal_dup or sem_dup):
                keep.append(t)
        return r, keep
    except Exception:
        # Fallback: literal-only de-dup if embeddings unavailable
        rset = {_norm(x) for x in r}
        keep = [t for t in a if _norm(t) not in rset]
        return r, keep

def enforce_caps_and_limits(payload: Dict, max_items_per_list: int = 10) -> Dict:
    def norm_list(xs: List[str]) -> List[str]:
        xs = [x.strip() for x in (xs or []) if x and x.strip()]
        xs = _unique(xs)
        xs = [x.lower() for x in xs]
        return xs[:max_items_per_list]

    out = {
        "research": {
            "domain": norm_list(payload.get("research", {}).get("domain", [])),
            "specialization": norm_list(payload.get("research", {}).get("specialization", [])),
        },
        "application": {
            "domain": norm_list(payload.get("application", {}).get("domain", [])),
            "specialization": norm_list(payload.get("application", {}).get("specialization", [])),
        },
    }
    r_spec, a_spec = _semantic_filter(out["research"]["specialization"], out["application"]["specialization"])
    out["research"]["specialization"] = r_spec
    out["application"]["specialization"] = a_spec
    return out
