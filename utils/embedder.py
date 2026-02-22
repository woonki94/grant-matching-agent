from __future__ import annotations
import numpy as np
from typing import List, Optional

from config import get_embedding_client


def embed_texts(texts: List[str]) -> np.ndarray:
    clean = [(t or "").strip() for t in texts]
    if not clean:
        return np.zeros((0, 0), dtype=np.float32)

    embedding_client = get_embedding_client().build()
    vecs = embedding_client.embed_documents(clean)

    return np.array(vecs, dtype=np.float32)

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A @ B.T


def embed_domain_bucket(domains: List[str]) -> Optional[List[float]]:
    """Embed joined domain labels using the configured embedding client."""
    domains = [d.strip() for d in (domains or []) if d and str(d).strip()]
    if not domains:
        return None
    emb = get_embedding_client().build()
    text = " ; ".join(domains)
    vec = emb.embed_query(text)
    return vec if vec else None
