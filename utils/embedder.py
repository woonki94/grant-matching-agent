from __future__ import annotations
import os, time, numpy as np
from typing import List, Optional
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from dotenv import load_dotenv
from pathlib import Path

from config import settings, get_embedding_client

def _retry(fn, n: int = 4, backoff: float = 0.75):
    for i in range(n):
        try:
            return fn()
        except (RateLimitError, APITimeoutError, APIError):
            if i == n - 1:
                raise
            time.sleep(backoff * (2 ** i))

def embed_texts(texts: List[str]) -> np.ndarray:
    clean = [(t or "").strip() for t in texts]
    if not clean:
        return np.zeros((0, 0), dtype=np.float32)

    embedding_client =  get_embedding_client().build()
    vecs = _retry(lambda: embedding_client.embed_documents(clean))

    return np.array(vecs, dtype=np.float32)

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A @ B.T

def _centroid(vecs: np.ndarray) -> Optional[List[float]]:
    if vecs.size == 0:
        return None
    c = vecs.mean(axis=0)
    n = np.linalg.norm(c)
    if n == 0:
        return None
    return (c / n).tolist()

def embed_domain_bucket(domains: List[str]) -> Optional[List[float]]:
    domains = [d.strip() for d in domains if isinstance(d, str) and d.strip()]
    if not domains:
        return None
    vecs = embed_texts(domains)     # (N, D)
    return _centroid(vecs)

def extract_domains(keywords: dict) -> tuple[list[str], list[str]]:
    r = [(x or "").strip() for x in (keywords.get("research") or {}).get("domain", [])]
    a = [(x or "").strip() for x in (keywords.get("application") or {}).get("domain", [])]
    return [x for x in r if x], [x for x in a if x]
