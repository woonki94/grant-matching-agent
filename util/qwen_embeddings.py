# util/qwen_embeddings.py
from __future__ import annotations
import os, time, numpy as np
from typing import List
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from dotenv import load_dotenv
from pathlib import Path

DEFAULT_OR_BASE = "https://openrouter.ai/api/v1"

def _load_env():
    # robust load: repo root api.env, or cwd/api.env
    for p in (Path(__file__).resolve().parents[2] / "api.env", Path.cwd() / "api.env"):
        if p.exists():
            load_dotenv(p, override=False)
            break

def _get_openrouter_client() -> OpenAI:
    _load_env()
    key = os.getenv("OPENROUTER_API_KEY")
    base = os.getenv("OPENROUTER_BASE_URL", DEFAULT_OR_BASE).rstrip("/")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (api.env).")
    client = OpenAI(api_key=key, base_url=base)
    # sanity check – but no over-strict equality
    base_url_str = str(getattr(client, "base_url", "")).lower().rstrip("/")
    if "openrouter.ai" not in base_url_str:
        raise RuntimeError(f"Embeddings client base_url is not OpenRouter: {base_url_str!r}")
    return client

def _get_model() -> str:
    _load_env()
    # allow either name; keep QWEN_EMBED_MODEL for backward compat
    model = (os.getenv("QWEN_EMBED_MODEL") or os.getenv("EMBED_MODEL") or "").strip()
    if not model:
        raise RuntimeError(
            "No embedding model configured. Set QWEN_EMBED_MODEL or EMBED_MODEL "
            "in api.env (e.g., 'qwen/qwen3-embedding-0.6b')."
        )
    # ❌ DO NOT enforce any ':embedding' suffix here – model can be any valid embedding id
    return model

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

    client = _get_openrouter_client()
    model = _get_model()
    res = _retry(lambda: client.embeddings.create(model=model, input=clean))
    vecs = [d.embedding for d in res.data]
    return np.array(vecs, dtype=np.float32)

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A @ B.T
