from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from utils.embedder import cosine_sim_matrix, embed_texts


def _parse_s3_bucket_key(content_path: str) -> Tuple[str, str]:
    cp = (content_path or "").strip()
    if not cp:
        raise ValueError("empty content_path")
    if cp.startswith("s3://"):
        rest = cp[5:]
        parts = rest.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"invalid s3 uri: {content_path}")
        return parts[0], parts[1]
    return settings.extracted_content_bucket, cp.lstrip("/")


def _get_s3_client():
    session = (
        boto3.Session(profile_name=settings.aws_profile, region_name=settings.aws_region)
        if settings.aws_profile
        else boto3.Session(region_name=settings.aws_region)
    )
    return session.client("s3")


def _read_s3_text(content_path: str) -> str:
    bucket, key = _parse_s3_bucket_key(content_path)
    s3 = _get_s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8", errors="ignore").strip()


def _chunk_text(text: str, words_per_chunk: int = 220) -> List[str]:
    words = [w for w in (text or "").split() if w]
    if not words:
        return []
    chunks: List[str] = []
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i : i + words_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _extract_keyword_phrases(keywords: Dict, include_domains: bool = False) -> List[str]:
    phrases: List[str] = []
    for sec in ("research", "application"):
        sec_obj = keywords.get(sec) if isinstance(keywords, dict) else None
        if not isinstance(sec_obj, dict):
            continue

        specs = sec_obj.get("specialization")
        if isinstance(specs, list):
            for item in specs:
                if isinstance(item, dict):
                    txt = str(item.get("t") or "").strip()
                else:
                    txt = str(item).strip()
                if txt:
                    phrases.append(txt)

        if include_domains:
            domains = sec_obj.get("domain")
            if isinstance(domains, list):
                for d in domains:
                    d_txt = str(d).strip()
                    if d_txt:
                        phrases.append(d_txt)

    deduped: List[str] = []
    seen = set()
    for p in phrases:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def _build_grant_context_texts(opp, max_docs: int = 12, max_chars_per_doc: int = 10000) -> List[str]:
    texts: List[str] = []

    title = str(getattr(opp, "opportunity_title", "") or "").strip()
    summary = str(getattr(opp, "summary_description", "") or "").strip()
    if title or summary:
        texts.append(f"{title}\n\n{summary}".strip())

    # attachments
    attachments = list(getattr(opp, "attachments", []) or [])
    for a in attachments[:max_docs]:
        if getattr(a, "extract_status", None) != "done":
            continue
        if getattr(a, "extract_error", None):
            continue
        cp = getattr(a, "content_path", None)
        if not cp:
            continue
        try:
            doc = _read_s3_text(str(cp))
            if doc:
                texts.append(doc[:max_chars_per_doc])
        except Exception:
            continue

    # additional info
    infos = list(getattr(opp, "additional_info", []) or [])
    for ai in infos[:max_docs]:
        if getattr(ai, "extract_status", None) != "done":
            continue
        if getattr(ai, "extract_error", None):
            continue
        cp = getattr(ai, "content_path", None)
        if not cp:
            continue
        try:
            doc = _read_s3_text(str(cp))
            if doc:
                texts.append(doc[:max_chars_per_doc])
        except Exception:
            continue

    return texts


def _percent(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return (100.0 * n) / float(d)

def _build_redundancy_clusters(sim: np.ndarray, threshold: float) -> List[List[int]]:
    n = sim.shape[0]
    visited = [False] * n
    clusters: List[List[int]] = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp: List[int] = []
        visited[i] = True
        while stack:
            cur = stack.pop()
            comp.append(cur)
            nbrs = np.where(sim[cur] >= threshold)[0].tolist()
            for nb in nbrs:
                if nb == cur:
                    continue
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(comp) > 1:
            clusters.append(sorted(comp))
    return clusters


def run_keyword_quality_for_opp(
    *,
    opp_id: str,
    include_domains: bool = False,
    words_per_chunk: int = 220,
) -> None:
    with SessionLocal() as sess:
        odao = OpportunityDAO(sess)
        rows = odao.read_opportunities_by_ids_with_relations([opp_id])
        if not rows:
            raise ValueError(f"Opportunity not found: {opp_id}")
        opp = rows[0]

        keywords = (getattr(getattr(opp, "keyword", None), "keywords", {}) or {})
        phrases = _extract_keyword_phrases(keywords, include_domains=include_domains)
        if not phrases:
            raise ValueError("No keyword phrases found for this opportunity.")

        context_docs = _build_grant_context_texts(opp)
        if not context_docs:
            # fallback to title+summary only if extracted docs are unavailable
            title = str(getattr(opp, "opportunity_title", "") or "").strip()
            summary = str(getattr(opp, "summary_description", "") or "").strip()
            fallback = f"{title}\n\n{summary}".strip()
            if not fallback:
                raise ValueError("No grant context found (summary/extracted docs missing).")
            context_docs = [fallback]

        chunks: List[str] = []
        for doc in context_docs:
            chunks.extend(_chunk_text(doc, words_per_chunk=words_per_chunk))
        if not chunks:
            raise ValueError("Context chunking produced zero chunks.")

        phrase_emb = embed_texts(phrases)
        chunk_emb = embed_texts(chunks)

        rel = cosine_sim_matrix(phrase_emb, chunk_emb)  # (P, C)
        max_rel = rel.max(axis=1) if rel.size else np.array([], dtype=np.float32)

        red = cosine_sim_matrix(phrase_emb, phrase_emb)  # (P, P)
        np.fill_diagonal(red, -1.0)
        upper = red[np.triu_indices(red.shape[0], k=1)] if red.size else np.array([], dtype=np.float32)

        n = len(phrases)
        strong = int((max_rel >= 0.75).sum())
        medium = int(((max_rel >= 0.50) & (max_rel < 0.75)).sum())
        weak = int(((max_rel >= 0.35) & (max_rel < 0.50)).sum())
        off = int((max_rel < 0.35).sum())

        dup_pairs = int((upper > 0.85).sum()) if upper.size else 0
        near_dup_pairs = int(((upper > 0.65) & (upper <= 0.85)).sum()) if upper.size else 0

        low_idx = np.argsort(max_rel)[: min(5, n)]

        print("")
        print("=== Keyword Quality Report (Single Grant) ===")
        print(f"Opportunity ID: {opp_id}")
        print(f"Title: {getattr(opp, 'opportunity_title', None)}")
        print(f"Keyword phrases analyzed: {n}")
        print(f"Context documents used: {len(context_docs)}")
        print(f"Context chunks used: {len(chunks)}")
        print("")
        print("Relevance (keyword phrase -> grant context chunks, max cosine)")
        print(f"- strong (>= 0.75): {strong} ({_percent(strong, n):.1f}%)")
        print(f"- medium (0.50 - 0.75): {medium} ({_percent(medium, n):.1f}%)")
        print(f"- weak (0.35 - 0.50): {weak} ({_percent(weak, n):.1f}%)")
        print(f"- off-topic (< 0.35): {off} ({_percent(off, n):.1f}%)")
        print(f"- avg relevance: {float(max_rel.mean()):.4f}")
        print(f"- min relevance: {float(max_rel.min()):.4f}")
        print(f"- max relevance: {float(max_rel.max()):.4f}")
        print("")
        print("Redundancy (keyword phrase <-> keyword phrase cosine)")
        print(f"- duplicate pairs (> 0.85): {dup_pairs}")
        print(f"- near-duplicate pairs (0.65 - 0.85): {near_dup_pairs}")
        if upper.size:
            print(f"- avg pairwise similarity: {float(upper.mean()):.4f}")
        print("")
        print("Redundant keyword sets:")
        if red.size:
            pairs: List[Tuple[float, int, int]] = []
            for i in range(n):
                for j in range(i + 1, n):
                    s = float(red[i, j])
                    if s > 0.75:
                        pairs.append((s, i, j))
            pairs.sort(key=lambda x: x[0], reverse=True)

            if pairs:
                print("- Top redundant pairs:")
                for s, i, j in pairs[:8]:
                    print(f"  - ({s:.4f}) \"{phrases[i]}\"  <->  \"{phrases[j]}\"")
            else:
                print("- No high-redundancy pairs above 0.65.")

            clusters = _build_redundancy_clusters(red, threshold=0.65)
            if clusters:
                print("- Redundancy clusters (threshold 0.65):")
                for c_idx, cluster in enumerate(clusters[:6], start=1):
                    print(f"  - Cluster {c_idx}:")
                    for idx in cluster:
                        print(f"    - {phrases[idx]}")
            else:
                print("- No redundancy clusters found.")
        else:
            print("- Not enough keyword phrases to assess redundancy sets.")
        print("")

        print("Lowest-relevance phrases (top 5):")
        for i in low_idx:
            print(f"- {phrases[int(i)]}  | score={float(max_rel[int(i)]):.4f}")
        print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Measure keyword quality for one grant opportunity.")
    parser.add_argument("--opp-id", required=True, help="Opportunity ID to evaluate.")
    parser.add_argument(
        "--include-domains",
        action="store_true",
        help="Include domain keywords in addition to specialization keywords.",
    )
    parser.add_argument(
        "--words-per-chunk",
        type=int,
        default=220,
        help="Approximate words per context chunk (default: 220).",
    )
    args = parser.parse_args()

    run_keyword_quality_for_opp(
        opp_id=args.opp_id,
        include_domains=args.include_domains,
        words_per_chunk=args.words_per_chunk,
    )


