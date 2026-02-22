from __future__ import annotations

import argparse
import json
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


# ----------------------------
# S3 Helpers (UNCHANGED)
# ----------------------------

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


# ----------------------------
# Keyword Extraction (FROM JSON)
# ----------------------------

def _extract_keyword_phrases(keywords: Dict, include_domains: bool = False) -> List[str]:
    phrases: List[str] = []

    for sec in ("research", "application"):
        sec_obj = keywords.get(sec)
        if not isinstance(sec_obj, dict):
            continue

        specs = sec_obj.get("specialization", [])
        for item in specs:
            txt = str(item).strip()
            if txt:
                phrases.append(txt)

        if include_domains:
            domains = sec_obj.get("domain", [])
            for d in domains:
                d_txt = str(d).strip()
                if d_txt:
                    phrases.append(d_txt)

    # dedupe
    deduped = []
    seen = set()
    for p in phrases:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    return deduped


# ----------------------------
# Context Builder (DB + S3)
# ----------------------------

def _build_grant_context_texts(opp, max_docs: int = 12, max_chars_per_doc: int = 10000) -> List[str]:
    texts: List[str] = []

    title = str(getattr(opp, "opportunity_title", "") or "").strip()
    summary = str(getattr(opp, "summary_description", "") or "").strip()
    if title or summary:
        texts.append(f"{title}\n\n{summary}".strip())

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


# ----------------------------
# Utilities
# ----------------------------

def _chunk_text(text: str, words_per_chunk: int = 220) -> List[str]:
    words = [w for w in (text or "").split() if w]
    return [
        " ".join(words[i:i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]


def _percent(n: int, d: int) -> float:
    return (100.0 * n / d) if d > 0 else 0.0


# ----------------------------
# MAIN RUNNER
# ----------------------------

def run_keyword_quality_from_json(
    *,
    json_path: str,
    include_domains: bool = False,
    words_per_chunk: int = 220,
) -> None:

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON must be a list of opportunity objects.")

    with SessionLocal() as sess:
        odao = OpportunityDAO(sess)

        for entry in data:
            opp_id = entry.get("opportunity_id")
            keywords = entry.get("keywords", {})

            if not opp_id:
                continue

            rows = odao.read_opportunities_by_ids_with_relations([opp_id])
            if not rows:
                print(f"\nSkipping {opp_id} (not found in DB)")
                continue

            opp = rows[0]

            phrases = _extract_keyword_phrases(keywords, include_domains)
            if not phrases:
                print(f"\nSkipping {opp_id} (no keywords)")
                continue

            context_docs = _build_grant_context_texts(opp)
            if not context_docs:
                print(f"\nSkipping {opp_id} (no S3 context found)")
                continue

            chunks: List[str] = []
            for doc in context_docs:
                chunks.extend(_chunk_text(doc, words_per_chunk))

            if not chunks:
                print(f"\nSkipping {opp_id} (chunking failed)")
                continue

            phrase_emb = embed_texts(phrases)
            chunk_emb = embed_texts(chunks)

            rel = cosine_sim_matrix(phrase_emb, chunk_emb)
            max_rel = rel.max(axis=1)

            n = len(phrases)

            strong = int((max_rel >= 0.75).sum())
            medium = int(((max_rel >= 0.50) & (max_rel < 0.75)).sum())
            weak = int(((max_rel >= 0.35) & (max_rel < 0.50)).sum())
            off = int((max_rel < 0.35).sum())

            print("\n========================================")
            print(f"Opportunity ID: {opp_id}")
            print(f"Title: {getattr(opp, 'opportunity_title', None)}")
            print(f"Keyword phrases analyzed: {n}")
            print(f"Context chunks used: {len(chunks)}\n")

            print("Relevance:")
            print(f"- strong (>= 0.75): {strong} ({_percent(strong, n):.1f}%)")
            print(f"- medium (0.50 - 0.75): {medium} ({_percent(medium, n):.1f}%)")
            print(f"- weak (0.35 - 0.50): {weak} ({_percent(weak, n):.1f}%)")
            print(f"- off-topic (< 0.35): {off} ({_percent(off, n):.1f}%)")
            print(f"- avg relevance: {float(max_rel.mean()):.4f}")


# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure keyword quality for multiple grants from JSON."
    )

    parser.add_argument("--json-path", required=True)
    parser.add_argument("--include-domains", action="store_true")
    parser.add_argument("--words-per-chunk", type=int, default=220)

    args = parser.parse_args()

    run_keyword_quality_from_json(
        json_path=args.json_path,
        include_domains=args.include_domains,
        words_per_chunk=args.words_per_chunk,
    )