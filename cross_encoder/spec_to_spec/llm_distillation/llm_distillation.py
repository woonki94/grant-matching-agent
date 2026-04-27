from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Ensure project root on sys.path for direct script execution.
def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "cross_encoder").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Requested model family (closest available instruct model in this setup).
MODEL_ID_DEFAULT = "Qwen/Qwen2.5-14B-Instruct"

GRANT_DB_DEFAULT = "cross_encoder/spec_to_spec/dataset/grant_keywords_spec_keywords_db.json"
FAC_DB_DEFAULT = "cross_encoder/spec_to_spec/dataset/fac_specs_db.json"

RAW_OUTPUT_DEFAULT = "cross_encoder/spec_to_spec/dataset/llm_distill_raw_scores.jsonl"
PAIRWISE_OUTPUT_DEFAULT = "cross_encoder/spec_to_spec/dataset/llm_distill_pairwise.jsonl"
LISTWISE_OUTPUT_DEFAULT = "cross_encoder/spec_to_spec/dataset/llm_distill_listwise.jsonl"
MANIFEST_OUTPUT_DEFAULT = "cross_encoder/spec_to_spec/dataset/llm_distill_manifest.json"
PREFILTER_SCORE_CACHE_DEFAULT = "cross_encoder/spec_to_spec/dataset/spec_facspec_cosine_cache.jsonl"


SYSTEM_PROMPT = """
You are evaluating whether a candidate specialization satisfies a requirement.

This is NOT general similarity — it is REQUIREMENT MATCHING.


Return ONLY strict JSON:
{"score": <float between 0.0 and 1.0>}


Evaluation steps (IMPORTANT — follow strictly):

1. Extract the core required concepts from the requirement text.
   - Keep them short (2–6 key phrases)
   - Do NOT invent new concepts

2. For each extracted concept:
   - Classify it as:
     - CORE (central to the requirement)
     - SUPPORTING (secondary detail)

3. For each concept:
   - Check if the candidate expresses it
   - Mark as: FULL, PARTIAL, or MISSING

4. Evaluate coverage with priority:
   - First consider CORE concepts
   - Missing a CORE concept should significantly reduce the score
   - SUPPORTING concepts influence the score only after CORE coverage is considered

5. Score based on coverage:
   - All CORE = FULL → 0.9–1.0
   - CORE mostly FULL + minor gaps → 0.75–0.9
   - Some CORE PARTIAL/MISSING → 0.5–0.75
   - Most CORE MISSING but some SUPPORTING overlap → 0.1–0.5
   - No meaningful overlap → 0.0–0.1

IMPORTANT:
- Only evaluate concepts present in the requirement
- Do NOT penalize for unrelated missing topics
- Avoid assigning identical scores when coverage differs
- Prefer slightly different scores when candidates differ in which CORE concepts they satisfy
- If a candidate covers the same broad domain but changes the main objective, method, or intended use, treat it as a partial match and cap the score at 0.65 unless most CORE concepts are still satisfied.
- A candidate that lacks one CORE concept should not receive the same score as a candidate that covers all CORE concepts partially.

Do not output explanation text.
"""

USER_PROMPT_TEMPLATE = """
Grant specialization keyword:
{spec_text}

Faculty specialization:
{fac_spec_text}
""".strip()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_path(value: Any) -> Path:
    path = Path(_clean_text(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _safe_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _safe_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _truncate(value: str, *, max_chars: int) -> str:
    text = _clean_text(value)
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        score = 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _extract_score(raw_text: str) -> Tuple[float, bool]:
    raw = _clean_text(raw_text)
    if not raw:
        return 0.0, False

    if raw.startswith("{") and raw.count("{") > raw.count("}"):
        raw = raw + ("}" * (raw.count("{") - raw.count("}")))

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and ("score" in obj):
            return _clamp_score(obj.get("score")), True
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and ("score" in obj):
                return _clamp_score(obj.get("score")), True
        except Exception:
            pass

    n = re.search(r"[-+]?\d*\.?\d+", raw)
    if n:
        return _clamp_score(n.group(0)), False
    return 0.0, False


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON: {path} ({type(e).__name__}: {e})") from e
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected top-level JSON object in {path}")
    return payload


def _build_prompt(tokenizer: Any, *, spec_text: str, fac_spec_text: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not support apply_chat_template().")
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("tokenizer.chat_template is missing for this model/tokenizer.")

    user_prompt = USER_PROMPT_TEMPLATE.format(spec_text=spec_text, fac_spec_text=fac_spec_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _flatten_specs(grant_payload: Dict[str, Any], *, max_specs: int) -> List[Dict[str, Any]]:
    grants = list(grant_payload.get("grants") or [])
    out: List[Dict[str, Any]] = []
    for grant in grants:
        if not isinstance(grant, dict):
            continue
        grant_id = _clean_text(grant.get("grant_id"))
        if not grant_id:
            continue
        specs = list(grant.get("grant_spec_keywords") or [])
        for spec_idx, spec_text in enumerate(specs):
            text_value = _clean_text(spec_text)
            if not text_value:
                continue
            out.append(
                {
                    "grant_id": grant_id,
                    "spec_idx": int(spec_idx),
                    "spec_text": text_value,
                    "query_text": text_value,
                }
            )
            if max_specs > 0 and len(out) >= max_specs:
                return out
    return out


def _flatten_fac_specs(
    fac_payload: Dict[str, Any],
    *,
    max_fac_specs: int,
    max_fac_spec_chars: int,
) -> List[Dict[str, Any]]:
    fac_specs = list(fac_payload.get("fac_specs") or [])
    out: List[Dict[str, Any]] = []
    for fac_spec in fac_specs:
        if not isinstance(fac_spec, dict):
            continue
        fac_id = _safe_int(fac_spec.get("fac_id"), default=0, minimum=0, maximum=2_147_483_647)
        fac_spec_id = _safe_int(fac_spec.get("fac_spec_id"), default=0, minimum=0, maximum=9_223_372_036_854_775_807)
        fac_spec_idx = _safe_int(fac_spec.get("fac_spec_idx"), default=0, minimum=0, maximum=1_000_000)
        section = _clean_text(fac_spec.get("section")) or "unknown"
        fac_spec_text = _truncate(_clean_text(fac_spec.get("text")), max_chars=max_fac_spec_chars)
        if fac_id <= 0 or fac_spec_id <= 0 or not fac_spec_text:
            continue
        out.append(
            {
                "fac_id": fac_id,
                "fac_spec_id": fac_spec_id,
                "fac_spec_idx": fac_spec_idx,
                "section": section,
                "fac_spec_text": fac_spec_text,
            }
        )
        if max_fac_specs > 0 and len(out) >= max_fac_specs:
            return out
    return out


def _fac_spec_key(*, fac_id: int, section: str, fac_spec_id: int, fac_spec_idx: int) -> Tuple[int, str, int, int]:
    return (
        int(fac_id),
        _clean_text(section) or "unknown",
        int(fac_spec_id),
        int(fac_spec_idx),
    )


def _spec_key(spec_item: Dict[str, Any]) -> Tuple[str, int]:
    return (
        _clean_text(spec_item.get("grant_id")),
        _safe_int(spec_item.get("spec_idx"), default=0, minimum=0, maximum=50_000_000),
    )


def _load_prefilter_score_cache(
    *,
    cache_path: Path,
    fac_specs: Sequence[Dict[str, Any]],
) -> Tuple[Dict[Tuple[str, int], List[int]], Dict[str, int]]:
    fac_spec_idx_by_key: Dict[Tuple[int, str, int, int], int] = {}
    for idx, fac_spec in enumerate(fac_specs):
        key = _fac_spec_key(
            fac_id=int(fac_spec["fac_id"]),
            section=_clean_text(fac_spec["section"]),
            fac_spec_id=int(fac_spec["fac_spec_id"]),
            fac_spec_idx=int(fac_spec["fac_spec_idx"]),
        )
        fac_spec_idx_by_key[key] = int(idx)

    out: Dict[Tuple[str, int], List[int]] = {}
    line_count = 0
    spec_kept = 0
    candidate_seen = 0
    candidate_matched = 0
    with cache_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = _clean_text(raw_line)
            if not line:
                continue
            line_count += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            grant_id = _clean_text(obj.get("grant_id"))
            spec_idx = _safe_int(obj.get("spec_idx"), default=0, minimum=0, maximum=50_000_000)
            if not grant_id:
                continue

            ranked: List[int] = []
            seen_local = set()
            for cand in list(obj.get("candidates") or []):
                if not isinstance(cand, dict):
                    continue
                candidate_seen += 1
                key = _fac_spec_key(
                    fac_id=_safe_int(cand.get("fac_id"), default=0, minimum=0, maximum=2_147_483_647),
                    section=_clean_text(cand.get("section")) or "unknown",
                    fac_spec_id=_safe_int(cand.get("fac_spec_id"), default=0, minimum=0, maximum=9_223_372_036_854_775_807),
                    fac_spec_idx=_safe_int(cand.get("fac_spec_idx"), default=0, minimum=0, maximum=1_000_000),
                )
                fac_spec_idx = fac_spec_idx_by_key.get(key)
                if fac_spec_idx is None:
                    continue
                candidate_matched += 1
                if fac_spec_idx in seen_local:
                    continue
                seen_local.add(fac_spec_idx)
                ranked.append(int(fac_spec_idx))

            if ranked:
                out[(grant_id, spec_idx)] = ranked
                spec_kept += 1

    stats = {
        "cache_line_count": int(line_count),
        "cache_spec_count": int(spec_kept),
        "cache_candidate_seen": int(candidate_seen),
        "cache_candidate_matched": int(candidate_matched),
    }
    return out, stats


def _select_fac_spec_candidates_from_ranked_indices(
    *,
    spec_item: Dict[str, Any],
    fac_specs: Sequence[Dict[str, Any]],
    ranked_indices: Sequence[int],
    top_k: int,
    mid_k: int,
    rand_k: int,
    base_seed: int,
) -> List[Dict[str, Any]]:
    total = len(fac_specs)
    if total <= 0:
        return []
    t = max(0, int(top_k))
    m = max(0, int(mid_k))
    r = max(0, int(rand_k))
    target = t + m + r
    if target <= 0 or target >= total:
        return list(fac_specs)

    grant_id = _clean_text(spec_item.get("grant_id"))
    spec_idx = int(spec_item.get("spec_idx") or 0)
    rng = _rng_for_spec(base_seed=base_seed, grant_id=grant_id, spec_idx=spec_idx)

    ranked = [int(i) for i in ranked_indices if 0 <= int(i) < total]
    selected: List[int] = []
    selected_set = set()

    def _add(idx: int) -> None:
        if idx in selected_set:
            return
        if idx < 0 or idx >= total:
            return
        selected.append(idx)
        selected_set.add(idx)

    for idx in ranked[: min(t, len(ranked))]:
        _add(idx)

    if m > 0:
        non_top = ranked[min(t, len(ranked)) :]
        if non_top:
            start = len(non_top) // 3
            end = (2 * len(non_top)) // 3
            mid_pool = non_top[start:end] if end > start else non_top
            avail = [i for i in mid_pool if i not in selected_set]
            if avail:
                pick = min(m, len(avail))
                if pick == len(avail):
                    for idx in avail:
                        _add(idx)
                else:
                    for idx in rng.sample(avail, k=pick):
                        _add(idx)

    if r > 0:
        remainder = [i for i in range(total) if i not in selected_set]
        if remainder:
            pick = min(r, len(remainder))
            if pick == len(remainder):
                for idx in remainder:
                    _add(idx)
            else:
                for idx in rng.sample(remainder, k=pick):
                    _add(idx)

    if len(selected) < target:
        for idx in ranked:
            if len(selected) >= target:
                break
            _add(idx)
    if len(selected) < target:
        remainder = [i for i in range(total) if i not in selected_set]
        if remainder:
            rng.shuffle(remainder)
            for idx in remainder:
                if len(selected) >= target:
                    break
                _add(idx)

    return [fac_specs[i] for i in selected[:target]]


def _tokenize_for_sparse(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _clean_text(text).lower())


def _build_sparse_fac_spec_index(fac_specs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a lightweight BM25-style sparse index for fast prefiltering.
    """
    postings: Dict[str, List[Tuple[int, int]]] = {}
    doc_len: List[int] = []
    n_docs = len(fac_specs)

    for idx, fac_spec in enumerate(fac_specs):
        tokens = _tokenize_for_sparse(_clean_text(fac_spec.get("fac_spec_text")))
        tf = Counter(tokens)
        doc_len.append(len(tokens))
        for term, freq in tf.items():
            postings.setdefault(term, []).append((idx, int(freq)))

    avg_doc_len = (sum(doc_len) / float(max(1, n_docs))) if n_docs > 0 else 1.0
    idf: Dict[str, float] = {}
    for term, plist in postings.items():
        df = len(plist)
        if df <= 0:
            continue
        # Standard BM25-style IDF variant with +1 smoothing inside log.
        idf[term] = float(math.log(1.0 + ((n_docs - df + 0.5) / float(df + 0.5))))

    return {
        "n_docs": int(n_docs),
        "avg_doc_len": float(max(1.0, avg_doc_len)),
        "postings": postings,
        "idf": idf,
        "doc_len": doc_len,
    }


def _rank_fac_spec_indices_for_spec(
    spec_text: str,
    *,
    sparse_index: Dict[str, Any],
    k1: float = 1.2,
    b: float = 0.75,
) -> List[int]:
    postings: Dict[str, List[Tuple[int, int]]] = sparse_index["postings"]
    idf: Dict[str, float] = sparse_index["idf"]
    doc_len: List[int] = sparse_index["doc_len"]
    avg_doc_len: float = float(sparse_index["avg_doc_len"])

    query_terms = list(set(_tokenize_for_sparse(spec_text)))
    scores: Dict[int, float] = defaultdict(float)

    for term in query_terms:
        plist = postings.get(term)
        if not plist:
            continue
        term_idf = float(idf.get(term, 0.0))
        if term_idf <= 0.0:
            continue
        for doc_idx, tf in plist:
            dl = float(doc_len[doc_idx] if doc_idx < len(doc_len) else 1)
            denom = float(tf) + float(k1) * (1.0 - float(b) + float(b) * (dl / max(1.0, avg_doc_len)))
            if denom <= 0.0:
                continue
            scores[doc_idx] += term_idf * ((float(tf) * (float(k1) + 1.0)) / denom)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [int(i) for i, _ in ranked]


def _rng_for_spec(*, base_seed: int, grant_id: str, spec_idx: int) -> random.Random:
    seed_text = f"{int(base_seed)}::{_clean_text(grant_id)}::{int(spec_idx)}"
    digest = hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:16]
    seed_int = int(digest, 16)
    return random.Random(seed_int)


def _select_fac_spec_candidates_for_spec(
    *,
    spec_item: Dict[str, Any],
    fac_specs: Sequence[Dict[str, Any]],
    sparse_index: Optional[Dict[str, Any]],
    top_k: int,
    mid_k: int,
    rand_k: int,
    base_seed: int,
) -> List[Dict[str, Any]]:
    total = len(fac_specs)
    if total <= 0:
        return []

    t = max(0, int(top_k))
    m = max(0, int(mid_k))
    r = max(0, int(rand_k))
    target = t + m + r
    if target <= 0 or target >= total or sparse_index is None:
        return list(fac_specs)

    grant_id = _clean_text(spec_item.get("grant_id"))
    spec_idx = int(spec_item.get("spec_idx") or 0)
    rng = _rng_for_spec(base_seed=base_seed, grant_id=grant_id, spec_idx=spec_idx)

    ranked = _rank_fac_spec_indices_for_spec(_clean_text(spec_item.get("spec_text")), sparse_index=sparse_index)
    selected: List[int] = []
    selected_set = set()

    def _add(idx: int) -> None:
        if idx in selected_set:
            return
        if idx < 0 or idx >= total:
            return
        selected.append(idx)
        selected_set.add(idx)

    # 1) Strong candidates from sparse rank top.
    for idx in ranked[: min(t, len(ranked))]:
        _add(idx)

    # 2) Medium band candidates from middle of non-top ranked pool.
    if m > 0:
        non_top = ranked[min(t, len(ranked)) :]
        if non_top:
            start = len(non_top) // 3
            end = (2 * len(non_top)) // 3
            mid_pool = non_top[start:end] if end > start else non_top
            if mid_pool:
                pick = min(m, len(mid_pool))
                for idx in rng.sample(mid_pool, k=pick):
                    _add(idx)

    # 3) Easy/diverse negatives from full remainder.
    if r > 0:
        remainder = [i for i in range(total) if i not in selected_set]
        if remainder:
            pick = min(r, len(remainder))
            for idx in rng.sample(remainder, k=pick):
                _add(idx)

    # Backfill if collisions reduced count.
    if len(selected) < target:
        for idx in ranked:
            if len(selected) >= target:
                break
            _add(idx)
    if len(selected) < target:
        remainder = [i for i in range(total) if i not in selected_set]
        if remainder:
            rng.shuffle(remainder)
            for idx in remainder:
                if len(selected) >= target:
                    break
                _add(idx)

    return [fac_specs[i] for i in selected[:target]]


def _chunked(items: Sequence[Any], batch_size: int) -> List[Sequence[Any]]:
    size = max(1, int(batch_size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _score_one_spec_against_all_fac_specs(
    *,
    llm: Any,
    tokenizer: Any,
    sampling_params: Any,
    spec_item: Dict[str, Any],
    fac_specs: Sequence[Dict[str, Any]],
    batch_size: int,
    progress_cb: Optional[Callable[[int], None]] = None,
    use_vllm_tqdm: bool = False,
) -> Tuple[List[Dict[str, Any]], int, int]:
    candidates: List[Dict[str, Any]] = []
    json_ok_count = 0
    fallback_count = 0

    spec_text = _clean_text(spec_item.get("spec_text"))
    for batch in _chunked(fac_specs, batch_size):
        prompts = [
            _build_prompt(
                tokenizer,
                spec_text=spec_text,
                fac_spec_text=_clean_text(c.get("fac_spec_text")),
            )
            for c in batch
        ]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=bool(use_vllm_tqdm))
        scored_in_batch = 0
        for fac_spec_item, out in zip(batch, outputs):
            raw_text = ""
            if out and getattr(out, "outputs", None):
                first = out.outputs[0] if out.outputs else None
                raw_text = _clean_text(getattr(first, "text", ""))
            score, is_json_ok = _extract_score(raw_text)
            if is_json_ok:
                json_ok_count += 1
            else:
                fallback_count += 1
            candidates.append(
                {
                    "fac_id": int(fac_spec_item["fac_id"]),
                    "fac_spec_id": int(fac_spec_item["fac_spec_id"]),
                    "fac_spec_idx": int(fac_spec_item["fac_spec_idx"]),
                    "section": _clean_text(fac_spec_item["section"]),
                    "fac_spec_text": _clean_text(fac_spec_item["fac_spec_text"]),
                    "score": float(score),
                }
            )
            scored_in_batch += 1
        if progress_cb is not None and scored_in_batch > 0:
            progress_cb(scored_in_batch)
    return candidates, json_ok_count, fallback_count


def _build_pairwise_records(
    *,
    query_text: str,
    grant_id: str,
    spec_idx: int,
    sorted_candidates: Sequence[Dict[str, Any]],
    pos_top_k: int,
    weak_bottom_k: int,
    hard_neg_k: int,
    min_margin: float,
    max_pairs_per_query: int,
) -> List[Dict[str, Any]]:
    if not sorted_candidates:
        return []

    pos_k = max(1, int(pos_top_k))
    weak_k = max(1, int(weak_bottom_k))
    hard_k = max(0, int(hard_neg_k))
    max_pairs = max(1, int(max_pairs_per_query))

    positives = list(sorted_candidates[: min(pos_k, len(sorted_candidates))])
    weak_negatives = list(sorted_candidates[-min(weak_k, len(sorted_candidates)) :])
    weak_negatives.sort(key=lambda x: float(x.get("score") or 0.0))

    hard_start = min(len(sorted_candidates), len(positives))
    hard_end = min(len(sorted_candidates), hard_start + hard_k)
    hard_negatives = list(sorted_candidates[hard_start:hard_end])

    out: List[Dict[str, Any]] = []

    for pos in positives:
        p_score = float(pos.get("score") or 0.0)

        for neg in weak_negatives:
            if int(pos.get("fac_spec_id") or 0) == int(neg.get("fac_spec_id") or 0) and int(pos.get("fac_id") or 0) == int(
                neg.get("fac_id") or 0
            ):
                continue
            n_score = float(neg.get("score") or 0.0)
            margin = p_score - n_score
            if margin <= 0:
                continue
            out.append(
                {
                    "grant_id": grant_id,
                    "spec_idx": int(spec_idx),
                    "query_text": query_text,
                    "pos_text": _clean_text(pos.get("fac_spec_text")),
                    "neg_text": _clean_text(neg.get("fac_spec_text")),
                    "teacher_pos_score": p_score,
                    "teacher_neg_score": n_score,
                    "teacher_margin": float(margin),
                    "pair_type": "strong_vs_weak",
                    "pos_fac_id": int(pos.get("fac_id") or 0),
                    "pos_fac_spec_id": int(pos.get("fac_spec_id") or 0),
                    "pos_fac_spec_idx": int(pos.get("fac_spec_idx") or 0),
                    "pos_section": _clean_text(pos.get("section")) or "unknown",
                    "neg_fac_id": int(neg.get("fac_id") or 0),
                    "neg_fac_spec_id": int(neg.get("fac_spec_id") or 0),
                    "neg_fac_spec_idx": int(neg.get("fac_spec_idx") or 0),
                    "neg_section": _clean_text(neg.get("section")) or "unknown",
                }
            )
            if len(out) >= max_pairs:
                return out

        for neg in hard_negatives:
            if int(pos.get("fac_spec_id") or 0) == int(neg.get("fac_spec_id") or 0) and int(pos.get("fac_id") or 0) == int(
                neg.get("fac_id") or 0
            ):
                continue
            n_score = float(neg.get("score") or 0.0)
            margin = p_score - n_score
            if margin < float(min_margin):
                continue
            out.append(
                {
                    "grant_id": grant_id,
                    "spec_idx": int(spec_idx),
                    "query_text": query_text,
                    "pos_text": _clean_text(pos.get("fac_spec_text")),
                    "neg_text": _clean_text(neg.get("fac_spec_text")),
                    "teacher_pos_score": p_score,
                    "teacher_neg_score": n_score,
                    "teacher_margin": float(margin),
                    "pair_type": "strong_vs_hard",
                    "pos_fac_id": int(pos.get("fac_id") or 0),
                    "pos_fac_spec_id": int(pos.get("fac_spec_id") or 0),
                    "pos_fac_spec_idx": int(pos.get("fac_spec_idx") or 0),
                    "pos_section": _clean_text(pos.get("section")) or "unknown",
                    "neg_fac_id": int(neg.get("fac_id") or 0),
                    "neg_fac_spec_id": int(neg.get("fac_spec_id") or 0),
                    "neg_fac_spec_idx": int(neg.get("fac_spec_idx") or 0),
                    "neg_section": _clean_text(neg.get("section")) or "unknown",
                }
            )
            if len(out) >= max_pairs:
                return out

    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate three distillation datasets from all grant-spec x fac-spec combinations "
            "with batched vLLM scoring: raw score, pairwise, and listwise."
        )
    )
    p.add_argument("--grant-db", type=str, default=GRANT_DB_DEFAULT, help="Grant JSON DB path.")
    p.add_argument("--fac-db", type=str, default=FAC_DB_DEFAULT, help="Faculty specialization JSON DB path.")
    p.add_argument("--raw-output", type=str, default=RAW_OUTPUT_DEFAULT, help="Raw-score JSONL output path.")
    p.add_argument("--pairwise-output", type=str, default=PAIRWISE_OUTPUT_DEFAULT, help="Pairwise JSONL output path.")
    p.add_argument("--listwise-output", type=str, default=LISTWISE_OUTPUT_DEFAULT, help="Listwise JSONL output path.")
    p.add_argument("--manifest-output", type=str, default=MANIFEST_OUTPUT_DEFAULT, help="Manifest JSON output path.")

    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT, help="vLLM model id.")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    p.add_argument("--batch-size", type=int, default=64, help="vLLM generation batch size.")
    p.add_argument("--max-new-tokens", type=int, default=24, help="max_tokens for score generation.")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")

    p.add_argument("--max-specs", type=int, default=0, help="Limit specs for smoke run (0 = all).")
    p.add_argument("--max-fac-specs", type=int, default=0, help="Limit faculty specs for smoke run (0 = all).")
    p.add_argument("--max-fac-spec-chars", type=int, default=0, help="Trim faculty spec text to this many chars (0 = no trim).")
    p.add_argument("--start-spec-index", type=int, default=0, help="Skip first N specs (manual resume).")
    p.add_argument(
        "--prefilter-top-k",
        type=int,
        default=0,
        help="Top-k faculty specs per spec before LLM scoring (used with score-cache ranking or sparse ranking).",
    )
    p.add_argument(
        "--prefilter-mid-k",
        type=int,
        default=0,
        help="Additional medium-band faculty specs sampled per spec for diversity.",
    )
    p.add_argument(
        "--prefilter-rand-k",
        type=int,
        default=0,
        help="Additional random faculty specs per spec for easy/low-score coverage.",
    )
    p.add_argument(
        "--prefilter-seed",
        type=int,
        default=42,
        help="Base random seed for deterministic per-spec prefilter sampling.",
    )
    p.add_argument(
        "--prefilter-score-cache",
        type=str,
        default=PREFILTER_SCORE_CACHE_DEFAULT,
        help="Optional JSONL cache file with precomputed cosine-ranked candidates per (grant_id, spec_idx).",
    )

    p.add_argument("--pair-pos-top-k", type=int, default=4, help="Top-K positive pool per query.")
    p.add_argument("--pair-weak-bottom-k", type=int, default=4, help="Bottom-K weak negative pool per query.")
    p.add_argument("--pair-hard-neg-k", type=int, default=4, help="Near-top hard negative pool per query.")
    p.add_argument("--pair-min-margin", type=float, default=0.05, help="Minimum margin for strong_vs_hard pairs.")
    p.add_argument("--pair-max-per-query", type=int, default=80, help="Cap pairwise rows generated per query.")

    p.add_argument("--listwise-top-k", type=int, default=0, help="Keep top-K docs in listwise output (0 = all).")
    p.add_argument("--no-tqdm", action="store_true", help="Disable global tqdm progress bar.")
    p.add_argument(
        "--vllm-tqdm",
        action="store_true",
        help="Enable vLLM internal tqdm bars (off by default so global tqdm stays clean).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        raise RuntimeError(
            "vLLM is required but not installed in this environment. "
            "Install `vllm` in your runtime, then rerun."
        ) from e

    grant_db = _resolve_path(args.grant_db)
    fac_db = _resolve_path(args.fac_db)
    raw_output = _resolve_path(args.raw_output)
    pairwise_output = _resolve_path(args.pairwise_output)
    listwise_output = _resolve_path(args.listwise_output)
    manifest_output = _resolve_path(args.manifest_output)

    model_id = _clean_text(args.model_id) or MODEL_ID_DEFAULT
    tensor_parallel_size = _safe_int(args.tensor_parallel_size, default=1, minimum=1, maximum=64)
    batch_size = _safe_int(args.batch_size, default=64, minimum=1, maximum=4096)
    max_new_tokens = _safe_int(args.max_new_tokens, default=24, minimum=1, maximum=2048)
    temperature = _safe_float(args.temperature, default=0.0, minimum=0.0, maximum=2.0)

    max_specs = _safe_int(args.max_specs, default=0, minimum=0, maximum=50_000_000)
    max_fac_specs = _safe_int(args.max_fac_specs, default=0, minimum=0, maximum=50_000_000)
    max_fac_spec_chars = _safe_int(args.max_fac_spec_chars, default=2400, minimum=0, maximum=1_000_000)
    start_spec_index = _safe_int(args.start_spec_index, default=0, minimum=0, maximum=50_000_000)
    prefilter_top_k = _safe_int(args.prefilter_top_k, default=0, minimum=0, maximum=200_000)
    prefilter_mid_k = _safe_int(args.prefilter_mid_k, default=0, minimum=0, maximum=200_000)
    prefilter_rand_k = _safe_int(args.prefilter_rand_k, default=0, minimum=0, maximum=200_000)
    prefilter_seed = _safe_int(args.prefilter_seed, default=42, minimum=0, maximum=2_147_483_647)
    prefilter_score_cache_path = _resolve_path(args.prefilter_score_cache) if _clean_text(args.prefilter_score_cache) else None
    use_prefilter = (prefilter_top_k + prefilter_mid_k + prefilter_rand_k) > 0

    pair_pos_top_k = _safe_int(args.pair_pos_top_k, default=4, minimum=1, maximum=10_000)
    pair_weak_bottom_k = _safe_int(args.pair_weak_bottom_k, default=4, minimum=1, maximum=10_000)
    pair_hard_neg_k = _safe_int(args.pair_hard_neg_k, default=4, minimum=0, maximum=10_000)
    pair_min_margin = _safe_float(args.pair_min_margin, default=0.05, minimum=0.0, maximum=1.0)
    pair_max_per_query = _safe_int(args.pair_max_per_query, default=80, minimum=1, maximum=1_000_000)
    listwise_top_k = _safe_int(args.listwise_top_k, default=0, minimum=0, maximum=10_000_000)
    use_tqdm = not bool(args.no_tqdm)
    use_vllm_tqdm = bool(args.vllm_tqdm)

    if not grant_db.exists():
        raise RuntimeError(f"Grant DB not found: {grant_db}")
    if not fac_db.exists():
        raise RuntimeError(f"Faculty DB not found: {fac_db}")

    grant_payload = _load_json(grant_db)
    fac_payload = _load_json(fac_db)

    specs_all = _flatten_specs(grant_payload, max_specs=max_specs)
    fac_specs = _flatten_fac_specs(
        fac_payload,
        max_fac_specs=max_fac_specs,
        max_fac_spec_chars=max_fac_spec_chars,
    )
    if not specs_all:
        raise RuntimeError("No grant specialization keywords found.")
    if not fac_specs:
        raise RuntimeError("No faculty specializations found with non-empty text.")

    specs = specs_all[start_spec_index:] if start_spec_index > 0 else specs_all
    if not specs:
        raise RuntimeError("No specs left after start-spec-index filtering.")

    full_total_pairs = int(len(specs) * len(fac_specs))
    prefilter_score_cache: Dict[Tuple[str, int], List[int]] = {}
    prefilter_score_cache_stats: Dict[str, int] = {}
    if prefilter_score_cache_path is not None:
        if not prefilter_score_cache_path.exists():
            raise RuntimeError(f"prefilter score cache not found: {prefilter_score_cache_path}")
        prefilter_score_cache, prefilter_score_cache_stats = _load_prefilter_score_cache(
            cache_path=prefilter_score_cache_path,
            fac_specs=fac_specs,
        )

    sparse_index: Optional[Dict[str, Any]] = None
    use_prefilter_score_cache = bool(prefilter_score_cache)
    if use_prefilter and (not use_prefilter_score_cache):
        sparse_index = _build_sparse_fac_spec_index(fac_specs)
        prefilter_target_per_spec = min(
            len(fac_specs),
            int(prefilter_top_k + prefilter_mid_k + prefilter_rand_k),
        )
    elif use_prefilter:
        prefilter_target_per_spec = min(
            len(fac_specs),
            int(prefilter_top_k + prefilter_mid_k + prefilter_rand_k),
        )
    else:
        prefilter_target_per_spec = len(fac_specs)

    total_pairs = int(len(specs) * prefilter_target_per_spec)

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    pairwise_output.parent.mkdir(parents=True, exist_ok=True)
    listwise_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model_id,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature)

    started = time.time()
    total_scored = 0
    total_pairwise = 0
    total_json_ok = 0
    total_fallback = 0
    sum_candidates_per_spec = 0
    min_candidates_per_spec = 2_147_483_647
    max_candidates_per_spec = 0

    print(f"model_id={model_id}")
    print(f"spec_count={len(specs)}")
    print(f"fac_spec_count={len(fac_specs)}")
    print(f"full_total_pairs={full_total_pairs}")
    print(f"total_pairs={total_pairs}")
    print(f"use_prefilter={use_prefilter}")
    if use_prefilter:
        print(f"use_prefilter_score_cache={use_prefilter_score_cache}")
        if prefilter_score_cache_path is not None:
            print(f"prefilter_score_cache_path={prefilter_score_cache_path}")
            print(f"prefilter_score_cache_spec_count={len(prefilter_score_cache)}")
            if prefilter_score_cache_stats:
                print(f"prefilter_score_cache_candidate_seen={prefilter_score_cache_stats.get('cache_candidate_seen')}")
                print(f"prefilter_score_cache_candidate_matched={prefilter_score_cache_stats.get('cache_candidate_matched')}")
        print(
            "prefilter_k="
            f"top:{prefilter_top_k},mid:{prefilter_mid_k},rand:{prefilter_rand_k} "
            f"(target_per_spec={prefilter_target_per_spec})"
        )
    print(f"batch_size={batch_size}")
    print(f"raw_output={raw_output}")
    print(f"pairwise_output={pairwise_output}")
    print(f"listwise_output={listwise_output}")

    tqdm_bar = None
    if use_tqdm:
        try:
            from tqdm.auto import tqdm

            tqdm_bar = tqdm(total=total_pairs, desc="Scoring pairs", unit="pair", dynamic_ncols=True)
        except Exception:
            print("tqdm_unavailable=true")

    def _update_progress(step: int) -> None:
        if tqdm_bar is not None:
            tqdm_bar.update(int(max(0, step)))

    try:
        with (
            raw_output.open("w", encoding="utf-8") as raw_f,
            pairwise_output.open("w", encoding="utf-8") as pair_f,
            listwise_output.open("w", encoding="utf-8") as list_f,
        ):
            for spec_rank, spec_item in enumerate(specs, start=1):
                if use_prefilter:
                    ranked_indices = prefilter_score_cache.get(_spec_key(spec_item)) if use_prefilter_score_cache else None
                    if use_prefilter_score_cache:
                        selected_fac_specs = _select_fac_spec_candidates_from_ranked_indices(
                            spec_item=spec_item,
                            fac_specs=fac_specs,
                            ranked_indices=ranked_indices or [],
                            top_k=prefilter_top_k,
                            mid_k=prefilter_mid_k,
                            rand_k=prefilter_rand_k,
                            base_seed=prefilter_seed,
                        )
                    else:
                        selected_fac_specs = _select_fac_spec_candidates_for_spec(
                            spec_item=spec_item,
                            fac_specs=fac_specs,
                            sparse_index=sparse_index,
                            top_k=prefilter_top_k,
                            mid_k=prefilter_mid_k,
                            rand_k=prefilter_rand_k,
                            base_seed=prefilter_seed,
                        )
                else:
                    selected_fac_specs = list(fac_specs)

                candidates, json_ok, fallback = _score_one_spec_against_all_fac_specs(
                    llm=llm,
                    tokenizer=tokenizer,
                    sampling_params=sampling_params,
                    spec_item=spec_item,
                    fac_specs=selected_fac_specs,
                    batch_size=batch_size,
                    progress_cb=_update_progress,
                    use_vllm_tqdm=use_vllm_tqdm,
                )

                total_scored += len(candidates)
                total_json_ok += int(json_ok)
                total_fallback += int(fallback)
                sum_candidates_per_spec += len(candidates)
                min_candidates_per_spec = min(min_candidates_per_spec, len(candidates))
                max_candidates_per_spec = max(max_candidates_per_spec, len(candidates))

                raw_obj = {
                    "grant_id": _clean_text(spec_item.get("grant_id")),
                    "spec_idx": int(spec_item.get("spec_idx") or 0),
                    "spec_text": _clean_text(spec_item.get("spec_text")),
                    "candidates": [
                        {
                            "fac_id": int(c["fac_id"]),
                            "fac_spec_id": int(c["fac_spec_id"]),
                            "fac_spec_idx": int(c["fac_spec_idx"]),
                            "section": _clean_text(c["section"]),
                            "fac_spec_text": _clean_text(c["fac_spec_text"]),
                            "score": float(c["score"]),
                        }
                        for c in candidates
                    ],
                }
                raw_f.write(json.dumps(raw_obj, ensure_ascii=False) + "\n")

                sorted_candidates = sorted(
                    candidates,
                    key=lambda x: float(x.get("score") or 0.0),
                    reverse=True,
                )

                docs = sorted_candidates
                if listwise_top_k > 0:
                    docs = docs[: min(listwise_top_k, len(docs))]
                listwise_obj = {
                    "grant_id": _clean_text(spec_item.get("grant_id")),
                    "spec_idx": int(spec_item.get("spec_idx") or 0),
                    "query_text": _clean_text(spec_item.get("query_text")),
                    "docs": [
                        {
                            "text": _clean_text(d["fac_spec_text"]),
                            "teacher_score": float(d["score"]),
                            "fac_id": int(d["fac_id"]),
                            "fac_spec_id": int(d["fac_spec_id"]),
                            "fac_spec_idx": int(d["fac_spec_idx"]),
                            "section": _clean_text(d["section"]),
                        }
                        for d in docs
                    ],
                }
                list_f.write(json.dumps(listwise_obj, ensure_ascii=False) + "\n")

                pairwise_rows = _build_pairwise_records(
                    query_text=_clean_text(spec_item.get("query_text")),
                    grant_id=_clean_text(spec_item.get("grant_id")),
                    spec_idx=int(spec_item.get("spec_idx") or 0),
                    sorted_candidates=sorted_candidates,
                    pos_top_k=pair_pos_top_k,
                    weak_bottom_k=pair_weak_bottom_k,
                    hard_neg_k=pair_hard_neg_k,
                    min_margin=pair_min_margin,
                    max_pairs_per_query=pair_max_per_query,
                )
                for row in pairwise_rows:
                    pair_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_pairwise += len(pairwise_rows)

                raw_f.flush()
                list_f.flush()
                pair_f.flush()

                if spec_rank % 10 == 0 or spec_rank == len(specs):
                    elapsed = max(1e-6, time.time() - started)
                    speed = total_scored / elapsed
                    avg_cand = (sum_candidates_per_spec / float(max(1, spec_rank)))
                    if tqdm_bar is not None:
                        tqdm_bar.set_postfix(
                            {
                                "spec": f"{spec_rank}/{len(specs)}",
                                "cand/spec": f"{avg_cand:.1f}",
                                "pairs_per_s": f"{speed:.2f}",
                            }
                        )
                    print(
                        f"spec_progress={spec_rank}/{len(specs)} "
                        f"scored_pairs={total_scored}/{total_pairs} "
                        f"avg_candidates_per_spec={avg_cand:.2f} "
                        f"pairwise_rows={total_pairwise} "
                        f"speed={speed:.2f} pairs/sec"
                    )
    finally:
        if tqdm_bar is not None:
            tqdm_bar.close()

    elapsed = max(1e-6, time.time() - started)
    avg_candidates = float(sum_candidates_per_spec / float(max(1, len(specs))))
    if min_candidates_per_spec == 2_147_483_647:
        min_candidates_per_spec = 0
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "grant_db": str(grant_db),
        "fac_db": str(fac_db),
        "raw_output": str(raw_output),
        "pairwise_output": str(pairwise_output),
        "listwise_output": str(listwise_output),
        "spec_count": int(len(specs)),
        "fac_spec_count": int(len(fac_specs)),
        "full_total_pairs": int(full_total_pairs),
        "total_pairs": int(total_pairs),
        "total_scored_pairs": int(total_scored),
        "reduction_ratio_vs_full": float(total_scored / float(max(1, full_total_pairs))),
        "use_prefilter": bool(use_prefilter),
        "prefilter_top_k": int(prefilter_top_k),
        "prefilter_mid_k": int(prefilter_mid_k),
        "prefilter_rand_k": int(prefilter_rand_k),
        "prefilter_target_per_spec": int(prefilter_target_per_spec),
        "prefilter_seed": int(prefilter_seed),
        "use_prefilter_score_cache": bool(use_prefilter_score_cache),
        "prefilter_score_cache_path": str(prefilter_score_cache_path) if prefilter_score_cache_path is not None else "",
        "prefilter_score_cache_spec_count": int(len(prefilter_score_cache)),
        "prefilter_score_cache_stats": prefilter_score_cache_stats,
        "avg_candidates_per_spec": float(avg_candidates),
        "min_candidates_per_spec": int(min_candidates_per_spec),
        "max_candidates_per_spec": int(max_candidates_per_spec),
        "total_pairwise_rows": int(total_pairwise),
        "parsed_json_ok_count": int(total_json_ok),
        "parsed_fallback_count": int(total_fallback),
        "tensor_parallel_size": int(tensor_parallel_size),
        "batch_size": int(batch_size),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "max_fac_spec_chars": int(max_fac_spec_chars),
        "start_spec_index": int(start_spec_index),
        "max_specs": int(max_specs),
        "max_fac_specs": int(max_fac_specs),
        "pair_pos_top_k": int(pair_pos_top_k),
        "pair_weak_bottom_k": int(pair_weak_bottom_k),
        "pair_hard_neg_k": int(pair_hard_neg_k),
        "pair_min_margin": float(pair_min_margin),
        "pair_max_per_query": int(pair_max_per_query),
        "listwise_top_k": int(listwise_top_k),
        "elapsed_seconds": float(elapsed),
        "pairs_per_second": float(total_scored / elapsed),
    }
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("done=true")
    print(f"total_scored_pairs={total_scored}")
    print(f"full_total_pairs={full_total_pairs}")
    print(f"reduction_ratio_vs_full={(total_scored / float(max(1, full_total_pairs))):.6f}")
    print(f"avg_candidates_per_spec={avg_candidates:.2f}")
    print(f"total_pairwise_rows={total_pairwise}")
    print(f"parsed_json_ok_count={total_json_ok}")
    print(f"parsed_fallback_count={total_fallback}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"pairs_per_second={(total_scored / elapsed):.4f}")
    print(f"manifest_output={manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
