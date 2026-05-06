from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "cross_encoder").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local augmentation helper.
try:
    from cross_encoder.data_preparation.llm_distillation.augmentation import LLMDistillationAugmenter
except Exception:
    try:
        from augmentation import LLMDistillationAugmenter  # type: ignore
    except Exception:
        LLMDistillationAugmenter = None  # type: ignore


GRANT_DB_DEFAULT = "cross_encoder/dataset/source/grant_keywords_spec_keywords_db.json"
FAC_DB_DEFAULT = "cross_encoder/dataset/source/fac_specs_db.json"
PREFILTER_CACHE_DEFAULT = "cross_encoder/dataset/source/spec_facspec_sts_cache.jsonl"
RAW_OUTPUT_DEFAULT = "cross_encoder/dataset/distill/llm_distill2_raw_scores.jsonl"
PAIRWISE_OUTPUT_DEFAULT = "cross_encoder/dataset/distill/llm_distill2_pairwise.jsonl"
LISTWISE_OUTPUT_DEFAULT = "cross_encoder/dataset/distill/llm_distill2_listwise.jsonl"
MANIFEST_DEFAULT = "cross_encoder/dataset/distill/llm_distill2_manifest.json"

MODEL_ID_DEFAULT = "Qwen/Qwen2.5-14B-Instruct"
BATCH_SIZE_DEFAULT = 64
MAX_NEW_TOKENS_DEFAULT = 24
TEMPERATURE_DEFAULT = 0.0
SEED_DEFAULT = 42

# Pairwise control defaults (minimal complexity, aligned with requested recipe).
PAIR_MAX_PER_SPEC_DEFAULT = 30
PAIR_MAX_DISAGREE_PER_SPEC_DEFAULT = 6
PAIR_MAX_BOUNDARY_PER_SPEC_DEFAULT = 6
PAIR_WEAK_MIN_DEFAULT = 10
DISAGREE_TOP_PERCENTILE_DEFAULT = 0.20
DISAGREE_LOW_SCORE_NORM_DEFAULT = 0.30
DISAGREE_MIN_MARGIN_DEFAULT = 0.15
BOUNDARY_MIN_MARGIN_DEFAULT = 0.05
ALLOW_EXTRA_DISAGREE_FROM_UNSELECTED_DEFAULT = True

# Post-score quality controls (fixed, keep code simple).
COVERAGE_GATE_MIN_MID_HIGH_DEFAULT = 0.60
AUGMENT_ENABLE_DEFAULT = True
AUGMENT_MAX_ATTEMPTS_DEFAULT = 3
AUGMENT_MAX_TRIES_PER_MISSING_DEFAULT = 3
AUGMENT_MAX_NEW_TOKENS_DEFAULT = 384
AUGMENT_VALIDATION_MAX_NEW_TOKENS_DEFAULT = 160

# LLM score banding for real-candidate target assignment.
HIGH_SCORE_MIN_DEFAULT = 0.70
MID_SCORE_MIN_DEFAULT = 0.30
MID_SCORE_MAX_DEFAULT = 0.69

SYSTEM_PROMPT = """
You are evaluating whether a candidate specialization satisfies a requirement.

This is NOT general similarity - it is REQUIREMENT MATCHING.

Return ONLY strict JSON:
{"score": <float between 0.0 and 1.0>}

Evaluation steps (IMPORTANT - follow strictly):

1. Extract the core required concepts from the requirement text.
   - Keep them short (2-6 key phrases)
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
   - All CORE = FULL -> 0.9-1.0
   - CORE mostly FULL + minor gaps -> 0.75-0.9
   - Some CORE PARTIAL/MISSING -> 0.5-0.75
   - Most CORE MISSING but some SUPPORTING overlap -> 0.1-0.5
   - No meaningful overlap -> 0.0-0.1

IMPORTANT:
- Only evaluate concepts present in the requirement
- Do NOT penalize for unrelated missing topics
- Avoid assigning identical scores when coverage differs
- Prefer slightly different scores when candidates differ in which CORE concepts they satisfy
- If a candidate covers the same broad domain but changes the main objective, method, or intended use,
  treat it as a partial match and cap the score at 0.65 unless most CORE concepts are still satisfied.
- A candidate that lacks one CORE concept should not receive the same score as a candidate that covers all CORE concepts partially.

Do not output explanation text.
""".strip()

USER_PROMPT_TEMPLATE = """
Grant specialization keyword:
{spec_text}

Faculty specialization:
{fac_spec_text}
""".strip()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_path(value: Any) -> Path:
    p = Path(_clean_text(value)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


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
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON: {path} ({type(exc).__name__}: {exc})") from exc
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected top-level JSON object in {path}")
    return obj


def _flatten_specs(grant_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for grant in list(grant_payload.get("grants") or []):
        if not isinstance(grant, dict):
            continue
        grant_id = _clean_text(grant.get("grant_id"))
        if not grant_id:
            continue
        for spec_idx, spec_text in enumerate(list(grant.get("grant_spec_keywords") or [])):
            text_value = _clean_text(spec_text)
            if not text_value:
                continue
            out.append(
                {
                    "grant_id": grant_id,
                    "spec_idx": int(spec_idx),
                    "spec_text": text_value,
                }
            )
    return out


def _flatten_fac_specs(fac_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for fac_spec in list(fac_payload.get("fac_specs") or []):
        if not isinstance(fac_spec, dict):
            continue
        fac_id = _safe_int(fac_spec.get("fac_id"), default=0, minimum=0, maximum=2_147_483_647)
        fac_spec_id = _safe_int(
            fac_spec.get("fac_spec_id"),
            default=0,
            minimum=0,
            maximum=9_223_372_036_854_775_807,
        )
        fac_spec_idx = _safe_int(fac_spec.get("fac_spec_idx"), default=0, minimum=0, maximum=1_000_000)
        section = _clean_text(fac_spec.get("section")) or "unknown"
        fac_spec_text = _clean_text(fac_spec.get("text"))
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
    return out


def _fac_spec_key(*, fac_id: int, section: str, fac_spec_id: int, fac_spec_idx: int) -> Tuple[int, str, int, int]:
    return (int(fac_id), _clean_text(section) or "unknown", int(fac_spec_id), int(fac_spec_idx))


def _load_prefilter_score_cache(
    *,
    cache_path: Path,
    fac_specs: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, int], List[int]]:
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
    with cache_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = _clean_text(raw_line)
            if not line:
                continue
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
                key = _fac_spec_key(
                    fac_id=_safe_int(cand.get("fac_id"), default=0, minimum=0, maximum=2_147_483_647),
                    section=_clean_text(cand.get("section")) or "unknown",
                    fac_spec_id=_safe_int(cand.get("fac_spec_id"), default=0, minimum=0, maximum=9_223_372_036_854_775_807),
                    fac_spec_idx=_safe_int(cand.get("fac_spec_idx"), default=0, minimum=0, maximum=1_000_000),
                )
                fac_idx = fac_spec_idx_by_key.get(key)
                if fac_idx is None or fac_idx in seen_local:
                    continue
                seen_local.add(fac_idx)
                ranked.append(int(fac_idx))

            if ranked:
                out[(grant_id, spec_idx)] = ranked
    return out


def _rng_for_spec(*, base_seed: int, grant_id: str, spec_idx: int) -> random.Random:
    seed_text = f"{int(base_seed)}::{_clean_text(grant_id)}::{int(spec_idx)}"
    digest = hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:16]
    return random.Random(int(digest, 16))


def _select_prefilter_buckets(
    *,
    total: int,
    ranked_indices: Sequence[int],
    high_k: int,
    mid_k: int,
    low_k: int,
    rng: random.Random,
) -> Tuple[List[int], List[int], List[int]]:
    if total <= 0:
        return [], [], []

    ranked: List[int] = []
    ranked_seen = set()
    for raw_idx in ranked_indices:
        idx = int(raw_idx)
        if idx < 0 or idx >= total or idx in ranked_seen:
            continue
        ranked_seen.add(idx)
        ranked.append(idx)

    high: List[int] = []
    high_set = set()
    for idx in ranked[: min(len(ranked), int(high_k))]:
        high.append(idx)
        high_set.add(idx)

    mid: List[int] = []
    if int(mid_k) > 0:
        non_top = ranked[min(len(ranked), int(high_k)) :]
        if non_top:
            start = len(non_top) // 3
            end = (2 * len(non_top)) // 3
            mid_pool = non_top[start:end] if end > start else non_top
            mid_pool = [i for i in mid_pool if i not in high_set]
            if mid_pool:
                pick = min(int(mid_k), len(mid_pool))
                mid = mid_pool if pick == len(mid_pool) else rng.sample(mid_pool, k=pick)

    selected_set = set(high) | set(mid)
    low: List[int] = []
    if int(low_k) > 0:
        pool = [i for i in range(total) if i not in selected_set]
        if pool:
            pick = min(int(low_k), len(pool))
            low = pool if pick == len(pool) else rng.sample(pool, k=pick)
    return high, mid, low


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
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _chunked(items: Sequence[Any], batch_size: int) -> List[Sequence[Any]]:
    size = max(1, int(batch_size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _score_prefiltered_candidates(
    *,
    llm: Any,
    tokenizer: Any,
    sampling_params: Any,
    spec_text: str,
    candidates: Sequence[Dict[str, Any]],
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    out: List[Dict[str, Any]] = []
    json_ok_count = 0
    fallback_count = 0

    for batch in _chunked(candidates, batch_size):
        prompts = [
            _build_prompt(tokenizer, spec_text=spec_text, fac_spec_text=_clean_text(c.get("fac_spec_text")))
            for c in batch
        ]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        for cand, model_out in zip(batch, outputs):
            raw_text = ""
            if model_out and getattr(model_out, "outputs", None):
                first = model_out.outputs[0] if model_out.outputs else None
                raw_text = _clean_text(getattr(first, "text", ""))
            score, is_json_ok = _extract_score(raw_text)
            if is_json_ok:
                json_ok_count += 1
            else:
                fallback_count += 1
            row = dict(cand)
            row["llm_score_raw"] = float(score)
            out.append(row)
    return out, json_ok_count, fallback_count


def _normalize_llm_scores(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = [dict(c) for c in candidates]
    if not out:
        return out
    vals = [float(c.get("llm_score_raw") or 0.0) for c in out]
    mn = float(min(vals))
    mx = float(max(vals))
    denom = float(mx - mn)
    for c in out:
        raw = float(c.get("llm_score_raw") or 0.0)
        if denom > 1e-12:
            c["llm_score_norm"] = float((raw - mn) / denom)
        else:
            c["llm_score_norm"] = float(raw)
    return out


def _dedup_text_key(text: Any) -> str:
    return re.sub(r"\s+", " ", _clean_text(text)).strip().lower()


def _assign_target_clusters(
    *,
    candidates: Sequence[Dict[str, Any]],
    target_high: int,
    target_mid: int,
    target_low: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    out = [dict(c) for c in candidates]
    for c in out:
        c["llm_target_cluster"] = "unused"
        c["selected_for_target"] = False

    high_pool_idx = [
        i for i, c in enumerate(out)
        if float(c.get("llm_score_raw") or 0.0) >= float(HIGH_SCORE_MIN_DEFAULT)
    ]
    high_pool_idx.sort(key=lambda i: float(out[i].get("llm_score_raw") or 0.0), reverse=True)

    mid_pool_idx = [
        i for i, c in enumerate(out)
        if float(MID_SCORE_MIN_DEFAULT) <= float(c.get("llm_score_raw") or 0.0) <= float(MID_SCORE_MAX_DEFAULT)
    ]
    mid_pool_idx.sort(key=lambda i: float(out[i].get("llm_score_raw") or 0.0), reverse=True)

    low_pool_idx = [
        i for i, c in enumerate(out)
        if float(c.get("llm_score_raw") or 0.0) < float(MID_SCORE_MIN_DEFAULT)
    ]
    low_pool_idx.sort(key=lambda i: float(out[i].get("llm_score_raw") or 0.0))

    sel_high_idx = high_pool_idx[: max(0, int(target_high))]
    sel_mid_idx = mid_pool_idx[: max(0, int(target_mid))]
    sel_low_idx = low_pool_idx[: max(0, int(target_low))]

    selected_set = set(sel_high_idx) | set(sel_mid_idx) | set(sel_low_idx)
    if len(sel_low_idx) < max(0, int(target_low)):
        missing_low = int(max(0, int(target_low) - len(sel_low_idx)))
        fallback_low = [i for i in range(len(out)) if i not in selected_set]
        fallback_low.sort(key=lambda i: float(out[i].get("llm_score_raw") or 0.0))
        for i in fallback_low[:missing_low]:
            sel_low_idx.append(i)
            selected_set.add(i)

    for i in sel_high_idx:
        out[i]["llm_target_cluster"] = "high"
        out[i]["selected_for_target"] = True
    for i in sel_mid_idx:
        out[i]["llm_target_cluster"] = "mid"
        out[i]["selected_for_target"] = True
    for i in sel_low_idx:
        out[i]["llm_target_cluster"] = "low"
        out[i]["selected_for_target"] = True

    sel_high = len(sel_high_idx)
    sel_mid = len(sel_mid_idx)
    sel_low = len(sel_low_idx)

    info = {
        "target_high": int(target_high),
        "target_mid": int(target_mid),
        "target_low": int(target_low),
        "available_high": int(len(high_pool_idx)),
        "available_mid": int(len(mid_pool_idx)),
        "available_low": int(len(low_pool_idx)),
        "selected_high": int(sel_high),
        "selected_mid": int(sel_mid),
        "selected_low": int(sel_low),
        "missing_high": int(max(0, int(target_high) - int(sel_high))),
        "missing_mid": int(max(0, int(target_mid) - int(sel_mid))),
        "missing_low": int(max(0, int(target_low) - int(sel_low))),
    }
    info["selected_total"] = int(sel_high + sel_mid + sel_low)
    info["missing_total"] = int(info["missing_high"] + info["missing_mid"] + info["missing_low"])
    return out, info


def _summarize_selected_clusters(
    *,
    candidates: Sequence[Dict[str, Any]],
    target_high: int,
    target_mid: int,
    target_low: int,
) -> Dict[str, int]:
    selected = [c for c in candidates if bool(c.get("selected_for_target"))]
    selected_high = sum(1 for c in selected if _clean_text(c.get("llm_target_cluster")) == "high")
    selected_mid = sum(1 for c in selected if _clean_text(c.get("llm_target_cluster")) == "mid")
    selected_low = sum(1 for c in selected if _clean_text(c.get("llm_target_cluster")) == "low")
    out = {
        "target_high": int(target_high),
        "target_mid": int(target_mid),
        "target_low": int(target_low),
        "selected_high": int(selected_high),
        "selected_mid": int(selected_mid),
        "selected_low": int(selected_low),
        "missing_high": int(max(0, int(target_high) - int(selected_high))),
        "missing_mid": int(max(0, int(target_mid) - int(selected_mid))),
        "missing_low": int(max(0, int(target_low) - int(selected_low))),
    }
    out["selected_total"] = int(selected_high + selected_mid + selected_low)
    out["missing_total"] = int(out["missing_high"] + out["missing_mid"] + out["missing_low"])
    return out


def _coverage_ratio(actual: int, target: int) -> float:
    t = int(target)
    if t <= 0:
        return 1.0
    return float(max(0, int(actual)) / float(t))


def _band_from_raw_score(raw_score: float) -> str:
    s = float(raw_score)
    if s >= float(HIGH_SCORE_MIN_DEFAULT):
        return "strong"
    if s >= float(MID_SCORE_MIN_DEFAULT):
        return "boundary"
    return "weak"


def _augment_missing_high_mid_for_spec(
    *,
    augmenter: Any,
    spec_text: str,
    existing_candidates: Sequence[Dict[str, Any]],
    missing_high: int,
    missing_mid: int,
    next_synthetic_id: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], int]:
    synthetic_rows: List[Dict[str, Any]] = []
    cursor = int(max(1, next_synthetic_id))

    used_text = {_dedup_text_key(c.get("fac_spec_text")) for c in existing_candidates if _clean_text(c.get("fac_spec_text"))}
    used_text = {k for k in used_text if k}

    stats = {
        "requested_high": int(max(0, int(missing_high))),
        "requested_mid": int(max(0, int(missing_mid))),
        "created_high": 0,
        "created_mid": 0,
        "attempts_total": 0,
        "rejected_empty": 0,
        "rejected_duplicate": 0,
        "rejected_validation": 0,
        "unfilled_high": 0,
        "unfilled_mid": 0,
    }

    for cluster, required in (("high", int(max(0, int(missing_high)))), ("mid", int(max(0, int(missing_mid))))):
        for _ in range(required):
            accepted = False
            for _try in range(int(max(1, AUGMENT_MAX_TRIES_PER_MISSING_DEFAULT))):
                stats["attempts_total"] += 1
                out = augmenter.augment(query=spec_text, target_cluster=cluster)
                augmented_text = _clean_text(out.get("augmented_text"))
                if not augmented_text:
                    stats["rejected_empty"] += 1
                    continue
                key = _dedup_text_key(augmented_text)
                if not key or key in used_text:
                    stats["rejected_duplicate"] += 1
                    continue
                validation = dict(out.get("validation") or {})
                if not bool(validation.get("pass_valid_range")):
                    stats["rejected_validation"] += 1
                    continue
                score = _clamp_score(validation.get("score"))
                syn_id = int(cursor)
                cursor += 1
                synthetic_rows.append(
                    {
                        "fac_id": int(-syn_id),
                        "fac_spec_id": int(-syn_id),
                        "fac_spec_idx": int(-syn_id),
                        "section": f"augmented_{cluster}",
                        "fac_spec_text": augmented_text,
                        "prefilter_bucket": f"augmented_{cluster}",
                        "sts_rank": -1,
                        "sts_rank_percentile": 1.0,
                        "llm_score_raw": float(score),
                        "llm_score_norm": float(score),
                        "llm_target_cluster": cluster,
                        "selected_for_target": True,
                        "is_augmented": True,
                    }
                )
                used_text.add(key)
                stats[f"created_{cluster}"] += 1
                accepted = True
                break
            if not accepted:
                stats[f"unfilled_{cluster}"] += 1

    return synthetic_rows, stats, cursor


def _candidate_id(c: Dict[str, Any]) -> Tuple[int, int, int]:
    return (
        _safe_int(c.get("fac_id"), default=0, minimum=-2_147_483_648, maximum=2_147_483_647),
        _safe_int(c.get("fac_spec_id"), default=0, minimum=-9_223_372_036_854_775_808, maximum=9_223_372_036_854_775_807),
        _safe_int(c.get("fac_spec_idx"), default=0, minimum=-2_147_483_648, maximum=2_147_483_647),
    )


def _attach_sts_rank_percentile(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = [dict(c) for c in candidates]
    ranked = [int(c.get("sts_rank", -1)) for c in out if int(c.get("sts_rank", -1)) >= 0]
    uniq = sorted(set(ranked))
    rank_to_pct: Dict[int, float] = {}
    if uniq:
        denom = float(max(1, len(uniq) - 1))
        for i, val in enumerate(uniq):
            rank_to_pct[int(val)] = float(i) / denom if len(uniq) > 1 else 0.0
    for c in out:
        r = int(c.get("sts_rank", -1))
        c["sts_rank_percentile"] = float(rank_to_pct.get(r, 1.0))
    return out


def _is_disagreement_candidate(c: Dict[str, Any]) -> bool:
    pct = float(c.get("sts_rank_percentile", 1.0))
    score = float(c.get("llm_score_norm", 1.0))
    return bool(pct <= float(DISAGREE_TOP_PERCENTILE_DEFAULT) and score <= float(DISAGREE_LOW_SCORE_NORM_DEFAULT))


def _build_pair_row(
    *,
    grant_id: str,
    spec_idx: int,
    query_text: str,
    pos: Dict[str, Any],
    neg: Dict[str, Any],
    pair_type: str,
) -> Dict[str, Any]:
    p_score = float(pos.get("llm_score_raw") or 0.0)
    n_score = float(neg.get("llm_score_raw") or 0.0)
    return {
        "grant_id": grant_id,
        "spec_idx": int(spec_idx),
        "query_text": query_text,
        "pair_type": pair_type,
        "pos_text": _clean_text(pos.get("fac_spec_text")),
        "neg_text": _clean_text(neg.get("fac_spec_text")),
        "teacher_pos_score": float(p_score),
        "teacher_neg_score": float(n_score),
        "teacher_margin": float(p_score - n_score),
        "pos_fac_id": int(pos.get("fac_id") or 0),
        "pos_fac_spec_id": int(pos.get("fac_spec_id") or 0),
        "pos_fac_spec_idx": int(pos.get("fac_spec_idx") or 0),
        "pos_section": _clean_text(pos.get("section")) or "unknown",
        "pos_sts_rank": int(pos.get("sts_rank", -1)),
        "pos_sts_rank_percentile": float(pos.get("sts_rank_percentile", 1.0)),
        "neg_fac_id": int(neg.get("fac_id") or 0),
        "neg_fac_spec_id": int(neg.get("fac_spec_id") or 0),
        "neg_fac_spec_idx": int(neg.get("fac_spec_idx") or 0),
        "neg_section": _clean_text(neg.get("section")) or "unknown",
        "neg_sts_rank": int(neg.get("sts_rank", -1)),
        "neg_sts_rank_percentile": float(neg.get("sts_rank_percentile", 1.0)),
    }


def _build_controlled_pairwise_records(
    *,
    grant_id: str,
    spec_idx: int,
    query_text: str,
    candidates: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    out: List[Dict[str, Any]] = []
    emitted = set()

    def _append(pos: Dict[str, Any], neg: Dict[str, Any], *, pair_type: str, min_margin: float = 0.0) -> bool:
        pos_id = _candidate_id(pos)
        neg_id = _candidate_id(neg)
        if pos_id == neg_id:
            return False
        sig = (pair_type, pos_id, neg_id)
        if sig in emitted:
            return False
        p_score = float(pos.get("llm_score_raw") or 0.0)
        n_score = float(neg.get("llm_score_raw") or 0.0)
        if (p_score - n_score) < float(min_margin):
            return False
        out.append(
            _build_pair_row(
                grant_id=grant_id,
                spec_idx=spec_idx,
                query_text=query_text,
                pos=pos,
                neg=neg,
                pair_type=pair_type,
            )
        )
        emitted.add(sig)
        return True

    selected = [c for c in candidates if bool(c.get("selected_for_target"))]
    pos = sorted(
        [c for c in selected if _clean_text(c.get("llm_target_cluster")) == "high"],
        key=lambda x: float(x.get("llm_score_raw") or 0.0),
        reverse=True,
    )
    mid = sorted(
        [c for c in selected if _clean_text(c.get("llm_target_cluster")) == "mid"],
        key=lambda x: float(x.get("llm_score_raw") or 0.0),
    )
    low = sorted(
        [c for c in selected if _clean_text(c.get("llm_target_cluster")) == "low"],
        key=lambda x: float(x.get("llm_score_raw") or 0.0),
    )

    if not pos:
        return out, {
            "pair_total": 0,
            "pair_disagreement": 0,
            "pair_strong_vs_boundary": 0,
            "pair_strong_vs_weak": 0,
            "pair_strong_vs_hard": 0,
        }

    max_pairs = int(max(1, PAIR_MAX_PER_SPEC_DEFAULT))
    weak_target = int(min(max(0, PAIR_WEAK_MIN_DEFAULT), max_pairs, len(pos) * len(low)))
    disagree_cap_base = int(min(max_pairs, PAIR_MAX_DISAGREE_PER_SPEC_DEFAULT, math.floor(0.3 * max_pairs)))
    disagree_cap = int(min(disagree_cap_base, max(0, max_pairs - weak_target)))
    boundary_cap_base = int(min(max_pairs, PAIR_MAX_BOUNDARY_PER_SPEC_DEFAULT, math.floor(0.3 * max_pairs)))
    boundary_cap = int(min(boundary_cap_base, max(0, max_pairs - weak_target - disagree_cap)))

    neg_selected = list(low) + list(mid)
    dis_selected = [c for c in neg_selected if _is_disagreement_candidate(c)]
    dis_selected.sort(key=lambda x: (float(x.get("llm_score_norm", 1.0)), float(x.get("sts_rank_percentile", 1.0))))

    dis_negatives: List[Dict[str, Any]] = []
    seen_neg = set()
    for c in dis_selected:
        cid = _candidate_id(c)
        if cid in seen_neg:
            continue
        seen_neg.add(cid)
        dis_negatives.append(c)

    if ALLOW_EXTRA_DISAGREE_FROM_UNSELECTED_DEFAULT and len(dis_negatives) < disagree_cap:
        dis_extra = [
            c for c in candidates
            if (not bool(c.get("selected_for_target"))) and _is_disagreement_candidate(c)
        ]
        dis_extra.sort(key=lambda x: (float(x.get("llm_score_norm", 1.0)), float(x.get("sts_rank_percentile", 1.0))))
        for c in dis_extra:
            cid = _candidate_id(c)
            if cid in seen_neg:
                continue
            seen_neg.add(cid)
            dis_negatives.append(c)
            if len(dis_negatives) >= disagree_cap:
                break

    # 1) Disagreement pairs (bounded).
    dis_added = 0
    for p in pos:
        for n in dis_negatives:
            if len(out) >= max_pairs or dis_added >= disagree_cap:
                break
            if _append(p, n, pair_type="llm_disagreement", min_margin=DISAGREE_MIN_MARGIN_DEFAULT):
                dis_added += 1
        if len(out) >= max_pairs or dis_added >= disagree_cap:
            break

    # 2) Boundary negatives (bounded).
    boundary_added = 0
    for p in pos:
        for n in mid:
            if len(out) >= max_pairs or boundary_added >= boundary_cap:
                break
            if _append(p, n, pair_type="strong_vs_boundary", min_margin=BOUNDARY_MIN_MARGIN_DEFAULT):
                boundary_added += 1
        if len(out) >= max_pairs or boundary_added >= boundary_cap:
            break

    # 3) True weak negatives (minimum target if available).
    weak_added = 0
    for p in pos:
        for n in low:
            if len(out) >= max_pairs or weak_added >= weak_target:
                break
            if _append(p, n, pair_type="strong_vs_weak", min_margin=0.0):
                weak_added += 1
        if len(out) >= max_pairs or weak_added >= weak_target:
            break

    # 4) Fill remaining with hard negatives from selected mid/low.
    hard_pool = sorted(neg_selected, key=lambda x: float(x.get("llm_score_raw") or 0.0), reverse=True)
    for p in pos:
        for n in hard_pool:
            if len(out) >= max_pairs:
                break
            _append(p, n, pair_type="strong_vs_hard", min_margin=0.0)
        if len(out) >= max_pairs:
            break

    info = {
        "pair_total": int(len(out)),
        "pair_disagreement": int(sum(1 for r in out if _clean_text(r.get("pair_type")) == "llm_disagreement")),
        "pair_strong_vs_boundary": int(sum(1 for r in out if _clean_text(r.get("pair_type")) == "strong_vs_boundary")),
        "pair_strong_vs_weak": int(sum(1 for r in out if _clean_text(r.get("pair_type")) == "strong_vs_weak")),
        "pair_strong_vs_hard": int(sum(1 for r in out if _clean_text(r.get("pair_type")) == "strong_vs_hard")),
    }
    return out, info


def main() -> int:
    # ======================================================
    # Step 1) Parse minimal args
    # ======================================================
    parser = argparse.ArgumentParser(
        description=(
            "Prefilter + LLM scoring + score-band target selection + "
            "coverage-gate drop + high/mid augmentation recovery."
        )
    )
    parser.add_argument(
        "--prefilter-multiplier",
        type=float,
        default=3.0,
        help=(
            "Multiplier applied to each target cluster count to derive prefilter sizes. "
            "Example: target high=5 with multiplier=3.0 -> prefilter high=15."
        ),
    )
    parser.add_argument("--target-high", type=int, default=5, help="Exact target high count per spec after LLM scoring.")
    parser.add_argument("--target-mid", type=int, default=10, help="Exact target mid count per spec after LLM scoring.")
    parser.add_argument("--target-low", type=int, default=5, help="Exact target low count per spec after LLM scoring.")
    args = parser.parse_args()

    prefilter_multiplier = _safe_float(args.prefilter_multiplier, default=2.0, minimum=0.0, maximum=100.0)
    target_high = _safe_int(args.target_high, default=5, minimum=0, maximum=100_000)
    target_mid = _safe_int(args.target_mid, default=10, minimum=0, maximum=100_000)
    target_low = _safe_int(args.target_low, default=5, minimum=0, maximum=100_000)
    prefilter_high = int(max(target_high, math.ceil(float(target_high) * float(prefilter_multiplier)))) if target_high > 0 else 0
    prefilter_mid = int(max(target_mid, math.ceil(float(target_mid) * float(prefilter_multiplier)))) if target_mid > 0 else 0
    prefilter_low = int(max(target_low, math.ceil(float(target_low) * float(prefilter_multiplier)))) if target_low > 0 else 0

    # ======================================================
    # Step 2) Resolve fixed paths
    # ======================================================
    grant_db = _resolve_path(GRANT_DB_DEFAULT)
    fac_db = _resolve_path(FAC_DB_DEFAULT)
    prefilter_cache = _resolve_path(PREFILTER_CACHE_DEFAULT)
    raw_output_path = _resolve_path(RAW_OUTPUT_DEFAULT)
    pairwise_output_path = _resolve_path(PAIRWISE_OUTPUT_DEFAULT)
    listwise_output_path = _resolve_path(LISTWISE_OUTPUT_DEFAULT)
    manifest_path = _resolve_path(MANIFEST_DEFAULT)

    if not grant_db.exists():
        raise RuntimeError(f"Grant DB not found: {grant_db}")
    if not fac_db.exists():
        raise RuntimeError(f"Faculty DB not found: {fac_db}")
    if not prefilter_cache.exists():
        raise RuntimeError(f"STS prefilter cache not found: {prefilter_cache}")

    # ======================================================
    # Step 3) Load datasets + STS cache
    # ======================================================
    grant_payload = _load_json(grant_db)
    fac_payload = _load_json(fac_db)
    specs = _flatten_specs(grant_payload)
    fac_specs = _flatten_fac_specs(fac_payload)
    if not specs:
        raise RuntimeError("No specs found in grant DB.")
    if not fac_specs:
        raise RuntimeError("No faculty specs found in faculty DB.")

    prefilter_map = _load_prefilter_score_cache(cache_path=prefilter_cache, fac_specs=fac_specs)

    # ======================================================
    # Step 4) Load LLM scorer
    # ======================================================
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise RuntimeError(
            "vLLM is required but not installed in this environment. Install `vllm` and rerun."
        ) from exc

    llm = LLM(
        MODEL_ID_DEFAULT,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS_DEFAULT, temperature=TEMPERATURE_DEFAULT)

    augmenter = None
    if AUGMENT_ENABLE_DEFAULT:
        if LLMDistillationAugmenter is None:
            raise RuntimeError("Augmentation is enabled but LLMDistillationAugmenter could not be imported.")
        augmenter = LLMDistillationAugmenter(
            llm=llm,
            tokenizer=tokenizer,
            model_id=MODEL_ID_DEFAULT,
            max_attempts=AUGMENT_MAX_ATTEMPTS_DEFAULT,
            max_new_tokens=AUGMENT_MAX_NEW_TOKENS_DEFAULT,
            temperature=0.2,
            top_p=0.9,
            enable_validation=True,
            validation_max_new_tokens=AUGMENT_VALIDATION_MAX_NEW_TOKENS_DEFAULT,
        )

    # ======================================================
    # Step 5) Prefilter -> score real -> gate -> recover high/mid -> pairwise
    # ======================================================
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    pairwise_output_path.parent.mkdir(parents=True, exist_ok=True)
    listwise_output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    total_prefilter_selected = 0
    total_scored_candidates = 0
    total_json_ok = 0
    total_fallback = 0
    kept_specs = 0
    dropped_by_coverage_gate_specs = 0
    dropped_after_augmentation_specs = 0
    shortage_specs_pre = 0
    shortage_specs_final = 0
    missing_high_total_pre = 0
    missing_mid_total_pre = 0
    missing_low_total_pre = 0
    missing_high_total_final = 0
    missing_mid_total_final = 0
    missing_low_total_final = 0
    augment_requested_high_total = 0
    augment_requested_mid_total = 0
    augment_created_high_total = 0
    augment_created_mid_total = 0
    augment_attempts_total = 0
    augment_rejected_empty_total = 0
    augment_rejected_duplicate_total = 0
    augment_rejected_validation_total = 0
    total_pairwise_rows = 0
    total_pair_disagreement = 0
    total_pair_strong_vs_boundary = 0
    total_pair_strong_vs_weak = 0
    total_pair_strong_vs_hard = 0
    total_disagreement_candidates = 0
    total_strong_candidates = 0
    total_boundary_candidates = 0
    total_weak_candidates = 0
    synthetic_id_cursor = 1

    try:
        from tqdm import tqdm as _tqdm  # type: ignore
    except Exception:
        _tqdm = None

    base_iter = enumerate(specs, start=1)
    spec_iter = _tqdm(base_iter, total=len(specs), desc="Scoring specs") if _tqdm is not None else base_iter

    with (
        raw_output_path.open("w", encoding="utf-8") as raw_f,
        pairwise_output_path.open("w", encoding="utf-8") as pair_f,
        listwise_output_path.open("w", encoding="utf-8") as list_f,
    ):
        for spec_rank, spec in spec_iter:
            grant_id = _clean_text(spec.get("grant_id"))
            spec_idx = _safe_int(spec.get("spec_idx"), default=0, minimum=0, maximum=50_000_000)
            spec_text = _clean_text(spec.get("spec_text"))
            ranked_indices = list(prefilter_map.get((grant_id, spec_idx)) or [])
            rank_map = {int(idx): int(i) for i, idx in enumerate(ranked_indices)}

            rng = _rng_for_spec(base_seed=SEED_DEFAULT, grant_id=grant_id, spec_idx=spec_idx)
            high_indices, mid_indices, low_indices = _select_prefilter_buckets(
                total=len(fac_specs),
                ranked_indices=ranked_indices,
                high_k=prefilter_high,
                mid_k=prefilter_mid,
                low_k=prefilter_low,
                rng=rng,
            )
            high_index_set = set(high_indices)
            mid_index_set = set(mid_indices)

            prefilter_candidates: List[Dict[str, Any]] = []
            for idx in list(high_indices) + list(mid_indices) + list(low_indices):
                if idx < 0 or idx >= len(fac_specs):
                    continue
                fac = fac_specs[idx]
                bucket = "low"
                if idx in high_index_set:
                    bucket = "high"
                elif idx in mid_index_set:
                    bucket = "mid"
                prefilter_candidates.append(
                    {
                        "fac_id": int(fac["fac_id"]),
                        "fac_spec_id": int(fac["fac_spec_id"]),
                        "fac_spec_idx": int(fac["fac_spec_idx"]),
                        "section": _clean_text(fac["section"]),
                        "fac_spec_text": _clean_text(fac["fac_spec_text"]),
                        "prefilter_bucket": bucket,
                        "sts_rank": int(rank_map.get(int(idx), -1)),
                    }
                )

            total_prefilter_selected += int(len(prefilter_candidates))

            scored_candidates, json_ok, fallback = _score_prefiltered_candidates(
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                spec_text=spec_text,
                candidates=prefilter_candidates,
                batch_size=BATCH_SIZE_DEFAULT,
            )
            total_scored_candidates += int(len(scored_candidates))
            total_json_ok += int(json_ok)
            total_fallback += int(fallback)

            scored_candidates = _normalize_llm_scores(scored_candidates)
            scored_candidates, quota_info = _assign_target_clusters(
                candidates=scored_candidates,
                target_high=target_high,
                target_mid=target_mid,
                target_low=target_low,
            )
            missing_high_total_pre += int(quota_info["missing_high"])
            missing_mid_total_pre += int(quota_info["missing_mid"])
            missing_low_total_pre += int(quota_info["missing_low"])
            if int(quota_info["missing_total"]) > 0:
                shortage_specs_pre += 1

            high_cov_pre = _coverage_ratio(int(quota_info["selected_high"]), int(target_high))
            mid_cov_pre = _coverage_ratio(int(quota_info["selected_mid"]), int(target_mid))
            passed_coverage_gate = bool(
                high_cov_pre >= float(COVERAGE_GATE_MIN_MID_HIGH_DEFAULT)
                and mid_cov_pre >= float(COVERAGE_GATE_MIN_MID_HIGH_DEFAULT)
            )
            if not passed_coverage_gate:
                dropped_by_coverage_gate_specs += 1
                continue

            augment_info = {
                "requested_high": 0,
                "requested_mid": 0,
                "created_high": 0,
                "created_mid": 0,
                "attempts_total": 0,
                "rejected_empty": 0,
                "rejected_duplicate": 0,
                "rejected_validation": 0,
                "unfilled_high": 0,
                "unfilled_mid": 0,
            }
            if (
                augmenter is not None
                and (int(quota_info["missing_high"]) > 0 or int(quota_info["missing_mid"]) > 0)
            ):
                synthetic_rows, augment_info, synthetic_id_cursor = _augment_missing_high_mid_for_spec(
                    augmenter=augmenter,
                    spec_text=spec_text,
                    existing_candidates=scored_candidates,
                    missing_high=int(quota_info["missing_high"]),
                    missing_mid=int(quota_info["missing_mid"]),
                    next_synthetic_id=synthetic_id_cursor,
                )
                if synthetic_rows:
                    scored_candidates.extend(synthetic_rows)

            augment_requested_high_total += int(augment_info["requested_high"])
            augment_requested_mid_total += int(augment_info["requested_mid"])
            augment_created_high_total += int(augment_info["created_high"])
            augment_created_mid_total += int(augment_info["created_mid"])
            augment_attempts_total += int(augment_info["attempts_total"])
            augment_rejected_empty_total += int(augment_info["rejected_empty"])
            augment_rejected_duplicate_total += int(augment_info["rejected_duplicate"])
            augment_rejected_validation_total += int(augment_info["rejected_validation"])

            final_quota_info = _summarize_selected_clusters(
                candidates=scored_candidates,
                target_high=target_high,
                target_mid=target_mid,
                target_low=target_low,
            )

            if int(final_quota_info["missing_high"]) > 0 or int(final_quota_info["missing_mid"]) > 0:
                dropped_after_augmentation_specs += 1
                continue

            missing_high_total_final += int(final_quota_info["missing_high"])
            missing_mid_total_final += int(final_quota_info["missing_mid"])
            missing_low_total_final += int(final_quota_info["missing_low"])
            if int(final_quota_info["missing_total"]) > 0:
                shortage_specs_final += 1

            scored_candidates = _attach_sts_rank_percentile(scored_candidates)

            for c in scored_candidates:
                score_raw = float(c.get("llm_score_raw") or 0.0)
                c["band"] = _band_from_raw_score(score_raw)
                c["is_disagreement"] = bool(_is_disagreement_candidate(c))
                c["type"] = "hard_negative" if bool(c["is_disagreement"]) else _clean_text(c["band"])

            total_disagreement_candidates += int(sum(1 for c in scored_candidates if bool(c.get("is_disagreement"))))
            total_strong_candidates += int(sum(1 for c in scored_candidates if _clean_text(c.get("band")) == "strong"))
            total_boundary_candidates += int(sum(1 for c in scored_candidates if _clean_text(c.get("band")) == "boundary"))
            total_weak_candidates += int(sum(1 for c in scored_candidates if _clean_text(c.get("band")) == "weak"))

            pair_rows, pair_info = _build_controlled_pairwise_records(
                grant_id=grant_id,
                spec_idx=int(spec_idx),
                query_text=spec_text,
                candidates=scored_candidates,
            )
            for prow in pair_rows:
                pair_f.write(json.dumps(prow, ensure_ascii=False) + "\n")
            total_pairwise_rows += int(pair_info.get("pair_total", 0))
            total_pair_disagreement += int(pair_info.get("pair_disagreement", 0))
            total_pair_strong_vs_boundary += int(pair_info.get("pair_strong_vs_boundary", 0))
            total_pair_strong_vs_weak += int(pair_info.get("pair_strong_vs_weak", 0))
            total_pair_strong_vs_hard += int(pair_info.get("pair_strong_vs_hard", 0))

            kept_specs += 1

            raw_obj = {
                "grant_id": grant_id,
                "spec_idx": int(spec_idx),
                "spec_text": spec_text,
                "candidates": [
                    {
                        "fac_id": int(c.get("fac_id") or 0),
                        "fac_spec_id": int(c.get("fac_spec_id") or 0),
                        "fac_spec_idx": int(c.get("fac_spec_idx") or 0),
                        "section": _clean_text(c.get("section")) or "unknown",
                        "fac_spec_text": _clean_text(c.get("fac_spec_text")),
                        "sts_rank": int(c.get("sts_rank", -1)),
                        "score_raw": float(c.get("llm_score_raw") or 0.0),
                        "score": float(c.get("llm_score_norm") if c.get("llm_score_norm") is not None else c.get("llm_score_raw") or 0.0),
                        "band": _clean_text(c.get("band")) or "unknown",
                        "type": _clean_text(c.get("type")) or "unknown",
                        "is_disagreement": bool(c.get("is_disagreement", False)),
                        "is_augmented": bool(c.get("is_augmented", False)),
                        "augment_target_cluster": _clean_text(c.get("llm_target_cluster")) if bool(c.get("is_augmented", False)) else "",
                        "augment_validation_score": (
                            float(c.get("llm_score_raw") or 0.0) if bool(c.get("is_augmented", False)) else None
                        ),
                        "augment_validation_pass": bool(c.get("is_augmented", False)),
                    }
                    for c in scored_candidates
                ],
            }
            raw_f.write(json.dumps(raw_obj, ensure_ascii=False) + "\n")

            sorted_candidates = sorted(
                scored_candidates,
                key=lambda x: float(x.get("llm_score_norm") if x.get("llm_score_norm") is not None else x.get("llm_score_raw") or 0.0),
                reverse=True,
            )
            listwise_obj = {
                "grant_id": grant_id,
                "spec_idx": int(spec_idx),
                "query_text": spec_text,
                "docs": [
                    {
                        "text": _clean_text(d.get("fac_spec_text")),
                        "teacher_score": float(d.get("llm_score_norm") if d.get("llm_score_norm") is not None else d.get("llm_score_raw") or 0.0),
                        "teacher_score_raw": float(d.get("llm_score_raw") or 0.0),
                        "fac_id": int(d.get("fac_id") or 0),
                        "fac_spec_id": int(d.get("fac_spec_id") or 0),
                        "fac_spec_idx": int(d.get("fac_spec_idx") or 0),
                        "section": _clean_text(d.get("section")) or "unknown",
                        "sts_rank": int(d.get("sts_rank", -1)),
                        "band": _clean_text(d.get("band")) or "unknown",
                        "type": _clean_text(d.get("type")) or "unknown",
                        "is_disagreement": bool(d.get("is_disagreement", False)),
                        "is_augmented": bool(d.get("is_augmented", False)),
                        "augment_target_cluster": _clean_text(d.get("llm_target_cluster")) if bool(d.get("is_augmented", False)) else "",
                        "augment_validation_score": (
                            float(d.get("llm_score_raw") or 0.0) if bool(d.get("is_augmented", False)) else None
                        ),
                        "augment_validation_pass": bool(d.get("is_augmented", False)),
                    }
                    for d in sorted_candidates
                ],
            }
            list_f.write(json.dumps(listwise_obj, ensure_ascii=False) + "\n")

            if (_tqdm is None) and (spec_rank % 10 == 0 or spec_rank == len(specs)):
                elapsed = max(1e-6, time.time() - started)
                speed = float(total_scored_candidates / elapsed)
                print(
                    f"spec_progress={spec_rank}/{len(specs)} "
                    f"scored_candidates={total_scored_candidates} "
                    f"kept_specs={kept_specs} "
                    f"speed={speed:.2f} cand/sec"
                )

    # ======================================================
    # Step 6) Save manifest + print summary
    # ======================================================
    elapsed = max(1e-6, time.time() - started)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "grant_db": str(grant_db),
        "fac_db": str(fac_db),
        "prefilter_cache": str(prefilter_cache),
        "raw_output": str(raw_output_path),
        "pairwise_output_path": str(pairwise_output_path),
        "pairwise_output": str(pairwise_output_path),
        "listwise_output": str(listwise_output_path),
        "manifest_output": str(manifest_path),
        "output_path": str(raw_output_path),
        "model_id": MODEL_ID_DEFAULT,
        "batch_size": int(BATCH_SIZE_DEFAULT),
        "max_new_tokens": int(MAX_NEW_TOKENS_DEFAULT),
        "temperature": float(TEMPERATURE_DEFAULT),
        "prefilter_multiplier": float(prefilter_multiplier),
        "prefilter_requested": {
            "high": int(prefilter_high),
            "mid": int(prefilter_mid),
            "low": int(prefilter_low),
        },
        "target_requested": {
            "high": int(target_high),
            "mid": int(target_mid),
            "low": int(target_low),
        },
        "spec_count_total": int(len(specs)),
        "spec_count": int(len(specs)),
        "spec_count_kept": int(kept_specs),
        "spec_count_dropped_by_coverage_gate": int(dropped_by_coverage_gate_specs),
        "spec_count_dropped_after_augmentation": int(dropped_after_augmentation_specs),
        "fac_spec_count": int(len(fac_specs)),
        "cache_spec_count": int(len(prefilter_map)),
        "total_prefilter_selected_candidates": int(total_prefilter_selected),
        "total_scored_candidates": int(total_scored_candidates),
        "score_band_ranges": {
            "high_min": float(HIGH_SCORE_MIN_DEFAULT),
            "mid_min": float(MID_SCORE_MIN_DEFAULT),
            "mid_max": float(MID_SCORE_MAX_DEFAULT),
        },
        "coverage_gate_min_mid_high": float(COVERAGE_GATE_MIN_MID_HIGH_DEFAULT),
        "target_missing_high_total_pre_augmentation": int(missing_high_total_pre),
        "target_missing_mid_total_pre_augmentation": int(missing_mid_total_pre),
        "target_missing_low_total_pre_augmentation": int(missing_low_total_pre),
        "target_shortage_specs_pre_augmentation": int(shortage_specs_pre),
        "target_missing_high_total_final_kept": int(missing_high_total_final),
        "target_missing_mid_total_final_kept": int(missing_mid_total_final),
        "target_missing_low_total_final_kept": int(missing_low_total_final),
        "target_shortage_specs_final_kept": int(shortage_specs_final),
        "target_missing_high_total": int(missing_high_total_final),
        "target_missing_mid_total": int(missing_mid_total_final),
        "target_missing_low_total": int(missing_low_total_final),
        "target_shortage_specs": int(shortage_specs_final),
        "augmentation_enabled": bool(augmenter is not None),
        "augmentation_requested_high_total": int(augment_requested_high_total),
        "augmentation_requested_mid_total": int(augment_requested_mid_total),
        "augmentation_created_high_total": int(augment_created_high_total),
        "augmentation_created_mid_total": int(augment_created_mid_total),
        "augmentation_attempts_total": int(augment_attempts_total),
        "augmentation_rejected_empty_total": int(augment_rejected_empty_total),
        "augmentation_rejected_duplicate_total": int(augment_rejected_duplicate_total),
        "augmentation_rejected_validation_total": int(augment_rejected_validation_total),
        "pair_max_per_spec": int(PAIR_MAX_PER_SPEC_DEFAULT),
        "pair_max_disagreement_per_spec": int(PAIR_MAX_DISAGREE_PER_SPEC_DEFAULT),
        "pair_max_boundary_per_spec": int(PAIR_MAX_BOUNDARY_PER_SPEC_DEFAULT),
        "pair_weak_min_per_spec": int(PAIR_WEAK_MIN_DEFAULT),
        "pair_disagree_top_percentile": float(DISAGREE_TOP_PERCENTILE_DEFAULT),
        "pair_disagree_low_score_norm_max": float(DISAGREE_LOW_SCORE_NORM_DEFAULT),
        "pair_disagree_min_margin": float(DISAGREE_MIN_MARGIN_DEFAULT),
        "pair_boundary_min_margin": float(BOUNDARY_MIN_MARGIN_DEFAULT),
        "allow_extra_disagreement_from_unselected": bool(ALLOW_EXTRA_DISAGREE_FROM_UNSELECTED_DEFAULT),
        "total_pairwise_rows": int(total_pairwise_rows),
        "total_pair_disagreement": int(total_pair_disagreement),
        "total_pair_strong_vs_boundary": int(total_pair_strong_vs_boundary),
        "total_pair_strong_vs_weak": int(total_pair_strong_vs_weak),
        "total_pair_strong_vs_hard": int(total_pair_strong_vs_hard),
        "total_disagreement_candidates": int(total_disagreement_candidates),
        "total_strong_candidates": int(total_strong_candidates),
        "total_boundary_candidates": int(total_boundary_candidates),
        "total_weak_candidates": int(total_weak_candidates),
        "parsed_json_ok_count": int(total_json_ok),
        "parsed_fallback_count": int(total_fallback),
        "elapsed_seconds": float(elapsed),
        "candidates_per_second": float(total_scored_candidates / elapsed),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"spec_count_total={len(specs)}")
    print(f"spec_count_kept={kept_specs}")
    print(f"spec_count_dropped_by_coverage_gate={dropped_by_coverage_gate_specs}")
    print(f"spec_count_dropped_after_augmentation={dropped_after_augmentation_specs}")
    print(f"fac_spec_count={len(fac_specs)}")
    print(f"prefilter_multiplier={prefilter_multiplier:.4f}")
    print(f"prefilter_requested=high:{prefilter_high},mid:{prefilter_mid},low:{prefilter_low}")
    print(f"target_requested=high:{target_high},mid:{target_mid},low:{target_low}")
    print(
        "score_band_ranges="
        f"high:[{HIGH_SCORE_MIN_DEFAULT:.2f},1.00],"
        f"mid:[{MID_SCORE_MIN_DEFAULT:.2f},{MID_SCORE_MAX_DEFAULT:.2f}],"
        f"low:[0.00,{MID_SCORE_MIN_DEFAULT:.2f})"
    )
    print(f"coverage_gate_min_mid_high={COVERAGE_GATE_MIN_MID_HIGH_DEFAULT:.2f}")
    print(f"cache_spec_count={len(prefilter_map)}")
    print(f"total_prefilter_selected_candidates={total_prefilter_selected}")
    print(f"total_scored_candidates={total_scored_candidates}")
    print(
        "pairwise_rows="
        f"total:{total_pairwise_rows},"
        f"disagreement:{total_pair_disagreement},"
        f"strong_vs_boundary:{total_pair_strong_vs_boundary},"
        f"strong_vs_weak:{total_pair_strong_vs_weak},"
        f"strong_vs_hard:{total_pair_strong_vs_hard}"
    )
    print(f"total_disagreement_candidates={total_disagreement_candidates}")
    print(
        "candidate_bands="
        f"high:{total_strong_candidates},"
        f"mid:{total_boundary_candidates},"
        f"low:{total_weak_candidates}"
    )
    print(
        "target_missing_pairs_pre_augmentation="
        f"high:{missing_high_total_pre},"
        f"mid:{missing_mid_total_pre},"
        f"low:{missing_low_total_pre}"
    )
    print(
        "target_missing_pairs_final_kept="
        f"high:{missing_high_total_final},"
        f"mid:{missing_mid_total_final},"
        f"low:{missing_low_total_final}"
    )
    print(f"target_shortage_specs_pre_augmentation={shortage_specs_pre}/{len(specs)}")
    print(f"target_shortage_specs_final_kept={shortage_specs_final}/{max(1, kept_specs)}")
    print(
        "augmentation_totals="
        f"requested_high:{augment_requested_high_total},"
        f"requested_mid:{augment_requested_mid_total},"
        f"created_high:{augment_created_high_total},"
        f"created_mid:{augment_created_mid_total},"
        f"attempts:{augment_attempts_total},"
        f"rejected_empty:{augment_rejected_empty_total},"
        f"rejected_duplicate:{augment_rejected_duplicate_total},"
        f"rejected_validation:{augment_rejected_validation_total}"
    )
    print(f"parsed_json_ok_count={total_json_ok}")
    print(f"parsed_fallback_count={total_fallback}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"candidates_per_second={(total_scored_candidates / elapsed):.4f}")
    print(f"raw_output={raw_output_path}")
    print(f"pairwise_output={pairwise_output_path}")
    print(f"listwise_output={listwise_output_path}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
