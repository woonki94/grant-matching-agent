from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "cross_encoder").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cross_encoder.data_preparation.llm_distillation.augmentation import LLMDistillationAugmenter
from cross_encoder.data_preparation.llm_distillation.llm_distill2 import (  # noqa: E402
    AUGMENT_BATCH_SIZE_DEFAULT,
    AUGMENT_MAX_ATTEMPTS_DEFAULT,
    AUGMENT_MAX_NEW_TOKENS_DEFAULT,
    AUGMENT_MAX_TRIES_PER_MISSING_DEFAULT,
    AUGMENT_VALIDATION_MAX_NEW_TOKENS_DEFAULT,
    GRANT_DB_DEFAULT,
    MODEL_ID_DEFAULT,
    _load_vllm_bundle,
    _release_vllm_bundle,
)

# ======================================================
# Eval constants (no CLI args)
# ======================================================
GRANT_DB_PATH = GRANT_DB_DEFAULT
MODEL_ID = "Qwen/Qwen3-14B"
RANDOM_SEED = 42
NEED_HIGH = 100
NEED_MID = 100
ENABLE_VALIDATION = False


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_path(value: Any) -> Path:
    p = Path(_clean_text(value)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


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
            out.append({"grant_id": grant_id, "spec_idx": int(spec_idx), "spec_text": text_value})
    return out


def _dedup_text_key(text: Any) -> str:
    return " ".join(_clean_text(text).lower().split())


def _augment_specs_batch(
    *,
    augmenter: LLMDistillationAugmenter,
    picked_specs: List[Dict[str, Any]],
    need_high: int,
    need_mid: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    results: List[Dict[str, Any]] = []
    stats = {
        "requested_high": int(max(0, int(need_high)) * len(picked_specs)),
        "requested_mid": int(max(0, int(need_mid)) * len(picked_specs)),
        "created_high": 0,
        "created_mid": 0,
        "attempts_total": 0,
        "rejected_empty": 0,
        "rejected_duplicate": 0,
        "rejected_validation": 0,
        "unfilled_high": 0,
        "unfilled_mid": 0,
    }
    if not picked_specs:
        return results, stats

    query_by_idx: Dict[int, str] = {i: _clean_text(spec.get("spec_text")) for i, spec in enumerate(picked_specs)}
    used_text_by_idx: Dict[int, set] = {i: set() for i in range(len(picked_specs))}
    accepted_by_idx: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(picked_specs))}

    max_tries = int(max(1, AUGMENT_MAX_TRIES_PER_MISSING_DEFAULT))
    for cluster, need in (("high", int(max(0, int(need_high)))), ("mid", int(max(0, int(need_mid))))):
        unresolved: List[int] = []
        for i in range(len(picked_specs)):
            for _ in range(need):
                unresolved.append(int(i))

        for _try in range(max_tries):
            if not unresolved:
                break
            jobs = [{"query": query_by_idx[int(spec_i)], "target_cluster": cluster} for spec_i in unresolved]
            stats["attempts_total"] += int(len(jobs))
            outs = augmenter.augment_batch(jobs, batch_size=int(AUGMENT_BATCH_SIZE_DEFAULT))
            next_unresolved: List[int] = []
            for spec_i, out in zip(unresolved, outs):
                spec_i = int(spec_i)
                augmented_text = _clean_text((out or {}).get("augmented_text"))
                if not augmented_text:
                    stats["rejected_empty"] += 1
                    next_unresolved.append(spec_i)
                    continue
                key = _dedup_text_key(augmented_text)
                if not key or key in used_text_by_idx[spec_i]:
                    stats["rejected_duplicate"] += 1
                    next_unresolved.append(spec_i)
                    continue
                validation = dict((out or {}).get("validation") or {})
                if bool(getattr(augmenter, "enable_validation", True)):
                    if not bool(validation.get("pass_valid_range")):
                        stats["rejected_validation"] += 1
                        next_unresolved.append(spec_i)
                        continue
                    score = float(validation.get("score") or 0.0)
                else:
                    score = 0.0
                accepted_by_idx[spec_i].append(
                    {
                        "cluster": cluster,
                        "text": augmented_text,
                        "score": score,
                    }
                )
                used_text_by_idx[spec_i].add(key)
                if cluster == "high":
                    stats["created_high"] += 1
                else:
                    stats["created_mid"] += 1
            unresolved = next_unresolved

        if unresolved:
            if cluster == "high":
                stats["unfilled_high"] += int(len(unresolved))
            else:
                stats["unfilled_mid"] += int(len(unresolved))

    for i, spec in enumerate(picked_specs):
        results.append(
            {
                "grant_id": _clean_text(spec.get("grant_id")),
                "spec_idx": int(spec.get("spec_idx") or 0),
                "query": _clean_text(spec.get("spec_text")),
                "rows": list(accepted_by_idx.get(i) or []),
            }
        )
    return results, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch augmentation eval over random grant specs.")
    parser.add_argument(
        "--num-specs",
        type=int,
        default=1,
        help="How many random specs to fetch from grant DB and augment.",
    )
    args = parser.parse_args()
    num_specs = max(1, int(args.num_specs))

    grant_db_path = _resolve_path(GRANT_DB_PATH)
    if not grant_db_path.exists():
        raise RuntimeError(f"Grant DB not found: {grant_db_path}")

    try:
        payload = json.loads(grant_db_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse grant db: {grant_db_path} ({type(exc).__name__}: {exc})") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected top-level JSON object: {grant_db_path}")

    specs = _flatten_specs(payload)
    if not specs:
        raise RuntimeError("No specs found in grant DB.")

    rng = random.Random(int(RANDOM_SEED))
    if num_specs >= len(specs):
        picked_specs = list(specs)
    else:
        picked_specs = rng.sample(specs, k=int(num_specs))

    print(f"fetched_specs={len(picked_specs)}")
    print(f"need_high={int(NEED_HIGH)} need_mid={int(NEED_MID)}")
    print(
        "augment_runtime="
        f"validation_enabled:{str(bool(ENABLE_VALIDATION)).lower()},"
        f"batch_size:{int(AUGMENT_BATCH_SIZE_DEFAULT)},"
        f"max_attempts:{int(AUGMENT_MAX_ATTEMPTS_DEFAULT)},"
        f"max_tries_per_missing:{int(AUGMENT_MAX_TRIES_PER_MISSING_DEFAULT)},"
        f"gen_tokens:{int(AUGMENT_MAX_NEW_TOKENS_DEFAULT)},"
        f"val_tokens:{int(AUGMENT_VALIDATION_MAX_NEW_TOKENS_DEFAULT)}"
    )

    llm = None
    tokenizer = None
    try:
        llm, tokenizer, _sampling = _load_vllm_bundle(
            model_id=_clean_text(MODEL_ID) or MODEL_ID_DEFAULT,
            max_new_tokens=AUGMENT_MAX_NEW_TOKENS_DEFAULT,
            temperature=0.2,
        )
        augmenter = LLMDistillationAugmenter(
            llm=llm,
            tokenizer=tokenizer,
            model_id=_clean_text(MODEL_ID) or MODEL_ID_DEFAULT,
            max_attempts=AUGMENT_MAX_ATTEMPTS_DEFAULT,
            max_new_tokens=AUGMENT_MAX_NEW_TOKENS_DEFAULT,
            temperature=0.2,
            top_p=0.9,
            enable_validation=bool(ENABLE_VALIDATION),
            validation_max_new_tokens=AUGMENT_VALIDATION_MAX_NEW_TOKENS_DEFAULT,
        )
        gen_started = time.perf_counter()
        per_spec_results, stats = _augment_specs_batch(
            augmenter=augmenter,
            picked_specs=picked_specs,
            need_high=max(0, int(NEED_HIGH)),
            need_mid=max(0, int(NEED_MID)),
        )
        generation_seconds = max(1e-9, float(time.perf_counter() - gen_started))
    finally:
        _release_vllm_bundle(llm)

    for i, item in enumerate(per_spec_results, start=1):
        grant_id = _clean_text(item.get("grant_id"))
        spec_idx = int(item.get("spec_idx") or 0)
        query = _clean_text(item.get("query"))
        rows = list(item.get("rows") or [])
        high_rows = [r for r in rows if _clean_text(r.get("cluster")) == "high"]
        mid_rows = [r for r in rows if _clean_text(r.get("cluster")) == "mid"]
        print(f"spec[{i}] grant_id={grant_id} spec_idx={spec_idx}")
        print(f"query={query}")
        for j, row in enumerate(high_rows, start=1):
            print(f"high[{j}] score={float(row.get('score') or 0.0):.4f} text={_clean_text(row.get('text'))}")
        for j, row in enumerate(mid_rows, start=1):
            print(f"mid[{j}] score={float(row.get('score') or 0.0):.4f} text={_clean_text(row.get('text'))}")
        print(
            "per_spec_summary="
            f"created_high:{len(high_rows)}/{max(0, int(NEED_HIGH))},"
            f"created_mid:{len(mid_rows)}/{max(0, int(NEED_MID))}"
        )

    print(
        "summary="
        f"created_high:{int(stats.get('created_high', 0))}/{max(0, int(NEED_HIGH)) * len(picked_specs)},"
        f"created_mid:{int(stats.get('created_mid', 0))}/{max(0, int(NEED_MID)) * len(picked_specs)},"
        f"attempts:{int(stats.get('attempts_total', 0))},"
        f"rejected_validation:{int(stats.get('rejected_validation', 0))},"
        f"rejected_duplicate:{int(stats.get('rejected_duplicate', 0))},"
        f"rejected_empty:{int(stats.get('rejected_empty', 0))},"
        f"unfilled_high:{int(stats.get('unfilled_high', 0))},"
        f"unfilled_mid:{int(stats.get('unfilled_mid', 0))}"
    )
    total_requested = int((max(0, int(NEED_HIGH)) + max(0, int(NEED_MID))) * len(picked_specs))
    print(
        "timing="
        f"generation_seconds:{generation_seconds:.4f},"
        f"requested_slots:{total_requested},"
        f"slots_per_second:{(float(total_requested) / generation_seconds):.4f},"
        f"attempts_per_second:{(float(int(stats.get('attempts_total', 0))) / generation_seconds):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
