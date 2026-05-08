from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List


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
    _augment_missing_high_mid_for_spec,
)

# ======================================================
# Eval constants (no CLI args)
# ======================================================
GRANT_DB_PATH = GRANT_DB_DEFAULT
MODEL_ID = "Qwen/Qwen3-14B"
RANDOM_SEED = 42
NEED_HIGH = 2
NEED_MID = 2


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


def main() -> int:
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
    picked = specs[rng.randrange(len(specs))]
    query = _clean_text(picked.get("spec_text"))
    grant_id = _clean_text(picked.get("grant_id"))
    spec_idx = int(picked.get("spec_idx") or 0)

    print(f"picked_grant_id={grant_id}")
    print(f"picked_spec_idx={spec_idx}")
    print(f"query={query}")
    print(f"need_high={int(NEED_HIGH)} need_mid={int(NEED_MID)}")
    print(
        "augment_runtime="
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
            enable_validation=True,
            validation_max_new_tokens=AUGMENT_VALIDATION_MAX_NEW_TOKENS_DEFAULT,
        )

        synthetic_rows, stats, _ = _augment_missing_high_mid_for_spec(
            augmenter=augmenter,
            spec_text=query,
            existing_candidates=[],
            missing_high=max(0, int(NEED_HIGH)),
            missing_mid=max(0, int(NEED_MID)),
            next_synthetic_id=1,
        )
    finally:
        _release_vllm_bundle(llm)

    high_rows = [r for r in synthetic_rows if _clean_text(r.get("llm_target_cluster")) == "high"]
    mid_rows = [r for r in synthetic_rows if _clean_text(r.get("llm_target_cluster")) == "mid"]

    print("generated:")
    for i, row in enumerate(high_rows, start=1):
        print(f"high[{i}] score={float(row.get('llm_score_raw') or 0.0):.4f} text={_clean_text(row.get('fac_spec_text'))}")
    for i, row in enumerate(mid_rows, start=1):
        print(f"mid[{i}] score={float(row.get('llm_score_raw') or 0.0):.4f} text={_clean_text(row.get('fac_spec_text'))}")

    print(
        "summary="
        f"created_high:{len(high_rows)}/{max(0, int(NEED_HIGH))},"
        f"created_mid:{len(mid_rows)}/{max(0, int(NEED_MID))},"
        f"attempts:{int(stats.get('attempts_total', 0))},"
        f"rejected_validation:{int(stats.get('rejected_validation', 0))},"
        f"rejected_duplicate:{int(stats.get('rejected_duplicate', 0))},"
        f"rejected_empty:{int(stats.get('rejected_empty', 0))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
