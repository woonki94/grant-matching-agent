from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce2").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce2.data_preparation.utils import (  # noqa: E402
    ASPECTS,
    DECOMPOSITION_OUTPUT_DEFAULT,
    FAC_DB_DEFAULT,
    GRANT_DB_DEFAULT,
    MAX_FAC_SPECS_DEFAULT,
    MAX_GRANT_SPECS_DEFAULT,
    PREFILTER_CACHE_BATCH_SIZE_DEFAULT,
    PREFILTER_CACHE_MANIFEST_DEFAULT,
    PREFILTER_CACHE_MAX_LENGTH_DEFAULT,
    PREFILTER_CACHE_MODEL_ID_DEFAULT,
    PREFILTER_CACHE_OUTPUT_DEFAULT,
    PREFILTER_CACHE_TOP_K_PER_ASPECT_DEFAULT,
    SEED_DEFAULT,
    iter_topk_candidates,
    load_fac_specs,
    load_grant_specs,
    load_jsonl_by_key,
    load_specs_from_decompositions,
    prefilter_cache_paths,
    resolve_path,
    score_query_against_docs,
    _aspect_text_for_item,
    _pick_device,
)
from ce2.data_preparation.llm_runtime import clean_text  # noqa: E402


GRANT_DB_CE2_DEFAULT = "ce2/dataset/source/grant_keywords_spec_keywords_db.json"
FAC_DB_CE2_DEFAULT = "ce2/dataset/source/fac_specs_db.json"


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


def _try_tqdm(total: int) -> Any:
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, desc="CE prefilter", unit="grant", dynamic_ncols=True)
    except Exception:
        return None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build the CE2 prefilter cache using cross-encoder pair logits."
        )
    )
    p.add_argument("--model-id", type=str, default=PREFILTER_CACHE_MODEL_ID_DEFAULT)
    p.add_argument("--grant-db", type=str, default=GRANT_DB_CE2_DEFAULT)
    p.add_argument("--fac-db", type=str, default=FAC_DB_CE2_DEFAULT)
    p.add_argument("--decomposition-output", type=str, default=DECOMPOSITION_OUTPUT_DEFAULT)
    p.add_argument("--output", type=str, default=PREFILTER_CACHE_OUTPUT_DEFAULT)
    p.add_argument("--output-domain", type=str, default="")
    p.add_argument("--output-method", type=str, default="")
    p.add_argument("--output-target", type=str, default="")
    p.add_argument("--manifest", type=str, default=PREFILTER_CACHE_MANIFEST_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--max-grant-specs", type=int, default=MAX_GRANT_SPECS_DEFAULT)
    p.add_argument("--max-fac-specs", type=int, default=MAX_FAC_SPECS_DEFAULT)
    p.add_argument("--top-k-per-aspect", type=int, default=PREFILTER_CACHE_TOP_K_PER_ASPECT_DEFAULT)
    p.add_argument("--batch-size", type=int, default=PREFILTER_CACHE_BATCH_SIZE_DEFAULT)
    p.add_argument("--max-length", type=int, default=PREFILTER_CACHE_MAX_LENGTH_DEFAULT)
    return p


def _derive_aspect_output_paths(
    *,
    output_base_path: Path,
    output_domain: str,
    output_method: str,
    output_target: str,
) -> Dict[str, Path]:
    out = prefilter_cache_paths(output_base_path)
    if clean_text(output_domain):
        out["domain"] = resolve_path(PROJECT_ROOT, output_domain)
    if clean_text(output_method):
        out["method"] = resolve_path(PROJECT_ROOT, output_method)
    if clean_text(output_target):
        out["target"] = resolve_path(PROJECT_ROOT, output_target)
    return out


def main() -> int:
    args = _build_parser().parse_args()
    started = time.time()

    decomposition_path = resolve_path(PROJECT_ROOT, args.decomposition_output)
    output_base_path = resolve_path(PROJECT_ROOT, args.output)
    aspect_output_paths = _derive_aspect_output_paths(
        output_base_path=output_base_path,
        output_domain=args.output_domain,
        output_method=args.output_method,
        output_target=args.output_target,
    )
    manifest_path = resolve_path(PROJECT_ROOT, args.manifest)

    if not decomposition_path.exists():
        raise RuntimeError(f"Decomposition JSONL not found: {decomposition_path}")

    top_k_per_aspect = _safe_int(args.top_k_per_aspect, default=256, minimum=0, maximum=5_000_000)
    batch_size = _safe_int(args.batch_size, default=64, minimum=1, maximum=4096)
    max_length = _safe_int(args.max_length, default=256, minimum=32, maximum=4096)

    decompositions = load_jsonl_by_key(decomposition_path, "item_id")
    grant_specs, fac_specs = load_specs_from_decompositions(
        decompositions=decompositions,
        max_grant_specs=int(args.max_grant_specs),
        max_fac_specs=int(args.max_fac_specs),
        seed=int(args.seed),
    )
    spec_source = "decomposition"

    if not grant_specs or not fac_specs:
        grant_db = resolve_path(PROJECT_ROOT, args.grant_db)
        fac_db = resolve_path(PROJECT_ROOT, args.fac_db)
        if clean_text(args.grant_db) == GRANT_DB_CE2_DEFAULT and not grant_db.exists():
            grant_db = resolve_path(PROJECT_ROOT, GRANT_DB_DEFAULT)
        if clean_text(args.fac_db) == FAC_DB_CE2_DEFAULT and not fac_db.exists():
            fac_db = resolve_path(PROJECT_ROOT, FAC_DB_DEFAULT)
        if grant_db.exists() and fac_db.exists():
            grant_specs = load_grant_specs(grant_db, max_items=int(args.max_grant_specs), seed=int(args.seed))
            fac_specs = load_fac_specs(fac_db, max_items=int(args.max_fac_specs), seed=int(args.seed))
            spec_source = "decomposition+db_fallback"
        else:
            grant_db = None
            fac_db = None

    if not grant_specs:
        raise RuntimeError("No grant specs loaded from decomposition rows (and DB fallback unavailable).")
    if not fac_specs:
        raise RuntimeError("No faculty specs loaded from decomposition rows (and DB fallback unavailable).")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        raise RuntimeError("Missing dependencies: install torch and transformers.") from e

    device = _pick_device()
    model_ref = clean_text(args.model_id) or PREFILTER_CACHE_MODEL_ID_DEFAULT
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_ref, trust_remote_code=True)
    model.to(device)
    model.eval()

    fac_aspect_texts: Dict[str, List[str]] = {}
    for aspect in ASPECTS:
        fac_aspect_texts[aspect] = [
            _aspect_text_for_item(f, aspect=aspect, decompositions=decompositions)
            for f in fac_specs
        ]

    for path in aspect_output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    bar = _try_tqdm(total=len(grant_specs))
    rows_written = 0
    per_aspect_candidate_counts = {aspect: 0 for aspect in ASPECTS}

    with (
        aspect_output_paths["domain"].open("w", encoding="utf-8") as out_domain,
        aspect_output_paths["method"].open("w", encoding="utf-8") as out_method,
        aspect_output_paths["target"].open("w", encoding="utf-8") as out_target,
    ):
        writers = {
            "domain": out_domain,
            "method": out_method,
            "target": out_target,
        }
        for grant in grant_specs:
            for aspect in ASPECTS:
                query_text = _aspect_text_for_item(grant, aspect=aspect, decompositions=decompositions)
                logits = score_query_against_docs(
                    model=model,
                    tokenizer=tokenizer,
                    query_text=query_text,
                    docs=fac_aspect_texts[aspect],
                    device=device,
                    batch_size=batch_size,
                    max_length=max_length,
                )
                candidates = list(
                    iter_topk_candidates(
                        fac_specs=fac_specs,
                        logits=logits,
                        top_k=top_k_per_aspect,
                    )
                )
                per_aspect_candidate_counts[aspect] += int(len(candidates))
                row: Dict[str, Any] = {
                    "grant_item_id": grant.item_id,
                    "grant_id": grant.meta.get("grant_id"),
                    "grant_spec_idx": grant.meta.get("grant_spec_idx"),
                    "grant_text": grant.text,
                    "aspect": aspect,
                    "query_text": query_text,
                    "candidates": candidates,
                }
                writers[aspect].write(json.dumps(row, ensure_ascii=False) + "\n")

            rows_written += 1
            if bar is not None:
                bar.update(1)

    if bar is not None:
        bar.close()

    try:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    elapsed = max(1e-6, time.time() - started)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "true_cross_encoder",
        "spec_source": spec_source,
        "model_id": model_ref,
        "device": str(device),
        "grant_db": str(grant_db) if "grant_db" in locals() and grant_db is not None else "",
        "fac_db": str(fac_db) if "fac_db" in locals() and fac_db is not None else "",
        "decomposition_output": str(decomposition_path),
        "outputs": {k: str(v) for k, v in aspect_output_paths.items()},
        "aspects": list(ASPECTS),
        "seed": int(args.seed),
        "max_grant_specs": int(args.max_grant_specs),
        "max_fac_specs": int(args.max_fac_specs),
        "grant_specs_loaded": int(len(grant_specs)),
        "fac_specs_loaded": int(len(fac_specs)),
        "decompositions_loaded": int(len(decompositions)),
        "top_k_per_aspect": int(top_k_per_aspect),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "rows_written": int(rows_written),
        "per_aspect_candidate_counts": {k: int(v) for k, v in per_aspect_candidate_counts.items()},
        "total_candidates_written": int(sum(per_aspect_candidate_counts.values())),
        "elapsed_seconds": float(elapsed),
        "grants_per_second": float(rows_written / elapsed),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"output_domain={aspect_output_paths['domain']}")
    print(f"output_method={aspect_output_paths['method']}")
    print(f"output_target={aspect_output_paths['target']}")
    print(f"manifest={manifest_path}")
    print(f"device={device}")
    print(f"rows_written={rows_written}")
    print(f"total_candidates_written={manifest['total_candidates_written']}")
    print(f"elapsed_seconds={elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
