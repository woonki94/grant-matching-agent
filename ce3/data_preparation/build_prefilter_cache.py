from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce3").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce3.data_preparation.llm_runtime import normalize_ws  # noqa: E402
from ce3.data_preparation.utils import (  # noqa: E402
    SpecItem,
    cached_row_matches_item,
    load_faculty_specializations,
    load_grant_specializations,
    load_jsonl_by_key,
    resolve_path,
)

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # type: ignore[assignment]


MODEL_ID_DEFAULT = "dleemiller/ModernCE-base-sts"
GRANT_DB_DEFAULT = "ce3/dataset/source/grant_keywords_spec_keywords_db.json"
FAC_DB_DEFAULT = "ce3/dataset/source/fac_specs_db.json"
DECOMPOSITION_OUTPUT_DEFAULT = "ce3/dataset/decomposed/spec_decompositions_topic_approach_objective.jsonl"
OUTPUT_BASE_DEFAULT = "ce3/dataset/source/prefilter_cache.jsonl"
SEED_DEFAULT = 42
MAX_GRANT_SPECS_DEFAULT = 0
MAX_FAC_SPECS_DEFAULT = 0
BATCH_SIZE_DEFAULT = 64
MAX_LENGTH_DEFAULT = 256
ASPECTS = ("topic", "approach", "objective")


def prefilter_cache_paths(output_base_path: Path) -> Dict[str, Path]:
    stem = output_base_path.stem if output_base_path.suffix else output_base_path.name
    return {
        aspect: (output_base_path.parent / f"{stem}_{aspect}.jsonl").resolve()
        for aspect in ASPECTS
    }


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [normalize_ws(x) for x in value if normalize_ws(x)]
    if isinstance(value, str) and normalize_ws(value):
        return [normalize_ws(value)]
    return []


def _safe_decomposition(row: Dict[str, Any]) -> Dict[str, List[str]]:
    decomp = row.get("decomposition") if isinstance(row, dict) else {}
    if not isinstance(decomp, dict):
        return {aspect: [] for aspect in ASPECTS}
    return {aspect: _as_list(decomp.get(aspect)) for aspect in ASPECTS}


def _aspect_text(item: SpecItem, decompositions: Dict[str, Dict[str, Any]], aspect: str) -> str:
    decomp = _safe_decomposition(decompositions.get(item.item_id, {}))
    text = " ".join(decomp.get(aspect, []))
    return normalize_ws(text)


def _validate_coverage(items: Sequence[SpecItem], decompositions: Dict[str, Dict[str, Any]], *, label: str) -> None:
    bad: List[str] = []
    for item in items:
        row = decompositions.get(item.item_id)
        if row is None:
            reason = "missing"
        elif not cached_row_matches_item(row, item):
            reason = "stale_or_mismatched_text"
        else:
            continue
        if len(bad) < 5:
            bad.append(f"{reason}: {item.item_id} text={item.text[:140]}")
    if bad:
        raise RuntimeError(
            f"Decomposition coverage failed for {label}. Examples:\n"
            + "\n".join(bad)
            + "\nRun CE3 decomposition on the same source DBs before building prefilter cache."
        )


def _sigmoid(value: float) -> float:
    x = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-x))


def _load_cross_encoder(model_id: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _score_pairs(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    query_texts: Sequence[str],
    doc_texts: Sequence[str],
    batch_size: int,
    max_length: int,
) -> List[float]:
    import torch

    scores: List[float] = []
    step = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, len(doc_texts), step):
            docs = list(doc_texts[start : start + step])
            queries = list(query_texts[start : start + step])
            scored_positions = [
                pos
                for pos, (query, doc) in enumerate(zip(queries, docs))
                if normalize_ws(query) and normalize_ws(doc)
            ]
            if not scored_positions:
                scores.extend(0.0 for _ in docs)
                continue
            scored_queries = [queries[pos] for pos in scored_positions]
            scored_docs = [docs[pos] for pos in scored_positions]
            enc = tokenizer(
                scored_queries,
                scored_docs,
                max_length=int(max_length),
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            if logits.ndim == 2 and logits.shape[-1] > 1:
                vals = logits[:, -1]
            else:
                vals = logits.squeeze(-1)
            batch_scores = [0.0 for _ in docs]
            for pos, value in zip(scored_positions, vals.detach().cpu().tolist()):
                batch_scores[pos] = float(_sigmoid(value))
            scores.extend(batch_scores)
    return scores


def _build_all_candidates(
    *,
    fac_specs: Sequence[SpecItem],
    fac_texts: Sequence[str],
    scores: Sequence[float],
) -> List[Dict[str, Any]]:
    ranked = sorted(range(len(fac_specs)), key=lambda i: (-float(scores[i]), fac_specs[i].item_id))
    candidates: List[Dict[str, Any]] = []
    for rank, idx in enumerate(ranked, start=1):
        fac = fac_specs[idx]
        candidates.append(
            {
                "rank": int(rank),
                "fac_item_id": fac.item_id,
                "fac_text": fac.text,
                "fac_meta": fac.meta,
                "doc_text": fac_texts[idx],
                "ce_score": float(scores[idx]),
            }
        )
    return candidates


def build_prefilter_cache(
    *,
    grant_specs: Sequence[SpecItem],
    fac_specs: Sequence[SpecItem],
    decompositions: Dict[str, Dict[str, Any]],
    output_base_path: Path,
    model_id: str,
    batch_size: int,
    max_length: int,
) -> Dict[str, Any]:
    model, tokenizer, device = _load_cross_encoder(model_id)
    output_paths = prefilter_cache_paths(output_base_path)
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    fac_texts_by_aspect = {
        aspect: [_aspect_text(fac, decompositions, aspect) for fac in fac_specs]
        for aspect in ASPECTS
    }
    written = {aspect: 0 for aspect in ASPECTS}
    iterator: Iterable[tuple[str, SpecItem]] = (
        (aspect, grant)
        for aspect in ASPECTS
        for grant in grant_specs
    )
    total = len(ASPECTS) * len(grant_specs)
    if tqdm is not None:
        iterator = tqdm(iterator, total=total, desc="CE3 prefilter", unit="grant-aspect", dynamic_ncols=True)

    open_files = {
        aspect: output_paths[aspect].open("a", encoding="utf-8")
        for aspect in ASPECTS
    }
    try:
        for aspect, grant in iterator:
            query_text = _aspect_text(grant, decompositions, aspect)
            query_texts = [query_text] * len(fac_specs)
            scores = _score_pairs(
                model=model,
                tokenizer=tokenizer,
                device=device,
                query_texts=query_texts,
                doc_texts=fac_texts_by_aspect[aspect],
                batch_size=batch_size,
                max_length=max_length,
            )
            candidates = _build_all_candidates(
                fac_specs=fac_specs,
                fac_texts=fac_texts_by_aspect[aspect],
                scores=scores,
            )
            row = {
                "grant_item_id": grant.item_id,
                "grant_text": grant.text,
                "grant_meta": grant.meta,
                "aspect": aspect,
                "query_text": query_text,
                "candidates": candidates,
            }
            open_files[aspect].write(json.dumps(row, ensure_ascii=False) + "\n")
            written[aspect] += len(candidates)
    finally:
        for f in open_files.values():
            f.close()

    return {
        "outputs": {aspect: str(path) for aspect, path in output_paths.items()},
        "candidate_counts": written,
        "total_candidates": int(sum(written.values())),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build CE3 aspect prefilter cache with a cross-encoder STS model.")
    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    p.add_argument("--grant-db", type=str, default=GRANT_DB_DEFAULT)
    p.add_argument("--fac-db", type=str, default=FAC_DB_DEFAULT)
    p.add_argument("--decomposition-output", type=str, default=DECOMPOSITION_OUTPUT_DEFAULT)
    p.add_argument("--output-base", type=str, default=OUTPUT_BASE_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--max-grant-specs", type=int, default=MAX_GRANT_SPECS_DEFAULT)
    p.add_argument("--max-fac-specs", type=int, default=MAX_FAC_SPECS_DEFAULT)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH_DEFAULT)
    return p.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    grant_db = resolve_path(args.grant_db)
    fac_db = resolve_path(args.fac_db)
    decomposition_path = resolve_path(args.decomposition_output)
    output_base_path = resolve_path(args.output_base)
    if not decomposition_path.exists():
        raise FileNotFoundError(f"Missing decomposition output: {decomposition_path}")

    grant_specs = load_grant_specializations(grant_db, max_items=args.max_grant_specs, seed=args.seed)
    fac_specs = load_faculty_specializations(fac_db, max_items=args.max_fac_specs, seed=args.seed)
    decompositions = load_jsonl_by_key(decomposition_path, "item_id")
    _validate_coverage(grant_specs, decompositions, label="grant specs")
    _validate_coverage(fac_specs, decompositions, label="faculty specs")

    print(
        json.dumps(
            {
                "stage": "ce3_prefilter_setup",
                "model_id": args.model_id,
                "grant_db": str(grant_db),
                "fac_db": str(fac_db),
                "decomposition_output": str(decomposition_path),
                "output_base": str(output_base_path),
                "aspects": list(ASPECTS),
                "grant_specs_loaded": int(len(grant_specs)),
                "fac_specs_loaded": int(len(fac_specs)),
                "expected_scores_per_aspect": int(len(grant_specs) * len(fac_specs)),
                "expected_scores_total": int(len(ASPECTS) * len(grant_specs) * len(fac_specs)),
            },
            ensure_ascii=False,
        )
    )
    stats = build_prefilter_cache(
        grant_specs=grant_specs,
        fac_specs=fac_specs,
        decompositions=decompositions,
        output_base_path=output_base_path,
        model_id=args.model_id,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(json.dumps(stats, ensure_ascii=False))
    print(f"elapsed_sec={time.time() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
