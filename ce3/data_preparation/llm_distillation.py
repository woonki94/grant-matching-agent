from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce3").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce3.data_preparation.llm_runtime import (  # noqa: E402
    batched,
    build_prompt,
    clean_text,
    coerce_score,
    extract_json_object,
    generate_responses_batch,
    load_llm,
    normalize_ws,
    score_to_band,
    unload_llm,
)
from ce3.data_preparation.utils import (  # noqa: E402
    SpecItem,
    append_jsonl,
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


MODEL_ID_DEFAULT = "Qwen/Qwen3-14B"
GRANT_DB_DEFAULT = "ce3/dataset/source/grant_keywords_spec_keywords_db.json"
FAC_DB_DEFAULT = "ce3/dataset/source/fac_specs_db.json"
DECOMPOSITION_OUTPUT_DEFAULT = "ce3/dataset/decomposed/spec_decompositions_topic_approach_objective.jsonl"
OUTPUT_DIR_DEFAULT = "ce3/dataset/distill"
DISTILLATION_OUTPUT_DEFAULT = "ce3/dataset/distill/llm_distillation.jsonl"
SEED_DEFAULT = 42
MAX_GRANT_SPECS_DEFAULT = 0
MAX_FAC_SPECS_DEFAULT = 0
PREFILTER_HIGH_PER_ASPECT_DEFAULT = 3
PREFILTER_MID_PER_ASPECT_DEFAULT = 3
PREFILTER_LOW_PER_ASPECT_DEFAULT = 3
DISTILL_BATCH_SIZE_DEFAULT = 24
DISTILL_MAX_NEW_TOKENS_DEFAULT = 32
MAX_ATTEMPTS_DEFAULT = 2
MAX_MODEL_LEN_DEFAULT = 4096
TEMPERATURE_DEFAULT = 0.0
TOP_P_DEFAULT = 0.9

ASPECTS = ("topic", "approach", "objective")


CONDITION_BY_ASPECT = {
    "topic": "topic, problem area, technical context, field, or service area",
    "approach": "capability, method, technique, action, workflow, model, intervention, or work performed",
    "objective": "target object, beneficiary, system, material, outcome, condition, use case, or intended purpose",
}


SYSTEM_PROMPT = """
You are a strict teacher model for cross-encoder distillation.

You score similarity between one grant specialization and one faculty specialization under exactly one requested attention lens.

Rules:
- Score only the requested lens.
- Use the primary lens phrases as the main evidence.
- Use full specialization text and other decomposition lenses only for context or disambiguation.
- Do not reward topic overlap when scoring approach unless it supports approach similarity.
- Do not reward approach overlap when scoring objective unless it supports objective similarity.
- The decomposition is guidance, not ground truth. If it is incomplete or slightly misplaced, use the full text to make the best lens-specific judgment.
- Return exactly one JSON object and no markdown.

Scoring rubric:
- 0.90-1.00: near-identical or directly substitutable lens match
- 0.70-0.89: strong lens match with minor wording or scope differences
- 0.50-0.69: related but partial lens match
- 0.30-0.49: weak broad relation only
- 0.10-0.29: incidental overlap
- 0.00-0.09: no meaningful lens match

Required schema:
{"score": 0.0}
""".strip()


USER_PROMPT_TEMPLATE = """
/no_think

Requested lens:
{aspect}

Lens definition:
{condition}

Grant specialization:
{grant_text}

Faculty specialization:
{faculty_text}

Grant full decomposition:
{grant_decomposition_json}

Faculty full decomposition:
{faculty_decomposition_json}

Primary grant {aspect} phrases:
{grant_aspect_items_json}

Primary faculty {aspect} phrases:
{faculty_aspect_items_json}

Score only the requested lens similarity from 0.0 to 1.0.
""".strip()


@dataclass(frozen=True)
class CandidatePair:
    grant: SpecItem
    faculty: SpecItem
    aspect: str
    prefilter_score: float
    source_band: str


def _unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


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


def _aspect_text(item: SpecItem, decomp: Dict[str, List[str]], aspect: str, *, fallback_to_full_text: bool) -> str:
    text = " ".join(x for x in decomp.get(aspect, []) if normalize_ws(x))
    if text:
        return normalize_ws(text)
    return item.text if fallback_to_full_text else ""


TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-]*")


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in TOKEN_RE.finditer(normalize_ws(text))}


def _lexical_score(query: str, doc: str) -> float:
    q = _tokens(query)
    d = _tokens(doc)
    if not q or not d:
        return 0.0
    inter = len(q & d)
    precision = inter / max(1, len(d))
    recall = inter / max(1, len(q))
    if precision + recall <= 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


def _dedupe_candidates(candidates: Iterable[CandidatePair]) -> List[CandidatePair]:
    best: Dict[Tuple[str, str, str], CandidatePair] = {}
    for cand in candidates:
        key = (cand.grant.item_id, cand.faculty.item_id, cand.aspect)
        old = best.get(key)
        if old is None or cand.prefilter_score > old.prefilter_score:
            best[key] = cand
    return sorted(best.values(), key=lambda x: (x.grant.item_id, x.aspect, -x.prefilter_score, x.faculty.item_id))


def select_candidate_pairs(
    *,
    grant_specs: Sequence[SpecItem],
    fac_specs: Sequence[SpecItem],
    decompositions: Dict[str, Dict[str, Any]],
    high_per_aspect: int,
    mid_per_aspect: int,
    low_per_aspect: int,
) -> List[CandidatePair]:
    selected: List[CandidatePair] = []
    fac_decomps = {
        fac.item_id: _safe_decomposition(decompositions.get(fac.item_id, {}))
        for fac in fac_specs
    }

    for grant in grant_specs:
        grant_dec = _safe_decomposition(decompositions.get(grant.item_id, {}))
        for aspect in ASPECTS:
            grant_aspect_text = _aspect_text(grant, grant_dec, aspect, fallback_to_full_text=False)
            if not grant_aspect_text:
                continue

            ranked: List[Tuple[float, SpecItem]] = []
            for fac in fac_specs:
                fac_text = _aspect_text(fac, fac_decomps[fac.item_id], aspect, fallback_to_full_text=True)
                ranked.append((_lexical_score(grant_aspect_text, fac_text), fac))
            if not ranked:
                continue

            ranked.sort(key=lambda x: (-float(x[0]), x[1].item_id))
            high_k = max(0, int(high_per_aspect))
            mid_k = max(0, int(mid_per_aspect))
            low_k = max(0, int(low_per_aspect))

            for score, fac in ranked[:high_k]:
                selected.append(CandidatePair(grant, fac, aspect, float(score), "high"))

            mid_pool = sorted(ranked, key=lambda x: (abs(float(x[0]) - 0.5), -float(x[0]), x[1].item_id))
            for score, fac in mid_pool[:mid_k]:
                selected.append(CandidatePair(grant, fac, aspect, float(score), "mid"))

            low_pool = sorted(ranked, key=lambda x: (float(x[0]), x[1].item_id))
            for score, fac in low_pool[:low_k]:
                selected.append(CandidatePair(grant, fac, aspect, float(score), "low"))

    return _dedupe_candidates(selected)


def load_existing_pair_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            pair_id = clean_text(obj.get("pair_id")) if isinstance(obj, dict) else ""
            if pair_id:
                out.add(pair_id)
    return out


def _parse_score(obj: Optional[Dict[str, Any]]) -> tuple[float, bool]:
    if not isinstance(obj, dict) or "score" not in obj:
        return 0.0, False
    return coerce_score(obj.get("score")), True


def validate_decomposition_coverage(
    *,
    items: Sequence[SpecItem],
    decompositions: Dict[str, Dict[str, Any]],
    label: str,
) -> None:
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
            + "\nRun CE3 decomposition on the same source DBs before distillation."
        )


def distill_pairs(
    *,
    llm_bundle: Dict[str, Any],
    model_id: str,
    pairs: Sequence[CandidatePair],
    decompositions: Dict[str, Dict[str, Any]],
    existing_pair_ids: set[str],
    output_path: Path,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_attempts: int,
) -> List[Dict[str, Any]]:
    tokenizer = llm_bundle["tokenizer"]
    pending = [
        p for p in pairs
        if f"{p.grant.item_id}::{p.faculty.item_id}::{p.aspect}" not in existing_pair_ids
    ]
    print(f"distill_existing={len(existing_pair_ids)} distill_pending={len(pending)}")
    written_rows: List[Dict[str, Any]] = []

    for attempt in range(max(1, int(max_attempts))):
        if not pending:
            break
        next_pending: List[CandidatePair] = []
        chunks = list(batched(pending, int(batch_size)))
        chunk_iter: Iterable[Sequence[CandidatePair]] = chunks
        bar = None
        if tqdm is not None:
            bar = tqdm(
                chunks,
                total=len(chunks),
                desc=f"CE3 distill {attempt + 1}/{max(1, int(max_attempts))}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )
            chunk_iter = bar

        written_this_attempt = 0
        for chunk in chunk_iter:
            prompts: List[str] = []
            for cand in chunk:
                g_dec = _safe_decomposition(decompositions.get(cand.grant.item_id, {}))
                f_dec = _safe_decomposition(decompositions.get(cand.faculty.item_id, {}))
                prompts.append(
                    build_prompt(
                        tokenizer,
                        model_id=model_id,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=USER_PROMPT_TEMPLATE.format(
                            aspect=cand.aspect,
                            condition=CONDITION_BY_ASPECT[cand.aspect],
                            grant_text=cand.grant.text,
                            faculty_text=cand.faculty.text,
                            grant_decomposition_json=json.dumps(g_dec, ensure_ascii=False),
                            faculty_decomposition_json=json.dumps(f_dec, ensure_ascii=False),
                            grant_aspect_items_json=json.dumps(g_dec.get(cand.aspect, []), ensure_ascii=False),
                            faculty_aspect_items_json=json.dumps(f_dec.get(cand.aspect, []), ensure_ascii=False),
                        ),
                    )
                )

            responses = generate_responses_batch(
                llm_bundle=llm_bundle,
                prompts=prompts,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            )

            rows: List[Dict[str, Any]] = []
            for cand, response in zip(chunk, responses):
                score, ok = _parse_score(extract_json_object(response))
                if not ok:
                    next_pending.append(cand)
                    continue
                g_dec = _safe_decomposition(decompositions.get(cand.grant.item_id, {}))
                f_dec = _safe_decomposition(decompositions.get(cand.faculty.item_id, {}))
                pair_id = f"{cand.grant.item_id}::{cand.faculty.item_id}::{cand.aspect}"
                row = {
                    "pair_id": pair_id,
                    "aspect": cand.aspect,
                    "score": float(score),
                    "band": score_to_band(float(score)),
                    "grant": {
                        "item_id": cand.grant.item_id,
                        "text": cand.grant.text,
                        "meta": cand.grant.meta,
                        "decomposition": g_dec,
                    },
                    "faculty": {
                        "item_id": cand.faculty.item_id,
                        "text": cand.faculty.text,
                        "meta": cand.faculty.meta,
                        "decomposition": f_dec,
                    },
                    "lexical_prefilter_score": float(cand.prefilter_score),
                    "pair_source": f"lexical_{cand.aspect}_{cand.source_band}",
                    "parse_ok": True,
                    "attempt": int(attempt + 1),
                    "model_id": model_id,
                    "raw_response": response,
                }
                rows.append(row)
                written_rows.append(row)
                existing_pair_ids.add(pair_id)

            written_this_attempt += append_jsonl(output_path, rows)
            if bar is not None:
                bar.set_postfix(written=int(written_this_attempt), retry_pending=int(len(next_pending)))

        if bar is not None:
            bar.close()
        pending = next_pending
        if pending:
            print(f"distill_retry_pending={len(pending)} attempt={attempt + 1}")

    return written_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CE3 LLM distillation over decomposed specialization pairs.")
    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    p.add_argument("--grant-db", type=str, default=GRANT_DB_DEFAULT)
    p.add_argument("--fac-db", type=str, default=FAC_DB_DEFAULT)
    p.add_argument("--decomposition-output", type=str, default=DECOMPOSITION_OUTPUT_DEFAULT)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--distillation-output", type=str, default=DISTILLATION_OUTPUT_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--max-grant-specs", type=int, default=MAX_GRANT_SPECS_DEFAULT)
    p.add_argument("--max-fac-specs", type=int, default=MAX_FAC_SPECS_DEFAULT)
    p.add_argument("--prefilter-high-per-aspect", type=int, default=PREFILTER_HIGH_PER_ASPECT_DEFAULT)
    p.add_argument("--prefilter-mid-per-aspect", type=int, default=PREFILTER_MID_PER_ASPECT_DEFAULT)
    p.add_argument("--prefilter-low-per-aspect", type=int, default=PREFILTER_LOW_PER_ASPECT_DEFAULT)
    p.add_argument("--distill-batch-size", type=int, default=DISTILL_BATCH_SIZE_DEFAULT)
    p.add_argument("--distill-max-new-tokens", type=int, default=DISTILL_MAX_NEW_TOKENS_DEFAULT)
    p.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    p.add_argument("--top-p", type=float, default=TOP_P_DEFAULT)
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN_DEFAULT)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--overwrite", action="store_true", help="Delete existing distillation output before running.")
    return p.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    grant_db = resolve_path(args.grant_db)
    fac_db = resolve_path(args.fac_db)
    decomposition_path = resolve_path(args.decomposition_output)
    output_dir = resolve_path(args.output_dir)
    output_path = resolve_path(args.distillation_output)
    if not decomposition_path.exists():
        raise FileNotFoundError(f"Missing decomposition output: {decomposition_path}")
    if args.overwrite:
        _unlink_if_exists(output_path)

    grant_specs = load_grant_specializations(grant_db, max_items=args.max_grant_specs, seed=args.seed)
    fac_specs = load_faculty_specializations(fac_db, max_items=args.max_fac_specs, seed=args.seed)
    decompositions = load_jsonl_by_key(decomposition_path, "item_id")
    validate_decomposition_coverage(items=grant_specs, decompositions=decompositions, label="grant specs")
    validate_decomposition_coverage(items=fac_specs, decompositions=decompositions, label="faculty specs")

    pairs = select_candidate_pairs(
        grant_specs=grant_specs,
        fac_specs=fac_specs,
        decompositions=decompositions,
        high_per_aspect=args.prefilter_high_per_aspect,
        mid_per_aspect=args.prefilter_mid_per_aspect,
        low_per_aspect=args.prefilter_low_per_aspect,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "stage": "ce3_distill_setup",
                "model_id": args.model_id,
                "grant_db": str(grant_db),
                "fac_db": str(fac_db),
                "decomposition_output": str(decomposition_path),
                "distillation_output": str(output_path),
                "aspects": list(ASPECTS),
                "grant_specs_loaded": int(len(grant_specs)),
                "fac_specs_loaded": int(len(fac_specs)),
                "candidate_pairs": int(len(pairs)),
                "prefilter_high_per_aspect": int(args.prefilter_high_per_aspect),
                "prefilter_mid_per_aspect": int(args.prefilter_mid_per_aspect),
                "prefilter_low_per_aspect": int(args.prefilter_low_per_aspect),
            },
            ensure_ascii=False,
        )
    )

    bundle = None
    try:
        bundle = load_llm(
            args.model_id,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        rows = distill_pairs(
            llm_bundle=bundle,
            model_id=args.model_id,
            pairs=pairs,
            decompositions=decompositions,
            existing_pair_ids=load_existing_pair_ids(output_path),
            output_path=output_path,
            batch_size=args.distill_batch_size,
            max_new_tokens=args.distill_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_attempts=args.max_attempts,
        )
    finally:
        unload_llm(bundle)

    aspect_counts = Counter(row.get("aspect") for row in rows)
    band_counts = Counter(row.get("band") for row in rows)
    print(f"elapsed_sec={time.time() - started:.2f}")
    print(f"distillation_jsonl={output_path}")
    print(f"written_rows={len(rows)}")
    print(f"aspect_counts={json.dumps(dict(aspect_counts), ensure_ascii=False)}")
    print(f"band_counts={json.dumps(dict(band_counts), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
