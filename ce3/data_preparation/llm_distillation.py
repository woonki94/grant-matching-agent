from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


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
    load_items_from_decomposition_rows,
    resolve_path,
)

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # type: ignore[assignment]


MODEL_ID_DEFAULT = "Qwen/Qwen3-14B"
DECOMPOSITION_OUTPUT_DEFAULT = "ce3/dataset/decomposed/spec_decompositions_combined.jsonl"
OUTPUT_DIR_DEFAULT = "ce3/dataset/distill"
DISTILLATION_OUTPUT_DEFAULT = "ce3/dataset/distill/llm_distillation.jsonl"
PREFILTER_CACHE_BASE_DEFAULT = "ce3/dataset/source/prefilter_cache.jsonl"
AUGMENTED_DECOMPOSITION_OUTPUT_DEFAULT = "ce3/dataset/decomposed/spec_decompositions_augmented.jsonl"
TARGET_HIGH_PER_GRANT_ASPECT_DEFAULT = 4
TARGET_MID_PER_GRANT_ASPECT_DEFAULT = 8
TARGET_LOW_PER_GRANT_ASPECT_DEFAULT = 4
PREFILTER_HIGH_MULTIPLIER_DEFAULT = 4.0
PREFILTER_MID_MULTIPLIER_DEFAULT = 2.0
PREFILTER_LOW_MULTIPLIER_DEFAULT = 1.25
AUGMENTED_CANDIDATES_PER_SOURCE_ASPECT_DEFAULT = 3
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
    prefilter_rank: int
    target_cluster: str
    query_text: str
    doc_text: str
    pair_source: str = "ce_prefilter"


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


def prefilter_cache_paths(output_base_path: Path) -> Dict[str, Path]:
    stem = output_base_path.stem if output_base_path.suffix else output_base_path.name
    return {
        aspect: (output_base_path.parent / f"{stem}_{aspect}.jsonl").resolve()
        for aspect in ASPECTS
    }


def _positive_int(value: int) -> int:
    return max(0, int(value))


def _overfetch_count(target: int, multiplier: float) -> int:
    if int(target) <= 0:
        return 0
    import math

    return max(int(target), int(math.ceil(float(target) * max(1.0, float(multiplier)))))


def _dedupe_candidates(candidates: Iterable[CandidatePair]) -> List[CandidatePair]:
    best: Dict[tuple[str, str, str], CandidatePair] = {}
    for cand in candidates:
        key = (cand.grant.item_id, cand.faculty.item_id, cand.aspect)
        old = best.get(key)
        if old is None or _candidate_priority(cand) < _candidate_priority(old):
            best[key] = cand
    return sorted(
        best.values(),
        key=lambda x: (x.grant.item_id, x.aspect, _cluster_sort_key(x.target_cluster), x.prefilter_rank, x.faculty.item_id),
    )


def _cluster_sort_key(cluster: str) -> int:
    return {"high": 0, "mid": 1, "low": 2}.get(cluster, 9)


def _candidate_priority(cand: CandidatePair) -> tuple[int, int, float, str]:
    score_sort = -cand.prefilter_score if cand.target_cluster != "low" else cand.prefilter_score
    source_sort = 0 if cand.pair_source == "augmented_anchor" else 1
    return (
        _cluster_sort_key(cand.target_cluster),
        source_sort,
        int(cand.prefilter_rank),
        float(score_sort),
        cand.faculty.item_id,
    )


def _select_from_ranked_candidates(
    *,
    candidates: Sequence[Dict[str, Any]],
    target_high: int,
    target_mid: int,
    target_low: int,
    high_multiplier: float,
    mid_multiplier: float,
    low_multiplier: float,
) -> List[tuple[str, Dict[str, Any]]]:
    valid = [
        cand
        for cand in candidates
        if normalize_ws(cand.get("doc_text")) and clean_text(cand.get("fac_item_id"))
    ]
    selected: List[tuple[str, Dict[str, Any]]] = []
    selected_fac_ids: set[str] = set()

    def add(cluster: str, pool: Sequence[Dict[str, Any]], limit: int) -> None:
        for cand in pool:
            if len([x for x in selected if x[0] == cluster]) >= limit:
                return
            fac_id = clean_text(cand.get("fac_item_id"))
            if not fac_id or fac_id in selected_fac_ids:
                continue
            selected.append((cluster, cand))
            selected_fac_ids.add(fac_id)

    high_n = _overfetch_count(target_high, high_multiplier)
    mid_n = _overfetch_count(target_mid, mid_multiplier)
    low_n = _overfetch_count(target_low, low_multiplier)

    high_pool = sorted(valid, key=lambda x: (-float(x.get("ce_score", 0.0)), int(x.get("rank", 0)), clean_text(x.get("fac_item_id"))))
    mid_pool = sorted(valid, key=lambda x: (abs(float(x.get("ce_score", 0.0)) - 0.5), -float(x.get("ce_score", 0.0)), int(x.get("rank", 0)), clean_text(x.get("fac_item_id"))))
    low_pool = sorted(valid, key=lambda x: (float(x.get("ce_score", 0.0)), -int(x.get("rank", 0)), clean_text(x.get("fac_item_id"))))

    add("high", high_pool, high_n)
    add("mid", mid_pool, mid_n)
    add("low", low_pool, low_n)
    return selected


def select_candidate_pairs(
    *,
    grant_specs: Sequence[SpecItem],
    fac_by_id: Dict[str, SpecItem],
    prefilter_cache_base: Path,
    target_high: int,
    target_mid: int,
    target_low: int,
    high_multiplier: float,
    mid_multiplier: float,
    low_multiplier: float,
) -> List[CandidatePair]:
    grant_by_id = {item.item_id: item for item in grant_specs}
    paths = prefilter_cache_paths(prefilter_cache_base)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prefilter cache files:\n"
            + "\n".join(missing)
            + "\nRun CE3 prefilter cache generation before distillation."
        )

    selected: List[CandidatePair] = []
    for aspect, path in paths.items():
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                try:
                    row = json.loads(raw)
                except Exception:
                    continue
                grant_id = clean_text(row.get("grant_item_id"))
                grant = grant_by_id.get(grant_id)
                if grant is None:
                    continue
                query_text = normalize_ws(row.get("query_text"))
                if not query_text:
                    continue
                row_aspect = clean_text(row.get("aspect")) or aspect
                if row_aspect != aspect:
                    continue
                raw_candidates = row.get("candidates")
                if not isinstance(raw_candidates, list):
                    continue
                clustered = _select_from_ranked_candidates(
                    candidates=raw_candidates,
                    target_high=_positive_int(target_high),
                    target_mid=_positive_int(target_mid),
                    target_low=_positive_int(target_low),
                    high_multiplier=high_multiplier,
                    mid_multiplier=mid_multiplier,
                    low_multiplier=low_multiplier,
                )
                for cluster, raw_cand in clustered:
                    fac_id = clean_text(raw_cand.get("fac_item_id"))
                    fac = fac_by_id.get(fac_id)
                    doc_text = normalize_ws(raw_cand.get("doc_text"))
                    if fac is None or not doc_text:
                        continue
                    selected.append(
                        CandidatePair(
                            grant=grant,
                            faculty=fac,
                            aspect=aspect,
                            prefilter_score=float(raw_cand.get("ce_score", 0.0)),
                            prefilter_rank=int(raw_cand.get("rank", 0)),
                            target_cluster=cluster,
                            query_text=query_text,
                            doc_text=doc_text,
                            pair_source="ce_prefilter",
                        )
                    )

    return _dedupe_candidates(selected)


def _source_slot(item: SpecItem) -> int:
    try:
        return int(item.meta.get("slot", 0))
    except Exception:
        return 0


def _aspect_items_text(item_id: str, decompositions: Dict[str, Dict[str, Any]], aspect: str) -> str:
    return normalize_ws(" ".join(_safe_decomposition(decompositions.get(item_id, {})).get(aspect, [])))


def select_augmented_candidate_pairs(
    *,
    original_grants_by_id: Dict[str, SpecItem],
    original_faculty_by_id: Dict[str, SpecItem],
    augmented_grants: Sequence[SpecItem],
    augmented_faculty: Sequence[SpecItem],
    decompositions: Dict[str, Dict[str, Any]],
    max_per_source_aspect: int,
) -> List[CandidatePair]:
    limit = max(0, int(max_per_source_aspect))
    if limit <= 0:
        return []
    selected: List[CandidatePair] = []
    counts: Counter[tuple[str, str, str]] = Counter()

    def has_room(side: str, source_item_id: str, aspect: str) -> bool:
        key = (side, source_item_id, aspect)
        return counts[key] < limit

    def mark_added(side: str, source_item_id: str, aspect: str) -> None:
        key = (side, source_item_id, aspect)
        counts[key] += 1

    for fac_aug in sorted(augmented_faculty, key=lambda x: (clean_text(x.meta.get("source_item_id")), clean_text(x.meta.get("target_aspect")), _source_slot(x), x.item_id)):
        source_item_id = clean_text(fac_aug.meta.get("source_item_id"))
        aspect = clean_text(fac_aug.meta.get("target_aspect"))
        grant = original_grants_by_id.get(source_item_id)
        if grant is None or aspect not in ASPECTS or not has_room("fac_aug", source_item_id, aspect):
            continue
        query_text = _aspect_items_text(grant.item_id, decompositions, aspect)
        doc_text = _aspect_items_text(fac_aug.item_id, decompositions, aspect)
        if not query_text or not doc_text:
            continue
        mark_added("fac_aug", source_item_id, aspect)
        selected.append(
            CandidatePair(
                grant=grant,
                faculty=fac_aug,
                aspect=aspect,
                prefilter_score=1.0,
                prefilter_rank=1 + _source_slot(fac_aug),
                target_cluster="high",
                query_text=query_text,
                doc_text=doc_text,
                pair_source="augmented_anchor",
            )
        )

    for grant_aug in sorted(augmented_grants, key=lambda x: (clean_text(x.meta.get("source_item_id")), clean_text(x.meta.get("target_aspect")), _source_slot(x), x.item_id)):
        source_item_id = clean_text(grant_aug.meta.get("source_item_id"))
        aspect = clean_text(grant_aug.meta.get("target_aspect"))
        faculty = original_faculty_by_id.get(source_item_id)
        if faculty is None or aspect not in ASPECTS or not has_room("grant_aug", source_item_id, aspect):
            continue
        query_text = _aspect_items_text(grant_aug.item_id, decompositions, aspect)
        doc_text = _aspect_items_text(faculty.item_id, decompositions, aspect)
        if not query_text or not doc_text:
            continue
        mark_added("grant_aug", source_item_id, aspect)
        selected.append(
            CandidatePair(
                grant=grant_aug,
                faculty=faculty,
                aspect=aspect,
                prefilter_score=1.0,
                prefilter_rank=1 + _source_slot(grant_aug),
                target_cluster="high",
                query_text=query_text,
                doc_text=doc_text,
                pair_source="augmented_anchor",
            )
        )
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
                        "kind": cand.grant.kind,
                        "text": cand.grant.text,
                        "meta": cand.grant.meta,
                        "decomposition": g_dec,
                    },
                    "faculty": {
                        "item_id": cand.faculty.item_id,
                        "kind": cand.faculty.kind,
                        "text": cand.faculty.text,
                        "meta": cand.faculty.meta,
                        "decomposition": f_dec,
                    },
                    "ce_prefilter_score": float(cand.prefilter_score),
                    "ce_prefilter_rank": int(cand.prefilter_rank),
                    "target_cluster": cand.target_cluster,
                    "prefilter_query_text": cand.query_text,
                    "prefilter_doc_text": cand.doc_text,
                    "pair_source": f"{cand.pair_source}_{cand.aspect}_{cand.target_cluster}",
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
    p.add_argument("--decomposition-output", type=str, default=DECOMPOSITION_OUTPUT_DEFAULT)
    p.add_argument(
        "--augmented-decomposition-output",
        type=str,
        default="",
        help="Optional augmented decomposition JSONL. When set, bounded high-intent augmented anchor pairs are added directly to distillation.",
    )
    p.add_argument("--prefilter-cache-base", type=str, default=PREFILTER_CACHE_BASE_DEFAULT)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--distillation-output", type=str, default=DISTILLATION_OUTPUT_DEFAULT)
    p.add_argument("--target-high-per-grant-aspect", type=int, default=TARGET_HIGH_PER_GRANT_ASPECT_DEFAULT)
    p.add_argument("--target-mid-per-grant-aspect", type=int, default=TARGET_MID_PER_GRANT_ASPECT_DEFAULT)
    p.add_argument("--target-low-per-grant-aspect", type=int, default=TARGET_LOW_PER_GRANT_ASPECT_DEFAULT)
    p.add_argument("--prefilter-high-multiplier", type=float, default=PREFILTER_HIGH_MULTIPLIER_DEFAULT)
    p.add_argument("--prefilter-mid-multiplier", type=float, default=PREFILTER_MID_MULTIPLIER_DEFAULT)
    p.add_argument("--prefilter-low-multiplier", type=float, default=PREFILTER_LOW_MULTIPLIER_DEFAULT)
    p.add_argument("--augmented-candidates-per-source-aspect", type=int, default=AUGMENTED_CANDIDATES_PER_SOURCE_ASPECT_DEFAULT)
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
    decomposition_path = resolve_path(args.decomposition_output)
    augmented_decomposition_path = (
        resolve_path(args.augmented_decomposition_output)
        if clean_text(args.augmented_decomposition_output)
        else None
    )
    prefilter_cache_base = resolve_path(args.prefilter_cache_base)
    output_dir = resolve_path(args.output_dir)
    output_path = resolve_path(args.distillation_output)
    if not decomposition_path.exists():
        raise FileNotFoundError(f"Missing decomposition output: {decomposition_path}")
    if args.overwrite:
        _unlink_if_exists(output_path)

    grant_specs, fac_specs, decompositions = load_items_from_decomposition_rows(
        decomposition_path,
        grant_kinds=("grant",),
        faculty_kinds=("faculty",),
    )
    if not grant_specs or not fac_specs:
        raise RuntimeError(
            f"Original decomposition must contain at least one grant-side and one faculty-side item: {decomposition_path}"
        )

    augmented_pairs: List[CandidatePair] = []
    if augmented_decomposition_path is not None:
        if not augmented_decomposition_path.exists():
            raise FileNotFoundError(f"Missing augmented decomposition output: {augmented_decomposition_path}")
        aug_grants, aug_faculty, aug_decompositions = load_items_from_decomposition_rows(
            augmented_decomposition_path,
            grant_kinds=("grant_aug",),
            faculty_kinds=("fac_aug",),
        )
        merged_decompositions = dict(decompositions)
        merged_decompositions.update(aug_decompositions)
        decompositions = merged_decompositions
        augmented_pairs = select_augmented_candidate_pairs(
            original_grants_by_id={item.item_id: item for item in grant_specs},
            original_faculty_by_id={item.item_id: item for item in fac_specs},
            augmented_grants=aug_grants,
            augmented_faculty=aug_faculty,
            decompositions=decompositions,
            max_per_source_aspect=int(args.augmented_candidates_per_source_aspect),
        )

    fac_by_id = {item.item_id: item for item in fac_specs}
    prefilter_pairs = select_candidate_pairs(
        grant_specs=grant_specs,
        fac_by_id=fac_by_id,
        prefilter_cache_base=prefilter_cache_base,
        target_high=args.target_high_per_grant_aspect,
        target_mid=args.target_mid_per_grant_aspect,
        target_low=args.target_low_per_grant_aspect,
        high_multiplier=args.prefilter_high_multiplier,
        mid_multiplier=args.prefilter_mid_multiplier,
        low_multiplier=args.prefilter_low_multiplier,
    )
    pairs = _dedupe_candidates([*prefilter_pairs, *augmented_pairs])
    candidate_aspect_counts = Counter(pair.aspect for pair in pairs)
    candidate_cluster_counts = Counter(pair.target_cluster for pair in pairs)
    candidate_source_counts = Counter(pair.pair_source for pair in pairs)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "stage": "ce3_distill_setup",
                "model_id": args.model_id,
                "decomposition_output": str(decomposition_path),
                "augmented_decomposition_output": str(augmented_decomposition_path or ""),
                "prefilter_cache_base": str(prefilter_cache_base),
                "distillation_output": str(output_path),
                "aspects": list(ASPECTS),
                "grant_specs_loaded": int(len(grant_specs)),
                "fac_specs_loaded": int(len(fac_specs)),
                "prefilter_candidate_pairs": int(len(prefilter_pairs)),
                "augmented_candidate_pairs": int(len(augmented_pairs)),
                "candidate_pairs": int(len(pairs)),
                "candidate_aspect_counts": dict(candidate_aspect_counts),
                "candidate_target_cluster_counts": dict(candidate_cluster_counts),
                "candidate_source_counts": dict(candidate_source_counts),
                "target_high_per_grant_aspect": int(args.target_high_per_grant_aspect),
                "target_mid_per_grant_aspect": int(args.target_mid_per_grant_aspect),
                "target_low_per_grant_aspect": int(args.target_low_per_grant_aspect),
                "prefilter_high_multiplier": float(args.prefilter_high_multiplier),
                "prefilter_mid_multiplier": float(args.prefilter_mid_multiplier),
                "prefilter_low_multiplier": float(args.prefilter_low_multiplier),
                "augmented_candidates_per_source_aspect": int(args.augmented_candidates_per_source_aspect),
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
