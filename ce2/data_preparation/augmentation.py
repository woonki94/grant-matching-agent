from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


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
    DISTILL_BATCH_SIZE_DEFAULT,
    DISTILL_MAX_NEW_TOKENS_DEFAULT,
    DISTILLATION_OUTPUT_DEFAULT,
    MAX_MODEL_LEN_DEFAULT,
    MODEL_ID_DEFAULT,
    SEED_DEFAULT,
    SpecItem,
    append_jsonl,
    load_jsonl_by_key,
    resolve_path,
    write_json,
    _normalize_aspect_items,
)
from ce2.data_preparation.llm_runtime import (  # noqa: E402
    build_prompt,
    coerce_score,
    extract_json_object,
    generate_responses_batch,
    load_llm,
    normalize_ws,
    score_to_band,
    unload_llm,
)
from ce2.data_preparation.prompt.augmentation import (  # noqa: E402
    AUGMENT_SYSTEM_PROMPTS_BY_ASPECT,
    AUGMENT_USER_PROMPT_TEMPLATE,
)
from ce2.data_preparation.prompt.distillation import SCORE_SYSTEM_PROMPTS_BY_ASPECT, SCORE_USER_PROMPT_TEMPLATE  # noqa: E402


AUGMENT_OUTPUT_DEFAULT = "ce2/dataset/distill/augmentation.jsonl"
AUGMENT_SUMMARY_DEFAULT = "ce2/dataset/distill/augmentation_summary.json"
AUGMENT_TARGET_POLICY_DEFAULT = "median"
AUGMENT_MAX_ADD_PER_BAND_DEFAULT = 300
AUGMENT_GEN_BATCH_SIZE_DEFAULT = 24
AUGMENT_GEN_MAX_NEW_TOKENS_DEFAULT = 512
AUGMENT_MAX_TRIES_PER_MISSING_DEFAULT = 8
TEMPERATURE_DEFAULT = 0.0
TOP_P_DEFAULT = 0.9

BANDS = ("high", "mid", "low")


@dataclass(frozen=True)
class _Candidate:
    grant: SpecItem
    faculty: SpecItem
    grant_dec_row: Dict[str, Any]
    fac_dec_row: Dict[str, Any]
    target_aspect: str
    target_band: str
    generation_note: str
    generation_raw: str


@dataclass(frozen=True)
class _GenJob:
    grant: SpecItem
    grant_dec_row: Dict[str, Any]
    aspect: str
    band: str


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


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = normalize_ws(raw)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _build_grant_specs(decompositions: Dict[str, Dict[str, Any]]) -> List[Tuple[SpecItem, Dict[str, Any]]]:
    out: List[Tuple[SpecItem, Dict[str, Any]]] = []
    for item_id, row in decompositions.items():
        if not item_id.startswith("grant:"):
            continue
        text = normalize_ws(row.get("text"))
        if not text:
            continue
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        item = SpecItem(item_id=item_id, kind="grant", text=text, meta=meta)
        out.append((item, row))
    out.sort(key=lambda x: x[0].item_id)
    return out


def _count_bands(distilled_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {
        aspect: {band: 0 for band in BANDS}
        for aspect in ASPECTS
    }
    for row in distilled_rows:
        if not bool(row.get("parse_ok")):
            continue
        row_aspect = normalize_ws(row.get("aspect")).lower()
        row_band = normalize_ws(row.get("band")).lower()
        if row_aspect in ASPECTS and row_band in BANDS:
            counts[row_aspect][row_band] += 1
            continue
        bands = row.get("bands") if isinstance(row.get("bands"), dict) else {}
        for aspect in ASPECTS:
            b = normalize_ws(bands.get(aspect)).lower()
            if b in BANDS:
                counts[aspect][b] += 1
    return counts


def _compute_targets(
    *,
    counts: Dict[str, Dict[str, int]],
    target_policy: str,
    target_high: int,
    target_mid: int,
    target_low: int,
) -> Dict[str, Dict[str, int]]:
    targets: Dict[str, Dict[str, int]] = {}
    mode = normalize_ws(target_policy).lower()
    for aspect in ASPECTS:
        c = counts.get(aspect, {})
        if mode == "explicit":
            targets[aspect] = {
                "high": int(max(0, target_high)),
                "mid": int(max(0, target_mid)),
                "low": int(max(0, target_low)),
            }
            continue
        vals = [int(c.get("high", 0)), int(c.get("mid", 0)), int(c.get("low", 0))]
        med = int(statistics.median(vals)) if vals else 0
        targets[aspect] = {"high": med, "mid": med, "low": med}
    return targets


def _normalize_generated_decomposition(obj: Dict[str, Any]) -> Dict[str, List[str]]:
    raw = obj.get("decomposition") if isinstance(obj.get("decomposition"), dict) else {}
    dec: Dict[str, List[str]] = {}
    for aspect in ASPECTS:
        vals = raw.get(aspect)
        if not isinstance(vals, list):
            vals = obj.get(aspect) if isinstance(obj.get(aspect), list) else []
        dec[aspect] = _normalize_aspect_items(aspect, vals if isinstance(vals, list) else [])
    return dec


def _parse_generated_response(response: str) -> Tuple[str, Dict[str, List[str]], str, bool]:
    parsed = extract_json_object(response)
    if not isinstance(parsed, dict):
        return "", {aspect: [] for aspect in ASPECTS}, "", False

    text = normalize_ws(parsed.get("faculty_text") or parsed.get("augmented_text") or parsed.get("text"))
    dec = _normalize_generated_decomposition(parsed)
    note = normalize_ws(parsed.get("note"))

    if not text:
        return "", dec, note, False
    if not any(dec.get(a) for a in ASPECTS):
        return text, dec, note, False
    return text, dec, note, True


def _resolve_effective_model_id(distilled_rows: Sequence[Dict[str, Any]], requested: str) -> Tuple[str, str]:
    req = normalize_ws(requested)
    ids = [normalize_ws(r.get("model_id")) for r in distilled_rows if isinstance(r, dict) and normalize_ws(r.get("model_id"))]
    from_distillation = Counter(ids).most_common(1)[0][0] if ids else ""
    if req.lower() in {"", "auto", "same", "same-as-distill"}:
        if from_distillation:
            return from_distillation, "distillation_input"
        return MODEL_ID_DEFAULT, "default"
    return req, "arg"


def _build_generation_jobs(
    *,
    rng: random.Random,
    grant_specs: Sequence[Tuple[SpecItem, Dict[str, Any]]],
    aspect: str,
    band: str,
    count: int,
) -> List[_GenJob]:
    jobs: List[_GenJob] = []
    if not grant_specs or int(count) <= 0:
        return jobs
    for _ in range(int(count)):
        grant_item, grant_row = grant_specs[rng.randrange(0, len(grant_specs))]
        jobs.append(_GenJob(grant=grant_item, grant_dec_row=grant_row, aspect=aspect, band=band))
    return jobs


def _generate_candidates_llm(
    *,
    llm_bundle: Dict[str, Any],
    model_id: str,
    jobs: Sequence[_GenJob],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    synthetic_counter_start: int,
) -> Tuple[List[_Candidate], int]:
    if not jobs:
        return [], synthetic_counter_start

    tokenizer = llm_bundle["tokenizer"]
    prompts: List[str] = []
    for job in jobs:
        grant_dec = job.grant_dec_row.get("decomposition") if isinstance(job.grant_dec_row.get("decomposition"), dict) else {}
        user_prompt = AUGMENT_USER_PROMPT_TEMPLATE.format(
            aspect=job.aspect,
            target_band=job.band,
            grant_text=job.grant.text,
            grant_decomposition_json=json.dumps(grant_dec, ensure_ascii=False),
            grant_aspect_items_json=json.dumps(grant_dec.get(job.aspect, []), ensure_ascii=False),
        )
        prompts.append(
            build_prompt(
                tokenizer,
                model_id=model_id,
                system_prompt=AUGMENT_SYSTEM_PROMPTS_BY_ASPECT[job.aspect],
                user_prompt=user_prompt,
            )
        )

    responses = generate_responses_batch(
        llm_bundle=llm_bundle,
        prompts=prompts,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )

    out: List[_Candidate] = []
    counter = int(synthetic_counter_start)
    for job, response in zip(jobs, responses):
        text, dec, note, ok = _parse_generated_response(response)
        if not ok:
            continue
        if not dec.get(job.aspect):
            continue

        syn_id = f"augfac:{job.aspect}:{job.band}:{counter}"
        counter += 1
        fac_item = SpecItem(
            item_id=syn_id,
            kind="faculty",
            text=text,
            meta={
                "fac_id": None,
                "fac_spec_id": syn_id,
                "fac_spec_idx": counter,
                "section": f"augmented_{job.aspect}_{job.band}",
                "is_augmented": True,
            },
        )
        fac_dec_row = {
            "item_id": syn_id,
            "kind": "faculty",
            "text": text,
            "meta": fac_item.meta,
            "decomposition": dec,
            "parse_ok": True,
        }
        out.append(
            _Candidate(
                grant=job.grant,
                faculty=fac_item,
                grant_dec_row=job.grant_dec_row,
                fac_dec_row=fac_dec_row,
                target_aspect=job.aspect,
                target_band=job.band,
                generation_note=note,
                generation_raw=response,
            )
        )
    return out, counter


def _distill_candidate_pairs(
    *,
    llm_bundle: Dict[str, Any],
    model_id: str,
    candidates: Sequence[_Candidate],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    tokenizer = llm_bundle["tokenizer"]
    prompts: List[str] = []
    task_meta: List[Tuple[int, str]] = []
    for idx, cand in enumerate(candidates):
        g_dec = cand.grant_dec_row.get("decomposition", {})
        f_dec = cand.fac_dec_row.get("decomposition", {})
        aspect = cand.target_aspect
        user_prompt = SCORE_USER_PROMPT_TEMPLATE.format(
            aspect=aspect,
            grant_text=cand.grant.text,
            grant_aspect_items_json=json.dumps(g_dec.get(aspect, []), ensure_ascii=False),
            grant_decomposition_json=json.dumps(g_dec, ensure_ascii=False),
            fac_text=cand.faculty.text,
            fac_aspect_items_json=json.dumps(f_dec.get(aspect, []), ensure_ascii=False),
            fac_decomposition_json=json.dumps(f_dec, ensure_ascii=False),
        )
        prompts.append(
            build_prompt(
                tokenizer,
                model_id=model_id,
                system_prompt=SCORE_SYSTEM_PROMPTS_BY_ASPECT[aspect],
                user_prompt=user_prompt,
            )
        )
        task_meta.append((idx, aspect))

    responses = generate_responses_batch(
        llm_bundle=llm_bundle,
        prompts=prompts,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )

    grouped: Dict[int, Dict[str, Any]] = {}
    for (idx, aspect), response in zip(task_meta, responses):
        parsed = extract_json_object(response)
        score = 0.0
        ok = False
        if isinstance(parsed, dict) and "score" in parsed:
            score = coerce_score(parsed.get("score"))
            ok = True
        bucket = grouped.setdefault(idx, {"scores": {}, "raw": {}, "ok": {}})
        bucket["scores"][aspect] = float(score)
        bucket["raw"][aspect] = response
        bucket["ok"][aspect] = bool(ok)

    out_rows: List[Dict[str, Any]] = []
    for idx, cand in enumerate(candidates):
        bucket = grouped.get(idx, {})
        score_map = bucket.get("scores", {})
        ok_map = bucket.get("ok", {})
        aspect = cand.target_aspect
        parse_ok = bool(ok_map.get(aspect)) and aspect in score_map
        if not parse_ok:
            continue
        score = float(score_map[aspect])
        band = score_to_band(score)
        row = {
            "pair_id": f"{cand.grant.item_id}::{cand.faculty.item_id}::{aspect}",
            "aspect": aspect,
            "score": score,
            "band": band,
            "grant": {
                "item_id": cand.grant.item_id,
                "text": cand.grant.text,
                "meta": cand.grant.meta,
                "decomposition": cand.grant_dec_row.get("decomposition", {}),
            },
            "faculty": {
                "item_id": cand.faculty.item_id,
                "text": cand.faculty.text,
                "meta": cand.faculty.meta,
                "decomposition": cand.fac_dec_row.get("decomposition", {}),
            },
            "scores": {aspect: score},
            "bands": {aspect: band},
            "lexical_prefilter_score": 0.0,
            "pair_source": f"augmented_{cand.target_aspect}_{cand.target_band}",
            "parse_ok": True,
            "attempt": 1,
            "model_id": model_id,
            "raw_response": (bucket.get("raw", {}) or {}).get(aspect, ""),
            "raw_responses": {aspect: (bucket.get("raw", {}) or {}).get(aspect, "")},
            "is_augmented": True,
            "augment_target_aspect": cand.target_aspect,
            "augment_target_band": cand.target_band,
            "augment_generation_note": cand.generation_note,
            "augment_generation_raw": cand.generation_raw,
        }
        out_rows.append(row)
    return out_rows


def _unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment CE2 distillation rows to fill missing aspect-band clusters.")
    p.add_argument("--model-id", type=str, default="auto", help='Use "auto" to reuse model_id from distillation input.')
    p.add_argument("--distillation-input", type=str, default=DISTILLATION_OUTPUT_DEFAULT)
    p.add_argument("--decomposition-output", type=str, default=DECOMPOSITION_OUTPUT_DEFAULT)
    p.add_argument("--output", type=str, default=AUGMENT_OUTPUT_DEFAULT)
    p.add_argument("--summary-output", type=str, default=AUGMENT_SUMMARY_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--target-policy", type=str, choices=["median", "explicit"], default=AUGMENT_TARGET_POLICY_DEFAULT)
    p.add_argument("--target-high", type=int, default=0)
    p.add_argument("--target-mid", type=int, default=0)
    p.add_argument("--target-low", type=int, default=0)
    p.add_argument("--max-add-per-band", type=int, default=AUGMENT_MAX_ADD_PER_BAND_DEFAULT)
    p.add_argument("--gen-batch-size", type=int, default=AUGMENT_GEN_BATCH_SIZE_DEFAULT)
    p.add_argument("--gen-max-new-tokens", type=int, default=AUGMENT_GEN_MAX_NEW_TOKENS_DEFAULT)
    p.add_argument("--max-tries-per-missing", type=int, default=AUGMENT_MAX_TRIES_PER_MISSING_DEFAULT)
    p.add_argument("--distill-batch-size", type=int, default=DISTILL_BATCH_SIZE_DEFAULT)
    p.add_argument("--distill-max-new-tokens", type=int, default=DISTILL_MAX_NEW_TOKENS_DEFAULT)
    p.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    p.add_argument("--top-p", type=float, default=TOP_P_DEFAULT)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN_DEFAULT)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    rng = random.Random(int(args.seed))

    distillation_input = resolve_path(PROJECT_ROOT, args.distillation_input)
    decomposition_path = resolve_path(PROJECT_ROOT, args.decomposition_output)
    output_path = resolve_path(PROJECT_ROOT, args.output)
    summary_path = resolve_path(PROJECT_ROOT, args.summary_output)

    if not distillation_input.exists():
        raise FileNotFoundError(f"Distillation input not found: {distillation_input}")
    if not decomposition_path.exists():
        raise FileNotFoundError(f"Decomposition file not found: {decomposition_path}")

    if args.overwrite:
        _unlink_if_exists(output_path)
        _unlink_if_exists(summary_path)

    distilled_rows = [row for row in _iter_jsonl(distillation_input)]
    decompositions = load_jsonl_by_key(decomposition_path, "item_id")
    grant_specs = _build_grant_specs(decompositions)
    if not grant_specs:
        raise RuntimeError("No grant specs found in decomposition file.")

    counts_before = _count_bands(distilled_rows)
    targets = _compute_targets(
        counts=counts_before,
        target_policy=args.target_policy,
        target_high=int(args.target_high),
        target_mid=int(args.target_mid),
        target_low=int(args.target_low),
    )
    max_add = _safe_int(args.max_add_per_band, default=AUGMENT_MAX_ADD_PER_BAND_DEFAULT, minimum=0, maximum=1_000_000)
    deficits: Dict[str, Dict[str, int]] = {a: {b: 0 for b in BANDS} for a in ASPECTS}
    for aspect in ASPECTS:
        for band in BANDS:
            miss = int(targets[aspect][band]) - int(counts_before[aspect][band])
            deficits[aspect][band] = min(max(0, miss), max_add)

    existing_pair_ids = {
        normalize_ws(row.get("pair_id"))
        for row in distilled_rows
        if normalize_ws(row.get("pair_id"))
    }

    total_needed = sum(int(deficits[a][b]) for a in ASPECTS for b in BANDS)
    print(
        json.dumps(
            {
                "stage": "augment_setup",
                "counts_before": counts_before,
                "targets": targets,
                "deficits": deficits,
                "total_needed": total_needed,
            },
            ensure_ascii=False,
        )
    )

    if total_needed <= 0:
        summary = {
            "created_at": time.time(),
            "elapsed_sec": time.time() - started,
            "distillation_input": str(distillation_input),
            "decomposition_output": str(decomposition_path),
            "output": str(output_path),
            "counts_before": counts_before,
            "targets": targets,
            "deficits": deficits,
            "augmented_rows_written": 0,
            "counts_after": counts_before,
        }
        write_json(summary_path, summary)
        print("augment_needed=0")
        print(f"summary_json={summary_path}")
        return 0

    effective_model_id, model_source = _resolve_effective_model_id(distilled_rows, requested=args.model_id)
    print(f"augment_model_resolved requested={args.model_id} effective={effective_model_id} source={model_source}")

    bundle = None
    generated_rows: List[Dict[str, Any]] = []
    synthetic_counter = 0
    tries_used: Dict[str, Dict[str, int]] = {a: {b: 0 for b in BANDS} for a in ASPECTS}
    accepted_counts: Dict[str, Dict[str, int]] = {a: {b: 0 for b in BANDS} for a in ASPECTS}
    seen_grant_text_keys: set[str] = set()

    try:
        bundle = load_llm(
            effective_model_id,
            max_model_len=int(args.max_model_len),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            tensor_parallel_size=int(args.tensor_parallel_size),
        )

        for aspect in ASPECTS:
            for band in BANDS:
                need = int(deficits[aspect][band])
                if need <= 0:
                    continue
                max_tries = int(max(1, int(args.max_tries_per_missing))) * need
                while need > 0 and tries_used[aspect][band] < max_tries:
                    gen_batch = min(
                        int(max(1, int(args.gen_batch_size))),
                        need * 2,
                        max_tries - tries_used[aspect][band],
                    )
                    jobs = _build_generation_jobs(
                        rng=rng,
                        grant_specs=grant_specs,
                        aspect=aspect,
                        band=band,
                        count=gen_batch,
                    )
                    tries_used[aspect][band] += len(jobs)

                    candidates, synthetic_counter = _generate_candidates_llm(
                        llm_bundle=bundle,
                        model_id=effective_model_id,
                        jobs=jobs,
                        max_new_tokens=int(args.gen_max_new_tokens),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        synthetic_counter_start=synthetic_counter,
                    )
                    if not candidates:
                        print(
                            f"augment_gen_empty aspect={aspect} band={band} "
                            f"remaining={need} tries={tries_used[aspect][band]}/{max_tries}"
                        )
                        continue

                    rows: List[Dict[str, Any]] = []
                    distill_step = max(1, int(args.distill_batch_size))
                    for i in range(0, len(candidates), distill_step):
                        rows.extend(
                            _distill_candidate_pairs(
                                llm_bundle=bundle,
                                model_id=effective_model_id,
                                candidates=candidates[i : i + distill_step],
                                max_new_tokens=int(args.distill_max_new_tokens),
                                temperature=float(args.temperature),
                                top_p=float(args.top_p),
                            )
                        )

                    accepted: List[Dict[str, Any]] = []
                    for row in rows:
                        pair_id = normalize_ws(row.get("pair_id"))
                        if not pair_id or pair_id in existing_pair_ids:
                            continue
                        row_band = normalize_ws(row.get("band")).lower()
                        if not row_band:
                            row_bands = row.get("bands") if isinstance(row.get("bands"), dict) else {}
                            row_band = normalize_ws(row_bands.get(aspect)).lower()
                        if row_band != band:
                            continue

                        grant_item_id = normalize_ws((row.get("grant") or {}).get("item_id"))
                        faculty_text = normalize_ws((row.get("faculty") or {}).get("text"))
                        dedupe_key = f"{grant_item_id}::{faculty_text.casefold()}"
                        if dedupe_key in seen_grant_text_keys:
                            continue

                        seen_grant_text_keys.add(dedupe_key)
                        existing_pair_ids.add(pair_id)
                        accepted.append(row)
                        if len(accepted) >= need:
                            break

                    if accepted:
                        append_jsonl(output_path, accepted)
                        generated_rows.extend(accepted)
                        got = len(accepted)
                        need -= got
                        accepted_counts[aspect][band] += got
                        print(
                            f"augment_accept aspect={aspect} band={band} "
                            f"accepted={got} remaining={need} tries={tries_used[aspect][band]}/{max_tries}"
                        )
                    else:
                        print(
                            f"augment_retry aspect={aspect} band={band} "
                            f"remaining={need} tries={tries_used[aspect][band]}/{max_tries}"
                        )
    finally:
        unload_llm(bundle)

    counts_after = {
        aspect: {
            band: int(counts_before[aspect][band] + accepted_counts[aspect][band])
            for band in BANDS
        }
        for aspect in ASPECTS
    }
    unfilled = {
        aspect: {
            band: int(max(0, int(targets[aspect][band]) - int(counts_after[aspect][band])))
            for band in BANDS
        }
        for aspect in ASPECTS
    }

    summary = {
        "created_at": time.time(),
        "elapsed_sec": time.time() - started,
        "distillation_input": str(distillation_input),
        "decomposition_output": str(decomposition_path),
        "output": str(output_path),
        "model_id_requested": args.model_id,
        "model_id_effective": effective_model_id,
        "model_id_source": model_source,
        "target_policy": args.target_policy,
        "counts_before": counts_before,
        "targets": targets,
        "deficits": deficits,
        "accepted_counts": accepted_counts,
        "tries_used": tries_used,
        "counts_after": counts_after,
        "unfilled": unfilled,
        "augmented_rows_written": len(generated_rows),
    }
    write_json(summary_path, summary)
    print(f"augmented_rows_written={len(generated_rows)}")
    print(f"augmented_jsonl={output_path}")
    print(f"summary_json={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
