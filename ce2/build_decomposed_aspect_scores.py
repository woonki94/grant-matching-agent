from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce2").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce2.llm_runtime import (  # noqa: E402
    batched,
    build_prompt,
    clean_text,
    coerce_score,
    extract_json_object,
    generate_responses_batch,
    load_llm,
    model_slug,
    normalize_ws,
    score_to_band,
    unload_llm,
)
from ce2.prompt.aspect_scoring_prompts import (  # noqa: E402
    SCORE_SYSTEM_PROMPTS_BY_ASPECT,
    SCORE_USER_PROMPT_TEMPLATE,
)
from ce2.prompt.decomposition_prompt import (  # noqa: E402
    DECOMPOSE_SYSTEM_PROMPT,
    DECOMPOSE_USER_PROMPT_TEMPLATE,
)


MODEL_ID_DEFAULT = "Qwen/Qwen3-14B"
GRANT_DB_DEFAULT = "ce/dataset/source/grant_keywords_spec_keywords_db.json"
FAC_DB_DEFAULT = "ce/dataset/source/fac_specs_db.json"
OUTPUT_DIR_DEFAULT = "ce2/dataset/distill"
DECOMPOSITION_OUTPUT_DEFAULT = "ce2/dataset/distill/spec_decompositions_5aspect.jsonl"
SCORES_OUTPUT_DEFAULT = "ce2/dataset/distill/decomposed_5aspect_pair_scores.jsonl"
SUMMARY_OUTPUT_DEFAULT = "ce2/dataset/distill/decomposed_5aspect_pair_scores_summary.json"

SEED_DEFAULT = 42
MAX_GRANT_SPECS_DEFAULT = 80
MAX_FAC_SPECS_DEFAULT = 2500
CANDIDATES_PER_GRANT_SPEC_DEFAULT = 16
RANDOM_CANDIDATES_PER_GRANT_SPEC_DEFAULT = 4
DECOMPOSE_BATCH_SIZE_DEFAULT = 64
SCORE_BATCH_SIZE_DEFAULT = 64
DECOMPOSE_MAX_NEW_TOKENS_DEFAULT = 220
SCORE_MAX_NEW_TOKENS_DEFAULT = 300
MAX_MODEL_LEN_DEFAULT = 4096
TEMPERATURE_DEFAULT = 0.0
TOP_P_DEFAULT = 0.9
MAX_ATTEMPTS_DEFAULT = 2

ASPECTS = ("domain", "method", "target", "deliverable", "application_context")


@dataclass(frozen=True)
class SpecItem:
    item_id: str
    kind: str
    text: str
    meta: Dict[str, Any]


def _resolve_path(value: str) -> Path:
    p = Path(clean_text(value)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _load_jsonl_by_key(path: Path, key: str) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = clean_text(raw)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                k = clean_text(obj.get(key))
                if k:
                    out[k] = obj
    return out


def _load_scored_pair_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = clean_text(raw)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pair_id = clean_text(obj.get("pair_id"))
            if pair_id:
                keys.add(pair_id)
    return keys


def _load_grant_specs(path: Path, *, max_items: int, seed: int) -> List[SpecItem]:
    db = _read_json(path)
    grants = db.get("grants") if isinstance(db, dict) else []
    items: List[SpecItem] = []
    for grant in grants or []:
        grant_id = clean_text(grant.get("grant_id"))
        specs = grant.get("grant_spec_keywords") if isinstance(grant, dict) else []
        for idx, text in enumerate(specs or []):
            norm = normalize_ws(text)
            if not norm:
                continue
            items.append(
                SpecItem(
                    item_id=f"grant:{grant_id}:{idx}",
                    kind="grant",
                    text=norm,
                    meta={
                        "grant_id": grant_id,
                        "grant_spec_idx": int(idx),
                    },
                )
            )
    rng = random.Random(int(seed))
    rng.shuffle(items)
    if int(max_items) > 0:
        items = items[: int(max_items)]
    return items


def _load_fac_specs(path: Path, *, max_items: int, seed: int) -> List[SpecItem]:
    db = _read_json(path)
    rows = db.get("fac_specs") if isinstance(db, dict) else []
    items: List[SpecItem] = []
    for row in rows or []:
        text = normalize_ws(row.get("text"))
        if not text:
            continue
        fac_id = row.get("fac_id")
        fac_spec_id = row.get("fac_spec_id")
        fac_spec_idx = row.get("fac_spec_idx")
        items.append(
            SpecItem(
                item_id=f"fac:{fac_id}:{fac_spec_id}:{fac_spec_idx}",
                kind="faculty",
                text=text,
                meta={
                    "fac_id": fac_id,
                    "fac_spec_id": fac_spec_id,
                    "fac_spec_idx": fac_spec_idx,
                    "section": row.get("section"),
                },
            )
        )
    rng = random.Random(int(seed) + 13)
    rng.shuffle(items)
    if int(max_items) > 0:
        items = items[: int(max_items)]
    return items


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "including",
    "into",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "using",
    "with",
}


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", clean_text(text).lower()) if t not in _STOPWORDS]


def _build_idf(items: Sequence[SpecItem]) -> Dict[str, float]:
    df: Counter[str] = Counter()
    for item in items:
        df.update(set(_tokens(item.text)))
    n = max(1, len(items))
    return {tok: math.log((1.0 + n) / (1.0 + cnt)) + 1.0 for tok, cnt in df.items()}


def _weighted_overlap_score(query_tokens: Sequence[str], doc_tokens: Sequence[str], idf: Dict[str, float]) -> float:
    q = set(query_tokens)
    d = set(doc_tokens)
    if not q or not d:
        return 0.0
    inter = q & d
    if not inter:
        return 0.0
    inter_w = sum(idf.get(t, 1.0) for t in inter)
    denom = math.sqrt(sum(idf.get(t, 1.0) ** 2 for t in q)) * math.sqrt(sum(idf.get(t, 1.0) ** 2 for t in d))
    return float(inter_w / max(denom, 1e-9))


def _select_pairs(
    grant_specs: Sequence[SpecItem],
    fac_specs: Sequence[SpecItem],
    *,
    candidates_per_grant_spec: int,
    random_candidates_per_grant_spec: int,
    seed: int,
) -> List[Tuple[SpecItem, SpecItem, float, str]]:
    rng = random.Random(int(seed) + 29)
    idf = _build_idf(list(grant_specs) + list(fac_specs))
    fac_token_cache = {f.item_id: _tokens(f.text) for f in fac_specs}
    pairs: List[Tuple[SpecItem, SpecItem, float, str]] = []
    for grant in grant_specs:
        gtoks = _tokens(grant.text)
        scored: List[Tuple[float, SpecItem]] = [
            (_weighted_overlap_score(gtoks, fac_token_cache[fac.item_id], idf), fac) for fac in fac_specs
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        selected: Dict[str, Tuple[SpecItem, float, str]] = {}
        for lexical_score, fac in scored[: max(0, int(candidates_per_grant_spec))]:
            selected[fac.item_id] = (fac, float(lexical_score), "lexical_top")
        random_pool = list(fac_specs)
        rng.shuffle(random_pool)
        for fac in random_pool[: max(0, int(random_candidates_per_grant_spec))]:
            if fac.item_id not in selected:
                lexical_score = _weighted_overlap_score(gtoks, fac_token_cache[fac.item_id], idf)
                selected[fac.item_id] = (fac, float(lexical_score), "random")
        for fac, lexical_score, source in selected.values():
            pairs.append((grant, fac, lexical_score, source))
    return pairs


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [normalize_ws(x) for x in value if normalize_ws(x)]
    if isinstance(value, str) and normalize_ws(value):
        return [normalize_ws(value)]
    return []


def _clean_decomposition(obj: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    if not isinstance(obj, dict):
        return {aspect: [] for aspect in ASPECTS}
    return {
        "domain": _as_list(obj.get("domain")),
        "method": _as_list(obj.get("method")),
        "target": _as_list(obj.get("target")),
        "deliverable": _as_list(obj.get("deliverable")),
        "application_context": _as_list(obj.get("application_context", obj.get("context"))),
    }


def _decompose_specs(
    *,
    llm_bundle: Dict[str, Any],
    model_id: str,
    items: Sequence[SpecItem],
    existing: Dict[str, Dict[str, Any]],
    output_path: Path,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_attempts: int,
) -> Dict[str, Dict[str, Any]]:
    tokenizer = llm_bundle["tokenizer"]
    pending = [item for item in items if item.item_id not in existing]
    print(f"decompose_existing={len(existing)} decompose_pending={len(pending)}")
    for attempt in range(max(1, int(max_attempts))):
        if not pending:
            break
        next_pending: List[SpecItem] = []
        for chunk in batched(pending, int(batch_size)):
            prompts = [
                build_prompt(
                    tokenizer,
                    model_id=model_id,
                    system_prompt=DECOMPOSE_SYSTEM_PROMPT,
                    user_prompt=DECOMPOSE_USER_PROMPT_TEMPLATE.format(text=item.text),
                )
                for item in chunk
            ]
            responses = generate_responses_batch(
                llm_bundle=llm_bundle,
                prompts=prompts,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            )
            rows: List[Dict[str, Any]] = []
            for item, response in zip(chunk, responses):
                parsed = extract_json_object(response)
                decomp = _clean_decomposition(parsed)
                parse_ok = isinstance(parsed, dict) and any(decomp[a] for a in ASPECTS)
                row = {
                    "item_id": item.item_id,
                    "kind": item.kind,
                    "text": item.text,
                    "meta": item.meta,
                    "decomposition": decomp,
                    "parse_ok": bool(parse_ok),
                    "attempt": int(attempt + 1),
                    "model_id": model_id,
                    "raw_response": response,
                }
                if parse_ok:
                    existing[item.item_id] = row
                    rows.append(row)
                else:
                    next_pending.append(item)
            _append_jsonl(output_path, rows)
        pending = next_pending
        if pending:
            print(f"decompose_retry_pending={len(pending)} attempt={attempt + 1}")

    if pending:
        rows = []
        for item in pending:
            row = {
                "item_id": item.item_id,
                "kind": item.kind,
                "text": item.text,
                "meta": item.meta,
                "decomposition": {aspect: [] for aspect in ASPECTS},
                "parse_ok": False,
                "attempt": int(max(1, int(max_attempts))),
                "model_id": model_id,
                "raw_response": "",
            }
            existing[item.item_id] = row
            rows.append(row)
        _append_jsonl(output_path, rows)
        print(f"decompose_failed_written={len(rows)}")
    return existing


def _parse_single_score(obj: Optional[Dict[str, Any]]) -> Tuple[float, str, bool]:
    if not isinstance(obj, dict):
        return 0.0, "", False
    if "score" not in obj:
        return 0.0, "", False
    return coerce_score(obj.get("score")), normalize_ws(obj.get("reason")), True


def _score_pairs(
    *,
    llm_bundle: Dict[str, Any],
    model_id: str,
    pairs: Sequence[Tuple[SpecItem, SpecItem, float, str]],
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
    pending = [p for p in pairs if f"{p[0].item_id}::{p[1].item_id}" not in existing_pair_ids]
    print(f"score_existing={len(existing_pair_ids)} score_pending={len(pending)}")
    written_rows: List[Dict[str, Any]] = []
    for attempt in range(max(1, int(max_attempts))):
        if not pending:
            break
        next_pending: List[Tuple[SpecItem, SpecItem, float, str]] = []
        for chunk in batched(pending, int(batch_size)):
            prompts: List[str] = []
            task_items: List[Tuple[str, str, SpecItem, SpecItem, float, str, Dict[str, Any], Dict[str, Any]]] = []
            # Group by aspect so consecutive prompts share the same system prompt.
            # This improves vLLM prefix/cache locality without increasing batch size.
            for aspect in ASPECTS:
                for grant, fac, lexical_score, pair_source in chunk:
                    g_dec_row = decompositions.get(grant.item_id, {})
                    f_dec_row = decompositions.get(fac.item_id, {})
                    g_dec = g_dec_row.get("decomposition") if isinstance(g_dec_row, dict) else {}
                    f_dec = f_dec_row.get("decomposition") if isinstance(f_dec_row, dict) else {}
                    pair_id = f"{grant.item_id}::{fac.item_id}"
                    user_prompt = SCORE_USER_PROMPT_TEMPLATE.format(
                        aspect=aspect,
                        grant_text=grant.text,
                        grant_aspect_items_json=json.dumps(g_dec.get(aspect, []), ensure_ascii=False),
                        grant_decomposition_json=json.dumps(g_dec, ensure_ascii=False),
                        fac_text=fac.text,
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
                    task_items.append((pair_id, aspect, grant, fac, lexical_score, pair_source, g_dec_row, f_dec_row))
            responses = generate_responses_batch(
                llm_bundle=llm_bundle,
                prompts=prompts,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            )
            rows: List[Dict[str, Any]] = []
            grouped: Dict[str, Dict[str, Any]] = {}
            for item, response in zip(task_items, responses):
                pair_id, aspect, grant, fac, lexical_score, pair_source, g_dec_row, f_dec_row = item
                parsed = extract_json_object(response)
                score, reason, ok = _parse_single_score(parsed)
                bucket = grouped.setdefault(
                    pair_id,
                    {
                        "grant": grant,
                        "fac": fac,
                        "lexical_score": lexical_score,
                        "pair_source": pair_source,
                        "g_dec_row": g_dec_row,
                        "f_dec_row": f_dec_row,
                        "scores": {},
                        "reasons": {},
                        "raw_responses": {},
                        "ok": {},
                    },
                )
                bucket["scores"][aspect] = float(score)
                bucket["reasons"][aspect] = reason
                bucket["raw_responses"][aspect] = response
                bucket["ok"][aspect] = bool(ok)

            for pair_id, bucket in grouped.items():
                grant = bucket["grant"]
                fac = bucket["fac"]
                lexical_score = bucket["lexical_score"]
                pair_source = bucket["pair_source"]
                g_dec_row = bucket["g_dec_row"]
                f_dec_row = bucket["f_dec_row"]
                score_map = bucket["scores"]
                reason_map = bucket["reasons"]
                ok_map = bucket["ok"]
                parse_ok = all(bool(ok_map.get(aspect)) and aspect in score_map for aspect in ASPECTS)
                if not parse_ok:
                    next_pending.append((grant, fac, lexical_score, pair_source))
                    continue
                aspect_scores = {aspect: float(score_map[aspect]) for aspect in ASPECTS}
                overall_score = float(sum(aspect_scores.values()) / max(1, len(ASPECTS)))
                row = {
                    "pair_id": pair_id,
                    "grant": {
                        "item_id": grant.item_id,
                        "text": grant.text,
                        "meta": grant.meta,
                        "decomposition": g_dec_row.get("decomposition", {}),
                    },
                    "faculty": {
                        "item_id": fac.item_id,
                        "text": fac.text,
                        "meta": fac.meta,
                        "decomposition": f_dec_row.get("decomposition", {}),
                    },
                    "scores": {
                        **aspect_scores,
                        "overall": overall_score,
                    },
                    "bands": {
                        **{aspect: score_to_band(score) for aspect, score in aspect_scores.items()},
                        "overall": score_to_band(overall_score),
                    },
                    "reasons": {aspect: reason_map.get(aspect, "") for aspect in ASPECTS},
                    "lexical_prefilter_score": float(lexical_score),
                    "pair_source": pair_source,
                    "parse_ok": True,
                    "attempt": int(attempt + 1),
                    "model_id": model_id,
                    "raw_responses": bucket["raw_responses"],
                }
                existing_pair_ids.add(pair_id)
                rows.append(row)
                written_rows.append(row)
            _append_jsonl(output_path, rows)
        pending = next_pending
        if pending:
            print(f"score_retry_pending={len(pending)} attempt={attempt + 1}")
    if pending:
        print(f"score_failed={len(pending)}")
    return written_rows


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = clean_text(raw)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    idx = min(len(vals) - 1, max(0, int(round(float(q) * (len(vals) - 1)))))
    return float(vals[idx])


def _score_stats(values: Sequence[float]) -> Dict[str, Any]:
    vals = [float(v) for v in values]
    if not vals:
        return {"count": 0}
    avg = sum(vals) / len(vals)
    var = sum((x - avg) ** 2 for x in vals) / max(1, len(vals))
    return {
        "count": len(vals),
        "mean": avg,
        "std": math.sqrt(var),
        "min": min(vals),
        "p10": _percentile(vals, 0.10),
        "p25": _percentile(vals, 0.25),
        "p50": _percentile(vals, 0.50),
        "p75": _percentile(vals, 0.75),
        "p90": _percentile(vals, 0.90),
        "max": max(vals),
        "bands": dict(Counter(score_to_band(x) for x in vals)),
    }


def _build_summary(*, scores_path: Path, decomposition_path: Path, started_at: float, config: Dict[str, Any]) -> Dict[str, Any]:
    rows = list(_iter_jsonl(scores_path))
    decomps = list(_iter_jsonl(decomposition_path))
    score_values = {
        aspect: [float(row.get("scores", {}).get(aspect, 0.0)) for row in rows if isinstance(row.get("scores"), dict)]
        for aspect in (*ASPECTS, "overall")
    }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - started_at,
        "config": config,
        "decompositions": {
            "count": len(decomps),
            "parse_ok": sum(1 for row in decomps if bool(row.get("parse_ok"))),
            "parse_failed": sum(1 for row in decomps if not bool(row.get("parse_ok"))),
            "by_kind": dict(Counter(clean_text(row.get("kind")) for row in decomps)),
        },
        "pairs": {
            "count": len(rows),
            "by_source": dict(Counter(clean_text(row.get("pair_source")) for row in rows)),
        },
        "score_stats": {aspect: _score_stats(values) for aspect, values in score_values.items()},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CE2 pilot: decompose specs and score five aspect pair matches.")
    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    p.add_argument("--grant-db", type=str, default=GRANT_DB_DEFAULT)
    p.add_argument("--fac-db", type=str, default=FAC_DB_DEFAULT)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--decomposition-output", type=str, default=DECOMPOSITION_OUTPUT_DEFAULT)
    p.add_argument("--scores-output", type=str, default=SCORES_OUTPUT_DEFAULT)
    p.add_argument("--summary-output", type=str, default=SUMMARY_OUTPUT_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--max-grant-specs", type=int, default=MAX_GRANT_SPECS_DEFAULT)
    p.add_argument("--max-fac-specs", type=int, default=MAX_FAC_SPECS_DEFAULT)
    p.add_argument("--candidates-per-grant-spec", type=int, default=CANDIDATES_PER_GRANT_SPEC_DEFAULT)
    p.add_argument("--random-candidates-per-grant-spec", type=int, default=RANDOM_CANDIDATES_PER_GRANT_SPEC_DEFAULT)
    p.add_argument("--decompose-batch-size", type=int, default=DECOMPOSE_BATCH_SIZE_DEFAULT)
    p.add_argument("--score-batch-size", type=int, default=SCORE_BATCH_SIZE_DEFAULT)
    p.add_argument("--decompose-max-new-tokens", type=int, default=DECOMPOSE_MAX_NEW_TOKENS_DEFAULT)
    p.add_argument("--score-max-new-tokens", type=int, default=SCORE_MAX_NEW_TOKENS_DEFAULT)
    p.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    p.add_argument("--top-p", type=float, default=TOP_P_DEFAULT)
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN_DEFAULT)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--overwrite", action="store_true", help="Remove existing ce2 decomposition/score outputs before running.")
    p.add_argument("--decompose-only", action="store_true")
    p.add_argument("--score-only", action="store_true")
    return p.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()

    output_dir = _resolve_path(args.output_dir)
    decomposition_path = _resolve_path(args.decomposition_output)
    scores_path = _resolve_path(args.scores_output)
    summary_path = _resolve_path(args.summary_output)
    if args.overwrite:
        for path in (decomposition_path, scores_path, summary_path):
            if path.exists():
                path.unlink()

    grant_specs = _load_grant_specs(_resolve_path(args.grant_db), max_items=args.max_grant_specs, seed=args.seed)
    fac_specs = _load_fac_specs(_resolve_path(args.fac_db), max_items=args.max_fac_specs, seed=args.seed)
    pairs = _select_pairs(
        grant_specs,
        fac_specs,
        candidates_per_grant_spec=args.candidates_per_grant_spec,
        random_candidates_per_grant_spec=args.random_candidates_per_grant_spec,
        seed=args.seed,
    )
    config = {
        "model_id": args.model_id,
        "grant_db": str(_resolve_path(args.grant_db)),
        "fac_db": str(_resolve_path(args.fac_db)),
        "max_grant_specs": int(args.max_grant_specs),
        "max_fac_specs": int(args.max_fac_specs),
        "grant_specs_loaded": len(grant_specs),
        "fac_specs_loaded": len(fac_specs),
        "candidate_pairs": len(pairs),
        "candidates_per_grant_spec": int(args.candidates_per_grant_spec),
        "random_candidates_per_grant_spec": int(args.random_candidates_per_grant_spec),
        "seed": int(args.seed),
    }
    print(json.dumps({"stage": "setup", **config}, ensure_ascii=False))
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle: Optional[Dict[str, Any]] = None
    try:
        bundle = load_llm(
            args.model_id,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        decompositions = _load_jsonl_by_key(decomposition_path, "item_id")
        if not args.score_only:
            decompositions = _decompose_specs(
                llm_bundle=bundle,
                model_id=args.model_id,
                items=list(grant_specs) + list(fac_specs),
                existing=decompositions,
                output_path=decomposition_path,
                batch_size=args.decompose_batch_size,
                max_new_tokens=args.decompose_max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                max_attempts=args.max_attempts,
            )
        if not args.decompose_only:
            existing_pair_ids = _load_scored_pair_keys(scores_path)
            _score_pairs(
                llm_bundle=bundle,
                model_id=args.model_id,
                pairs=pairs,
                decompositions=decompositions,
                existing_pair_ids=existing_pair_ids,
                output_path=scores_path,
                batch_size=args.score_batch_size,
                max_new_tokens=args.score_max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                max_attempts=args.max_attempts,
            )
    finally:
        unload_llm(bundle)

    summary = _build_summary(scores_path=scores_path, decomposition_path=decomposition_path, started_at=started, config=config)
    _write_json(summary_path, summary)
    print(json.dumps({"stage": "summary", "summary_path": str(summary_path), **summary["pairs"]}, ensure_ascii=False))
    for aspect, stats in summary.get("score_stats", {}).items():
        if stats.get("count"):
            print(
                f"{aspect:12s} count={stats['count']} mean={stats['mean']:.4f} "
                f"p25={stats['p25']:.4f} p50={stats['p50']:.4f} p75={stats['p75']:.4f} "
                f"bands={stats['bands']}"
            )
    print(f"decomposition_jsonl={decomposition_path}")
    print(f"scores_jsonl={scores_path}")
    print(f"summary_json={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
