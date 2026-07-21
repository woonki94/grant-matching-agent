from __future__ import annotations

import argparse
import json
import re
import sys
import time
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
    extract_json_object,
    generate_responses_batch,
    load_llm,
    normalize_ws,
    unload_llm,
)

from ce3.data_preparation.utils import (  # noqa: E402
    SpecItem,
    append_jsonl,
    drop_stale_cached_rows,
    load_faculty_specializations,
    load_grant_specializations,
    load_jsonl_by_key,
    refresh_decomposition_cache,
    resolve_path,
)

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # type: ignore[assignment]


MODEL_ID_DEFAULT = "Qwen/Qwen3-14B"
GRANT_DB_DEFAULT = "ce3/dataset/source/grant_keywords_spec_keywords_db.json"
FAC_DB_DEFAULT = "ce3/dataset/source/fac_specs_db.json"
OUTPUT_DIR_DEFAULT = "ce3/dataset/decomposed"
DECOMPOSITION_OUTPUT_DEFAULT = "ce3/dataset/decomposed/spec_decompositions_topic_approach_objective.jsonl"
SEED_DEFAULT = 42
MAX_GRANT_SPECS_DEFAULT = 0
MAX_FAC_SPECS_DEFAULT = 0
DECOMPOSE_BATCH_SIZE_DEFAULT = 16
DECOMPOSE_MAX_NEW_TOKENS_DEFAULT = 256
MAX_ATTEMPTS_DEFAULT = 2
MAX_MODEL_LEN_DEFAULT = 4096
TEMPERATURE_DEFAULT = 0.0
TOP_P_DEFAULT = 0.9

ASPECTS = ("topic", "approach", "objective")


SYSTEM_PROMPT = """
You decompose one grant or faculty specialization phrase into three attention lenses for cross-encoder training.

The three lenses are:
- topic: the main subject, problem area, technical context, field, or service area.
- approach: the capability, method, technique, action, workflow, model, intervention, or work performed.
- objective: the object, system, population, material, outcome, condition, use case, or intended purpose.

Rules:
- Use only information present, strongly implied, or safely inferable from the phrase.
- Make a grounded best effort to fill every lens, but prefer [] over vague filler.
- Each lens should add distinct retrieval value.
- Return [] when a lens would only repeat another lens or require a vague/generic filler phrase.
- Do not copy the same phrase into multiple lenses unless it truly plays multiple roles.
- Avoid generic standalone terms such as "program", "project", "strategy", "initiative", "intervention", "services", or "management" unless paired with a specific modifier.
- The objective lens should capture the target object, beneficiary, system, material, outcome, condition, use case, or intended purpose, not merely restate the approach.
- If the objective is not explicit but the target object or beneficiary is clear, use that target object or beneficiary as the objective.
- When a phrase contains verbs such as reduce, increase, improve, maintain, maximize, strengthen, prevent, support, or enhance, put the desired state/result/beneficiary in objective unless the phrase clearly describes how the work is performed.
- Prefer role-specific paraphrases over repeated source spans.
- Keep items short keyword phrases, not sentences.
- Lowercase unless proper nouns or technical capitalization is needed.
- Return exactly one JSON object and no markdown.

Required schema:
{
  "topic": ["..."],
  "approach": ["..."],
  "objective": ["..."]
}
""".strip()


USER_PROMPT_TEMPLATE = """
/no_think

Specialization phrase:
{text}

Return the coordinated topic/approach/objective decomposition.
""".strip()


def _dedupe_phrases(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        phrase = normalize_ws(value)
        if not phrase:
            continue
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    return out


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [normalize_ws(x) for x in value if normalize_ws(x)]
    if isinstance(value, str) and normalize_ws(value):
        return [normalize_ws(value)]
    return []


def _normalize_items(aspect: str, values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in values:
        phrase = normalize_ws(raw)
        if not phrase:
            continue
        phrase = re.sub(r"^[\-\*\d\.\)\(]+\s*", "", phrase)
        if phrase:
            out.append(phrase)
    return _dedupe_phrases(out)


def _parse_decomposition(obj: Optional[Dict[str, Any]]) -> tuple[Dict[str, List[str]], bool]:
    out = {aspect: [] for aspect in ASPECTS}
    if not isinstance(obj, dict):
        return out, False
    parsed_any = False
    for aspect in ASPECTS:
        values = _as_list(obj.get(aspect))
        if values or aspect in obj:
            parsed_any = True
        out[aspect] = _normalize_items(aspect, values)
    return out, parsed_any


def decompose_specializations(
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
    existing, stale = drop_stale_cached_rows(existing=existing, items=items)
    pending = [item for item in items if item.item_id not in existing]
    print(f"decompose_existing={len(existing)} decompose_stale_cached={stale} decompose_pending={len(pending)}")

    for attempt in range(max(1, int(max_attempts))):
        if not pending:
            break
        next_pending: List[SpecItem] = []
        chunks = list(batched(pending, int(batch_size)))
        chunk_iter: Iterable[Sequence[SpecItem]] = chunks
        bar = None
        if tqdm is not None:
            bar = tqdm(
                chunks,
                total=len(chunks),
                desc=f"CE3 decompose {attempt + 1}/{max(1, int(max_attempts))}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )
            chunk_iter = bar

        written_this_attempt = 0
        for chunk in chunk_iter:
            prompts = [
                build_prompt(
                    tokenizer,
                    model_id=model_id,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=USER_PROMPT_TEMPLATE.format(text=item.text),
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
                decomp, parsed = _parse_decomposition(extract_json_object(response))
                nonempty = [aspect for aspect in ASPECTS if decomp.get(aspect)]
                parse_ok = bool(parsed) and bool(nonempty)
                row = {
                    "item_id": item.item_id,
                    "kind": item.kind,
                    "text": item.text,
                    "meta": item.meta,
                    "decomposition": decomp,
                    "parse_ok": bool(parse_ok),
                    "decomposition_parse": {
                        "parsed_aspects_count": int(len(ASPECTS) if parsed else 0),
                        "nonempty_aspects_count": int(len(nonempty)),
                        "parsed_aspects": list(ASPECTS) if parsed else [],
                        "nonempty_aspects": nonempty,
                    },
                    "attempt": int(attempt + 1),
                    "model_id": model_id,
                    "raw_response": response,
                }
                if not parse_ok:
                    next_pending.append(item)
                    continue
                existing[item.item_id] = row
                rows.append(row)

            written_this_attempt += append_jsonl(output_path, rows)
            if bar is not None:
                bar.set_postfix(written=int(written_this_attempt), retry_pending=int(len(next_pending)))

        if bar is not None:
            bar.close()
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
                "decomposition_parse": {
                    "parsed_aspects_count": 0,
                    "nonempty_aspects_count": 0,
                    "parsed_aspects": [],
                    "nonempty_aspects": [],
                },
                "attempt": int(max(1, int(max_attempts))),
                "model_id": model_id,
                "raw_response": "",
            }
            existing[item.item_id] = row
            rows.append(row)
        append_jsonl(output_path, rows)
        print(f"decompose_failed_written={len(rows)}")

    return existing


def _unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CE3 specialization decomposition runner (topic/approach/objective).")
    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    p.add_argument("--grant-db", type=str, default=GRANT_DB_DEFAULT)
    p.add_argument("--fac-db", type=str, default=FAC_DB_DEFAULT)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--decomposition-output", type=str, default=DECOMPOSITION_OUTPUT_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--max-grant-specs", type=int, default=MAX_GRANT_SPECS_DEFAULT)
    p.add_argument("--max-fac-specs", type=int, default=MAX_FAC_SPECS_DEFAULT)
    p.add_argument("--decompose-batch-size", type=int, default=DECOMPOSE_BATCH_SIZE_DEFAULT)
    p.add_argument("--decompose-max-new-tokens", type=int, default=DECOMPOSE_MAX_NEW_TOKENS_DEFAULT)
    p.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    p.add_argument("--top-p", type=float, default=TOP_P_DEFAULT)
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN_DEFAULT)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--overwrite", action="store_true", help="Delete existing decomposition output before running.")
    p.add_argument(
        "--refresh-failed-decompositions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-run parse-failed or all-empty cached rows.",
    )
    p.add_argument(
        "--refresh-all-decompositions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore all cached decomposition rows and regenerate every item.",
    )
    return p.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    grant_db = resolve_path(args.grant_db)
    fac_db = resolve_path(args.fac_db)
    output_dir = resolve_path(args.output_dir)
    output_path = resolve_path(args.decomposition_output)
    if args.overwrite:
        _unlink_if_exists(output_path)

    grant_specs = load_grant_specializations(grant_db, max_items=args.max_grant_specs, seed=args.seed)
    fac_specs = load_faculty_specializations(fac_db, max_items=args.max_fac_specs, seed=args.seed)
    items = [*grant_specs, *fac_specs]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "stage": "ce3_decompose_setup",
                "model_id": args.model_id,
                "grant_db": str(grant_db),
                "fac_db": str(fac_db),
                "decomposition_output": str(output_path),
                "aspects": list(ASPECTS),
                "seed": int(args.seed),
                "grant_specs_loaded": int(len(grant_specs)),
                "fac_specs_loaded": int(len(fac_specs)),
                "items_total": int(len(items)),
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
        existing = load_jsonl_by_key(output_path, "item_id")
        before = len(existing)
        existing = refresh_decomposition_cache(
            existing,
            refresh_failed_only=bool(args.refresh_failed_decompositions),
            refresh_all=bool(args.refresh_all_decompositions),
            aspects=ASPECTS,
        )
        refreshed = before - len(existing)
        if refreshed:
            mode = "all" if args.refresh_all_decompositions else "failed_or_empty"
            print(f"decompose_refresh_{mode}={refreshed} decompose_cached_kept={len(existing)}")

        decompose_specializations(
            llm_bundle=bundle,
            model_id=args.model_id,
            items=items,
            existing=existing,
            output_path=output_path,
            batch_size=args.decompose_batch_size,
            max_new_tokens=args.decompose_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_attempts=args.max_attempts,
        )
    finally:
        unload_llm(bundle)

    print(f"elapsed_sec={time.time() - started:.2f}")
    print(f"decomposition_jsonl={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
