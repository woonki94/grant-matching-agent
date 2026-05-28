from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
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
    resolve_path,
)

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # type: ignore[assignment]


MODEL_ID_DEFAULT = "Qwen/Qwen3-14B"
DECOMPOSITION_OUTPUT_DEFAULT = "ce3/dataset/decomposed/spec_decompositions_topic_approach_objective.jsonl"
OUTPUT_DIR_DEFAULT = "ce3/dataset/augmented"
AUGMENTATION_OUTPUT_DEFAULT = "ce3/dataset/augmented/spec_augmentations_high.jsonl"
AUGMENTATIONS_PER_ASPECT_DEFAULT = 4
AUGMENT_BATCH_SIZE_DEFAULT = 12
AUGMENT_MAX_NEW_TOKENS_DEFAULT = 384
MAX_ATTEMPTS_DEFAULT = 2
MAX_MODEL_LEN_DEFAULT = 4096
TEMPERATURE_DEFAULT = 0.7
TOP_P_DEFAULT = 0.9

ASPECTS = ("topic", "approach", "objective")


ASPECT_DEFINITIONS = {
    "topic": "topic, problem area, technical context, field, or service area",
    "approach": "capability, method, technique, action, workflow, model, intervention, or work performed",
    "objective": "target object, beneficiary, system, material, outcome, condition, use case, or intended purpose",
}


SYSTEM_PROMPT = """
You generate synthetic specialization phrases for cross-encoder data augmentation.

The synthetic phrase is not a label. It is only a candidate that will be scored later by a teacher model.

Goal:
- Match the source specialization strongly on exactly one requested attention lens.
- Make the other two lenses meaningfully different when possible, so the generated phrase creates useful contrast.
- Keep the phrase realistic as a grant/faculty specialization keyword.

Attention lenses:
- topic: the main subject, problem area, technical context, field, or service area.
- approach: the capability, method, technique, action, workflow, model, intervention, or work performed.
- objective: the object, system, population, material, outcome, condition, use case, or intended purpose.

Rules:
- Preserve the requested high-match lens clearly and specifically.
- Vary the non-target lenses. Do not make a near-duplicate of the source phrase.
- Do not add names of real people, institutions, grants, or private entities.
- Do not invent overly broad filler.
- Each output should be one concise standalone specialization phrase, not a sentence with explanations.
- Return exactly one JSON object and no markdown.

Required schema:
{"augmentations": ["..."]}
""".strip()


USER_PROMPT_TEMPLATE = """
/no_think

Source specialization:
{source_text}

Source decomposition:
{source_decomposition_json}

Requested high-match lens:
{target_aspect}

Requested lens definition:
{target_definition}

Requested lens phrases to preserve semantically:
{target_items_json}

Generate {n} diverse synthetic specialization phrases.

The generated phrases should be high for {target_aspect} against the source, while the other two lenses should be different enough to create contrast.
""".strip()


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


def _parse_augmentations(obj: Optional[Dict[str, Any]], *, limit: int) -> tuple[List[str], bool]:
    if not isinstance(obj, dict):
        return [], False
    raw = obj.get("augmentations")
    values: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                values.extend(_as_list(item.get("text") or item.get("specialization") or item.get("phrase")))
            else:
                values.extend(_as_list(item))
    else:
        values = _as_list(raw)
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_ws(value)
        if not text:
            continue
        text = text.strip("\"'` ")
        text = text.rstrip(".")
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= int(limit):
            break
    return out, bool(values)


def _source_key(source_item_id: str, target_aspect: str, slot: int) -> str:
    return f"{source_item_id}::{target_aspect}::{int(slot)}"


def _item_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _augmented_kind(source: SpecItem) -> str:
    if source.kind == "grant":
        return "fac_aug"
    if source.kind == "faculty":
        return "grant_aug"
    return f"{source.kind}_aug"


def _target_side(source: SpecItem) -> str:
    return "faculty" if source.kind == "grant" else "grant"


def _make_aug_item_id(source: SpecItem, target_aspect: str, slot: int, text: str) -> str:
    h = _item_hash(f"{source.item_id}|{target_aspect}|{slot}|{normalize_ws(text).casefold()}")
    return f"aug:{_target_side(source)}:{source.item_id}:{target_aspect}:{int(slot)}:{h}"


def load_existing_source_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            meta = obj.get("meta")
            if not isinstance(meta, dict):
                continue
            source_item_id = clean_text(meta.get("source_item_id"))
            target_aspect = clean_text(meta.get("target_aspect"))
            slot = meta.get("slot")
            if source_item_id and target_aspect and slot is not None:
                out.add(_source_key(source_item_id, target_aspect, int(slot)))
    return out


def load_items_from_decompositions(path: Path) -> tuple[List[SpecItem], Dict[str, Dict[str, Any]]]:
    items: List[SpecItem] = []
    decompositions: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            item_id = clean_text(row.get("item_id"))
            kind = clean_text(row.get("kind"))
            text = normalize_ws(row.get("text"))
            if not item_id or not kind or not text:
                continue
            if kind not in {"grant", "faculty"}:
                continue
            if item_id in decompositions:
                continue
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            items.append(SpecItem(item_id=item_id, kind=kind, text=text, meta=dict(meta)))
            decompositions[item_id] = row
    return items, decompositions


def _pending_requests(
    *,
    items: Sequence[SpecItem],
    decompositions: Dict[str, Dict[str, Any]],
    existing_source_keys: set[str],
    augmentations_per_aspect: int,
) -> List[tuple[SpecItem, str]]:
    pending: List[tuple[SpecItem, str]] = []
    n = max(0, int(augmentations_per_aspect))
    for item in items:
        decomp = _safe_decomposition(decompositions.get(item.item_id, {}))
        for aspect in ASPECTS:
            if not decomp.get(aspect):
                continue
            if all(_source_key(item.item_id, aspect, slot) in existing_source_keys for slot in range(n)):
                continue
            pending.append((item, aspect))
    return pending


def augment_specializations(
    *,
    llm_bundle: Dict[str, Any],
    model_id: str,
    items: Sequence[SpecItem],
    decompositions: Dict[str, Dict[str, Any]],
    existing_source_keys: set[str],
    output_path: Path,
    augmentations_per_aspect: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_attempts: int,
) -> List[Dict[str, Any]]:
    tokenizer = llm_bundle["tokenizer"]
    pending = _pending_requests(
        items=items,
        decompositions=decompositions,
        existing_source_keys=existing_source_keys,
        augmentations_per_aspect=augmentations_per_aspect,
    )
    print(f"augment_existing_slots={len(existing_source_keys)} augment_pending_requests={len(pending)}")
    written_rows: List[Dict[str, Any]] = []

    for attempt in range(max(1, int(max_attempts))):
        if not pending:
            break
        next_pending: List[tuple[SpecItem, str]] = []
        chunks = list(batched(pending, int(batch_size)))
        chunk_iter: Iterable[Sequence[tuple[SpecItem, str]]] = chunks
        bar = None
        if tqdm is not None:
            bar = tqdm(
                chunks,
                total=len(chunks),
                desc=f"CE3 augment {attempt + 1}/{max(1, int(max_attempts))}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )
            chunk_iter = bar

        written_this_attempt = 0
        for chunk in chunk_iter:
            prompts: List[str] = []
            for item, aspect in chunk:
                decomp = _safe_decomposition(decompositions.get(item.item_id, {}))
                prompts.append(
                    build_prompt(
                        tokenizer,
                        model_id=model_id,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=USER_PROMPT_TEMPLATE.format(
                            source_text=item.text,
                            source_decomposition_json=json.dumps(decomp, ensure_ascii=False),
                            target_aspect=aspect,
                            target_definition=ASPECT_DEFINITIONS[aspect],
                            target_items_json=json.dumps(decomp.get(aspect, []), ensure_ascii=False),
                            n=max(1, int(augmentations_per_aspect)),
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
            for (item, aspect), response in zip(chunk, responses):
                decomp = _safe_decomposition(decompositions.get(item.item_id, {}))
                texts, parsed = _parse_augmentations(
                    extract_json_object(response),
                    limit=max(1, int(augmentations_per_aspect)),
                )
                if not texts:
                    next_pending.append((item, aspect))
                    continue

                accepted_for_request = 0
                for text in texts:
                    slot = None
                    for candidate_slot in range(max(0, int(augmentations_per_aspect))):
                        key = _source_key(item.item_id, aspect, candidate_slot)
                        if key not in existing_source_keys:
                            slot = candidate_slot
                            break
                    if slot is None:
                        break
                    key = _source_key(item.item_id, aspect, int(slot))
                    row = {
                        "item_id": _make_aug_item_id(item, aspect, int(slot), text),
                        "kind": _augmented_kind(item),
                        "text": text,
                        "meta": {
                            "source_item_id": item.item_id,
                            "source_kind": item.kind,
                            "source_text": item.text,
                            "source_meta": item.meta,
                            "target_side": _target_side(item),
                            "target_aspect": aspect,
                            "intended_cluster": "high",
                            "slot": int(slot),
                            "contrast_aspects": [x for x in ASPECTS if x != aspect],
                        },
                        "source_decomposition": decomp,
                        "parse_ok": bool(parsed),
                        "attempt": int(attempt + 1),
                        "model_id": model_id,
                        "raw_response": response,
                    }
                    existing_source_keys.add(key)
                    rows.append(row)
                    written_rows.append(row)
                    accepted_for_request += 1

                if accepted_for_request < max(1, int(augmentations_per_aspect)):
                    next_pending.append((item, aspect))

            written_this_attempt += append_jsonl(output_path, rows)
            if bar is not None:
                bar.set_postfix(written=int(written_this_attempt), retry_pending=int(len(next_pending)))

        if bar is not None:
            bar.close()
        pending = next_pending
        if pending:
            print(f"augment_retry_pending={len(pending)} attempt={attempt + 1}")

    return written_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CE3 high-intent specialization augmentation runner.")
    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    p.add_argument("--decomposition-output", type=str, default=DECOMPOSITION_OUTPUT_DEFAULT)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--augmentation-output", type=str, default=AUGMENTATION_OUTPUT_DEFAULT)
    p.add_argument("--augmentations-per-aspect", type=int, default=AUGMENTATIONS_PER_ASPECT_DEFAULT)
    p.add_argument("--augment-batch-size", type=int, default=AUGMENT_BATCH_SIZE_DEFAULT)
    p.add_argument("--augment-max-new-tokens", type=int, default=AUGMENT_MAX_NEW_TOKENS_DEFAULT)
    p.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    p.add_argument("--top-p", type=float, default=TOP_P_DEFAULT)
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN_DEFAULT)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--overwrite", action="store_true", help="Delete existing augmentation output before running.")
    return p.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    decomposition_path = resolve_path(args.decomposition_output)
    output_dir = resolve_path(args.output_dir)
    output_path = resolve_path(args.augmentation_output)
    if not decomposition_path.exists():
        raise FileNotFoundError(f"Missing decomposition output: {decomposition_path}")
    if args.overwrite:
        _unlink_if_exists(output_path)

    items, decompositions = load_items_from_decompositions(decomposition_path)
    if not items:
        raise RuntimeError(f"No original grant/faculty decomposition rows found in {decomposition_path}")
    kind_counts = Counter(item.kind for item in items)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "stage": "ce3_augment_setup",
                "model_id": args.model_id,
                "decomposition_output": str(decomposition_path),
                "augmentation_output": str(output_path),
                "aspects": list(ASPECTS),
                "grant_specs_loaded": int(kind_counts.get("grant", 0)),
                "fac_specs_loaded": int(kind_counts.get("faculty", 0)),
                "items_total": int(len(items)),
                "augmentations_per_aspect": int(args.augmentations_per_aspect),
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
        rows = augment_specializations(
            llm_bundle=bundle,
            model_id=args.model_id,
            items=items,
            decompositions=decompositions,
            existing_source_keys=load_existing_source_keys(output_path),
            output_path=output_path,
            augmentations_per_aspect=args.augmentations_per_aspect,
            batch_size=args.augment_batch_size,
            max_new_tokens=args.augment_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_attempts=args.max_attempts,
        )
    finally:
        unload_llm(bundle)

    kind_counts = Counter(row.get("kind") for row in rows)
    aspect_counts = Counter((row.get("meta") or {}).get("target_aspect") for row in rows)
    print(f"elapsed_sec={time.time() - started:.2f}")
    print(f"augmentation_jsonl={output_path}")
    print(f"written_rows={len(rows)}")
    print(f"kind_counts={json.dumps(dict(kind_counts), ensure_ascii=False)}")
    print(f"target_aspect_counts={json.dumps(dict(aspect_counts), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
