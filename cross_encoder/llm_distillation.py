from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


# Requested model family (closest available instruct model in this setup).
MODEL_ID_DEFAULT = "Qwen/Qwen2.5-32B-Instruct"

GRANT_DB_DEFAULT = "cross_encoder/dataset/grant_keywords_spec_keywords_db.json"
FAC_DB_DEFAULT = "cross_encoder/dataset/fac_chunks_db.json"

RAW_OUTPUT_DEFAULT = "cross_encoder/dataset/llm_distill_raw_scores.jsonl"
PAIRWISE_OUTPUT_DEFAULT = "cross_encoder/dataset/llm_distill_pairwise.jsonl"
LISTWISE_OUTPUT_DEFAULT = "cross_encoder/dataset/llm_distill_listwise.jsonl"
MANIFEST_OUTPUT_DEFAULT = "cross_encoder/dataset/llm_distill_manifest.json"


SYSTEM_PROMPT = """
You are evaluating whether a candidate text chunk satisfies a specialization requirement.

This is NOT general similarity — it is REQUIREMENT MATCHING.

Return ONLY strict JSON:
{
  "score": <float between 0.0 and 1.0>,
  "reason": "<one short sentence>"
}

Evaluation rules:

1. Identify core requirements:
   - learning-based methods (RL, multimodal learning)
   - sim-to-real transfer
   - contact-rich manipulation
   - humanoid locomotion control

2. Scoring principles:
   - Strong match requires BOTH domain AND method alignment.
   - Domain-only overlap (e.g., robotics/control without learning) is weak, not zero.

3. Penalties (soft, not absolute):
   - Missing learning-based methods → score should NOT exceed 0.4
   - Missing sim-to-real → score should NOT exceed 0.5
   - Missing multiple core elements → push score toward lower range (0.1–0.3)

4. Scoring meaning:
- 0.9–1.0: strong match with required methods
- 0.7–0.89: good match, minor gaps
- 0.4–0.69: partial match
- 0.1–0.39: domain overlap only
- 0.0–0.09: completely unrelated

IMPORTANT:
Do NOT assign 0.0 unless the topic is completely unrelated.
"""

USER_PROMPT_TEMPLATE = """
Grant specialization keyword:
{spec_text}

Faculty chunk:
{chunk_text}
""".strip()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


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


def _safe_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _truncate(value: str, *, max_chars: int) -> str:
    text = _clean_text(value)
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        score = 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _extract_score(raw_text: str) -> Tuple[float, bool]:
    raw = _clean_text(raw_text)
    if not raw:
        return 0.0, False

    if raw.startswith("{") and raw.count("{") > raw.count("}"):
        raw = raw + ("}" * (raw.count("{") - raw.count("}")))

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and ("score" in obj):
            return _clamp_score(obj.get("score")), True
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and ("score" in obj):
                return _clamp_score(obj.get("score")), True
        except Exception:
            pass

    n = re.search(r"[-+]?\d*\.?\d+", raw)
    if n:
        return _clamp_score(n.group(0)), False
    return 0.0, False


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON: {path} ({type(e).__name__}: {e})") from e
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected top-level JSON object in {path}")
    return payload


def _build_prompt(tokenizer: Any, *, spec_text: str, chunk_text: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not support apply_chat_template().")
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("tokenizer.chat_template is missing for this model/tokenizer.")

    user_prompt = USER_PROMPT_TEMPLATE.format(spec_text=spec_text, chunk_text=chunk_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _flatten_specs(grant_payload: Dict[str, Any], *, max_specs: int) -> List[Dict[str, Any]]:
    grants = list(grant_payload.get("grants") or [])
    out: List[Dict[str, Any]] = []
    for grant in grants:
        if not isinstance(grant, dict):
            continue
        grant_id = _clean_text(grant.get("grant_id"))
        if not grant_id:
            continue
        specs = list(grant.get("grant_spec_keywords") or [])
        for spec_idx, spec_text in enumerate(specs):
            text_value = _clean_text(spec_text)
            if not text_value:
                continue
            out.append(
                {
                    "grant_id": grant_id,
                    "spec_idx": int(spec_idx),
                    "spec_text": text_value,
                    "query_text": text_value,
                }
            )
            if max_specs > 0 and len(out) >= max_specs:
                return out
    return out


def _flatten_chunks(
    fac_payload: Dict[str, Any],
    *,
    max_chunks: int,
    max_chunk_chars: int,
) -> List[Dict[str, Any]]:
    chunks = list(fac_payload.get("fac_chunks") or [])
    out: List[Dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        fac_id = _safe_int(chunk.get("fac_id"), default=0, minimum=0, maximum=2_147_483_647)
        chunk_id = _safe_int(chunk.get("chunk_id"), default=0, minimum=0, maximum=2_147_483_647)
        chunk_index = _safe_int(chunk.get("chunk_index"), default=0, minimum=0, maximum=1_000_000)
        source_type = _clean_text(chunk.get("source_type")) or "unknown"
        chunk_text = _truncate(_clean_text(chunk.get("text")), max_chars=max_chunk_chars)
        if fac_id <= 0 or chunk_id <= 0 or not chunk_text:
            continue
        out.append(
            {
                "fac_id": fac_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "source_type": source_type,
                "chunk_text": chunk_text,
            }
        )
        if max_chunks > 0 and len(out) >= max_chunks:
            return out
    return out


def _chunked(items: Sequence[Any], batch_size: int) -> List[Sequence[Any]]:
    size = max(1, int(batch_size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _score_one_spec_against_all_chunks(
    *,
    llm: Any,
    tokenizer: Any,
    sampling_params: Any,
    spec_item: Dict[str, Any],
    chunks: Sequence[Dict[str, Any]],
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    candidates: List[Dict[str, Any]] = []
    json_ok_count = 0
    fallback_count = 0

    spec_text = _clean_text(spec_item.get("spec_text"))
    for batch in _chunked(chunks, batch_size):
        prompts = [
            _build_prompt(
                tokenizer,
                spec_text=spec_text,
                chunk_text=_clean_text(c.get("chunk_text")),
            )
            for c in batch
        ]
        outputs = llm.generate(prompts, sampling_params)
        for chunk_item, out in zip(batch, outputs):
            raw_text = ""
            if out and getattr(out, "outputs", None):
                first = out.outputs[0] if out.outputs else None
                raw_text = _clean_text(getattr(first, "text", ""))
            score, is_json_ok = _extract_score(raw_text)
            if is_json_ok:
                json_ok_count += 1
            else:
                fallback_count += 1
            candidates.append(
                {
                    "fac_id": int(chunk_item["fac_id"]),
                    "chunk_id": int(chunk_item["chunk_id"]),
                    "chunk_index": int(chunk_item["chunk_index"]),
                    "source_type": _clean_text(chunk_item["source_type"]),
                    "chunk_text": _clean_text(chunk_item["chunk_text"]),
                    "score": float(score),
                }
            )
    return candidates, json_ok_count, fallback_count


def _build_pairwise_records(
    *,
    query_text: str,
    grant_id: str,
    spec_idx: int,
    sorted_candidates: Sequence[Dict[str, Any]],
    pos_top_k: int,
    weak_bottom_k: int,
    hard_neg_k: int,
    min_margin: float,
    max_pairs_per_query: int,
) -> List[Dict[str, Any]]:
    if not sorted_candidates:
        return []

    pos_k = max(1, int(pos_top_k))
    weak_k = max(1, int(weak_bottom_k))
    hard_k = max(0, int(hard_neg_k))
    max_pairs = max(1, int(max_pairs_per_query))

    positives = list(sorted_candidates[: min(pos_k, len(sorted_candidates))])
    weak_negatives = list(sorted_candidates[-min(weak_k, len(sorted_candidates)) :])
    weak_negatives.sort(key=lambda x: float(x.get("score") or 0.0))

    hard_start = min(len(sorted_candidates), len(positives))
    hard_end = min(len(sorted_candidates), hard_start + hard_k)
    hard_negatives = list(sorted_candidates[hard_start:hard_end])

    out: List[Dict[str, Any]] = []

    for pos in positives:
        p_score = float(pos.get("score") or 0.0)

        for neg in weak_negatives:
            if int(pos.get("chunk_id") or 0) == int(neg.get("chunk_id") or 0) and int(pos.get("fac_id") or 0) == int(
                neg.get("fac_id") or 0
            ):
                continue
            n_score = float(neg.get("score") or 0.0)
            margin = p_score - n_score
            if margin <= 0:
                continue
            out.append(
                {
                    "grant_id": grant_id,
                    "spec_idx": int(spec_idx),
                    "query_text": query_text,
                    "pos_text": _clean_text(pos.get("chunk_text")),
                    "neg_text": _clean_text(neg.get("chunk_text")),
                    "teacher_pos_score": p_score,
                    "teacher_neg_score": n_score,
                    "teacher_margin": float(margin),
                    "pair_type": "strong_vs_weak",
                    "pos_fac_id": int(pos.get("fac_id") or 0),
                    "pos_chunk_id": int(pos.get("chunk_id") or 0),
                    "pos_chunk_index": int(pos.get("chunk_index") or 0),
                    "pos_source_type": _clean_text(pos.get("source_type")) or "unknown",
                    "neg_fac_id": int(neg.get("fac_id") or 0),
                    "neg_chunk_id": int(neg.get("chunk_id") or 0),
                    "neg_chunk_index": int(neg.get("chunk_index") or 0),
                    "neg_source_type": _clean_text(neg.get("source_type")) or "unknown",
                }
            )
            if len(out) >= max_pairs:
                return out

        for neg in hard_negatives:
            if int(pos.get("chunk_id") or 0) == int(neg.get("chunk_id") or 0) and int(pos.get("fac_id") or 0) == int(
                neg.get("fac_id") or 0
            ):
                continue
            n_score = float(neg.get("score") or 0.0)
            margin = p_score - n_score
            if margin < float(min_margin):
                continue
            out.append(
                {
                    "grant_id": grant_id,
                    "spec_idx": int(spec_idx),
                    "query_text": query_text,
                    "pos_text": _clean_text(pos.get("chunk_text")),
                    "neg_text": _clean_text(neg.get("chunk_text")),
                    "teacher_pos_score": p_score,
                    "teacher_neg_score": n_score,
                    "teacher_margin": float(margin),
                    "pair_type": "strong_vs_hard",
                    "pos_fac_id": int(pos.get("fac_id") or 0),
                    "pos_chunk_id": int(pos.get("chunk_id") or 0),
                    "pos_chunk_index": int(pos.get("chunk_index") or 0),
                    "pos_source_type": _clean_text(pos.get("source_type")) or "unknown",
                    "neg_fac_id": int(neg.get("fac_id") or 0),
                    "neg_chunk_id": int(neg.get("chunk_id") or 0),
                    "neg_chunk_index": int(neg.get("chunk_index") or 0),
                    "neg_source_type": _clean_text(neg.get("source_type")) or "unknown",
                }
            )
            if len(out) >= max_pairs:
                return out

    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate three distillation datasets from all grant-spec x fac-chunk combinations "
            "with batched vLLM scoring: raw score, pairwise, and listwise."
        )
    )
    p.add_argument("--grant-db", type=str, default=GRANT_DB_DEFAULT, help="Grant JSON DB path.")
    p.add_argument("--fac-db", type=str, default=FAC_DB_DEFAULT, help="Faculty chunk JSON DB path.")
    p.add_argument("--raw-output", type=str, default=RAW_OUTPUT_DEFAULT, help="Raw-score JSONL output path.")
    p.add_argument("--pairwise-output", type=str, default=PAIRWISE_OUTPUT_DEFAULT, help="Pairwise JSONL output path.")
    p.add_argument("--listwise-output", type=str, default=LISTWISE_OUTPUT_DEFAULT, help="Listwise JSONL output path.")
    p.add_argument("--manifest-output", type=str, default=MANIFEST_OUTPUT_DEFAULT, help="Manifest JSON output path.")

    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT, help="vLLM model id.")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    p.add_argument("--batch-size", type=int, default=64, help="vLLM generation batch size.")
    p.add_argument("--max-new-tokens", type=int, default=24, help="max_tokens for score generation.")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")

    p.add_argument("--max-specs", type=int, default=0, help="Limit specs for smoke run (0 = all).")
    p.add_argument("--max-chunks", type=int, default=0, help="Limit chunks for smoke run (0 = all).")
    p.add_argument("--max-chunk-chars", type=int, default=2400, help="Trim chunk text to this many chars (0 = no trim).")
    p.add_argument("--start-spec-index", type=int, default=0, help="Skip first N specs (manual resume).")

    p.add_argument("--pair-pos-top-k", type=int, default=4, help="Top-K positive pool per query.")
    p.add_argument("--pair-weak-bottom-k", type=int, default=4, help="Bottom-K weak negative pool per query.")
    p.add_argument("--pair-hard-neg-k", type=int, default=4, help="Near-top hard negative pool per query.")
    p.add_argument("--pair-min-margin", type=float, default=0.05, help="Minimum margin for strong_vs_hard pairs.")
    p.add_argument("--pair-max-per-query", type=int, default=80, help="Cap pairwise rows generated per query.")

    p.add_argument("--listwise-top-k", type=int, default=0, help="Keep top-K docs in listwise output (0 = all).")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        raise RuntimeError(
            "vLLM is required but not installed in this environment. "
            "Install `vllm` in your runtime, then rerun."
        ) from e

    grant_db = Path(_clean_text(args.grant_db)).expanduser().resolve()
    fac_db = Path(_clean_text(args.fac_db)).expanduser().resolve()
    raw_output = Path(_clean_text(args.raw_output)).expanduser().resolve()
    pairwise_output = Path(_clean_text(args.pairwise_output)).expanduser().resolve()
    listwise_output = Path(_clean_text(args.listwise_output)).expanduser().resolve()
    manifest_output = Path(_clean_text(args.manifest_output)).expanduser().resolve()

    model_id = _clean_text(args.model_id) or MODEL_ID_DEFAULT
    tensor_parallel_size = _safe_int(args.tensor_parallel_size, default=1, minimum=1, maximum=64)
    batch_size = _safe_int(args.batch_size, default=64, minimum=1, maximum=4096)
    max_new_tokens = _safe_int(args.max_new_tokens, default=24, minimum=1, maximum=2048)
    temperature = _safe_float(args.temperature, default=0.0, minimum=0.0, maximum=2.0)

    max_specs = _safe_int(args.max_specs, default=0, minimum=0, maximum=50_000_000)
    max_chunks = _safe_int(args.max_chunks, default=0, minimum=0, maximum=50_000_000)
    max_chunk_chars = _safe_int(args.max_chunk_chars, default=2400, minimum=0, maximum=1_000_000)
    start_spec_index = _safe_int(args.start_spec_index, default=0, minimum=0, maximum=50_000_000)

    pair_pos_top_k = _safe_int(args.pair_pos_top_k, default=4, minimum=1, maximum=10_000)
    pair_weak_bottom_k = _safe_int(args.pair_weak_bottom_k, default=4, minimum=1, maximum=10_000)
    pair_hard_neg_k = _safe_int(args.pair_hard_neg_k, default=4, minimum=0, maximum=10_000)
    pair_min_margin = _safe_float(args.pair_min_margin, default=0.05, minimum=0.0, maximum=1.0)
    pair_max_per_query = _safe_int(args.pair_max_per_query, default=80, minimum=1, maximum=1_000_000)
    listwise_top_k = _safe_int(args.listwise_top_k, default=0, minimum=0, maximum=10_000_000)

    if not grant_db.exists():
        raise RuntimeError(f"Grant DB not found: {grant_db}")
    if not fac_db.exists():
        raise RuntimeError(f"Faculty DB not found: {fac_db}")

    grant_payload = _load_json(grant_db)
    fac_payload = _load_json(fac_db)

    specs_all = _flatten_specs(grant_payload, max_specs=max_specs)
    chunks = _flatten_chunks(fac_payload, max_chunks=max_chunks, max_chunk_chars=max_chunk_chars)
    if not specs_all:
        raise RuntimeError("No grant specialization keywords found.")
    if not chunks:
        raise RuntimeError("No faculty chunks found with non-empty text.")

    specs = specs_all[start_spec_index:] if start_spec_index > 0 else specs_all
    if not specs:
        raise RuntimeError("No specs left after start-spec-index filtering.")

    total_pairs = int(len(specs) * len(chunks))

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    pairwise_output.parent.mkdir(parents=True, exist_ok=True)
    listwise_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model_id,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature)

    started = time.time()
    total_scored = 0
    total_pairwise = 0
    total_json_ok = 0
    total_fallback = 0

    print(f"model_id={model_id}")
    print(f"spec_count={len(specs)}")
    print(f"chunk_count={len(chunks)}")
    print(f"total_pairs={total_pairs}")
    print(f"batch_size={batch_size}")
    print(f"raw_output={raw_output}")
    print(f"pairwise_output={pairwise_output}")
    print(f"listwise_output={listwise_output}")

    with (
        raw_output.open("w", encoding="utf-8") as raw_f,
        pairwise_output.open("w", encoding="utf-8") as pair_f,
        listwise_output.open("w", encoding="utf-8") as list_f,
    ):
        for spec_rank, spec_item in enumerate(specs, start=1):
            candidates, json_ok, fallback = _score_one_spec_against_all_chunks(
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                spec_item=spec_item,
                chunks=chunks,
                batch_size=batch_size,
            )

            total_scored += len(candidates)
            total_json_ok += int(json_ok)
            total_fallback += int(fallback)

            raw_obj = {
                "grant_id": _clean_text(spec_item.get("grant_id")),
                "spec_idx": int(spec_item.get("spec_idx") or 0),
                "spec_text": _clean_text(spec_item.get("spec_text")),
                "candidates": [
                    {
                        "fac_id": int(c["fac_id"]),
                        "chunk_id": int(c["chunk_id"]),
                        "chunk_index": int(c["chunk_index"]),
                        "source_type": _clean_text(c["source_type"]),
                        "chunk_text": _clean_text(c["chunk_text"]),
                        "score": float(c["score"]),
                    }
                    for c in candidates
                ],
            }
            raw_f.write(json.dumps(raw_obj, ensure_ascii=False) + "\n")

            sorted_candidates = sorted(
                candidates,
                key=lambda x: float(x.get("score") or 0.0),
                reverse=True,
            )

            docs = sorted_candidates
            if listwise_top_k > 0:
                docs = docs[: min(listwise_top_k, len(docs))]
            listwise_obj = {
                "grant_id": _clean_text(spec_item.get("grant_id")),
                "spec_idx": int(spec_item.get("spec_idx") or 0),
                "query_text": _clean_text(spec_item.get("query_text")),
                "docs": [
                    {
                        "text": _clean_text(d["chunk_text"]),
                        "teacher_score": float(d["score"]),
                        "fac_id": int(d["fac_id"]),
                        "chunk_id": int(d["chunk_id"]),
                        "chunk_index": int(d["chunk_index"]),
                        "source_type": _clean_text(d["source_type"]),
                    }
                    for d in docs
                ],
            }
            list_f.write(json.dumps(listwise_obj, ensure_ascii=False) + "\n")

            pairwise_rows = _build_pairwise_records(
                query_text=_clean_text(spec_item.get("query_text")),
                grant_id=_clean_text(spec_item.get("grant_id")),
                spec_idx=int(spec_item.get("spec_idx") or 0),
                sorted_candidates=sorted_candidates,
                pos_top_k=pair_pos_top_k,
                weak_bottom_k=pair_weak_bottom_k,
                hard_neg_k=pair_hard_neg_k,
                min_margin=pair_min_margin,
                max_pairs_per_query=pair_max_per_query,
            )
            for row in pairwise_rows:
                pair_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_pairwise += len(pairwise_rows)

            raw_f.flush()
            list_f.flush()
            pair_f.flush()

            if spec_rank % 10 == 0 or spec_rank == len(specs):
                elapsed = max(1e-6, time.time() - started)
                speed = total_scored / elapsed
                print(
                    f"spec_progress={spec_rank}/{len(specs)} "
                    f"scored_pairs={total_scored}/{total_pairs} "
                    f"pairwise_rows={total_pairwise} "
                    f"speed={speed:.2f} pairs/sec"
                )

    elapsed = max(1e-6, time.time() - started)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "grant_db": str(grant_db),
        "fac_db": str(fac_db),
        "raw_output": str(raw_output),
        "pairwise_output": str(pairwise_output),
        "listwise_output": str(listwise_output),
        "spec_count": int(len(specs)),
        "chunk_count": int(len(chunks)),
        "total_pairs": int(total_pairs),
        "total_scored_pairs": int(total_scored),
        "total_pairwise_rows": int(total_pairwise),
        "parsed_json_ok_count": int(total_json_ok),
        "parsed_fallback_count": int(total_fallback),
        "tensor_parallel_size": int(tensor_parallel_size),
        "batch_size": int(batch_size),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "max_chunk_chars": int(max_chunk_chars),
        "start_spec_index": int(start_spec_index),
        "max_specs": int(max_specs),
        "max_chunks": int(max_chunks),
        "pair_pos_top_k": int(pair_pos_top_k),
        "pair_weak_bottom_k": int(pair_weak_bottom_k),
        "pair_hard_neg_k": int(pair_hard_neg_k),
        "pair_min_margin": float(pair_min_margin),
        "pair_max_per_query": int(pair_max_per_query),
        "listwise_top_k": int(listwise_top_k),
        "elapsed_seconds": float(elapsed),
        "pairs_per_second": float(total_scored / elapsed),
    }
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("done=true")
    print(f"total_scored_pairs={total_scored}")
    print(f"total_pairwise_rows={total_pairwise}")
    print(f"parsed_json_ok_count={total_json_ok}")
    print(f"parsed_fallback_count={total_fallback}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"pairs_per_second={(total_scored / elapsed):.4f}")
    print(f"manifest_output={manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
