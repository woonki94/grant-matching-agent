from __future__ import annotations

import gc
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce.llm_runtime_util import (  # noqa: E402
    batched as _batched,
    build_prompt as _build_prompt,
    clean_text as _clean_text,
    coerce_score as _coerce_score,
    extract_json_object as _extract_json_object,
    generate_responses_batch as _generate_responses_batch,
    load_llm as _load_llm,
    normalize_band as _normalize_band,
    normalize_ws as _normalize_ws,
    score_to_band as _score_to_band,
    unload_llm as _unload_llm,
)


# ======================================================
# Constants (no CLI args)
# ======================================================
# Source rows can be:
# 1) distill2-style rows with `ranked_docs` list
# 2) rows with `scored_candidates` list
# 3) flat pair rows with query/doc fields
SOURCE_JSONL = "ce/dataset/distill/llm_distill2_distill_raw_scores.jsonl"

METHOD_OUTPUT_JSONL = "ce/dataset/distill/llm_distill_method.jsonl"
DOMAIN_OUTPUT_JSONL = "ce/dataset/distill/llm_distill_domain.jsonl"

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
TOP_P = 1.0
BATCH_SIZE = 64
SAVE_RAW_RESPONSE = False

# If >0, only first N source rows are loaded (for quick tests).
MAX_SOURCE_ROWS = 0


METHOD_SYSTEM_PROMPT = """
You are a strict evaluator of methodological similarity between a requirement query and a candidate specialization.
Evaluate overlap in methods, techniques, procedures, and analytical approaches, not broad topical match.
Final output must be exactly one JSON object.

Required JSON schema:
{
  "score": <float in [0,1]>,
  "reason": "<one short sentence>",
  "band": "<high|mid|low>"
}

Band guidance:
- high: score >= 0.70 (strong method overlap)
- mid: 0.40 <= score < 0.70 (partial method overlap)
- low: score < 0.40 (little method overlap)

No markdown or extra text outside JSON.
""".strip()

METHOD_USER_PROMPT_TEMPLATE = """
Requirement query:
{query}

Candidate specialization:
{candidate}
""".strip()


DOMAIN_SYSTEM_PROMPT = """
You are a strict evaluator of domain/topic similarity between a requirement query and a candidate specialization.
Evaluate whether they operate in the same or very close application domain, not merely sharing generic methods.
Final output must be exactly one JSON object.

Required JSON schema:
{
  "score": <float in [0,1]>,
  "reason": "<one short sentence>",
  "band": "<high|mid|low>"
}

Band guidance:
- high: score >= 0.70 (same/very close domain)
- mid: 0.40 <= score < 0.70 (related domain)
- low: score < 0.40 (different domain)

No markdown or extra text outside JSON.
""".strip()

DOMAIN_USER_PROMPT_TEMPLATE = """
Requirement query:
{query}

Candidate specialization:
{candidate}
""".strip()
def _resolve_path(value: Any) -> Path:
    p = Path(_clean_text(value)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)




def _parse_score_response(raw: str) -> Dict[str, Any]:
    parsed = _extract_json_object(raw)
    score = _coerce_score((parsed or {}).get("score"))
    band = _normalize_band((parsed or {}).get("band"))
    reason = _clean_text((parsed or {}).get("reason"))

    if parsed is None:
        m = re.search(r"(?<!\\d)(0(?:\\.\\d+)?|1(?:\\.0+)?)", _clean_text(raw))
        if m:
            try:
                score = _coerce_score(float(m.group(1)))
            except Exception:
                pass
        band = _score_to_band(float(score))

    if band not in {"high", "mid", "low"}:
        band = _score_to_band(float(score))

    return {
        "teacher_score": float(score),
        "teacher_band": str(band),
        "teacher_reason": reason,
        "parsed_ok": bool(parsed is not None),
        "raw_response": _clean_text(raw),
    }


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _iter_source_rows(path: Path, *, max_rows: int) -> Sequence[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            raw = _clean_text(line)
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                continue
            rows.append(obj)
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def _extract_pairs(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    pair_seq = 0
    for row in rows:
        grant_id = _clean_text(row.get("grant_id"))
        spec_idx = _safe_int(row.get("spec_idx"), default=-1)
        query_text = _normalize_ws(row.get("query_text") or row.get("spec_text") or row.get("query"))
        if not query_text:
            continue

        ranked_docs = row.get("ranked_docs")
        if isinstance(ranked_docs, list) and ranked_docs:
            for doc in ranked_docs:
                if not isinstance(doc, dict):
                    continue
                doc_text = _normalize_ws(doc.get("text") or doc.get("doc_text") or doc.get("fac_spec_text"))
                if not doc_text:
                    continue
                pair_seq += 1
                pairs.append(
                    {
                        "pair_seq": int(pair_seq),
                        "grant_id": grant_id,
                        "spec_idx": int(spec_idx),
                        "query_text": query_text,
                        "doc_text": doc_text,
                        "fac_id": _safe_int(doc.get("fac_id"), default=0),
                        "fac_spec_id": _safe_int(doc.get("fac_spec_id"), default=0),
                        "fac_spec_idx": _safe_int(doc.get("fac_spec_idx"), default=0),
                        "section": _clean_text(doc.get("section")),
                        "source_rank": _safe_int(doc.get("rank"), default=-1),
                    }
                )
            continue

        scored_candidates = row.get("scored_candidates")
        if isinstance(scored_candidates, list) and scored_candidates:
            for idx, doc in enumerate(scored_candidates):
                if not isinstance(doc, dict):
                    continue
                doc_text = _normalize_ws(doc.get("fac_spec_text") or doc.get("text") or doc.get("doc_text"))
                if not doc_text:
                    continue
                pair_seq += 1
                pairs.append(
                    {
                        "pair_seq": int(pair_seq),
                        "grant_id": grant_id,
                        "spec_idx": int(spec_idx),
                        "query_text": query_text,
                        "doc_text": doc_text,
                        "fac_id": _safe_int(doc.get("fac_id"), default=0),
                        "fac_spec_id": _safe_int(doc.get("fac_spec_id"), default=0),
                        "fac_spec_idx": _safe_int(doc.get("fac_spec_idx"), default=0),
                        "section": _clean_text(doc.get("section")),
                        "source_rank": _safe_int(doc.get("sts_rank"), default=int(idx)),
                    }
                )
            continue

        doc_text = _normalize_ws(
            row.get("doc_text")
            or row.get("candidate_text")
            or row.get("fac_spec_text")
            or row.get("text")
        )
        if not doc_text:
            continue
        pair_seq += 1
        pairs.append(
            {
                "pair_seq": int(pair_seq),
                "grant_id": grant_id,
                "spec_idx": int(spec_idx),
                "query_text": query_text,
                "doc_text": doc_text,
                "fac_id": _safe_int(row.get("fac_id"), default=0),
                "fac_spec_id": _safe_int(row.get("fac_spec_id"), default=0),
                "fac_spec_idx": _safe_int(row.get("fac_spec_idx"), default=0),
                "section": _clean_text(row.get("section")),
                "source_rank": _safe_int(row.get("rank"), default=-1),
            }
        )
    return pairs


def _score_aspect(
    *,
    llm_bundle: Dict[str, Any],
    model_id: str,
    pairs: Sequence[Dict[str, Any]],
    aspect: str,
    system_prompt: str,
    user_template: str,
) -> List[Dict[str, Any]]:
    tokenizer = llm_bundle["tokenizer"]
    prompts: List[str] = []
    for row in pairs:
        user_prompt = user_template.format(
            query=_clean_text(row.get("query_text")),
            candidate=_clean_text(row.get("doc_text")),
        )
        prompts.append(
            _build_prompt(
                tokenizer,
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        )

    out_rows: List[Dict[str, Any]] = []
    done = 0
    batch_size = max(1, int(BATCH_SIZE))
    for prompt_batch in _batched(prompts, batch_size):
        raw_batch = _generate_responses_batch(
            llm_bundle=llm_bundle,
            prompts=prompt_batch,
            max_new_tokens=int(MAX_NEW_TOKENS),
            temperature=float(TEMPERATURE),
            top_p=float(TOP_P),
        )
        for local_i, raw_text in enumerate(raw_batch):
            global_i = done + local_i
            pair = dict(pairs[global_i])
            parsed = _parse_score_response(raw_text)
            row = {
                **pair,
                "aspect": aspect,
                "model_id": model_id,
                "teacher_score": float(parsed["teacher_score"]),
                "teacher_score_raw": float(parsed["teacher_score"]),
                "teacher_band": _clean_text(parsed["teacher_band"]),
                "teacher_reason": _clean_text(parsed["teacher_reason"]),
                "parsed_ok": bool(parsed["parsed_ok"]),
            }
            if bool(SAVE_RAW_RESPONSE):
                row["teacher_raw_response"] = _clean_text(parsed["raw_response"])
            out_rows.append(row)
        done += len(raw_batch)
        print(f"progress aspect={aspect} pair={done}/{len(pairs)}")
    return out_rows


def _print_summary(*, name: str, rows: Sequence[Dict[str, Any]]) -> None:
    n = len(rows)
    if n <= 0:
        print(f"{name}: rows=0")
        return
    ok = sum(1 for r in rows if bool(r.get("parsed_ok")))
    mean_score = sum(float(r.get("teacher_score") or 0.0) for r in rows) / float(n)
    high = sum(1 for r in rows if _clean_text(r.get("teacher_band")) == "high")
    mid = sum(1 for r in rows if _clean_text(r.get("teacher_band")) == "mid")
    low = sum(1 for r in rows if _clean_text(r.get("teacher_band")) == "low")
    print(
        f"{name}: rows={n} parsed_ok={ok}/{n} "
        f"mean_score={mean_score:.4f} bands(high/mid/low)={high}/{mid}/{low}"
    )


def main() -> int:
    source_path = _resolve_path(SOURCE_JSONL)
    method_output_path = _resolve_path(METHOD_OUTPUT_JSONL)
    domain_output_path = _resolve_path(DOMAIN_OUTPUT_JSONL)

    if not source_path.exists():
        raise RuntimeError(f"Source JSONL not found: {source_path}")

    source_rows = _iter_source_rows(source_path, max_rows=int(max(0, MAX_SOURCE_ROWS)))
    pairs = _extract_pairs(source_rows)
    if not pairs:
        raise RuntimeError(f"No valid query-doc pairs extracted from source: {source_path}")

    print(f"source_jsonl={source_path}")
    print(f"source_rows={len(source_rows)} extracted_pairs={len(pairs)}")
    print(f"model_id={MODEL_ID} batch_size={BATCH_SIZE}")
    print(f"method_output={method_output_path}")
    print(f"domain_output={domain_output_path}")

    llm_bundle = _load_llm(MODEL_ID)
    try:
        domain_rows = _score_aspect(
            llm_bundle=llm_bundle,
            model_id=MODEL_ID,
            pairs=pairs,
            aspect="domain",
            system_prompt=DOMAIN_SYSTEM_PROMPT,
            user_template=DOMAIN_USER_PROMPT_TEMPLATE,
        )
        method_rows = _score_aspect(
            llm_bundle=llm_bundle,
            model_id=MODEL_ID,
            pairs=pairs,
            aspect="method",
            system_prompt=METHOD_SYSTEM_PROMPT,
            user_template=METHOD_USER_PROMPT_TEMPLATE,
        )
    finally:
        _unload_llm(llm_bundle)
        gc.collect()

    _write_jsonl(domain_output_path, domain_rows)
    _write_jsonl(method_output_path, method_rows)

    _print_summary(name="domain", rows=domain_rows)
    _print_summary(name="method", rows=method_rows)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
