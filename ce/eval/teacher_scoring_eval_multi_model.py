from __future__ import annotations

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
# Run configuration (edit these directly)
# ======================================================
MODEL_IDS = [
    "Qwen/Qwen2.5-14B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "prithivMLmods/Ophiuchi-Qwen3-14B-Instruct",
    "Qwen/Qwen3-14B",
]

MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0
TOP_P = 1.0
BATCH_SIZE = 10

# Use JSON input file so one run can include many queries, each with many docs.
USE_EVAL_INPUT_JSON = True
EVAL_INPUT_JSON = "ce/dataset/eval/teacher_scoring_eval_input.json"
OUTPUT_JSON = "ce/eval/results/teacher_scoring_eval_output.json"

METHOD_SYSTEM_PROMPT = """
You are a strict evaluator of methodological similarity between a requirement query and a candidate specialization.

Evaluate overlap in:
- methods
- techniques
- procedures
- workflows
- analytical approaches
- technical mechanisms

Do NOT reward overlap that is only:
- same application area
- same domain/topic
- same population or environment
- generic data work
- broad scientific interest

Two texts can be in the same domain and still have low method similarity if they use different approaches.

Final output must be exactly one JSON object.

Required JSON schema:
{
  "score": <float in [0,1]>,
  "reason": "<one short sentence>",
  "band": "<high|mid|low>"
}

Band guidance:
- high: score >= 0.70 (strong overlap in methods/techniques/procedures)
- mid: 0.40 <= score < 0.70 (partial or indirect method overlap)
- low: score < 0.40 (little or no real method overlap)

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

Evaluate overlap in:
- application domain
- subject area
- problem space
- research or operational context

Do NOT reward overlap that is only:
- same generic method
- same data-processing language
- same technical workflow
- same analytical style

Two texts can use similar methods and still have low domain similarity if they address different subject areas.

Final output must be exactly one JSON object.

Required JSON schema:
{
  "score": <float in [0,1]>,
  "reason": "<one short sentence>",
  "band": "<high|mid|low>"
}

Band guidance:
- high: score >= 0.70 (same or very close domain/topic)
- mid: 0.40 <= score < 0.70 (related but not the same domain)
- low: score < 0.40 (different domain/topic)

No markdown or extra text outside JSON.
""".strip()


DOMAIN_USER_PROMPT_TEMPLATE = """
Requirement query:
{query}

Candidate specialization:
{candidate}
""".strip()

REQUIREMENT_SYSTEM_PROMPT = """
You are a strict requirement-match judge.
Final output must be exactly one JSON object.

Required JSON schema:
{
  "score": <float in [0,1]>,
  "reason": "<one short sentence>",
  "band": "<high|mid|low>"
}

Band mapping guidance:
- high: score >= 0.70
- mid: 0.40 <= score < 0.70
- low: score < 0.40

No markdown or extra text outside JSON.
""".strip()

REQUIREMENT_USER_PROMPT_TEMPLATE = """
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


def _load_eval_pairs_from_json(path: Path) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse eval JSON: {path} ({type(exc).__name__}: {exc})") from exc

    if isinstance(obj, dict):
        items = obj.get("items")
        if not isinstance(items, list):
            raise RuntimeError("Eval JSON object must contain `items` as a list.")
    elif isinstance(obj, list):
        items = obj
    else:
        raise RuntimeError("Eval JSON must be either a list or an object with `items` list.")

    pairs: List[Dict[str, Any]] = []
    pair_seq = 0
    for qi, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        query_text = _normalize_ws(item.get("query") or item.get("query_text") or item.get("text"))
        if not query_text:
            continue
        docs_raw = item.get("docs")
        if not isinstance(docs_raw, list):
            continue
        for di, d in enumerate(docs_raw):
            if isinstance(d, dict):
                doc_text = _normalize_ws(d.get("text") or d.get("doc_text") or d.get("candidate"))
            else:
                doc_text = _normalize_ws(d)
            if not doc_text:
                continue
            pair_seq += 1
            pairs.append(
                {
                    "pair_idx": int(pair_seq - 1),
                    "query_idx": int(qi),
                    "doc_idx": int(di),
                    "query_text": query_text,
                    "doc_text": doc_text,
                }
            )

    return pairs


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
        "score": float(score),
        "band": str(band),
        "reason": reason,
        "parsed_ok": bool(parsed is not None),
    }


def _build_output_json(
    *,
    pairs: Sequence[Dict[str, Any]],
    model_ids: Sequence[str],
    aspects: Sequence[str],
    scored: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> Dict[str, Any]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for p in pairs:
        grouped.setdefault(int(p["query_idx"]), []).append(p)

    queries_out: List[Dict[str, Any]] = []
    for qidx in sorted(grouped.keys()):
        rows = grouped[qidx]
        query_text = _clean_text(rows[0].get("query_text"))
        docs_out: List[Dict[str, Any]] = []
        for p in rows:
            pi = int(p["pair_idx"])
            doc_row: Dict[str, Any] = {"doc": _clean_text(p.get("doc_text"))}
            for aspect in aspects:
                score_cols: Dict[str, float] = {}
                for model_id in model_ids:
                    score_cols[model_id] = float(scored[(model_id, aspect, pi)]["score"])
                doc_row[f"{aspect}_scores"] = score_cols
            docs_out.append(doc_row)
        queries_out.append(
            {
                "query": query_text,
                "docs": docs_out,
            }
        )

    return {"queries": queries_out}


def main() -> int:
    output_path = _resolve_path(OUTPUT_JSON)
    if bool(USE_EVAL_INPUT_JSON):
        input_path = _resolve_path(EVAL_INPUT_JSON)
        if not input_path.exists():
            raise RuntimeError(f"Eval input JSON not found: {input_path}")
        pairs = _load_eval_pairs_from_json(input_path)
        if not pairs:
            raise RuntimeError(f"No valid query-doc pairs loaded from eval JSON: {input_path}")
    else:
        query = _normalize_ws(EVAL_QUERY)
        docs = [_normalize_ws(d) for d in EVAL_DOCS if _normalize_ws(d)]
        if not query:
            raise RuntimeError("EVAL_QUERY is empty.")
        if len(docs) != 10:
            raise RuntimeError(f"EVAL_DOCS must contain exactly 10 non-empty docs. Got {len(docs)}")
        pairs = []
        for di, doc in enumerate(docs):
            pairs.append(
                {
                    "pair_idx": int(di),
                    "query_idx": 0,
                    "doc_idx": int(di),
                    "query_text": query,
                    "doc_text": doc,
                }
            )

    model_ids = list(MODEL_IDS)
    batch_size = max(1, int(BATCH_SIZE))

    aspects: List[Tuple[str, str, str]] = [
        ("method", METHOD_SYSTEM_PROMPT, METHOD_USER_PROMPT_TEMPLATE),
        ("domain", DOMAIN_SYSTEM_PROMPT, DOMAIN_USER_PROMPT_TEMPLATE),
        ("requirement", REQUIREMENT_SYSTEM_PROMPT, REQUIREMENT_USER_PROMPT_TEMPLATE),
    ]

    scored: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    for model_id in model_ids:
        llm_bundle = _load_llm(model_id)
        try:
            tokenizer = llm_bundle["tokenizer"]
            for aspect_name, system_prompt, user_template in aspects:
                prompts: List[str] = []
                for p in pairs:
                    user_prompt = user_template.format(
                        query=_clean_text(p.get("query_text")),
                        candidate=_clean_text(p.get("doc_text")),
                    )
                    prompts.append(
                        _build_prompt(
                            tokenizer,
                            model_id=model_id,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                        )
                    )

                done = 0
                for prompt_batch in _batched(prompts, batch_size):
                    raw_batch = _generate_responses_batch(
                        llm_bundle=llm_bundle,
                        prompts=prompt_batch,
                        max_new_tokens=int(MAX_NEW_TOKENS),
                        temperature=float(TEMPERATURE),
                        top_p=float(TOP_P),
                    )
                    for local_i, raw_text in enumerate(raw_batch):
                        pi = done + local_i
                        scored[(model_id, aspect_name, pi)] = _parse_score_response(raw_text)
                    done += len(raw_batch)
        finally:
            _unload_llm(llm_bundle)

    aspect_names = [a[0] for a in aspects]
    output_obj = _build_output_json(
        pairs=pairs,
        model_ids=model_ids,
        aspects=aspect_names,
        scored=scored,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
