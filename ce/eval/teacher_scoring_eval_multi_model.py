from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "cross_encoder").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse proven local-LLM backend + requirement prompt style from cross_encoder.
from cross_encoder.eval.test_augment_with_multi_model import (  # noqa: E402
    JUDGE_SYSTEM_PROMPT as REQUIREMENT_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT_TEMPLATE as REQUIREMENT_USER_PROMPT_TEMPLATE,
    _build_prompt,
    _coerce_score,
    _extract_json_object,
    _generate_single_response,
    _load_llm,
    _normalize_band,
    _unload_llm,
)


DEFAULT_DB_JSON = "ce/dataset/source/grant_keywords_spec_keywords_db.json"
DEFAULT_FAC_DB_JSON = "ce/dataset/source/fac_specs_db.json"
DEFAULT_OUTPUT_DIR = "ce/eval/results"
DEFAULT_QUERY_COUNT = 10
DEFAULT_DOC_COUNT = 10
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MODEL_IDS = [
    "Qwen/Qwen2.5-14B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "prithivMLmods/Ophiuchi-Qwen3-14B-Instruct",
    "Qwen/Qwen3-14B",
]


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


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_path(value: Any) -> Path:
    p = Path(_clean_text(value)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


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


def _normalize_ws(text: Any) -> str:
    return " ".join(_clean_text(text).split())


def _short(text: Any, limit: int = 110) -> str:
    s = _normalize_ws(text)
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)] + "..."


def _model_slug(model_id: str) -> str:
    token = _clean_text(model_id).split("/")[-1].lower()
    token = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
    return token or "model"


def _score_to_band(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.40:
        return "mid"
    return "low"


def _extract_strings(values: Any) -> List[str]:
    out: List[str] = []
    if isinstance(values, list):
        for v in values:
            t = _normalize_ws(v)
            if t:
                out.append(t)
    return out


def _load_queries_from_db(db_obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(db_obj, dict):
        grants = db_obj.get("grants")
        if isinstance(grants, list):
            for g in grants:
                if not isinstance(g, dict):
                    continue
                out.extend(_extract_strings(g.get("grant_spec_keywords")))
                # fallback only if no spec-keywords
                if not g.get("grant_spec_keywords"):
                    out.extend(_extract_strings(g.get("grant_keywords")))
        if not out:
            out.extend(_extract_strings(db_obj.get("queries")))
            out.extend(_extract_strings(db_obj.get("query_texts")))
    if isinstance(db_obj, list):
        for row in db_obj:
            if isinstance(row, str):
                t = _normalize_ws(row)
                if t:
                    out.append(t)
            elif isinstance(row, dict):
                for key in ("query_text", "query", "text", "keyword", "spec_keyword"):
                    t = _normalize_ws(row.get(key))
                    if t:
                        out.append(t)
                        break
    return out


def _load_docs_from_db(db_obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(db_obj, dict):
        fac_specs = db_obj.get("fac_specs")
        if isinstance(fac_specs, list):
            for row in fac_specs:
                if not isinstance(row, dict):
                    continue
                t = _normalize_ws(row.get("text"))
                if t:
                    out.append(t)
        if not out:
            out.extend(_extract_strings(db_obj.get("docs")))
            out.extend(_extract_strings(db_obj.get("doc_texts")))
    if isinstance(db_obj, list):
        for row in db_obj:
            if isinstance(row, str):
                t = _normalize_ws(row)
                if t:
                    out.append(t)
            elif isinstance(row, dict):
                for key in ("text", "doc_text", "candidate", "fac_spec_text"):
                    t = _normalize_ws(row.get(key))
                    if t:
                        out.append(t)
                        break
    return out


def _sample_unique(items: Sequence[str], *, count: int, rng: random.Random, label: str) -> List[str]:
    uniq: List[str] = []
    seen = set()
    for x in items:
        key = _normalize_ws(x).lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        uniq.append(_normalize_ws(x))
    if len(uniq) < count:
        raise RuntimeError(f"Not enough unique {label}: need {count}, got {len(uniq)}")
    return rng.sample(uniq, count)


def _score_one_aspect(
    *,
    llm_bundle: Dict[str, Any],
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    tokenizer = llm_bundle["tokenizer"]
    prompt = _build_prompt(
        tokenizer,
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    raw = _generate_single_response(
        llm_bundle=llm_bundle,
        prompt=prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )
    parsed = _extract_json_object(raw)
    score = _coerce_score((parsed or {}).get("score"))
    band = _normalize_band((parsed or {}).get("band"))
    reason = _clean_text((parsed or {}).get("reason"))

    # Lightweight fallback when model returned non-JSON text.
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
        "raw_response": raw,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Teacher scoring evaluator across multiple local LLMs for method/domain/requirement similarity.",
    )
    p.add_argument("--db-json", type=str, default=DEFAULT_DB_JSON, help="Primary DB JSON (queries expected here).")
    p.add_argument(
        "--fac-db-json",
        type=str,
        default=DEFAULT_FAC_DB_JSON,
        help="Fallback/secondary DB JSON for docs (fac_specs text).",
    )
    p.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT, help="Random query count.")
    p.add_argument("--doc-count", type=int, default=DEFAULT_DOC_COUNT, help="Random doc count.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed.")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max generation tokens per score call.")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Generation temperature for scoring.")
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Generation top_p for scoring.")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for results.")
    p.add_argument(
        "--save-raw-responses",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include raw LLM responses in JSON output (large).",
    )
    p.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Model id override (repeatable). If omitted, uses built-in 4-model list.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    query_count = _safe_int(args.query_count, default=DEFAULT_QUERY_COUNT, minimum=1, maximum=10_000)
    doc_count = _safe_int(args.doc_count, default=DEFAULT_DOC_COUNT, minimum=1, maximum=10_000)
    seed = _safe_int(args.seed, default=DEFAULT_SEED, minimum=0, maximum=2_147_483_647)
    max_new_tokens = _safe_int(args.max_new_tokens, default=DEFAULT_MAX_NEW_TOKENS, minimum=32, maximum=8192)
    temperature = _safe_float(args.temperature, default=DEFAULT_TEMPERATURE, minimum=0.0, maximum=2.0)
    top_p = _safe_float(args.top_p, default=DEFAULT_TOP_P, minimum=0.01, maximum=1.0)
    save_raw_responses = bool(args.save_raw_responses)

    model_ids = list(args.model_id or []) or list(DEFAULT_MODEL_IDS)

    db_path = _resolve_path(args.db_json)
    fac_db_path = _resolve_path(args.fac_db_json)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise RuntimeError(f"db_json not found: {db_path}")
    if not fac_db_path.exists():
        raise RuntimeError(f"fac_db_json not found: {fac_db_path}")

    db_obj = json.loads(db_path.read_text(encoding="utf-8"))
    fac_db_obj = json.loads(fac_db_path.read_text(encoding="utf-8"))

    all_queries = _load_queries_from_db(db_obj)
    all_docs = _load_docs_from_db(db_obj)
    if not all_docs:
        all_docs = _load_docs_from_db(fac_db_obj)

    rng = random.Random(seed)
    sampled_queries = _sample_unique(all_queries, count=query_count, rng=rng, label="queries")
    sampled_docs = _sample_unique(all_docs, count=doc_count, rng=rng, label="docs")

    pairs: List[Dict[str, Any]] = []
    pair_id = 0
    for qi, query in enumerate(sampled_queries):
        for di, doc in enumerate(sampled_docs):
            pair_id += 1
            pairs.append(
                {
                    "pair_id": int(pair_id),
                    "query_idx": int(qi),
                    "doc_idx": int(di),
                    "query_text": query,
                    "doc_text": doc,
                }
            )

    aspects: List[Tuple[str, str, str]] = [
        ("method", METHOD_SYSTEM_PROMPT, METHOD_USER_PROMPT_TEMPLATE),
        ("domain", DOMAIN_SYSTEM_PROMPT, DOMAIN_USER_PROMPT_TEMPLATE),
        ("requirement", REQUIREMENT_SYSTEM_PROMPT, REQUIREMENT_USER_PROMPT_TEMPLATE),
    ]

    long_rows: List[Dict[str, Any]] = []
    wide_by_pair: Dict[int, Dict[str, Any]] = {}
    for pair in pairs:
        wide_by_pair[int(pair["pair_id"])] = dict(pair)

    print(f"db_json={db_path}")
    print(f"fac_db_json={fac_db_path}")
    print(f"query_count={len(sampled_queries)} doc_count={len(sampled_docs)} pair_count={len(pairs)}")
    print(f"models={len(model_ids)} aspects={len(aspects)}")

    for model_id in model_ids:
        print(f"model_load={model_id}")
        llm_bundle = _load_llm(model_id)
        slug = _model_slug(model_id)
        try:
            for idx, pair in enumerate(pairs, start=1):
                query = _clean_text(pair["query_text"])
                doc = _clean_text(pair["doc_text"])
                for aspect_name, system_prompt, user_template in aspects:
                    user_prompt = user_template.format(query=query, candidate=doc)
                    scored = _score_one_aspect(
                        llm_bundle=llm_bundle,
                        model_id=model_id,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    rec = {
                        "pair_id": int(pair["pair_id"]),
                        "query_idx": int(pair["query_idx"]),
                        "doc_idx": int(pair["doc_idx"]),
                        "query_text": query,
                        "doc_text": doc,
                        "model_id": model_id,
                        "model_slug": slug,
                        "aspect": aspect_name,
                        "score": float(scored["score"]),
                        "band": str(scored["band"]),
                        "reason": _clean_text(scored["reason"]),
                        "parsed_ok": bool(scored["parsed_ok"]),
                    }
                    if save_raw_responses:
                        rec["raw_response"] = _clean_text(scored.get("raw_response"))
                    long_rows.append(rec)

                    wide = wide_by_pair[int(pair["pair_id"])]
                    wide[f"{slug}_{aspect_name}_score"] = float(scored["score"])
                    wide[f"{slug}_{aspect_name}_band"] = str(scored["band"])
                    wide[f"{slug}_{aspect_name}_parsed_ok"] = int(1 if bool(scored["parsed_ok"]) else 0)
                if idx % 10 == 0:
                    print(f"model_progress model={model_id} pair={idx}/{len(pairs)}")
        finally:
            _unload_llm(llm_bundle)
            print(f"model_unload={model_id}")

    # Wide table columns
    base_cols = ["pair_id", "query_idx", "doc_idx", "query_text", "doc_text"]
    dynamic_cols: List[str] = []
    for model_id in model_ids:
        slug = _model_slug(model_id)
        for aspect_name, _, _ in aspects:
            dynamic_cols.extend(
                [
                    f"{slug}_{aspect_name}_score",
                    f"{slug}_{aspect_name}_band",
                    f"{slug}_{aspect_name}_parsed_ok",
                ]
            )
    wide_cols = base_cols + dynamic_cols
    wide_rows = [wide_by_pair[k] for k in sorted(wide_by_pair.keys())]

    # Summary stats per model/aspect
    summary_rows: List[Dict[str, Any]] = []
    for model_id in model_ids:
        slug = _model_slug(model_id)
        for aspect_name, _, _ in aspects:
            vals = [
                float(r.get("score") or 0.0)
                for r in long_rows
                if _clean_text(r.get("model_slug")) == slug and _clean_text(r.get("aspect")) == aspect_name
            ]
            parsed_ok_count = sum(
                1
                for r in long_rows
                if _clean_text(r.get("model_slug")) == slug
                and _clean_text(r.get("aspect")) == aspect_name
                and bool(r.get("parsed_ok"))
            )
            n = len(vals)
            mean = (sum(vals) / n) if n > 0 else 0.0
            summary_rows.append(
                {
                    "model_id": model_id,
                    "model_slug": slug,
                    "aspect": aspect_name,
                    "count": int(n),
                    "mean_score": float(mean),
                    "min_score": float(min(vals) if vals else 0.0),
                    "max_score": float(max(vals) if vals else 0.0),
                    "parsed_ok_count": int(parsed_ok_count),
                    "parsed_ok_rate": float(parsed_ok_count / max(1, n)),
                }
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"teacher_scoring_eval_{ts}.json"
    wide_csv_path = output_dir / f"teacher_scoring_eval_wide_{ts}.csv"
    summary_csv_path = output_dir / f"teacher_scoring_eval_summary_{ts}.csv"
    md_path = output_dir / f"teacher_scoring_eval_{ts}.md"

    payload = {
        "meta": {
            "created_at_local": datetime.now().isoformat(),
            "db_json": str(db_path),
            "fac_db_json": str(fac_db_path),
            "query_count": int(len(sampled_queries)),
            "doc_count": int(len(sampled_docs)),
            "pair_count": int(len(pairs)),
            "aspects": [x[0] for x in aspects],
            "model_ids": list(model_ids),
            "seed": int(seed),
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "save_raw_responses": bool(save_raw_responses),
            "output_dir": str(output_dir),
        },
        "sampled_queries": sampled_queries,
        "sampled_docs": sampled_docs,
        "summary_rows": summary_rows,
        "wide_rows": wide_rows,
        "long_rows": long_rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with wide_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=wide_cols)
        w.writeheader()
        for row in wide_rows:
            w.writerow({k: row.get(k, "") for k in wide_cols})

    summary_cols = [
        "model_id",
        "model_slug",
        "aspect",
        "count",
        "mean_score",
        "min_score",
        "max_score",
        "parsed_ok_count",
        "parsed_ok_rate",
    ]
    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    md_lines: List[str] = []
    md_lines.append("# Teacher Scoring Evaluation")
    md_lines.append("")
    md_lines.append(f"- Created: `{payload['meta']['created_at_local']}`")
    md_lines.append(f"- DB: `{db_path}`")
    md_lines.append(f"- Fac DB: `{fac_db_path}`")
    md_lines.append(f"- Queries: `{len(sampled_queries)}` Docs: `{len(sampled_docs)}` Pairs: `{len(pairs)}`")
    md_lines.append(f"- Models: `{len(model_ids)}` Aspects: `{len(aspects)}`")
    md_lines.append("")
    md_lines.append("## Model Summary")
    md_lines.append("")
    md_lines.append("| model | aspect | n | mean | min | max | parsed_ok |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        md_lines.append(
            f"| `{row['model_slug']}` | `{row['aspect']}` | {int(row['count'])} | "
            f"{float(row['mean_score']):.4f} | {float(row['min_score']):.4f} | {float(row['max_score']):.4f} | "
            f"{int(row['parsed_ok_count'])}/{int(row['count'])} |"
        )
    md_lines.append("")
    md_lines.append("## Pairwise Comparison (scores)")
    md_lines.append("")
    # Keep markdown table reasonably readable by only showing score columns.
    score_cols: List[str] = []
    for model_id in model_ids:
        slug = _model_slug(model_id)
        for aspect_name, _, _ in aspects:
            score_cols.append(f"{slug}_{aspect_name}_score")
    header = ["pair_id", "query", "doc"] + score_cols
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in wide_rows:
        cells = [
            str(int(row.get("pair_id") or 0)),
            _short(row.get("query_text"), 80),
            _short(row.get("doc_text"), 80),
        ]
        for c in score_cols:
            v = row.get(c)
            try:
                cells.append(f"{float(v):.4f}")
            except Exception:
                cells.append("")
        md_lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    print(f"saved_json={json_path}")
    print(f"saved_wide_csv={wide_csv_path}")
    print(f"saved_summary_csv={summary_csv_path}")
    print(f"saved_markdown={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
