from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


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
    model_slug as _model_slug,
    normalize_band as _normalize_band,
    normalize_ws as _normalize_ws,
    score_to_band as _score_to_band,
    short_text as _short,
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

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
TOP_P = 1.0
BATCH_SIZE = 10

# One query + 10 docs (replace these with your own later).

EVAL_QUERY = "demonstrated expertise in open geospatial data standards, metadata documentation, and biological field survey data management for environmental research programs"

EVAL_DOCS = [

    # HIGH (expected score: 0.92 - 0.98)
    # Strong overlap in metadata standards, ecological surveys, interoperability, and geospatial biodiversity systems.
    "Development of standardized metadata frameworks for ecological field surveys and interoperable geospatial biodiversity databases",

    # HIGH (expected score: 0.90 - 0.96)
    # Strong match on FAIR/open geospatial practices, metadata documentation, and habitat monitoring workflows.
    "Implementation of FAIR geospatial data practices and metadata documentation pipelines for multi-agency habitat monitoring programs",

    # HIGH (expected score: 0.88 - 0.95)
    # Directly relevant to biological records, spatial schemas, and environmental data exchange standards.
    "Management of biological observation records using standardized spatial schemas and environmental data exchange protocols",

    # MID (expected score: 0.60 - 0.72)
    # Relevant geospatial workflow terminology, but lacks biological survey and metadata documentation emphasis.
    "Geospatial data quality control workflows for remote sensing and land-use classification systems",

    # MID (expected score: 0.55 - 0.68)
    # Ecological database relevance is present, but standards and metadata concepts are mostly absent.
    "Design of biodiversity monitoring databases for long-term ecological restoration projects",

    # MID / HARD NEGATIVE (expected score: 0.45 - 0.60)
    # Shares ontology/annotation/repository semantics with metadata management, but focuses on genomics instead of geospatial field surveys.
    "Ontology-backed annotation methods for environmental genomics and species occurrence repositories",

    # MID (expected score: 0.50 - 0.65)
    # Strong interoperability and environmental spatial data themes, but weaker alignment with biological surveys and metadata documentation.
    "Spatial interoperability techniques for integrating hydrology, forestry, and climate datasets across research institutions",

    # LOW / HARD NEGATIVE (expected score: 0.20 - 0.38)
    # Contains species/ecology language that may confuse embedding models, but focuses on ML vision pipelines rather than standards or metadata management.
    "Machine learning pipelines for automated species recognition in drone imagery",

    # LOW / HARD NEGATIVE (expected score: 0.18 - 0.35)
    # Heavy geospatial terminology overlap, but actually about storage infrastructure and distributed systems instead of environmental data stewardship.
    "Cloud-native architectures for large-scale geospatial raster storage and distributed query optimization",

    # LOW (expected score: 0.05 - 0.18)
    # Generic data management concepts exist, but domain mismatch makes this largely irrelevant.
    "Adaptive clinical data management systems for longitudinal public health studies",
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


def _print_aspect_table(
    *,
    aspect: str,
    docs: Sequence[str],
    model_ids: Sequence[str],
    scored: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> None:
    slugs = [_model_slug(m) for m in model_ids]
    print("")
    print(f"=== {aspect.upper()} ===")
    print("doc_idx | doc_text | " + " | ".join([f"{slug}:score/band" for slug in slugs]))
    print("-" * 140)
    for di, doc in enumerate(docs):
        cells = [str(di), _short(doc, 70)]
        for model_id in model_ids:
            r = scored[(model_id, aspect, di)]
            cells.append(f"{float(r['score']):.3f}/{r['band']}")
        print(" | ".join(cells))


def _print_model_summary(
    *,
    model_ids: Sequence[str],
    aspects: Sequence[str],
    docs: Sequence[str],
    scored: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> None:
    print("")
    print("=== SUMMARY (mean score by model/aspect) ===")
    print("model | " + " | ".join([f"{a}_mean" for a in aspects]) + " | parsed_ok")
    print("-" * 100)
    for model_id in model_ids:
        means: List[str] = []
        ok = 0
        total = 0
        for aspect in aspects:
            vals = []
            for di in range(len(docs)):
                row = scored[(model_id, aspect, di)]
                vals.append(float(row["score"]))
                total += 1
                ok += int(1 if bool(row["parsed_ok"]) else 0)
            means.append(f"{(sum(vals)/max(1, len(vals))):.3f}")
        print(f"{_model_slug(model_id)} | " + " | ".join(means) + f" | {ok}/{total}")


def main() -> int:
    query = _normalize_ws(EVAL_QUERY)
    docs = [_normalize_ws(d) for d in EVAL_DOCS if _normalize_ws(d)]
    if not query:
        raise RuntimeError("EVAL_QUERY is empty.")
    if len(docs) != 10:
        raise RuntimeError(f"EVAL_DOCS must contain exactly 10 non-empty docs. Got {len(docs)}")

    model_ids = list(MODEL_IDS)
    batch_size = max(1, int(BATCH_SIZE))

    aspects: List[Tuple[str, str, str]] = [
        ("method", METHOD_SYSTEM_PROMPT, METHOD_USER_PROMPT_TEMPLATE),
        ("domain", DOMAIN_SYSTEM_PROMPT, DOMAIN_USER_PROMPT_TEMPLATE),
        ("requirement", REQUIREMENT_SYSTEM_PROMPT, REQUIREMENT_USER_PROMPT_TEMPLATE),
    ]

    print("teacher_scoring_eval_multi_model.py")
    print(f"query={query}")
    print(f"doc_count={len(docs)} model_count={len(model_ids)} aspect_count={len(aspects)}")
    print(f"batch_size={batch_size} max_new_tokens={MAX_NEW_TOKENS} temperature={TEMPERATURE} top_p={TOP_P}")

    scored: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    for model_id in model_ids:
        print("")
        print(f"model_load={model_id}")
        llm_bundle = _load_llm(model_id)
        try:
            tokenizer = llm_bundle["tokenizer"]
            for aspect_name, system_prompt, user_template in aspects:
                prompts: List[str] = []
                for doc in docs:
                    user_prompt = user_template.format(query=query, candidate=doc)
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
                        di = done + local_i
                        scored[(model_id, aspect_name, di)] = _parse_score_response(raw_text)
                    done += len(raw_batch)
                    print(
                        f"model_progress model={model_id} aspect={aspect_name} doc={done}/{len(docs)}"
                    )
        finally:
            _unload_llm(llm_bundle)
            print(f"model_unload={model_id}")

    aspect_names = [a[0] for a in aspects]
    for aspect_name in aspect_names:
        _print_aspect_table(
            aspect=aspect_name,
            docs=docs,
            model_ids=model_ids,
            scored=scored,
        )
    _print_model_summary(
        model_ids=model_ids,
        aspects=aspect_names,
        docs=docs,
        scored=scored,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
