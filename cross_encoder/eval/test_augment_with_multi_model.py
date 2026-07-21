from __future__ import annotations

import gc
import inspect
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "cross_encoder").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ======================================================
# Constants (no CLI args)
# ======================================================
GRANT_DB_PATH = "cross_encoder/dataset/source/grant_keywords_spec_keywords_db.json"
RANDOM_DB_QUERY_COUNT = 10
RANDOM_DB_QUERY_SEED = 42
SAMPLES_PER_BAND = 2

MODEL_IDS = [
    "Qwen/Qwen2.5-14B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "prithivMLmods/Ophiuchi-Qwen3-14B-Instruct",
    "Qwen/Qwen3-14B",
]

TARGET_BANDS = ("high", "mid")
TARGET_SCORE_RANGES: Dict[str, tuple[float, float]] = {
    "high": (0.70, 0.80),
    "mid": (0.40, 0.50),
}

MAX_ATTEMPTS = 6
MAX_NEW_TOKENS = 4096
TENSOR_PARALLEL_SIZE = 1
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.9

MAX_QUERY_TOKEN_COVERAGE = 0.60
MAX_QUERY_BIGRAM_OVERLAP = 0.25
MAX_QUERY_TRIGRAM_OVERLAP = 0.10
MIN_NOVEL_TOKEN_RATIO = 0.50
MIN_QUERY_TOKEN_COVERAGE_HIGH = 0.20
MIN_QUERY_TOKEN_COVERAGE_MID = 0.10

VERBOSE = True


# ======================================================
# Shared prompts (DeepSeek now follows Qwen3 instruction style)
# ======================================================
GENERATION_SYSTEM_PROMPT = """
You are generating augmented training dataset for requirement matching.
Final output must be exactly one JSON object.

Rules:
- Keep semantic relevance to the query and target band.
- Avoid close paraphrase and repeated n-grams from the query.
- Preserve core meaning with alternate wording (concept-equivalent phrasing, not copy).
- For target "mid", intentionally miss at least one core concept and add a related but non-identical angle.

Required JSON schema:
{
  "augmented_text": "<D text only: concise capability phrase, 6-24 words>",
  "target_band": "<high|mid>",
  "intentionally_missing_core_concept": "<short phrase>",
  "notes": "<short phrase>"
}

Style for augmented_text:
- Dataset pair style (Q || D), compact and capability-focused.
- Prefer verb-noun phrase style; avoid biography style.
- Do NOT start with "Specializes in", "Focuses on", "Expert in", or similar.
- Keep a small topical anchor from Q while rewording the rest.

Quality target pattern:
- High: semantically very close but lexically rewritten.
- Mid: partially aligned to one/few core aspects, with one clear conceptual gap.

No markdown fences or extra text outside the JSON object.
""".strip()


GENERATION_USER_PROMPT_TEMPLATE = """
Requirement query:
{query}

Target band:
{target_band}

Desired judge score range:
{target_min} to {target_max}

Preferred center:
{target_center}
""".strip()


JUDGE_SYSTEM_PROMPT = """
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


JUDGE_USER_PROMPT_TEMPLATE = """
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


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        score = 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _normalize_band(value: Any) -> str:
    token = _clean_text(value).lower()
    if token in {"high", "mid", "low"}:
        return token
    return "low"


def _tokenize_simple(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _clean_text(text).lower())


def _query_token_coverage(query: str, candidate: str) -> float:
    q_tokens = set(_tokenize_simple(query))
    c_tokens = set(_tokenize_simple(candidate))
    if not q_tokens:
        return 0.0
    return float(len(q_tokens.intersection(c_tokens)) / float(max(1, len(q_tokens))))


def _query_bigram_overlap(query: str, candidate: str) -> float:
    q_toks = _tokenize_simple(query)
    c_toks = _tokenize_simple(candidate)
    if len(q_toks) < 2:
        return 0.0
    q_bi = set(zip(q_toks[:-1], q_toks[1:]))
    c_bi = set(zip(c_toks[:-1], c_toks[1:])) if len(c_toks) >= 2 else set()
    if not q_bi:
        return 0.0
    return float(len(q_bi.intersection(c_bi)) / float(len(q_bi)))


def _query_trigram_overlap(query: str, candidate: str) -> float:
    q_toks = _tokenize_simple(query)
    c_toks = _tokenize_simple(candidate)
    if len(q_toks) < 3:
        return 0.0
    q_tri = set(zip(q_toks[:-2], q_toks[1:-1], q_toks[2:]))
    c_tri = set(zip(c_toks[:-2], c_toks[1:-1], c_toks[2:])) if len(c_toks) >= 3 else set()
    if not q_tri:
        return 0.0
    return float(len(q_tri.intersection(c_tri)) / float(len(q_tri)))


def _novel_token_ratio(query: str, candidate: str) -> float:
    q_tokens = set(_tokenize_simple(query))
    c_tokens = set(_tokenize_simple(candidate))
    if not c_tokens:
        return 0.0
    novel = len([t for t in c_tokens if t not in q_tokens])
    return float(novel / float(len(c_tokens)))


def _looks_like_reasoning_spill(text: str) -> bool:
    t = _clean_text(text).lower()
    if not t:
        return True
    markers = ["<think>", "</think>", "```", "alright, i need to", "let me think", "reasoning:"]
    return any(m in t for m in markers)


def _normalize_augmented_d_text(text: str) -> str:
    s = _clean_text(text)
    if not s:
        return ""
    s = s.strip().strip('"').strip("'").strip()
    s = re.sub(
        r"^(specializes in|specialising in|focuses on|focused on|expert in|expertise in|works on|researches)\s+",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(" .;")
    if not s:
        return ""
    return s[0].upper() + s[1:] if len(s) > 1 else s.upper()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = _clean_text(text)
    if not raw:
        return None

    def _try_parse(candidate: str) -> Optional[Dict[str, Any]]:
        s = _clean_text(candidate)
        if not s:
            return None
        if s.startswith("{") and s.count("{") > s.count("}"):
            s = s + ("}" * (s.count("{") - s.count("}")))
        try:
            obj = json.loads(s)
        except Exception:
            return None
        if isinstance(obj, dict):
            return obj
        return None

    direct = _try_parse(raw)
    if direct is not None:
        return direct

    stripped = re.sub(r"(?is)<think>[\s\S]*?</think>", "", raw).strip()
    if "</think>" in stripped:
        stripped = stripped.split("</think>", 1)[-1].strip()
    via_stripped = _try_parse(stripped)
    if via_stripped is not None:
        return via_stripped

    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", stripped, flags=re.IGNORECASE)
    for block in reversed(fenced_blocks):
        obj = _try_parse(block)
        if obj is not None:
            return obj

    candidates: List[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(stripped):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(stripped[start : i + 1])
                    start = -1
    for cand in reversed(candidates):
        obj = _try_parse(cand)
        if obj is not None:
            return obj
    return None


def _extract_augmented_text(parsed: Dict[str, Any]) -> str:
    for key in ("augmented_text", "d_text", "domain_text", "candidate_text", "candidate", "text", "output"):
        text = _clean_text(parsed.get(key))
        if text:
            return text
    return ""


def _build_prompt(tokenizer: Any, *, model_id: str, system_prompt: str, user_prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not support apply_chat_template().")
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("tokenizer.chat_template is not set for this model.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in _clean_text(model_id).lower() or "ophiuchi-qwen3" in _clean_text(model_id).lower():
        try:
            sig = inspect.signature(tokenizer.apply_chat_template)
            if "enable_thinking" in sig.parameters:
                kwargs["enable_thinking"] = True
        except Exception:
            pass
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _load_llm(model_id: str) -> Dict[str, Any]:
    try:
        from vllm import LLM
    except Exception as vllm_err:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as hf_err:
            raise RuntimeError(
                "Failed to initialize model backend.\n"
                f"- vLLM error: {type(vllm_err).__name__}: {vllm_err}\n"
                f"- transformers error: {type(hf_err).__name__}: {hf_err}\n"
                "Install one working backend, then rerun."
            ) from hf_err

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        return {"backend": "hf", "model_id": model_id, "tokenizer": tokenizer, "client": model}

    llm = LLM(
        model_id,
        tensor_parallel_size=int(max(1, TENSOR_PARALLEL_SIZE)),
        max_model_len=int(max(512, MAX_MODEL_LEN)),
        gpu_memory_utilization=float(max(0.2, min(0.98, GPU_MEMORY_UTILIZATION))),
    )
    return {"backend": "vllm", "model_id": model_id, "tokenizer": llm.get_tokenizer(), "client": llm}


def _unload_llm(llm_bundle: Optional[Dict[str, Any]]) -> None:
    if not isinstance(llm_bundle, dict):
        return

    client = llm_bundle.get("client")
    tokenizer = llm_bundle.get("tokenizer")
    llm_bundle["client"] = None
    llm_bundle["tokenizer"] = None
    llm_bundle["backend"] = ""
    llm_bundle["model_id"] = ""

    if client is not None:
        for method_name in ("shutdown", "close"):
            fn = getattr(client, method_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    del client
    del tokenizer
    llm_bundle.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _generate_single_response(
    *,
    llm_bundle: Dict[str, Any],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    backend = _clean_text(llm_bundle.get("backend"))
    if backend == "vllm":
        from vllm import SamplingParams

        llm = llm_bundle["client"]
        params = SamplingParams(
            max_tokens=int(max_new_tokens),
            temperature=float(max(0.0, temperature)),
            top_p=float(max(0.01, min(1.0, top_p))),
        )
        outputs = llm.generate([prompt], params)
        if not outputs or not outputs[0].outputs:
            return ""
        return _clean_text(outputs[0].outputs[0].text)

    if backend == "hf":
        import torch

        tokenizer = llm_bundle["tokenizer"]
        model = llm_bundle["client"]
        enc = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        do_sample = float(temperature) > 0.0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "temperature": float(max(0.0, temperature)),
        }
        if tokenizer.eos_token_id is not None:
            gen_kwargs["pad_token_id"] = int(tokenizer.eos_token_id)
        if do_sample:
            gen_kwargs["top_p"] = float(max(0.01, min(1.0, top_p)))
        with torch.no_grad():
            out_ids = model.generate(**enc, **gen_kwargs)
        prefix_len = int(enc["input_ids"].shape[-1])
        new_ids = out_ids[0][prefix_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        return _clean_text(text)

    raise RuntimeError(f"Unsupported backend: {backend}")


def _model_generation_sampling(model_id: str) -> tuple[float, float]:
    token = _clean_text(model_id).lower()
    if "qwen3" in token or "ophiuchi-qwen3" in token or "deepseek-r1" in token:
        return 0.6, 0.95
    return 0.1, 0.9


def _generate_one_with_model(
    *,
    llm_bundle: Dict[str, Any],
    query: str,
    target_band: str,
    target_min: float,
    target_max: float,
) -> Dict[str, Any]:
    tokenizer = llm_bundle["tokenizer"]
    model_id = _clean_text(llm_bundle.get("model_id"))
    user_prompt = GENERATION_USER_PROMPT_TEMPLATE.format(
        query=query,
        target_band=target_band,
        target_min=f"{float(target_min):.2f}",
        target_max=f"{float(target_max):.2f}",
        target_center=f"{float((target_min + target_max) / 2.0):.2f}",
    )
    prompt = _build_prompt(
        tokenizer,
        model_id=model_id,
        system_prompt=GENERATION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    gen_temp, gen_top_p = _model_generation_sampling(model_id)
    raw_text = _generate_single_response(
        llm_bundle=llm_bundle,
        prompt=prompt,
        max_new_tokens=int(MAX_NEW_TOKENS),
        temperature=float(gen_temp),
        top_p=float(gen_top_p),
    )
    parsed_obj = _extract_json_object(raw_text)
    parsed = dict(parsed_obj or {})
    augmented_text = _normalize_augmented_d_text(_extract_augmented_text(parsed))
    parsed_target = _normalize_band(parsed.get("target_band"))
    if parsed_target not in {"high", "mid"}:
        parsed_target = target_band
    return {
        "raw_response": raw_text,
        "parsed": parsed,
        "parsed_ok": bool(parsed_obj is not None and bool(augmented_text)),
        "augmented_text": augmented_text,
        "target_band": parsed_target,
    }


def _judge_one(
    *,
    llm_bundle: Dict[str, Any],
    query: str,
    candidate_text: str,
    target_min: float,
    target_max: float,
) -> Dict[str, Any]:
    tokenizer = llm_bundle["tokenizer"]
    model_id = _clean_text(llm_bundle.get("model_id"))
    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(query=query, candidate=candidate_text)
    prompt = _build_prompt(
        tokenizer,
        model_id=model_id,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    raw_text = _generate_single_response(
        llm_bundle=llm_bundle,
        prompt=prompt,
        max_new_tokens=int(MAX_NEW_TOKENS),
        temperature=0.0,
        top_p=1.0,
    )
    parsed_obj = _extract_json_object(raw_text)
    parsed = dict(parsed_obj or {})
    score = _coerce_score(parsed.get("score"))
    pass_target = bool(parsed_obj is not None and float(target_min) <= float(score) <= float(target_max))
    return {
        "raw_response": raw_text,
        "parsed_ok": bool(parsed_obj is not None),
        "score": float(score),
        "band": _normalize_band(parsed.get("band")),
        "reason": _clean_text(parsed.get("reason")),
        "pass_target": pass_target,
    }


def _run_single_sample(
    *,
    llm_bundle: Dict[str, Any],
    model_id: str,
    query: str,
    target_band: str,
    target_min: float,
    target_max: float,
    target_center: float,
) -> Dict[str, Any]:
    if target_band == "high":
        min_coverage = float(MIN_QUERY_TOKEN_COVERAGE_HIGH)
    else:
        min_coverage = float(MIN_QUERY_TOKEN_COVERAGE_MID)

    attempts: List[Dict[str, Any]] = []
    chosen: Optional[Dict[str, Any]] = None

    for attempt in range(1, int(MAX_ATTEMPTS) + 1):
        gen = _generate_one_with_model(
            llm_bundle=llm_bundle,
            query=query,
            target_band=target_band,
            target_min=target_min,
            target_max=target_max,
        )
        text = _clean_text(gen.get("augmented_text"))
        if (not bool(gen.get("parsed_ok"))) or (not text) or _looks_like_reasoning_spill(text):
            continue

        coverage = _query_token_coverage(query=query, candidate=text)
        if float(coverage) < float(min_coverage):
            continue
        if float(coverage) > float(MAX_QUERY_TOKEN_COVERAGE):
            continue

        bigram_overlap = _query_bigram_overlap(query=query, candidate=text)
        if float(bigram_overlap) > float(MAX_QUERY_BIGRAM_OVERLAP):
            continue

        trigram_overlap = _query_trigram_overlap(query=query, candidate=text)
        if float(trigram_overlap) > float(MAX_QUERY_TRIGRAM_OVERLAP):
            continue

        novel_ratio = _novel_token_ratio(query=query, candidate=text)
        if float(novel_ratio) < float(MIN_NOVEL_TOKEN_RATIO):
            continue

        judge = _judge_one(
            llm_bundle=llm_bundle,
            query=query,
            candidate_text=text,
            target_min=target_min,
            target_max=target_max,
        )
        if not bool(judge.get("parsed_ok")):
            continue

        row = {
            "attempt": int(attempt),
            "selected_candidate_text": text,
            "selected_score": float(judge.get("score") or 0.0),
            "selected_pass_target": bool(judge.get("pass_target", False)),
            "selected_query_token_coverage": float(coverage),
            "selected_query_bigram_overlap": float(bigram_overlap),
            "selected_query_trigram_overlap": float(trigram_overlap),
            "selected_novel_token_ratio": float(novel_ratio),
        }
        attempts.append(row)
        if row["selected_pass_target"]:
            chosen = row
            break

    if chosen is None and attempts:
        chosen = min(
            attempts,
            key=lambda x: (
                abs(float(x["selected_score"]) - float(target_center)),
                float(x["selected_query_trigram_overlap"]),
                float(x["selected_query_bigram_overlap"]),
                -float(x["selected_novel_token_ratio"]),
            ),
        )

    if chosen is None:
        return {
            "model_id": model_id,
            "selected_candidate_text": "",
            "selected_score": None,
            "selected_pass_target": False,
            "failure_reason": "No usable candidate produced after filtering.",
        }

    return {
        "model_id": model_id,
        "selected_candidate_text": _clean_text(chosen.get("selected_candidate_text")),
        "selected_score": float(chosen.get("selected_score") or 0.0),
        "selected_pass_target": bool(chosen.get("selected_pass_target", False)),
        "failure_reason": "",
    }


def _load_random_queries_from_grant_db(path_value: str, *, count: int, seed: int) -> List[str]:
    db_path = _resolve_path(path_value)
    if not db_path.is_file():
        raise RuntimeError(f"grant db file not found: {db_path}")

    try:
        payload = json.loads(db_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to parse grant db JSON: {db_path}") from exc

    grants = list((payload or {}).get("grants") or [])
    pool: List[str] = []
    for grant in grants:
        if not isinstance(grant, dict):
            continue
        for spec in list(grant.get("grant_spec_keywords") or []):
            text = _clean_text(spec)
            if text:
                pool.append(text)
    pool = list(dict.fromkeys(pool))
    if not pool:
        raise RuntimeError(f"no valid grant spec queries found in DB: {db_path}")

    n = int(max(1, min(int(count), len(pool))))
    rng = random.Random(int(seed))
    if n >= len(pool):
        return list(pool)
    return list(rng.sample(pool, n))


def main() -> int:
    queries = _load_random_queries_from_grant_db(
        GRANT_DB_PATH,
        count=int(RANDOM_DB_QUERY_COUNT),
        seed=int(RANDOM_DB_QUERY_SEED),
    )

    if VERBOSE:
        print(f"query_count={len(queries)}")
        print(f"grant_db_path={_resolve_path(GRANT_DB_PATH)}")
        print(f"random_db_query_seed={RANDOM_DB_QUERY_SEED}")
        print(f"samples_per_band={SAMPLES_PER_BAND}")

    query_reports: List[Dict[str, Any]] = [
        {"query_index": int(i + 1), "query": q, "models": {}}
        for i, q in enumerate(queries)
    ]

    for model_id in MODEL_IDS:
        if VERBOSE:
            print(f"[{model_id}] step=load_model")
        llm_bundle = _load_llm(model_id)
        try:
            for q_idx, query in enumerate(queries, start=1):
                if VERBOSE:
                    print(f"[{model_id}] step=query_{q_idx}/{len(queries)}")
                per_band: Dict[str, List[Dict[str, Any]]] = {}
                for band in TARGET_BANDS:
                    target_min, target_max = TARGET_SCORE_RANGES[band]
                    center = float((target_min + target_max) / 2.0)
                    results: List[Dict[str, Any]] = []
                    for sample_index in range(1, int(SAMPLES_PER_BAND) + 1):
                        if VERBOSE:
                            print(f"[{model_id}] step={band}_sample_{sample_index}")
                        one = _run_single_sample(
                            llm_bundle=llm_bundle,
                            model_id=model_id,
                            query=query,
                            target_band=band,
                            target_min=float(target_min),
                            target_max=float(target_max),
                            target_center=float(center),
                        )
                        one["target_band"] = band
                        one["sample_index"] = int(sample_index)
                        results.append(one)
                    per_band[band] = results
                query_reports[q_idx - 1]["models"][model_id] = per_band
        finally:
            if VERBOSE:
                print(f"[{model_id}] step=unload_model")
            _unload_llm(llm_bundle)
            llm_bundle = None

    for item in query_reports:
        query = _clean_text(item.get("query"))
        print(f"query: {query}")
        models_map = dict(item.get("models") or {})
        for model_id in MODEL_IDS:
            by_band = dict(models_map.get(model_id) or {})
            cells: List[str] = []
            for band in TARGET_BANDS:
                rows = list(by_band.get(band) or [])
                for sample_index in range(1, int(SAMPLES_PER_BAND) + 1):
                    row = rows[sample_index - 1] if sample_index - 1 < len(rows) else {}
                    s = row.get("selected_score")
                    score_text = f"{float(s):.4f}" if s is not None else "n/a"
                    text = _clean_text(row.get("selected_candidate_text")) or "[no valid candidate]"
                    cells.append(f"{band}[{sample_index}] score={score_text} text={text}")
            print(f"{model_id} | " + " | ".join(cells))
        print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

