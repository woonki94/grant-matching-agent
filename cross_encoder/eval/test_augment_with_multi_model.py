from __future__ import annotations

import argparse
import gc
import json
import random
import re
import sys
import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "cross_encoder").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# Quick-edit defaults (you can just edit QUERY_TEXT and rerun)
# ------------------------------------------------------------
QUERY_TEXT = (
    "Reinforcement learning combined with symbolic planning and reasoning "
    "for long-horizon decision-making in dynamic, uncertain environments."
)
GRANT_DB_PATH_DEFAULT = "cross_encoder/dataset/source/grant_keywords_spec_keywords_db.json"
RANDOM_DB_QUERY_COUNT_DEFAULT = 100
RANDOM_DB_QUERY_SEED_DEFAULT = 42
SAMPLES_PER_BAND_DEFAULT = 2
DEFAULT_BATCH_QUERIES = [
    "Graph neural networks for molecular property prediction and structure-aware drug discovery pipelines.",
    "Federated learning with differential privacy for secure healthcare analytics across multi-hospital dataset silos.",
    "Computer vision and sensor fusion for autonomous driving perception in adverse weather and low-light conditions.",
    "Large language models for legal-document retrieval, citation-grounded summarization, and compliance risk analysis.",
    "Climate modeling with physics-informed machine learning for extreme weather forecasting and uncertainty quantification.",
    "Cybersecurity anomaly detection using streaming graph analytics and adversarially robust representation learning.",
    "Energy systems optimization for smart grids using stochastic control and demand-response forecasting.",
    "Educational dataset mining to personalize tutoring interventions and improve long-term student retention outcomes.",
]

TARGET_BAND = "mid"  # "high" or "mid"
MAX_ATTEMPTS_DEFAULT = 6
MAX_QUERY_TOKEN_COVERAGE_DEFAULT = 0.60
MAX_QUERY_BIGRAM_OVERLAP_DEFAULT = 0.25
MAX_QUERY_TRIGRAM_OVERLAP_DEFAULT = 0.10
MIN_NOVEL_TOKEN_RATIO_DEFAULT = 0.50
MIN_QUERY_TOKEN_COVERAGE_HIGH_DEFAULT = 0.20
MIN_QUERY_TOKEN_COVERAGE_MID_DEFAULT = 0.10
HIGH_TARGET_MIN_DEFAULT = 0.70
HIGH_TARGET_MAX_DEFAULT = 0.80
MID_TARGET_MIN_DEFAULT = 0.40
MID_TARGET_MAX_DEFAULT = 0.50

QWEN_MODEL_ID_DEFAULT = "Qwen/Qwen2.5-14B-Instruct"
DEEPSEEK_MODEL_ID_DEFAULT = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
QWEN3_MODEL_ID_DEFAULT = "prithivMLmods/Ophiuchi-Qwen3-14B-Instruct"
QWEN3_OFFICIAL_MODEL_ID_DEFAULT = "Qwen/Qwen3-14B"
MAX_NEW_TOKENS_DEFAULT = 4096
GPU_MEMORY_UTIL_DEFAULT = 0.9
MAX_MODEL_LEN_DEFAULT = 4096
TENSOR_PARALLEL_SIZE_DEFAULT = 1


QWEN_GENERATION_SYSTEM_PROMPT = """
You are augmenting training dataset for requirement matching.
Given one requirement query and a requested target profile, produce exactly one
candidate faculty specialization text.

Band rules:
- high: strong and direct fit to the core requirement concepts.
- mid: partially relevant; related domain but missing at least one core concept
  or changing one key objective/method.

Critical anti-paraphrase rules:
- Do NOT closely rephrase the query.
- Do NOT copy long spans from the query.
- Avoid reusing the same 2-3 word phrases from the query.
- Keep literal phrase reuse minimal; rely on alternate wording and structure.
- Preserve conceptual meaning using synonym/abstraction replacements rather than lexical copying.
- Keep the same intent and domain, but rewrite key concepts with alternate terminology.
- For target "mid", intentionally miss at least one core concept from the query.
- For target "mid", add one adjacent-but-not-identical technical angle.

Return strict JSON only:
{
  "augmented_text": "<D text only: concise capability phrase, 6-24 words>",
  "target_band": "<high|mid>",
  "intentionally_missing_core_concept": "<short phrase>",
  "notes": "<short phrase>"
}

Style requirement for augmented_text (D):
- Write it like dataset pair text, not profile prose.
- Prefer compact action/capability phrasing (e.g., "Standardizing ...", "Providing ...", "Optimizing ...").
- Do NOT start with "Specializes in", "Focuses on", "Expert in", or similar biography phrasing.
- Avoid trailing period when possible.
- Keep 1-3 anchor terms from Q for topicality, but rewrite most wording.

Quality examples to mimic:
- High-quality semantic rewrite:
  Q: creating standardized curriculum frameworks implementable across multiple institute locations while allowing institutional flexibility
  D: standardizing course structure and curriculum across online computer science degree programs for consistency
- Mid-quality partial match:
  Q: capability to provide comprehensive participant support including health, safety, and academic monitoring
  D: providing technical advice on safety and health issues related to physical, chemical, and biological workplace stressors

Do not output reasoning traces, <think> tags, markdown fences, or any text
outside the JSON object.
""".strip()


QWEN_GENERATION_USER_PROMPT_TEMPLATE = """
Requirement query:
{query}

Target band to generate:
{target_band}

Desired judge score range:
{target_min} to {target_max}

Preferred center:
{target_center}
""".strip()


QWEN_JUDGE_SYSTEM_PROMPT = """
You are a strict requirement-match judge.

Score how well candidate specialization matches the requirement query.
Return strict JSON only:
{
  "score": <float in [0,1]>,
  "reason": "<one short sentence>",
  "band": "<high|mid|low>"
}

Band mapping guidance:
- high: score >= 0.80
- mid: 0.45 <= score < 0.80
- low: score < 0.45

Use score/band based on core-concept coverage from the query.

Do not output reasoning traces, <think> tags, markdown fences, or any text
outside the JSON object.
""".strip()


QWEN_JUDGE_USER_PROMPT_TEMPLATE = """
Requirement query:
{query}

Candidate specialization:
{candidate}
""".strip()


QWEN3_GENERATION_SYSTEM_PROMPT = """
You are generating augmented training dataset for requirement matching.
Thinking mode is allowed, but final output must be exactly one JSON object.

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

Do not return markdown fences or text outside the JSON object.
""".strip()


QWEN3_GENERATION_USER_PROMPT_TEMPLATE = """
/think
Requirement query:
{query}

Target band:
{target_band}

Desired judge score range:
{target_min} to {target_max}

Preferred center:
{target_center}
""".strip()


QWEN3_JUDGE_SYSTEM_PROMPT = """
You are a strict requirement-match judge.
Thinking mode is allowed, but final output must be exactly one JSON object.

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


QWEN3_JUDGE_USER_PROMPT_TEMPLATE = """
/think
Requirement query:
{query}

Candidate specialization:
{candidate}
""".strip()


DEEPSEEK_GENERATION_SYSTEM_PROMPT = """
Return exactly one JSON object and nothing else.
No chain-of-thought, no explanation, no markdown.
The first non-whitespace character must be '{' and the last must be '}'.
""".strip()


DEEPSEEK_GENERATION_USER_PROMPT_TEMPLATE = """
Generate one candidate faculty specialization for requirement matching.

Rules:
- target_band=high => score target range {target_min}..{target_max}, strong fit
- target_band=mid => score target range {target_min}..{target_max}, partial fit
- avoid close paraphrase of the query
- avoid repeating the same 2-3 word phrases from the query
- preserve conceptual similarity with alternate vocabulary
- output JSON only, no prose

Required JSON schema:
{{
  "augmented_text": "D text only, concise capability phrase",
  "target_band": "high|mid",
  "intentionally_missing_core_concept": "string",
  "notes": "string"
}}

augmented_text style:
- dataset pair style, concise capability phrase, 6-24 words
- do NOT start with "Specializes in", "Focuses on", "Expert in"
- keep a small topical anchor from Q, but rewrite most terms
- high should be conceptually close with non-literal wording
- mid should be partially aligned with at least one conceptual omission

Requirement query:
{query}

Target band:
{target_band}
""".strip()


DEEPSEEK_JUDGE_SYSTEM_PROMPT = """
Return exactly one JSON object and nothing else.
No chain-of-thought, no explanation, no markdown.
The first non-whitespace character must be '{' and the last must be '}'.
""".strip()


DEEPSEEK_JUDGE_USER_PROMPT_TEMPLATE = """
Judge requirement-match quality and output JSON only.

Required JSON schema:
{{
  "score": 0.0,
  "reason": "short string",
  "band": "high|mid|low"
}}

Band mapping:
- high >= 0.70
- mid >= 0.40 and < 0.70
- low < 0.40

Requirement query:
{query}

Candidate specialization:
{candidate}
""".strip()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_band(value: Any) -> str:
    token = _clean_text(value).lower()
    if token in {"high", "mid", "low"}:
        return token
    return "low"


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


def _safe_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out < minimum:
        return float(minimum)
    if out > maximum:
        return float(maximum)
    return float(out)


def _safe_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    if out < minimum:
        return int(minimum)
    if out > maximum:
        return int(maximum)
    return int(out)


def _was_cli_arg_set(flag_name: str) -> bool:
    for raw in sys.argv[1:]:
        token = _clean_text(raw)
        if token == flag_name or token.startswith(f"{flag_name}="):
            return True
    return False


def _load_queries_from_file(path_value: str) -> list[str]:
    path = Path(_clean_text(path_value))
    if not path.is_file():
        raise RuntimeError(f"queries file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            obj = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"failed to parse JSON queries file: {path}") from exc
        if not isinstance(obj, list):
            raise RuntimeError(f"JSON queries file must be a list of strings: {path}")
        out = [_clean_text(x) for x in obj if _clean_text(x)]
        if not out:
            raise RuntimeError(f"no valid queries found in JSON file: {path}")
        return out
    out: list[str] = []
    for line in raw.splitlines():
        token = _clean_text(line)
        if not token or token.startswith("#"):
            continue
        out.append(token)
    if not out:
        raise RuntimeError(f"no valid queries found in text file: {path}")
    return out


def _resolve_path(value: Any) -> Path:
    p = Path(_clean_text(value)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _load_random_queries_from_grant_db(path_value: str, *, count: int, seed: int) -> list[str]:
    db_path = _resolve_path(path_value)
    if not db_path.is_file():
        raise RuntimeError(f"grant db file not found: {db_path}")
    try:
        payload = json.loads(db_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to parse grant db JSON: {db_path}") from exc
    grants = list((payload or {}).get("grants") or [])
    pool: list[str] = []
    for grant in grants:
        if not isinstance(grant, dict):
            continue
        for spec in list(grant.get("grant_spec_keywords") or []):
            text = _clean_text(spec)
            if text:
                pool.append(text)
    pool = _dedupe_keep_order(pool)
    if not pool:
        raise RuntimeError(f"no valid grant spec queries found in DB: {db_path}")
    take_n = int(max(1, min(int(count), len(pool))))
    rng = random.Random(int(seed))
    if take_n >= len(pool):
        return list(pool)
    return list(rng.sample(pool, take_n))


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        token = _clean_text(value)
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _band_from_score(score: float) -> str:
    s = _coerce_score(score)
    if s >= 0.70:
        return "high"
    if s >= 0.40:
        return "mid"
    return "low"


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
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None

    # 1) Direct parse.
    direct = _try_parse(raw)
    if direct is not None:
        return direct

    # 2) Strip explicit reasoning wrappers.
    stripped = re.sub(r"(?is)<think>[\s\S]*?</think>", "", raw).strip()
    if "</think>" in stripped:
        stripped = stripped.split("</think>", 1)[-1].strip()
    via_stripped = _try_parse(stripped)
    if via_stripped is not None:
        return via_stripped

    # 3) Try fenced blocks first.
    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", stripped, flags=re.IGNORECASE)
    for block in reversed(fenced_blocks):
        obj = _try_parse(block)
        if obj is not None:
            return obj

    # 4) Balanced-brace object scan; choose last parsable object.
    candidates: list[str] = []
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


def _tokenize_simple(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _clean_text(text).lower())


def _query_token_coverage(query: str, candidate: str) -> float:
    q_tokens = set(_tokenize_simple(query))
    c_tokens = set(_tokenize_simple(candidate))
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens.intersection(c_tokens))
    return float(overlap / float(max(1, len(q_tokens))))


def _query_bigram_overlap(query: str, candidate: str) -> float:
    q_toks = _tokenize_simple(query)
    c_toks = _tokenize_simple(candidate)
    if len(q_toks) < 2:
        return 0.0
    q_bi = set(zip(q_toks[:-1], q_toks[1:]))
    c_bi = set(zip(c_toks[:-1], c_toks[1:])) if len(c_toks) >= 2 else set()
    if not q_bi:
        return 0.0
    shared = len(q_bi.intersection(c_bi))
    return float(shared / float(len(q_bi)))


def _query_trigram_overlap(query: str, candidate: str) -> float:
    q_toks = _tokenize_simple(query)
    c_toks = _tokenize_simple(candidate)
    if len(q_toks) < 3:
        return 0.0
    q_tri = set(zip(q_toks[:-2], q_toks[1:-1], q_toks[2:]))
    c_tri = set(zip(c_toks[:-2], c_toks[1:-1], c_toks[2:])) if len(c_toks) >= 3 else set()
    if not q_tri:
        return 0.0
    shared = len(q_tri.intersection(c_tri))
    return float(shared / float(len(q_tri)))


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
    markers = [
        "<think>",
        "</think>",
        "```",
        "alright, i need to",
        "let me think",
        "reasoning:",
    ]
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


def _is_deepseek_r1_model(model_id: str) -> bool:
    token = _clean_text(model_id).lower()
    return "deepseek-r1" in token


def _is_qwen3_model(model_id: str) -> bool:
    token = _clean_text(model_id).lower()
    return "qwen3" in token or "ophiuchi-qwen3" in token


def _get_generation_prompts(model_id: str) -> tuple[str, str]:
    if _is_deepseek_r1_model(model_id):
        return DEEPSEEK_GENERATION_SYSTEM_PROMPT, DEEPSEEK_GENERATION_USER_PROMPT_TEMPLATE
    if _is_qwen3_model(model_id):
        return QWEN3_GENERATION_SYSTEM_PROMPT, QWEN3_GENERATION_USER_PROMPT_TEMPLATE
    return QWEN_GENERATION_SYSTEM_PROMPT, QWEN_GENERATION_USER_PROMPT_TEMPLATE


def _get_judge_prompts(model_id: str) -> tuple[str, str]:
    if _is_deepseek_r1_model(model_id):
        return DEEPSEEK_JUDGE_SYSTEM_PROMPT, DEEPSEEK_JUDGE_USER_PROMPT_TEMPLATE
    if _is_qwen3_model(model_id):
        return QWEN3_JUDGE_SYSTEM_PROMPT, QWEN3_JUDGE_USER_PROMPT_TEMPLATE
    return QWEN_JUDGE_SYSTEM_PROMPT, QWEN_JUDGE_USER_PROMPT_TEMPLATE


def _build_prompt(tokenizer: Any, *, model_id: str, system_prompt: str, user_prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not support apply_chat_template().")
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("tokenizer.chat_template is not set for this model.")
    # DeepSeek-R1 usage guidance generally prefers user-only instructions.
    if _is_deepseek_r1_model(model_id):
        merged = f"{_clean_text(system_prompt)}\n\n{_clean_text(user_prompt)}"
        messages = [{"role": "user", "content": merged}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if _is_qwen3_model(model_id):
        # Prefer enabling explicit Qwen3 thinking mode when tokenizer supports it.
        try:
            sig = inspect.signature(tokenizer.apply_chat_template)
            if "enable_thinking" in sig.parameters:
                kwargs["enable_thinking"] = True
        except Exception:
            pass
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Fallback for tokenizers without the extra kwarg.
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _load_llm(model_id: str, *, tensor_parallel_size: int, max_model_len: int, gpu_memory_utilization: float) -> Dict[str, Any]:
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
        return {
            "backend": "hf",
            "model_id": model_id,
            "tokenizer": tokenizer,
            "client": model,
        }

    llm = LLM(
        model_id,
        tensor_parallel_size=int(max(1, tensor_parallel_size)),
        max_model_len=int(max(512, max_model_len)),
        gpu_memory_utilization=float(max(0.2, min(0.98, gpu_memory_utilization))),
    )
    return {
        "backend": "vllm",
        "model_id": model_id,
        "tokenizer": llm.get_tokenizer(),
        "client": llm,
    }


def _unload_llm(llm_bundle: Dict[str, Any]) -> None:
    del llm_bundle
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


def _extract_augmented_text(parsed: Dict[str, Any]) -> str:
    for key in ("augmented_text", "d_text", "domain_text", "candidate_text", "candidate", "text", "output"):
        text = _clean_text(parsed.get(key))
        if text:
            return text
    return ""


def _model_generation_sampling(model_id: str) -> tuple[float, float]:
    if _is_qwen3_model(model_id):
        # Qwen3 thinking mode generally benefits from non-greedy decoding.
        return 0.6, 0.95
    if _is_deepseek_r1_model(model_id):
        return 0.2, 0.9
    return 0.1, 0.9


def _generate_one_with_model(
    *,
    llm_bundle: Dict[str, Any],
    target_band: str,
    target_min: float,
    target_max: float,
    query: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    tokenizer = llm_bundle["tokenizer"]
    model_id = _clean_text(llm_bundle.get("model_id"))
    system_prompt, user_template = _get_generation_prompts(model_id)
    user_prompt = user_template.format(
        query=query,
        target_band=target_band,
        target_min=f"{float(target_min):.2f}",
        target_max=f"{float(target_max):.2f}",
        target_center=f"{float((target_min + target_max) / 2.0):.2f}",
    )
    prompt = _build_prompt(
        tokenizer,
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    gen_temp, gen_top_p = _model_generation_sampling(model_id)
    raw_text = _generate_single_response(
        llm_bundle=llm_bundle,
        prompt=prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(gen_temp),
        top_p=float(gen_top_p),
    )
    if not raw_text:
        raise RuntimeError(f"No generation output from {model_id}.")
    parsed_obj = _extract_json_object(raw_text)
    parsed = dict(parsed_obj or {})
    parsed_ok = bool(parsed_obj is not None)
    augmented_text = _normalize_augmented_d_text(_extract_augmented_text(parsed))
    parsed_target = _normalize_band(parsed.get("target_band"))
    if parsed_target not in {"high", "mid"}:
        parsed_target = target_band
    return {
        "raw_response": raw_text,
        "parsed": parsed,
        "parsed_ok": bool(parsed_ok and bool(augmented_text)),
        "augmented_text": augmented_text,
        "target_band": parsed_target,
        "intentionally_missing_core_concept": _clean_text(parsed.get("intentionally_missing_core_concept")),
        "notes": _clean_text(parsed.get("notes")),
    }


def _judge_one(
    *,
    llm_bundle: Dict[str, Any],
    model_id: str,
    query: str,
    candidate_text: str,
    target_min: float,
    target_max: float,
    max_new_tokens: int,
) -> Dict[str, Any]:
    tokenizer = llm_bundle["tokenizer"]
    model_id_for_prompt = _clean_text(llm_bundle.get("model_id"))
    system_prompt, user_template = _get_judge_prompts(model_id_for_prompt)
    user_prompt = user_template.format(
        query=query,
        candidate=candidate_text,
    )
    prompt = _build_prompt(
        tokenizer,
        model_id=model_id_for_prompt,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    raw_text = _generate_single_response(
        llm_bundle=llm_bundle,
        prompt=prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=0.0,
        top_p=1.0,
    )
    if not raw_text:
        raise RuntimeError(f"No judge output from {model_id}.")
    parsed_obj = _extract_json_object(raw_text)
    parsed = dict(parsed_obj or {})
    parsed_ok = bool(parsed_obj is not None)
    score = _coerce_score(parsed.get("score"))
    model_band = _normalize_band(parsed.get("band"))
    computed_band = _band_from_score(score)
    pass_target = bool(parsed_ok and (float(target_min) <= float(score) <= float(target_max)))
    return {
        "model_id": model_id,
        "raw_response": raw_text,
        "parsed_ok": bool(parsed_ok),
        "score": float(score),
        "reason": _clean_text(parsed.get("reason")),
        "model_band": model_band,
        "computed_band": computed_band,
        "pass_target": bool(pass_target),
    }


def _run_single_model_with_loaded_bundle(
    *,
    llm_bundle: Dict[str, Any],
    model_id: str,
    query: str,
    target_band: str,
    target_min: float,
    target_max: float,
    target_center: float,
    max_attempts: int,
    min_query_token_coverage: float,
    max_query_token_coverage: float,
    max_query_bigram_overlap: float,
    max_query_trigram_overlap: float,
    min_novel_token_ratio: float,
    max_new_tokens: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    attempt_rows: list[Dict[str, Any]] = []
    selected: Optional[Dict[str, Any]] = None
    skip_counts = {
        "generation_not_json": 0,
        "empty_augmented_text": 0,
        "reasoning_spill_in_augmented_text": 0,
        "judge_not_json": 0,
        "too_low_token_coverage": 0,
        "too_high_bigram_overlap": 0,
        "too_high_trigram_overlap": 0,
        "too_low_novel_token_ratio": 0,
    }
    for attempt in range(1, int(max_attempts) + 1):
        if verbose:
            print(f"[{model_id}] step=augment_attempt_{attempt}")
        gen = _generate_one_with_model(
            llm_bundle=llm_bundle,
            target_band=target_band,
            target_min=float(target_min),
            target_max=float(target_max),
            query=query,
            max_new_tokens=int(max_new_tokens),
        )
        if not bool(gen.get("parsed_ok")):
            skip_counts["generation_not_json"] += 1
            if verbose:
                print(f"[{model_id}] skip_attempt_{attempt}=generation_not_json")
            continue
        candidate_text = _clean_text(gen.get("augmented_text"))
        if not candidate_text:
            skip_counts["empty_augmented_text"] += 1
            continue
        if _looks_like_reasoning_spill(candidate_text):
            skip_counts["reasoning_spill_in_augmented_text"] += 1
            if verbose:
                print(f"[{model_id}] skip_attempt_{attempt}=reasoning_spill_in_augmented_text")
            continue
        coverage = _query_token_coverage(query=query, candidate=candidate_text)
        if float(coverage) < float(min_query_token_coverage):
            skip_counts["too_low_token_coverage"] += 1
            if verbose:
                print(
                    f"[{model_id}] skip_attempt_{attempt}=too_low_token_coverage "
                    f"value={coverage:.4f} threshold={min_query_token_coverage:.4f}"
                )
            continue
        bigram_overlap = _query_bigram_overlap(query=query, candidate=candidate_text)
        if float(bigram_overlap) > float(max_query_bigram_overlap):
            skip_counts["too_high_bigram_overlap"] += 1
            if verbose:
                print(
                    f"[{model_id}] skip_attempt_{attempt}=too_high_bigram_overlap "
                    f"value={bigram_overlap:.4f} threshold={max_query_bigram_overlap:.4f}"
                )
            continue
        trigram_overlap = _query_trigram_overlap(query=query, candidate=candidate_text)
        if float(trigram_overlap) > float(max_query_trigram_overlap):
            skip_counts["too_high_trigram_overlap"] += 1
            if verbose:
                print(
                    f"[{model_id}] skip_attempt_{attempt}=too_high_trigram_overlap "
                    f"value={trigram_overlap:.4f} threshold={max_query_trigram_overlap:.4f}"
                )
            continue
        novel_ratio = _novel_token_ratio(query=query, candidate=candidate_text)
        if float(novel_ratio) < float(min_novel_token_ratio):
            skip_counts["too_low_novel_token_ratio"] += 1
            if verbose:
                print(
                    f"[{model_id}] skip_attempt_{attempt}=too_low_novel_token_ratio "
                    f"value={novel_ratio:.4f} threshold={min_novel_token_ratio:.4f}"
                )
            continue

        if verbose:
            print(f"[{model_id}] step=judge_attempt_{attempt}")
        judge_out = _judge_one(
            llm_bundle=llm_bundle,
            model_id=model_id,
            query=query,
            candidate_text=candidate_text,
            target_min=float(target_min),
            target_max=float(target_max),
            max_new_tokens=int(max_new_tokens),
        )
        if not bool(judge_out.get("parsed_ok")):
            skip_counts["judge_not_json"] += 1
            if verbose:
                print(f"[{model_id}] skip_attempt_{attempt}=judge_not_json")
            continue
        score = float(judge_out["score"])
        coverage = _query_token_coverage(query=query, candidate=candidate_text)
        bigram_overlap = _query_bigram_overlap(query=query, candidate=candidate_text)
        trigram_overlap = _query_trigram_overlap(query=query, candidate=candidate_text)
        novel_ratio = _novel_token_ratio(query=query, candidate=candidate_text)
        score_ok = float(target_min) <= float(score) <= float(target_max)
        min_coverage_ok = float(coverage) >= float(min_query_token_coverage)
        max_coverage_ok = float(coverage) <= float(max_query_token_coverage)
        coverage_ok = bool(min_coverage_ok and max_coverage_ok)
        bigram_ok = float(bigram_overlap) <= float(max_query_bigram_overlap)
        trigram_ok = float(trigram_overlap) <= float(max_query_trigram_overlap)
        novel_ok = float(novel_ratio) >= float(min_novel_token_ratio)
        row = {
            "attempt": int(attempt),
            "augmentation": gen,
            "judge": judge_out,
            "query_token_coverage": float(coverage),
            "query_bigram_overlap": float(bigram_overlap),
            "query_trigram_overlap": float(trigram_overlap),
            "novel_token_ratio": float(novel_ratio),
            "score_ok": bool(score_ok),
            "min_coverage_ok": bool(min_coverage_ok),
            "max_coverage_ok": bool(max_coverage_ok),
            "coverage_ok": bool(coverage_ok),
            "bigram_ok": bool(bigram_ok),
            "trigram_ok": bool(trigram_ok),
            "novel_ok": bool(novel_ok),
            "accepted": bool(score_ok and coverage_ok and bigram_ok and trigram_ok and novel_ok),
        }
        attempt_rows.append(row)
        if row["accepted"]:
            selected = row
            break

    if selected is None and attempt_rows:
        selected = min(
            attempt_rows,
            key=lambda x: (
                abs(float(x["judge"]["score"]) - float(target_center)),
                float(x.get("query_trigram_overlap") or 0.0),
                float(x.get("query_bigram_overlap") or 0.0),
                -float(x.get("novel_token_ratio") or 0.0),
                float(x["query_token_coverage"]),
            ),
        )
    if selected is None:
        return {
            "model_id": model_id,
            "attempt_count": int(len(attempt_rows)),
            "attempts": attempt_rows,
            "selected_attempt": 0,
            "selected_candidate_text": "",
            "selected_score": None,
            "selected_pass_target": False,
            "selected_query_token_coverage": None,
            "selected_query_bigram_overlap": None,
            "selected_query_trigram_overlap": None,
            "selected_novel_token_ratio": None,
            "selected_augmentation": {},
            "selected_judge": {},
            "recommend_keep": False,
            "failure_reason": "No usable candidate produced after filtering.",
            "skip_counts": skip_counts,
        }

    chosen_aug = dict(selected["augmentation"])
    chosen_judge = dict(selected["judge"])
    chosen_text = _clean_text(chosen_aug.get("augmented_text"))

    return {
        "model_id": model_id,
        "attempt_count": int(len(attempt_rows)),
        "attempts": attempt_rows,
        "selected_attempt": int(selected["attempt"]),
        "selected_candidate_text": chosen_text,
        "selected_score": float(chosen_judge["score"]),
        "selected_pass_target": bool(selected["accepted"]),
        "selected_query_token_coverage": float(selected["query_token_coverage"]),
        "selected_query_bigram_overlap": float(selected.get("query_bigram_overlap") or 0.0),
        "selected_query_trigram_overlap": float(selected.get("query_trigram_overlap") or 0.0),
        "selected_novel_token_ratio": float(selected.get("novel_token_ratio") or 0.0),
        "selected_augmentation": chosen_aug,
        "selected_judge": chosen_judge,
        "recommend_keep": bool(selected["accepted"]),
        "failure_reason": "",
        "skip_counts": skip_counts,
    }


def _run_single_model(
    *,
    model_id: str,
    query: str,
    target_band: str,
    target_min: float,
    target_max: float,
    target_center: float,
    max_attempts: int,
    min_query_token_coverage: float,
    max_query_token_coverage: float,
    max_query_bigram_overlap: float,
    max_query_trigram_overlap: float,
    min_novel_token_ratio: float,
    max_new_tokens: int,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    verbose: bool = False,
) -> Dict[str, Any]:
    if verbose:
        print(f"[{model_id}] step=load_model")
    llm_bundle = _load_llm(
        model_id,
        tensor_parallel_size=int(tensor_parallel_size),
        max_model_len=int(max_model_len),
        gpu_memory_utilization=float(gpu_memory_utilization),
    )
    try:
        return _run_single_model_with_loaded_bundle(
            llm_bundle=llm_bundle,
            model_id=model_id,
            query=query,
            target_band=target_band,
            target_min=target_min,
            target_max=target_max,
            target_center=target_center,
            max_attempts=max_attempts,
            min_query_token_coverage=min_query_token_coverage,
            max_query_token_coverage=max_query_token_coverage,
            max_query_bigram_overlap=max_query_bigram_overlap,
            max_query_trigram_overlap=max_query_trigram_overlap,
            min_novel_token_ratio=min_novel_token_ratio,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
        )
    finally:
        if verbose:
            print(f"[{model_id}] step=unload_model")
        _unload_llm(llm_bundle)


def _run_model_across_queries(
    *,
    model_id: str,
    queries: list[str],
    target_band: str,
    target_min: float,
    target_max: float,
    target_center: float,
    max_attempts: int,
    min_query_token_coverage: float,
    max_query_token_coverage: float,
    max_query_bigram_overlap: float,
    max_query_trigram_overlap: float,
    min_novel_token_ratio: float,
    max_new_tokens: int,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    verbose: bool = False,
) -> list[Dict[str, Any]]:
    if verbose:
        print(f"[{model_id}] step=load_model")
    llm_bundle = _load_llm(
        model_id,
        tensor_parallel_size=int(tensor_parallel_size),
        max_model_len=int(max_model_len),
        gpu_memory_utilization=float(gpu_memory_utilization),
    )
    results: list[Dict[str, Any]] = []
    try:
        for query_index, query in enumerate(queries, start=1):
            if verbose and len(queries) > 1:
                print(f"[{model_id}] step=query_{query_index}/{len(queries)}")
            result = _run_single_model_with_loaded_bundle(
                llm_bundle=llm_bundle,
                model_id=model_id,
                query=query,
                target_band=target_band,
                target_min=target_min,
                target_max=target_max,
                target_center=target_center,
                max_attempts=max_attempts,
                min_query_token_coverage=min_query_token_coverage,
                max_query_token_coverage=max_query_token_coverage,
                max_query_bigram_overlap=max_query_bigram_overlap,
                max_query_trigram_overlap=max_query_trigram_overlap,
                min_novel_token_ratio=min_novel_token_ratio,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
            results.append(result)
    finally:
        if verbose:
            print(f"[{model_id}] step=unload_model")
        _unload_llm(llm_bundle)
    return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Multi-model augmentation comparison on random grant DB queries. "
            "Loads one model, runs all queries, unloads it, then moves to the next model."
        )
    )
    p.add_argument(
        "--grant-db-path",
        type=str,
        default=GRANT_DB_PATH_DEFAULT,
        help="Grant DB JSON path containing grant_spec_keywords.",
    )
    p.add_argument(
        "--random-db-query-count",
        type=int,
        default=RANDOM_DB_QUERY_COUNT_DEFAULT,
        help="Number of random queries to sample from grant DB.",
    )
    p.add_argument(
        "--random-db-query-seed",
        type=int,
        default=RANDOM_DB_QUERY_SEED_DEFAULT,
        help="Random seed for grant DB query sampling.",
    )
    p.add_argument(
        "--samples-per-band",
        type=int,
        default=SAMPLES_PER_BAND_DEFAULT,
        help="How many augmentation samples to generate per target band per query.",
    )
    p.add_argument("--query", type=str, default=QUERY_TEXT, help="Requirement query text.")
    p.add_argument(
        "--batch-default-examples",
        action="store_true",
        help="Run built-in multi-domain example queries in batch mode.",
    )
    p.add_argument(
        "--queries-file",
        type=str,
        default="",
        help="Optional query file for batch mode (.txt one-per-line or .json list of strings).",
    )
    p.add_argument(
        "--max-batch-queries",
        type=int,
        default=0,
        help="Cap number of batch queries (0 means no cap).",
    )
    p.add_argument("--target-band", type=str, default=TARGET_BAND, choices=["high", "mid"])
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT, help="Max augmentation retries.")
    p.add_argument(
        "--min-query-token-coverage",
        type=float,
        default=-1.0,
        help=(
            "Minimum query token overlap floor to keep semantic topicality. "
            "Use -1 for auto (high=0.20, mid=0.10)."
        ),
    )
    p.add_argument(
        "--max-query-token-coverage",
        type=float,
        default=MAX_QUERY_TOKEN_COVERAGE_DEFAULT,
        help="Reject candidate when too many query tokens are reused (anti-paraphrase).",
    )
    p.add_argument(
        "--max-query-bigram-overlap",
        type=float,
        default=MAX_QUERY_BIGRAM_OVERLAP_DEFAULT,
        help="Reject candidate when too many query bigrams are reused.",
    )
    p.add_argument(
        "--max-query-trigram-overlap",
        type=float,
        default=MAX_QUERY_TRIGRAM_OVERLAP_DEFAULT,
        help="Reject candidate when too many query trigrams are reused.",
    )
    p.add_argument(
        "--min-novel-token-ratio",
        type=float,
        default=MIN_NOVEL_TOKEN_RATIO_DEFAULT,
        help="Reject candidate when too few candidate tokens are novel vs query tokens.",
    )
    p.add_argument("--qwen-model-id", type=str, default=QWEN_MODEL_ID_DEFAULT)
    p.add_argument("--deepseek-model-id", type=str, default=DEEPSEEK_MODEL_ID_DEFAULT)
    p.add_argument("--qwen3-model-id", type=str, default=QWEN3_MODEL_ID_DEFAULT)
    p.add_argument("--qwen3-official-model-id", type=str, default=QWEN3_OFFICIAL_MODEL_ID_DEFAULT)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT)
    p.add_argument("--tensor-parallel-size", type=int, default=TENSOR_PARALLEL_SIZE_DEFAULT)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN_DEFAULT)
    p.add_argument("--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTIL_DEFAULT)
    p.add_argument("--verbose", action="store_true", help="Print detailed progress and full JSON payload.")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    max_attempts = _safe_int(args.max_attempts, default=MAX_ATTEMPTS_DEFAULT, minimum=1, maximum=50)
    shared_min_query_token_coverage = _safe_float(
        args.min_query_token_coverage,
        default=MIN_QUERY_TOKEN_COVERAGE_MID_DEFAULT,
        minimum=-1.0,
        maximum=1.0,
    )
    max_query_token_coverage = _safe_float(
        args.max_query_token_coverage,
        default=MAX_QUERY_TOKEN_COVERAGE_DEFAULT,
        minimum=0.05,
        maximum=1.0,
    )
    max_query_bigram_overlap = _safe_float(
        args.max_query_bigram_overlap,
        default=MAX_QUERY_BIGRAM_OVERLAP_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    max_query_trigram_overlap = _safe_float(
        args.max_query_trigram_overlap,
        default=MAX_QUERY_TRIGRAM_OVERLAP_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    min_novel_token_ratio = _safe_float(
        args.min_novel_token_ratio,
        default=MIN_NOVEL_TOKEN_RATIO_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    random_db_query_count = _safe_int(
        args.random_db_query_count,
        default=RANDOM_DB_QUERY_COUNT_DEFAULT,
        minimum=1,
        maximum=5000,
    )
    random_db_query_seed = _safe_int(
        args.random_db_query_seed,
        default=RANDOM_DB_QUERY_SEED_DEFAULT,
        minimum=0,
        maximum=2_147_483_647,
    )
    samples_per_band = _safe_int(
        args.samples_per_band,
        default=SAMPLES_PER_BAND_DEFAULT,
        minimum=1,
        maximum=20,
    )

    batch_queries = _load_random_queries_from_grant_db(
        _clean_text(args.grant_db_path) or GRANT_DB_PATH_DEFAULT,
        count=int(random_db_query_count),
        seed=int(random_db_query_seed),
    )

    qwen_model_id = _clean_text(args.qwen_model_id) or QWEN_MODEL_ID_DEFAULT
    deepseek_model_id = _clean_text(args.deepseek_model_id) or DEEPSEEK_MODEL_ID_DEFAULT
    qwen3_model_id = _clean_text(args.qwen3_model_id) or QWEN3_MODEL_ID_DEFAULT
    qwen3_official_model_id = _clean_text(args.qwen3_official_model_id) or QWEN3_OFFICIAL_MODEL_ID_DEFAULT

    target_band_ranges: Dict[str, tuple[float, float]] = {
        "high": (float(HIGH_TARGET_MIN_DEFAULT), float(HIGH_TARGET_MAX_DEFAULT)),
        "mid": (float(MID_TARGET_MIN_DEFAULT), float(MID_TARGET_MAX_DEFAULT)),
    }
    target_band_order = ["high", "mid"]

    verbose = bool(args.verbose)
    if verbose:
        print(f"query_count={len(batch_queries)}")
        print(f"query_source=grant_db_random")
        print(f"grant_db_path={_clean_text(args.grant_db_path) or GRANT_DB_PATH_DEFAULT}")
        print(f"random_db_query_count={random_db_query_count}")
        print(f"random_db_query_seed={random_db_query_seed}")
        print(f"samples_per_band={samples_per_band}")
        print(
            "target_ranges="
            f"high:{HIGH_TARGET_MIN_DEFAULT:.2f}..{HIGH_TARGET_MAX_DEFAULT:.2f},"
            f"mid:{MID_TARGET_MIN_DEFAULT:.2f}..{MID_TARGET_MAX_DEFAULT:.2f}"
        )
        print(f"max_attempts={max_attempts}")
        if float(shared_min_query_token_coverage) < 0.0:
            print(
                "min_query_token_coverage=auto("
                f"high:{MIN_QUERY_TOKEN_COVERAGE_HIGH_DEFAULT:.2f},"
                f"mid:{MIN_QUERY_TOKEN_COVERAGE_MID_DEFAULT:.2f})"
            )
        else:
            print(f"min_query_token_coverage={shared_min_query_token_coverage:.3f}")
        print(f"max_query_token_coverage={max_query_token_coverage:.3f}")
        print(f"max_query_bigram_overlap={max_query_bigram_overlap:.3f}")
        print(f"max_query_trigram_overlap={max_query_trigram_overlap:.3f}")
        print(f"min_novel_token_ratio={min_novel_token_ratio:.3f}")
        print(f"qwen_model_id={qwen_model_id}")
        print(f"deepseek_model_id={deepseek_model_id}")
        print(f"qwen3_model_id={qwen3_model_id}")
        print(f"qwen3_official_model_id={qwen3_official_model_id}")
    model_batches: Dict[str, list[Dict[str, Any]]] = {}
    model_run_specs = [
        ("qwen", qwen_model_id),
        ("deepseek", deepseek_model_id),
        ("qwen3", qwen3_model_id),
        ("qwen3_official", qwen3_official_model_id),
    ]
    query_reports: list[Dict[str, Any]] = [
        {"query_index": int(i + 1), "query": q, "models": {}}
        for i, q in enumerate(batch_queries)
    ]

    for model_key, model_id in model_run_specs:
        if verbose:
            print(f"[{model_key}] step=load_model")
        llm_bundle = _load_llm(
            model_id,
            tensor_parallel_size=int(args.tensor_parallel_size),
            max_model_len=int(args.max_model_len),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
        )
        try:
            for query_index, query in enumerate(batch_queries, start=1):
                if verbose:
                    print(f"[{model_key}] step=query_{query_index}/{len(batch_queries)}")
                per_band: Dict[str, list[Dict[str, Any]]] = {}
                for band in target_band_order:
                    target_min, target_max = target_band_ranges[band]
                    target_center = float((target_min + target_max) / 2.0)
                    if float(shared_min_query_token_coverage) < 0.0:
                        min_query_token_coverage = (
                            float(MIN_QUERY_TOKEN_COVERAGE_HIGH_DEFAULT)
                            if band == "high"
                            else float(MIN_QUERY_TOKEN_COVERAGE_MID_DEFAULT)
                        )
                    else:
                        min_query_token_coverage = float(shared_min_query_token_coverage)

                    runs: list[Dict[str, Any]] = []
                    for sample_index in range(1, int(samples_per_band) + 1):
                        if verbose:
                            print(f"[{model_key}] step={band}_sample_{sample_index}")
                        out = _run_single_model_with_loaded_bundle(
                            llm_bundle=llm_bundle,
                            model_id=model_id,
                            query=query,
                            target_band=band,
                            target_min=float(target_min),
                            target_max=float(target_max),
                            target_center=float(target_center),
                            max_attempts=int(max_attempts),
                            min_query_token_coverage=float(min_query_token_coverage),
                            max_query_token_coverage=float(max_query_token_coverage),
                            max_query_bigram_overlap=float(max_query_bigram_overlap),
                            max_query_trigram_overlap=float(max_query_trigram_overlap),
                            min_novel_token_ratio=float(min_novel_token_ratio),
                            max_new_tokens=int(args.max_new_tokens),
                            verbose=verbose,
                        )
                        out["target_band"] = band
                        out["sample_index"] = int(sample_index)
                        runs.append(out)
                    per_band[band] = runs
                query_reports[query_index - 1]["models"][model_id] = per_band
        finally:
            if verbose:
                print(f"[{model_key}] step=unload_model")
            _unload_llm(llm_bundle)

    for query_item in query_reports:
        query = _clean_text(query_item.get("query"))
        print(f"query: {query}")
        for _, model_id in model_run_specs:
            per_band = dict((query_item.get("models") or {}).get(model_id) or {})
            cells: list[str] = []
            for band in target_band_order:
                for sample_index in range(1, int(samples_per_band) + 1):
                    runs = list(per_band.get(band) or [])
                    row = runs[sample_index - 1] if sample_index - 1 < len(runs) else {}
                    score_raw = row.get("selected_score")
                    score_text = f"{float(score_raw):.4f}" if score_raw is not None else "n/a"
                    d_text = _clean_text(row.get("selected_candidate_text")) or "[no valid candidate]"
                    cells.append(f"{band}[{sample_index}] score={score_text} text={d_text}")
            print(f"{model_id} | " + " | ".join(cells))
        print("")

    if verbose:
        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "query_count": int(len(batch_queries)),
            "samples_per_band": int(samples_per_band),
            "target_bands": list(target_band_order),
            "models": [model_id for _, model_id in model_run_specs],
        }
        print("batch_summary=")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
