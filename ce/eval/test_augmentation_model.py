from __future__ import annotations

import argparse
import gc
import inspect
import json
import random
import re
import sys
import time
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


# ======================================================
# Local runtime constants (independent from llm_distillation.py)
# ======================================================
GRANT_DB_DEFAULT = "ce/dataset/source/grant_keywords_spec_keywords_db.json"
MODEL_ID_DEFAULT = "Qwen/Qwen2.5-14B-Instruct"
AUGMENT_BATCH_SIZE_DEFAULT = 256
AUGMENT_MAX_ATTEMPTS_DEFAULT = 1
AUGMENT_MAX_NEW_TOKENS_DEFAULT = 512
AUGMENT_MAX_TRIES_PER_MISSING_DEFAULT = 1
AUGMENT_VALIDATION_MAX_NEW_TOKENS_DEFAULT = 512

# ======================================================
# Eval constants (no CLI args)
# ======================================================
GRANT_DB_PATH = GRANT_DB_DEFAULT
MODEL_ID = "Qwen/Qwen3-14B"
RANDOM_SEED = 42
NEED_HIGH = 100
NEED_MID = 100
ENABLE_VALIDATION = False
JUDGE_ASPECTS: Tuple[str, str] = ("domain", "method")

# ======================================================
# Local prompts + scoring ranges (self-contained)
# ======================================================
AIM_SCORE_RANGES: Dict[str, Tuple[float, float]] = {
    "high": (0.70, 0.80),
    "mid": (0.40, 0.50),
    "low": (0.00, 0.39),
}

VALID_SCORE_RANGES: Dict[str, Tuple[float, float]] = {
    "high": (0.70, 1.00),
    "mid": (0.30, 0.69),
    "low": (0.00, 0.39),
}

DOMAIN_AUGMENT_SYSTEM_PROMPT = """
You are generating augmented training data for DOMAIN similarity.
Final output must be exactly one JSON object and nothing else.

Rules:
- Focus on domain/topic/problem-space overlap (WHAT area it is about).
- Do not optimize for method overlap.
- Avoid direct copy from the query.
- Preserve core meaning with alternate wording.
- For target "mid", keep domain adjacent but not fully equivalent.

Required JSON schema:
{
  "augmented_text": "<D text only: concise capability phrase, 8-26 words>",
  "target_band": "<high|mid|low>",
  "notes": "<short phrase>"
}

Output rules:
- No markdown.
- No reasoning.
- Output JSON only.
""".strip()

METHOD_AUGMENT_SYSTEM_PROMPT = """
You are generating augmented training data for METHOD similarity.
Final output must be exactly one JSON object and nothing else.

Rules:
- Focus on methods/techniques/workflows overlap (HOW it is done).
- Do not optimize for domain overlap alone.
- Avoid direct copy from the query.
- Preserve method intent with alternate wording.
- For target "mid", keep partial method overlap with at least one missing component.

Required JSON schema:
{
  "augmented_text": "<D text only: concise capability phrase, 8-26 words>",
  "target_band": "<high|mid|low>",
  "notes": "<short phrase>"
}

Output rules:
- No markdown.
- No reasoning.
- Output JSON only.
""".strip()

AUGMENT_USER_PROMPT_TEMPLATE = """
Requirement query:
{query}

Judge aspect:
{aspect_label}

Target band:
{target_band}

Desired judge score range:
{target_min} to {target_max}

Preferred center:
{target_center}
""".strip()

DOMAIN_VALIDATION_SYSTEM_PROMPT = """
You are a strict DOMAIN similarity judge.
Return exactly one JSON object:
{"score": <float in [0,1]>, "reason": "<short sentence>", "band": "<high|mid|low>"}
Judge domain/topic overlap only.
""".strip()

METHOD_VALIDATION_SYSTEM_PROMPT = """
You are a strict METHOD similarity judge.
Return exactly one JSON object:
{"score": <float in [0,1]>, "reason": "<short sentence>", "band": "<high|mid|low>"}
Judge methods/techniques/workflow overlap only.
""".strip()

VALIDATION_USER_PROMPT_TEMPLATE = """
Requirement query:
{query}

Candidate specialization:
{candidate}
""".strip()

AUGMENT_PROMPT_CONFIGS: Dict[str, Tuple[str, str]] = {
    "domain": (DOMAIN_AUGMENT_SYSTEM_PROMPT, "domain"),
    "method": (METHOD_AUGMENT_SYSTEM_PROMPT, "method"),
}

VALIDATION_PROMPT_CONFIGS: Dict[str, str] = {
    "domain": DOMAIN_VALIDATION_SYSTEM_PROMPT,
    "method": METHOD_VALIDATION_SYSTEM_PROMPT,
}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_path(value: Any) -> Path:
    p = Path(_clean_text(value)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _load_vllm_bundle(
    *,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[Any, Any, Any]:
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise RuntimeError("vLLM is required but not installed in this environment.") from exc

    llm = LLM(
        _clean_text(model_id),
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        max_tokens=int(max(1, max_new_tokens)),
        temperature=float(max(0.0, temperature)),
    )
    return llm, tokenizer, sampling_params


def _release_vllm_bundle(llm: Any) -> None:
    if llm is None:
        return
    for method_name in ("shutdown", "close"):
        fn = getattr(llm, method_name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _normalize_judge_aspect(value: Any) -> str:
    token = _clean_text(value).lower()
    if token in {"method", "methods"}:
        return "method"
    return "domain"


def _normalize_target_cluster(value: Any) -> str:
    token = _clean_text(value).lower()
    if token in {"high", "top", "strong"}:
        return "high"
    if token in {"mid", "middle", "boundary"}:
        return "mid"
    if token in {"low", "weak", "rand", "random"}:
        return "low"
    return "mid"


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


def _range_for_cluster(cluster: str, table: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    key = _normalize_target_cluster(cluster)
    out = table.get(key, (0.0, 1.0))
    lo, hi = float(out[0]), float(out[1])
    if lo > hi:
        lo, hi = hi, lo
    return max(0.0, lo), min(1.0, hi)


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = _clean_text(text)
    if not raw:
        return {}

    def _try_parse(candidate: str) -> Dict[str, Any]:
        s = _clean_text(candidate)
        if not s:
            return {}
        if s.startswith("{") and s.count("{") > s.count("}"):
            s = s + ("}" * (s.count("{") - s.count("}")))
        try:
            obj = json.loads(s)
        except Exception:
            return {}
        return obj if isinstance(obj, dict) else {}

    direct = _try_parse(raw)
    if direct:
        return direct
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        extracted = _try_parse(m.group(0))
        if extracted:
            return extracted
    return {}


def _extract_score(raw_text: str) -> Tuple[float, bool]:
    obj = _extract_json_object(raw_text)
    if obj and ("score" in obj):
        return _clamp_score(obj.get("score")), True
    n = re.search(r"[-+]?\d*\.?\d+", _clean_text(raw_text))
    if n:
        return _clamp_score(n.group(0)), False
    return 0.0, False


def _extract_augmented_text(parsed: Dict[str, Any]) -> str:
    for key in ("augmented_text", "d_text", "domain_text", "candidate_text", "candidate", "text", "output"):
        value = _clean_text(parsed.get(key))
        if value:
            return value
    return ""


def _normalize_augmented_text(text: str) -> str:
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
    s = re.sub(r"\s+", " ", s).strip().rstrip(" .;")
    if not s:
        return ""
    return s[0].upper() + s[1:] if len(s) > 1 else s.upper()


def _apply_chat_template(
    *,
    tokenizer: Any,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not support apply_chat_template().")
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("tokenizer.chat_template is not set for this model/tokenizer.")

    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    token = _clean_text(model_id).lower()
    if "qwen3" in token or "ophiuchi-qwen3" in token:
        try:
            sig = inspect.signature(tokenizer.apply_chat_template)
            if "enable_thinking" in sig.parameters:
                kwargs["enable_thinking"] = True
        except Exception:
            pass
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _generate_raw_batch(
    *,
    llm: Any,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    try:
        from vllm import SamplingParams
    except Exception as exc:
        raise RuntimeError("vLLM SamplingParams import failed.") from exc
    params = SamplingParams(
        max_tokens=int(max(1, max_new_tokens)),
        temperature=float(max(0.0, temperature)),
        top_p=float(max(0.01, min(1.0, top_p))),
    )
    outputs = llm.generate(list(prompts), params, use_tqdm=False)
    out_texts: List[str] = []
    for row in outputs:
        if not row.outputs:
            out_texts.append("")
            continue
        out_texts.append(_clean_text(row.outputs[0].text))
    return out_texts


def _batched(items: List[Any], batch_size: int) -> List[List[Any]]:
    size = max(1, int(batch_size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _build_augment_prompt(
    *,
    tokenizer: Any,
    model_id: str,
    query: str,
    target_cluster: str,
    judge_aspect: str,
) -> str:
    cluster = _normalize_target_cluster(target_cluster)
    aspect = _normalize_judge_aspect(judge_aspect)
    system_prompt, aspect_label = AUGMENT_PROMPT_CONFIGS.get(aspect, AUGMENT_PROMPT_CONFIGS["domain"])
    lo, hi = _range_for_cluster(cluster, AIM_SCORE_RANGES)
    user_prompt = AUGMENT_USER_PROMPT_TEMPLATE.format(
        query=_clean_text(query),
        aspect_label=aspect_label,
        target_band=cluster,
        target_min=f"{lo:.2f}",
        target_max=f"{hi:.2f}",
        target_center=f"{((lo + hi) / 2.0):.2f}",
    )
    return _apply_chat_template(
        tokenizer=tokenizer,
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _validate_candidate(
    *,
    llm: Any,
    tokenizer: Any,
    model_id: str,
    judge_aspect: str,
    query: str,
    candidate_text: str,
    target_cluster: str,
) -> Dict[str, Any]:
    aspect = _normalize_judge_aspect(judge_aspect)
    system_prompt = VALIDATION_PROMPT_CONFIGS.get(aspect, VALIDATION_PROMPT_CONFIGS["domain"])
    user_prompt = VALIDATION_USER_PROMPT_TEMPLATE.format(
        query=_clean_text(query),
        candidate=_clean_text(candidate_text),
    )
    prompt = _apply_chat_template(
        tokenizer=tokenizer,
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    raw = _generate_raw_batch(
        llm=llm,
        prompts=[prompt],
        max_new_tokens=AUGMENT_VALIDATION_MAX_NEW_TOKENS_DEFAULT,
        temperature=0.0,
        top_p=1.0,
    )[0]
    score, parsed_ok = _extract_score(raw)
    lo, hi = _range_for_cluster(target_cluster, VALID_SCORE_RANGES)
    pass_valid = bool(lo <= float(score) <= hi)
    return {
        "score": float(score),
        "parsed_ok": bool(parsed_ok),
        "raw_response": raw,
        "pass_valid_range": pass_valid,
        "valid_min": float(lo),
        "valid_max": float(hi),
    }


def _flatten_specs(grant_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for grant in list(grant_payload.get("grants") or []):
        if not isinstance(grant, dict):
            continue
        grant_id = _clean_text(grant.get("grant_id"))
        if not grant_id:
            continue
        for spec_idx, spec_text in enumerate(list(grant.get("grant_spec_keywords") or [])):
            text_value = _clean_text(spec_text)
            if not text_value:
                continue
            out.append({"grant_id": grant_id, "spec_idx": int(spec_idx), "spec_text": text_value})
    return out


def _dedup_text_key(text: Any) -> str:
    return " ".join(_clean_text(text).lower().split())


def _augment_specs_batch(
    *,
    llm: Any,
    tokenizer: Any,
    model_id: str,
    judge_aspect: str,
    picked_specs: List[Dict[str, Any]],
    need_high: int,
    need_mid: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    results: List[Dict[str, Any]] = []
    stats = {
        "requested_high": int(max(0, int(need_high)) * len(picked_specs)),
        "requested_mid": int(max(0, int(need_mid)) * len(picked_specs)),
        "created_high": 0,
        "created_mid": 0,
        "attempts_total": 0,
        "rejected_empty": 0,
        "rejected_duplicate": 0,
        "rejected_validation": 0,
        "unfilled_high": 0,
        "unfilled_mid": 0,
    }
    if not picked_specs:
        return results, stats

    query_by_idx: Dict[int, str] = {i: _clean_text(spec.get("spec_text")) for i, spec in enumerate(picked_specs)}
    used_text_by_idx: Dict[int, set] = {i: set() for i in range(len(picked_specs))}
    accepted_by_idx: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(picked_specs))}

    max_tries = int(max(1, AUGMENT_MAX_TRIES_PER_MISSING_DEFAULT, AUGMENT_MAX_ATTEMPTS_DEFAULT))
    for cluster, need in (("high", int(max(0, int(need_high)))), ("mid", int(max(0, int(need_mid))))):
        unresolved: List[int] = []
        for i in range(len(picked_specs)):
            for _ in range(need):
                unresolved.append(int(i))

        for _try in range(max_tries):
            if not unresolved:
                break
            jobs = [{"query": query_by_idx[int(spec_i)], "target_cluster": cluster} for spec_i in unresolved]
            stats["attempts_total"] += int(len(jobs))

            raws: List[str] = []
            for jobs_chunk in _batched(jobs, AUGMENT_BATCH_SIZE_DEFAULT):
                prompts = [
                    _build_augment_prompt(
                        tokenizer=tokenizer,
                        model_id=model_id,
                        query=_clean_text(job.get("query")),
                        target_cluster=_clean_text(job.get("target_cluster")) or "mid",
                        judge_aspect=judge_aspect,
                    )
                    for job in jobs_chunk
                ]
                raws.extend(
                    _generate_raw_batch(
                        llm=llm,
                        prompts=prompts,
                        max_new_tokens=AUGMENT_MAX_NEW_TOKENS_DEFAULT,
                        temperature=0.2,
                        top_p=0.9,
                    )
                )
            
            next_unresolved: List[int] = []
            for spec_i, raw in zip(unresolved, raws):
                spec_i = int(spec_i)
                parsed = _extract_json_object(raw)
                augmented_text = _normalize_augmented_text(_extract_augmented_text(parsed))
                if not augmented_text:
                    stats["rejected_empty"] += 1
                    next_unresolved.append(spec_i)
                    continue
                key = _dedup_text_key(augmented_text)
                if not key or key in used_text_by_idx[spec_i]:
                    stats["rejected_duplicate"] += 1
                    next_unresolved.append(spec_i)
                    continue

                if bool(ENABLE_VALIDATION):
                    validation = _validate_candidate(
                        llm=llm,
                        tokenizer=tokenizer,
                        model_id=model_id,
                        judge_aspect=judge_aspect,
                        query=query_by_idx[spec_i],
                        candidate_text=augmented_text,
                        target_cluster=cluster,
                    )
                    if not bool(validation.get("pass_valid_range")):
                        stats["rejected_validation"] += 1
                        next_unresolved.append(spec_i)
                        continue
                    score = float(validation.get("score") or 0.0)
                else:
                    score = 0.0
                accepted_by_idx[spec_i].append(
                    {
                        "cluster": cluster,
                        "text": augmented_text,
                        "score": score,
                    }
                )
                used_text_by_idx[spec_i].add(key)
                if cluster == "high":
                    stats["created_high"] += 1
                else:
                    stats["created_mid"] += 1
            unresolved = next_unresolved

        if unresolved:
            if cluster == "high":
                stats["unfilled_high"] += int(len(unresolved))
            else:
                stats["unfilled_mid"] += int(len(unresolved))

    for i, spec in enumerate(picked_specs):
        results.append(
            {
                "grant_id": _clean_text(spec.get("grant_id")),
                "spec_idx": int(spec.get("spec_idx") or 0),
                "query": _clean_text(spec.get("spec_text")),
                "rows": list(accepted_by_idx.get(i) or []),
            }
        )
    return results, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch augmentation eval over random grant specs.")
    parser.add_argument(
        "--num-specs",
        type=int,
        default=1,
        help="How many random specs to fetch from grant DB and augment.",
    )
    args = parser.parse_args()
    num_specs = max(1, int(args.num_specs))

    grant_db_path = _resolve_path(GRANT_DB_PATH)
    if not grant_db_path.exists():
        raise RuntimeError(f"Grant DB not found: {grant_db_path}")

    try:
        payload = json.loads(grant_db_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse grant db: {grant_db_path} ({type(exc).__name__}: {exc})") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected top-level JSON object: {grant_db_path}")

    specs = _flatten_specs(payload)
    if not specs:
        raise RuntimeError("No specs found in grant DB.")

    rng = random.Random(int(RANDOM_SEED))
    if num_specs >= len(specs):
        picked_specs = list(specs)
    else:
        picked_specs = rng.sample(specs, k=int(num_specs))

    print(f"fetched_specs={len(picked_specs)}")
    print(f"need_high={int(NEED_HIGH)} need_mid={int(NEED_MID)}")
    print(f"judge_aspects={','.join(JUDGE_ASPECTS)}")
    print(
        "augment_runtime="
        f"validation_enabled:{str(bool(ENABLE_VALIDATION)).lower()},"
        f"batch_size:{int(AUGMENT_BATCH_SIZE_DEFAULT)},"
        f"max_attempts:{int(AUGMENT_MAX_ATTEMPTS_DEFAULT)},"
        f"max_tries_per_missing:{int(AUGMENT_MAX_TRIES_PER_MISSING_DEFAULT)},"
        f"gen_tokens:{int(AUGMENT_MAX_NEW_TOKENS_DEFAULT)},"
        f"val_tokens:{int(AUGMENT_VALIDATION_MAX_NEW_TOKENS_DEFAULT)}"
    )

    llm = None
    tokenizer = None
    results_by_aspect: Dict[str, Dict[str, Any]] = {}
    try:
        llm, tokenizer, _sampling = _load_vllm_bundle(
            model_id=_clean_text(MODEL_ID) or MODEL_ID_DEFAULT,
            max_new_tokens=AUGMENT_MAX_NEW_TOKENS_DEFAULT,
            temperature=0.2,
        )
        for judge_aspect in JUDGE_ASPECTS:
            aspect_key = _normalize_judge_aspect(judge_aspect)
            gen_started = time.perf_counter()
            per_spec_results, stats = _augment_specs_batch(
                llm=llm,
                tokenizer=tokenizer,
                model_id=_clean_text(MODEL_ID) or MODEL_ID_DEFAULT,
                judge_aspect=aspect_key,
                picked_specs=picked_specs,
                need_high=max(0, int(NEED_HIGH)),
                need_mid=max(0, int(NEED_MID)),
            )
            generation_seconds = max(1e-9, float(time.perf_counter() - gen_started))
            results_by_aspect[aspect_key] = {
                "per_spec_results": per_spec_results,
                "stats": stats,
                "generation_seconds": generation_seconds,
            }
    finally:
        _release_vllm_bundle(llm)

    for judge_aspect in JUDGE_ASPECTS:
        aspect_key = _normalize_judge_aspect(judge_aspect)
        payload = dict(results_by_aspect.get(aspect_key) or {})
        per_spec_results = list(payload.get("per_spec_results") or [])
        stats = dict(payload.get("stats") or {})
        generation_seconds = float(payload.get("generation_seconds") or 1e-9)

        print(f"aspect={aspect_key}")
        for i, item in enumerate(per_spec_results, start=1):
            grant_id = _clean_text(item.get("grant_id"))
            spec_idx = int(item.get("spec_idx") or 0)
            query = _clean_text(item.get("query"))
            rows = list(item.get("rows") or [])
            high_rows = [r for r in rows if _clean_text(r.get("cluster")) == "high"]
            mid_rows = [r for r in rows if _clean_text(r.get("cluster")) == "mid"]
            print(f"spec[{i}] grant_id={grant_id} spec_idx={spec_idx}")
            print(f"query={query}")
            for j, row in enumerate(high_rows, start=1):
                print(f"high[{j}] score={float(row.get('score') or 0.0):.4f} text={_clean_text(row.get('text'))}")
            for j, row in enumerate(mid_rows, start=1):
                print(f"mid[{j}] score={float(row.get('score') or 0.0):.4f} text={_clean_text(row.get('text'))}")
            print(
                "per_spec_summary="
                f"created_high:{len(high_rows)}/{max(0, int(NEED_HIGH))},"
                f"created_mid:{len(mid_rows)}/{max(0, int(NEED_MID))}"
            )

        print(
            "summary="
            f"created_high:{int(stats.get('created_high', 0))}/{max(0, int(NEED_HIGH)) * len(picked_specs)},"
            f"created_mid:{int(stats.get('created_mid', 0))}/{max(0, int(NEED_MID)) * len(picked_specs)},"
            f"attempts:{int(stats.get('attempts_total', 0))},"
            f"rejected_validation:{int(stats.get('rejected_validation', 0))},"
            f"rejected_duplicate:{int(stats.get('rejected_duplicate', 0))},"
            f"rejected_empty:{int(stats.get('rejected_empty', 0))},"
            f"unfilled_high:{int(stats.get('unfilled_high', 0))},"
            f"unfilled_mid:{int(stats.get('unfilled_mid', 0))}"
        )
        total_requested = int((max(0, int(NEED_HIGH)) + max(0, int(NEED_MID))) * len(picked_specs))
        print(
            "timing="
            f"generation_seconds:{generation_seconds:.4f},"
            f"requested_slots:{total_requested},"
            f"slots_per_second:{(float(total_requested) / generation_seconds):.4f},"
            f"attempts_per_second:{(float(int(stats.get('attempts_total', 0))) / generation_seconds):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
