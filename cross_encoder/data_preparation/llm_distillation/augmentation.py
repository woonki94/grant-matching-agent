from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

QWEN_AUGMENT_SYSTEM_PROMPT = """
You are generating augmented training dataset for requirement matching.
Final output must be exactly one JSON object and nothing else.

Rules:
- Keep semantic relevance to the query and target band.
- Avoid direct copy from the query.
- Preserve core meaning with alternate wording (concept-equivalent phrasing, not copy).
- For target "mid", intentionally miss at least one core concept and add a related but non-identical angle.

Required JSON schema:
{
  "augmented_text": "<D text only: concise capability phrase, 8-26 words>",
  "target_band": "<high|mid|low>",
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
- Low: weak/adjacent relevance with clear concept gaps.

Output rules (strict):
- Do not output reasoning, analysis, or explanations.
- Do not output markdown fences.
- Do not output <think> tags.
- Output only valid JSON object with the keys above.
""".strip()

QWEN_AUGMENT_USER_PROMPT_TEMPLATE = """
Requirement query:
{query}

Target band:
{target_band}

Desired judge score range:
{target_min} to {target_max}

Preferred center:
{target_center}
""".strip()


# Sharp generation target range.
AIM_SCORE_RANGES: Dict[str, tuple[float, float]] = {
    "high": (0.70, 0.80),
    "mid": (0.40, 0.50),
    "low": (0.00, 0.39),
}

# Loose acceptance range after validation.
VALID_SCORE_RANGES: Dict[str, tuple[float, float]] = {
    "high": (0.70, 1.00),
    "mid": (0.30, 0.69),
    "low": (0.00, 0.39),
}

# Lexical-diversity filters (same style as eval multi-model test).
MAX_QUERY_TOKEN_COVERAGE = 0.60
MAX_QUERY_BIGRAM_OVERLAP = 0.45
MAX_QUERY_TRIGRAM_OVERLAP = 0.25
MIN_NOVEL_TOKEN_RATIO = 0.50
MIN_QUERY_TOKEN_COVERAGE_HIGH = 0.20
MIN_QUERY_TOKEN_COVERAGE_MID = 0.10
MIN_QUERY_TOKEN_COVERAGE_LOW = 0.05


VALIDATION_SYSTEM_PROMPT = """
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

VALIDATION_USER_PROMPT_TEMPLATE = """
Requirement query:
{query}

Candidate specialization:
{candidate}
""".strip()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_target_cluster(value: Any) -> str:
    token = _clean_text(value).lower()
    if token in {"high", "top", "strong"}:
        return "high"
    if token in {"mid", "middle", "boundary"}:
        return "mid"
    if token in {"low", "rand", "random", "weak"}:
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


def _extract_score(raw_text: str) -> tuple[float, bool]:
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


def _range_for_cluster(cluster: str, table: Dict[str, tuple[float, float]]) -> tuple[float, float]:
    key = _normalize_target_cluster(cluster)
    out = table.get(key)
    if out is None:
        return (0.0, 1.0)
    lo, hi = float(out[0]), float(out[1])
    if lo > hi:
        lo, hi = hi, lo
    return (max(0.0, lo), min(1.0, hi))


def _score_distance_to_range(score: float, lo: float, hi: float) -> float:
    s = _clamp_score(score)
    if lo <= s <= hi:
        return 0.0
    if s < lo:
        return float(lo - s)
    return float(s - hi)


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
    return any(marker in t for marker in markers)


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
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(" .;")
    if not s:
        return ""
    return s[0].upper() + s[1:] if len(s) > 1 else s.upper()


def _model_generation_sampling(model_id: str) -> tuple[float, float]:
    token = _clean_text(model_id).lower()
    if "qwen3" in token or "ophiuchi-qwen3" in token or "deepseek-r1" in token:
        return 0.6, 0.95
    return 0.1, 0.9


@dataclass
class AugmentationResult:
    query: str
    target_cluster: str
    augmented_text: str
    parsed_ok: bool
    attempt: int
    raw_response: str
    parsed: Dict[str, Any]
    notes: str
    validation: Dict[str, Any]
    failure_reason: str


class LLMDistillationAugmenter:
    """
    Reusable augmentation helper that keeps one loaded LLM in memory.

    Pass an already-loaded LLM object (typically vLLM `LLM`) so repeated calls
    do not re-load model weights.
    """

    def __init__(
        self,
        *,
        llm: Any,
        tokenizer: Optional[Any] = None,
        model_id: str = "Qwen/Qwen2.5-14B-Instruct",
        max_attempts: int = 3,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        enable_validation: bool = True,
        validation_max_new_tokens: int = 160,
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer if tokenizer is not None else self._infer_tokenizer(llm)
        self.model_id = _clean_text(model_id) or "Qwen/Qwen2.5-14B-Instruct"
        self.max_attempts = max(1, int(max_attempts))
        self.max_new_tokens = max(32, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.top_p = max(0.01, min(1.0, float(top_p)))
        self.enable_validation = bool(enable_validation)
        self.validation_max_new_tokens = max(32, int(validation_max_new_tokens))

        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError("Tokenizer does not support apply_chat_template().")
        if not getattr(self.tokenizer, "chat_template", None):
            raise RuntimeError("tokenizer.chat_template is not set for this model/tokenizer.")

    @staticmethod
    def _infer_tokenizer(llm: Any) -> Any:
        if hasattr(llm, "get_tokenizer"):
            return llm.get_tokenizer()
        raise RuntimeError("Tokenizer was not provided and could not be inferred from llm.get_tokenizer().")

    def _apply_chat_template(self, messages: Sequence[Dict[str, str]]) -> str:
        kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        token = _clean_text(self.model_id).lower()
        if "qwen3" in token or "ophiuchi-qwen3" in token:
            try:
                sig = inspect.signature(self.tokenizer.apply_chat_template)
                if "enable_thinking" in sig.parameters:
                    kwargs["enable_thinking"] = False
            except Exception:
                pass
        try:
            return self.tokenizer.apply_chat_template(list(messages), **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(list(messages), **kwargs)

    def _build_prompt(self, *, query: str, target_cluster: str) -> str:
        target_min, target_max = _range_for_cluster(target_cluster, AIM_SCORE_RANGES)
        target_band = _normalize_target_cluster(target_cluster)
        user_prompt = QWEN_AUGMENT_USER_PROMPT_TEMPLATE.format(
            query=_clean_text(query),
            target_band=target_band,
            target_min=f"{target_min:.2f}",
            target_max=f"{target_max:.2f}",
            target_center=f"{((target_min + target_max) / 2.0):.2f}",
        )
        messages = [
            {"role": "system", "content": QWEN_AUGMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self._apply_chat_template(messages)

    def _generate_raw_batch(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> List[str]:
        try:
            from vllm import SamplingParams
        except Exception as exc:
            raise RuntimeError(
                "vLLM SamplingParams is required for this augmenter. "
                "Install/import vllm in the same environment."
            ) from exc

        params = SamplingParams(
            max_tokens=int(self.max_new_tokens if max_new_tokens is None else max_new_tokens),
            temperature=float(self.temperature if temperature is None else temperature),
            top_p=float(self.top_p if top_p is None else top_p),
        )
        outputs = self.llm.generate(list(prompts), params, use_tqdm=False)
        out_texts: List[str] = []
        for row in outputs:
            if not row.outputs:
                out_texts.append("")
                continue
            out_texts.append(_clean_text(row.outputs[0].text))
        return out_texts

    def _build_validation_prompt(self, *, query: str, candidate_text: str) -> str:
        user_prompt = VALIDATION_USER_PROMPT_TEMPLATE.format(
            query=_clean_text(query),
            candidate=_clean_text(candidate_text),
        )
        messages = [
            {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self._apply_chat_template(messages)

    def validate_pair(
        self,
        *,
        query: str,
        candidate_text: str,
        target_cluster: str = "mid",
    ) -> Dict[str, Any]:
        query_text = _clean_text(query)
        candidate = _clean_text(candidate_text)
        cluster = _normalize_target_cluster(target_cluster)
        valid_min, valid_max = _range_for_cluster(cluster, VALID_SCORE_RANGES)
        aim_min, aim_max = _range_for_cluster(cluster, AIM_SCORE_RANGES)
        if not query_text:
            raise ValueError("query must be non-empty.")
        if not candidate:
            return {
                "score": 0.0,
                "parsed_ok": False,
                "raw_response": "",
                "target_cluster": cluster,
                "valid_min": float(valid_min),
                "valid_max": float(valid_max),
                "aim_min": float(aim_min),
                "aim_max": float(aim_max),
                "pass_valid_range": False,
                "distance_to_aim_range": 1.0,
            }
        prompt = self._build_validation_prompt(query=query_text, candidate_text=candidate)
        raw = self._generate_raw_batch(
            [prompt],
            max_new_tokens=int(self.validation_max_new_tokens),
            temperature=0.0,
            top_p=1.0,
        )[0]
        parsed_obj = _extract_json_object(raw)
        score, parsed_ok = _extract_score(raw)
        if parsed_obj is not None and ("score" in parsed_obj):
            score = _clamp_score(parsed_obj.get("score"))
            parsed_ok = True
        pass_valid = bool(valid_min <= float(score) <= valid_max)
        return {
            "score": float(score),
            "parsed_ok": bool(parsed_ok),
            "raw_response": raw,
            "target_cluster": cluster,
            "valid_min": float(valid_min),
            "valid_max": float(valid_max),
            "aim_min": float(aim_min),
            "aim_max": float(aim_max),
            "pass_valid_range": bool(pass_valid),
            "distance_to_aim_range": float(_score_distance_to_range(float(score), float(aim_min), float(aim_max))),
        }

    def _validate_batch(
        self,
        *,
        items: Sequence[Tuple[str, str, str]],
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not items:
            return out

        for start in range(0, len(items), max(1, int(batch_size))):
            chunk = list(items[start : start + max(1, int(batch_size))])
            prompts = [
                self._build_validation_prompt(query=query_text, candidate_text=candidate_text)
                for query_text, candidate_text, _cluster in chunk
            ]
            raws = self._generate_raw_batch(
                prompts,
                max_new_tokens=int(self.validation_max_new_tokens),
                temperature=0.0,
                top_p=1.0,
            )
            for (query_text, candidate_text, cluster), raw in zip(chunk, raws):
                valid_min, valid_max = _range_for_cluster(cluster, VALID_SCORE_RANGES)
                aim_min, aim_max = _range_for_cluster(cluster, AIM_SCORE_RANGES)
                parsed_obj = _extract_json_object(raw)
                score, parsed_ok = _extract_score(raw)
                if parsed_obj is not None and ("score" in parsed_obj):
                    score = _clamp_score(parsed_obj.get("score"))
                    parsed_ok = True
                pass_valid = bool(valid_min <= float(score) <= valid_max)
                out.append(
                    {
                        "score": float(score),
                        "parsed_ok": bool(parsed_ok),
                        "raw_response": raw,
                        "target_cluster": cluster,
                        "valid_min": float(valid_min),
                        "valid_max": float(valid_max),
                        "aim_min": float(aim_min),
                        "aim_max": float(aim_max),
                        "pass_valid_range": bool(pass_valid),
                        "distance_to_aim_range": float(
                            _score_distance_to_range(float(score), float(aim_min), float(aim_max))
                        ),
                    }
                )
        return out

    def augment(self, *, query: str, target_cluster: str) -> Dict[str, Any]:
        query_text = _clean_text(query)
        if not query_text:
            raise ValueError("query must be non-empty.")
        cluster = _normalize_target_cluster(target_cluster)
        out = self.augment_batch(
            [{"query": query_text, "target_cluster": cluster}],
            batch_size=1,
        )
        if not out:
            return {
                "query": query_text,
                "target_cluster": cluster,
                "augmented_text": "",
                "parsed_ok": False,
                "attempt": int(self.max_attempts),
                "raw_response": "",
                "parsed": {},
                "notes": "",
                "validation": {},
                "failure_reason": "No usable candidate produced.",
            }
        return out[0]

    def augment_batch(self, items: Sequence[Dict[str, Any]], *, batch_size: int = 32) -> List[Dict[str, Any]]:
        batch_n = max(1, int(batch_size))
        results: List[Optional[Dict[str, Any]]] = [None] * len(items)
        jobs: List[Dict[str, Any]] = []
        gen_temp, gen_top_p = _model_generation_sampling(self.model_id)

        for idx, row in enumerate(items):
            query_text = _clean_text(row.get("query"))
            cluster = _normalize_target_cluster(row.get("target_cluster") or row.get("target_band") or "mid")
            if not query_text:
                results[idx] = {
                    "query": "",
                    "target_cluster": cluster,
                    "augmented_text": "",
                    "parsed_ok": False,
                    "attempt": 0,
                    "raw_response": "",
                    "parsed": {},
                    "notes": "",
                    "validation": {},
                    "failure_reason": "query must be non-empty.",
                }
                continue
            target_min, target_max = _range_for_cluster(cluster, AIM_SCORE_RANGES)
            target_center = float((target_min + target_max) / 2.0)
            min_coverage = (
                float(MIN_QUERY_TOKEN_COVERAGE_HIGH)
                if cluster == "high"
                else float(MIN_QUERY_TOKEN_COVERAGE_MID if cluster == "mid" else MIN_QUERY_TOKEN_COVERAGE_LOW)
            )
            jobs.append(
                {
                    "idx": int(idx),
                    "query": query_text,
                    "cluster": cluster,
                    "target_center": target_center,
                    "min_coverage": float(min_coverage),
                    "attempt": 0,
                    "done": False,
                    "last_raw": "",
                    "last_parsed": {},
                    "best_candidate": None,
                    "best_distance": float("inf"),
                    "best_key": (float("inf"), float("inf"), float("inf"), float("inf")),
                }
            )

        for _round in range(1, self.max_attempts + 1):
            active = [j for j in jobs if (not bool(j["done"])) and int(j["attempt"]) < self.max_attempts]
            if not active:
                break

            validate_queue: List[Dict[str, Any]] = []
            for start in range(0, len(active), batch_n):
                chunk = active[start : start + batch_n]
                prompts = [self._build_prompt(query=j["query"], target_cluster=j["cluster"]) for j in chunk]
                raws = self._generate_raw_batch(
                    prompts,
                    max_new_tokens=int(self.max_new_tokens),
                    temperature=float(gen_temp),
                    top_p=float(gen_top_p),
                )
                for j, raw in zip(chunk, raws):
                    j["attempt"] = int(j["attempt"]) + 1
                    j["last_raw"] = raw
                    parsed_obj = _extract_json_object(raw)
                    parsed = dict(parsed_obj or {})
                    j["last_parsed"] = parsed

                    candidate = _normalize_augmented_text(_extract_augmented_text(parsed))
                    if not candidate:
                        continue
                    if _looks_like_reasoning_spill(candidate):
                        continue
                    coverage = _query_token_coverage(query=j["query"], candidate=candidate)
                    if float(coverage) < float(j["min_coverage"]):
                        continue
                    if float(coverage) > float(MAX_QUERY_TOKEN_COVERAGE):
                        continue
                    bigram_overlap = _query_bigram_overlap(query=j["query"], candidate=candidate)
                    if float(bigram_overlap) > float(MAX_QUERY_BIGRAM_OVERLAP):
                        continue
                    trigram_overlap = _query_trigram_overlap(query=j["query"], candidate=candidate)
                    if float(trigram_overlap) > float(MAX_QUERY_TRIGRAM_OVERLAP):
                        continue
                    novel_ratio = _novel_token_ratio(query=j["query"], candidate=candidate)
                    if float(novel_ratio) < float(MIN_NOVEL_TOKEN_RATIO):
                        continue

                    out_cluster = _normalize_target_cluster(
                        parsed.get("target_cluster") or parsed.get("target_band") or j["cluster"]
                    )
                    if not self.enable_validation:
                        result = AugmentationResult(
                            query=j["query"],
                            target_cluster=out_cluster,
                            augmented_text=candidate,
                            parsed_ok=bool(parsed_obj is not None),
                            attempt=int(j["attempt"]),
                            raw_response=raw,
                            parsed=parsed,
                            notes=_clean_text(parsed.get("notes")),
                            validation={},
                            failure_reason="",
                        )
                        results[int(j["idx"])] = result.__dict__
                        j["done"] = True
                        continue

                    validate_queue.append(
                        {
                            "job": j,
                            "candidate": candidate,
                            "out_cluster": out_cluster,
                            "parsed_ok": bool(parsed_obj is not None),
                            "raw_response": raw,
                            "parsed": parsed,
                            "notes": _clean_text(parsed.get("notes")),
                            "bigram_overlap": float(bigram_overlap),
                            "trigram_overlap": float(trigram_overlap),
                            "novel_ratio": float(novel_ratio),
                        }
                    )

            if self.enable_validation and validate_queue:
                validate_items = [
                    (v["job"]["query"], v["candidate"], _normalize_target_cluster(v["job"]["cluster"]))
                    for v in validate_queue
                ]
                validations = self._validate_batch(items=validate_items, batch_size=batch_n)
                for item, validation in zip(validate_queue, validations):
                    j = item["job"]
                    result = AugmentationResult(
                        query=j["query"],
                        target_cluster=item["out_cluster"],
                        augmented_text=item["candidate"],
                        parsed_ok=bool(item["parsed_ok"]),
                        attempt=int(j["attempt"]),
                        raw_response=item["raw_response"],
                        parsed=item["parsed"],
                        notes=item["notes"],
                        validation=validation,
                        failure_reason="",
                    )
                    if bool(validation.get("pass_valid_range")):
                        results[int(j["idx"])] = result.__dict__
                        j["done"] = True
                        continue

                    dist = float(validation.get("distance_to_aim_range", 1.0))
                    score = _clamp_score(validation.get("score"))
                    rank_key = (
                        abs(float(score) - float(j["target_center"])),
                        float(item["trigram_overlap"]),
                        float(item["bigram_overlap"]),
                        -float(item["novel_ratio"]),
                    )
                    if (dist < float(j["best_distance"])) or (
                        dist == float(j["best_distance"]) and rank_key < tuple(j["best_key"])
                    ):
                        j["best_distance"] = float(dist)
                        j["best_key"] = rank_key
                        j["best_candidate"] = result

        for j in jobs:
            idx = int(j["idx"])
            if results[idx] is not None:
                continue
            best = j.get("best_candidate")
            if isinstance(best, AugmentationResult):
                out = best.__dict__.copy()
                out["failure_reason"] = "No candidate passed validation range; returning closest-to-aim candidate."
                results[idx] = out
                continue
            results[idx] = AugmentationResult(
                query=j["query"],
                target_cluster=j["cluster"],
                augmented_text="",
                parsed_ok=False,
                attempt=int(j["attempt"]),
                raw_response=_clean_text(j.get("last_raw")),
                parsed=dict(j.get("last_parsed") or {}),
                notes="",
                validation={},
                failure_reason="No usable candidate produced.",
            ).__dict__

        return [r if r is not None else {
            "query": "",
            "target_cluster": "mid",
            "augmented_text": "",
            "parsed_ok": False,
            "attempt": 0,
            "raw_response": "",
            "parsed": {},
            "notes": "",
            "validation": {},
            "failure_reason": "No usable candidate produced.",
        } for r in results]
