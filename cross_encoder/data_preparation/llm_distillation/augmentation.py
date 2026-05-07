from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

QWEN_AUGMENT_SYSTEM_PROMPT = """
You are generating synthetic faculty-specialization text for requirement-matching distillation.
Return exactly one JSON object and nothing else.

Target cluster rules:
- high: strong semantic fit to the requirement, but not a close paraphrase.
- mid: partial fit with controlled overlap.
  - Keep domain/topic relevance.
  - Deliberately miss at least one CORE concept from the requirement.
  - Do NOT include all key method + objective + context concepts together.
  - Keep it plausible but incomplete.
- low: weak/adjacent fit; related domain only, with clear concept gaps.

Anti-copy rules:
- Do not copy long spans from the query.
- Avoid repeating the same 2-3 word phrases from the query.
- Use alternate wording and structure while preserving intended relevance for target cluster.

MID guardrails (important):
- Aim for partial semantic coverage (roughly 50-75% of major concepts).
- Include exactly one deliberate concept gap for MID.
- If output sounds fully complete, it is NOT MID.

Style for augmented_text:
- Concise capability phrase suitable for pair dataset.
- Avoid biography style openings like "Specializes in", "Focuses on", "Expert in".
- Prefer 6-28 words.

Return strict JSON only:
{
  "augmented_text": "<candidate specialization text>",
  "target_cluster": "<high|mid|low>",
  "intentionally_missing_core_concept": "<short phrase; required for mid, empty for high>",
  "notes": "<very short note>"
}
""".strip()

QWEN_AUGMENT_USER_PROMPT_TEMPLATE = """
Requirement query:
{query}

Target cluster:
{target_cluster}

Desired judge score range (aim):
{target_min} to {target_max}

Preferred center:
{target_center}

Cluster-specific instruction:
- If target is MID, keep clear topical relevance but intentionally leave one core concept unmet.
- If target is HIGH, cover all core concepts without copying phrasing.
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


VALIDATION_SYSTEM_PROMPT = """
You are evaluating whether a candidate specialization satisfies a requirement.

This is NOT general similarity — it is REQUIREMENT MATCHING.


Return ONLY strict JSON:
{"score": <float between 0.0 and 1.0>}


Evaluation steps (IMPORTANT — follow strictly):

1. Extract the core required concepts from the requirement text.
   - Keep them short (2–6 key phrases)
   - Do NOT invent new concepts

2. For each extracted concept:
   - Classify it as:
     - CORE (central to the requirement)
     - SUPPORTING (secondary detail)

3. For each concept:
   - Check if the candidate expresses it
   - Mark as: FULL, PARTIAL, or MISSING

4. Evaluate coverage with priority:
   - First consider CORE concepts
   - Missing a CORE concept should significantly reduce the score
   - SUPPORTING concepts influence the score only after CORE coverage is considered

5. Score based on coverage:
   - All CORE = FULL → 0.9–1.0
   - CORE mostly FULL + minor gaps → 0.75–0.9
   - Some CORE PARTIAL/MISSING → 0.5–0.75
   - Most CORE MISSING but some SUPPORTING overlap → 0.1–0.5
   - No meaningful overlap → 0.0–0.1

IMPORTANT:
- Only evaluate concepts present in the requirement
- Do NOT penalize for unrelated missing topics
- Avoid assigning identical scores when coverage differs
- Prefer slightly different scores when candidates differ in which CORE concepts they satisfy
- If a candidate covers the same broad domain but changes the main objective, method, or intended use, treat it as a partial match and cap the score at 0.65 unless most CORE concepts are still satisfied.
- A candidate that lacks one CORE concept should not receive the same score as a candidate that covers all CORE concepts partially.

Do not output explanation text.
""".strip()

VALIDATION_USER_PROMPT_TEMPLATE = """
Grant specialization keyword:
{spec_text}

Faculty specialization:
{fac_spec_text}
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
    for key in ("augmented_text", "d_text", "candidate_text", "candidate", "text", "output"):
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

    def _build_prompt(self, *, query: str, target_cluster: str) -> str:
        target_min, target_max = _range_for_cluster(target_cluster, AIM_SCORE_RANGES)
        user_prompt = QWEN_AUGMENT_USER_PROMPT_TEMPLATE.format(
            query=_clean_text(query),
            target_cluster=_normalize_target_cluster(target_cluster),
            target_min=f"{target_min:.2f}",
            target_max=f"{target_max:.2f}",
            target_center=f"{((target_min + target_max) / 2.0):.2f}",
        )
        messages = [
            {"role": "system", "content": QWEN_AUGMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

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
            spec_text=_clean_text(query),
            fac_spec_text=_clean_text(candidate_text),
        )
        messages = [
            {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

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
        score, parsed_ok = _extract_score(raw)
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

    def augment(self, *, query: str, target_cluster: str) -> Dict[str, Any]:
        query_text = _clean_text(query)
        if not query_text:
            raise ValueError("query must be non-empty.")
        cluster = _normalize_target_cluster(target_cluster)

        last_raw = ""
        last_parsed: Dict[str, Any] = {}
        best_candidate: Optional[AugmentationResult] = None
        best_distance = float("inf")

        for attempt in range(1, self.max_attempts + 1):
            prompt = self._build_prompt(query=query_text, target_cluster=cluster)
            raw = self._generate_raw_batch([prompt])[0]
            last_raw = raw

            parsed_obj = _extract_json_object(raw)
            parsed = dict(parsed_obj or {})
            last_parsed = parsed

            candidate = _normalize_augmented_text(_extract_augmented_text(parsed))
            if not candidate:
                continue
            if _looks_like_reasoning_spill(candidate):
                continue

            out_cluster = _normalize_target_cluster(parsed.get("target_cluster") or cluster)
            validation = (
                self.validate_pair(query=query_text, candidate_text=candidate, target_cluster=cluster)
                if self.enable_validation
                else {}
            )
            result = AugmentationResult(
                query=query_text,
                target_cluster=out_cluster,
                augmented_text=candidate,
                parsed_ok=bool(parsed_obj is not None),
                attempt=int(attempt),
                raw_response=raw,
                parsed=parsed,
                notes=_clean_text(parsed.get("notes")),
                validation=validation,
                failure_reason="",
            )
            if not self.enable_validation:
                return result.__dict__
            if bool(validation.get("pass_valid_range")):
                return result.__dict__

            dist = float(validation.get("distance_to_aim_range", 1.0))
            if dist < best_distance:
                best_distance = dist
                best_candidate = result

        if best_candidate is not None:
            out = best_candidate.__dict__.copy()
            out["failure_reason"] = "No candidate passed validation range; returning closest-to-aim candidate."
            return out

        failed = AugmentationResult(
            query=query_text,
            target_cluster=cluster,
            augmented_text="",
            parsed_ok=False,
            attempt=int(self.max_attempts),
            raw_response=last_raw,
            parsed=last_parsed,
            notes="",
            validation={},
            failure_reason="No usable candidate produced.",
        )
        return failed.__dict__

    def augment_batch(self, items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for row in items:
            query_text = _clean_text(row.get("query"))
            target_cluster = _clean_text(row.get("target_cluster") or row.get("target_band") or "mid")
            if not query_text:
                continue
            results.append(self.augment(query=query_text, target_cluster=target_cluster))
        return results
