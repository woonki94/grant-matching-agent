from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from vllm import LLM, SamplingParams

# Ensure project root on sys.path for direct script execution.
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
# Edit these manually
# ------------------------------------------------------------
KEYWORD_SET_TEXT = """

Reinforcement learning combined with symbolic planning and reasoning for long-horizon decision-making in dynamic, uncertain real-world environments

""".strip()

# --- PERFECT MATCH ---

FAC_SPEC_TEXT1 = """

Reinforcement learning integrated with symbolic planning and reasoning for long-horizon sequential decision-making in dynamic and uncertain environments, including structured representations for goal decomposition and constraint-aware policy learning

""".strip()

# Expected score: 0.92 ~ 0.97

# Reason: exact coverage of RL + planning + reasoning + long-horizon + uncertainty

# --- VERY STRONG (near paraphrase, slight shift in wording) ---

FAC_SPEC_TEXT2 = """

Hybrid reinforcement learning and planning methods for multi-step decision-making in dynamic environments, leveraging structured reasoning and learned policies to improve long-horizon performance and adaptability under uncertainty

""".strip()

# Expected score: 0.85 ~ 0.92

# Reason: all components present, reasoning slightly implicit

# --- PARTIAL (missing explicit reasoning) ---

FAC_SPEC_TEXT3 = """

Reinforcement learning for long-horizon decision-making in dynamic environments using hierarchical policies, temporal abstraction, and adaptive control strategies for complex sequential tasks

""".strip()

# Expected score: 0.70 ~ 0.80

# Reason: RL + long-horizon present, but no planning/reasoning

# --- PARTIAL (planning + reasoning, no RL) ---

FAC_SPEC_TEXT4 = """

Symbolic planning and reasoning for long-horizon decision-making in dynamic environments using structured representations, search-based methods, and constraint-driven task decomposition

""".strip()

# Expected score: 0.65 ~ 0.75

# Reason: planning + reasoning strong, RL missing

# --- MISLEADING OVERLAP (RL but different objective) ---

FAC_SPEC_TEXT5 = """

Reinforcement learning for optimizing decision-making policies in dynamic environments with emphasis on cost efficiency, reward shaping, and scalable policy optimization across large state spaces

""".strip()

# Expected score: 0.60 ~ 0.70

# Reason: RL + dynamic env present, but no planning/reasoning focus

# --- WEAK (short-horizon RL) ---

FAC_SPEC_TEXT6 = """

Reinforcement learning for reactive control and short-horizon decision-making in dynamic systems with focus on fast adaptation and stability in real-time environments

""".strip()

# Expected score: 0.45 ~ 0.60

# Reason: lacks long-horizon, planning, reasoning

# --- IRRELEVANT ---

FAC_SPEC_TEXT7 = """

Statistical forecasting and optimization for supply chain systems including demand prediction, inventory control, and logistics planning using probabilistic models and operations research techniques

""".strip()

# Expected score: 0.05 ~ 0.20

# Reason: completely different domain

FAC_SPEC_TEXTS = [

    FAC_SPEC_TEXT1,

    FAC_SPEC_TEXT2,

    FAC_SPEC_TEXT3,

    FAC_SPEC_TEXT4,

    FAC_SPEC_TEXT5,

    FAC_SPEC_TEXT6,

    FAC_SPEC_TEXT7,

] # Add more faculty specialization strings for batched scoring.

# Hugging Face model id (Qwen2.5 32B Instruct)
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

SYSTEM_PROMPT = """
You are evaluating whether a candidate faculty specialization text satisfies a specialization requirement.

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
Keyword set:
{keyword_set}

Candidate faculty specialization:
{fac_spec}
""".strip()

MAX_NEW_TOKENS = 80
TEMPERATURE = 0.0  # deterministic scoring for teacher labels
STOP_STRINGS = ["}\n", "}"]  # stop early after JSON object
TENSOR_PARALLEL_SIZE = 1  # set 2/4/8 on multi-GPU nodes


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None

    if raw.startswith("{") and raw.count("{") > raw.count("}"):
        raw = raw + ("}" * (raw.count("{") - raw.count("}")))

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


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


def _build_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            "Tokenizer does not support apply_chat_template(). "
            "Use an instruct/chat tokenizer for this model."
        )
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            "tokenizer.chat_template is not set. "
            "This script requires a tokenizer with a valid chat template."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> int:
    llm = LLM(
        MODEL_ID,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    )
    tokenizer = llm.get_tokenizer()

    prompts = []
    for fac_spec_text in FAC_SPEC_TEXTS:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            keyword_set=KEYWORD_SET_TEXT,
            fac_spec=fac_spec_text,
        )
        prompts.append(_build_prompt(tokenizer, SYSTEM_PROMPT, user_prompt))

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        stop=STOP_STRINGS,
    )
    outputs = llm.generate(prompts, sampling_params)
    if not outputs:
        raise RuntimeError("vLLM returned no outputs.")

    print(f"model_id={MODEL_ID}")
    print("backend=vllm")
    print(f"tensor_parallel_size={TENSOR_PARALLEL_SIZE}")
    print(f"batch_size={len(prompts)}")
    print("")
    print("keyword_set:")
    print(KEYWORD_SET_TEXT)

    for idx, (fac_spec_text, out) in enumerate(zip(FAC_SPEC_TEXTS, outputs), start=1):
        if not out.outputs:
            raw_response = ""
        else:
            raw_response = str(out.outputs[0].text or "").strip()
        parsed = _extract_json_object(raw_response) or {}
        score = _coerce_score(parsed.get("score"))
        reason = str(parsed.get("reason") or "").strip() or "No reason returned."

        print("")
        print(f"[item {idx}]")
        print("faculty_specialization:")
        print(fac_spec_text)
        print("")
        print(f"score={score:.4f}")
        print(f"reason={reason}")
        print("")
        print("raw_response:")
        print(raw_response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
