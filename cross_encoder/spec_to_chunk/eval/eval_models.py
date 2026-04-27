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
CHUNK_TEXT1 = """
We present a framework that integrates reinforcement learning with symbolic planning and reasoning to enable long-horizon decision-making in complex, dynamic environments. The system combines model-free reinforcement learning for low-level policy acquisition with a symbolic planner that encodes structured domain knowledge and supports high-level reasoning over task sequences. A key contribution is the bidirectional interaction between learned policies and symbolic representations, allowing the system to refine plans based on real-time observations and adapt strategies under uncertainty. The reasoning component supports constraint satisfaction, goal decomposition, and temporal abstraction, enabling robust behavior across extended horizons. Experiments in simulated and real robotic domains demonstrate improved performance in tasks requiring sequential decision-making, adaptability, and generalization across environments with varying dynamics. The approach shows strong results in navigation, manipulation, and multi-stage planning scenarios, highlighting the benefits of combining learning-based control with structured reasoning mechanisms.
""".strip()
# Expected score: 0.92 ~ 0.97
# Reason: covers ALL key components (RL + planning + reasoning + long-horizon + dynamic uncertainty)


# --- NEAR PARAPHRASE (VERY STRONG) ---
CHUNK_TEXT2 = """
This work explores the integration of reinforcement learning and planning techniques for sequential decision-making in dynamic environments. The proposed system leverages learned representations to guide planning across extended horizons, enabling consistent performance in complex tasks. A hybrid architecture is introduced where planning modules interact with learned policies, supporting reasoning over multi-step decision processes and adapting to environmental changes. The framework is evaluated in robotic navigation and control scenarios, demonstrating improvements in long-term performance and robustness. By combining structured decision processes with reinforcement learning, the system enhances its ability to manage uncertainty and maintain stable behavior across diverse environments. The results highlight the effectiveness of integrating planning and learning-based methods for complex sequential decision-making tasks.
""".strip()
# Expected score: 0.85 ~ 0.92
# Reason: RL + planning + long-horizon present, reasoning is implicit/weaker → slightly lower


# --- PARTIAL (MISSING SYMBOLIC REASONING) ---
CHUNK_TEXT3 = """
We investigate reinforcement learning approaches for long-horizon decision-making in dynamic environments, focusing on policy optimization and value estimation techniques. The proposed method emphasizes stability and scalability by leveraging actor-critic architectures and hierarchical policy representations. To improve performance over extended time horizons, the system incorporates temporal abstraction and structured exploration strategies. Experiments across multiple control domains show that the approach effectively manages uncertainty and adapts to changing dynamics. The framework is applied to robotic manipulation and navigation tasks, demonstrating strong performance in sequential decision problems. The approach relies on learned representations and optimization techniques to guide decision-making across complex environments.
""".strip()
# Expected score: 0.70 ~ 0.80
# Reason: RL + long-horizon present, but no planning/reasoning structure → classic "partial match"


# --- PARTIAL (PLANNING + REASONING, NO RL) ---
CHUNK_TEXT4 = """
This paper presents a symbolic planning and reasoning framework for long-horizon decision-making in dynamic environments. The system encodes domain knowledge using structured representations and applies search-based planning algorithms to generate action sequences that satisfy complex constraints. A reasoning module enables inference over environmental states, supporting adaptability and robustness in uncertain conditions. The framework is evaluated in robotic and logistics scenarios, demonstrating strong performance in tasks requiring coordination and multi-step reasoning. By leveraging structured representations and logical inference, the system achieves consistent results across extended horizons and varying environmental conditions.
""".strip()
# Expected score: 0.65 ~ 0.75
# Reason: planning + reasoning strong, but missing RL → still relevant but incomplete


# --- MISLEADING OVERLAP (OPTIMIZATION RL) ---
CHUNK_TEXT5 = """
We propose a reinforcement learning framework for optimizing decision-making policies in dynamic environments with complex cost structures. The approach focuses on efficient policy learning through reward shaping and cost-aware optimization techniques, enabling improved performance in resource-constrained scenarios. By incorporating environment feedback and adaptive learning strategies, the system refines policies to achieve stable outcomes over time. The method is evaluated in control and scheduling domains, demonstrating strong results in minimizing cost and improving operational efficiency. The framework emphasizes scalability and robustness across large state spaces and varying environmental conditions.
""".strip()
# Expected score: 0.60 ~ 0.70
# Reason: RL + dynamic env present, BUT objective is optimization (not planning/reasoning) → deceptive overlap


# --- WEAK (SHORT-HORIZON RL) ---
CHUNK_TEXT6 = """
This study explores reinforcement learning techniques for reactive control in static and moderately dynamic environments. The approach focuses on learning policies that respond efficiently to immediate environmental changes, emphasizing fast adaptation and low computational overhead. Using model-free reinforcement learning methods, the system achieves strong performance in tasks requiring rapid response and stability. Experiments in control benchmarks demonstrate the effectiveness of the approach in short-horizon scenarios. The framework is designed for real-time applications where quick decision-making is essential.
""".strip()
# Expected score: 0.45 ~ 0.60
# Reason: RL present but NO long-horizon / planning / reasoning → weak alignment


# --- IRRELEVANT ---
CHUNK_TEXT7 = """
This paper investigates statistical forecasting techniques for supply chain optimization, focusing on inventory management and demand prediction. The proposed models leverage time-series analysis and probabilistic methods to improve forecasting accuracy and reduce operational costs. By incorporating historical data and external factors, the system enhances decision-making in logistics and distribution networks. Experiments demonstrate improved performance compared to baseline forecasting approaches. The framework is applied to real-world datasets, showing its effectiveness in managing uncertainty and variability in supply chain operations.
""".strip()
# Expected score: 0.05 ~ 0.20
# Reason: completely different domain


CHUNK_TEXTS = [CHUNK_TEXT1, CHUNK_TEXT2, CHUNK_TEXT3, CHUNK_TEXT4, CHUNK_TEXT5, CHUNK_TEXT6, CHUNK_TEXT7]  # Add more chunk strings for batched scoring.

# Hugging Face model id (Qwen2.5 32B Instruct)
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

SYSTEM_PROMPT = """
You are evaluating whether a candidate text chunk satisfies a specialization requirement.

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

Candidate chunk:
{chunk}
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
    for chunk_text in CHUNK_TEXTS:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            keyword_set=KEYWORD_SET_TEXT,
            chunk=chunk_text,
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

    for idx, (chunk_text, out) in enumerate(zip(CHUNK_TEXTS, outputs), start=1):
        if not out.outputs:
            raw_response = ""
        else:
            raw_response = str(out.outputs[0].text or "").strip()
        parsed = _extract_json_object(raw_response) or {}
        score = _coerce_score(parsed.get("score"))
        reason = str(parsed.get("reason") or "").strip() or "No reason returned."

        print("")
        print(f"[item {idx}]")
        print("chunk:")
        print(chunk_text)
        print("")
        print(f"score={score:.4f}")
        print(f"reason={reason}")
        print("")
        print("raw_response:")
        print(raw_response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
