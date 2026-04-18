from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------
# Edit these manually
# ------------------------------------------------------------
KEYWORD_SET_TEXT = """
- multimodal robot learning
- sim-to-real transfer
- humanoid locomotion control
- reinforcement learning for contact-rich manipulation
""".strip()

CHUNK_TEXT = """
Our lab develops model-based and learning-based control for humanoid robots,
including sim-to-real adaptation, robust gait generation, and manipulation under
contact uncertainty.
""".strip()

# Hugging Face model id (LLaMA 3.1 8B Instruct)
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = """
You are evaluating whether a candidate text chunk satisfies a specialization requirement.

This is NOT general semantic similarity.
This is REQUIREMENT MATCHING.

A high score means the candidate demonstrates the REQUIRED CAPABILITIES.

Return ONLY strict JSON:
{
  "score": <float between 0.0 and 1.0>,
  "reason": "<one short sentence>"
}

Evaluation rules:

1. Identify CORE REQUIREMENTS from the keyword set:
   - learning-based methods (reinforcement learning, multimodal learning)
   - sim-to-real transfer
   - contact-rich manipulation
   - humanoid locomotion control

2. A candidate MUST demonstrate MOST of these to score high.

3. Strong penalties:
   - If the text is robotics/control but WITHOUT learning → score <= 0.4
   - If it lacks RL or sim-to-real → score <= 0.5
   - If it is only general robotics → score <= 0.4

4. Scoring meaning:
- 0.9–1.0: directly demonstrates required methods (RL + sim-to-real + control)
- 0.7–0.89: strong but missing minor components
- 0.4–0.69: partial overlap (missing key methods)
- 0.1–0.39: domain overlap only (e.g., robotics without learning)
- 0.0–0.09: unrelated

IMPORTANT:
Do NOT reward general robotics/control unless learning-based methods are present.
""".strip()

USER_PROMPT_TEMPLATE = """
Keyword set:
{keyword_set}

Candidate chunk:
{chunk}
""".strip()

MAX_NEW_TOKENS = 80


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
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


def _select_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def main() -> int:
    device, dtype = _select_device_and_dtype()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
    )
    model.to(device)
    model.eval()

    user_prompt = USER_PROMPT_TEMPLATE.format(
        keyword_set=KEYWORD_SET_TEXT,
        chunk=CHUNK_TEXT,
    )
    prompt_text = _build_prompt(tokenizer, SYSTEM_PROMPT, user_prompt)

    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][encoded["input_ids"].shape[1] :]
    raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    parsed = _extract_json_object(raw_response) or {}
    score = _coerce_score(parsed.get("score"))
    reason = str(parsed.get("reason") or "").strip() or "No reason returned."

    print(f"model_id={MODEL_ID}")
    print(f"device={device}")
    print("")
    print("keyword_set:")
    print(KEYWORD_SET_TEXT)
    print("")
    print("chunk:")
    print(CHUNK_TEXT)
    print("")
    print(f"score={score:.4f}")
    print(f"reason={reason}")
    print("")
    print("raw_response:")
    print(raw_response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
