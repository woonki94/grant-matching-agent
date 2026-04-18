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
MODEL_ID = "meta-llama/Llama-3.1-8B"

SYSTEM_PROMPT = """
You are scoring relevance between:
1) a specialization keyword set
2) a candidate text chunk

Return ONLY strict JSON:
{
  "score": <float between 0.0 and 1.0>,
  "reason": "<one short sentence>"
}

Scoring rubric:
- 0.9-1.0: direct and strong semantic match
- 0.7-0.89: clearly relevant with good overlap
- 0.4-0.69: partial/indirect relevance
- 0.1-0.39: weak relevance
- 0.0-0.09: unrelated
""".strip()

USER_PROMPT_TEMPLATE = """
Keyword set:
{keyword_set}

Candidate chunk:
{chunk}
""".strip()

MAX_NEW_TOKENS = 120


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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return (
        "System:\n"
        + system_prompt
        + "\n\nUser:\n"
        + user_prompt
        + "\n\nAssistant:\n"
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

    gen_cfg = model.generation_config
    gen_cfg.do_sample = False
    if hasattr(gen_cfg, "temperature"):
        gen_cfg.temperature = None
    if hasattr(gen_cfg, "top_p"):
        gen_cfg.top_p = None
    if tokenizer.pad_token_id is not None:
        gen_cfg.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        gen_cfg.eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            generation_config=gen_cfg,
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
