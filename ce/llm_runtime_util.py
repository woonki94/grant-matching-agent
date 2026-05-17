from __future__ import annotations

import gc
import inspect
import json
import re
from typing import Any, Dict, List, Optional, Sequence


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_ws(value: Any) -> str:
    return " ".join(clean_text(value).split())


def short_text(value: Any, limit: int = 72) -> str:
    s = normalize_ws(value)
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)] + "..."


def model_slug(model_id: str) -> str:
    token = clean_text(model_id).split("/")[-1].lower()
    token = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
    return token or "model"


def score_to_band(score: float) -> str:
    if float(score) >= 0.70:
        return "high"
    if float(score) >= 0.40:
        return "mid"
    return "low"


def coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        score = 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def normalize_band(value: Any) -> str:
    token = clean_text(value).lower()
    if token in {"high", "mid", "low"}:
        return token
    return "low"


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = clean_text(text)
    if not raw:
        return None

    def _try_parse(candidate: str) -> Optional[Dict[str, Any]]:
        s = clean_text(candidate)
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


def build_prompt(tokenizer: Any, *, model_id: str, system_prompt: str, user_prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not support apply_chat_template().")
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("tokenizer.chat_template is not set for this model.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in clean_text(model_id).lower() or "ophiuchi-qwen3" in clean_text(model_id).lower():
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


def load_llm(model_id: str) -> Dict[str, Any]:
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
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )
    return {"backend": "vllm", "model_id": model_id, "tokenizer": llm.get_tokenizer(), "client": llm}


def unload_llm(llm_bundle: Optional[Dict[str, Any]]) -> None:
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


def batched(seq: Sequence[Any], size: int) -> Sequence[Sequence[Any]]:
    step = max(1, int(size))
    for i in range(0, len(seq), step):
        yield seq[i : i + step]


def generate_responses_batch(
    *,
    llm_bundle: Dict[str, Any],
    prompts: Sequence[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    backend = clean_text(llm_bundle.get("backend"))
    if backend == "vllm":
        from vllm import SamplingParams

        llm = llm_bundle["client"]
        params = SamplingParams(
            max_tokens=int(max_new_tokens),
            temperature=float(max(0.0, temperature)),
            top_p=float(max(0.01, min(1.0, top_p))),
        )
        outputs = llm.generate(list(prompts), params)
        texts: List[str] = []
        for out in outputs:
            if not out.outputs:
                texts.append("")
            else:
                texts.append(clean_text(out.outputs[0].text))
        return texts

    if backend == "hf":
        import torch

        tokenizer = llm_bundle["tokenizer"]
        model = llm_bundle["client"]
        enc = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True)
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

        attn = enc.get("attention_mask")
        input_ids = enc["input_ids"]
        texts: List[str] = []
        for i in range(out_ids.shape[0]):
            if attn is not None:
                prefix_len = int(attn[i].sum().item())
            else:
                prefix_len = int(input_ids.shape[-1])
            new_ids = out_ids[i][prefix_len:]
            texts.append(clean_text(tokenizer.decode(new_ids, skip_special_tokens=True)))
        return texts

    raise RuntimeError(f"Unsupported backend: {backend}")

