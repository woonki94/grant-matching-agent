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


def score_to_band(score: float) -> str:
    s = float(score)
    if s >= 0.70:
        return "high"
    if s >= 0.30:
        return "mid"
    return "low"


def model_slug(model_id: str) -> str:
    token = clean_text(model_id).split("/")[-1].lower()
    token = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
    return token or "model"


def batched(seq: Sequence[Any], size: int):
    step = max(1, int(size))
    for i in range(0, len(seq), step):
        yield seq[i : i + step]


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = clean_text(text)
    if not raw:
        return None

    def _try(candidate: str) -> Optional[Dict[str, Any]]:
        s = clean_text(candidate)
        if not s:
            return None
        if s.startswith("{") and s.count("{") > s.count("}"):
            s += "}" * (s.count("{") - s.count("}"))
        try:
            obj = json.loads(s)
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    direct = _try(raw)
    if direct is not None:
        return direct

    stripped = re.sub(r"(?is)<think>[\s\S]*?</think>", "", raw).strip()
    if "</think>" in stripped:
        stripped = stripped.split("</think>", 1)[-1].strip()
    direct = _try(stripped)
    if direct is not None:
        return direct

    for block in reversed(re.findall(r"```(?:json)?\s*([\s\S]*?)```", stripped, flags=re.IGNORECASE)):
        obj = _try(block)
        if obj is not None:
            return obj

    spans: List[str] = []
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
                    spans.append(stripped[start : i + 1])
                    start = -1
    for span in reversed(spans):
        obj = _try(span)
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
    if "qwen3" in clean_text(model_id).lower():
        try:
            sig = inspect.signature(tokenizer.apply_chat_template)
            if "enable_thinking" in sig.parameters:
                kwargs["enable_thinking"] = False
        except Exception:
            pass
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_llm(
    model_id: str,
    *,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 1,
) -> Dict[str, Any]:
    try:
        from vllm import LLM
    except Exception as vllm_err:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as hf_err:
            raise RuntimeError(
                "Failed to initialize an LLM backend.\n"
                f"- vLLM error: {type(vllm_err).__name__}: {vllm_err}\n"
                f"- transformers error: {type(hf_err).__name__}: {hf_err}"
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

    llm_kwargs: Dict[str, Any] = {
        "tensor_parallel_size": int(tensor_parallel_size),
        "max_model_len": int(max_model_len),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "trust_remote_code": True,
        "disable_log_stats": True,
    }
    try:
        llm = LLM(model_id, **llm_kwargs)
    except TypeError:
        llm_kwargs.pop("disable_log_stats", None)
        llm = LLM(model_id, **llm_kwargs)
    return {"backend": "vllm", "model_id": model_id, "tokenizer": llm.get_tokenizer(), "client": llm}


def unload_llm(bundle: Optional[Dict[str, Any]]) -> None:
    if not isinstance(bundle, dict):
        return
    client = bundle.get("client")
    tokenizer = bundle.get("tokenizer")
    bundle.clear()
    for obj in (client, tokenizer):
        del obj
    if client is not None:
        for name in ("shutdown", "close"):
            fn = getattr(client, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def generate_responses_batch(
    *,
    llm_bundle: Dict[str, Any],
    prompts: Sequence[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    backend = clean_text(llm_bundle.get("backend"))

    def _generate_vllm(batch_prompts: Sequence[str]) -> List[str]:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=int(max_new_tokens),
            temperature=float(max(0.0, temperature)),
            top_p=float(max(0.01, min(1.0, top_p))),
        )
        generate_kwargs: Dict[str, Any] = {"use_tqdm": False}
        try:
            outputs = llm_bundle["client"].generate(list(batch_prompts), params, **generate_kwargs)
        except TypeError:
            outputs = llm_bundle["client"].generate(list(batch_prompts), params)
        texts: List[str] = []
        for out in outputs:
            texts.append(clean_text(out.outputs[0].text) if getattr(out, "outputs", None) else "")
        return texts

    def _generate_hf(batch_prompts: Sequence[str]) -> List[str]:
        import torch

        tokenizer = llm_bundle["tokenizer"]
        model = llm_bundle["client"]
        enc = tokenizer(list(batch_prompts), return_tensors="pt", padding=True, truncation=True)
        device = next(model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        do_sample = float(temperature) > 0.0
        kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
        }
        if do_sample:
            kwargs["temperature"] = float(max(0.0, temperature))
            kwargs["top_p"] = float(max(0.01, min(1.0, top_p)))
        if tokenizer.eos_token_id is not None:
            kwargs["pad_token_id"] = int(tokenizer.eos_token_id)
        with torch.no_grad():
            out_ids = model.generate(**enc, **kwargs)
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask")
        texts: List[str] = []
        for i in range(out_ids.shape[0]):
            prefix_len = int(attn[i].sum().item()) if attn is not None else int(input_ids.shape[-1])
            texts.append(clean_text(tokenizer.decode(out_ids[i][prefix_len:], skip_special_tokens=True)))
        return texts

    if backend == "vllm":
        texts = _generate_vllm(prompts)
    elif backend == "hf":
        texts = _generate_hf(prompts)
    else:
        raise RuntimeError(f"Unsupported LLM backend: {backend}")

    # Some vLLM/HF batched generations may return empty strings for a subset.
    # Retry empty outputs one-by-one to recover without increasing batch memory.
    missing = [i for i, txt in enumerate(texts) if not clean_text(txt)]
    if missing:
        retry_prompts = [prompts[i] for i in missing]
        if backend == "vllm":
            retry_texts = _generate_vllm(retry_prompts)
        else:
            retry_texts = _generate_hf(retry_prompts)
        for idx, rtxt in zip(missing, retry_texts):
            if clean_text(rtxt):
                texts[idx] = rtxt

    return texts
