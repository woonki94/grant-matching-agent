from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_limit(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _resolve_model_dir(model_dir: str) -> Path:
    def _has_model_artifacts(p: Path) -> bool:
        if not p.exists() or not p.is_dir():
            return False
        has_config = (p / "config.json").exists()
        has_model = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
        has_tokenizer = (p / "tokenizer.json").exists() or (p / "tokenizer_config.json").exists()
        return bool(has_config and has_model and has_tokenizer)

    def _pick_best_checkpoint_dir(p: Path) -> Optional[Path]:
        cands = [x for x in p.glob("checkpoint-*") if x.is_dir()]
        if not cands:
            return None

        def _score(x: Path) -> Tuple[int, float]:
            name = x.name
            step = -1
            if name.startswith("checkpoint-"):
                try:
                    step = int(name.split("-", 1)[1])
                except Exception:
                    step = -1
            return (step, x.stat().st_mtime)

        cands.sort(key=_score, reverse=True)
        for c in cands:
            if _has_model_artifacts(c):
                return c.resolve()
        return None

    if _clean_text(model_dir):
        p = Path(_clean_text(model_dir)).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Model directory not found: {p}")
        if _has_model_artifacts(p):
            return p
        ckpt = _pick_best_checkpoint_dir(p)
        if ckpt is not None:
            return ckpt
        raise FileNotFoundError(
            f"No loadable model artifacts found in {p}. "
            "Expected config + model + tokenizer files either in the directory or checkpoint-* subdirs."
        )

    base = Path(__file__).resolve().parent / "models"
    candidates = sorted([p for p in base.glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No model directories found under {base}")
    for p in candidates:
        if _has_model_artifacts(p):
            return p.resolve()
        ckpt = _pick_best_checkpoint_dir(p)
        if ckpt is not None:
            return ckpt
    raise FileNotFoundError(f"No loadable model artifacts found under {base}")


def _select_device(*, cpu_only: bool) -> Tuple[Any, str]:
    import torch

    if not cpu_only and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def _load_model(model_dir: Path, device, device_name: str):
    try:
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Missing inference dependencies. Install in your venv:\n"
            "pip install torch transformers"
        ) from e

    tok_last_err: Optional[Exception] = None
    tokenizer = None
    tok_attempts = (
        {"use_fast": True},
        {"use_fast": False},
        {},
    )
    for kwargs in tok_attempts:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
                **kwargs,
            )
            break
        except Exception as e:
            tok_last_err = e

    # Fallback: tokenizer from base model id in config.
    if tokenizer is None:
        base_name = ""
        try:
            cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
            base_name = _clean_text(getattr(cfg, "_name_or_path", ""))
        except Exception:
            base_name = ""
        if base_name and base_name != str(model_dir):
            for kwargs in tok_attempts:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True, **kwargs)
                    break
                except Exception as e:
                    tok_last_err = e

    if tokenizer is None:
        raise RuntimeError(
            "Failed to load tokenizer for inference. "
            "Install tokenizer deps if needed: `pip install sentencepiece tiktoken`.\n"
            f"Model dir: {model_dir}\n"
            f"Last tokenizer error: {tok_last_err}"
        )

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), trust_remote_code=True)
    model.to(device)
    model.eval()

    # Explicitly avoid CUDA usage in this inference script.
    if device_name != "mps":
        torch.set_num_threads(max(1, torch.get_num_threads()))
    return tokenizer, model


def _score_pairs(
    *,
    tokenizer,
    model,
    device,
    pairs: Sequence[Dict[str, Any]],
    batch_size: int,
    max_length: int,
) -> List[float]:
    import torch

    out: List[float] = []
    bs = max(1, int(batch_size))
    mx = max(32, int(max_length))
    pair_list = list(pairs or [])
    for start in range(0, len(pair_list), bs):
        batch = pair_list[start : start + bs]
        queries = [_clean_text(x.get("query")) for x in batch]
        docs = [_clean_text(x.get("doc")) for x in batch]
        enc = tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=mx,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            if logits.ndim == 2 and logits.shape[-1] > 1:
                scores = logits[:, -1]
            else:
                scores = logits.squeeze(-1)
            out.extend([float(x) for x in scores.detach().cpu().tolist()])
    return out


def _load_candidates_from_file(path: str) -> List[str]:
    if not _clean_text(path):
        return []
    p = Path(_clean_text(path)).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Candidates file not found: {p}")
    out: List[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        t = _clean_text(raw)
        if t:
            out.append(t)
    return out


def _load_pair_rows(input_jsonl: str) -> List[Dict[str, Any]]:
    p = Path(_clean_text(input_jsonl)).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {p}")
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                item = dict(json.loads(line) or {})
            except Exception:
                continue
            query = _clean_text(item.get("query"))
            doc = _clean_text(item.get("doc"))
            if not query or not doc:
                continue
            out.append({"query": query, "doc": doc, "raw": item})
    return out


def run_inference(
    *,
    model_dir: str,
    query: str,
    candidates: Sequence[str],
    candidates_file: str,
    input_jsonl: str,
    output_jsonl: str,
    batch_size: int,
    max_length: int,
    top_k: int,
    cpu_only: bool,
) -> Dict[str, Any]:
    resolved_model_dir = _resolve_model_dir(model_dir)
    device, device_name = _select_device(cpu_only=bool(cpu_only))
    tokenizer, model = _load_model(resolved_model_dir, device, device_name)

    safe_batch = _safe_limit(batch_size, default=64, minimum=1, maximum=4096)
    safe_max_len = _safe_limit(max_length, default=256, minimum=32, maximum=4096)

    q = _clean_text(query)
    cands = list(candidates or [])
    cands.extend(_load_candidates_from_file(candidates_file))
    cands = [_clean_text(x) for x in cands if _clean_text(x)]

    mode = ""
    payload: Dict[str, Any] = {
        "model_dir": str(resolved_model_dir),
        "device": device_name,
        "batch_size": safe_batch,
        "max_length": safe_max_len,
    }

    if q and cands:
        mode = "query_rank"
        pairs = [{"query": q, "doc": c} for c in cands]
        scores = _score_pairs(
            tokenizer=tokenizer,
            model=model,
            device=device,
            pairs=pairs,
            batch_size=safe_batch,
            max_length=safe_max_len,
        )
        rows = []
        for idx, (cand, score) in enumerate(zip(cands, scores), start=1):
            rows.append({"candidate": cand, "score": float(score), "input_order": int(idx)})
        rows.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        kk = _safe_limit(top_k, default=10, minimum=1, maximum=1000000)
        payload.update(
            {
                "mode": mode,
                "query": q,
                "candidate_count": len(rows),
                "top_k": kk,
                "ranked": rows[:kk],
            }
        )
        return payload

    if _clean_text(input_jsonl):
        mode = "batch_jsonl"
        pair_rows = _load_pair_rows(input_jsonl)
        pairs = [{"query": x["query"], "doc": x["doc"]} for x in pair_rows]
        scores = _score_pairs(
            tokenizer=tokenizer,
            model=model,
            device=device,
            pairs=pairs,
            batch_size=safe_batch,
            max_length=safe_max_len,
        )

        scored_rows: List[Dict[str, Any]] = []
        for item, score in zip(pair_rows, scores):
            row = dict(item.get("raw") or {})
            row["score"] = float(score)
            scored_rows.append(row)

        out_path = ""
        if _clean_text(output_jsonl):
            out_p = Path(_clean_text(output_jsonl)).expanduser().resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with out_p.open("w", encoding="utf-8") as f:
                for row in scored_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_path = str(out_p)

        payload.update(
            {
                "mode": mode,
                "input_rows": len(pair_rows),
                "scored_rows": len(scored_rows),
                "output_jsonl": out_path,
                "preview": scored_rows[: min(10, len(scored_rows))],
            }
        )
        return payload

    raise ValueError("Provide either --query + --candidate(s), or --input-jsonl.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference for fine-tuned bge reranker. Device policy: MPS if available, otherwise CPU (never CUDA)."
    )
    parser.add_argument("--model-dir", type=str, default="", help="Fine-tuned model directory. Default: latest under train_cross_encoder/models.")

    parser.add_argument("--query", type=str, default="", help="Single query specialization text.")
    parser.add_argument("--candidate", action="append", default=[], help="Candidate specialization text (repeatable).")
    parser.add_argument("--candidates-file", type=str, default="", help="Text file with one candidate per line.")
    parser.add_argument("--top-k", type=int, default=10, help="Top K results for query mode.")

    parser.add_argument("--input-jsonl", type=str, default="", help="Batch input JSONL with {'query','doc'} per line.")
    parser.add_argument("--output-jsonl", type=str, default="", help="Optional scored output JSONL path for batch mode.")

    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max length.")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU even if MPS is available.")
    parser.add_argument("--json-only", action="store_true", help="Print only JSON payload.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    payload = run_inference(
        model_dir=_clean_text(args.model_dir),
        query=_clean_text(args.query),
        candidates=list(args.candidate or []),
        candidates_file=_clean_text(args.candidates_file),
        input_jsonl=_clean_text(args.input_jsonl),
        output_jsonl=_clean_text(args.output_jsonl),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        top_k=int(args.top_k),
        cpu_only=bool(args.cpu_only),
    )

    if not args.json_only:
        print("Reranker inference complete.")
        print(f"  device      : {payload.get('device', '')}")
        print(f"  model_dir   : {payload.get('model_dir', '')}")
        print(f"  mode        : {payload.get('mode', '')}")
        if payload.get("mode") == "query_rank":
            print(f"  candidates  : {payload.get('candidate_count', 0)}")
        elif payload.get("mode") == "batch_jsonl":
            print(f"  scored_rows : {payload.get('scored_rows', 0)}")
        print()

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
