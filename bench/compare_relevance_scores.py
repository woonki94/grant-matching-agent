from __future__ import annotations

import argparse
import json
import os
import sys
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Editable benchmark inputs
QUERIES = [
    "sim-to-real reinforcement learning for bipedal robot locomotion under dynamic loads"
]
DOCS = [
     "data-driven systems-based approaches using artificial intelligence and machine learning for molecular mechanism discovery",
    "criminal history record data quality improvement and interstate accessibility enhancement systems",
    "fundamental engineering and computer science research challenges in robotics",
    "interdisciplinary robotics research integrating intelligence computation and embodiment",
    "project timeline planning and workplan development for groundwater monitoring objectives",
    "computational modeling and machine learning for chemical reaction optimization",
    "create artificial intelligence and machine learning models for naval platform integrity assessment",
    "artificial intelligence and machine learning algorithms for automated glucose control systems",
    "hybrid physical-virtual co-simulation for multi-agent reinforcement learning validation and testing",
    "learning evaluation measures and learner satisfaction assessment methods",
]

# Avoid OpenMP duplicate-runtime crashes in mixed local environments.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from utils.embedder import cosine_sim_matrix, embed_texts
from config import get_llm_client, settings

logger = logging.getLogger("bench.compare_relevance_scores")


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
    p = Path(_clean_text(model_dir)).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Model directory not found: {p}")
    logger.info("Resolved finetuned model dir: %s", p)
    return p


def _select_device(*, cpu_only: bool):
    import torch

    if not cpu_only and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def _load_cross_encoder(model_ref: str, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading cross-encoder model: %s", model_ref)
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_ref, trust_remote_code=True)
    model.to(device)
    model.eval()
    logger.info("Loaded model on device: %s", device)
    return tokenizer, model


def _score_pairs_cross_encoder(
    *,
    tokenizer,
    model,
    device,
    pairs: Sequence[Tuple[str, str]],
    batch_size: int,
    max_length: int,
) -> List[float]:
    import torch

    out: List[float] = []
    safe_batch = _safe_limit(batch_size, default=32, minimum=1, maximum=4096)
    safe_max_len = _safe_limit(max_length, default=256, minimum=32, maximum=4096)

    pair_list = list(pairs or [])
    logger.info(
        "Scoring cross-encoder pairs: count=%d batch_size=%d max_length=%d",
        len(pair_list),
        safe_batch,
        safe_max_len,
    )
    for start in range(0, len(pair_list), safe_batch):
        batch = pair_list[start : start + safe_batch]
        logger.debug("Cross-encoder batch: start=%d size=%d", start, len(batch))
        queries = [_clean_text(x[0]) for x in batch]
        docs = [_clean_text(x[1]) for x in batch]
        enc = tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=safe_max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            if logits.ndim == 2 and logits.shape[-1] > 1:
                scores = logits[:, -1]
            else:
                scores = logits.squeeze(-1)
            out.extend([float(v) for v in scores.detach().cpu().tolist()])
    logger.info("Completed cross-encoder scoring: produced=%d scores", len(out))
    return out


def _build_pairs(queries: Sequence[str], docs: Sequence[str], mode: str) -> List[Tuple[str, str]]:
    q = [_clean_text(x) for x in (queries or []) if _clean_text(x)]
    d = [_clean_text(x) for x in (docs or []) if _clean_text(x)]
    logger.info("Building pairs: queries=%d docs=%d mode=%s", len(q), len(d), mode)
    if not q or not d:
        return []
    if mode == "zip":
        n = min(len(q), len(d))
        pairs = [(q[i], d[i]) for i in range(n)]
        logger.info("Built zip pairs: %d", len(pairs))
        return pairs
    pairs: List[Tuple[str, str]] = []
    for qq in q:
        for dd in d:
            pairs.append((qq, dd))
    logger.info("Built cartesian pairs: %d", len(pairs))
    return pairs


def _score_pairs_cosine(pairs: Sequence[Tuple[str, str]]) -> List[float]:
    if not pairs:
        return []
    logger.info("Scoring cosine pairs: count=%d", len(pairs))
    queries = [x[0] for x in pairs]
    docs = [x[1] for x in pairs]
    q_emb = embed_texts(queries)
    d_emb = embed_texts(docs)
    sims = cosine_sim_matrix(q_emb, d_emb)
    out: List[float] = []
    for i in range(len(pairs)):
        out.append(float(sims[i, i]))
    logger.info("Completed cosine scoring: produced=%d scores", len(out))
    return out


class LLMPairNeedScore(BaseModel):
    i: int = Field(default=0)
    score: float = Field(default=0.0)
    reason: str = Field(default="")


class LLMGrantFacultyNeedOut(BaseModel):
    items: List[LLMPairNeedScore] = Field(default_factory=list)


GRANT_FACULTY_NEED_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You evaluate how much a faculty specialization is needed for a grant requirement.\n"
            "You will receive a JSON list of pairs with index i.\n"
            "Return one result item per input pair.\n"
            "For each pair, output: i, score in [0,1], and a short reason.\n"
            "Scoring guidelines:\n"
            "- 1.0: direct capability match and strong domain alignment.\n"
            "- 0.7-0.9: strong fit with minor scope differences.\n"
            "- 0.4-0.6: partial overlap.\n"
            "- 0.1-0.3: weak relation.\n"
            "- 0.0: unrelated.\n"
            "Penalize keyword-only overlap when the real domain/capability is different.\n"
            "Keep reason concise (one sentence).",
        ),
        (
            "human",
            "Pairs JSON:\n{pairs_json}",
        ),
    ]
)


def _score_pairs_llm(
    *,
    pairs: Sequence[Tuple[str, str]],
    model_id: Optional[str],
) -> Tuple[List[float], List[str]]:
    pair_list = list(pairs or [])
    if not pair_list:
        return [], []

    logger.info(
        "Scoring LLM pairs in one batch call: count=%d model=%s",
        len(pair_list),
        _clean_text(model_id) or "default",
    )

    llm = get_llm_client(_clean_text(model_id) or None).build()
    chain = GRANT_FACULTY_NEED_PROMPT | llm.with_structured_output(LLMGrantFacultyNeedOut)

    payload = [
        {
            "i": idx,
            "grant_requirement": _clean_text(q),
            "faculty_specialization": _clean_text(d),
        }
        for idx, (q, d) in enumerate(pair_list)
    ]
    logger.info("Invoking LLM once for all pairs")
    out = chain.invoke({"pairs_json": json.dumps(payload, ensure_ascii=False)})

    scores: List[float] = [0.0 for _ in pair_list]
    reasons: List[str] = ["" for _ in pair_list]
    missing = set(range(len(pair_list)))
    for item in (out.items or []):
        idx = int(item.i) if item and item.i is not None else -1
        if idx < 0 or idx >= len(pair_list):
            continue
        score = float(item.score if item.score is not None else 0.0)
        scores[idx] = min(1.0, max(0.0, score))
        reasons[idx] = _clean_text(item.reason)
        missing.discard(idx)

    if missing:
        logger.warning(
            "LLM batch output missing %d pair scores; filled with 0.0",
            len(missing),
        )

    return scores, reasons


def compare_scores(
    *,
    queries: Sequence[str],
    docs: Sequence[str],
    pair_mode: str,
    pure_ce_model: str,
    finetuned_model_dir: str,
    batch_size: int,
    max_length: int,
    cpu_only: bool,
    include_llm: bool,
    llm_model: Optional[str],
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    pairs = _build_pairs(queries, docs, pair_mode)
    if not pairs:
        raise ValueError("No valid query/doc pairs. Check QUERIES/DOCS or inputs.")

    device, device_name = _select_device(cpu_only=bool(cpu_only))
    logger.info("Selected device: %s", device_name)

    # 1) Cosine (bi-encoder embedding)
    t_cos = time.perf_counter()
    cosine_scores = _score_pairs_cosine(pairs)
    logger.info("Cosine stage finished in %.2fs", time.perf_counter() - t_cos)

    # 2) Pure cross-encoder (base model)
    t_pure = time.perf_counter()
    pure_tok, pure_model = _load_cross_encoder(pure_ce_model, device)
    pure_scores = _score_pairs_cross_encoder(
        tokenizer=pure_tok,
        model=pure_model,
        device=device,
        pairs=pairs,
        batch_size=batch_size,
        max_length=max_length,
    )
    logger.info("Pure cross-encoder stage finished in %.2fs", time.perf_counter() - t_pure)

    # 3) Finetuned cross-encoder (local checkpoint)
    t_ft = time.perf_counter()
    resolved_ft = _resolve_model_dir(finetuned_model_dir)
    ft_tok, ft_model = _load_cross_encoder(str(resolved_ft), device)
    ft_scores = _score_pairs_cross_encoder(
        tokenizer=ft_tok,
        model=ft_model,
        device=device,
        pairs=pairs,
        batch_size=batch_size,
        max_length=max_length,
    )
    logger.info("Finetuned cross-encoder stage finished in %.2fs", time.perf_counter() - t_ft)

    # 4) LLM comparison stage
    llm_scores: List[float] = [0.0 for _ in pairs]
    llm_reasons: List[str] = ["" for _ in pairs]
    if include_llm:
        t_llm = time.perf_counter()
        llm_scores, llm_reasons = _score_pairs_llm(
            pairs=pairs,
            model_id=llm_model,
        )
        logger.info("LLM stage finished in %.2fs", time.perf_counter() - t_llm)
    else:
        logger.info("LLM stage skipped")

    rows: List[Dict[str, Any]] = []
    for idx, (q, d) in enumerate(pairs):
        rows.append(
            {
                "query": q,
                "doc": d,
                "cosine_score": float(cosine_scores[idx]),
                "pure_ce_score": float(pure_scores[idx]),
                "finetuned_ce_score": float(ft_scores[idx]),
                "llm_need_score": float(llm_scores[idx]),
                "llm_reason": _clean_text(llm_reasons[idx]),
            }
        )

    rows.sort(
        key=lambda x: (
            _clean_text(x.get("query")),
            float(x.get("finetuned_ce_score") or 0.0),
        ),
        reverse=True,
    )
    logger.info("Score comparison completed in %.2fs (rows=%d)", time.perf_counter() - t0, len(rows))

    return {
        "pair_mode": pair_mode,
        "pair_count": len(rows),
        "device": device_name,
        "pure_ce_model": pure_ce_model,
        "finetuned_model_dir": str(resolved_ft),
        "llm_model": _clean_text(llm_model) or _clean_text(settings.haiku),
        "llm_enabled": bool(include_llm),
        "results": rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare cosine-sim, base cross-encoder, and finetuned cross-encoder scores "
            "for query/doc pairs."
        )
    )
    parser.add_argument("--pair-mode", type=str, default="all", choices=["all", "zip"])
    parser.add_argument("--pure-ce-model", type=str, default="BAAI/bge-reranker-base")
    parser.add_argument(
        "--finetuned-model-dir",
        type=str,
        default=(
            "/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/"
            "train_cross_encoder/models/bge-reranker-base-finetuned/checkpoint-47022"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--llm-model", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, _clean_text(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger.info("Starting compare_relevance_scores")
    try:
        payload = compare_scores(
            queries=QUERIES,
            docs=DOCS,
            pair_mode=_clean_text(args.pair_mode).lower() or "all",
            pure_ce_model=_clean_text(args.pure_ce_model) or "BAAI/bge-reranker-base",
            finetuned_model_dir=_clean_text(args.finetuned_model_dir),
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
            cpu_only=bool(args.cpu_only),
            include_llm=not bool(args.skip_llm),
            llm_model=_clean_text(args.llm_model) or None,
        )
    except Exception:
        logger.exception("compare_relevance_scores failed")
        return 1

    output_json = _clean_text(args.output_json)
    if output_json:
        out_path = Path(output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Wrote output JSON: %s", out_path)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    logger.info("Finished compare_relevance_scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
