from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import get_llm_client, settings
from train_cross_encoder.infer_bge_reranker import (
    _resolve_model_dir,
    _score_pairs,
    _select_device,
)


DEFAULT_QUERY_COUNT = 120
DEFAULT_MIN_CANDIDATES = 4
DEFAULT_FAILURE_SAMPLE_SIZE = 20
DEFAULT_INFER_BATCH_SIZE = 64
DEFAULT_INFER_MAX_LENGTH = 256
DEFAULT_LOW_MARGIN_THRESHOLD = 0.05
DEFAULT_LOW_CONF_THRESHOLD = 0.55
DEFAULT_HARD_NEG_WINDOW_SIZE = 3


class QualityReviewOut(BaseModel):
    overall_summary: str = Field(default="")
    likely_failure_patterns: List[str] = Field(default_factory=list)
    dataset_fix_actions: List[str] = Field(default_factory=list)
    training_fix_actions: List[str] = Field(default_factory=list)

    @field_validator(
        "likely_failure_patterns",
        "dataset_fix_actions",
        "training_fix_actions",
        mode="before",
    )
    @classmethod
    def _coerce_list_fields(cls, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(x or "").strip() for x in value if str(x or "").strip()]
        if isinstance(value, str):
            txt = value.strip()
            if not txt:
                return []
            try:
                parsed = json.loads(txt)
                if isinstance(parsed, list):
                    return [str(x or "").strip() for x in parsed if str(x or "").strip()]
            except Exception:
                pass
            lines = [x.strip() for x in txt.splitlines() if x.strip()]
            out: List[str] = []
            for line in lines:
                t = line
                if t[:1] in {"-", "*", "•"}:
                    t = t[1:].strip()
                out.append(t)
            return out
        return []


QUALITY_REVIEW_SYSTEM_PROMPT = (
    "You are auditing cross-encoder ranking quality for spec->chunk matching.\n"
    "The model should rank true support chunks above semantically close negatives.\n"
    "Given deterministic metrics and failure samples, provide practical guidance.\n"
    "Return JSON only."
)

QUALITY_REVIEW_HUMAN_PROMPT = (
    "Metrics JSON:\n{metrics_json}\n\n"
    "Failure examples JSON:\n{failures_json}\n\n"
    "Output keys:\n"
    "- overall_summary\n"
    "- likely_failure_patterns\n"
    "- dataset_fix_actions\n"
    "- training_fix_actions\n"
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    if out < minimum:
        return minimum
    if out > maximum:
        return maximum
    return out


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return float(out)


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip(value: Any, max_chars: int = 240) -> str:
    t = _clean_text(value)
    if len(t) <= int(max_chars):
        return t
    return t[: int(max_chars)].rstrip() + " ...<truncated>"


def _pick_best_tokenizer_error(errors: Sequence[Exception]) -> Optional[Exception]:
    if not errors:
        return None
    # Prefer actionable dependency errors over downstream tokenizer conversion crashes.
    for e in errors:
        msg = _clean_text(e).lower()
        if "sentencepiece" in msg:
            return e
    for e in errors:
        msg = _clean_text(e).lower()
        if "tiktoken" in msg:
            return e
    return errors[-1]


def _load_model_stable_for_eval(model_dir: Path, device) -> Tuple[Any, Any]:
    try:
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Missing inference dependencies. Install in your venv:\n"
            "pip install torch transformers"
        ) from e

    target = str(model_dir)
    is_local = bool(model_dir.exists())
    tok_errors: List[Exception] = []
    tokenizer = None

    # Slow tokenizer first: avoids known fast tokenizer conversion edge-cases.
    attempts = (
        {"use_fast": False, "fix_mistral_regex": True},
        {"use_fast": False},
        {"use_fast": True, "fix_mistral_regex": True},
        {"use_fast": True},
    )

    for kwargs in attempts:
        try_kwargs = dict(kwargs)
        if is_local:
            try_kwargs["local_files_only"] = True
        try:
            tokenizer = AutoTokenizer.from_pretrained(target, trust_remote_code=True, **try_kwargs)
            break
        except Exception as e:
            tok_errors.append(e)

    if tokenizer is None:
        base_name = ""
        try:
            cfg_kwargs = {"trust_remote_code": True}
            if is_local:
                cfg_kwargs["local_files_only"] = True
            cfg = AutoConfig.from_pretrained(target, **cfg_kwargs)
            base_name = _clean_text(getattr(cfg, "_name_or_path", ""))
        except Exception:
            base_name = ""

        if base_name and base_name != target and not is_local:
            for kwargs in attempts:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True, **dict(kwargs))
                    break
                except Exception as e:
                    tok_errors.append(e)

    if tokenizer is None:
        best_err = _pick_best_tokenizer_error(tok_errors)
        raise RuntimeError(
            "Failed to load tokenizer for inference. "
            "Install tokenizer deps if needed: `pip install sentencepiece tiktoken`.\n"
            f"Model dir: {model_dir}\n"
            f"Last tokenizer error: {best_err}"
        )

    model_kwargs = {"trust_remote_code": True}
    if is_local:
        model_kwargs["local_files_only"] = True
    model = AutoModelForSequenceClassification.from_pretrained(target, **model_kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model


def _resolve_latest_boundary_listwise(dataset_dir: Path) -> Path:
    cands = sorted(
        dataset_dir.glob("*_listwise_boundary_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if cands:
        return cands[0].resolve()
    cands = sorted(
        dataset_dir.glob("*_listwise_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"No listwise dataset found under {dataset_dir}")
    return cands[0].resolve()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_topks(value: str) -> List[int]:
    raw = _clean_text(value)
    if not raw:
        return [1, 3, 5]
    out: List[int] = []
    for part in raw.split(","):
        p = _clean_text(part)
        if not p:
            continue
        try:
            k = int(p)
        except Exception:
            continue
        if k <= 0:
            continue
        out.append(int(k))
    if not out:
        out = [1, 3, 5]
    out = sorted(set(out))
    return out


def _load_listwise_rows(
    *,
    path: Path,
    query_count: int,
    min_candidates: int,
    require_boundary_labels: bool,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    safe_q = _safe_int(query_count, default=DEFAULT_QUERY_COUNT, minimum=0, maximum=1_000_000)
    safe_min = _safe_int(min_candidates, default=DEFAULT_MIN_CANDIDATES, minimum=2, maximum=512)
    rows: List[Dict[str, Any]] = []
    skip = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = dict(json.loads(line) or {})
            except Exception:
                skip["parse_error"] += 1
                continue

            query = _clean_text(item.get("query"))
            query_group = _clean_text(item.get("query_group")) or f"q::{line_no}"
            ranking = [int(x) for x in list(item.get("ranking") or []) if int(x) > 0]
            candidates = list(item.get("candidates") or [])
            if not query or len(ranking) < safe_min or len(candidates) < safe_min:
                skip["bad_shape"] += 1
                continue

            by_i: Dict[int, str] = {}
            for c in candidates:
                i = int(c.get("i") or 0)
                t = _clean_text(c.get("t") or c.get("text"))
                if i > 0 and t:
                    by_i[i] = t

            ordered = [i for i in ranking if i in by_i]
            if len(ordered) < safe_min:
                skip["too_few_candidates"] += 1
                continue

            positives = [int(x) for x in list(item.get("positive_indices") or []) if int(x) in by_i]
            negatives = [int(x) for x in list(item.get("negative_indices") or []) if int(x) in by_i]
            uncertain = [int(x) for x in list(item.get("uncertain_indices") or []) if int(x) in by_i]
            if bool(require_boundary_labels):
                if not positives:
                    skip["no_positive"] += 1
                    continue
                if not negatives:
                    skip["no_negative"] += 1
                    continue

            rows.append(
                {
                    "row_idx": int(item.get("row_idx") or line_no),
                    "query": query,
                    "query_group": query_group,
                    "ranking": ordered,
                    "candidates": [{"i": int(i), "t": by_i[int(i)]} for i in ordered],
                    "first_negative_rank": int(item.get("first_negative_rank") or 0),
                    "positive_indices": positives,
                    "negative_indices": negatives,
                    "uncertain_indices": uncertain,
                    "label_source": _clean_text(item.get("label_source")),
                }
            )

    if safe_q > 0 and len(rows) > safe_q:
        rng = random.Random(int(seed))
        rng.shuffle(rows)
        rows = rows[:safe_q]

    meta = {
        "input_path": str(path),
        "loaded_queries": int(len(rows)),
        "rows_skipped": int(sum(skip.values())),
        "skip_counts": dict(skip),
        "require_boundary_labels": bool(require_boundary_labels),
    }
    return rows, meta


def _infer_scores(
    *,
    model_dir: str,
    rows: Sequence[Dict[str, Any]],
    cpu_only: bool,
    batch_size: int,
    max_length: int,
    hard_negative_window_size: int,
) -> Dict[str, Any]:
    resolved_model_dir = _resolve_model_dir(model_dir)
    device, device_name = _select_device(cpu_only=bool(cpu_only))
    tokenizer, model = _load_model_stable_for_eval(Path(resolved_model_dir), device)

    pair_payloads: List[Dict[str, Any]] = []
    backref: List[Tuple[int, int]] = []
    for ridx, row in enumerate(list(rows or [])):
        q = _clean_text(row.get("query"))
        cands = list(row.get("candidates") or [])
        for cidx, c in enumerate(cands):
            text = _clean_text(c.get("t"))
            if not q or not text:
                continue
            pair_payloads.append({"query": q, "doc": text})
            backref.append((int(ridx), int(cidx)))

    scores = _score_pairs(
        tokenizer=tokenizer,
        model=model,
        device=device,
        pairs=pair_payloads,
        batch_size=_safe_int(batch_size, default=DEFAULT_INFER_BATCH_SIZE, minimum=1, maximum=8192),
        max_length=_safe_int(max_length, default=DEFAULT_INFER_MAX_LENGTH, minimum=32, maximum=4096),
    )

    scored_rows: List[Dict[str, Any]] = [dict(x) for x in list(rows or [])]
    rank_maps: List[Dict[int, int]] = []
    for row in scored_rows:
        ranking = [int(x) for x in list(row.get("ranking") or []) if int(x) > 0]
        rank_maps.append({idx: pos for pos, idx in enumerate(ranking, start=1)})

    for score, (ridx, cidx) in zip(scores, backref):
        if not (0 <= ridx < len(scored_rows)):
            continue
        cands = list(scored_rows[ridx].get("candidates") or [])
        if not (0 <= cidx < len(cands)):
            continue
        cands[cidx]["score"] = float(score)
        cands[cidx]["confidence"] = float(_sigmoid(float(score)))
        scored_rows[ridx]["candidates"] = cands

    safe_hard_window = _safe_int(
        hard_negative_window_size,
        default=DEFAULT_HARD_NEG_WINDOW_SIZE,
        minimum=1,
        maximum=32,
    )

    final_rows: List[Dict[str, Any]] = []
    for ridx, row in enumerate(scored_rows):
        rank_map = rank_maps[ridx] if ridx < len(rank_maps) else {}
        positives = set(int(x) for x in list(row.get("positive_indices") or []))
        negatives = set(int(x) for x in list(row.get("negative_indices") or []))
        uncertain = set(int(x) for x in list(row.get("uncertain_indices") or []))
        boundary = int(row.get("first_negative_rank") or 0)

        cand_out: List[Dict[str, Any]] = []
        for c in list(row.get("candidates") or []):
            i = int(c.get("i") or 0)
            t = _clean_text(c.get("t"))
            s = _safe_float(c.get("score"), default=-1e9)
            conf = _safe_float(c.get("confidence"), default=0.0)
            teacher_rank = int(rank_map.get(i) or 0)
            label_kind = "unknown"
            if i in positives:
                label_kind = "positive"
            elif i in uncertain:
                label_kind = "uncertain"
            elif i in negatives:
                if boundary > 0 and teacher_rank >= boundary and teacher_rank < (boundary + safe_hard_window):
                    label_kind = "hard_negative"
                else:
                    label_kind = "easy_negative"
            cand_out.append(
                {
                    "i": int(i),
                    "text": t,
                    "score": float(s),
                    "confidence": float(conf),
                    "teacher_rank": int(teacher_rank),
                    "label_kind": label_kind,
                }
            )
        cand_out.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        for rank_pos, c in enumerate(cand_out, start=1):
            c["model_rank"] = int(rank_pos)

        final_rows.append(
            {
                "row_idx": int(row.get("row_idx") or 0),
                "query": _clean_text(row.get("query")),
                "query_group": _clean_text(row.get("query_group")),
                "label_source": _clean_text(row.get("label_source")),
                "first_negative_rank": int(row.get("first_negative_rank") or 0),
                "positive_indices": list(row.get("positive_indices") or []),
                "negative_indices": list(row.get("negative_indices") or []),
                "uncertain_indices": list(row.get("uncertain_indices") or []),
                "ranked": cand_out,
            }
        )

    return {
        "model_dir": str(resolved_model_dir),
        "device": device_name,
        "pair_count": int(len(pair_payloads)),
        "scored_queries": final_rows,
    }


def _compute_metrics(
    *,
    scored_rows: Sequence[Dict[str, Any]],
    top_ks: Sequence[int],
    low_margin_threshold: float,
    low_conf_threshold: float,
) -> Dict[str, Any]:
    rows = list(scored_rows or [])
    if not rows:
        return {
            "query_count": 0,
            "top1_accuracy": 0.0,
            "mrr": 0.0,
            "false_positive_rate": 0.0,
            "hard_negative_top1_rate": 0.0,
            "false_negative_rate_at_3": 0.0,
            "low_confidence_ratio": 0.0,
            "low_margin_ratio": 0.0,
            "mean_top1_top2_margin": 0.0,
            "recall_at": {},
            "low_margin_threshold": float(low_margin_threshold),
            "low_confidence_threshold": float(low_conf_threshold),
        }

    ks = sorted(set(int(k) for k in list(top_ks or [1, 3, 5]) if int(k) > 0))
    if not ks:
        ks = [1, 3, 5]

    top1_correct = 0
    mrr_sum = 0.0
    fp_top1 = 0
    hard_top1 = 0
    fn_at_3 = 0
    low_conf = 0
    low_margin = 0
    margin_sum = 0.0
    margin_n = 0
    recall_hits = {int(k): 0 for k in ks}

    labeled_rows = 0
    for row in rows:
        ranked = list(row.get("ranked") or [])
        if not ranked:
            continue
        has_labels = any(_clean_text(x.get("label_kind")) in {"positive", "hard_negative", "easy_negative"} for x in ranked)
        if not has_labels:
            continue
        labeled_rows += 1

        top1 = dict(ranked[0])
        top1_kind = _clean_text(top1.get("label_kind"))
        top1_is_pos = top1_kind == "positive"
        if top1_is_pos:
            top1_correct += 1
        else:
            fp_top1 += 1
        if top1_kind == "hard_negative":
            hard_top1 += 1

        c1 = _safe_float(top1.get("confidence"), default=0.0)
        if c1 < float(low_conf_threshold):
            low_conf += 1

        if len(ranked) >= 2:
            margin = _safe_float(ranked[0].get("score")) - _safe_float(ranked[1].get("score"))
        else:
            margin = 0.0
        margin_sum += float(margin)
        margin_n += 1
        if margin < float(low_margin_threshold):
            low_margin += 1

        first_pos_rank: Optional[int] = None
        for pos, item in enumerate(ranked, start=1):
            if _clean_text(item.get("label_kind")) == "positive":
                first_pos_rank = int(pos)
                break
        if first_pos_rank is not None:
            mrr_sum += 1.0 / float(first_pos_rank)

        for k in ks:
            topk = ranked[: int(k)]
            if any(_clean_text(x.get("label_kind")) == "positive" for x in topk):
                recall_hits[int(k)] += 1

        if not any(_clean_text(x.get("label_kind")) == "positive" for x in ranked[:3]):
            fn_at_3 += 1

    n = max(1, labeled_rows)
    return {
        "query_count": int(labeled_rows),
        "top1_accuracy": float(top1_correct) / float(n),
        "mrr": float(mrr_sum) / float(n),
        "false_positive_rate": float(fp_top1) / float(n),
        "hard_negative_top1_rate": float(hard_top1) / float(n),
        "false_negative_rate_at_3": float(fn_at_3) / float(n),
        "low_confidence_ratio": float(low_conf) / float(n),
        "low_margin_ratio": float(low_margin) / float(max(1, margin_n)),
        "mean_top1_top2_margin": float(margin_sum) / float(max(1, margin_n)),
        "recall_at": {str(k): float(recall_hits[k]) / float(n) for k in ks},
        "low_margin_threshold": float(low_margin_threshold),
        "low_confidence_threshold": float(low_conf_threshold),
    }


def _compute_acceptance_gates(
    *,
    metrics: Dict[str, Any],
    gate_hard_negative_top1_rate_max: Optional[float],
    gate_false_positive_rate_max: Optional[float],
    gate_mean_top1_top2_margin_min: Optional[float],
) -> Dict[str, Any]:
    m_hard = float(metrics.get("hard_negative_top1_rate") or 0.0)
    m_fp = float(metrics.get("false_positive_rate") or 0.0)
    m_margin = float(metrics.get("mean_top1_top2_margin") or 0.0)

    hard_enabled = gate_hard_negative_top1_rate_max is not None
    fp_enabled = gate_false_positive_rate_max is not None
    margin_enabled = gate_mean_top1_top2_margin_min is not None

    hard_pass = (not hard_enabled) or (m_hard <= float(gate_hard_negative_top1_rate_max or 0.0))
    fp_pass = (not fp_enabled) or (m_fp <= float(gate_false_positive_rate_max or 0.0))
    margin_pass = (not margin_enabled) or (m_margin >= float(gate_mean_top1_top2_margin_min or 0.0))
    all_pass = bool(hard_pass and fp_pass and margin_pass)

    return {
        "enabled": bool(hard_enabled or fp_enabled or margin_enabled),
        "all_pass": bool(all_pass),
        "hard_negative_top1_rate": {
            "enabled": bool(hard_enabled),
            "threshold_max": (float(gate_hard_negative_top1_rate_max) if hard_enabled else None),
            "actual": float(m_hard),
            "pass": bool(hard_pass),
        },
        "false_positive_rate": {
            "enabled": bool(fp_enabled),
            "threshold_max": (float(gate_false_positive_rate_max) if fp_enabled else None),
            "actual": float(m_fp),
            "pass": bool(fp_pass),
        },
        "mean_top1_top2_margin": {
            "enabled": bool(margin_enabled),
            "threshold_min": (float(gate_mean_top1_top2_margin_min) if margin_enabled else None),
            "actual": float(m_margin),
            "pass": bool(margin_pass),
        },
    }


def _select_failure_examples(
    *,
    scored_rows: Sequence[Dict[str, Any]],
    sample_size: int,
    low_margin_threshold: float,
    low_conf_threshold: float,
) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for row in list(scored_rows or []):
        ranked = list(row.get("ranked") or [])
        if not ranked:
            continue
        top1 = dict(ranked[0])
        top1_kind = _clean_text(top1.get("label_kind"))
        top1_pos = top1_kind == "positive"
        conf = _safe_float(top1.get("confidence"), default=0.0)
        margin = (
            _safe_float(ranked[0].get("score")) - _safe_float(ranked[1].get("score"))
            if len(ranked) >= 2
            else 0.0
        )
        bad = (not top1_pos) or (conf < float(low_conf_threshold)) or (margin < float(low_margin_threshold))
        if not bad:
            continue
        failures.append(
            {
                "query": _clean_text(row.get("query")),
                "query_group": _clean_text(row.get("query_group")),
                "top1_kind": top1_kind,
                "top1_score": float(_safe_float(top1.get("score"))),
                "top1_confidence": float(conf),
                "margin_top1_top2": float(margin),
                "top1_text": _clip(top1.get("text"), max_chars=220),
                "top_ranked_preview": [
                    {
                        "rank": int(x.get("model_rank") or 0),
                        "kind": _clean_text(x.get("label_kind")),
                        "score": float(_safe_float(x.get("score"))),
                        "text": _clip(x.get("text"), max_chars=160),
                    }
                    for x in ranked[:5]
                ],
            }
        )
    failures.sort(
        key=lambda x: (
            0 if _clean_text(x.get("top1_kind")) != "positive" else 1,
            float(x.get("top1_confidence") or 0.0),
            float(x.get("margin_top1_top2") or 0.0),
        )
    )
    return failures[: max(1, int(sample_size))]


def _build_quality_chain(model_id: str):
    try:
        from langchain_core.prompts import ChatPromptTemplate
    except Exception as e:
        raise RuntimeError("Missing dependency for LLM quality review. Install: pip install langchain-core") from e

    llm = get_llm_client(model_id).build()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUALITY_REVIEW_SYSTEM_PROMPT),
            ("human", QUALITY_REVIEW_HUMAN_PROMPT),
        ]
    )
    return prompt | llm.with_structured_output(QualityReviewOut)


def _llm_quality_review(
    *,
    llm_model: str,
    metrics: Dict[str, Any],
    failures: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    def _fallback(error_text: str) -> Dict[str, Any]:
        fp = float(metrics.get("false_positive_rate") or 0.0)
        hard = float(metrics.get("hard_negative_top1_rate") or 0.0)
        conf = float(metrics.get("low_confidence_ratio") or 0.0)
        margin = float(metrics.get("low_margin_ratio") or 0.0)
        patterns: List[str] = []
        if hard >= 0.20:
            patterns.append("Hard negatives frequently outrank positives at top-1.")
        if fp >= 0.25:
            patterns.append("False-positive rate is elevated; lexical overlap may dominate.")
        if conf >= 0.40:
            patterns.append("Low-confidence top predictions indicate weak calibration.")
        if margin >= 0.35:
            patterns.append("Top1-top2 margin compression suggests unstable ranking separation.")
        if not patterns:
            patterns.append("No dominant failure mode detected from deterministic metrics.")
        return {
            "overall_summary": (
                "LLM review fallback used due to structured output failure. "
                f"error={error_text}"
            ),
            "likely_failure_patterns": patterns,
            "dataset_fix_actions": [
                "Increase query-local hard negatives that share topic words but differ in intent.",
                "Remove noisy profiles/citation-dump chunks that are not valid semantic evidence.",
                "Keep only query groups with at least one high-confidence positive chunk.",
            ],
            "training_fix_actions": [
                "Increase hard-negative sampling weight and keep per-query negative diversity.",
                "Run ANCE-style remine after each round and add false-top negatives back.",
                "Track margin-focused metrics and stop when margin gains plateau.",
            ],
            "model": llm_model,
            "fallback": "deterministic_heuristic",
        }

    if not failures:
        return {
            "overall_summary": "No major failure examples in sampled labeled queries.",
            "likely_failure_patterns": [],
            "dataset_fix_actions": [],
            "training_fix_actions": [],
            "model": llm_model,
        }

    try:
        chain = _build_quality_chain(llm_model)
        out = chain.invoke(
            {
                "metrics_json": json.dumps(metrics, ensure_ascii=False, indent=2),
                "failures_json": json.dumps(list(failures), ensure_ascii=False, indent=2),
            }
        )
        payload = out.model_dump() if hasattr(out, "model_dump") else dict(out or {})
        payload["likely_failure_patterns"] = list(payload.get("likely_failure_patterns") or [])
        payload["dataset_fix_actions"] = list(payload.get("dataset_fix_actions") or [])
        payload["training_fix_actions"] = list(payload.get("training_fix_actions") or [])
        payload["model"] = llm_model
        return payload
    except Exception as e:
        return _fallback(f"{type(e).__name__}: {e}")


def run_eval(
    *,
    model_dir: str,
    listwise_jsonl: str,
    llm_model: str,
    query_count: int,
    min_candidates: int,
    require_boundary_labels: bool,
    output_dir: Path,
    cpu_only: bool,
    seed: int,
    infer_batch_size: int,
    infer_max_length: int,
    hard_negative_window_size: int,
    top_ks: Sequence[int],
    low_margin_threshold: float,
    low_conf_threshold: float,
    failure_sample_size: int,
    gate_hard_negative_top1_rate_max: Optional[float],
    gate_false_positive_rate_max: Optional[float],
    gate_mean_top1_top2_margin_min: Optional[float],
) -> Dict[str, Any]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir.expanduser().resolve() / f"llm_quality_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(_clean_text(listwise_jsonl)).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input listwise file not found: {input_path}")

    loaded_rows, load_meta = _load_listwise_rows(
        path=input_path,
        query_count=int(query_count),
        min_candidates=int(min_candidates),
        require_boundary_labels=bool(require_boundary_labels),
        seed=int(seed),
    )
    if not loaded_rows:
        raise RuntimeError("No valid query rows to evaluate.")

    inferred = _infer_scores(
        model_dir=model_dir,
        rows=loaded_rows,
        cpu_only=bool(cpu_only),
        batch_size=int(infer_batch_size),
        max_length=int(infer_max_length),
        hard_negative_window_size=int(hard_negative_window_size),
    )
    scored_rows = list(inferred.get("scored_queries") or [])

    metrics = _compute_metrics(
        scored_rows=scored_rows,
        top_ks=list(top_ks),
        low_margin_threshold=float(low_margin_threshold),
        low_conf_threshold=float(low_conf_threshold),
    )
    gates = _compute_acceptance_gates(
        metrics=metrics,
        gate_hard_negative_top1_rate_max=gate_hard_negative_top1_rate_max,
        gate_false_positive_rate_max=gate_false_positive_rate_max,
        gate_mean_top1_top2_margin_min=gate_mean_top1_top2_margin_min,
    )
    failures = _select_failure_examples(
        scored_rows=scored_rows,
        sample_size=int(failure_sample_size),
        low_margin_threshold=float(low_margin_threshold),
        low_conf_threshold=float(low_conf_threshold),
    )
    review = _llm_quality_review(
        llm_model=_clean_text(llm_model),
        metrics=metrics,
        failures=failures,
    )

    selected_queries_path = run_dir / "queries_selected.json"
    scored_queries_path = run_dir / "queries_scored.json"
    summary_path = run_dir / "quality_summary.json"

    _write_json(
        selected_queries_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "load_meta": load_meta,
            "queries": loaded_rows,
        },
    )
    _write_json(
        scored_queries_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_dir": inferred.get("model_dir"),
            "device": inferred.get("device"),
            "pair_count": inferred.get("pair_count"),
            "query_count": len(scored_rows),
            "queries": scored_rows,
        },
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "model_dir": inferred.get("model_dir"),
        "device": inferred.get("device"),
        "input_listwise_jsonl": str(input_path),
        "query_sampling": load_meta,
        "metrics": metrics,
        "acceptance_gates": gates,
        "failure_examples_count": int(len(failures)),
        "failure_examples_preview": failures[: min(5, len(failures))],
        "llm_quality_review": review,
        "files": {
            "queries_selected": str(selected_queries_path),
            "queries_scored": str(scored_queries_path),
            "summary": str(summary_path),
        },
    }
    _write_json(summary_path, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "dataset"
    eval_dir = base_dir / "eval_runs"
    model_default = str((base_dir / "models" / "promoted_best").resolve())
    try:
        listwise_default = str(_resolve_latest_boundary_listwise(dataset_dir))
    except Exception:
        listwise_default = ""

    p = argparse.ArgumentParser(
        description=(
            "Evaluate finetuned spec->chunk cross-encoder on sampled listwise queries "
            "and ask LLM for quality review."
        )
    )
    p.add_argument("--model-dir", type=str, default=model_default)
    p.add_argument("--listwise-jsonl", type=str, default=listwise_default)
    p.add_argument("--llm-model", type=str, default=_clean_text(settings.haiku))
    p.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    p.add_argument("--min-candidates", type=int, default=DEFAULT_MIN_CANDIDATES)
    p.add_argument("--require-boundary-labels", action="store_true")
    p.add_argument("--allow-unlabeled", action="store_true")
    p.add_argument("--output-dir", type=str, default=str(eval_dir))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--infer-batch-size", type=int, default=DEFAULT_INFER_BATCH_SIZE)
    p.add_argument("--infer-max-length", type=int, default=DEFAULT_INFER_MAX_LENGTH)
    p.add_argument("--hard-negative-window-size", type=int, default=DEFAULT_HARD_NEG_WINDOW_SIZE)
    p.add_argument("--top-ks", type=str, default="1,3,5")
    p.add_argument("--low-margin-threshold", type=float, default=DEFAULT_LOW_MARGIN_THRESHOLD)
    p.add_argument("--low-confidence-threshold", type=float, default=DEFAULT_LOW_CONF_THRESHOLD)
    p.add_argument("--failure-sample-size", type=int, default=DEFAULT_FAILURE_SAMPLE_SIZE)
    p.add_argument(
        "--gate-hard-negative-top1-rate-max",
        type=float,
        default=-1.0,
        help="Acceptance gate: pass only if hard_negative_top1_rate <= threshold. Negative = disabled.",
    )
    p.add_argument(
        "--gate-false-positive-rate-max",
        type=float,
        default=-1.0,
        help="Acceptance gate: pass only if false_positive_rate <= threshold. Negative = disabled.",
    )
    p.add_argument(
        "--gate-mean-top1-top2-margin-min",
        type=float,
        default=-1.0,
        help="Acceptance gate: pass only if mean_top1_top2_margin >= threshold. Negative = disabled.",
    )
    p.add_argument("--json-only", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    require_boundary = bool(args.require_boundary_labels)
    if bool(args.allow_unlabeled):
        require_boundary = False

    summary = run_eval(
        model_dir=_clean_text(args.model_dir),
        listwise_jsonl=_clean_text(args.listwise_jsonl),
        llm_model=_clean_text(args.llm_model) or _clean_text(settings.haiku),
        query_count=int(args.query_count),
        min_candidates=int(args.min_candidates),
        require_boundary_labels=bool(require_boundary),
        output_dir=Path(_clean_text(args.output_dir)),
        cpu_only=bool(args.cpu_only),
        seed=int(args.seed),
        infer_batch_size=int(args.infer_batch_size),
        infer_max_length=int(args.infer_max_length),
        hard_negative_window_size=int(args.hard_negative_window_size),
        top_ks=_parse_topks(_clean_text(args.top_ks)),
        low_margin_threshold=float(args.low_margin_threshold),
        low_conf_threshold=float(args.low_confidence_threshold),
        failure_sample_size=int(args.failure_sample_size),
        gate_hard_negative_top1_rate_max=(
            float(args.gate_hard_negative_top1_rate_max)
            if float(args.gate_hard_negative_top1_rate_max) >= 0.0
            else None
        ),
        gate_false_positive_rate_max=(
            float(args.gate_false_positive_rate_max)
            if float(args.gate_false_positive_rate_max) >= 0.0
            else None
        ),
        gate_mean_top1_top2_margin_min=(
            float(args.gate_mean_top1_top2_margin_min)
            if float(args.gate_mean_top1_top2_margin_min) >= 0.0
            else None
        ),
    )

    if not bool(args.json_only):
        metrics = dict(summary.get("metrics") or {})
        print("Spec-chunk LLM quality evaluation complete.")
        print(f"  run dir                 : {summary.get('run_dir', '')}")
        print(f"  model dir               : {summary.get('model_dir', '')}")
        print(f"  device                  : {summary.get('device', '')}")
        print(f"  query count             : {metrics.get('query_count', 0)}")
        print(f"  top1 accuracy           : {metrics.get('top1_accuracy', 0.0):.4f}")
        print(f"  mrr                     : {metrics.get('mrr', 0.0):.4f}")
        print(f"  false positive rate     : {metrics.get('false_positive_rate', 0.0):.4f}")
        print(f"  hard negative top1 rate : {metrics.get('hard_negative_top1_rate', 0.0):.4f}")
        print(f"  mean top1-top2 margin   : {metrics.get('mean_top1_top2_margin', 0.0):.4f}")
        gate = dict(summary.get("acceptance_gates") or {})
        if bool(gate.get("enabled")):
            print(f"  acceptance gates pass   : {bool(gate.get('all_pass'))}")
        print(f"  summary file            : {summary.get('files', {}).get('summary', '')}")
        print()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
