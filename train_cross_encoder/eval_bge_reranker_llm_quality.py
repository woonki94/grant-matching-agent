from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field, field_validator

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import get_llm_client, settings
from train_cross_encoder.infer_bge_reranker import (
    _load_model,
    _resolve_model_dir,
    _score_pairs,
    _select_device,
)

# =========================
# Static Config (Simple Step 3)
# =========================
DEFAULT_CASE_COUNT = 120
PROBE_CASES_PER_LLM_CALL = 8
LLM_MAX_RETRIES = 2
INFER_BATCH_SIZE = 64
INFER_MAX_LENGTH = 256
LOW_MARGIN_THRESHOLD = 0.05
LOW_CONFIDENCE_THRESHOLD = 0.55
FAILURE_SAMPLE_SIZE = 20
TOP_KS = (1, 3, 5)


def _to_text_list(value: Any) -> List[str]:
    def _dedupe(values: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for item in values:
            t = str(item or "").strip()
            if not t:
                continue
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
        return out

    if isinstance(value, (list, tuple)):
        return _dedupe([str(x or "").strip() for x in list(value)])

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        try:
            parsed = json.loads(text)
            if isinstance(parsed, (list, tuple)):
                return _to_text_list(parsed)
        except Exception:
            pass

        lines: List[str] = []
        for raw in text.splitlines():
            line = str(raw or "").strip()
            if not line:
                continue
            if line[:1] in {"-", "*", "•"}:
                line = line[1:].strip()
            if line and line[0].isdigit():
                i = 0
                while i < len(line) and line[i].isdigit():
                    i += 1
                if i < len(line) and line[i] in {".", ")"}:
                    line = line[i + 1 :].strip()
            if line:
                lines.append(line)

        if not lines:
            lines = [x.strip() for x in text.replace(";", "\n").splitlines() if x.strip()]
        return _dedupe(lines)

    return []


class ProbeCase(BaseModel):
    query: str = Field(..., description="Grant/funding specialization query text.")
    positives: List[str] = Field(default_factory=list, description="Clearly relevant faculty specialization candidates.")
    hard_negatives: List[str] = Field(
        default_factory=list,
        description="Semantically close but incorrect candidates (false-positive traps).",
    )
    easy_negatives: List[str] = Field(
        default_factory=list,
        description="Clearly unrelated candidates.",
    )
    case_tag: str = Field(
        default="",
        description="Case type: hard_negative | false_positive_trap | false_negative_trap",
    )


class ProbeCaseBatch(BaseModel):
    items: List[ProbeCase] = Field(default_factory=list)


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
        return _to_text_list(value)


PROBE_GEN_SYSTEM_PROMPT = (
    "You are generating evaluation probes for a grant-to-faculty specialization reranker.\n"
    "Produce realistic research-specialization cases.\n"
    "\n"
    "For each case:\n"
    "- query: a grant specialization need.\n"
    "- positives: 1-2 truly relevant faculty specialization candidates.\n"
    "- hard_negatives: 2-3 near-topic but wrong candidates (false-positive traps).\n"
    "- easy_negatives: 2-3 clearly unrelated candidates.\n"
    "- case_tag: choose one: hard_negative | false_positive_trap | false_negative_trap.\n"
    "\n"
    "Rules:\n"
    "- Keep text concise (one line each).\n"
    "- Avoid duplicate candidates in the same case.\n"
    "- Avoid generic fluff; use concrete specialization language.\n"
    "- Return JSON only."
)

PROBE_GEN_HUMAN_PROMPT = "Generate {count} cases. Use this seed for variation only: {seed_hint}"

QUALITY_REVIEW_SYSTEM_PROMPT = (
    "You are auditing reranker quality. Focus on practical, data-centric advice.\n"
    "You are given deterministic metrics and failure examples.\n"
    "Return JSON only."
)

QUALITY_REVIEW_HUMAN_PROMPT = (
    "Metrics JSON:\n{metrics_json}\n\n"
    "Failure examples JSON:\n{failures_json}\n\n"
    "Provide:\n"
    "- overall_summary\n"
    "- likely_failure_patterns (short bullets)\n"
    "- dataset_fix_actions (specific data generation/labeling fixes)\n"
    "- training_fix_actions (small training adjustments)"
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _default_probe_case(idx: int) -> Dict[str, Any]:
    templates = [
        (
            "federated learning for clinical imaging privacy",
            ["federated optimization for medical image segmentation"],
            ["privacy-preserving language model alignment in social media"],
            ["marine ecology field survey instrumentation"],
            "hard_negative",
        ),
        (
            "graph neural networks for drug-target interaction prediction",
            ["geometric deep learning for molecular property prediction"],
            ["drug adherence prediction from claims data"],
            ["coastal erosion mapping with remote sensing"],
            "false_positive_trap",
        ),
        (
            "reinforcement learning for power grid demand response",
            ["safe reinforcement learning for energy dispatch control"],
            ["time-series forecasting for retail demand"],
            ["soil microbiome metagenomics for crop resilience"],
            "false_negative_trap",
        ),
    ]
    q, pos, hard, easy, tag = templates[idx % len(templates)]
    return {
        "query": q,
        "positives": list(pos),
        "hard_negatives": list(hard),
        "easy_negatives": list(easy),
        "case_tag": tag,
        "source": "fallback",
    }


def _sanitize_probe_case(raw: Dict[str, Any], *, fallback_idx: int) -> Dict[str, Any]:
    query = _clean_text(raw.get("query"))
    positives = [_clean_text(x) for x in list(raw.get("positives") or []) if _clean_text(x)]
    hard_negatives = [_clean_text(x) for x in list(raw.get("hard_negatives") or []) if _clean_text(x)]
    easy_negatives = [_clean_text(x) for x in list(raw.get("easy_negatives") or []) if _clean_text(x)]
    case_tag = _clean_text(raw.get("case_tag")).lower()

    if not query:
        return _default_probe_case(fallback_idx)
    if not positives:
        return _default_probe_case(fallback_idx)

    if case_tag not in {"hard_negative", "false_positive_trap", "false_negative_trap"}:
        case_tag = "hard_negative"

    seen = set()

    def _uniq_keep(values: Sequence[str]) -> List[str]:
        out: List[str] = []
        for item in values:
            t = _clean_text(item)
            if not t:
                continue
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
        return out

    positives_u = _uniq_keep(positives)
    hard_u = _uniq_keep(hard_negatives)
    easy_u = _uniq_keep(easy_negatives)
    negatives_total = len(hard_u) + len(easy_u)
    if not positives_u or negatives_total < 2:
        return _default_probe_case(fallback_idx)

    return {
        "query": query,
        "positives": positives_u[:2],
        "hard_negatives": hard_u[:3],
        "easy_negatives": easy_u[:3],
        "case_tag": case_tag,
        "source": "llm",
    }


def _build_probe_chain(model_id: str):
    try:
        from langchain_core.prompts import ChatPromptTemplate
    except Exception as e:
        raise RuntimeError("Missing dependency for LLM probe generation. Install: pip install langchain-core") from e

    llm = get_llm_client(model_id).build()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROBE_GEN_SYSTEM_PROMPT),
            ("human", PROBE_GEN_HUMAN_PROMPT),
        ]
    )
    return prompt | llm.with_structured_output(ProbeCaseBatch)


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


def _generate_probe_cases(
    *,
    llm_model: str,
    case_count: int,
    seed: int,
) -> Dict[str, Any]:
    target = _safe_int(case_count, default=DEFAULT_CASE_COUNT, minimum=10, maximum=2000)
    per_call = int(PROBE_CASES_PER_LLM_CALL)
    retries = int(LLM_MAX_RETRIES)
    chain = _build_probe_chain(llm_model)
    rng = random.Random(int(seed))

    cases: List[Dict[str, Any]] = []
    llm_calls = 0
    llm_failures = 0
    fallback_count = 0

    while len(cases) < target:
        needed = target - len(cases)
        ask_count = min(per_call, needed)

        batch_items: List[ProbeCase] = []
        success = False
        for attempt in range(retries):
            llm_calls += 1
            try:
                out = chain.invoke(
                    {
                        "count": int(ask_count),
                        "seed_hint": f"{seed}-{rng.randint(0, 10_000_000)}",
                    }
                )
                batch_items = list(getattr(out, "items", []) or [])
                if batch_items:
                    success = True
                    break
            except Exception:
                continue

        if not success:
            llm_failures += 1

        if success:
            for item in batch_items:
                raw = item.model_dump() if hasattr(item, "model_dump") else dict(item or {})
                fixed = _sanitize_probe_case(raw, fallback_idx=len(cases))
                if fixed.get("source") == "fallback":
                    fallback_count += 1
                cases.append(fixed)
                if len(cases) >= target:
                    break
        else:
            while len(cases) < target and ask_count > 0:
                d = _default_probe_case(len(cases))
                fallback_count += 1
                cases.append(d)
                ask_count -= 1

    return {
        "cases": cases[:target],
        "meta": {
            "target_case_count": int(target),
            "generated_case_count": int(len(cases[:target])),
            "llm_calls": int(llm_calls),
            "llm_failures": int(llm_failures),
            "fallback_case_count": int(fallback_count),
            "llm_model": llm_model,
        },
    }


def _score_cases(
    *,
    model_dir: str,
    cases: Sequence[Dict[str, Any]],
    cpu_only: bool,
    seed: int,
) -> Dict[str, Any]:
    resolved_model_dir = _resolve_model_dir(model_dir)
    device, device_name = _select_device(cpu_only=bool(cpu_only))
    tokenizer, model = _load_model(resolved_model_dir, device, device_name)

    rng = random.Random(int(seed))
    pair_rows: List[Dict[str, Any]] = []
    for idx, case in enumerate(list(cases or [])):
        query = _clean_text(case.get("query"))
        pos = [_clean_text(x) for x in list(case.get("positives") or []) if _clean_text(x)]
        hard = [_clean_text(x) for x in list(case.get("hard_negatives") or []) if _clean_text(x)]
        easy = [_clean_text(x) for x in list(case.get("easy_negatives") or []) if _clean_text(x)]
        if not query or not pos:
            continue

        candidates: List[Dict[str, Any]] = []
        for text in pos:
            candidates.append({"text": text, "kind": "positive", "is_positive": 1})
        for text in hard:
            candidates.append({"text": text, "kind": "hard_negative", "is_positive": 0})
        for text in easy:
            candidates.append({"text": text, "kind": "easy_negative", "is_positive": 0})
        if len(candidates) < 2:
            continue

        rng.shuffle(candidates)
        for cand in candidates:
            pair_rows.append(
                {
                    "case_index": int(idx),
                    "query": query,
                    "candidate": cand["text"],
                    "kind": cand["kind"],
                    "is_positive": int(cand["is_positive"]),
                    "case_tag": _clean_text(case.get("case_tag")),
                }
            )

    scores = _score_pairs(
        tokenizer=tokenizer,
        model=model,
        device=device,
        pairs=[{"query": x["query"], "doc": x["candidate"]} for x in pair_rows],
        batch_size=int(INFER_BATCH_SIZE),
        max_length=int(INFER_MAX_LENGTH),
    )

    for row, score in zip(pair_rows, scores):
        row["score"] = float(score)
        row["confidence"] = float(_sigmoid(float(score)))

    by_case: Dict[int, List[Dict[str, Any]]] = {}
    for row in pair_rows:
        by_case.setdefault(int(row["case_index"]), []).append(dict(row))

    scored_cases: List[Dict[str, Any]] = []
    for idx, case in enumerate(list(cases or [])):
        ranked = sorted(by_case.get(int(idx), []), key=lambda x: float(x.get("score") or 0.0), reverse=True)
        if not ranked:
            continue
        scored_cases.append(
            {
                "case_index": int(idx),
                "query": _clean_text(case.get("query")),
                "case_tag": _clean_text(case.get("case_tag")),
                "source": _clean_text(case.get("source")),
                "positives": list(case.get("positives") or []),
                "hard_negatives": list(case.get("hard_negatives") or []),
                "easy_negatives": list(case.get("easy_negatives") or []),
                "ranked": ranked,
            }
        )

    return {
        "model_dir": str(resolved_model_dir),
        "device": device_name,
        "pair_count": int(len(pair_rows)),
        "scored_case_count": int(len(scored_cases)),
        "scored_cases": scored_cases,
    }


def _compute_quality_metrics(scored_cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    cases = list(scored_cases or [])
    if not cases:
        return {
            "case_count": 0,
            "top1_accuracy": 0.0,
            "mrr": 0.0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate_at_3": 0.0,
            "hard_negative_top1_rate": 0.0,
            "low_confidence_ratio": 0.0,
            "low_margin_ratio": 0.0,
            "low_margin_threshold": float(LOW_MARGIN_THRESHOLD),
            "low_confidence_threshold": float(LOW_CONFIDENCE_THRESHOLD),
        }

    n = len(cases)
    top1_correct = 0
    mrr_sum = 0.0
    fp_top1 = 0
    fn_at_3 = 0
    hard_neg_top1 = 0
    low_conf = 0
    low_margin = 0
    recall_hits = {k: 0 for k in TOP_KS}

    for case in cases:
        ranked = list(case.get("ranked") or [])
        if not ranked:
            continue

        top1 = dict(ranked[0])
        top1_is_pos = int(top1.get("is_positive") or 0) == 1
        if top1_is_pos:
            top1_correct += 1
        else:
            fp_top1 += 1
        if _clean_text(top1.get("kind")) == "hard_negative":
            hard_neg_top1 += 1

        c1 = float(top1.get("confidence") or 0.0)
        if c1 < float(LOW_CONFIDENCE_THRESHOLD):
            low_conf += 1
        if len(ranked) >= 2:
            margin = float(ranked[0].get("score") or 0.0) - float(ranked[1].get("score") or 0.0)
            if margin < float(LOW_MARGIN_THRESHOLD):
                low_margin += 1
        else:
            low_margin += 1

        first_pos_rank: Optional[int] = None
        for pos, row in enumerate(ranked, start=1):
            if int(row.get("is_positive") or 0) == 1:
                first_pos_rank = int(pos)
                break
        if first_pos_rank is not None:
            mrr_sum += 1.0 / float(first_pos_rank)

        for k in TOP_KS:
            topk = ranked[: int(k)]
            has_positive = any(int(x.get("is_positive") or 0) == 1 for x in topk)
            if has_positive:
                recall_hits[int(k)] += 1

        if not any(int(x.get("is_positive") or 0) == 1 for x in ranked[:3]):
            fn_at_3 += 1

    return {
        "case_count": int(n),
        "top1_accuracy": float(top1_correct) / float(n),
        "mrr": float(mrr_sum) / float(n),
        "recall_at_1": float(recall_hits[1]) / float(n),
        "recall_at_3": float(recall_hits[3]) / float(n),
        "recall_at_5": float(recall_hits[5]) / float(n),
        "false_positive_rate": float(fp_top1) / float(n),
        "false_negative_rate_at_3": float(fn_at_3) / float(n),
        "hard_negative_top1_rate": float(hard_neg_top1) / float(n),
        "low_confidence_ratio": float(low_conf) / float(n),
        "low_margin_ratio": float(low_margin) / float(n),
        "low_margin_threshold": float(LOW_MARGIN_THRESHOLD),
        "low_confidence_threshold": float(LOW_CONFIDENCE_THRESHOLD),
    }


def _select_failure_samples(scored_cases: Sequence[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for case in list(scored_cases or []):
        ranked = list(case.get("ranked") or [])
        if not ranked:
            continue
        top1 = dict(ranked[0])
        top1_is_pos = int(top1.get("is_positive") or 0) == 1
        top2_score = float(ranked[1].get("score") or 0.0) if len(ranked) >= 2 else float(top1.get("score") or 0.0)
        margin = float(top1.get("score") or 0.0) - top2_score
        low_conf = float(top1.get("confidence") or 0.0) < float(LOW_CONFIDENCE_THRESHOLD)
        low_margin = margin < float(LOW_MARGIN_THRESHOLD)

        if top1_is_pos and not low_conf and not low_margin:
            continue

        failures.append(
            {
                "query": _clean_text(case.get("query")),
                "case_tag": _clean_text(case.get("case_tag")),
                "top1_is_positive": bool(top1_is_pos),
                "top1_kind": _clean_text(top1.get("kind")),
                "top1_score": float(top1.get("score") or 0.0),
                "top1_confidence": float(top1.get("confidence") or 0.0),
                "top1_text": _clean_text(top1.get("candidate")),
                "margin_top1_top2": float(margin),
                "expected_positives": list(case.get("positives") or []),
                "top_ranked_preview": [
                    {
                        "candidate": _clean_text(x.get("candidate")),
                        "kind": _clean_text(x.get("kind")),
                        "is_positive": int(x.get("is_positive") or 0),
                        "score": float(x.get("score") or 0.0),
                    }
                    for x in ranked[:5]
                ],
            }
        )

    failures.sort(key=lambda x: (x["top1_is_positive"], x["top1_confidence"], x["margin_top1_top2"]))
    return failures[: max(1, int(sample_size))]


def _llm_quality_review(
    *,
    llm_model: str,
    metrics: Dict[str, Any],
    failures: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    def _fallback_review(error_text: str) -> Dict[str, Any]:
        fp = float(metrics.get("false_positive_rate") or 0.0)
        fn = float(metrics.get("false_negative_rate_at_3") or 0.0)
        hard = float(metrics.get("hard_negative_top1_rate") or 0.0)
        margin = float(metrics.get("low_margin_ratio") or 0.0)
        conf = float(metrics.get("low_confidence_ratio") or 0.0)

        patterns: List[str] = []
        if hard >= 0.20:
            patterns.append("Model frequently ranks hard negatives at top-1 for near-topic candidates.")
        if fp >= 0.30:
            patterns.append("False-positive pressure is high; lexical overlap likely dominates semantic fit.")
        if fn >= 0.30:
            patterns.append("False negatives are frequent in top-3; relevant candidates are not reliably surfaced.")
        if margin >= 0.35:
            patterns.append("Decision margins are often small, indicating ambiguous ranking boundaries.")
        if conf >= 0.35:
            patterns.append("Top predictions are often low-confidence, suggesting weak calibration.")
        if not patterns:
            patterns.append("No single dominant error pattern; failures appear distributed across case types.")

        return {
            "overall_summary": (
                "LLM structured quality review fell back to deterministic heuristics due to parse error. "
                f"Root error: {error_text}"
            ),
            "likely_failure_patterns": patterns,
            "dataset_fix_actions": [
                "Increase hard-negative cases that share domain terms but differ in research objective.",
                "Add targeted false-negative probes with paraphrased positives and low lexical overlap.",
                "Balance probe case tags across hard_negative, false_positive_trap, and false_negative_trap.",
            ],
            "training_fix_actions": [
                "Upweight hard-negative pairs or oversample them in training batches.",
                "Use shorter eval intervals to catch early overfitting and stop sooner.",
                "Tune learning rate/epochs conservatively when loss drops too quickly.",
            ],
            "model": llm_model,
            "fallback": "deterministic_heuristic",
        }

    if not failures:
        return {
            "overall_summary": "No major failure examples were detected in the sampled probes.",
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
        review = out.model_dump() if hasattr(out, "model_dump") else dict(out or {})
        review["likely_failure_patterns"] = _to_text_list(review.get("likely_failure_patterns"))
        review["dataset_fix_actions"] = _to_text_list(review.get("dataset_fix_actions"))
        review["training_fix_actions"] = _to_text_list(review.get("training_fix_actions"))
        review["model"] = llm_model
        return review
    except Exception as e:
        return _fallback_review(f"{type(e).__name__}: {e}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_llm_quality_eval(
    *,
    model_dir: str,
    llm_model: str,
    num_cases: int,
    output_dir: Path,
    cpu_only: bool,
    seed: int,
) -> Dict[str, Any]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir.expanduser().resolve() / f"llm_quality_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    probe_result = _generate_probe_cases(
        llm_model=llm_model,
        case_count=num_cases,
        seed=seed,
    )
    probe_cases = list(probe_result.get("cases") or [])

    scored = _score_cases(
        model_dir=model_dir,
        cases=probe_cases,
        cpu_only=cpu_only,
        seed=seed,
    )
    scored_cases = list(scored.get("scored_cases") or [])

    metrics = _compute_quality_metrics(scored_cases)
    failures = _select_failure_samples(scored_cases, FAILURE_SAMPLE_SIZE)
    review = _llm_quality_review(
        llm_model=llm_model,
        metrics=metrics,
        failures=failures,
    )

    generated_path = run_dir / "probes_generated.json"
    scored_path = run_dir / "probes_scored.json"
    summary_path = run_dir / "quality_summary.json"

    _write_json(
        generated_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "meta": probe_result.get("meta", {}),
            "cases": probe_cases,
        },
    )
    _write_json(
        scored_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_dir": scored.get("model_dir"),
            "device": scored.get("device"),
            "pair_count": scored.get("pair_count"),
            "scored_case_count": scored.get("scored_case_count"),
            "cases": scored_cases,
        },
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "model_dir": scored.get("model_dir"),
        "device": scored.get("device"),
        "probe_generation": probe_result.get("meta", {}),
        "metrics": metrics,
        "failure_examples_count": len(failures),
        "failure_examples_preview": failures[: min(5, len(failures))],
        "llm_quality_review": review,
        "files": {
            "generated_probes": str(generated_path),
            "scored_probes": str(scored_path),
            "summary": str(summary_path),
        },
    }
    _write_json(summary_path, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    default_out = Path(__file__).resolve().parent / "eval_runs"
    parser = argparse.ArgumentParser(
        description=(
            "Step-3 quality check: generate LLM probe cases, run final model inference, "
            "and produce deterministic + LLM quality report."
        )
    )
    parser.add_argument("--model-dir", type=str, default="", help="Final model directory. Default: latest under train_cross_encoder/models.")
    parser.add_argument("--llm-model", type=str, default=(settings.haiku or "").strip(), help="Bedrock model id used for probe generation and quality review.")
    parser.add_argument("--num-cases", type=int, default=DEFAULT_CASE_COUNT, help="Number of LLM-generated probe cases.")
    parser.add_argument("--output-dir", type=str, default=str(default_out), help="Base output directory for eval artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU even if CUDA/MPS is available.")
    parser.add_argument("--json-only", action="store_true", help="Print JSON only.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    summary = run_llm_quality_eval(
        model_dir=_clean_text(args.model_dir),
        llm_model=_clean_text(args.llm_model) or (settings.haiku or "").strip(),
        num_cases=int(args.num_cases),
        output_dir=Path(_clean_text(args.output_dir)),
        cpu_only=bool(args.cpu_only),
        seed=int(args.seed),
    )

    if not bool(args.json_only):
        metrics = dict(summary.get("metrics") or {})
        print("LLM quality evaluation complete.")
        print(f"  run dir                 : {summary.get('run_dir', '')}")
        print(f"  model dir               : {summary.get('model_dir', '')}")
        print(f"  device                  : {summary.get('device', '')}")
        print(f"  probe cases             : {summary.get('probe_generation', {}).get('generated_case_count', 0)}")
        print(f"  top1 accuracy           : {metrics.get('top1_accuracy', 0.0):.4f}")
        print(f"  false positive rate     : {metrics.get('false_positive_rate', 0.0):.4f}")
        print(f"  false negative rate@3   : {metrics.get('false_negative_rate_at_3', 0.0):.4f}")
        print(f"  hard negative top1 rate : {metrics.get('hard_negative_top1_rate', 0.0):.4f}")
        print(f"  low confidence ratio    : {metrics.get('low_confidence_ratio', 0.0):.4f}")
        print(f"  low margin ratio        : {metrics.get('low_margin_ratio', 0.0):.4f}")
        print(f"  summary file            : {summary.get('files', {}).get('summary', '')}")
        print()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
