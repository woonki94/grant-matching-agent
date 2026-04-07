from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, TypedDict

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


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


def _safe_float(value: Any, *, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    return out


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@lru_cache(maxsize=1)
def _load_build_module():
    try:
        return importlib.import_module("train_cross_encoder.build_llm_spec_pair_dataset")
    except Exception as e:
        raise RuntimeError(
            "Failed to import dataset builder module. Use the project venv "
            "(for example `venv2/bin/python`)."
        ) from e


@lru_cache(maxsize=1)
def _load_train_module():
    try:
        return importlib.import_module("train_cross_encoder.train_bge_reranker_v2_simple")
    except Exception as e:
        raise RuntimeError(
            "Failed to import v2 training module. Use the project venv "
            "(for example `venv2/bin/python`)."
        ) from e


@lru_cache(maxsize=1)
def _load_infer_module():
    try:
        return importlib.import_module("train_cross_encoder.infer_bge_reranker")
    except Exception as e:
        raise RuntimeError(
            "Failed to import inference module. Use the project venv "
            "(for example `venv2/bin/python`)."
        ) from e


@lru_cache(maxsize=1)
def _load_config_module():
    try:
        return importlib.import_module("config")
    except Exception as e:
        raise RuntimeError(
            "Failed to import config module. Use the project venv "
            "(for example `venv2/bin/python`)."
        ) from e


def _resolve_default_llm_model() -> str:
    try:
        cfg = _load_config_module()
        return _clean_text(getattr(cfg.settings, "haiku", ""))
    except Exception:
        return ""


@dataclass
class DatasetTuningConfig:
    top_k_candidates: int = 8
    hard_negatives_per_grant: int = 10
    random_negatives_per_grant: int = 10
    candidates_per_query: int = 20
    max_queries: int = 5000
    max_pairs: int = 200000
    llm_batch_size: int = 8
    llm_model: str = ""
    llm_max_retries: int = 2
    llm_max_workers: int = 8
    faculty_min_spec_weight: float = 0.0
    grant_min_spec_weight: float = 0.0
    faculty_limit: int = 200000
    grant_limit: int = 200000
    embed_batch_size: int = 64
    seed: int = 42


@dataclass
class LoopConfig:
    max_iterations: int = 3
    target_eval_loss: float = 0.045
    target_low_conf_ratio: float = 0.25
    target_low_margin_ratio: float = 0.35
    target_false_positive_rate: float = 0.30
    target_false_negative_rate: float = 0.30
    target_hard_negative_top1_rate: float = 0.25
    probe_query_count: int = 24
    probe_candidate_count: int = 120
    probe_top_k: int = 5
    low_conf_threshold: float = 0.45
    low_margin_threshold: float = 0.05
    use_wandb: bool = True


class FinetuneWorkflowState(TypedDict, total=False):
    run_dir: str
    iteration: int
    max_iterations: int
    dataset_cfg: Dict[str, Any]
    loop_cfg: Dict[str, Any]
    dataset_payload: Dict[str, Any]
    dataset_jsonl: str
    model_dir: str
    train_summary: Dict[str, Any]
    probe_summary: Dict[str, Any]
    history: List[Dict[str, Any]]
    best_eval_loss: float | None
    best_iteration: int
    next_action: str
    stop_reason: str
    error: str
    result: Dict[str, Any]


class FinetuneTools:
    """Tool wrappers used by graph nodes."""

    @staticmethod
    def build_llm_spec_pair_dataset(
        *,
        cfg: DatasetTuningConfig,
        output_dir: Path,
        output_prefix: str,
    ) -> Dict[str, Any]:
        mod = _load_build_module()
        payload = mod.build_dataset(
            top_k_candidates=int(cfg.top_k_candidates),
            hard_negatives_per_grant=int(cfg.hard_negatives_per_grant),
            random_negatives_per_grant=int(cfg.random_negatives_per_grant),
            candidates_per_query=int(cfg.candidates_per_query),
            max_queries=int(cfg.max_queries),
            max_pairs=int(cfg.max_pairs),
            llm_batch_size=int(cfg.llm_batch_size),
            llm_model=_clean_text(cfg.llm_model),
            llm_max_retries=int(cfg.llm_max_retries),
            llm_max_workers=int(cfg.llm_max_workers),
            faculty_min_spec_weight=float(cfg.faculty_min_spec_weight),
            grant_min_spec_weight=float(cfg.grant_min_spec_weight),
            faculty_limit=int(cfg.faculty_limit),
            grant_limit=int(cfg.grant_limit),
            embed_batch_size=int(cfg.embed_batch_size),
            seed=int(cfg.seed),
            output_dir=output_dir,
            output_prefix=output_prefix,
        )
        return dict(payload)

    @staticmethod
    def train_v2_simple(
        *,
        dataset_jsonl: Path,
        output_dir: Path,
        use_wandb: bool,
    ) -> Dict[str, Any]:
        mod = _load_train_module()
        payload = mod.run_train(
            dataset_jsonl=dataset_jsonl,
            output_dir=output_dir,
            use_wandb=bool(use_wandb),
        )
        return dict(payload)

    @staticmethod
    def infer_scores(
        *,
        model_dir: Path,
        query: str,
        candidates: Sequence[str],
        top_k: int,
    ) -> Dict[str, Any]:
        mod = _load_infer_module()
        payload = mod.run_inference(
            model_dir=str(model_dir),
            query=_clean_text(query),
            candidates=list(candidates),
            candidates_file="",
            input_jsonl="",
            output_jsonl="",
            batch_size=64,
            max_length=256,
            top_k=int(max(1, top_k)),
            cpu_only=False,
        )
        return dict(payload)

    @staticmethod
    def fetch_faculty_specs(*, min_spec_weight: float, limit: int) -> List[Any]:
        mod = _load_build_module()
        return list(mod._fetch_faculty_specs(min_spec_weight=float(min_spec_weight), limit=int(limit)))

    @staticmethod
    def fetch_grant_specs(*, min_spec_weight: float, limit: int) -> List[Any]:
        mod = _load_build_module()
        return list(mod._fetch_grant_specs(min_spec_weight=float(min_spec_weight), limit=int(limit)))


class PlanRouter:
    """Planner/router for the graph flow."""

    @staticmethod
    def _passes_targets(
        *,
        train_summary: Dict[str, Any],
        probe_summary: Dict[str, Any],
        loop_cfg: LoopConfig,
    ) -> bool:
        eval_loss = train_summary.get("eval_loss")
        eval_loss_val = None if eval_loss is None else _safe_float(eval_loss, default=999.0)
        low_conf_ratio = _safe_float(probe_summary.get("low_conf_ratio"), default=1.0)
        low_margin_ratio = _safe_float(probe_summary.get("low_margin_ratio"), default=1.0)
        false_positive_rate = _safe_float(probe_summary.get("false_positive_rate"), default=1.0)
        false_negative_rate = _safe_float(probe_summary.get("false_negative_rate"), default=1.0)
        hard_negative_top1_rate = _safe_float(probe_summary.get("hard_negative_top1_rate"), default=1.0)
        pass_eval = bool(eval_loss_val is not None and eval_loss_val <= float(loop_cfg.target_eval_loss))
        pass_conf = bool(low_conf_ratio <= float(loop_cfg.target_low_conf_ratio))
        pass_margin = bool(low_margin_ratio <= float(loop_cfg.target_low_margin_ratio))
        pass_fp = bool(false_positive_rate <= float(loop_cfg.target_false_positive_rate))
        pass_fn = bool(false_negative_rate <= float(loop_cfg.target_false_negative_rate))
        pass_hard = bool(hard_negative_top1_rate <= float(loop_cfg.target_hard_negative_top1_rate))
        return bool(pass_eval and pass_conf and pass_margin and pass_fp and pass_fn and pass_hard)

    def decide(self, state: FinetuneWorkflowState) -> Dict[str, Any]:
        if _clean_text(state.get("error")):
            return {"next_action": "finish", "stop_reason": "error"}

        raw_iteration = state.get("iteration")
        raw_max_iterations = state.get("max_iterations")
        iteration = int(raw_iteration) if raw_iteration is not None else 1
        max_iterations = int(raw_max_iterations) if raw_max_iterations is not None else 1
        loop_cfg = LoopConfig(**dict(state.get("loop_cfg") or {}))

        if iteration > max_iterations:
            return {"next_action": "finish", "stop_reason": "max_iterations"}

        if not _clean_text(state.get("dataset_jsonl")):
            return {"next_action": "tool_build_dataset", "stop_reason": ""}
        if not dict(state.get("train_summary") or {}):
            return {"next_action": "tool_train_v2_simple", "stop_reason": ""}
        if not dict(state.get("probe_summary") or {}):
            return {"next_action": "tool_probe_quality", "stop_reason": ""}

        if self._passes_targets(
            train_summary=dict(state.get("train_summary") or {}),
            probe_summary=dict(state.get("probe_summary") or {}),
            loop_cfg=loop_cfg,
        ):
            return {"next_action": "finish", "stop_reason": "satisfactory"}

        if iteration >= max_iterations:
            return {"next_action": "finish", "stop_reason": "max_iterations"}

        return {"next_action": "retune_and_advance", "stop_reason": "continue"}


def _extract_json_object(text: str) -> str:
    s = _clean_text(text)
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        return s[i : j + 1]
    return ""


def _tokenize(text: str) -> set[str]:
    return set(x for x in re.findall(r"[a-z0-9]+", _clean_text(text).lower()) if x)


def _token_overlap(a: str, b: str) -> float:
    aa = _tokenize(a)
    bb = _tokenize(b)
    if not aa or not bb:
        return 0.0
    return float(len(aa & bb)) / float(len(aa | bb))


def _normalize_probe_case_from_indices(
    *,
    case: Dict[str, Any],
    candidate_pool: Sequence[str],
    query_fallback: str,
    tag_fallback: str,
) -> Dict[str, Any]:
    query = _clean_text(case.get("query")) or _clean_text(query_fallback)
    case_tag = _clean_text(case.get("case_tag")).lower() or _clean_text(tag_fallback)
    if case_tag not in {"hard_negative", "false_positive_trap", "false_negative_trap"}:
        case_tag = _clean_text(tag_fallback) or "hard_negative"

    n = len(candidate_pool)
    if n <= 0:
        return {}

    try:
        pos_idx = int(case.get("positive_idx"))
    except Exception:
        pos_idx = -1
    if pos_idx < 0 or pos_idx >= n:
        pos_idx = 0

    def _idx_list(raw: Any) -> List[int]:
        out: List[int] = []
        for item in list(raw or []):
            try:
                idx = int(item)
            except Exception:
                continue
            if 0 <= idx < n and idx not in out and idx != pos_idx:
                out.append(idx)
        return out

    hard_idxs = _idx_list(case.get("hard_negative_idxs"))
    easy_idxs = _idx_list(case.get("easy_negative_idxs"))

    used = {pos_idx, *hard_idxs, *easy_idxs}
    fallback_pool = [i for i in range(n) if i not in used]
    if not hard_idxs and fallback_pool:
        hard_idxs.append(fallback_pool.pop(0))
    if not easy_idxs and fallback_pool:
        easy_idxs.append(fallback_pool.pop(0))

    return {
        "query": query,
        "case_tag": case_tag,
        "positives": [candidate_pool[pos_idx]],
        "hard_negatives": [candidate_pool[i] for i in hard_idxs[:4]],
        "easy_negatives": [candidate_pool[i] for i in easy_idxs[:4]],
    }


def _build_fallback_probe_cases(
    *,
    n: int,
    seed: int,
    grant_seed_texts: Sequence[str],
    candidate_pool: Sequence[str],
) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed))
    cands = list(candidate_pool or [])
    seeds = [_clean_text(x) for x in list(grant_seed_texts or []) if _clean_text(x)]
    if not cands:
        return []

    if not seeds:
        seeds = list(cands)
    rng.shuffle(seeds)
    tags = ["hard_negative", "false_positive_trap", "false_negative_trap"]
    out: List[Dict[str, Any]] = []
    for i in range(int(max(1, n))):
        query = _clean_text(seeds[i % len(seeds)])
        scores = [(idx, _token_overlap(query, cand)) for idx, cand in enumerate(cands)]
        scores.sort(key=lambda x: x[1], reverse=True)

        pos_idx = int(scores[0][0]) if scores else 0
        hard_idxs = [idx for idx, sc in scores[1:8] if sc > 0.0 and idx != pos_idx][:2]
        if not hard_idxs:
            hard_idxs = [idx for idx, _ in scores[1:4] if idx != pos_idx][:2]

        low_scores = [idx for idx, sc in sorted(scores, key=lambda x: x[1]) if idx != pos_idx and idx not in hard_idxs]
        easy_idxs = low_scores[:2]

        case = _normalize_probe_case_from_indices(
            case={
                "query": query,
                "case_tag": tags[i % len(tags)],
                "positive_idx": pos_idx,
                "hard_negative_idxs": hard_idxs,
                "easy_negative_idxs": easy_idxs,
            },
            candidate_pool=cands,
            query_fallback=query,
            tag_fallback=tags[i % len(tags)],
        )
        if case:
            out.append(case)
    return out


def _generate_probe_cases(
    *,
    tools: FinetuneTools,
    n: int,
    seed: int,
    llm_model: str,
    grant_seed_texts: Sequence[str],
    candidate_texts: Sequence[str],
) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    all_candidates = list(dict.fromkeys([_clean_text(x) for x in list(candidate_texts or []) if _clean_text(x)]))
    if not all_candidates:
        return {"cases": [], "source": "none"}

    rng.shuffle(all_candidates)
    candidate_pool = all_candidates[: min(len(all_candidates), 80)]
    seed_topics = list(dict.fromkeys([_clean_text(x) for x in list(grant_seed_texts or []) if _clean_text(x)]))
    rng.shuffle(seed_topics)
    seed_topics = seed_topics[:40]

    target_n = int(max(1, n))
    llm_cases: List[Dict[str, Any]] = []
    if _clean_text(llm_model) and seed_topics and candidate_pool:
        prompt = (
            "Create stress-test probe cases for a cross-encoder reranker.\n"
            "We need to evaluate hard negatives, false-positive traps, and false-negative traps.\n"
            f"Generate exactly {target_n} cases.\n"
            "Return ONLY JSON with schema:\n"
            "{\n"
            '  "cases": [\n'
            "    {\n"
            '      "query": string,\n'
            '      "case_tag": "hard_negative" | "false_positive_trap" | "false_negative_trap",\n'
            '      "positive_idx": integer,\n'
            '      "hard_negative_idxs": [integer,...],\n'
            '      "easy_negative_idxs": [integer,...]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- Use candidate indices ONLY from the list.\n"
            "- positive_idx must be a true relevant candidate for the query.\n"
            "- hard_negative_idxs must be semantically close but less relevant than positive.\n"
            "- easy_negative_idxs must be clearly irrelevant.\n"
            "- Do not overlap indices across positive/hard/easy inside a case.\n"
            "- Cover all three case_tag values across the output.\n\n"
            "Seed topics:\n"
            + "\n".join(f"- {x}" for x in seed_topics)
            + "\n\nCandidate index -> text:\n"
            + "\n".join(f"{i}: {t}" for i, t in enumerate(candidate_pool))
        )
        try:
            cfg = _load_config_module()
            llm = cfg.get_llm_client(_clean_text(llm_model)).build()
            res = llm.invoke(prompt)
            raw = getattr(res, "content", None)
            text = str(raw if raw is not None else res)
            obj = json.loads(_extract_json_object(text) or "{}")
            parsed_cases = list(dict(obj or {}).get("cases") or [])
            for i, c in enumerate(parsed_cases):
                if not isinstance(c, dict):
                    continue
                normalized = _normalize_probe_case_from_indices(
                    case=c,
                    candidate_pool=candidate_pool,
                    query_fallback=seed_topics[i % len(seed_topics)] if seed_topics else candidate_pool[0],
                    tag_fallback=["hard_negative", "false_positive_trap", "false_negative_trap"][i % 3],
                )
                if normalized:
                    llm_cases.append(normalized)
        except Exception:
            llm_cases = []

    fallback_cases = _build_fallback_probe_cases(
        n=target_n,
        seed=int(seed),
        grant_seed_texts=seed_topics,
        candidate_pool=candidate_pool,
    )
    required_tags = {"hard_negative", "false_positive_trap", "false_negative_trap"}
    llm_tags = set(_clean_text(c.get("case_tag")).lower() for c in llm_cases if isinstance(c, dict))
    if len(llm_cases) >= target_n and required_tags.issubset(llm_tags):
        return {"cases": llm_cases[:target_n], "source": "llm"}
    if not llm_cases:
        return {"cases": fallback_cases[:target_n], "source": "fallback"}

    out = list(llm_cases)
    existing_tags = set(_clean_text(c.get("case_tag")).lower() for c in out if isinstance(c, dict))
    missing_tags = [t for t in sorted(required_tags) if t not in existing_tags]
    for tag in missing_tags:
        for c in fallback_cases:
            if _clean_text(c.get("case_tag")).lower() != tag:
                continue
            out.append(c)
            break
    for c in fallback_cases:
        if len(out) >= target_n:
            break
        out.append(c)
    return {"cases": out[:target_n], "source": "hybrid"}


def _probe_inference(
    *,
    tools: FinetuneTools,
    model_dir: Path,
    cases: Sequence[Dict[str, Any]],
    top_k: int,
    low_conf_threshold: float,
    low_margin_threshold: float,
    case_source: str,
) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    low_conf = 0
    low_margin = 0
    false_positive = 0
    false_negative = 0
    hard_negative_top1 = 0
    total = 0
    tag_counts: Dict[str, int] = {}

    for case in list(cases or []):
        query = _clean_text(case.get("query"))
        positives = list(dict.fromkeys([_clean_text(x) for x in list(case.get("positives") or []) if _clean_text(x)]))
        hard_negs = list(dict.fromkeys([_clean_text(x) for x in list(case.get("hard_negatives") or []) if _clean_text(x)]))
        easy_negs = list(dict.fromkeys([_clean_text(x) for x in list(case.get("easy_negatives") or []) if _clean_text(x)]))
        case_tag = _clean_text(case.get("case_tag")).lower() or "hard_negative"

        if not query or not positives:
            continue
        candidates = list(dict.fromkeys([*positives, *hard_negs, *easy_negs]))
        if len(candidates) < 2:
            continue

        payload = tools.infer_scores(
            model_dir=model_dir,
            query=query,
            candidates=candidates,
            top_k=max(int(top_k), len(candidates)),
        )
        ranked = list(payload.get("ranked") or [])
        if not ranked:
            continue

        total += 1
        tag_counts[case_tag] = int(tag_counts.get(case_tag, 0)) + 1

        top1_row = ranked[0] or {}
        top2_row = ranked[1] or top1_row if len(ranked) > 1 else top1_row
        top1_candidate = _clean_text(top1_row.get("candidate"))
        top1_score = float(top1_row.get("score") or 0.0)
        top2_score = float(top2_row.get("score") or top1_score)
        margin = float(top1_score - top2_score)

        positive_set = set(positives)
        hard_set = set(hard_negs)
        negative_set = set(hard_negs) | set(easy_negs)
        score_by_candidate = {_clean_text(r.get("candidate")): float(r.get("score") or 0.0) for r in ranked}
        positive_rank = len(ranked) + 1
        for idx, row in enumerate(ranked, start=1):
            if _clean_text(row.get("candidate")) in positive_set:
                positive_rank = idx
                break
        positive_score = max([float(score_by_candidate.get(c, -1e9)) for c in positive_set] or [-1e9])

        is_low_conf = bool(top1_score < float(low_conf_threshold))
        is_low_margin = bool(margin < float(low_margin_threshold))
        is_false_positive = bool(top1_candidate in negative_set and top1_score >= float(low_conf_threshold))
        is_false_negative = bool(positive_rank > 1 or positive_score < float(low_conf_threshold))
        is_hard_negative_top1 = bool(top1_candidate in hard_set)

        if is_low_conf:
            low_conf += 1
        if is_low_margin:
            low_margin += 1
        if is_false_positive:
            false_positive += 1
        if is_false_negative:
            false_negative += 1
        if is_hard_negative_top1:
            hard_negative_top1 += 1

        details.append(
            {
                "query": query,
                "case_tag": case_tag,
                "top_candidate": top1_candidate,
                "top1_score": top1_score,
                "top2_score": top2_score,
                "margin": margin,
                "positive_rank": int(positive_rank),
                "positive_score": float(positive_score),
                "low_conf": is_low_conf,
                "low_margin": is_low_margin,
                "false_positive": is_false_positive,
                "false_negative": is_false_negative,
                "hard_negative_top1": is_hard_negative_top1,
            }
        )

    if total <= 0:
        return {
            "probe_case_source": case_source,
            "probe_cases_used": 0,
            "probe_case_tag_counts": {},
            "low_conf_count": 0,
            "low_margin_count": 0,
            "false_positive_count": 0,
            "false_negative_count": 0,
            "hard_negative_top1_count": 0,
            "low_conf_ratio": 1.0,
            "low_margin_ratio": 1.0,
            "false_positive_rate": 1.0,
            "false_negative_rate": 1.0,
            "hard_negative_top1_rate": 1.0,
            "quality_score": 0.0,
            "quality_decider": {
                "type": "rule_based",
                "note": "No probe cases evaluated.",
            },
            "examples": [],
        }

    details = sorted(
        details,
        key=lambda x: (
            not bool(x.get("false_negative")),
            not bool(x.get("false_positive")),
            not bool(x.get("hard_negative_top1")),
            float(x.get("top1_score") or 0.0),
        ),
    )
    low_conf_ratio = float(low_conf) / float(total)
    low_margin_ratio = float(low_margin) / float(total)
    false_positive_rate = float(false_positive) / float(total)
    false_negative_rate = float(false_negative) / float(total)
    hard_negative_top1_rate = float(hard_negative_top1) / float(total)
    penalty = (
        0.35 * false_positive_rate
        + 0.35 * false_negative_rate
        + 0.20 * hard_negative_top1_rate
        + 0.05 * low_conf_ratio
        + 0.05 * low_margin_ratio
    )
    quality_score = max(0.0, 1.0 - penalty)

    return {
        "probe_case_source": case_source,
        "probe_cases_used": int(total),
        "probe_case_tag_counts": dict(tag_counts),
        "low_conf_count": int(low_conf),
        "low_margin_count": int(low_margin),
        "false_positive_count": int(false_positive),
        "false_negative_count": int(false_negative),
        "hard_negative_top1_count": int(hard_negative_top1),
        "low_conf_ratio": low_conf_ratio,
        "low_margin_ratio": low_margin_ratio,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "hard_negative_top1_rate": hard_negative_top1_rate,
        "quality_score": quality_score,
        "quality_decider": {
            "type": "rule_based",
            "source_labels": "llm_generated_probe_cases_or_fallback",
            "false_positive_rule": "top1 is negative and top1_score >= low_conf_threshold",
            "false_negative_rule": "positive_rank > 1 or positive_score < low_conf_threshold",
            "hard_negative_top1_rule": "top1 is from hard_negative set",
        },
        "examples": details[:30],
    }


def _tune_dataset_config(
    *,
    cfg: DatasetTuningConfig,
    train_summary: Dict[str, Any],
    probe_summary: Dict[str, Any],
    prev_best_eval_loss: float | None,
) -> DatasetTuningConfig:
    out = replace(cfg)

    eval_loss = train_summary.get("eval_loss")
    train_loss = train_summary.get("train_loss")
    eval_loss = None if eval_loss is None else _safe_float(eval_loss, default=999.0)
    train_loss = _safe_float(train_loss, default=0.0)
    low_conf_ratio = _safe_float(probe_summary.get("low_conf_ratio"), default=1.0)
    low_margin_ratio = _safe_float(probe_summary.get("low_margin_ratio"), default=1.0)
    false_positive_rate = _safe_float(probe_summary.get("false_positive_rate"), default=1.0)
    false_negative_rate = _safe_float(probe_summary.get("false_negative_rate"), default=1.0)
    hard_negative_top1_rate = _safe_float(probe_summary.get("hard_negative_top1_rate"), default=1.0)

    if low_conf_ratio > 0.40:
        out.top_k_candidates = min(64, out.top_k_candidates + 2)
        out.hard_negatives_per_grant = min(64, out.hard_negatives_per_grant + 2)
    if low_margin_ratio > 0.40:
        out.candidates_per_query = min(64, out.candidates_per_query + 4)
        out.hard_negatives_per_grant = min(64, out.hard_negatives_per_grant + 2)
    if false_positive_rate > 0.30:
        out.hard_negatives_per_grant = min(64, out.hard_negatives_per_grant + 2)
        out.random_negatives_per_grant = min(64, out.random_negatives_per_grant + 2)
    if false_negative_rate > 0.30:
        out.top_k_candidates = min(64, out.top_k_candidates + 2)
        out.max_queries = min(25000, out.max_queries + 1000)
    if hard_negative_top1_rate > 0.25:
        out.hard_negatives_per_grant = min(64, out.hard_negatives_per_grant + 3)
    if eval_loss is not None and (train_loss + 0.01) < eval_loss:
        out.max_queries = min(20000, out.max_queries + 1000)
        out.random_negatives_per_grant = min(64, out.random_negatives_per_grant + 2)
    if eval_loss is not None and prev_best_eval_loss is not None and eval_loss > (prev_best_eval_loss * 1.02):
        out.random_negatives_per_grant = min(64, out.random_negatives_per_grant + 2)
        out.max_queries = min(25000, out.max_queries + 1500)

    out.hard_negatives_per_grant = _safe_int(out.hard_negatives_per_grant, default=10, minimum=0, maximum=64)
    out.random_negatives_per_grant = _safe_int(out.random_negatives_per_grant, default=10, minimum=0, maximum=64)
    out.top_k_candidates = _safe_int(out.top_k_candidates, default=8, minimum=1, maximum=64)
    out.candidates_per_query = _safe_int(out.candidates_per_query, default=20, minimum=2, maximum=128)
    if out.candidates_per_query < (out.top_k_candidates + out.hard_negatives_per_grant):
        out.candidates_per_query = min(128, out.top_k_candidates + out.hard_negatives_per_grant + 2)
    return out


class AgenticFinetuneGraph:
    def __init__(
        self,
        *,
        tools: FinetuneTools | None = None,
        router: PlanRouter | None = None,
    ):
        self.tools = tools or FinetuneTools()
        self.router = router or PlanRouter()
        self.graph = self._build_graph()

    @staticmethod
    def _edge_from_route(state: FinetuneWorkflowState) -> str:
        action = _clean_text(state.get("next_action"))
        return action or "finish"

    def _node_plan_route(self, state: FinetuneWorkflowState) -> FinetuneWorkflowState:
        out = self.router.decide(state)
        return {
            "next_action": _clean_text(out.get("next_action")),
            "stop_reason": _clean_text(out.get("stop_reason")),
        }

    def _node_tool_build_dataset(self, state: FinetuneWorkflowState) -> FinetuneWorkflowState:
        try:
            run_dir = Path(_clean_text(state.get("run_dir"))).resolve()
            iteration = int(state.get("iteration") or 1)
            dataset_cfg = DatasetTuningConfig(**dict(state.get("dataset_cfg") or {}))
            iter_dir = run_dir / f"iter_{iteration:02d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            payload = self.tools.build_llm_spec_pair_dataset(
                cfg=dataset_cfg,
                output_dir=iter_dir / "dataset",
                output_prefix=f"agentic_iter{iteration:02d}",
            )
            dataset_jsonl = _clean_text(dict(payload.get("output") or {}).get("jsonl_path"))
            return {
                "dataset_payload": payload,
                "dataset_jsonl": dataset_jsonl,
                "model_dir": str((iter_dir / "model").resolve()),
                "train_summary": {},
                "probe_summary": {},
                "error": "",
            }
        except Exception as e:
            return {"error": f"tool_build_dataset failed: {e}"}

    def _node_tool_train_v2_simple(self, state: FinetuneWorkflowState) -> FinetuneWorkflowState:
        try:
            dataset_jsonl = Path(_clean_text(state.get("dataset_jsonl"))).resolve()
            model_dir = Path(_clean_text(state.get("model_dir"))).resolve()
            loop_cfg = LoopConfig(**dict(state.get("loop_cfg") or {}))
            payload = self.tools.train_v2_simple(
                dataset_jsonl=dataset_jsonl,
                output_dir=model_dir,
                use_wandb=bool(loop_cfg.use_wandb),
            )
            return {"train_summary": payload, "error": ""}
        except Exception as e:
            return {"error": f"tool_train_v2_simple failed: {e}"}

    def _node_tool_probe_quality(self, state: FinetuneWorkflowState) -> FinetuneWorkflowState:
        try:
            run_dir = Path(_clean_text(state.get("run_dir"))).resolve()
            iteration = int(state.get("iteration") or 1)
            dataset_cfg = DatasetTuningConfig(**dict(state.get("dataset_cfg") or {}))
            loop_cfg = LoopConfig(**dict(state.get("loop_cfg") or {}))
            model_dir = Path(_clean_text(state.get("model_dir"))).resolve()

            faculty_specs = self.tools.fetch_faculty_specs(min_spec_weight=0.0, limit=5000)
            candidates = list(dict.fromkeys([_clean_text(x.text) for x in faculty_specs if _clean_text(x.text)]))
            grant_specs = self.tools.fetch_grant_specs(min_spec_weight=0.0, limit=3000)
            grant_seed_texts = list(dict.fromkeys([_clean_text(x.text) for x in grant_specs if _clean_text(x.text)]))
            rng = random.Random(int(dataset_cfg.seed) + int(iteration))
            rng.shuffle(candidates)
            candidates = candidates[: max(20, int(loop_cfg.probe_candidate_count))]

            probe_case_payload = _generate_probe_cases(
                tools=self.tools,
                n=int(loop_cfg.probe_query_count),
                seed=int(dataset_cfg.seed) + int(iteration),
                llm_model=_clean_text(dataset_cfg.llm_model),
                grant_seed_texts=grant_seed_texts,
                candidate_texts=candidates,
            )
            probe_cases = list(probe_case_payload.get("cases") or [])
            probe_summary = _probe_inference(
                tools=self.tools,
                model_dir=model_dir,
                cases=probe_cases,
                top_k=int(loop_cfg.probe_top_k),
                low_conf_threshold=float(loop_cfg.low_conf_threshold),
                low_margin_threshold=float(loop_cfg.low_margin_threshold),
                case_source=_clean_text(probe_case_payload.get("source")) or "fallback",
            )
            probe_summary["probe_cases_requested"] = int(loop_cfg.probe_query_count)
            probe_summary["probe_case_pool_size"] = int(len(candidates))

            train_summary = dict(state.get("train_summary") or {})
            dataset_payload = dict(state.get("dataset_payload") or {})
            eval_loss = train_summary.get("eval_loss")
            eval_loss_val = None if eval_loss is None else _safe_float(eval_loss, default=999.0)
            best_eval_loss = state.get("best_eval_loss")
            best_iteration = int(state.get("best_iteration") or -1)
            if eval_loss_val is not None and (best_eval_loss is None or eval_loss_val < float(best_eval_loss)):
                best_eval_loss = float(eval_loss_val)
                best_iteration = int(iteration)

            record = {
                "iteration": int(iteration),
                "timestamp_utc": _now_utc(),
                "dataset_config": asdict(dataset_cfg),
                "dataset_payload": dataset_payload,
                "train_summary": train_summary,
                "probe_summary": probe_summary,
            }
            iter_dir = run_dir / f"iter_{iteration:02d}"
            _write_json(iter_dir / "iteration_summary.json", record)

            history = list(state.get("history") or [])
            history.append(record)
            return {
                "probe_summary": probe_summary,
                "history": history,
                "best_eval_loss": best_eval_loss,
                "best_iteration": best_iteration,
                "error": "",
            }
        except Exception as e:
            return {"error": f"tool_probe_quality failed: {e}"}

    def _node_retune_and_advance(self, state: FinetuneWorkflowState) -> FinetuneWorkflowState:
        dataset_cfg = DatasetTuningConfig(**dict(state.get("dataset_cfg") or {}))
        train_summary = dict(state.get("train_summary") or {})
        probe_summary = dict(state.get("probe_summary") or {})
        best_eval_loss = state.get("best_eval_loss")
        tuned = _tune_dataset_config(
            cfg=dataset_cfg,
            train_summary=train_summary,
            probe_summary=probe_summary,
            prev_best_eval_loss=(None if best_eval_loss is None else float(best_eval_loss)),
        )
        next_iteration = int(state.get("iteration") or 1) + 1
        return {
            "dataset_cfg": asdict(tuned),
            "iteration": next_iteration,
            "dataset_payload": {},
            "dataset_jsonl": "",
            "model_dir": "",
            "train_summary": {},
            "probe_summary": {},
        }

    def _node_finish(self, state: FinetuneWorkflowState) -> FinetuneWorkflowState:
        run_dir = Path(_clean_text(state.get("run_dir"))).resolve()
        final_payload = {
            "created_at_utc": _now_utc(),
            "run_dir": str(run_dir),
            "iterations_ran": int(len(list(state.get("history") or []))),
            "best_iteration": int(state.get("best_iteration") or -1),
            "best_eval_loss": state.get("best_eval_loss"),
            "stop_reason": _clean_text(state.get("stop_reason")),
            "history": list(state.get("history") or []),
            "error": _clean_text(state.get("error")),
        }
        _write_json(run_dir / "agentic_summary.json", final_payload)
        return {"result": final_payload}

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception as e:  # pragma: no cover - dependency guard
            raise ImportError("langgraph is required. Install with: pip install langgraph") from e

        graph = StateGraph(FinetuneWorkflowState)
        graph.add_node("plan_route", self._node_plan_route)
        graph.add_node("tool_build_dataset", self._node_tool_build_dataset)
        graph.add_node("tool_train_v2_simple", self._node_tool_train_v2_simple)
        graph.add_node("tool_probe_quality", self._node_tool_probe_quality)
        graph.add_node("retune_and_advance", self._node_retune_and_advance)
        graph.add_node("finish", self._node_finish)

        graph.set_entry_point("plan_route")
        graph.add_conditional_edges("plan_route", self._edge_from_route)
        graph.add_edge("tool_build_dataset", "plan_route")
        graph.add_edge("tool_train_v2_simple", "plan_route")
        graph.add_edge("tool_probe_quality", "plan_route")
        graph.add_edge("retune_and_advance", "plan_route")
        graph.add_edge("finish", END)
        return graph.compile()

    def run(
        self,
        *,
        run_dir: Path,
        dataset_cfg: DatasetTuningConfig,
        loop_cfg: LoopConfig,
    ) -> Dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        init_state: FinetuneWorkflowState = {
            "run_dir": str(run_dir),
            "iteration": 1,
            "max_iterations": int(loop_cfg.max_iterations),
            "dataset_cfg": asdict(dataset_cfg),
            "loop_cfg": asdict(loop_cfg),
            "dataset_payload": {},
            "dataset_jsonl": "",
            "model_dir": "",
            "train_summary": {},
            "probe_summary": {},
            "history": [],
            "best_eval_loss": None,
            "best_iteration": -1,
            "next_action": "",
            "stop_reason": "",
            "error": "",
            "result": {},
        }
        out = self.graph.invoke(init_state)
        result = dict(out.get("result") or {})
        if result:
            return result
        return {
            "created_at_utc": _now_utc(),
            "run_dir": str(run_dir),
            "iterations_ran": int(len(list(out.get("history") or []))),
            "best_iteration": int(out.get("best_iteration") or -1),
            "best_eval_loss": out.get("best_eval_loss"),
            "stop_reason": _clean_text(out.get("stop_reason")),
            "history": list(out.get("history") or []),
            "error": _clean_text(out.get("error")),
        }


def _build_parser() -> argparse.ArgumentParser:
    default_run_dir = (
        Path(__file__).resolve().parent
        / "agentic_runs"
        / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    p = argparse.ArgumentParser(
        description=(
            "LangGraph agentic fine-tuning: plan_route + tool nodes "
            "(build_llm_spec_pair_dataset, train_v2_simple, probe, retune)."
        )
    )
    p.add_argument("--run-dir", type=str, default=str(default_run_dir), help="Output directory for iteration artifacts.")
    p.add_argument("--max-iterations", type=int, default=3, help="Maximum loop iterations.")
    p.add_argument("--target-eval-loss", type=float, default=0.045, help="Stopping threshold for eval loss.")
    p.add_argument("--target-low-conf-ratio", type=float, default=0.25, help="Stopping threshold for probe low-confidence ratio.")
    p.add_argument("--target-low-margin-ratio", type=float, default=0.35, help="Stopping threshold for probe low-margin ratio.")
    p.add_argument("--target-false-positive-rate", type=float, default=0.30, help="Stopping threshold for probe false-positive rate.")
    p.add_argument("--target-false-negative-rate", type=float, default=0.30, help="Stopping threshold for probe false-negative rate.")
    p.add_argument("--target-hard-negative-top1-rate", type=float, default=0.25, help="Stopping threshold for hard-negative@top1 rate.")
    p.add_argument("--probe-query-count", type=int, default=24, help="Number of probe queries each iteration.")
    p.add_argument("--probe-candidate-count", type=int, default=120, help="Number of candidate docs for probing.")
    p.add_argument("--llm-model", type=str, default="", help="Optional LLM model id for dataset/probe stages.")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B during training.")
    p.add_argument("--json-only", action="store_true", help="Print only final JSON payload.")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    run_dir = Path(_clean_text(args.run_dir)).expanduser().resolve()
    llm_model = _clean_text(args.llm_model) or _resolve_default_llm_model()

    dataset_cfg = DatasetTuningConfig(llm_model=llm_model)
    loop_cfg = LoopConfig(
        max_iterations=int(args.max_iterations),
        target_eval_loss=float(args.target_eval_loss),
        target_low_conf_ratio=float(args.target_low_conf_ratio),
        target_low_margin_ratio=float(args.target_low_margin_ratio),
        target_false_positive_rate=float(args.target_false_positive_rate),
        target_false_negative_rate=float(args.target_false_negative_rate),
        target_hard_negative_top1_rate=float(args.target_hard_negative_top1_rate),
        probe_query_count=int(args.probe_query_count),
        probe_candidate_count=int(args.probe_candidate_count),
        use_wandb=not bool(args.no_wandb),
    )

    runner = AgenticFinetuneGraph()
    payload = runner.run(
        run_dir=run_dir,
        dataset_cfg=dataset_cfg,
        loop_cfg=loop_cfg,
    )

    if not args.json_only:
        print("LangGraph agentic fine-tuning complete.")
        print(f"  run dir          : {payload.get('run_dir', '')}")
        print(f"  iterations ran   : {payload.get('iterations_ran', 0)}")
        print(f"  best iteration   : {payload.get('best_iteration', -1)}")
        print(f"  best eval loss   : {payload.get('best_eval_loss', None)}")
        print(f"  stop reason      : {payload.get('stop_reason', '')}")
        print(f"  summary          : {run_dir / 'agentic_summary.json'}")
        print()

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
