from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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


def _build_dataset_from_db(**kwargs) -> Dict[str, Any]:
    mod = _load_build_module()
    return dict(mod.build_dataset(**kwargs))


def _fetch_grant_specs_from_db(*, min_spec_weight: float, limit: int) -> List[Any]:
    mod = _load_build_module()
    return list(mod._fetch_grant_specs(min_spec_weight=float(min_spec_weight), limit=int(limit)))


def _fetch_faculty_specs_from_db(*, min_spec_weight: float, limit: int) -> List[Any]:
    mod = _load_build_module()
    return list(mod._fetch_faculty_specs(min_spec_weight=float(min_spec_weight), limit=int(limit)))


def _run_train_stage(*, dataset_jsonl: Path, output_dir: Path, use_wandb: bool) -> Dict[str, Any]:
    mod = _load_train_module()
    return dict(mod.run_train(dataset_jsonl=dataset_jsonl, output_dir=output_dir, use_wandb=bool(use_wandb)))


def _run_probe_inference(**kwargs) -> Dict[str, Any]:
    mod = _load_infer_module()
    return dict(mod.run_inference(**kwargs))


def _build_llm_client(model_id: str):
    cfg = _load_config_module()
    return cfg.get_llm_client(model_id).build()


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
    probe_query_count: int = 24
    probe_candidate_count: int = 120
    probe_top_k: int = 5
    low_conf_threshold: float = 0.45
    low_margin_threshold: float = 0.05
    use_wandb: bool = True


def _resolve_default_llm_model() -> str:
    try:
        cfg = _load_config_module()
        return _clean_text(getattr(cfg.settings, "haiku", ""))
    except Exception:
        return ""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _generate_probe_queries(
    *,
    n: int,
    seed: int,
    llm_model: str,
) -> List[str]:
    rng = random.Random(int(seed))
    grant_specs = _fetch_grant_specs_from_db(min_spec_weight=0.0, limit=3000)
    base = [str(x.text or "").strip() for x in grant_specs if str(x.text or "").strip()]
    base = list(dict.fromkeys(base))
    if not base:
        return []
    rng.shuffle(base)
    seeds = base[: max(8, min(len(base), n))]

    prompt = (
        "Generate short research-specialization queries for reranker probing.\n"
        "Return ONLY JSON array of strings. No markdown.\n"
        f"Need exactly {int(n)} queries.\n"
        "Use these seed topics as inspiration:\n"
        + "\n".join(f"- {s}" for s in seeds[:40])
    )

    try:
        llm = _build_llm_client(llm_model)
        res = llm.invoke(prompt)
        raw = getattr(res, "content", None)
        text = str(raw if raw is not None else res)
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            arr = json.loads(text[start : end + 1])
            out = []
            for item in list(arr or []):
                q = _clean_text(item)
                if q:
                    out.append(q)
            out = list(dict.fromkeys(out))
            if len(out) >= n:
                return out[:n]
    except Exception:
        pass

    # Fallback: templated variants (still synthetic queries).
    templates = [
        "advanced methods in {x}",
        "applied research on {x}",
        "scalable approaches for {x}",
        "interdisciplinary studies in {x}",
        "real-world deployment challenges in {x}",
    ]
    out: List[str] = []
    for s in seeds:
        for t in templates:
            out.append(t.format(x=s))
            if len(out) >= n:
                return out[:n]
    return out[:n]


def _probe_inference(
    *,
    model_dir: Path,
    queries: Sequence[str],
    candidate_texts: Sequence[str],
    top_k: int,
    low_conf_threshold: float,
    low_margin_threshold: float,
) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    low_conf = 0
    low_margin = 0
    total = 0

    for q in list(queries or []):
        query = _clean_text(q)
        if not query:
            continue
        payload = _run_probe_inference(
            model_dir=str(model_dir),
            query=query,
            candidates=list(candidate_texts),
            candidates_file="",
            input_jsonl="",
            output_jsonl="",
            batch_size=64,
            max_length=256,
            top_k=int(max(1, top_k)),
            cpu_only=False,
        )
        ranked = list(payload.get("ranked") or [])
        if not ranked:
            continue
        total += 1
        top1 = float((ranked[0] or {}).get("score") or 0.0)
        top2 = float((ranked[1] or {}).get("score") or top1) if len(ranked) > 1 else top1
        margin = float(top1 - top2)
        is_low_conf = bool(top1 < float(low_conf_threshold))
        is_low_margin = bool(margin < float(low_margin_threshold))
        if is_low_conf:
            low_conf += 1
        if is_low_margin:
            low_margin += 1
        details.append(
            {
                "query": query,
                "top1_score": top1,
                "top2_score": top2,
                "margin": margin,
                "low_conf": is_low_conf,
                "low_margin": is_low_margin,
                "top_candidate": _clean_text((ranked[0] or {}).get("candidate")),
            }
        )

    details = sorted(
        details,
        key=lambda x: (
            not bool(x.get("low_conf")),
            not bool(x.get("low_margin")),
            float(x.get("top1_score") or 0.0),
        ),
    )
    return {
        "probe_queries_used": int(total),
        "low_conf_count": int(low_conf),
        "low_margin_count": int(low_margin),
        "low_conf_ratio": (float(low_conf) / float(total) if total > 0 else 1.0),
        "low_margin_ratio": (float(low_margin) / float(total) if total > 0 else 1.0),
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

    # Probe-driven adjustments.
    if low_conf_ratio > 0.40:
        out.top_k_candidates = min(64, out.top_k_candidates + 2)
        out.hard_negatives_per_grant = min(64, out.hard_negatives_per_grant + 2)
    if low_margin_ratio > 0.40:
        out.candidates_per_query = min(64, out.candidates_per_query + 4)
        out.hard_negatives_per_grant = min(64, out.hard_negatives_per_grant + 2)

    # Overfitting-ish signal.
    if eval_loss is not None and (train_loss + 0.01) < eval_loss:
        out.max_queries = min(20000, out.max_queries + 1000)
        out.random_negatives_per_grant = min(64, out.random_negatives_per_grant + 2)

    # If loss got worse compared to previous best, add diversity.
    if eval_loss is not None and prev_best_eval_loss is not None and eval_loss > (prev_best_eval_loss * 1.02):
        out.random_negatives_per_grant = min(64, out.random_negatives_per_grant + 2)
        out.max_queries = min(25000, out.max_queries + 1500)

    # Keep parameters coherent.
    out.hard_negatives_per_grant = _safe_int(out.hard_negatives_per_grant, default=10, minimum=0, maximum=64)
    out.random_negatives_per_grant = _safe_int(out.random_negatives_per_grant, default=10, minimum=0, maximum=64)
    out.top_k_candidates = _safe_int(out.top_k_candidates, default=8, minimum=1, maximum=64)
    out.candidates_per_query = _safe_int(out.candidates_per_query, default=20, minimum=2, maximum=128)
    if out.candidates_per_query < (out.top_k_candidates + out.hard_negatives_per_grant):
        out.candidates_per_query = min(128, out.top_k_candidates + out.hard_negatives_per_grant + 2)
    return out


def run_loop(
    *,
    run_dir: Path,
    dataset_cfg: DatasetTuningConfig,
    loop_cfg: LoopConfig,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cur_cfg = replace(dataset_cfg)
    history: List[Dict[str, Any]] = []
    best_eval_loss: float | None = None
    best_iter = -1

    for it in range(1, int(loop_cfg.max_iterations) + 1):
        iter_dir = run_dir / f"iter_{it:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        print(f"[agentic] iteration={it} building dataset...")
        ds_payload = _build_dataset_from_db(
            top_k_candidates=int(cur_cfg.top_k_candidates),
            hard_negatives_per_grant=int(cur_cfg.hard_negatives_per_grant),
            random_negatives_per_grant=int(cur_cfg.random_negatives_per_grant),
            candidates_per_query=int(cur_cfg.candidates_per_query),
            max_queries=int(cur_cfg.max_queries),
            max_pairs=int(cur_cfg.max_pairs),
            llm_batch_size=int(cur_cfg.llm_batch_size),
            llm_model=_clean_text(cur_cfg.llm_model),
            llm_max_retries=int(cur_cfg.llm_max_retries),
            llm_max_workers=int(cur_cfg.llm_max_workers),
            faculty_min_spec_weight=float(cur_cfg.faculty_min_spec_weight),
            grant_min_spec_weight=float(cur_cfg.grant_min_spec_weight),
            faculty_limit=int(cur_cfg.faculty_limit),
            grant_limit=int(cur_cfg.grant_limit),
            embed_batch_size=int(cur_cfg.embed_batch_size),
            seed=int(cur_cfg.seed),
            output_dir=iter_dir / "dataset",
            output_prefix=f"agentic_iter{it:02d}",
        )
        dataset_jsonl = Path(_clean_text(ds_payload.get("output", {}).get("jsonl_path"))).resolve()

        print(f"[agentic] iteration={it} training...")
        model_dir = iter_dir / "model"
        train_summary = _run_train_stage(
            dataset_jsonl=dataset_jsonl,
            output_dir=model_dir,
            use_wandb=bool(loop_cfg.use_wandb),
        )

        print(f"[agentic] iteration={it} probe inference...")
        faculty_specs = _fetch_faculty_specs_from_db(min_spec_weight=0.0, limit=5000)
        candidates = list(dict.fromkeys([_clean_text(x.text) for x in faculty_specs if _clean_text(x.text)]))
        rng = random.Random(int(cur_cfg.seed) + int(it))
        rng.shuffle(candidates)
        candidates = candidates[: max(20, int(loop_cfg.probe_candidate_count))]
        queries = _generate_probe_queries(
            n=int(loop_cfg.probe_query_count),
            seed=int(cur_cfg.seed) + int(it),
            llm_model=_clean_text(cur_cfg.llm_model),
        )
        probe_summary = _probe_inference(
            model_dir=model_dir,
            queries=queries,
            candidate_texts=candidates,
            top_k=int(loop_cfg.probe_top_k),
            low_conf_threshold=float(loop_cfg.low_conf_threshold),
            low_margin_threshold=float(loop_cfg.low_margin_threshold),
        )

        eval_loss = train_summary.get("eval_loss")
        eval_loss_val = None if eval_loss is None else _safe_float(eval_loss, default=999.0)
        if eval_loss_val is not None and (best_eval_loss is None or eval_loss_val < best_eval_loss):
            best_eval_loss = float(eval_loss_val)
            best_iter = int(it)

        record = {
            "iteration": int(it),
            "timestamp_utc": _now_utc(),
            "dataset_config": asdict(cur_cfg),
            "dataset_payload": ds_payload,
            "train_summary": train_summary,
            "probe_summary": probe_summary,
        }
        _write_json(iter_dir / "iteration_summary.json", record)
        history.append(record)

        pass_eval = bool(eval_loss_val is not None and eval_loss_val <= float(loop_cfg.target_eval_loss))
        pass_low_conf = bool(_safe_float(probe_summary.get("low_conf_ratio"), default=1.0) <= float(loop_cfg.target_low_conf_ratio))
        pass_low_margin = bool(_safe_float(probe_summary.get("low_margin_ratio"), default=1.0) <= float(loop_cfg.target_low_margin_ratio))
        print(
            "[agentic] "
            f"iteration={it} eval_loss={eval_loss_val} "
            f"low_conf_ratio={probe_summary.get('low_conf_ratio')} "
            f"low_margin_ratio={probe_summary.get('low_margin_ratio')}"
        )
        if pass_eval and pass_low_conf and pass_low_margin:
            print(f"[agentic] stop: satisfactory at iteration={it}")
            break

        cur_cfg = _tune_dataset_config(
            cfg=cur_cfg,
            train_summary=train_summary,
            probe_summary=probe_summary,
            prev_best_eval_loss=best_eval_loss,
        )

    final_payload = {
        "created_at_utc": _now_utc(),
        "run_dir": str(run_dir),
        "iterations_ran": int(len(history)),
        "best_iteration": int(best_iter),
        "best_eval_loss": (None if best_eval_loss is None else float(best_eval_loss)),
        "history": history,
    }
    _write_json(run_dir / "agentic_summary.json", final_payload)
    return final_payload


def _build_parser() -> argparse.ArgumentParser:
    default_run_dir = (
        Path(__file__).resolve().parent
        / "agentic_runs"
        / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    p = argparse.ArgumentParser(
        description="Agentic fine-tuning loop: build dataset from DB, train, probe inference, retune dataset config, repeat."
    )
    p.add_argument("--run-dir", type=str, default=str(default_run_dir), help="Output directory for iteration artifacts.")
    p.add_argument("--max-iterations", type=int, default=3, help="Maximum loop iterations.")
    p.add_argument("--target-eval-loss", type=float, default=0.045, help="Stopping threshold for eval loss.")
    p.add_argument("--target-low-conf-ratio", type=float, default=0.25, help="Stopping threshold for probe low-confidence ratio.")
    p.add_argument("--target-low-margin-ratio", type=float, default=0.35, help="Stopping threshold for probe low-margin ratio.")
    p.add_argument("--probe-query-count", type=int, default=24, help="Number of probe queries each iteration.")
    p.add_argument("--probe-candidate-count", type=int, default=120, help="Number of candidate docs for probing.")
    p.add_argument("--llm-model", type=str, default="", help="Optional LLM model id for dataset/probe stages.")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B in training stage.")
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
        probe_query_count=int(args.probe_query_count),
        probe_candidate_count=int(args.probe_candidate_count),
        use_wandb=not bool(args.no_wandb),
    )

    payload = run_loop(
        run_dir=run_dir,
        dataset_cfg=dataset_cfg,
        loop_cfg=loop_cfg,
    )

    if not args.json_only:
        print("Agentic fine-tuning loop complete.")
        print(f"  run dir          : {payload.get('run_dir', '')}")
        print(f"  iterations ran   : {payload.get('iterations_ran', 0)}")
        print(f"  best iteration   : {payload.get('best_iteration', -1)}")
        print(f"  best eval loss   : {payload.get('best_eval_loss', None)}")
        print(f"  summary          : {run_dir / 'agentic_summary.json'}")
        print()

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
