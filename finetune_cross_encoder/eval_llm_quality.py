from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _clean_text(value) -> str:
    return str(value or "").strip()


def _default_llm_model() -> str:
    try:
        from config import settings  # local import to avoid hard dependency during --help

        return _clean_text(getattr(settings, "haiku", ""))
    except Exception:
        return ""


def _resolve_default_model_dir() -> Optional[Path]:
    """
    Prefer latest promoted checkpoint under finetune_cross_encoder/models.
    """
    base = Path(__file__).resolve().parent / "models"
    if not base.exists() or not base.is_dir():
        return None

    direct_promoted = base / "promoted_best"
    if direct_promoted.exists() and direct_promoted.is_dir():
        return direct_promoted.resolve()

    run_dirs = [p for p in base.glob("*") if p.is_dir()]
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        promoted = run_dir / "promoted_best"
        if promoted.exists() and promoted.is_dir():
            return promoted.resolve()
    return None


def _resolve_model_dir_arg(model_dir_arg: str) -> Path:
    raw = _clean_text(model_dir_arg)
    if raw:
        p = Path(raw).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Model directory not found: {p}")
        return p

    auto = _resolve_default_model_dir()
    if auto is None:
        raise FileNotFoundError(
            "No default promoted model found under finetune_cross_encoder/models. "
            "Pass --model-dir explicitly."
        )
    return auto


def _build_parser() -> argparse.ArgumentParser:
    default_out = Path(__file__).resolve().parent / "eval_runs"
    p = argparse.ArgumentParser(
        description=(
            "LLM probe-based quality evaluation for finetune_cross_encoder models. "
            "Defaults to latest promoted_best checkpoint under finetune_cross_encoder/models."
        )
    )
    p.add_argument("--model-dir", type=str, default="", help="Model directory. Default: latest promoted_best under finetune_cross_encoder/models.")
    p.add_argument("--llm-model", type=str, default="", help="Bedrock model id for probe generation and quality review. Default: settings.haiku.")
    p.add_argument("--num-cases", type=int, default=120, help="Number of probe cases.")
    p.add_argument("--output-dir", type=str, default=str(default_out), help="Output base directory for eval artifacts.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--cpu-only", action="store_true", help="Force CPU for model inference.")
    p.add_argument(
        "--gate-hard-negative-top1-rate-max",
        type=float,
        default=-1.0,
        help="Acceptance gate: pass if hard_negative_top1_rate <= threshold. Negative = disabled.",
    )
    p.add_argument(
        "--gate-false-positive-rate-max",
        type=float,
        default=-1.0,
        help="Acceptance gate: pass if false_positive_rate <= threshold. Negative = disabled.",
    )
    p.add_argument(
        "--gate-mean-top1-top2-margin-min",
        type=float,
        default=-1.0,
        help="Acceptance gate: pass if mean_top1_top2_margin >= threshold. Negative = disabled.",
    )
    p.add_argument("--json-only", action="store_true", help="Print JSON payload only.")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    try:
        from train_cross_encoder.eval_bge_reranker_llm_quality import run_llm_quality_eval
    except Exception as e:
        raise RuntimeError(
            "Unable to import evaluator. Make sure project dependencies are installed in the active environment."
        ) from e

    model_dir = _resolve_model_dir_arg(_clean_text(args.model_dir))
    llm_model = _clean_text(args.llm_model) or _default_llm_model()
    if not llm_model:
        raise RuntimeError(
            "Missing --llm-model and settings.haiku is unavailable. "
            "Set BEDROCK_CLAUDE_HAIKU in .env or pass --llm-model."
        )

    run_kwargs = {
        "model_dir": str(model_dir),
        "llm_model": llm_model,
        "num_cases": int(args.num_cases),
        "output_dir": Path(_clean_text(args.output_dir)),
        "cpu_only": bool(args.cpu_only),
        "seed": int(args.seed),
        "gate_hard_negative_top1_rate_max": (
            float(args.gate_hard_negative_top1_rate_max)
            if float(args.gate_hard_negative_top1_rate_max) >= 0.0
            else None
        ),
        "gate_false_positive_rate_max": (
            float(args.gate_false_positive_rate_max)
            if float(args.gate_false_positive_rate_max) >= 0.0
            else None
        ),
        "gate_mean_top1_top2_margin_min": (
            float(args.gate_mean_top1_top2_margin_min)
            if float(args.gate_mean_top1_top2_margin_min) >= 0.0
            else None
        ),
    }
    sig = inspect.signature(run_llm_quality_eval)
    supported = set(sig.parameters.keys())
    call_kwargs = {k: v for k, v in run_kwargs.items() if k in supported}
    dropped = [k for k in run_kwargs.keys() if k not in supported]
    summary = run_llm_quality_eval(**call_kwargs)
    if dropped:
        summary["compat_note"] = (
            "Evaluator version does not support some optional args; ignored: "
            + ", ".join(dropped)
        )

    if not bool(args.json_only):
        metrics = dict(summary.get("metrics") or {})
        gates = dict(summary.get("acceptance_gates") or {})
        print("Finetune LLM quality evaluation complete.")
        print(f"  model dir               : {summary.get('model_dir', '')}")
        print(f"  run dir                 : {summary.get('run_dir', '')}")
        print(f"  probe cases             : {summary.get('probe_generation', {}).get('generated_case_count', 0)}")
        print(f"  top1 accuracy           : {metrics.get('top1_accuracy', 0.0):.4f}")
        print(f"  false positive rate     : {metrics.get('false_positive_rate', 0.0):.4f}")
        print(f"  hard negative top1 rate : {metrics.get('hard_negative_top1_rate', 0.0):.4f}")
        print(f"  mean top1-top2 margin   : {metrics.get('mean_top1_top2_margin', 0.0):.4f}")
        if bool(gates.get("enabled")):
            print(f"  acceptance gates pass   : {bool(gates.get('all_pass'))}")
        print(f"  summary file            : {summary.get('files', {}).get('summary', '')}")
        print()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
