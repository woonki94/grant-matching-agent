from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce4_aspect_attention").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce3.eval.compare_overall_gt_to_finetuned_ce import (  # noqa: E402
    _print_aggregator_table,
    _resolve_path,
    _write_jsonl,
)
from ce4_aspect_attention.train_plain_sts_attention import (  # noqa: E402
    PlainStsAttentionConfig,
    GroundTruthLoader,
    PlainStsScorer,
    CoverageTrainer,
    ScalarCalibratorModel,
    RepeatedScoreAttentionModel,
    clean_text,
)


class PlainStsAttentionTestEvaluator:
    def __init__(self, run_dir: Path, config: PlainStsAttentionConfig) -> None:
        self.run_dir = run_dir
        self.config = config
        self.device = self.pick_device()
        self.loader = GroundTruthLoader()
        self.scorer = PlainStsScorer(config, self.device)
        self.trainer = CoverageTrainer(config, self.device)

    def pick_device(self) -> Any:
        from ce3.eval.compare_overall_gt_to_finetuned_ce import _pick_device

        return _pick_device(self.config.device)

    def evaluate_test(self) -> Dict[str, Any]:
        started = time.time()
        test_rows = self.loader.load_split(_resolve_path(self.config.gt_test))
        scored_test = self.scorer.score_splits({"test": test_rows})["test"]
        models = self.load_models()
        metrics: Dict[str, Dict[str, Any]] = {"raw": self.trainer.evaluate_rows(scored_test)}
        predictions: Dict[str, List[Dict[str, Any]]] = {"plain_sts": scored_test}
        for model_name, model in models.items():
            rows = self.trainer.predict_rows(scored_test, model=model, score_name=model_name)
            predictions[model_name] = rows
            metrics[model_name] = self.trainer.evaluate_rows(rows)

        output = {
            "mode": "ce4_eval_plain_sts_repeated_score_attention",
            "run_dir": str(self.run_dir),
            "gt_test": str(_resolve_path(self.config.gt_test)),
            "metrics": metrics,
            "elapsed_sec": float(time.time() - started),
        }
        (self.run_dir / "test_metrics.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for model_name, rows in predictions.items():
            _write_jsonl(self.run_dir / f"test.{model_name}.details.jsonl", rows)
        self.print_summary(output)
        return output

    def load_models(self) -> Dict[str, Any]:
        import torch

        builders = {
            "scalar_linear": ScalarCalibratorModel.build_linear,
            "scalar_mlp": lambda: ScalarCalibratorModel.build_mlp(self.config.hidden_dim, self.config.dropout),
            "repeated_score_attention": lambda: RepeatedScoreAttentionModel.build_model(self.config),
        }
        models: Dict[str, Any] = {}
        for model_name, builder in builders.items():
            path = self.run_dir / f"{model_name}.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing trained model: {path}")
            model = builder()
            state = torch.load(path, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            models[model_name] = model
        return models

    def print_summary(self, output: Dict[str, Any]) -> None:
        print("\n=== Raw Plain STS Test Metrics ===")
        _print_aggregator_table(output["metrics"]["raw"])
        for model_name in ("scalar_linear", "scalar_mlp", "repeated_score_attention"):
            print(f"\n=== {model_name} Test Metrics ===")
            _print_aggregator_table(output["metrics"][model_name])
        print(f"\nrun_dir={output['run_dir']}")
        print(f"test_metrics={self.run_dir / 'test_metrics.json'}")
        print(f"elapsed_sec={float(output['elapsed_sec']):.2f}")


class EvalArgumentParserBuilder:
    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Evaluate saved plain STS attention models on the held-out test split.",
        )
        parser.add_argument("--run-dir", required=True, type=str)
        parser.add_argument("--gt-test", type=str, default="")
        parser.add_argument("--cache-dir", type=str, default="")
        parser.add_argument("--device", type=str, default="")
        parser.add_argument("--overwrite-cache", action="store_true")
        return parser.parse_args()


def load_config(args: argparse.Namespace) -> PlainStsAttentionConfig:
    run_dir = _resolve_path(args.run_dir)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing train config: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config = PlainStsAttentionConfig(**raw)
    if clean_text(args.gt_test):
        config = replace(config, gt_test=args.gt_test)
    if clean_text(args.cache_dir):
        config = replace(config, cache_dir=args.cache_dir)
    if clean_text(args.device):
        config = replace(config, device=args.device)
    if bool(args.overwrite_cache):
        config = replace(config, overwrite_cache=True)
    return config


def main() -> int:
    args = EvalArgumentParserBuilder().parse_args()
    run_dir = _resolve_path(args.run_dir)
    config = load_config(args)
    PlainStsAttentionTestEvaluator(run_dir, config).evaluate_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
