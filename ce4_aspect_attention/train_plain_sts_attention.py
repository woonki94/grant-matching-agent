from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


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
    _all_aggregator_metrics,
    _clamp_01,
    _load_gt_rows,
    _pick_device,
    _print_aggregator_table,
    _resolve_path,
    _write_jsonl,
)


ASPECTS: Tuple[str, ...] = ("topic", "approach", "objective")
GT_ROOT_DEFAULT = "ce4_aspect_attention/dataset/ground_truth"
GT_TRAIN_DEFAULT = f"{GT_ROOT_DEFAULT}/overall_coverage_test_subset_claude_opus_train.jsonl"
GT_VAL_DEFAULT = f"{GT_ROOT_DEFAULT}/overall_coverage_test_subset_claude_opus_val.jsonl"
GT_TEST_DEFAULT = f"{GT_ROOT_DEFAULT}/overall_coverage_test_subset_claude_opus_test.jsonl"
CACHE_DIR_DEFAULT = "ce4_aspect_attention/cache"
OUTPUT_DIR_DEFAULT = "ce4_aspect_attention/results"
PLAIN_STS_MODEL_DEFAULT = "dleemiller/ModernCE-base-sts"
HIGH_THRESHOLD = 0.70
MID_THRESHOLD = 0.30


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def seed_everything(seed: int) -> None:
    import torch

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def threshold_to_logit(threshold: float) -> float:
    import math

    value = min(1.0 - 1e-6, max(1e-6, float(threshold)))
    return float(math.log(value / (1.0 - value)))


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL row at {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                yield obj


@dataclass(frozen=True)
class PlainStsAttentionConfig:
    gt_train: str = GT_TRAIN_DEFAULT
    gt_val: str = GT_VAL_DEFAULT
    gt_test: str = GT_TEST_DEFAULT
    cache_dir: str = CACHE_DIR_DEFAULT
    output_dir: str = OUTPUT_DIR_DEFAULT
    run_name: str = ""
    plain_sts_model: str = PLAIN_STS_MODEL_DEFAULT
    device: str = ""
    infer_batch_size: int = 64
    train_batch_size: int = 64
    max_length: int = 512
    epochs: int = 500
    patience: int = 80
    learning_rate: float = 5e-3
    weight_decay: float = 1e-3
    hidden_dim: int = 16
    token_dim: int = 32
    num_heads: int = 4
    num_layers: int = 1
    dropout: float = 0.10
    loss_any_boundary_weight: float = 0.0
    loss_high_boundary_weight: float = 0.0
    high_threshold: float = HIGH_THRESHOLD
    mid_threshold: float = MID_THRESHOLD
    seed: int = 42
    trust_remote_code: bool = True
    overwrite_cache: bool = False


class GroundTruthLoader:
    def load_split(self, path: Path) -> List[Dict[str, Any]]:
        rows = _load_gt_rows(path)
        if not rows:
            raise RuntimeError(f"No usable GT rows loaded from {path}")
        return rows


class PlainStsScorer:
    def __init__(self, config: PlainStsAttentionConfig, device: Any) -> None:
        self.config = config
        self.device = device

    def score_splits(self, split_rows: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        cache_dir = _resolve_path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        scored: Dict[str, List[Dict[str, Any]]] = {}
        missing = []
        for split_name in split_rows:
            cache_path = cache_dir / f"plain_sts_{split_name}.jsonl"
            if cache_path.exists() and not self.config.overwrite_cache:
                scored[split_name] = list(iter_jsonl(cache_path))
            else:
                missing.append(split_name)

        if missing:
            inferred = self.infer_scores({name: split_rows[name] for name in missing})
            for split_name, rows in inferred.items():
                cache_path = cache_dir / f"plain_sts_{split_name}.jsonl"
                _write_jsonl(cache_path, rows)
                scored[split_name] = rows
        return scored

    def infer_scores(self, split_rows: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.plain_sts_model,
            trust_remote_code=bool(self.config.trust_remote_code),
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.plain_sts_model,
            num_labels=1,
            trust_remote_code=bool(self.config.trust_remote_code),
        )
        model.to(self.device)
        model.eval()

        scored: Dict[str, List[Dict[str, Any]]] = {}
        with torch.no_grad():
            for split_name, rows in split_rows.items():
                print(f"inferring_plain_sts split={split_name} rows={len(rows)}")
                scored[split_name] = self.score_rows(model, tokenizer, rows)

        try:
            model.to("cpu")
        except Exception:
            pass
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return scored

    def score_rows(self, model: Any, tokenizer: Any, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        import torch

        out = [copy.deepcopy(row) for row in rows]
        step = max(1, int(self.config.infer_batch_size))
        for start in range(0, len(out), step):
            chunk = out[start : start + step]
            enc = tokenizer(
                [row["grant_text"] for row in chunk],
                [row["faculty_text"] for row in chunk],
                max_length=int(self.config.max_length),
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {key: value.to(self.device) for key, value in enc.items()}
            logits = model(**enc).logits.view(-1)
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            for row, score in zip(chunk, probs):
                self.attach_score(row, score)
        return out

    def attach_score(self, row: Dict[str, Any], score: float) -> None:
        plain_score = float(_clamp_01(score))
        row["plain_sts_score"] = plain_score
        row["aspect_scores"] = {aspect: plain_score for aspect in ASPECTS}
        row["aggregate_scores"] = {
            "plain_sts": plain_score,
            "duplicate_mean": plain_score,
        }


class ScalarCalibratorModel:
    @staticmethod
    def build_linear() -> Any:
        from torch import nn

        return nn.Linear(1, 1)

    @staticmethod
    def build_mlp(hidden_dim: int, dropout: float) -> Any:
        from torch import nn

        return nn.Sequential(
            nn.Linear(1, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )


class RepeatedScoreAttentionModel:
    @staticmethod
    def build_model(config: PlainStsAttentionConfig) -> Any:
        import torch
        from torch import nn

        class RepeatedScoreAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if int(config.token_dim) % int(config.num_heads) != 0:
                    raise ValueError("token_dim must be divisible by num_heads.")
                self.score_projection = nn.Sequential(
                    nn.Linear(1, int(config.token_dim)),
                    nn.GELU(),
                    nn.LayerNorm(int(config.token_dim)),
                )
                self.aspect_embedding = nn.Embedding(len(ASPECTS), int(config.token_dim))
                self.cls_token = nn.Parameter(torch.zeros(1, 1, int(config.token_dim)))
                self.layers = nn.ModuleList(
                    [
                        nn.TransformerEncoderLayer(
                            d_model=int(config.token_dim),
                            nhead=int(config.num_heads),
                            dim_feedforward=int(config.token_dim) * 4,
                            dropout=float(config.dropout),
                            activation="gelu",
                            batch_first=True,
                            norm_first=True,
                        )
                        for _ in range(int(config.num_layers))
                    ]
                )
                self.output = nn.Sequential(
                    nn.LayerNorm(int(config.token_dim)),
                    nn.Linear(int(config.token_dim), int(config.token_dim)),
                    nn.GELU(),
                    nn.Dropout(float(config.dropout)),
                    nn.Linear(int(config.token_dim), 1),
                )
                nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

            def forward(self, scores: Any) -> Any:
                batch_size = int(scores.shape[0])
                repeated = scores.view(batch_size, 1, 1).expand(batch_size, len(ASPECTS), 1)
                aspect_ids = torch.arange(len(ASPECTS), device=scores.device).view(1, len(ASPECTS))
                tokens = self.score_projection(repeated) + self.aspect_embedding(aspect_ids)
                cls = self.cls_token.expand(batch_size, 1, -1)
                tokens = torch.cat([cls, tokens], dim=1)
                for layer in self.layers:
                    tokens = layer(tokens)
                return self.output(tokens[:, 0, :])

        return RepeatedScoreAttention()


class TensorDatasetBuilder:
    def __init__(self, device: Any) -> None:
        self.device = device

    def build_tensors(self, rows: Sequence[Dict[str, Any]]) -> Tuple[Any, Any]:
        import torch

        scores = [float(_clamp_01(row.get("plain_sts_score"))) for row in rows]
        targets = [float(_clamp_01(row.get("gt_score"))) for row in rows]
        x = torch.tensor(scores, dtype=torch.float32, device=self.device).view(-1, 1)
        y = torch.tensor(targets, dtype=torch.float32, device=self.device).view(-1, 1)
        return x, y


class CoverageTrainer:
    def __init__(self, config: PlainStsAttentionConfig, device: Any) -> None:
        self.config = config
        self.device = device
        self.tensor_builder = TensorDatasetBuilder(device)

    def train_model(
        self,
        *,
        name: str,
        model: Any,
        train_rows: Sequence[Dict[str, Any]],
        val_rows: Sequence[Dict[str, Any]],
    ) -> Tuple[Any, Dict[str, Any]]:
        import torch

        x_train, y_train = self.tensor_builder.build_tensors(train_rows)
        x_val, y_val = self.tensor_builder.build_tensors(val_rows)
        model.to(self.device)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )

        best_state = copy.deepcopy(model.state_dict())
        best_score = float("inf")
        best_epoch = 0
        wait = 0
        history: List[Dict[str, Any]] = []
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.seed))

        for epoch in range(1, max(1, int(self.config.epochs)) + 1):
            model.train()
            order = torch.randperm(int(x_train.shape[0]), generator=generator).to(self.device)
            losses: List[float] = []
            step = max(1, int(self.config.train_batch_size))
            for start in range(0, int(x_train.shape[0]), step):
                idx = order[start : start + step]
                logits = model(x_train[idx])
                loss = self.calculate_loss(logits, y_train[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu().item()))

            val_rows_scored = self.predict_rows(val_rows, model=model, score_name=name)
            val_metrics = self.evaluate_rows(val_rows_scored).get(name, {})
            selection_score = self.select_model(val_metrics)
            history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(sum(losses) / max(1, len(losses))),
                    "val_mae": float(val_metrics.get("mae", 999.0)),
                    "val_rmse": float(val_metrics.get("rmse", 999.0)),
                    "val_bias": float(val_metrics.get("bias_pred_minus_gt", 0.0)),
                    "selection_score": float(selection_score),
                }
            )
            if selection_score < best_score:
                best_score = float(selection_score)
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
            if int(self.config.patience) > 0 and wait >= int(self.config.patience):
                break

        model.load_state_dict(best_state)
        return model, {
            "name": name,
            "best_epoch": int(best_epoch),
            "best_selection_score": float(best_score),
            "epochs_run": int(len(history)),
            "history": history,
        }

    def calculate_loss(self, logits: Any, targets: Any) -> Any:
        import torch.nn.functional as F

        probs = logits.sigmoid()
        mse = F.mse_loss(probs, targets)
        any_loss = logits.new_tensor(0.0)
        high_loss = logits.new_tensor(0.0)
        if float(self.config.loss_any_boundary_weight) > 0.0:
            any_targets = (targets >= float(self.config.mid_threshold)).to(dtype=logits.dtype)
            any_loss = F.binary_cross_entropy_with_logits(
                logits - threshold_to_logit(self.config.mid_threshold),
                any_targets,
            )
        if float(self.config.loss_high_boundary_weight) > 0.0:
            high_targets = (targets >= float(self.config.high_threshold)).to(dtype=logits.dtype)
            high_loss = F.binary_cross_entropy_with_logits(
                logits - threshold_to_logit(self.config.high_threshold),
                high_targets,
            )
        return (
            mse
            + float(self.config.loss_any_boundary_weight) * any_loss
            + float(self.config.loss_high_boundary_weight) * high_loss
        )

    def predict_rows(self, rows: Sequence[Dict[str, Any]], *, model: Any, score_name: str) -> List[Dict[str, Any]]:
        import torch

        model.eval()
        x, _ = self.tensor_builder.build_tensors(rows)
        preds: List[float] = []
        step = max(1, int(self.config.train_batch_size))
        with torch.no_grad():
            for start in range(0, int(x.shape[0]), step):
                logits = model(x[start : start + step])
                preds.extend(float(_clamp_01(v)) for v in logits.sigmoid().detach().cpu().view(-1).tolist())
        out: List[Dict[str, Any]] = []
        for row, pred in zip(rows, preds):
            item = copy.deepcopy(row)
            item.setdefault("aggregate_scores", {})
            item["aggregate_scores"][score_name] = float(pred)
            out.append(item)
        return out

    def evaluate_rows(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return _all_aggregator_metrics(
            rows,
            high_threshold=float(self.config.high_threshold),
            mid_threshold=float(self.config.mid_threshold),
        )

    def select_model(self, metrics: Dict[str, Any]) -> float:
        mae = float(metrics.get("mae", 999.0))
        bias = abs(float(metrics.get("bias_pred_minus_gt", 0.0)))
        return float(mae + 0.25 * bias)


class PlainStsAttentionTrainer:
    def __init__(self, config: PlainStsAttentionConfig) -> None:
        self.config = config
        self.device = _pick_device(config.device)
        self.loader = GroundTruthLoader()
        self.scorer = PlainStsScorer(config, self.device)
        self.trainer = CoverageTrainer(config, self.device)

    def train_model(self) -> Dict[str, Any]:
        started = time.time()
        seed_everything(int(self.config.seed))
        output_dir = self.prepare_output_dir()
        split_rows = self.load_splits()
        scored_splits = self.scorer.score_splits(split_rows)
        models = self.build_models()
        trained = self.train_models(models, scored_splits)
        summary = self.write_outputs(
            output_dir=output_dir,
            scored_splits=scored_splits,
            trained=trained,
            elapsed_sec=float(time.time() - started),
        )
        self.print_summary(summary)
        return summary

    def prepare_output_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = clean_text(self.config.run_name) or f"plain_sts_attention_eval_{timestamp}"
        output_dir = _resolve_path(self.config.output_dir) / run_name
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir

    def load_splits(self) -> Dict[str, List[Dict[str, Any]]]:
        paths = {
            "train": _resolve_path(self.config.gt_train),
            "val": _resolve_path(self.config.gt_val),
            "test": _resolve_path(self.config.gt_test),
        }
        for split_name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing GT {split_name} file: {path}")
        return {split_name: self.loader.load_split(paths[split_name]) for split_name in ("train", "val")}

    def build_models(self) -> Dict[str, Any]:
        return {
            "scalar_linear": ScalarCalibratorModel.build_linear(),
            "scalar_mlp": ScalarCalibratorModel.build_mlp(self.config.hidden_dim, self.config.dropout),
            "repeated_score_attention": RepeatedScoreAttentionModel.build_model(self.config),
        }

    def train_models(
        self,
        models: Dict[str, Any],
        scored_splits: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        trained: Dict[str, Dict[str, Any]] = {}
        for name, model in models.items():
            print(f"training_model={name}")
            trained_model, meta = self.trainer.train_model(
                name=name,
                model=model,
                train_rows=scored_splits["train"],
                val_rows=scored_splits["val"],
            )
            split_predictions = {
                split_name: self.trainer.predict_rows(rows, model=trained_model, score_name=name)
                for split_name, rows in scored_splits.items()
            }
            split_metrics = {
                split_name: self.trainer.evaluate_rows(rows)
                for split_name, rows in split_predictions.items()
            }
            trained[name] = {
                "model": trained_model,
                "meta": meta,
                "predictions": split_predictions,
                "metrics": split_metrics,
            }
        return trained

    def write_outputs(
        self,
        *,
        output_dir: Path,
        scored_splits: Dict[str, List[Dict[str, Any]]],
        trained: Dict[str, Dict[str, Any]],
        elapsed_sec: float,
    ) -> Dict[str, Any]:
        import torch

        raw_metrics = {
            split_name: self.trainer.evaluate_rows(rows)
            for split_name, rows in scored_splits.items()
        }
        for split_name, rows in scored_splits.items():
            _write_jsonl(output_dir / f"{split_name}.plain_sts.details.jsonl", rows)
        for model_name, data in trained.items():
            torch.save(data["model"].state_dict(), output_dir / f"{model_name}.pt")
            for split_name, rows in data["predictions"].items():
                _write_jsonl(output_dir / f"{split_name}.{model_name}.details.jsonl", rows)

        metrics = {"raw": raw_metrics}
        metrics.update({name: data["metrics"] for name, data in trained.items()})
        summary = {
            "mode": "ce4_train_plain_sts_repeated_score_attention",
            "purpose": "Train post-hoc scalar and repeated-score attention models over plain ModernCE STS scores.",
            "config": asdict(self.config),
            "output_dir": str(output_dir),
            "split_sizes": {split_name: len(rows) for split_name, rows in scored_splits.items()},
            "train_meta": {name: data["meta"] for name, data in trained.items()},
            "metrics": metrics,
            "elapsed_sec": float(elapsed_sec),
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "config.json").write_text(
            json.dumps(asdict(self.config), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary

    def print_summary(self, summary: Dict[str, Any]) -> None:
        print("\n=== Raw Plain STS Validation Metrics ===")
        _print_aggregator_table(summary["metrics"]["raw"]["val"])
        for model_name in ("scalar_linear", "scalar_mlp", "repeated_score_attention"):
            print(f"\n=== {model_name} Validation Metrics ===")
            _print_aggregator_table(summary["metrics"][model_name]["val"])
        print(f"\noutput_dir={summary['output_dir']}")
        print(f"elapsed_sec={float(summary['elapsed_sec']):.2f}")


class ArgumentParserBuilder:
    def parse_args(self) -> PlainStsAttentionConfig:
        parser = argparse.ArgumentParser(
            description="Train post-hoc scalar and repeated-score attention models over plain ModernCE STS scores.",
        )
        parser.add_argument("--gt-train", type=str, default=GT_TRAIN_DEFAULT)
        parser.add_argument("--gt-val", type=str, default=GT_VAL_DEFAULT)
        parser.add_argument("--gt-test", type=str, default=GT_TEST_DEFAULT)
        parser.add_argument("--cache-dir", type=str, default=CACHE_DIR_DEFAULT)
        parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
        parser.add_argument("--run-name", type=str, default="")
        parser.add_argument("--plain-sts-model", type=str, default=PLAIN_STS_MODEL_DEFAULT)
        parser.add_argument("--device", type=str, default="")
        parser.add_argument("--infer-batch-size", type=int, default=64)
        parser.add_argument("--train-batch-size", type=int, default=64)
        parser.add_argument("--max-length", type=int, default=512)
        parser.add_argument("--epochs", type=int, default=500)
        parser.add_argument("--patience", type=int, default=80)
        parser.add_argument("--learning-rate", type=float, default=5e-3)
        parser.add_argument("--weight-decay", type=float, default=1e-3)
        parser.add_argument("--hidden-dim", type=int, default=16)
        parser.add_argument("--token-dim", type=int, default=32)
        parser.add_argument("--num-heads", type=int, default=4)
        parser.add_argument("--num-layers", type=int, default=1)
        parser.add_argument("--dropout", type=float, default=0.10)
        parser.add_argument("--loss-any-boundary-weight", type=float, default=0.0)
        parser.add_argument("--loss-high-boundary-weight", type=float, default=0.0)
        parser.add_argument("--high-threshold", type=float, default=HIGH_THRESHOLD)
        parser.add_argument("--mid-threshold", type=float, default=MID_THRESHOLD)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--overwrite-cache", action="store_true")
        args = parser.parse_args()
        return PlainStsAttentionConfig(
            gt_train=args.gt_train,
            gt_val=args.gt_val,
            gt_test=args.gt_test,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            run_name=args.run_name,
            plain_sts_model=args.plain_sts_model,
            device=args.device,
            infer_batch_size=args.infer_batch_size,
            train_batch_size=args.train_batch_size,
            max_length=args.max_length,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            token_dim=args.token_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            loss_any_boundary_weight=args.loss_any_boundary_weight,
            loss_high_boundary_weight=args.loss_high_boundary_weight,
            high_threshold=args.high_threshold,
            mid_threshold=args.mid_threshold,
            seed=args.seed,
            trust_remote_code=args.trust_remote_code,
            overwrite_cache=args.overwrite_cache,
        )


def main() -> int:
    config = ArgumentParserBuilder().parse_args()
    PlainStsAttentionTrainer(config).train_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
