"""Fine-tune plain single-head ModernCE as the controlled CE5 baseline.

This trainer intentionally reuses CE5's V2 data loading, owner-grouped split,
pair-type mixture, pointwise loss, ranking loss, optimizer schedule, and W&B
configuration.  It changes only the model architecture: the pretrained
ModernCE sequence-classification head is fine-tuned without latent experts or
a learned gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ce5.train import (
    DEFAULT_JUDGMENTS,
    DEFAULT_MODEL_ID,
    GroupedBatchSampler,
    JudgmentCollator,
    JudgmentDataset,
    _autocast_context,
    _confidence_weights,
    _dataset_summary,
    _flatten_numeric_metrics,
    _grant_faculty_high_score_weights,
    _init_wandb,
    _mix_training_examples,
    _move_encoded,
    _nonnegative_float,
    _parameter_counts,
    _parse_training_pair_mix,
    _pointwise_loss,
    _positive_float,
    _positive_int,
    _prediction_metrics,
    _ranking_loss,
    _resolve_device,
    _resolve_path,
    _resolve_precision,
    _set_seed,
    _split_manifest_entry,
    _unit_interval,
    _utc_now,
    _write_json,
    load_judgments,
    split_examples_three_way,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "ce5" / "models" / "single_head_distilled_v2"


def _batch_losses(
    model: Any,
    batch: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Tensor], Tensor, Tensor, int]:
    encoded = _move_encoded(batch["encoded"], device)
    targets = batch["labels"].to(device)
    confidences = batch["confidences"].to(device)
    confidence_weights = _confidence_weights(
        confidences,
        floor=args.confidence_weight_floor,
        power=args.confidence_weight_power,
    )
    score_weights = _grant_faculty_high_score_weights(
        targets,
        batch["pair_types"],
        high_score_threshold=args.high_score_threshold,
        high_score_weight=args.grant_faculty_high_weight,
    )
    pointwise_weights = confidence_weights * score_weights
    output = model(**encoded, return_dict=True)
    logits = output.logits.reshape(-1)
    scores = torch.sigmoid(logits)
    pointwise = _pointwise_loss(
        logits,
        scores,
        targets,
        pointwise_weights,
        loss_name=args.score_loss,
    )
    ranking, ranking_pairs = _ranking_loss(
        logits,
        targets,
        confidence_weights,
        batch["query_ids"],
        min_score_gap=args.ranking_min_score_gap,
        margin=args.ranking_margin,
        max_pairs=args.ranking_max_pairs_per_batch,
    )
    total = pointwise + args.ranking_loss_weight * ranking
    return {
        "total": total,
        "pointwise": pointwise,
        "ranking": ranking,
    }, scores, logits, ranking_pairs


def _parameter_groups(
    model: Any,
    *,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    encoder_parameter_ids = {id(parameter) for parameter in model.base_model.parameters()}
    grouped: dict[tuple[bool, bool], list[Tensor]] = {
        (True, True): [],
        (True, False): [],
        (False, True): [],
        (False, False): [],
    }
    for name, parameter in model.named_parameters():
        is_encoder = id(parameter) in encoder_parameter_ids
        use_decay = parameter.ndim > 1 and not name.endswith("bias")
        grouped[(is_encoder, use_decay)].append(parameter)
    output: list[dict[str, Any]] = []
    for (is_encoder, use_decay), parameters in grouped.items():
        if not parameters:
            continue
        output.append(
            {
                "params": parameters,
                "lr": encoder_lr if is_encoder else head_lr,
                "weight_decay": weight_decay if use_decay else 0.0,
                "group_name": (
                    f"{'encoder' if is_encoder else 'head'}_"
                    f"{'decay' if use_decay else 'no_decay'}"
                ),
            }
        )
    return output


def _set_encoder_trainable(model: Any, trainable: bool) -> None:
    for parameter in model.base_model.parameters():
        parameter.requires_grad = trainable


@torch.no_grad()
def evaluate(
    model: Any,
    loader: DataLoader[Any],
    device: torch.device,
    precision: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    query_ids: list[str] = []
    pair_types: list[str] = []
    loss_sums = {name: 0.0 for name in ("total", "pointwise", "ranking")}
    example_count = 0
    ranking_pairs = 0
    for batch in loader:
        with _autocast_context(device, precision):
            losses, scores, _, batch_ranking_pairs = _batch_losses(
                model,
                batch,
                device,
                args,
            )
        batch_size = int(batch["labels"].numel())
        for name, value in losses.items():
            loss_sums[name] += float(value.detach().float().item()) * batch_size
        predictions.extend(scores.detach().float().cpu().tolist())
        targets.extend(batch["labels"].tolist())
        query_ids.extend(batch["query_ids"])
        pair_types.extend(batch["pair_types"])
        example_count += batch_size
        ranking_pairs += batch_ranking_pairs
    if example_count == 0:
        return {}

    metrics = _prediction_metrics(
        predictions,
        targets,
        query_ids,
        min_score_gap=args.ranking_min_score_gap,
    )
    by_pair_type: dict[str, dict[str, float | int]] = {}
    for pair_type in ("grant_faculty", "grant_grant", "faculty_faculty"):
        indices = [
            index for index, value in enumerate(pair_types) if value == pair_type
        ]
        if not indices:
            continue
        by_pair_type[pair_type] = _prediction_metrics(
            [predictions[index] for index in indices],
            [targets[index] for index in indices],
            [query_ids[index] for index in indices],
            min_score_gap=args.ranking_min_score_gap,
        )
    return {
        "loss": {
            name: total / example_count for name, total in loss_sums.items()
        },
        **metrics,
        "by_pair_type": by_pair_type,
        "ranking_pairs_in_batches": ranking_pairs,
    }


def _train_epoch(
    model: Any,
    loader: DataLoader[Any],
    optimizer: AdamW,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
    precision: torch.dtype,
    args: argparse.Namespace,
    *,
    epoch: int,
    global_step: int,
    wandb_run: Any = None,
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {name: 0.0 for name in ("total", "pointwise", "ranking")}
    step_totals = {name: 0.0 for name in totals}
    example_count = 0
    step_example_count = 0
    ranking_pairs = 0
    step_ranking_pairs = 0
    progress: Iterable[Any] = loader
    if tqdm is not None and not args.no_progress:
        progress = tqdm(loader, desc=f"train epoch {epoch + 1}", dynamic_ncols=True)

    for batch_index, batch in enumerate(progress):
        with _autocast_context(device, precision):
            losses, _, _, batch_ranking_pairs = _batch_losses(
                model,
                batch,
                device,
                args,
            )
            backward_loss = losses["total"] / args.gradient_accumulation_steps
        scaler.scale(backward_loss).backward()
        should_step = (
            (batch_index + 1) % args.gradient_accumulation_steps == 0
            or batch_index + 1 == len(loader)
        )
        batch_size = int(batch["labels"].numel())
        for name, value in losses.items():
            detached = float(value.detach().float().item())
            totals[name] += detached * batch_size
            step_totals[name] += detached * batch_size
        example_count += batch_size
        step_example_count += batch_size
        ranking_pairs += batch_ranking_pairs
        step_ranking_pairs += batch_ranking_pairs

        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if (
                wandb_run is not None
                and global_step % args.wandb_log_every_steps == 0
            ):
                payload = {
                    f"train/loss_{name}": total / max(1, step_example_count)
                    for name, total in step_totals.items()
                }
                payload.update(
                    {
                        "trainer/global_step": global_step,
                        "train/epoch": epoch + 1,
                        "train/ranking_pairs": step_ranking_pairs,
                        "train/encoder_frozen": float(
                            epoch < args.frozen_encoder_epochs
                        ),
                    }
                )
                for group in optimizer.param_groups:
                    group_name = str(group.get("group_name", "parameters"))
                    payload[f"train/learning_rate/{group_name}"] = float(group["lr"])
                wandb_run.log(payload)
            step_totals = {name: 0.0 for name in totals}
            step_example_count = 0
            step_ranking_pairs = 0
        if tqdm is not None and hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{totals['total'] / example_count:.4f}")

    metrics = {
        name: total / max(1, example_count) for name, total in totals.items()
    }
    metrics["ranking_pairs"] = float(ranking_pairs)
    return metrics, global_step


def _save_pretrained(model: Any, tokenizer: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune plain single-head ModernCE on CE5 V2 judgments."
    )
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--min-confidence", type=_unit_interval, default=0.0)
    parser.add_argument(
        "--duplicate-policy",
        choices=("highest-confidence", "latest", "error"),
        default="highest-confidence",
    )
    parser.add_argument("--validation-ratio", type=_unit_interval, default=0.1)
    parser.add_argument("--test-ratio", type=_unit_interval, default=0.1)
    parser.add_argument(
        "--split-group",
        choices=("owner", "target_owner_id", "target_item_id", "pair_id"),
        default="owner",
    )
    parser.add_argument(
        "--training-pair-mix",
        type=_parse_training_pair_mix,
        default={
            "grant_faculty": 0.8,
            "grant_grant": 0.1,
            "faculty_faculty": 0.1,
        },
    )
    parser.add_argument("--use-all-training-pairs", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--score-loss",
        choices=("smooth_l1", "mse", "bce"),
        default="smooth_l1",
    )
    parser.add_argument("--epochs", type=_positive_int, default=5)
    parser.add_argument("--frozen-encoder-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--max-length", type=_positive_int, default=512)
    parser.add_argument("--encoder-lr", type=_positive_float, default=1e-5)
    parser.add_argument("--head-lr", type=_positive_float, default=1e-4)
    parser.add_argument("--weight-decay", type=_nonnegative_float, default=0.01)
    parser.add_argument("--warmup-ratio", type=_unit_interval, default=0.1)
    parser.add_argument("--scheduler", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--gradient-accumulation-steps", type=_positive_int, default=1)
    parser.add_argument("--max-grad-norm", type=_positive_float, default=1.0)
    parser.add_argument("--confidence-weight-floor", type=_unit_interval, default=0.25)
    parser.add_argument("--confidence-weight-power", type=_nonnegative_float, default=1.0)
    parser.add_argument(
        "--grant-faculty-high-weight",
        type=_positive_float,
        default=1.0,
        help=(
            "Pointwise-loss multiplier for G-F targets at or above "
            "--high-score-threshold. Same-side high pairs are not upweighted."
        ),
    )
    parser.add_argument(
        "--high-score-threshold",
        type=_unit_interval,
        default=0.75,
    )
    parser.add_argument("--ranking-loss-weight", type=_nonnegative_float, default=0.2)
    parser.add_argument("--ranking-min-score-gap", type=_unit_interval, default=0.1)
    parser.add_argument("--ranking-margin", type=_nonnegative_float, default=0.0)
    parser.add_argument("--ranking-max-pairs-per-batch", type=int, default=512)

    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="ce5-distillation")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument(
        "--wandb-tags",
        default="ce5,distillation,single-head-baseline",
    )
    parser.add_argument("--wandb-log-every-steps", type=_positive_int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_examples < 0:
        raise ValueError("--max-examples cannot be negative")
    if args.frozen_encoder_epochs < 0:
        raise ValueError("--frozen-encoder-epochs cannot be negative")
    if args.ranking_max_pairs_per_batch < 0:
        raise ValueError("--ranking-max-pairs-per-batch cannot be negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    started = time.time()
    _set_seed(args.seed)
    judgment_path = _resolve_path(args.judgments)
    output_dir = _resolve_path(args.output_dir)
    examples, loading_stats = load_judgments(
        judgment_path,
        min_confidence=args.min_confidence,
        duplicate_policy=args.duplicate_policy,
        max_examples=args.max_examples,
    )
    train_pool, validation_examples, test_examples = split_examples_three_way(
        examples,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        split_group=args.split_group,
        seed=args.seed,
    )
    if args.use_all_training_pairs:
        train_examples = list(train_pool)
        training_mix_mode = "all"
    else:
        train_examples = _mix_training_examples(
            train_pool,
            ratios=args.training_pair_mix,
            seed=args.seed,
        )
        training_mix_mode = "sampled_ratio"
    data_summary = {
        "loading": loading_stats,
        "all": _dataset_summary(examples),
        "train_before_pair_mix": _dataset_summary(train_pool),
        "train": _dataset_summary(train_examples),
        "validation": _dataset_summary(validation_examples),
        "test": _dataset_summary(test_examples),
        "split_group": args.split_group,
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
        "training_mix_mode": training_mix_mode,
        "training_pair_mix_requested": args.training_pair_mix,
    }
    if args.dry_run:
        print(json.dumps(data_summary, indent=2, ensure_ascii=False))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "split_manifest.json",
        {
            "schema_version": "ce5.split-manifest.v1",
            "created_at_utc": _utc_now(),
            "judgments": str(judgment_path),
            "seed": args.seed,
            "split_group": args.split_group,
            "validation_ratio": args.validation_ratio,
            "test_ratio": args.test_ratio,
            "training_mix_mode": training_mix_mode,
            "training_pair_mix_requested": args.training_pair_mix,
            "train_before_pair_mix": _split_manifest_entry(
                train_pool,
                split_group=args.split_group,
            ),
            "train": _split_manifest_entry(
                train_examples,
                split_group=args.split_group,
            ),
            "validation": _split_manifest_entry(
                validation_examples,
                split_group=args.split_group,
            ),
            "test": _split_manifest_entry(
                test_examples,
                split_group=args.split_group,
            ),
        },
    )

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )

    device = _resolve_device(args.device)
    precision = _resolve_precision(args.precision, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    model_source = (
        str(_resolve_path(args.resume_checkpoint))
        if args.resume_checkpoint
        else args.model_id
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        torch_dtype=precision,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    if args.gradient_checkpointing:
        enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
        if not callable(enable_checkpointing):
            raise RuntimeError("The selected model does not support gradient checkpointing")
        enable_checkpointing()

    collator = JudgmentCollator(tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        JudgmentDataset(train_examples),
        batch_sampler=GroupedBatchSampler(
            train_examples,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
        ),
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader: Optional[DataLoader[Any]] = None
    if validation_examples:
        validation_loader = DataLoader(
            JudgmentDataset(validation_examples),
            batch_sampler=GroupedBatchSampler(
                validation_examples,
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.seed,
            ),
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    if args.frozen_encoder_epochs > 0:
        _set_encoder_trainable(model, False)
    optimizer = AdamW(
        _parameter_groups(
            model,
            encoder_lr=args.encoder_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_steps = optimizer_steps_per_epoch * args.epochs
    warmup_steps = round(total_steps * args.warmup_ratio)
    scheduler_factory = (
        get_cosine_schedule_with_warmup
        if args.scheduler == "cosine"
        else get_linear_schedule_with_warmup
    )
    scheduler = scheduler_factory(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and precision == torch.float16,
    )

    run_config = {
        "schema_version": "ce5.single-head-training-run.v1",
        "created_at_utc": _utc_now(),
        "judgments": str(judgment_path),
        "output_dir": str(output_dir),
        "arguments": vars(args) | {
            "judgments": str(args.judgments),
            "output_dir": str(args.output_dir),
            "resume_checkpoint": (
                str(args.resume_checkpoint) if args.resume_checkpoint else None
            ),
        },
        "data": data_summary,
        "model_architecture": "plain_single_head_sequence_classifier",
        "training_objective": {
            "score_loss": args.score_loss,
            "confidence_weight_floor": args.confidence_weight_floor,
            "confidence_weight_power": args.confidence_weight_power,
            "grant_faculty_high_weight": args.grant_faculty_high_weight,
            "high_score_threshold": args.high_score_threshold,
            "ranking_loss_weight": args.ranking_loss_weight,
            "ranking_min_score_gap": args.ranking_min_score_gap,
            "ranking_margin": args.ranking_margin,
            "ranking_max_pairs_per_batch": args.ranking_max_pairs_per_batch,
        },
        "parameter_counts_at_start": _parameter_counts(model),
        "device": str(device),
        "precision": str(precision).replace("torch.", ""),
    }
    _write_json(output_dir / "run_config.json", run_config)
    wandb_run = _init_wandb(args, run_config=run_config, output_dir=output_dir)

    history: list[dict[str, Any]] = []
    best_rmse = math.inf
    global_step = 0
    for epoch in range(args.epochs):
        if epoch == args.frozen_encoder_epochs:
            _set_encoder_trainable(model, True)
        train_metrics, global_step = _train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            precision,
            args,
            epoch=epoch,
            global_step=global_step,
            wandb_run=wandb_run,
        )
        validation_metrics = (
            evaluate(model, validation_loader, device, precision, args)
            if validation_loader is not None
            else {}
        )
        epoch_record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "encoder_frozen": epoch < args.frozen_encoder_epochs,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        _write_json(output_dir / "history.json", {"epochs": history})
        _save_pretrained(model, tokenizer, output_dir / "last")
        grant_faculty_validation = (
            validation_metrics.get("by_pair_type", {}).get("grant_faculty", {})
            if validation_metrics
            else {}
        )
        current_rmse = float(
            grant_faculty_validation.get(
                "rmse",
                validation_metrics.get("rmse", train_metrics["pointwise"]),
            )
        )
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            _save_pretrained(model, tokenizer, output_dir / "best")
            _write_json(output_dir / "best_metrics.json", epoch_record)
        if wandb_run is not None:
            payload = {
                "trainer/global_step": global_step,
                "train_epoch/epoch": epoch + 1,
                "train_epoch/encoder_frozen": float(
                    epoch < args.frozen_encoder_epochs
                ),
            }
            payload.update(_flatten_numeric_metrics("train_epoch", train_metrics))
            payload.update(_flatten_numeric_metrics("validation", validation_metrics))
            wandb_run.log(payload)
        print(json.dumps(epoch_record, ensure_ascii=False))

    _save_pretrained(model, tokenizer, output_dir / "last")
    final_summary = {
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": time.time() - started,
        "epochs": args.epochs,
        "global_step": global_step,
        "best_validation_rmse": best_rmse,
        "best_metric": "validation/by_pair_type/grant_faculty/rmse",
        "best_checkpoint": str(output_dir / "best"),
        "last_checkpoint": str(output_dir / "last"),
    }
    if wandb_run is not None:
        final_summary["wandb_run_id"] = str(getattr(wandb_run, "id", ""))
        final_summary["wandb_run_url"] = str(getattr(wandb_run, "url", ""))
    _write_json(output_dir / "training_summary.json", final_summary)
    if wandb_run is not None:
        wandb_run.summary.update(final_summary)
        wandb_run.finish()
    print(json.dumps(final_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
