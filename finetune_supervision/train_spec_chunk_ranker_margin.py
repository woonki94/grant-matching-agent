from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import finetune_supervision.train_spec_chunk_ranker_supervision as base


@dataclass
class MarginLossBreakdown:
    total: float
    listwise: float
    margin: float
    query_count: int


def _batch_forward_and_margin_loss(
    *,
    batch: Sequence[base.QueryExample],
    tokenizer,
    model,
    device,
    max_length: int,
    supervision_mode: str,
    listwise_weight: float,
    margin_weight: float,
    margin_value: float,
    max_pairs_per_query: int,
    top_focus_pairs: int,
    rng: random.Random,
):
    import torch
    import torch.nn.functional as F

    mode = base._clean_text(supervision_mode).lower() or "hybrid"
    list_w = float(max(0.0, listwise_weight))
    margin_w = float(max(0.0, margin_weight))
    margin_v = float(max(0.0, margin_value))

    q_texts: List[str] = []
    d_texts: List[str] = []
    spans: List[Tuple[int, int, base.QueryExample]] = []
    for ex in list(batch or []):
        start = len(q_texts)
        for d in list(ex.docs or []):
            q_texts.append(base._clean_text(ex.query))
            d_texts.append(base._clean_text(d))
        end = len(q_texts)
        if end - start >= 2:
            spans.append((start, end, ex))

    if not spans:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return z, MarginLossBreakdown(total=0.0, listwise=0.0, margin=0.0, query_count=0)

    enc = tokenizer(
        q_texts,
        d_texts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
        verbose=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(**enc)
    logits = out.logits
    if logits.ndim == 2 and logits.shape[-1] > 1:
        flat_scores = logits[:, -1].float().view(-1)
    else:
        flat_scores = logits.squeeze(-1).float().view(-1)

    losses: List[torch.Tensor] = []
    list_losses: List[float] = []
    margin_losses: List[float] = []

    for start, end, ex in spans:
        scores = flat_scores[start:end]
        target = torch.tensor([float(x) for x in ex.target_scores], dtype=torch.float32, device=device)
        if target.numel() != scores.numel() or target.numel() < 2:
            continue

        # Listwise (ListNet-style): fit the full ranking distribution.
        list_loss = -(F.softmax(target, dim=0) * F.log_softmax(scores, dim=0)).sum()

        # Margin ranking: enforce (s_i - s_j) >= margin for preferred pairs.
        pair_idx: List[Tuple[int, int, float]] = []
        raw_constraints = list(ex.pair_constraints or [])
        if raw_constraints:
            for i, j, w in raw_constraints:
                ii = int(i)
                jj = int(j)
                if ii < 0 or jj < 0 or ii >= int(scores.numel()) or jj >= int(scores.numel()) or ii == jj:
                    continue
                ww = float(max(0.0, base._safe_float(w, default=1.0)))
                if ww <= 0.0:
                    continue
                pair_idx.append((ii, jj, ww))
            safe_cap = base._safe_limit(
                max_pairs_per_query,
                default=base.DEFAULT_MAX_PAIRS_PER_QUERY,
                minimum=1,
                maximum=4096,
            )
            if len(pair_idx) > safe_cap:
                pair_idx = sorted(pair_idx, key=lambda x: float(x[2]), reverse=True)[:safe_cap]
        if not pair_idx:
            pair_idx = base._build_pair_indices(
                int(scores.numel()),
                max_pairs=int(max_pairs_per_query),
                top_focus=int(top_focus_pairs),
                rng=rng,
            )

        margin_terms: List[torch.Tensor] = []
        margin_weight_sum = 0.0
        for i, j, w in pair_idx:
            # Hinge: max(0, margin - (s_i - s_j))
            margin_terms.append(F.relu(margin_v - (scores[i] - scores[j])) * float(w))
            margin_weight_sum += float(w)
        if margin_terms and margin_weight_sum > 0.0:
            margin_loss = torch.stack(margin_terms).sum() / float(margin_weight_sum)
        else:
            margin_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        if mode == "listwise":
            q_loss = list_w * list_loss
        elif mode == "margin":
            q_loss = margin_w * margin_loss
        else:  # hybrid
            q_loss = (list_w * list_loss) + (margin_w * margin_loss)

        losses.append(q_loss)
        list_losses.append(float(list_loss.detach().item()))
        margin_losses.append(float(margin_loss.detach().item()))

    if not losses:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return z, MarginLossBreakdown(total=0.0, listwise=0.0, margin=0.0, query_count=0)

    batch_loss = torch.stack(losses).mean()
    info = MarginLossBreakdown(
        total=float(batch_loss.detach().item()),
        listwise=float(np.mean(list_losses)) if list_losses else 0.0,
        margin=float(np.mean(margin_losses)) if margin_losses else 0.0,
        query_count=int(len(losses)),
    )
    return batch_loss, info


def _evaluate(
    *,
    examples: Sequence[base.QueryExample],
    tokenizer,
    model,
    device,
    max_length: int,
    eval_batch_size: int,
    supervision_mode: str,
    listwise_weight: float,
    margin_weight: float,
    margin_value: float,
    max_pairs_per_query: int,
    top_focus_pairs: int,
    seed: int,
) -> Dict[str, float]:
    import torch

    rows = list(examples or [])
    if not rows:
        return {
            "eval/loss": 0.0,
            "eval/listwise_loss": 0.0,
            "eval/margin_loss": 0.0,
            "eval/top1_accuracy": 0.0,
            "eval/mrr": 0.0,
            "eval/ndcg@5": 0.0,
            "eval/ndcg@10": 0.0,
            "eval/query_count": 0.0,
        }

    model.eval()
    rng = random.Random(int(seed) + 9919)

    sum_loss = 0.0
    sum_list = 0.0
    sum_margin = 0.0
    sum_q = 0

    score_lists: List[List[float]] = []
    with torch.no_grad():
        for batch in base._iter_batches(rows, batch_size=int(eval_batch_size), shuffle=False, seed=int(seed)):
            loss, info = _batch_forward_and_margin_loss(
                batch=batch,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=int(max_length),
                supervision_mode=supervision_mode,
                listwise_weight=float(listwise_weight),
                margin_weight=float(margin_weight),
                margin_value=float(margin_value),
                max_pairs_per_query=int(max_pairs_per_query),
                top_focus_pairs=int(top_focus_pairs),
                rng=rng,
            )
            _ = loss
            sum_loss += float(info.total) * int(max(1, info.query_count))
            sum_list += float(info.listwise) * int(max(1, info.query_count))
            sum_margin += float(info.margin) * int(max(1, info.query_count))
            sum_q += int(max(1, info.query_count))

            # Separate pass for exact per-query prediction list.
            q_texts: List[str] = []
            d_texts: List[str] = []
            spans: List[Tuple[int, int]] = []
            for ex in list(batch or []):
                s = len(q_texts)
                for d in list(ex.docs or []):
                    q_texts.append(base._clean_text(ex.query))
                    d_texts.append(base._clean_text(d))
                e = len(q_texts)
                spans.append((s, e))

            enc = tokenizer(
                q_texts,
                d_texts,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
                verbose=False,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits
            if logits.ndim == 2 and logits.shape[-1] > 1:
                flat = logits[:, -1].float().view(-1)
            else:
                flat = logits.squeeze(-1).float().view(-1)

            for s, e in spans:
                score_lists.append([float(x) for x in flat[s:e].detach().cpu().tolist()])

    denom = max(1, sum_q)
    ranking = base._compute_ranking_metrics(rows, score_lists)
    return {
        "eval/loss": float(sum_loss) / float(denom),
        "eval/listwise_loss": float(sum_list) / float(denom),
        "eval/margin_loss": float(sum_margin) / float(denom),
        "eval/top1_accuracy": float(ranking.get("top1_accuracy") or 0.0),
        "eval/mrr": float(ranking.get("mrr") or 0.0),
        "eval/ndcg@5": float(ranking.get("ndcg@5") or 0.0),
        "eval/ndcg@10": float(ranking.get("ndcg@10") or 0.0),
        "eval/query_count": float(ranking.get("query_count") or 0.0),
    }


def train(args) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup, set_seed
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()

    dataset_dir = Path(base._clean_text(args.dataset_dir)).expanduser().resolve()
    listwise_path = (
        Path(base._clean_text(args.listwise_jsonl)).expanduser().resolve()
        if base._clean_text(args.listwise_jsonl)
        else None
    )
    pairs_path = (
        Path(base._clean_text(args.pairs_jsonl)).expanduser().resolve()
        if base._clean_text(args.pairs_jsonl)
        else None
    )

    if listwise_path is None and pairs_path is None:
        listwise_path = base._resolve_latest_listwise(dataset_dir)

    if listwise_path is not None:
        examples, data_meta = base._load_listwise_examples(listwise_path, max_queries=int(args.max_queries))
    else:
        examples, data_meta = base._load_pairs_as_examples(
            pairs_path,
            max_queries=int(args.max_queries),
            min_candidates=int(args.min_candidates_per_query),
        )

    if len(examples) < 10:
        raise RuntimeError(f"Not enough examples to train. examples={len(examples)}")

    train_rows, val_rows = base._split_by_query_group(
        examples,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    if not train_rows:
        raise RuntimeError("Training split is empty.")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base._clean_text(args.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))
    device = base._pick_device()

    tokenizer = base._load_tokenizer_stable(base._clean_text(args.model_name) or base.DEFAULT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        base._clean_text(args.model_name) or base.DEFAULT_MODEL_NAME,
        num_labels=1,
        trust_remote_code=True,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    steps_per_epoch = max(1, math.ceil(len(train_rows) / max(1, int(args.batch_size))))
    total_steps = int(math.ceil(float(args.epochs) * float(steps_per_epoch)))
    total_opt_steps = int(math.ceil(total_steps / max(1, int(args.grad_accum))))
    warmup_steps = int(round(float(args.warmup_ratio) * float(total_opt_steps)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(0, int(warmup_steps)),
        num_training_steps=max(1, int(total_opt_steps)),
    )

    wandb = None
    if bool(args.wandb):
        env_file = Path(__file__).resolve().parents[1] / ".env"
        loaded_env = base._load_selected_env_from_file(
            env_file,
            keys=[base.WANDB_ENV_API_KEY, base.WANDB_ENV_PROJECT, base.WANDB_ENV_ENTITY],
        )
        for k, v in loaded_env.items():
            os.environ.setdefault(k, v)

        api_key = base._clean_text(os.getenv(base.WANDB_ENV_API_KEY))
        if not api_key:
            raise RuntimeError(
                f"--wandb set but {base.WANDB_ENV_API_KEY} is missing (shell env or {env_file})."
            )
        if len(api_key) < base.WANDB_API_KEY_MIN_LEN:
            raise RuntimeError(
                f"--wandb set but {base.WANDB_ENV_API_KEY} looks too short (len={len(api_key)})."
            )

        try:
            import wandb as _wandb  # type: ignore
        except Exception as e:
            raise RuntimeError("W&B requested but `wandb` is not installed. Run: pip install wandb") from e

        wandb = _wandb
        run_name = base._clean_text(args.wandb_run_name) or f"spec-chunk-margin-{run_ts}"
        wandb.init(
            project=base._clean_text(os.getenv(base.WANDB_ENV_PROJECT)) or None,
            entity=base._clean_text(os.getenv(base.WANDB_ENV_ENTITY)) or None,
            name=run_name,
            config={
                "model_name": base._clean_text(args.model_name) or base.DEFAULT_MODEL_NAME,
                "supervision_mode": base._clean_text(args.supervision_mode) or "hybrid",
                "epochs": float(args.epochs),
                "batch_size": int(args.batch_size),
                "grad_accum": int(args.grad_accum),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "warmup_ratio": float(args.warmup_ratio),
                "max_length": int(args.max_length),
                "margin_value": float(args.margin_value),
                "margin_weight": float(args.margin_weight),
                "max_pairs_per_query": int(args.max_pairs_per_query),
                "top_focus_pairs": int(args.top_focus_pairs),
                "train_queries": int(len(train_rows)),
                "val_queries": int(len(val_rows)),
                "device": str(device),
                "dataset": str((listwise_path or pairs_path) or ""),
            },
        )

    rng = random.Random(int(args.seed))
    global_step = 0
    opt_step = 0
    best_metric = -1e18
    best_eval: Dict[str, float] = {}
    best_checkpoint_dir = output_dir / "best_checkpoint"

    model.train()
    optimizer.zero_grad(set_to_none=True)

    train_loss_running = 0.0
    train_list_running = 0.0
    train_margin_running = 0.0
    train_count_running = 0

    epoch_count = max(1, int(math.ceil(float(args.epochs))))
    for epoch_idx in range(epoch_count):
        epoch_seed = int(args.seed) + epoch_idx * 1003
        epoch_iter: Iterable[List[base.QueryExample]] = base._iter_batches(
            train_rows,
            batch_size=int(args.batch_size),
            shuffle=True,
            seed=epoch_seed,
        )
        if base._tqdm is not None:
            epoch_iter = base._tqdm(
                epoch_iter,
                total=int(max(1, steps_per_epoch)),
                desc=f"epoch {epoch_idx + 1}/{epoch_count}",
                leave=False,
            )

        for batch in epoch_iter:
            global_step += 1

            loss, info = _batch_forward_and_margin_loss(
                batch=batch,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=int(args.max_length),
                supervision_mode=base._clean_text(args.supervision_mode) or "hybrid",
                listwise_weight=float(args.listwise_weight),
                margin_weight=float(args.margin_weight),
                margin_value=float(args.margin_value),
                max_pairs_per_query=int(args.max_pairs_per_query),
                top_focus_pairs=int(args.top_focus_pairs),
                rng=rng,
            )

            loss_to_backprop = loss / float(max(1, int(args.grad_accum)))
            loss_to_backprop.backward()

            train_loss_running += float(info.total)
            train_list_running += float(info.listwise)
            train_margin_running += float(info.margin)
            train_count_running += int(max(1, info.query_count))

            if global_step % int(max(1, int(args.grad_accum))) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

            if global_step % int(max(1, int(args.logging_steps))) == 0:
                denom = max(1, train_count_running)
                payload = {
                    "train/loss": float(train_loss_running) / float(denom),
                    "train/listwise_loss": float(train_list_running) / float(denom),
                    "train/margin_loss": float(train_margin_running) / float(denom),
                    "train/lr": float(scheduler.get_last_lr()[0]) if scheduler.get_last_lr() else float(args.learning_rate),
                    "train/epoch": float(epoch_idx + (global_step / max(1, steps_per_epoch))),
                    "train/global_step": float(global_step),
                    "train/opt_step": float(opt_step),
                }
                print(
                    f"[train] step={global_step} opt_step={opt_step} "
                    f"loss={payload['train/loss']:.6f} "
                    f"list={payload['train/listwise_loss']:.6f} "
                    f"margin={payload['train/margin_loss']:.6f}"
                )
                if base._tqdm is not None and hasattr(epoch_iter, "set_postfix"):
                    try:
                        epoch_iter.set_postfix(
                            {
                                "loss": f"{payload['train/loss']:.4f}",
                                "list": f"{payload['train/listwise_loss']:.4f}",
                                "margin": f"{payload['train/margin_loss']:.4f}",
                            }
                        )
                    except Exception:
                        pass
                if wandb is not None and getattr(wandb, "run", None) is not None:
                    wandb.log(payload, step=int(global_step))
                train_loss_running = 0.0
                train_list_running = 0.0
                train_margin_running = 0.0
                train_count_running = 0

            should_eval = bool(val_rows) and (global_step % int(max(1, int(args.eval_steps))) == 0)
            if should_eval:
                eval_metrics = _evaluate(
                    examples=val_rows,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_length=int(args.max_length),
                    eval_batch_size=int(args.eval_batch_size),
                    supervision_mode=base._clean_text(args.supervision_mode) or "hybrid",
                    listwise_weight=float(args.listwise_weight),
                    margin_weight=float(args.margin_weight),
                    margin_value=float(args.margin_value),
                    max_pairs_per_query=int(args.max_pairs_per_query),
                    top_focus_pairs=int(args.top_focus_pairs),
                    seed=int(args.seed) + global_step,
                )
                print(
                    f"[eval] step={global_step} loss={eval_metrics['eval/loss']:.6f} "
                    f"top1={eval_metrics['eval/top1_accuracy']:.4f} "
                    f"mrr={eval_metrics['eval/mrr']:.4f} ndcg10={eval_metrics['eval/ndcg@10']:.4f}"
                )
                if wandb is not None and getattr(wandb, "run", None) is not None:
                    wandb.log(eval_metrics, step=int(global_step))

                metric_key = base._clean_text(args.select_best_metric) or "eval/ndcg@10"
                metric_val = float(eval_metrics.get(metric_key) or 0.0)
                if metric_val > float(best_metric):
                    best_metric = float(metric_val)
                    best_eval = dict(eval_metrics)
                    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(best_checkpoint_dir))
                    tokenizer.save_pretrained(str(best_checkpoint_dir))

                model.train()

            if global_step % int(max(1, int(args.save_steps))) == 0:
                ckpt_dir = output_dir / f"checkpoint-step-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(ckpt_dir))
                tokenizer.save_pretrained(str(ckpt_dir))

    final_eval = (
        _evaluate(
            examples=val_rows,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=int(args.max_length),
            eval_batch_size=int(args.eval_batch_size),
            supervision_mode=base._clean_text(args.supervision_mode) or "hybrid",
            listwise_weight=float(args.listwise_weight),
            margin_weight=float(args.margin_weight),
            margin_value=float(args.margin_value),
            max_pairs_per_query=int(args.max_pairs_per_query),
            top_focus_pairs=int(args.top_focus_pairs),
            seed=int(args.seed) + 777,
        )
        if val_rows
        else {
            "eval/loss": 0.0,
            "eval/listwise_loss": 0.0,
            "eval/margin_loss": 0.0,
            "eval/top1_accuracy": 0.0,
            "eval/mrr": 0.0,
            "eval/ndcg@5": 0.0,
            "eval/ndcg@10": 0.0,
            "eval/query_count": 0.0,
        }
    )

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if not best_checkpoint_dir.exists():
        best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(best_checkpoint_dir))
        tokenizer.save_pretrained(str(best_checkpoint_dir))
        best_eval = dict(final_eval)
        best_metric = float(best_eval.get(base._clean_text(args.select_best_metric) or "eval/ndcg@10") or 0.0)

    promoted_dir = output_dir / "promoted_best"
    if promoted_dir.exists():
        shutil.rmtree(promoted_dir)
    shutil.copytree(best_checkpoint_dir, promoted_dir)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": base._clean_text(args.model_name) or base.DEFAULT_MODEL_NAME,
        "supervision_mode": base._clean_text(args.supervision_mode) or "hybrid",
        "dataset": str((listwise_path or pairs_path) or ""),
        "dataset_meta": data_meta,
        "train_query_count": int(len(train_rows)),
        "val_query_count": int(len(val_rows)),
        "device": str(device),
        "global_steps": int(global_step),
        "optimizer_steps": int(opt_step),
        "best_metric_name": base._clean_text(args.select_best_metric) or "eval/ndcg@10",
        "best_metric_value": float(best_metric),
        "best_eval": best_eval,
        "final_eval": final_eval,
        "margin_value": float(args.margin_value),
        "paths": {
            "output_dir": str(output_dir),
            "final_dir": str(final_dir),
            "best_checkpoint_dir": str(best_checkpoint_dir),
            "promoted_best_dir": str(promoted_dir),
        },
        "wandb_enabled": bool(args.wandb),
    }

    summary_path = output_dir / "train_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.summary["best_metric_name"] = str(summary.get("best_metric_name"))
        wandb.summary["best_metric_value"] = float(summary.get("best_metric_value") or 0.0)
        wandb.summary["final_top1_accuracy"] = float(final_eval.get("eval/top1_accuracy") or 0.0)
        wandb.summary["final_ndcg10"] = float(final_eval.get("eval/ndcg@10") or 0.0)
        wandb.finish()

    return summary


def _build_parser() -> argparse.ArgumentParser:
    default_output = (
        Path(__file__).resolve().parent
        / "models"
        / f"spec_chunk_ranker_margin_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    default_dataset_dir = Path(__file__).resolve().parent / "dataset"

    p = argparse.ArgumentParser(
        description=(
            "Spec->chunk cross-encoder trainer with margin ranking objective. "
            "Supports listwise, margin, or hybrid(listwise+margin)."
        )
    )
    p.add_argument("--dataset-dir", type=str, default=str(default_dataset_dir), help="Directory with generated dataset files.")
    p.add_argument("--listwise-jsonl", type=str, default="", help="Optional explicit listwise JSONL path.")
    p.add_argument("--pairs-jsonl", type=str, default="", help="Optional explicit pairs JSONL path (used if listwise not given).")
    p.add_argument("--max-queries", type=int, default=0, help="Max queries to load (0=all).")
    p.add_argument("--min-candidates-per-query", type=int, default=2)

    p.add_argument("--model-name", type=str, default=base.DEFAULT_MODEL_NAME)
    p.add_argument("--output-dir", type=str, default=str(default_output))
    p.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    p.add_argument("--val-ratio", type=float, default=base.DEFAULT_VAL_RATIO)

    p.add_argument("--epochs", type=float, default=base.DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=base.DEFAULT_BATCH_SIZE, help="Queries per training step.")
    p.add_argument("--eval-batch-size", type=int, default=base.DEFAULT_BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=base.DEFAULT_GRAD_ACCUM)
    p.add_argument("--learning-rate", type=float, default=base.DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=base.DEFAULT_WEIGHT_DECAY)
    p.add_argument("--warmup-ratio", type=float, default=base.DEFAULT_WARMUP_RATIO)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=base.DEFAULT_MAX_LENGTH)

    p.add_argument("--supervision-mode", type=str, default="hybrid", choices=["hybrid", "listwise", "margin"])
    p.add_argument("--listwise-weight", type=float, default=1.0)
    p.add_argument("--margin-weight", type=float, default=1.0)
    p.add_argument("--margin-value", type=float, default=0.15)
    p.add_argument("--max-pairs-per-query", type=int, default=base.DEFAULT_MAX_PAIRS_PER_QUERY)
    p.add_argument("--top-focus-pairs", type=int, default=base.DEFAULT_TOP_FOCUS)

    p.add_argument("--logging-steps", type=int, default=base.DEFAULT_LOGGING_STEPS)
    p.add_argument("--eval-steps", type=int, default=base.DEFAULT_EVAL_STEPS)
    p.add_argument("--save-steps", type=int, default=base.DEFAULT_SAVE_STEPS)
    p.add_argument("--select-best-metric", type=str, default="eval/ndcg@10")

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-run-name", type=str, default="")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    summary = train(args)

    print("Spec->chunk margin training complete.")
    print(f"  output dir        : {summary.get('paths', {}).get('output_dir', '')}")
    print(f"  train queries     : {summary.get('train_query_count', 0)}")
    print(f"  val queries       : {summary.get('val_query_count', 0)}")
    print(f"  best metric       : {summary.get('best_metric_name', '')}={summary.get('best_metric_value', 0.0):.6f}")
    print(f"  final top1        : {summary.get('final_eval', {}).get('eval/top1_accuracy', 0.0):.4f}")
    print(f"  final ndcg@10     : {summary.get('final_eval', {}).get('eval/ndcg@10', 0.0):.4f}")
    print(f"  promoted best dir : {summary.get('paths', {}).get('promoted_best_dir', '')}")
    print(f"  summary json      : {Path(summary.get('paths', {}).get('output_dir', '')) / 'train_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

