from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_ID_DEFAULT = "dleemiller/ModernCE-base-sts"
DOMAIN_LISTWISE_DEFAULT = "ce/dataset/splits/llm_distill_domain_listwise_train.jsonl"
METHOD_LISTWISE_DEFAULT = "ce/dataset/splits/llm_distill_method_listwise_train.jsonl"
DOMAIN_LISTWISE_FALLBACK = "ce/dataset/distill/llm_distill_domain_listwise.jsonl"
METHOD_LISTWISE_FALLBACK = "ce/dataset/distill/llm_distill_method_listwise.jsonl"
DOMAIN_VAL_LISTWISE_DEFAULT = "ce/dataset/splits/llm_distill_domain_listwise_val.jsonl"
METHOD_VAL_LISTWISE_DEFAULT = "ce/dataset/splits/llm_distill_method_listwise_val.jsonl"
OUTPUT_DIR_DEFAULT = "ce/models/mse_domain_method"
WANDB_PROJECT_DEFAULT = "ce_mse_distill"
WANDB_MODE_DEFAULT = "online"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_path(value: str) -> Path:
    p = Path(_clean_text(value)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


def _safe_weight(value: Any, default: float = 1.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out <= 0.0:
        return float(default)
    return out


def _prediction_from_logits(logits: torch.Tensor, *, prediction_space: str) -> torch.Tensor:
    mode = _clean_text(prediction_space).lower()
    if mode == "sigmoid":
        return torch.sigmoid(logits)
    return logits


def _per_sample_mse_weights(
    labels: torch.Tensor,
    *,
    high_threshold: float,
    mid_threshold: float,
    high_weight: float,
    mid_weight: float,
    low_weight: float,
) -> torch.Tensor:
    high_mask = labels >= float(high_threshold)
    mid_mask = (labels >= float(mid_threshold)) & (~high_mask)
    weights = torch.full_like(labels, float(low_weight))
    weights = torch.where(mid_mask, torch.full_like(weights, float(mid_weight)), weights)
    weights = torch.where(high_mask, torch.full_like(weights, float(high_weight)), weights)
    return weights


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass
class TrainExample:
    query_text: str
    doc_text: str
    label: float
    aspect: str
    query_key: str


class MSEDataset(Dataset):
    def __init__(self, rows: Sequence[TrainExample]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> TrainExample:
        return self.rows[idx]


class MSECollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, batch: Sequence[TrainExample]) -> Dict[str, Any]:
        queries = [row.query_text for row in batch]
        docs = [row.doc_text for row in batch]
        labels = torch.tensor([float(row.label) for row in batch], dtype=torch.float32)
        aspects = [row.aspect for row in batch]

        enc = self.tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {"enc": enc, "labels": labels, "aspects": aspects}


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = _clean_text(raw)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL row at {path}:{line_no} ({type(exc).__name__}: {exc})") from exc
            if not isinstance(obj, dict):
                continue
            yield obj


def _load_listwise_examples(
    *,
    path: Path,
    aspect: str,
    score_field: str,
    only_selected: bool,
) -> List[TrainExample]:
    examples: List[TrainExample] = []
    aspect_norm = _clean_text(aspect).lower()
    if aspect_norm not in {"domain", "method"}:
        raise RuntimeError(f"Unsupported aspect: {aspect}")
    prefix = "[DOMAIN]" if aspect_norm == "domain" else "[METHOD]"

    for row_idx, row in enumerate(_iter_jsonl(path), start=1):
        query_text = _clean_text(row.get("query_text"))
        if not query_text:
            continue
        grant_id = _clean_text(row.get("grant_id"))
        spec_idx = int(row.get("spec_idx") or 0)
        query_key = f"{grant_id}::{spec_idx}"
        docs = list(row.get("docs") or [])
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            if only_selected and (not bool(doc.get("selected_for_target", False))):
                continue
            doc_text = _clean_text(doc.get("text"))
            if not doc_text:
                continue
            label = _safe_float(doc.get(score_field), default=0.0)
            examples.append(
                TrainExample(
                    query_text=f"{prefix} {query_text}",
                    doc_text=f"{prefix} {doc_text}",
                    label=float(label),
                    aspect=aspect_norm,
                    query_key=query_key,
                )
            )
    return examples


def _split_train_val_by_query(
    *,
    rows: Sequence[TrainExample],
    val_ratio: float,
    seed: int,
) -> Tuple[List[TrainExample], List[TrainExample]]:
    if not rows:
        return [], []
    ratio = float(max(0.0, min(0.9, val_ratio)))
    if ratio <= 0.0:
        return list(rows), []

    keys = sorted(set(r.query_key for r in rows))
    rng = random.Random(int(seed))
    rng.shuffle(keys)
    n_val = int(max(1, round(len(keys) * ratio)))
    n_val = min(n_val, max(1, len(keys) - 1))
    val_keys = set(keys[:n_val])

    train_rows: List[TrainExample] = []
    val_rows: List[TrainExample] = []
    for row in rows:
        if row.query_key in val_keys:
            val_rows.append(row)
        else:
            train_rows.append(row)
    if not train_rows:
        return list(rows), []
    return train_rows, val_rows


def _evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    prediction_space: str,
) -> Dict[str, float]:
    model.eval()
    total_count = 0
    sum_sq = 0.0
    sum_abs = 0.0
    sum_sq_by_aspect = {"domain": 0.0, "method": 0.0}
    sum_abs_by_aspect = {"domain": 0.0, "method": 0.0}
    count_by_aspect = {"domain": 0, "method": 0}

    with torch.no_grad():
        for batch in loader:
            enc = {k: v.to(device) for k, v in dict(batch["enc"]).items()}
            labels = batch["labels"].to(device)
            aspects = list(batch["aspects"])
            out = model(**enc)
            preds = _prediction_from_logits(out.logits.view(-1), prediction_space=prediction_space)
            err = preds - labels

            sq = (err * err).detach().cpu()
            ab = err.abs().detach().cpu()
            labels_cpu = labels.detach().cpu()
            _ = labels_cpu

            batch_size = int(sq.numel())
            total_count += batch_size
            sum_sq += float(sq.sum().item())
            sum_abs += float(ab.sum().item())

            for i, asp in enumerate(aspects):
                key = "domain" if _clean_text(asp).lower() == "domain" else "method"
                sum_sq_by_aspect[key] += float(sq[i].item())
                sum_abs_by_aspect[key] += float(ab[i].item())
                count_by_aspect[key] += 1

    if total_count <= 0:
        return {
            "mse": 0.0,
            "mae": 0.0,
            "domain_mse": 0.0,
            "domain_mae": 0.0,
            "method_mse": 0.0,
            "method_mae": 0.0,
            "count": 0.0,
            "domain_count": 0.0,
            "method_count": 0.0,
        }

    def _safe_div(a: float, b: int) -> float:
        if int(b) <= 0:
            return 0.0
        return float(a / float(b))

    return {
        "mse": _safe_div(sum_sq, total_count),
        "mae": _safe_div(sum_abs, total_count),
        "domain_mse": _safe_div(sum_sq_by_aspect["domain"], count_by_aspect["domain"]),
        "domain_mae": _safe_div(sum_abs_by_aspect["domain"], count_by_aspect["domain"]),
        "method_mse": _safe_div(sum_sq_by_aspect["method"], count_by_aspect["method"]),
        "method_mae": _safe_div(sum_abs_by_aspect["method"], count_by_aspect["method"]),
        "count": float(total_count),
        "domain_count": float(count_by_aspect["domain"]),
        "method_count": float(count_by_aspect["method"]),
    }


def _save_metrics(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MSE-only CE trainer over domain+method listwise distillation files.")
    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    p.add_argument("--domain-listwise", type=str, default=DOMAIN_LISTWISE_DEFAULT)
    p.add_argument("--method-listwise", type=str, default=METHOD_LISTWISE_DEFAULT)
    p.add_argument("--domain-val-listwise", type=str, default=DOMAIN_VAL_LISTWISE_DEFAULT)
    p.add_argument("--method-val-listwise", type=str, default=METHOD_VAL_LISTWISE_DEFAULT)
    p.add_argument("--score-field", type=str, default="teacher_score_raw", choices=["teacher_score_raw", "teacher_score"])
    p.add_argument("--only-selected", action="store_true", help="Use only docs where selected_for_target=true.")
    p.add_argument("--val-ratio", type=float, default=0.05, help="Used only when explicit val files are not provided.")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-bf16", action="store_true")
    p.add_argument("--prediction-space", type=str, default="sigmoid", choices=["sigmoid", "logit"])
    p.add_argument("--high-threshold", type=float, default=0.70)
    p.add_argument("--mid-threshold", type=float, default=0.30)
    p.add_argument("--loss-high-weight", type=float, default=1.50)
    p.add_argument("--loss-mid-weight", type=float, default=1.00)
    p.add_argument("--loss-low-weight", type=float, default=1.40)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--wandb-project", type=str, default=WANDB_PROJECT_DEFAULT)
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--wandb-run-name", type=str, default="")
    p.add_argument("--wandb-mode", type=str, default=WANDB_MODE_DEFAULT, choices=["online", "offline", "disabled"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _set_seed(int(args.seed))
    prediction_space = _clean_text(args.prediction_space).lower()
    if prediction_space not in {"sigmoid", "logit"}:
        prediction_space = "sigmoid"
    high_threshold = float(max(0.0, min(1.0, float(args.high_threshold))))
    mid_threshold = float(max(0.0, min(1.0, float(args.mid_threshold))))
    if mid_threshold > high_threshold:
        mid_threshold = high_threshold
    loss_high_weight = _safe_weight(args.loss_high_weight, default=1.50)
    loss_mid_weight = _safe_weight(args.loss_mid_weight, default=1.00)
    loss_low_weight = _safe_weight(args.loss_low_weight, default=1.40)

    domain_arg = _clean_text(args.domain_listwise)
    method_arg = _clean_text(args.method_listwise)
    domain_path = _resolve_path(domain_arg)
    method_path = _resolve_path(method_arg)
    if (not domain_path.exists()) and (domain_arg == DOMAIN_LISTWISE_DEFAULT):
        fallback = _resolve_path(DOMAIN_LISTWISE_FALLBACK)
        if fallback.exists():
            domain_path = fallback
    if (not method_path.exists()) and (method_arg == METHOD_LISTWISE_DEFAULT):
        fallback = _resolve_path(METHOD_LISTWISE_FALLBACK)
        if fallback.exists():
            method_path = fallback
    if not domain_path.exists():
        raise RuntimeError(f"Domain listwise file not found: {domain_path}")
    if not method_path.exists():
        raise RuntimeError(f"Method listwise file not found: {method_path}")

    out_dir = _resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows_domain = _load_listwise_examples(
        path=domain_path,
        aspect="domain",
        score_field=args.score_field,
        only_selected=bool(args.only_selected),
    )
    train_rows_method = _load_listwise_examples(
        path=method_path,
        aspect="method",
        score_field=args.score_field,
        only_selected=bool(args.only_selected),
    )
    train_rows_all = list(train_rows_domain) + list(train_rows_method)
    if not train_rows_all:
        raise RuntimeError("No training examples were loaded.")

    val_rows_all: List[TrainExample] = []
    domain_val_arg = _clean_text(args.domain_val_listwise)
    method_val_arg = _clean_text(args.method_val_listwise)
    domain_val_path = _resolve_path(domain_val_arg) if domain_val_arg else None
    method_val_path = _resolve_path(method_val_arg) if method_val_arg else None

    has_explicit_val = bool(domain_val_path and domain_val_path.exists() and method_val_path and method_val_path.exists())
    if has_explicit_val:
        val_rows_all = _load_listwise_examples(
            path=domain_val_path,  # type: ignore[arg-type]
            aspect="domain",
            score_field=args.score_field,
            only_selected=bool(args.only_selected),
        ) + _load_listwise_examples(
            path=method_val_path,  # type: ignore[arg-type]
            aspect="method",
            score_field=args.score_field,
            only_selected=bool(args.only_selected),
        )
    else:
        train_rows_all, val_rows_all = _split_train_val_by_query(
            rows=train_rows_all,
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
        )

    rng = random.Random(int(args.seed))
    rng.shuffle(train_rows_all)
    if val_rows_all:
        rng.shuffle(val_rows_all)

    tokenizer = AutoTokenizer.from_pretrained(_clean_text(args.model_id), trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        _clean_text(args.model_id),
        num_labels=1,
        trust_remote_code=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_amp = bool(args.use_bf16 and device.type == "cuda")
    amp_dtype = torch.bfloat16 if use_amp else None

    train_ds = MSEDataset(train_rows_all)
    val_ds = MSEDataset(val_rows_all)
    collate = MSECollator(tokenizer=tokenizer, max_length=int(args.max_length))

    train_loader = DataLoader(
        train_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(args.eval_batch_size)),
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    ) if len(val_ds) > 0 else None

    optim = AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    steps_per_epoch = max(1, len(train_loader))
    updates_per_epoch = max(1, (steps_per_epoch + max(1, int(args.grad_accum)) - 1) // max(1, int(args.grad_accum)))
    total_updates = max(1, updates_per_epoch * max(1, int(args.epochs)))
    warmup_steps = int(max(0, round(float(args.warmup_ratio) * total_updates)))
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    run_meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": _clean_text(args.model_id),
        "device": str(device),
        "score_field": _clean_text(args.score_field),
        "only_selected": bool(args.only_selected),
        "domain_listwise": str(domain_path),
        "method_listwise": str(method_path),
        "domain_val_listwise": str(domain_val_path) if domain_val_path else "",
        "method_val_listwise": str(method_val_path) if method_val_path else "",
        "train_examples": int(len(train_ds)),
        "val_examples": int(len(val_ds)),
        "train_domain_examples": int(sum(1 for r in train_rows_all if r.aspect == "domain")),
        "train_method_examples": int(sum(1 for r in train_rows_all if r.aspect == "method")),
        "val_domain_examples": int(sum(1 for r in val_rows_all if r.aspect == "domain")),
        "val_method_examples": int(sum(1 for r in val_rows_all if r.aspect == "method")),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "grad_accum": int(args.grad_accum),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "warmup_steps": int(warmup_steps),
        "max_length": int(args.max_length),
        "use_bf16": bool(use_amp),
        "prediction_space": prediction_space,
        "high_threshold": float(high_threshold),
        "mid_threshold": float(mid_threshold),
        "loss_high_weight": float(loss_high_weight),
        "loss_mid_weight": float(loss_mid_weight),
        "loss_low_weight": float(loss_low_weight),
        "seed": int(args.seed),
        "wandb_project": _clean_text(args.wandb_project),
        "wandb_entity": _clean_text(args.wandb_entity),
        "wandb_run_name": _clean_text(args.wandb_run_name),
        "wandb_mode": _clean_text(args.wandb_mode),
    }
    _save_metrics(out_dir / "run_config.json", run_meta)

    print(f"device={device}")
    print(f"train_examples={len(train_ds)} val_examples={len(val_ds)}")
    print(f"train_domain={run_meta['train_domain_examples']} train_method={run_meta['train_method_examples']}")
    print(f"score_field={args.score_field} only_selected={bool(args.only_selected)}")
    print(
        f"prediction_space={prediction_space} "
        f"band_thresholds={high_threshold:.2f}/{mid_threshold:.2f} "
        f"loss_weights(high/mid/low)={loss_high_weight:.2f}/{loss_mid_weight:.2f}/{loss_low_weight:.2f}"
    )

    best_metric = float("inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    global_step = 0
    wandb_run = None

    if _clean_text(args.wandb_mode).lower() != "disabled":
        try:
            import wandb  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "W&B logging is enabled but `wandb` is not installed. "
                "Install it with `pip install wandb`, or run with `--wandb-mode disabled`."
            ) from exc
        run_name = _clean_text(args.wandb_run_name) or f"mse-domain-method-{int(time.time())}"
        wandb_kwargs: Dict[str, Any] = {
            "project": _clean_text(args.wandb_project) or WANDB_PROJECT_DEFAULT,
            "name": run_name,
            "config": dict(run_meta),
            "mode": _clean_text(args.wandb_mode).lower(),
            "dir": str(out_dir),
        }
        if _clean_text(args.wandb_entity):
            wandb_kwargs["entity"] = _clean_text(args.wandb_entity)
        wandb_run = wandb.init(**wandb_kwargs)

    try:
        for epoch in range(1, max(1, int(args.epochs)) + 1):
            model.train()
            started = time.time()
            running_loss = 0.0
            seen = 0
            optim.zero_grad(set_to_none=True)
            grad_accum = max(1, int(args.grad_accum))

            for step, batch in enumerate(train_loader, start=1):
                enc = {k: v.to(device) for k, v in dict(batch["enc"]).items()}
                labels = batch["labels"].to(device)
                bs = int(labels.numel())
                seen += bs

                amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
                with amp_ctx:
                    out = model(**enc)
                    preds = _prediction_from_logits(out.logits.view(-1), prediction_space=prediction_space)
                    sq = (preds - labels) ** 2
                    w = _per_sample_mse_weights(
                        labels,
                        high_threshold=high_threshold,
                        mid_threshold=mid_threshold,
                        high_weight=loss_high_weight,
                        mid_weight=loss_mid_weight,
                        low_weight=loss_low_weight,
                    )
                    loss = torch.mean(sq * w)

                running_loss += float(loss.item()) * bs
                (loss / grad_accum).backward()

                if (step % grad_accum == 0) or (step == len(train_loader)):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                    optim.step()
                    scheduler.step()
                    optim.zero_grad(set_to_none=True)
                    global_step += 1

            train_mse = float(running_loss / max(1, seen))
            val_metrics = (
                _evaluate(model=model, loader=val_loader, device=device, prediction_space=prediction_space)
                if val_loader is not None
                else {}
            )
            monitor = float(val_metrics.get("mse", train_mse))
            elapsed = max(1e-6, time.time() - started)

            epoch_dir = out_dir / f"epoch_{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
            epoch_metrics = {
                "epoch": int(epoch),
                "train_mse": train_mse,
                "monitor_mse": monitor,
                "elapsed_sec": float(elapsed),
                "steps": int(global_step),
                "val": val_metrics,
            }
            _save_metrics(epoch_dir / "metrics.json", epoch_metrics)
            history.append(epoch_metrics)

            if monitor <= best_metric:
                best_metric = float(monitor)
                best_epoch = int(epoch)
                best_dir = out_dir / "best"
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                shutil.copytree(epoch_dir, best_dir)

            if wandb_run is not None:
                wandb_log = {
                    "epoch": int(epoch),
                    "train/mse": float(train_mse),
                    "train/monitor_mse": float(monitor),
                    "train/elapsed_sec": float(elapsed),
                    "train/global_step": int(global_step),
                    "lr": float(optim.param_groups[0]["lr"]),
                    "best/monitor_mse": float(best_metric),
                }
                for k, v in dict(val_metrics).items():
                    wandb_log[f"val/{k}"] = float(v)
                wandb_run.log(wandb_log, step=int(epoch))

            print(
                f"epoch={epoch} train_mse={train_mse:.6f} "
                f"val_mse={float(val_metrics.get('mse', 0.0)):.6f} "
                f"val_mae={float(val_metrics.get('mae', 0.0)):.6f} "
                f"time_sec={elapsed:.2f}"
            )
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    summary = {
        "best_epoch": int(best_epoch),
        "best_monitor_mse": float(best_metric),
        "history": history,
        "final_dir": str(final_dir),
        "best_dir": str(out_dir / "best"),
    }
    _save_metrics(out_dir / "train_summary.json", summary)

    print(f"best_epoch={best_epoch}")
    print(f"best_monitor_mse={best_metric:.6f}")
    print(f"best_dir={out_dir / 'best'}")
    print(f"final_dir={final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
