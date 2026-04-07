from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# =========================
# Static Config (Simple V2)
# =========================
MODEL_NAME = "BAAI/bge-reranker-base"
OUTPUT_DIR_DEFAULT = Path(__file__).resolve().parent / "models" / "bge-reranker-base-v2-simple"
VAL_RATIO = 0.05
SEED = 42
EPOCHS = 2.0
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
MAX_LENGTH = 256
LOGGING_STEPS = 50
EVAL_STEPS = 500
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 2
MAX_PAIR_ROWS = 0  # 0 = all
SOFT_LABELS = True
HUMAN_CHECK_ROWS = 20
DEDUPE_POINTWISE_ROWS = True

WANDB_ENV_API_KEY = "WANDB_API_KEY"
WANDB_ENV_PROJECT = "WANDB_PROJECT"
WANDB_ENV_ENTITY = "WANDB_ENTITY"
WANDB_API_KEY_MIN_LEN = 30


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_unit_float(value: Any, *, default: float = 0.0) -> float:
    out = _safe_float(value, default=default)
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


def _resolve_dataset_path(path_str: str) -> Path:
    if _clean_text(path_str):
        p = Path(_clean_text(path_str)).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Dataset file not found: {p}")
        return p

    dataset_dir = Path(__file__).resolve().parent / "dataset"
    cands = sorted(
        dataset_dir.glob("spec_pair_rankdistill_train_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(
            f"No dataset under {dataset_dir}. "
            "Run build_llm_spec_pair_dataset.py first or pass --dataset-jsonl."
        )
    return cands[0].resolve()


def _load_selected_env_from_file(env_path: Path, keys: Sequence[str]) -> Dict[str, str]:
    wanted = set(str(k) for k in list(keys or []))
    if not wanted:
        return {}
    if not env_path.exists() or not env_path.is_file():
        return {}

    out: Dict[str, str] = {}
    try:
        raw_lines = env_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {}

    for raw in raw_lines:
        line = str(raw or "").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        k = _clean_text(key)
        if k not in wanted:
            continue
        v = str(value).strip()
        if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            v = v[1:-1]
        out[k] = v
    return out


@dataclass
class PointwiseRow:
    query: str
    doc: str
    label: float
    pair_id: str
    source: str
    group_id: str


def load_pointwise_dataset(dataset_jsonl: Path) -> List[PointwiseRow]:
    rows: List[PointwiseRow] = []
    used_pairs = 0
    seen_pointwise = set()
    with dataset_jsonl.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if MAX_PAIR_ROWS > 0 and used_pairs >= MAX_PAIR_ROWS:
                break
            try:
                item = dict(json.loads(line) or {})
            except Exception:
                continue

            query = _clean_text(item.get("query"))
            pos = _clean_text(item.get("positive"))
            neg = _clean_text(item.get("negative"))
            if not query or not pos or not neg:
                continue

            if SOFT_LABELS:
                pos_label = _safe_unit_float(item.get("positive_teacher_score"), default=1.0)
                neg_label = _safe_unit_float(item.get("negative_teacher_score"), default=0.0)
                if pos_label <= neg_label:
                    pos_label = min(1.0, neg_label + 1e-3)
            else:
                pos_label = 1.0
                neg_label = 0.0

            pair_id = (
                _clean_text(item.get("query_spec_id"))
                + "::"
                + _clean_text(item.get("positive_spec_id"))
                + "::"
                + _clean_text(item.get("negative_spec_id"))
            )
            source = _clean_text(item.get("label_source")) or "unknown"
            group_id = (
                _clean_text(item.get("query_spec_id"))
                or _clean_text(item.get("grant_id"))
                or query
            )

            pos_row = PointwiseRow(
                query=query,
                doc=pos,
                label=float(pos_label),
                pair_id=pair_id + "::pos",
                source=source,
                group_id=group_id,
            )
            neg_row = PointwiseRow(
                query=query,
                doc=neg,
                label=float(neg_label),
                pair_id=pair_id + "::neg",
                source=source,
                group_id=group_id,
            )

            if DEDUPE_POINTWISE_ROWS:
                pos_key = (pos_row.group_id, pos_row.query, pos_row.doc, round(pos_row.label, 6), pos_row.source)
                neg_key = (neg_row.group_id, neg_row.query, neg_row.doc, round(neg_row.label, 6), neg_row.source)
                if pos_key not in seen_pointwise:
                    seen_pointwise.add(pos_key)
                    rows.append(pos_row)
                if neg_key not in seen_pointwise:
                    seen_pointwise.add(neg_key)
                    rows.append(neg_row)
            else:
                rows.append(pos_row)
                rows.append(neg_row)
            used_pairs += 1

    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows


def split_rows(rows: Sequence[PointwiseRow]) -> Tuple[List[PointwiseRow], List[PointwiseRow]]:
    all_rows = list(rows or [])
    if not all_rows:
        return [], []
    n_total = len(all_rows)
    if VAL_RATIO <= 0.0 or n_total < 2:
        return all_rows, []

    group_to_rows: Dict[str, List[PointwiseRow]] = {}
    for r in all_rows:
        gid = _clean_text(r.group_id) or "unknown"
        group_to_rows.setdefault(gid, []).append(r)

    group_ids = list(group_to_rows.keys())
    if len(group_ids) <= 1:
        # Not enough independent groups to make a leakage-safe split.
        return all_rows, []

    rng = random.Random(SEED)
    rng.shuffle(group_ids)

    target_val_rows = int(round(n_total * VAL_RATIO))
    target_val_rows = max(1, min(n_total - 1, target_val_rows))
    selected_val_groups = set()
    cur_val_rows = 0
    for gid in group_ids:
        if cur_val_rows >= target_val_rows:
            break
        selected_val_groups.add(gid)
        cur_val_rows += len(group_to_rows.get(gid, []))

    if len(selected_val_groups) >= len(group_ids):
        # Guarantee at least one train group.
        selected_val_groups.remove(group_ids[-1])

    val_rows: List[PointwiseRow] = []
    train_rows: List[PointwiseRow] = []
    for gid, group_rows in group_to_rows.items():
        if gid in selected_val_groups:
            val_rows.extend(group_rows)
        else:
            train_rows.extend(group_rows)

    # Final safety fallback.
    if not train_rows or not val_rows:
        return all_rows, []

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def _to_hf_dicts(rows: Sequence[PointwiseRow]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "query": r.query,
                "doc": r.doc,
                "labels": float(r.label),
            }
        )
    return out


def run_train(
    *,
    dataset_jsonl: Path,
    output_dir: Path,
    use_wandb: bool,
) -> Dict[str, Any]:
    import inspect

    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

    wandb = None
    if use_wandb:
        env_file = Path(__file__).resolve().parents[1] / ".env"
        loaded_env = _load_selected_env_from_file(env_file, keys=[WANDB_ENV_API_KEY, WANDB_ENV_PROJECT, WANDB_ENV_ENTITY])
        for k, v in loaded_env.items():
            os.environ.setdefault(k, v)

        api_key = _clean_text(os.getenv(WANDB_ENV_API_KEY))
        if not api_key:
            raise RuntimeError(f"--wandb set but {WANDB_ENV_API_KEY} is missing (shell env or {env_file}).")
        if len(api_key) < WANDB_API_KEY_MIN_LEN:
            raise RuntimeError(f"--wandb set but {WANDB_ENV_API_KEY} looks too short (len={len(api_key)}).")

        try:
            import wandb as _wandb  # type: ignore
        except Exception as e:
            raise RuntimeError("W&B requested but package not installed. Run: pip install wandb") from e

        wandb = _wandb
        wandb.init(
            project=_clean_text(os.getenv(WANDB_ENV_PROJECT)) or None,
            entity=_clean_text(os.getenv(WANDB_ENV_ENTITY)) or None,
            name=f"bge-reranker-v2-simple-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_name": MODEL_NAME,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "val_ratio": VAL_RATIO,
                "max_length": MAX_LENGTH,
            },
        )

    set_seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_pointwise_dataset(dataset_jsonl)
    if len(rows) < 10:
        raise RuntimeError(f"Not enough rows to train. rows={len(rows)}")
    train_rows, eval_rows = split_rows(rows)
    if not train_rows:
        raise RuntimeError("Training split is empty.")
    train_groups = {r.group_id for r in train_rows}
    eval_groups = {r.group_id for r in eval_rows}
    group_overlap = len(train_groups & eval_groups)

    train_ds = Dataset.from_list(_to_hf_dicts(train_rows))
    eval_ds = Dataset.from_list(_to_hf_dicts(eval_rows)) if eval_rows else None
    do_eval = bool(eval_ds is not None and len(eval_ds) > 0)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        trust_remote_code=True,
    )

    def _tok(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tokenizer(
            batch["query"],
            batch["doc"],
            truncation=True,
            max_length=int(MAX_LENGTH),
        )

    train_ds = train_ds.map(_tok, batched=True)
    if do_eval:
        eval_ds = eval_ds.map(_tok, batched=True)

    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    drop_train = [c for c in train_ds.column_names if c not in keep_cols]
    if drop_train:
        train_ds = train_ds.remove_columns(drop_train)
    if do_eval and eval_ds is not None:
        drop_eval = [c for c in eval_ds.column_names if c not in keep_cols]
        if drop_eval:
            eval_ds = eval_ds.remove_columns(drop_eval)

    ta_sig = inspect.signature(TrainingArguments.__init__)
    ta_params = set(ta_sig.parameters.keys())
    eval_mode = "steps" if do_eval else "no"
    ta_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": float(EPOCHS),
        "per_device_train_batch_size": int(BATCH_SIZE),
        "per_device_eval_batch_size": int(BATCH_SIZE),
        "gradient_accumulation_steps": 1,
        "learning_rate": float(LEARNING_RATE),
        "weight_decay": float(WEIGHT_DECAY),
        "warmup_ratio": float(WARMUP_RATIO),
        "logging_steps": int(LOGGING_STEPS),
        "save_steps": int(SAVE_STEPS),
        "eval_steps": int(EVAL_STEPS),
        "save_total_limit": int(SAVE_TOTAL_LIMIT),
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "report_to": [],
        "remove_unused_columns": False,
        "load_best_model_at_end": bool(do_eval),
        "metric_for_best_model": "eval_loss" if do_eval else None,
        "greater_is_better": False if do_eval else None,
        "seed": int(SEED),
    }
    if "evaluation_strategy" in ta_params:
        ta_kwargs["evaluation_strategy"] = eval_mode
    elif "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = eval_mode
    if "no_cuda" in ta_params:
        ta_kwargs["no_cuda"] = False
    if "use_cpu" in ta_params:
        ta_kwargs["use_cpu"] = False
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in ta_params and v is not None}
    training_args = TrainingArguments(**ta_kwargs)

    class SimpleWandbLossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
            if wandb is None or getattr(wandb, "run", None) is None:
                return control
            payload = dict(logs or {})
            if not payload:
                return control
            to_log: Dict[str, float] = {}
            if "loss" in payload:
                to_log["train/loss"] = float(payload["loss"])
            if "eval_loss" in payload:
                to_log["eval/loss"] = float(payload["eval_loss"])
            if to_log:
                wandb.log(to_log, step=int(getattr(state, "global_step", 0) or 0))
            return control

    class ExplicitMSETrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if logits.ndim == 2 and logits.shape[-1] > 1:
                logits = logits[:, -1]
            else:
                logits = logits.squeeze(-1)
            y_hat = logits.float().view(-1)
            y = labels.float().view(-1)
            loss = F.mse_loss(y_hat, y)
            return (loss, outputs) if return_outputs else loss

    trainer = ExplicitMSETrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if do_eval else None,
        tokenizer=tokenizer,
    )
    trainer.add_callback(SimpleWandbLossCallback())

    train_result = trainer.train()
    eval_metrics = trainer.evaluate() if do_eval else {}
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if do_eval and eval_ds is not None and wandb is not None and getattr(wandb, "run", None) is not None:
        pred = trainer.predict(eval_ds)
        preds = np.asarray(pred.predictions).reshape(-1)
        labels = np.asarray(pred.label_ids).reshape(-1)
        if preds.size == len(eval_rows):
            abs_err = np.abs(preds - labels)
            worst_idx = np.argsort(-abs_err)[: max(1, int(HUMAN_CHECK_ROWS))]
            table = wandb.Table(columns=["pair_id", "source", "label", "prediction", "abs_error", "query", "doc"])
            for i in worst_idx.tolist():
                r = eval_rows[int(i)]
                table.add_data(
                    r.pair_id,
                    r.source,
                    float(labels[int(i)]),
                    float(preds[int(i)]),
                    float(abs_err[int(i)]),
                    r.query,
                    r.doc,
                )
            wandb.log({"human_check": table}, step=int(getattr(trainer.state, "global_step", 0) or 0))

    if wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.finish()

    summary = {
        "dataset_jsonl": str(dataset_jsonl),
        "output_dir": str(output_dir),
        "total_rows": int(len(rows)),
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "train_groups": int(len(train_groups)),
        "eval_groups": int(len(eval_groups)),
        "group_overlap": int(group_overlap),
        "train_loss": float((getattr(train_result, "metrics", {}) or {}).get("train_loss", 0.0)),
        "eval_loss": float((eval_metrics or {}).get("eval_loss", 0.0)) if do_eval else None,
        "wandb_enabled": bool(use_wandb),
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simple v2 training: load dataset, fine-tune with MSE, eval, optional W&B.")
    p.add_argument("--dataset-jsonl", type=str, default="", help="Path to dataset JSONL. Default: latest dataset file.")
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR_DEFAULT), help="Output model directory.")
    p.add_argument("--wandb", action="store_true", help="Log train/eval loss + human_check to W&B.")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    dataset_jsonl = _resolve_dataset_path(_clean_text(args.dataset_jsonl))
    output_dir = Path(_clean_text(args.output_dir)).expanduser().resolve()

    summary = run_train(
        dataset_jsonl=dataset_jsonl,
        output_dir=output_dir,
        use_wandb=bool(args.wandb),
    )

    print("Simple v2 training complete.")
    print(f"  dataset     : {summary.get('dataset_jsonl', '')}")
    print(f"  output dir  : {summary.get('output_dir', '')}")
    print(f"  total rows  : {summary.get('total_rows', 0)}")
    print(f"  train rows  : {summary.get('train_rows', 0)}")
    print(f"  eval rows   : {summary.get('eval_rows', 0)}")
    print(f"  train groups: {summary.get('train_groups', 0)}")
    print(f"  eval groups : {summary.get('eval_groups', 0)}")
    print(f"  overlap grp : {summary.get('group_overlap', 0)}")
    print(f"  train loss  : {summary.get('train_loss', 0.0)}")
    if summary.get("eval_loss") is not None:
        print(f"  eval loss   : {summary.get('eval_loss', 0.0)}")
    print(f"  wandb       : {'enabled' if summary.get('wandb_enabled') else 'disabled'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
