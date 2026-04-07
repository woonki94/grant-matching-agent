from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

WANDB_REPORT_TARGET = "wandb"
WANDB_WATCH_DISABLED = "false"
WANDB_ENV_WATCH = "WANDB_WATCH"
WANDB_ENV_API_KEY = "WANDB_API_KEY"
WANDB_ENV_PROJECT = "WANDB_PROJECT"
WANDB_ENV_ENTITY = "WANDB_ENTITY"
WANDB_API_KEY_MIN_LEN = 30
WANDB_ARG_ENABLE = "--wandb"
WANDB_HELP_ENABLE = "Enable Weights & Biases logging."


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_limit(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if math.isnan(parsed) or math.isinf(parsed):
        return float(default)
    return parsed


def _safe_unit_float(value: Any, *, default: float = 0.0) -> float:
    parsed = _safe_float(value, default=default)
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


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
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        k = _clean_text(key)
        if not k or k not in wanted:
            continue
        v = str(value).strip()
        if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            v = v[1:-1]
        out[k] = v
    return out


def _resolve_dataset_path(path_str: str) -> Path:
    if _clean_text(path_str):
        p = Path(_clean_text(path_str)).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Dataset file not found: {p}")
        return p

    dataset_dir = Path(__file__).resolve().parent / "dataset"
    candidates = sorted(
        dataset_dir.glob("spec_pair_rankdistill_train_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No ranking-distillation dataset found under {dataset_dir}. "
            "Run build_llm_spec_pair_dataset.py first or pass --dataset-jsonl."
        )
    return candidates[0].resolve()


@dataclass
class PointwiseRow:
    query: str
    doc: str
    labels: float
    weights: float
    source: str
    pair_id: str
    candidate_type: str
    domain_bucket: str


def _load_pointwise_from_rankdistill(
    *,
    dataset_jsonl: Path,
    max_pair_rows: int,
    soft_labels: bool,
    min_preference_strength: float,
    seed: int,
) -> Tuple[List[PointwiseRow], Dict[str, Any]]:
    safe_max_pairs = max(0, int(max_pair_rows))
    safe_min_pref = max(0.0, float(min_preference_strength))
    rows: List[PointwiseRow] = []

    pair_rows_total = 0
    pair_rows_used = 0
    pair_rows_skipped_empty = 0
    pair_rows_skipped_pref = 0
    source_counts: Dict[str, int] = {}
    candidate_type_counts: Dict[str, int] = {}
    domain_bucket_counts: Dict[str, int] = {}
    label_sum = 0.0
    label_sq_sum = 0.0
    weight_sum = 0.0

    with dataset_jsonl.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            pair_rows_total += 1
            if safe_max_pairs > 0 and pair_rows_used >= safe_max_pairs:
                break

            try:
                item = json.loads(line)
            except Exception:
                continue

            query = _clean_text(item.get("query"))
            pos = _clean_text(item.get("positive"))
            neg = _clean_text(item.get("negative"))
            if not query or not pos or not neg:
                pair_rows_skipped_empty += 1
                continue

            pref = _safe_unit_float(item.get("preference_strength"), default=0.0)
            if pref < safe_min_pref:
                pair_rows_skipped_pref += 1
                continue

            pair_id = _clean_text(item.get("query_spec_id")) + "::" + _clean_text(item.get("positive_spec_id")) + "::" + _clean_text(
                item.get("negative_spec_id")
            )
            src = _clean_text(item.get("label_source")) or "unknown"
            source_counts[src] = int(source_counts.get(src, 0)) + 1

            grant_domains_raw = item.get("grant_domains")
            grant_domains: List[str] = []
            if isinstance(grant_domains_raw, (list, tuple)):
                for d in grant_domains_raw:
                    tok = _clean_text(d).lower()
                    if tok:
                        grant_domains.append(tok)
            domain_bucket = sorted(set(grant_domains))[0] if grant_domains else "unknown"

            if soft_labels:
                pos_label = _safe_unit_float(item.get("positive_teacher_score"), default=1.0)
                neg_label = _safe_unit_float(item.get("negative_teacher_score"), default=0.0)
                if pos_label <= neg_label:
                    # Keep valid ordering signal.
                    pos_label = min(1.0, neg_label + 1e-3)
            else:
                pos_label = 1.0
                neg_label = 0.0

            # Weight examples by preference strength, with a floor.
            w = max(0.10, pref)
            pos_candidate_type = _clean_text(item.get("positive_candidate_type")) or "unknown"
            neg_candidate_type = _clean_text(item.get("negative_candidate_type")) or "unknown"
            candidate_type_counts[pos_candidate_type] = int(candidate_type_counts.get(pos_candidate_type, 0)) + 1
            candidate_type_counts[neg_candidate_type] = int(candidate_type_counts.get(neg_candidate_type, 0)) + 1
            domain_bucket_counts[domain_bucket] = int(domain_bucket_counts.get(domain_bucket, 0)) + 2
            label_sum += float(pos_label) + float(neg_label)
            label_sq_sum += float(pos_label) ** 2 + float(neg_label) ** 2
            weight_sum += float(w) * 2.0
            rows.append(
                PointwiseRow(
                    query=query,
                    doc=pos,
                    labels=float(pos_label),
                    weights=float(w),
                    source=src,
                    pair_id=pair_id + "::pos",
                    candidate_type=pos_candidate_type,
                    domain_bucket=domain_bucket,
                )
            )
            rows.append(
                PointwiseRow(
                    query=query,
                    doc=neg,
                    labels=float(neg_label),
                    weights=float(w),
                    source=src,
                    pair_id=pair_id + "::neg",
                    candidate_type=neg_candidate_type,
                    domain_bucket=domain_bucket,
                )
            )
            pair_rows_used += 1

    rng = random.Random(int(seed))
    rng.shuffle(rows)

    fallback_like_pair_rows = 0
    for key, count in source_counts.items():
        k = _clean_text(key).lower()
        if "fallback" in k or "partial" in k:
            fallback_like_pair_rows += int(count)

    pointwise_total = int(len(rows))
    label_mean = float(label_sum / pointwise_total) if pointwise_total > 0 else 0.0
    label_var = float((label_sq_sum / pointwise_total) - (label_mean ** 2)) if pointwise_total > 0 else 0.0
    label_std = float(math.sqrt(max(0.0, label_var)))

    stats = {
        "pair_rows_total_seen": int(pair_rows_total),
        "pair_rows_used": int(pair_rows_used),
        "pair_rows_skipped_empty": int(pair_rows_skipped_empty),
        "pair_rows_skipped_pref": int(pair_rows_skipped_pref),
        "pointwise_rows_total": pointwise_total,
        "pointwise_label_mean": float(label_mean),
        "pointwise_label_std": float(label_std),
        "pointwise_weight_mean": (float(weight_sum) / float(pointwise_total) if pointwise_total > 0 else 0.0),
        "fallback_like_pair_rows": int(fallback_like_pair_rows),
        "fallback_like_pair_ratio": (float(fallback_like_pair_rows) / float(pair_rows_used) if pair_rows_used > 0 else 0.0),
        "label_source_counts": dict(source_counts),
        "candidate_type_counts": dict(candidate_type_counts),
        "domain_bucket_counts": dict(domain_bucket_counts),
    }
    return rows, stats


def _split_rows(
    rows: Sequence[PointwiseRow],
    *,
    val_ratio: float,
    seed: int,
) -> Tuple[List[PointwiseRow], List[PointwiseRow]]:
    all_rows = list(rows or [])
    if not all_rows:
        return [], []
    rng = random.Random(int(seed))
    rng.shuffle(all_rows)

    if val_ratio <= 0.0:
        return all_rows, []
    n_total = len(all_rows)
    n_val = int(round(n_total * float(val_ratio)))
    n_val = max(1, min(n_total - 1, n_val)) if n_total >= 2 else 0
    val_rows = all_rows[:n_val]
    train_rows = all_rows[n_val:]
    return train_rows, val_rows


def _rows_to_hf_dicts(rows: Sequence[PointwiseRow]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "query": r.query,
                "doc": r.doc,
                "labels": float(r.labels),
                "weights": float(r.weights),
                "source": r.source,
                "pair_id": r.pair_id,
            }
        )
    return out


def _safe_metric_suffix(value: Any, *, max_len: int = 48) -> str:
    token = _clean_text(value).lower()
    if not token:
        return "unknown"
    out_chars: List[str] = []
    prev_us = False
    for ch in token:
        if ch.isalnum():
            out_chars.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out_chars.append("_")
                prev_us = True
    out = "".join(out_chars).strip("_") or "unknown"
    return out[: int(max_len)]


def _distribution_metrics(values: np.ndarray, *, prefix: str) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p05": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p95": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_p05": float(np.quantile(arr, 0.05)),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p95": float(np.quantile(arr, 0.95)),
    }


def _slice_metrics(
    *,
    preds: np.ndarray,
    labels: np.ndarray,
    slice_values: Sequence[str],
    prefix: str,
    top_n: int = 10,
) -> Dict[str, float]:
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if p.size <= 0 or y.size <= 0 or len(slice_values) != p.size:
        return {}

    by_slice: Dict[str, List[int]] = {}
    for i, raw in enumerate(slice_values):
        key = _clean_text(raw) or "unknown"
        by_slice.setdefault(key, []).append(int(i))

    ranked = sorted(by_slice.items(), key=lambda kv: len(kv[1]), reverse=True)
    selected = ranked[: max(1, int(top_n))]
    out: Dict[str, float] = {
        f"{prefix}_slice_total": float(len(by_slice)),
    }
    for raw_key, idxs in selected:
        idx = np.asarray(idxs, dtype=np.int64)
        key = _safe_metric_suffix(raw_key)
        yp = p[idx]
        yy = y[idx]
        mse = float(np.mean((yp - yy) ** 2)) if yp.size > 0 else 0.0
        mae = float(np.mean(np.abs(yp - yy))) if yp.size > 0 else 0.0
        out[f"{prefix}_{key}_count"] = float(idx.size)
        out[f"{prefix}_{key}_mse"] = mse
        out[f"{prefix}_{key}_mae"] = mae
        out[f"{prefix}_{key}_pred_mean"] = float(np.mean(yp)) if yp.size > 0 else 0.0
        out[f"{prefix}_{key}_label_mean"] = float(np.mean(yy)) if yy.size > 0 else 0.0
    return out


def _ranking_metrics(
    *,
    preds: np.ndarray,
    labels: np.ndarray,
    query_values: Sequence[str],
    ks: Sequence[int] = (1, 3, 5, 10),
    relevance_threshold: float = 0.5,
) -> Dict[str, float]:
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if p.size <= 0 or y.size <= 0 or len(query_values) != p.size:
        return {}

    safe_ks = sorted({max(1, int(k)) for k in list(ks or [])})
    groups: Dict[str, List[int]] = {}
    for i, q in enumerate(query_values):
        groups.setdefault(_clean_text(q) or "unknown", []).append(int(i))

    ndcg_sum = {k: 0.0 for k in safe_ks}
    mrr_sum = {k: 0.0 for k in safe_ks}
    recall_sum = {k: 0.0 for k in safe_ks}
    ndcg_cnt = {k: 0 for k in safe_ks}
    rel_cnt = {k: 0 for k in safe_ks}

    def _dcg(rels: np.ndarray) -> float:
        if rels.size <= 0:
            return 0.0
        ranks = np.arange(2, rels.size + 2, dtype=np.float64)
        gains = np.power(2.0, rels) - 1.0
        return float(np.sum(gains / np.log2(ranks)))

    for _, idxs in groups.items():
        idx = np.asarray(idxs, dtype=np.int64)
        yp = p[idx]
        yy = y[idx]
        if yp.size <= 0:
            continue

        order_pred = np.argsort(-yp)
        order_true = np.argsort(-yy)
        yy_pred_ranked = yy[order_pred]
        yy_true_ranked = yy[order_true]

        relevant_idx = set(int(i) for i, v in enumerate(yy.tolist()) if float(v) >= float(relevance_threshold))
        has_relevant = len(relevant_idx) > 0

        for k in safe_ks:
            kk = min(int(k), int(yy.size))
            if kk <= 0:
                continue
            dcg = _dcg(yy_pred_ranked[:kk])
            idcg = _dcg(yy_true_ranked[:kk])
            if idcg > 0.0:
                ndcg_sum[k] += float(dcg / idcg)
                ndcg_cnt[k] += 1
            if has_relevant:
                rel_cnt[k] += 1
                top_pred = order_pred[:kk].tolist()
                hits = [rank + 1 for rank, local_idx in enumerate(top_pred) if int(local_idx) in relevant_idx]
                mrr_sum[k] += (1.0 / float(min(hits))) if hits else 0.0
                recall_sum[k] += (float(len(set(top_pred) & relevant_idx)) / float(len(relevant_idx)))

    out: Dict[str, float] = {
        "rank_query_groups": float(len(groups)),
    }
    for k in safe_ks:
        out[f"rank_ndcg_at_{k}"] = (ndcg_sum[k] / float(max(1, ndcg_cnt[k])))
        out[f"rank_mrr_at_{k}"] = (mrr_sum[k] / float(max(1, rel_cnt[k])))
        out[f"rank_recall_at_{k}"] = (recall_sum[k] / float(max(1, rel_cnt[k])))
    out["rank_relevance_threshold"] = float(relevance_threshold)
    return out


def _spotcheck_indices(rows: Sequence[PointwiseRow], *, max_items: int = 24) -> List[int]:
    items: List[Tuple[str, int]] = []
    for idx, row in enumerate(list(rows or [])):
        key = "||".join(
            [
                _clean_text(row.pair_id),
                _clean_text(row.source),
                _clean_text(row.candidate_type),
                _clean_text(row.query),
                _clean_text(row.doc),
            ]
        )
        items.append((key, int(idx)))
    items.sort(key=lambda x: x[0])
    return [idx for _, idx in items[: max(1, int(max_items))]]


def train_bge_reranker(
    *,
    dataset_jsonl: Path,
    output_dir: Path,
    model_name: str,
    max_pair_rows: int,
    soft_labels: bool,
    min_preference_strength: float,
    val_ratio: float,
    seed: int,
    num_epochs: float,
    per_device_batch_size: int,
    grad_accum_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    max_length: int,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    save_total_limit: int,
    fp16: bool,
    bf16: bool,
    use_mps: bool,
    num_workers: int,
    resume_from_checkpoint: str,
    use_wandb: bool,
) -> Dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except Exception as e:
        raise RuntimeError(
            "Missing training dependencies. Install packages in your venv:\n"
            "pip install torch transformers datasets accelerate sentence-transformers"
        ) from e

    class WeightedMSETrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
            labels = inputs.pop("labels")
            weights = inputs.pop("weights", None)
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1)
            labels = labels.float().view(-1)
            logits = logits.float().view(-1)
            if weights is None:
                loss = F.mse_loss(logits, labels)
            else:
                w = weights.float().view(-1)
                denom = torch.clamp(w.sum(), min=1e-8)
                loss = torch.sum(w * (logits - labels) ** 2) / denom
            return (loss, outputs) if return_outputs else loss

    class PeriodicLossTrackerCallback(TrainerCallback):
        def __init__(self, *, log_jsonl_path: Path, wandb_enabled: bool, quality_state: Dict[str, Any]):
            self.log_jsonl_path = log_jsonl_path
            self.wandb_enabled = bool(wandb_enabled)
            self.quality_state = quality_state
            self.last_train_loss: Optional[float] = None
            if self.log_jsonl_path.exists():
                try:
                    self.log_jsonl_path.unlink()
                except Exception:
                    pass

        @staticmethod
        def _to_float(value: Any) -> Optional[float]:
            try:
                return float(value)
            except Exception:
                return None

        @staticmethod
        def _fmt(value: Any, *, precision: int = 6) -> str:
            parsed = PeriodicLossTrackerCallback._to_float(value)
            if parsed is None:
                return "n/a"
            return f"{parsed:.{precision}f}"

        @staticmethod
        def _fmt_sci(value: Any) -> str:
            parsed = PeriodicLossTrackerCallback._to_float(value)
            if parsed is None:
                return "n/a"
            return f"{parsed:.3e}"

        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
            payload = dict(logs or {})
            if not payload:
                return control

            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "step": int(getattr(state, "global_step", 0) or 0),
                **payload,
            }
            try:
                with self.log_jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass

            step = int(record.get("step") or 0)
            epoch = self._fmt(record.get("epoch"), precision=4)
            if "loss" in payload:
                self.last_train_loss = self._to_float(payload.get("loss"))
                print(
                    "[train_log] "
                    f"step={step} epoch={epoch} "
                    f"loss={self._fmt(payload.get('loss'))} "
                    f"lr={self._fmt_sci(payload.get('learning_rate'))} "
                    f"grad_norm={self._fmt(payload.get('grad_norm'), precision=4)}"
                )
            elif "eval_loss" in payload:
                gap = None
                eval_loss_val = self._to_float(payload.get("eval_loss"))
                if eval_loss_val is not None and self.last_train_loss is not None:
                    gap = float(eval_loss_val - self.last_train_loss)
                    print(f"[gap_log] step={step} train_eval_loss_gap={self._fmt(gap)}")
                    try:
                        with self.log_jsonl_path.open("a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                        "step": int(step),
                                        "train_eval_loss_gap": float(gap),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
                    if self.wandb_enabled:
                        try:
                            import wandb  # type: ignore

                            if getattr(wandb, "run", None) is not None:
                                wandb.log({"train_eval_loss_gap": float(gap)}, step=int(step))
                        except Exception:
                            pass
                print(
                    "[eval_log] "
                    f"step={step} epoch={epoch} "
                    f"eval_loss={self._fmt(payload.get('eval_loss'))} "
                    f"eval_mse={self._fmt(payload.get('eval_mse'))} "
                    f"eval_mae={self._fmt(payload.get('eval_mae'))} "
                    f"eval_pearson={self._fmt(payload.get('eval_pearson'))}"
                )
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # type: ignore[override]
            if not self.wandb_enabled:
                return control
            rows = list(self.quality_state.get("spotcheck_rows") or [])
            if not rows:
                return control
            try:
                import wandb  # type: ignore

                if getattr(wandb, "run", None) is None:
                    return control
                table = wandb.Table(
                    columns=[
                        "pair_id",
                        "source",
                        "candidate_type",
                        "domain_bucket",
                        "label",
                        "prediction",
                        "abs_error",
                        "query",
                        "doc",
                    ]
                )
                for row in rows:
                    table.add_data(
                        row.get("pair_id", ""),
                        row.get("source", ""),
                        row.get("candidate_type", ""),
                        row.get("domain_bucket", ""),
                        float(row.get("label", 0.0)),
                        float(row.get("prediction", 0.0)),
                        float(row.get("abs_error", 0.0)),
                        row.get("query", ""),
                        row.get("doc", ""),
                    )
                wandb.log({"spotcheck_table": table}, step=int(getattr(state, "global_step", 0) or 0))
            except Exception:
                pass
            return control

    dataset_jsonl = dataset_jsonl.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_log_jsonl = output_dir / "train_loss_log.jsonl"
    wandb_enabled = bool(use_wandb)
    if wandb_enabled:
        env_file = Path(__file__).resolve().parents[1] / ".env"
        loaded_env = _load_selected_env_from_file(
            env_file,
            keys=[WANDB_ENV_API_KEY, WANDB_ENV_PROJECT, WANDB_ENV_ENTITY],
        )
        for k, v in loaded_env.items():
            os.environ.setdefault(k, v)

        api_key = _clean_text(os.getenv(WANDB_ENV_API_KEY))
        if not api_key:
            raise RuntimeError(
                "W&B logging requested but WANDB_API_KEY is not set. "
                f"Set it in shell env or in {env_file}."
            )
        if len(api_key) < int(WANDB_API_KEY_MIN_LEN):
            raise RuntimeError(
                "W&B logging requested but WANDB_API_KEY looks too short "
                f"(len={len(api_key)}). Please paste the full API key."
            )
        if not _clean_text(os.getenv(WANDB_ENV_PROJECT)):
            print("[wandb_warn] WANDB_PROJECT is unset; W&B default project will be used.")
        if not _clean_text(os.getenv(WANDB_ENV_ENTITY)):
            print("[wandb_warn] WANDB_ENTITY is unset; W&B default account will be used.")
        try:
            import wandb  # type: ignore  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "W&B logging requested but `wandb` is not installed. Install packages in your venv:\n"
                "pip install wandb"
            ) from e
        os.environ.setdefault(WANDB_ENV_WATCH, WANDB_WATCH_DISABLED)

    safe_model_name = _clean_text(model_name) or "BAAI/bge-reranker-base"
    safe_seed = int(seed)
    set_seed(safe_seed)
    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    # Prefer CUDA whenever possible; use MPS only when CUDA is unavailable.
    use_cuda_device = bool(cuda_available)
    use_mps_device = bool((not use_cuda_device) and use_mps and mps_available)
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    rows, load_stats = _load_pointwise_from_rankdistill(
        dataset_jsonl=dataset_jsonl,
        max_pair_rows=max_pair_rows,
        soft_labels=bool(soft_labels),
        min_preference_strength=float(min_preference_strength),
        seed=safe_seed,
    )
    if len(rows) < 10:
        raise RuntimeError(f"Not enough rows to train. pointwise_rows_total={len(rows)}")

    train_rows, val_rows = _split_rows(rows, val_ratio=float(max(0.0, val_ratio)), seed=safe_seed)
    if not train_rows:
        raise RuntimeError("Training split is empty.")
    eval_spotcheck_idx = _spotcheck_indices(val_rows, max_items=24)
    quality_state: Dict[str, Any] = {"spotcheck_rows": []}

    train_dicts = _rows_to_hf_dicts(train_rows)
    val_dicts = _rows_to_hf_dicts(val_rows)
    train_ds = Dataset.from_list(train_dicts)
    eval_ds = Dataset.from_list(val_dicts) if val_dicts else None

    tokenizer = AutoTokenizer.from_pretrained(safe_model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        safe_model_name,
        num_labels=1,
        trust_remote_code=True,
    )

    safe_max_len = _safe_limit(max_length, default=256, minimum=32, maximum=4096)

    def _tok(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tokenizer(
            batch["query"],
            batch["doc"],
            truncation=True,
            max_length=int(safe_max_len),
        )

    train_ds = train_ds.map(_tok, batched=True, desc="Tokenizing train")
    if eval_ds is not None:
        eval_ds = eval_ds.map(_tok, batched=True, desc="Tokenizing eval")

    # Keep only tensor-friendly columns for collation.
    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels", "weights"}
    train_drop_cols = [c for c in train_ds.column_names if c not in keep_cols]
    if train_drop_cols:
        train_ds = train_ds.remove_columns(train_drop_cols)
    if eval_ds is not None:
        eval_drop_cols = [c for c in eval_ds.column_names if c not in keep_cols]
        if eval_drop_cols:
            eval_ds = eval_ds.remove_columns(eval_drop_cols)

    class WeightedDataCollator:
        def __init__(self, tok):
            self.tok = tok

        def __call__(self, features):
            labels = [float(f.pop("labels")) for f in features]
            weights = [float(f.pop("weights", 1.0)) for f in features]
            batch = self.tok.pad(
                features,
                padding=True,
                return_tensors="pt",
            )
            batch["labels"] = torch.tensor(labels, dtype=torch.float32)
            batch["weights"] = torch.tensor(weights, dtype=torch.float32)
            return batch

    data_collator = WeightedDataCollator(tokenizer)

    def _metrics(eval_pred) -> Dict[str, float]:
        preds, labels = eval_pred
        p = np.asarray(preds).reshape(-1)
        y = np.asarray(labels).reshape(-1)
        quality_state["spotcheck_rows"] = []
        if p.size <= 0 or y.size <= 0:
            return {"mse": 0.0, "mae": 0.0, "pearson": 0.0}
        mse = float(np.mean((p - y) ** 2))
        mae = float(np.mean(np.abs(p - y)))
        if p.size >= 2 and float(np.std(p)) > 0.0 and float(np.std(y)) > 0.0:
            pearson = float(np.corrcoef(p, y)[0, 1])
        else:
            pearson = 0.0
        out: Dict[str, float] = {"mse": mse, "mae": mae, "pearson": pearson}

        out.update(_distribution_metrics(p, prefix="pred"))
        out.update(_distribution_metrics(y, prefix="label"))
        out["pred_low_ratio"] = float(np.mean((p <= 0.05).astype(np.float32))) if p.size > 0 else 0.0
        out["pred_high_ratio"] = float(np.mean((p >= 0.95).astype(np.float32))) if p.size > 0 else 0.0
        out["pred_collapse_flag"] = 1.0 if float(np.std(p)) < 0.01 else 0.0

        if len(val_rows) == int(p.size):
            query_values = [r.query for r in val_rows]
            source_values = [r.source for r in val_rows]
            candidate_values = [r.candidate_type for r in val_rows]
            domain_values = [r.domain_bucket for r in val_rows]
            out.update(
                _ranking_metrics(
                    preds=p,
                    labels=y,
                    query_values=query_values,
                    ks=(1, 3, 5, 10),
                    relevance_threshold=0.5,
                )
            )
            out.update(_slice_metrics(preds=p, labels=y, slice_values=source_values, prefix="slice_source", top_n=10))
            out.update(_slice_metrics(preds=p, labels=y, slice_values=candidate_values, prefix="slice_candidate", top_n=10))
            out.update(_slice_metrics(preds=p, labels=y, slice_values=domain_values, prefix="slice_domain", top_n=10))

            spot_rows: List[Dict[str, Any]] = []
            for i in eval_spotcheck_idx:
                if i < 0 or i >= len(val_rows):
                    continue
                row = val_rows[i]
                pred_v = float(p[i])
                label_v = float(y[i])
                spot_rows.append(
                    {
                        "pair_id": row.pair_id,
                        "source": row.source,
                        "candidate_type": row.candidate_type,
                        "domain_bucket": row.domain_bucket,
                        "label": label_v,
                        "prediction": pred_v,
                        "abs_error": float(abs(pred_v - label_v)),
                        "query": row.query,
                        "doc": row.doc,
                    }
                )
            quality_state["spotcheck_rows"] = spot_rows

        return out

    do_eval = bool(eval_ds is not None and len(eval_ds) > 0)

    import inspect

    ta_sig = inspect.signature(TrainingArguments.__init__)
    ta_params = set(ta_sig.parameters.keys())

    ta_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": float(num_epochs),
        "per_device_train_batch_size": _safe_limit(per_device_batch_size, default=16, minimum=1, maximum=1024),
        "per_device_eval_batch_size": _safe_limit(per_device_batch_size, default=16, minimum=1, maximum=1024),
        "gradient_accumulation_steps": _safe_limit(grad_accum_steps, default=1, minimum=1, maximum=1024),
        "learning_rate": float(max(1e-8, learning_rate)),
        "weight_decay": float(max(0.0, weight_decay)),
        "warmup_ratio": float(min(1.0, max(0.0, warmup_ratio))),
        "logging_steps": _safe_limit(logging_steps, default=50, minimum=1, maximum=100000),
        "save_steps": _safe_limit(save_steps, default=500, minimum=1, maximum=10000000),
        "eval_steps": _safe_limit(eval_steps, default=500, minimum=1, maximum=10000000),
        "save_total_limit": _safe_limit(save_total_limit, default=3, minimum=1, maximum=100),
        # Mixed precision should be CUDA-only here.
        "fp16": bool(fp16 and use_cuda_device),
        "bf16": bool(bf16 and use_cuda_device),
        "dataloader_num_workers": _safe_limit(num_workers, default=0, minimum=0, maximum=64),
        "dataloader_pin_memory": False,
        "report_to": [WANDB_REPORT_TARGET] if wandb_enabled else [],
        "remove_unused_columns": False,
        "load_best_model_at_end": bool(do_eval),
        "metric_for_best_model": "mse" if do_eval else None,
        "greater_is_better": False if do_eval else None,
        "seed": safe_seed,
    }

    eval_mode = "steps" if do_eval else "no"
    if "evaluation_strategy" in ta_params:
        ta_kwargs["evaluation_strategy"] = eval_mode
    elif "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = eval_mode
    if "use_mps_device" in ta_params:
        ta_kwargs["use_mps_device"] = bool(use_mps_device)
    if "no_cuda" in ta_params:
        ta_kwargs["no_cuda"] = False
    if "use_cpu" in ta_params:
        ta_kwargs["use_cpu"] = False

    # Drop arguments unsupported by the installed transformers version.
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in ta_params and v is not None}
    training_args = TrainingArguments(**ta_kwargs)

    trainer_sig = inspect.signature(WeightedMSETrainer.__init__)
    trainer_params = set(trainer_sig.parameters.keys())
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds if do_eval else None,
        "data_collator": data_collator,
        "compute_metrics": _metrics if do_eval else None,
    }
    # HF Trainer API changed tokenizer -> processing_class in newer versions.
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in trainer_params}
    trainer = WeightedMSETrainer(**trainer_kwargs)
    trainer.add_callback(
        PeriodicLossTrackerCallback(
            log_jsonl_path=loss_log_jsonl,
            wandb_enabled=wandb_enabled,
            quality_state=quality_state,
        )
    )

    dataset_quality_log: Dict[str, float] = {
        "dq_pair_rows_total_seen": float(load_stats.get("pair_rows_total_seen") or 0),
        "dq_pair_rows_used": float(load_stats.get("pair_rows_used") or 0),
        "dq_pair_rows_skipped_empty": float(load_stats.get("pair_rows_skipped_empty") or 0),
        "dq_pair_rows_skipped_pref": float(load_stats.get("pair_rows_skipped_pref") or 0),
        "dq_pointwise_rows_total": float(load_stats.get("pointwise_rows_total") or 0),
        "dq_pointwise_label_mean": float(load_stats.get("pointwise_label_mean") or 0.0),
        "dq_pointwise_label_std": float(load_stats.get("pointwise_label_std") or 0.0),
        "dq_pointwise_weight_mean": float(load_stats.get("pointwise_weight_mean") or 0.0),
        "dq_fallback_like_pair_rows": float(load_stats.get("fallback_like_pair_rows") or 0),
        "dq_fallback_like_pair_ratio": float(load_stats.get("fallback_like_pair_ratio") or 0.0),
        "dq_train_rows": float(len(train_rows)),
        "dq_eval_rows": float(len(val_rows)),
        "dq_eval_ratio_effective": float((len(val_rows) / max(1, len(rows)))),
    }
    for raw_key, value in dict(load_stats.get("label_source_counts") or {}).items():
        dataset_quality_log[f"dq_source_count_{_safe_metric_suffix(raw_key)}"] = float(value or 0)
    for raw_key, value in dict(load_stats.get("candidate_type_counts") or {}).items():
        dataset_quality_log[f"dq_candidate_count_{_safe_metric_suffix(raw_key)}"] = float(value or 0)
    domain_counts = sorted(
        list(dict(load_stats.get("domain_bucket_counts") or {}).items()),
        key=lambda kv: int(kv[1] or 0),
        reverse=True,
    )
    for raw_key, value in domain_counts[:12]:
        dataset_quality_log[f"dq_domain_count_{_safe_metric_suffix(raw_key)}"] = float(value or 0)
    trainer.log(dataset_quality_log)

    train_result = trainer.train(resume_from_checkpoint=_clean_text(resume_from_checkpoint) or None)
    eval_metrics = trainer.evaluate() if do_eval else {}
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_jsonl": str(dataset_jsonl),
        "output_dir": str(output_dir),
        "model_name": safe_model_name,
        "soft_labels": bool(soft_labels),
        "load_stats": load_stats,
        "split": {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "val_ratio": float(max(0.0, val_ratio)),
        },
        "train_metrics": dict(getattr(train_result, "metrics", {}) or {}),
        "eval_metrics": dict(eval_metrics or {}),
        "loss_log_jsonl": str(loss_log_jsonl),
        "params": {
            "epochs": float(num_epochs),
            "batch_size": int(per_device_batch_size),
            "grad_accum_steps": int(grad_accum_steps),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "warmup_ratio": float(warmup_ratio),
            "max_length": int(safe_max_len),
            "fp16": bool(fp16),
            "bf16": bool(bf16),
            "cuda_available": bool(cuda_available),
            "use_cuda_effective": bool(use_cuda_device),
            "use_mps_requested": bool(use_mps),
            "mps_available": bool(mps_available),
            "use_mps_effective": bool(use_mps_device),
            "trainer_device": str(getattr(trainer.args, "device", "")),
            "seed": int(safe_seed),
            "wandb_enabled": bool(wandb_enabled),
        },
    }

    summary_path = output_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    default_out = Path(__file__).resolve().parent / "models" / "bge-reranker-base-finetuned"
    parser = argparse.ArgumentParser(
        description="Fine-tune BAAI/bge-reranker-base using ranking-distilled dataset JSONL."
    )
    parser.add_argument(
        "--dataset-jsonl",
        type=str,
        default="",
        help="Path to rank-distillation dataset JSONL. Default: latest train_cross_encoder/dataset/spec_pair_rankdistill_train_*.jsonl",
    )
    parser.add_argument("--output-dir", type=str, default=str(default_out), help="Output model directory.")
    parser.add_argument("--model-name", type=str, default="BAAI/bge-reranker-base", help="Base model to fine-tune.")
    parser.add_argument("--max-pair-rows", type=int, default=0, help="Cap pair rows loaded from JSONL (0 = all).")
    parser.add_argument("--min-preference-strength", type=float, default=0.0, help="Drop rows below this preference strength.")
    parser.add_argument("--hard-labels", action="store_true", help="Use binary labels (1/0) instead of teacher soft scores.")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=float, default=2.0, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device train/eval batch size.")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio.")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length.")
    parser.add_argument("--logging-steps", type=int, default=50, help="Logging steps.")
    parser.add_argument("--eval-steps", type=int, default=500, help="Eval steps.")
    parser.add_argument("--save-steps", type=int, default=500, help="Save steps.")
    parser.add_argument("--save-total-limit", type=int, default=3, help="Checkpoint retention.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training.")
    parser.add_argument("--use-mps", dest="use_mps", action="store_true", default=True, help="Use MPS when available (default: on).")
    parser.add_argument("--no-mps", dest="use_mps", action="store_false", help="Disable MPS and fall back to CPU/CUDA selection.")
    parser.add_argument("--resume-from-checkpoint", type=str, default="", help="Checkpoint path to resume.")
    parser.add_argument(WANDB_ARG_ENABLE, action="store_true", help=WANDB_HELP_ENABLE)
    parser.add_argument("--json-only", action="store_true", help="Print only JSON summary.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    dataset_jsonl = _resolve_dataset_path(_clean_text(args.dataset_jsonl))
    output_dir = Path(_clean_text(args.output_dir)).expanduser().resolve()

    summary = train_bge_reranker(
        dataset_jsonl=dataset_jsonl,
        output_dir=output_dir,
        model_name=_clean_text(args.model_name) or "BAAI/bge-reranker-base",
        max_pair_rows=int(args.max_pair_rows),
        soft_labels=not bool(args.hard_labels),
        min_preference_strength=float(max(0.0, args.min_preference_strength)),
        val_ratio=float(max(0.0, min(0.5, args.val_ratio))),
        seed=int(args.seed),
        num_epochs=float(max(0.1, args.epochs)),
        per_device_batch_size=int(args.batch_size),
        grad_accum_steps=int(args.grad_accum_steps),
        learning_rate=float(args.learning_rate),
        weight_decay=float(max(0.0, args.weight_decay)),
        warmup_ratio=float(max(0.0, min(1.0, args.warmup_ratio))),
        max_length=int(args.max_length),
        logging_steps=int(args.logging_steps),
        eval_steps=int(args.eval_steps),
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        use_mps=bool(args.use_mps),
        num_workers=int(args.num_workers),
        resume_from_checkpoint=_clean_text(args.resume_from_checkpoint),
        use_wandb=bool(args.wandb),
    )

    if not args.json_only:
        print("bge-reranker-base fine-tuning complete.")
        print(f"  dataset          : {summary.get('dataset_jsonl', '')}")
        print(f"  train rows       : {summary.get('split', {}).get('train_rows', 0)}")
        print(f"  val rows         : {summary.get('split', {}).get('val_rows', 0)}")
        print(f"  output dir       : {summary.get('output_dir', '')}")
        print(f"  summary          : {summary.get('summary_path', '')}")
        print(f"  loss log         : {summary.get('loss_log_jsonl', '')}")
        if bool(summary.get("params", {}).get("wandb_enabled")):
            print("  wandb            : enabled")
        print()

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
