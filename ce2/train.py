from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore[assignment]


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce2").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce2.data_preparation.utils import ASPECTS, resolve_path  # noqa: E402


MODEL_ID_DEFAULT = "dleemiller/ModernCE-base-sts"
SPLIT_DIR_DEFAULT = "ce2/dataset/splits"
OUTPUT_DIR_DEFAULT = "ce2/models/basic_distill"
WANDB_PROJECT_DEFAULT = "ce2_distill"
WANDB_MODE_DEFAULT = "disabled"
ASPECT_PREFIX = {
    "domain": "[DOMAIN]",
    "method": "[METHOD]",
    "target": "[TARGET]",
}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_ws(value: Any) -> str:
    return " ".join(_clean_text(value).split())


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if not math.isfinite(out):
        out = float(default)
    return max(0.0, min(1.0, out))


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = _clean_text(raw)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {type(exc).__name__}: {exc}") from exc
            if isinstance(obj, dict):
                yield obj


def _format_pair_texts(example: "Example") -> Tuple[str, str]:
    prefix = ASPECT_PREFIX.get(example.aspect, f"[{example.aspect.upper()}]")
    return f"{prefix} {example.query_text}", f"{prefix} {example.doc_text}"


def _model_logits(model: nn.Module, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(**enc).logits.view(-1)


@dataclass(frozen=True)
class Example:
    aspect: str
    query_id: str
    doc_id: str
    pair_id: str
    query_text: str
    doc_text: str
    score: float
    band: str
    source: str


@dataclass(frozen=True)
class PairExample:
    pos: Example
    neg: Example
    margin: float
    weight: float


class PairDataset(Dataset):
    def __init__(self, rows: Sequence[PairExample]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> PairExample:
        return self.rows[idx]


class ListwiseDataset(Dataset):
    def __init__(self, groups: Sequence[Sequence[Example]]) -> None:
        self.groups = [list(group) for group in groups if len(group) > 1]

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> List[Example]:
        return list(self.groups[idx])


class PairCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, batch: Sequence[PairExample]) -> Dict[str, Any]:
        pos_queries: List[str] = []
        pos_docs: List[str] = []
        neg_queries: List[str] = []
        neg_docs: List[str] = []
        for pair in batch:
            q_pos, d_pos = _format_pair_texts(pair.pos)
            q_neg, d_neg = _format_pair_texts(pair.neg)
            pos_queries.append(q_pos)
            pos_docs.append(d_pos)
            neg_queries.append(q_neg)
            neg_docs.append(d_neg)
        queries = pos_queries + neg_queries
        docs = pos_docs + neg_docs
        enc = self.tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "enc": enc,
            "margins": torch.tensor([p.margin for p in batch], dtype=torch.float32),
            "weights": torch.tensor([p.weight for p in batch], dtype=torch.float32),
        }


class ListwiseCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, batch: Sequence[Sequence[Example]]) -> Dict[str, Any]:
        queries: List[str] = []
        docs: List[str] = []
        scores: List[float] = []
        list_sizes: List[int] = []
        for group in batch:
            rows = list(group)
            list_sizes.append(len(rows))
            for row in rows:
                query, doc = _format_pair_texts(row)
                queries.append(query)
                docs.append(doc)
                scores.append(float(row.score))
        enc = self.tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "enc": enc,
            "scores": torch.tensor(scores, dtype=torch.float32),
            "list_sizes": list_sizes,
        }


def _load_split_rows(split_dir: Path, split: str, aspects: Sequence[str]) -> List[Example]:
    rows: List[Example] = []
    for aspect in aspects:
        path = split_dir / f"{aspect}_{split}.jsonl"
        if not path.exists():
            continue
        for obj in _iter_jsonl(path):
            query_text = _normalize_ws(obj.get("query_text"))
            doc_text = _normalize_ws(obj.get("doc_text"))
            query_id = _normalize_ws(obj.get("query_id"))
            doc_id = _normalize_ws(obj.get("doc_id"))
            if not query_text or not doc_text or not query_id or not doc_id:
                continue
            rows.append(
                Example(
                    aspect=aspect,
                    query_id=query_id,
                    doc_id=doc_id,
                    pair_id=_normalize_ws(obj.get("pair_id")) or f"{query_id}::{doc_id}",
                    query_text=query_text,
                    doc_text=doc_text,
                    score=_clamp01(obj.get("score")),
                    band=_normalize_ws(obj.get("band")).lower(),
                    source=_normalize_ws(obj.get("source")),
                )
            )
    return rows


def _group_by_query(rows: Sequence[Example]) -> List[List[Example]]:
    groups: Dict[Tuple[str, str], List[Example]] = {}
    for row in rows:
        groups.setdefault((row.aspect, row.query_id), []).append(row)
    out: List[List[Example]] = []
    for key in sorted(groups):
        group = groups[key]
        group.sort(key=lambda r: (-float(r.score), r.doc_id))
        if len(group) > 1:
            out.append(group)
    return out


def _make_pairs(
    rows: Sequence[Example],
    *,
    seed: int,
    pairs_per_query: int,
    min_score_delta: float,
    min_margin: float,
    max_margin: float,
    margin_scale: float,
) -> List[PairExample]:
    rng = random.Random(int(seed))
    pairs: List[PairExample] = []
    for group in _group_by_query(rows):
        candidates: List[PairExample] = []
        for i, high in enumerate(group):
            for low in group[i + 1 :]:
                delta = float(high.score) - float(low.score)
                if delta < float(min_score_delta):
                    continue
                margin = max(float(min_margin), min(float(max_margin), delta * float(margin_scale)))
                candidates.append(PairExample(pos=high, neg=low, margin=margin, weight=max(0.05, delta)))
        if not candidates:
            continue
        rng.shuffle(candidates)
        pairs.extend(candidates[: max(1, int(pairs_per_query))])
    rng.shuffle(pairs)
    return pairs


def _split_pair_logits(logits: torch.Tensor, pair_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = logits.view(-1)
    pos = flat[:pair_count]
    neg = flat[pair_count : pair_count * 2]
    return pos, neg


def _pairwise_margin_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    margins: torch.Tensor,
    weights: Optional[torch.Tensor],
) -> torch.Tensor:
    raw = F.relu(margins - (pos_logits - neg_logits))
    if weights is None:
        return raw.mean()
    w = weights.clamp(min=0.0)
    return (raw * w).sum() / torch.clamp(w.sum(), min=1e-6)


def _listwise_kl_and_mse(
    *,
    logits_flat: torch.Tensor,
    scores_flat: torch.Tensor,
    list_sizes: Sequence[int],
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kl_parts: List[torch.Tensor] = []
    mse_parts: List[torch.Tensor] = []
    cursor = 0
    temp = max(1e-6, float(temperature))
    for size in list_sizes:
        n = int(size)
        if n <= 1:
            cursor += max(0, n)
            continue
        logits = logits_flat[cursor : cursor + n]
        scores = scores_flat[cursor : cursor + n].clamp(0.0, 1.0)
        cursor += n
        teacher_probs = torch.softmax(scores / temp, dim=-1)
        student_log_probs = torch.log_softmax(logits / temp, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp * temp)
        mse = (torch.sigmoid(logits) - scores).pow(2).mean()
        kl_parts.append(kl)
        mse_parts.append(mse)
    if not kl_parts:
        zero = torch.zeros((), device=logits_flat.device, dtype=logits_flat.dtype)
        return zero, zero
    return torch.stack(kl_parts).mean(), torch.stack(mse_parts).mean()


def _move_encoder_to_device(enc: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in dict(enc).items()}


def _cycle(loader: DataLoader) -> Iterator[Dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def _evaluate_listwise(
    *,
    model: nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    temperature: float,
) -> Dict[str, float]:
    if loader is None:
        return {"kl": 0.0, "mse": 0.0, "count": 0.0}
    model.eval()
    kl_vals: List[float] = []
    mse_vals: List[float] = []
    count = 0
    with torch.no_grad():
        for batch in loader:
            enc = _move_encoder_to_device(batch["enc"], device)
            scores = batch["scores"].to(device)
            logits = _model_logits(model, enc)
            kl_loss, mse_loss = _listwise_kl_and_mse(
                logits_flat=logits,
                scores_flat=scores,
                list_sizes=batch["list_sizes"],
                temperature=temperature,
            )
            kl_vals.append(float(kl_loss.detach().cpu().item()))
            mse_vals.append(float(mse_loss.detach().cpu().item()))
            count += int(scores.numel())
    return {
        "kl": float(sum(kl_vals) / max(1, len(kl_vals))),
        "mse": float(sum(mse_vals) / max(1, len(mse_vals))),
        "count": float(count),
    }


def _evaluate_pairwise(*, model: nn.Module, loader: Optional[DataLoader], device: torch.device) -> Dict[str, float]:
    if loader is None:
        return {"pair_loss": 0.0, "pair_acc": 0.0, "count": 0.0}
    model.eval()
    losses: List[float] = []
    correct = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            enc = _move_encoder_to_device(batch["enc"], device)
            margins = batch["margins"].to(device)
            weights = batch["weights"].to(device)
            logits = _model_logits(model, enc)
            pos_logits, neg_logits = _split_pair_logits(logits, int(margins.shape[0]))
            loss = _pairwise_margin_loss(pos_logits, neg_logits, margins, weights)
            losses.append(float(loss.detach().cpu().item()))
            correct += int((pos_logits > neg_logits).detach().cpu().sum().item())
            count += int(margins.numel())
    return {
        "pair_loss": float(sum(losses) / max(1, len(losses))),
        "pair_acc": float(correct / max(1, count)),
        "count": float(count),
    }


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _save_model_dir(model: nn.Module, tokenizer: Any, path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def _flatten_metrics(prefix: str, obj: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in obj.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_metrics(name, value))
            continue
        if isinstance(value, bool):
            out[name] = float(1 if value else 0)
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[name] = float(value)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Basic CE2 finetuning: Stage 1 pairwise ranking warmup, then Stage 2 "
            "listwise KL distillation + pairwise ranking + small MSE."
        )
    )
    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    p.add_argument("--split-dir", type=str, default=SPLIT_DIR_DEFAULT)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--aspects", type=str, default=",".join(ASPECTS), help="Comma-separated aspects to train.")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage1-epochs", type=int, default=1)
    p.add_argument("--stage2-epochs", type=int, default=3)
    p.add_argument("--stage1-learning-rate", type=float, default=2e-5)
    p.add_argument("--stage2-learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--pair-batch-size", type=int, default=16)
    p.add_argument("--list-batch-size", type=int, default=4, help="Number of query groups per Stage 2 batch.")
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--pairs-per-query", type=int, default=16)
    p.add_argument("--min-pair-delta", type=float, default=0.15)
    p.add_argument("--min-pair-margin", type=float, default=0.05)
    p.add_argument("--max-pair-margin", type=float, default=0.75)
    p.add_argument("--pair-margin-scale", type=float, default=1.0)
    p.add_argument("--teacher-temperature", type=float, default=1.0)
    p.add_argument("--loss-kl-weight", type=float, default=1.0)
    p.add_argument("--loss-pair-weight", type=float, default=0.5)
    p.add_argument("--loss-mse-weight", type=float, default=0.05)
    p.add_argument("--use-bf16", action="store_true")
    p.add_argument("--no-tqdm", action="store_true")
    p.add_argument("--wandb-project", type=str, default=WANDB_PROJECT_DEFAULT)
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--wandb-run-name", type=str, default="")
    p.add_argument("--wandb-mode", type=str, default=WANDB_MODE_DEFAULT, choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-tags", type=str, default="", help="Comma-separated W&B tags.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _set_seed(int(args.seed))

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except Exception as exc:
        raise RuntimeError(
            "Training requires a working Transformers install. In this environment the import failed; "
            "check that `transformers` and `huggingface-hub` versions are compatible."
        ) from exc

    split_dir = resolve_path(PROJECT_ROOT, args.split_dir)
    output_dir = resolve_path(PROJECT_ROOT, args.output_dir)
    aspects = [a.strip().lower() for a in str(args.aspects).split(",") if a.strip()]
    aspects = [a for a in aspects if a in ASPECTS]
    if not aspects:
        raise RuntimeError(f"No valid aspects selected. Valid aspects: {', '.join(ASPECTS)}")

    train_rows = _load_split_rows(split_dir, "train", aspects)
    val_rows = _load_split_rows(split_dir, "val", aspects)
    test_rows = _load_split_rows(split_dir, "test", aspects)
    if not train_rows:
        raise RuntimeError(f"No train rows found in {split_dir}")

    train_pairs = _make_pairs(
        train_rows,
        seed=int(args.seed),
        pairs_per_query=int(args.pairs_per_query),
        min_score_delta=float(args.min_pair_delta),
        min_margin=float(args.min_pair_margin),
        max_margin=float(args.max_pair_margin),
        margin_scale=float(args.pair_margin_scale),
    )
    val_pairs = _make_pairs(
        val_rows,
        seed=int(args.seed) + 7,
        pairs_per_query=int(args.pairs_per_query),
        min_score_delta=float(args.min_pair_delta),
        min_margin=float(args.min_pair_margin),
        max_margin=float(args.max_pair_margin),
        margin_scale=float(args.pair_margin_scale),
    )
    train_groups = _group_by_query(train_rows)
    val_groups = _group_by_query(val_rows)
    test_groups = _group_by_query(test_rows)
    if int(args.stage1_epochs) > 0 and not train_pairs:
        raise RuntimeError("Stage 1 requested, but no pairwise examples could be made. Lower --min-pair-delta.")
    if int(args.stage2_epochs) > 0 and not train_groups:
        raise RuntimeError("Stage 2 requested, but no listwise query groups could be made.")

    output_dir.mkdir(parents=True, exist_ok=True)

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
    use_tqdm = (not bool(args.no_tqdm)) and tqdm is not None
    grad_accum_steps = max(1, int(args.grad_accum_steps))

    pair_collator = PairCollator(tokenizer, int(args.max_length))
    list_collator = ListwiseCollator(tokenizer, int(args.max_length))
    train_pair_loader = DataLoader(
        PairDataset(train_pairs),
        batch_size=max(1, int(args.pair_batch_size)),
        shuffle=True,
        collate_fn=pair_collator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    ) if train_pairs else None
    val_pair_loader = DataLoader(
        PairDataset(val_pairs),
        batch_size=max(1, int(args.eval_batch_size)),
        shuffle=False,
        collate_fn=pair_collator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    ) if val_pairs else None
    train_list_loader = DataLoader(
        ListwiseDataset(train_groups),
        batch_size=max(1, int(args.list_batch_size)),
        shuffle=True,
        collate_fn=list_collator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    ) if train_groups else None
    val_list_loader = DataLoader(
        ListwiseDataset(val_groups),
        batch_size=max(1, int(args.eval_batch_size)),
        shuffle=False,
        collate_fn=list_collator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    ) if val_groups else None
    test_list_loader = DataLoader(
        ListwiseDataset(test_groups),
        batch_size=max(1, int(args.eval_batch_size)),
        shuffle=False,
        collate_fn=list_collator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    ) if test_groups else None

    run_config = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": _clean_text(args.model_id),
        "split_dir": str(split_dir),
        "output_dir": str(output_dir),
        "aspects": aspects,
        "device": str(device),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "stage1_epochs": int(args.stage1_epochs),
        "stage2_epochs": int(args.stage2_epochs),
        "stage1_learning_rate": float(args.stage1_learning_rate),
        "stage2_learning_rate": float(args.stage2_learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "max_length": int(args.max_length),
        "pair_batch_size": int(args.pair_batch_size),
        "list_batch_size": int(args.list_batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "pairs_per_query": int(args.pairs_per_query),
        "min_pair_delta": float(args.min_pair_delta),
        "teacher_temperature": float(args.teacher_temperature),
        "use_bf16": bool(use_amp),
        "loss_weights": {
            "kl": float(args.loss_kl_weight),
            "pair": float(args.loss_pair_weight),
            "mse": float(args.loss_mse_weight),
        },
        "wandb": {
            "project": _clean_text(args.wandb_project),
            "entity": _clean_text(args.wandb_entity),
            "run_name": _clean_text(args.wandb_run_name),
            "mode": _clean_text(args.wandb_mode),
            "tags": [_clean_text(x) for x in str(args.wandb_tags).split(",") if _clean_text(x)],
        },
    }
    _save_json(output_dir / "run_config.json", run_config)
    print(json.dumps(run_config, ensure_ascii=False, indent=2))

    history: List[Dict[str, Any]] = []
    wandb_run = None
    wandb_mode = _clean_text(args.wandb_mode).lower()
    if wandb_mode != "disabled":
        try:
            import wandb  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "W&B logging is enabled but `wandb` is not installed. "
                "Install it or run with --wandb-mode disabled."
            ) from exc
        run_name = _clean_text(args.wandb_run_name) or f"ce2-basic-distill-{int(time.time())}"
        wandb_kwargs: Dict[str, Any] = {
            "project": _clean_text(args.wandb_project) or WANDB_PROJECT_DEFAULT,
            "name": run_name,
            "config": run_config,
            "mode": wandb_mode,
            "dir": str(output_dir),
        }
        tags = [_clean_text(x) for x in str(args.wandb_tags).split(",") if _clean_text(x)]
        if tags:
            wandb_kwargs["tags"] = tags
        if _clean_text(args.wandb_entity):
            wandb_kwargs["entity"] = _clean_text(args.wandb_entity)
        wandb_run = wandb.init(**wandb_kwargs)
        wandb_run.summary["train_rows"] = int(len(train_rows))
        wandb_run.summary["val_rows"] = int(len(val_rows))
        wandb_run.summary["test_rows"] = int(len(test_rows))
        wandb_run.summary["train_pairs"] = int(len(train_pairs))
        wandb_run.summary["train_groups"] = int(len(train_groups))

    best_metric = float("inf")
    best_epoch = 0
    try:
        if train_pair_loader is not None and int(args.stage1_epochs) > 0:
            total_steps = len(train_pair_loader) * int(args.stage1_epochs)
            total_updates = max(1, math.ceil(total_steps / grad_accum_steps))
            warmup_steps = int(round(float(args.warmup_ratio) * total_updates))
            optimizer = AdamW(model.parameters(), lr=float(args.stage1_learning_rate), weight_decay=float(args.weight_decay))
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)
            for epoch in range(1, int(args.stage1_epochs) + 1):
                model.train()
                losses: List[float] = []
                optimizer.zero_grad(set_to_none=True)
                iterator: Iterable[Dict[str, Any]] = train_pair_loader
                if use_tqdm:
                    iterator = tqdm(train_pair_loader, desc=f"Stage1 {epoch}/{int(args.stage1_epochs)}", leave=True)
                for step, batch in enumerate(iterator, start=1):
                    enc = _move_encoder_to_device(batch["enc"], device)
                    margins = batch["margins"].to(device)
                    weights = batch["weights"].to(device)
                    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
                    with amp_ctx:
                        logits = _model_logits(model, enc)
                        pos_logits, neg_logits = _split_pair_logits(logits, int(margins.shape[0]))
                        loss = _pairwise_margin_loss(pos_logits, neg_logits, margins, weights)
                        scaled = loss / float(grad_accum_steps)
                    scaled.backward()
                    losses.append(float(loss.detach().cpu().item()))
                    if step % grad_accum_steps == 0 or step == len(train_pair_loader):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                metrics = _evaluate_pairwise(model=model, loader=val_pair_loader, device=device)
                epoch_metrics = {
                    "stage": 1,
                    "epoch": epoch,
                    "train_pair_loss": float(sum(losses) / max(1, len(losses))),
                    "val_pair_loss": metrics["pair_loss"],
                    "val_pair_acc": metrics["pair_acc"],
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                history.append(epoch_metrics)
                print(json.dumps(epoch_metrics, ensure_ascii=False))
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "stage": 1,
                            "epoch": int(epoch),
                            "train/pair_loss": float(epoch_metrics["train_pair_loss"]),
                            "val/pair_loss": float(epoch_metrics["val_pair_loss"]),
                            "val/pair_acc": float(epoch_metrics["val_pair_acc"]),
                            "lr": float(epoch_metrics["learning_rate"]),
                        },
                        step=int(epoch),
                    )
            _save_model_dir(model, tokenizer, output_dir / "stage1_final")

        if train_list_loader is not None and int(args.stage2_epochs) > 0:
            pair_cycle = _cycle(train_pair_loader) if train_pair_loader is not None else None
            total_steps = len(train_list_loader) * int(args.stage2_epochs)
            total_updates = max(1, math.ceil(total_steps / grad_accum_steps))
            warmup_steps = int(round(float(args.warmup_ratio) * total_updates))
            optimizer = AdamW(model.parameters(), lr=float(args.stage2_learning_rate), weight_decay=float(args.weight_decay))
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)
            for epoch in range(1, int(args.stage2_epochs) + 1):
                model.train()
                loss_totals = {"total": 0.0, "kl": 0.0, "pair": 0.0, "mse": 0.0}
                step_count = 0
                optimizer.zero_grad(set_to_none=True)
                iterator2: Iterable[Dict[str, Any]] = train_list_loader
                if use_tqdm:
                    iterator2 = tqdm(train_list_loader, desc=f"Stage2 {epoch}/{int(args.stage2_epochs)}", leave=True)
                for step, list_batch in enumerate(iterator2, start=1):
                    step_count += 1
                    enc = _move_encoder_to_device(list_batch["enc"], device)
                    scores = list_batch["scores"].to(device)
                    pair_loss = torch.zeros((), device=device)
                    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
                    with amp_ctx:
                        logits = _model_logits(model, enc)
                        kl_loss, mse_loss = _listwise_kl_and_mse(
                            logits_flat=logits,
                            scores_flat=scores,
                            list_sizes=list_batch["list_sizes"],
                            temperature=float(args.teacher_temperature),
                        )
                        if pair_cycle is not None:
                            pair_batch = next(pair_cycle)
                            pair_enc = _move_encoder_to_device(pair_batch["enc"], device)
                            margins = pair_batch["margins"].to(device)
                            weights = pair_batch["weights"].to(device)
                            pair_logits = _model_logits(model, pair_enc)
                            pos_logits, neg_logits = _split_pair_logits(pair_logits, int(margins.shape[0]))
                            pair_loss = _pairwise_margin_loss(pos_logits, neg_logits, margins, weights)
                        total_loss = (
                            float(args.loss_kl_weight) * kl_loss
                            + float(args.loss_pair_weight) * pair_loss
                            + float(args.loss_mse_weight) * mse_loss
                        )
                        scaled_total = total_loss / float(grad_accum_steps)
                    scaled_total.backward()
                    if step % grad_accum_steps == 0 or step == len(train_list_loader):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    loss_totals["total"] += float(total_loss.detach().cpu().item())
                    loss_totals["kl"] += float(kl_loss.detach().cpu().item())
                    loss_totals["pair"] += float(pair_loss.detach().cpu().item())
                    loss_totals["mse"] += float(mse_loss.detach().cpu().item())

                val_list_metrics = _evaluate_listwise(
                    model=model,
                    loader=val_list_loader,
                    device=device,
                    temperature=float(args.teacher_temperature),
                )
                val_pair_metrics = _evaluate_pairwise(model=model, loader=val_pair_loader, device=device)
                monitor = float(val_list_metrics["kl"] + float(args.loss_mse_weight) * val_list_metrics["mse"])
                epoch_metrics = {
                    "stage": 2,
                    "epoch": epoch,
                    "train_total_loss": loss_totals["total"] / max(1, step_count),
                    "train_kl_loss": loss_totals["kl"] / max(1, step_count),
                    "train_pair_loss": loss_totals["pair"] / max(1, step_count),
                    "train_mse_loss": loss_totals["mse"] / max(1, step_count),
                    "val_kl_loss": val_list_metrics["kl"],
                    "val_mse_loss": val_list_metrics["mse"],
                    "val_pair_loss": val_pair_metrics["pair_loss"],
                    "val_pair_acc": val_pair_metrics["pair_acc"],
                    "monitor": monitor,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                history.append(epoch_metrics)
                print(json.dumps(epoch_metrics, ensure_ascii=False))
                epoch_dir = output_dir / f"stage2_epoch_{epoch}"
                _save_model_dir(model, tokenizer, epoch_dir)
                _save_json(epoch_dir / "metrics.json", epoch_metrics)
                improved = monitor <= best_metric
                if improved:
                    best_metric = monitor
                    best_epoch = epoch
                    _save_model_dir(model, tokenizer, output_dir / "best")
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "stage": 2,
                            "epoch": int(epoch),
                            "train/total_loss": float(epoch_metrics["train_total_loss"]),
                            "train/kl_loss": float(epoch_metrics["train_kl_loss"]),
                            "train/pair_loss": float(epoch_metrics["train_pair_loss"]),
                            "train/mse_loss": float(epoch_metrics["train_mse_loss"]),
                            "val/kl_loss": float(epoch_metrics["val_kl_loss"]),
                            "val/mse_loss": float(epoch_metrics["val_mse_loss"]),
                            "val/pair_loss": float(epoch_metrics["val_pair_loss"]),
                            "val/pair_acc": float(epoch_metrics["val_pair_acc"]),
                            "val/monitor": float(epoch_metrics["monitor"]),
                            "best/monitor": float(best_metric),
                            "best/epoch": float(best_epoch),
                            "checkpoint/improved": float(1 if improved else 0),
                            "lr": float(epoch_metrics["learning_rate"]),
                        },
                        step=int(args.stage1_epochs) + int(epoch),
                    )

        test_metrics = _evaluate_listwise(
            model=model,
            loader=test_list_loader,
            device=device,
            temperature=float(args.teacher_temperature),
        )
        _save_model_dir(model, tokenizer, output_dir / "final")
        summary = {
            "best_stage2_epoch": best_epoch,
            "best_monitor": best_metric if math.isfinite(best_metric) else None,
            "test": test_metrics,
            "history": history,
            "final_dir": str(output_dir / "final"),
            "best_dir": str(output_dir / "best"),
        }
        _save_json(output_dir / "train_summary.json", summary)
        if wandb_run is not None:
            wandb_run.summary["best_stage2_epoch"] = int(best_epoch)
            if math.isfinite(best_metric):
                wandb_run.summary["best_monitor"] = float(best_metric)
            for key, value in _flatten_metrics("test", test_metrics).items():
                wandb_run.summary[key] = value
            wandb_run.log(
                _flatten_metrics("test", test_metrics),
                step=int(args.stage1_epochs) + int(args.stage2_epochs) + 1,
            )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
