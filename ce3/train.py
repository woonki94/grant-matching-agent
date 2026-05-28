from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# ModernBERT may invoke torch.compile for embeddings. In this training setup the
# aspect-head routing wrapper can make Dynamo tracing fail with fake CPU/CUDA
# device propagation errors, so keep eager execution unless explicitly changed.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce3").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce3.aspect_modeling import (  # noqa: E402
    ASPECTS,
    aspect_from_prefixed_query,
    aspect_id_from_name,
    format_aspect_pair,
    load_sequence_classifier_model,
    model_logits,
)


MODEL_ID_DEFAULT = "dleemiller/ModernCE-base-sts"
SPLIT_DIR_DEFAULT = "ce3/dataset/splits"
OUTPUT_DIR_DEFAULT = "ce3/models/aspect_reranker"
PAIR_TYPE_WEIGHT_MAP_DEFAULT = "default=1.0,llm_disagreement=1.15,strong_vs_boundary=1.05,strong_vs_weak=0.95,strong_vs_hard=1.0"


def _wandb_enabled(args: argparse.Namespace) -> bool:
    return _clean_text(getattr(args, "wandb_mode", "")).lower() not in {"", "disabled", "off", "false", "none"}


def _flatten_metrics(prefix: str, obj: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in obj.items():
        metric_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_metrics(metric_key, value))
            continue
        if isinstance(value, bool):
            out[metric_key] = float(value)
            continue
        if isinstance(value, (int, float)):
            out[metric_key] = float(value)
    return out


def _init_wandb(args: argparse.Namespace, *, run_config: Dict[str, Any]) -> Any:
    if not _wandb_enabled(args):
        return None
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError("W&B logging requested, but wandb is not installed in this environment.") from exc
    tags = [_clean_text(x) for x in _clean_text(args.wandb_tags).split(",") if _clean_text(x)]
    return wandb.init(
        project=_clean_text(args.wandb_project) or "ce3_distill",
        entity=_clean_text(args.wandb_entity) or None,
        name=_clean_text(args.wandb_run_name) or None,
        mode=_clean_text(args.wandb_mode) or "online",
        tags=tags or None,
        config=run_config,
    )


def _wandb_log(wandb_run: Any, payload: Dict[str, Any], *, step: Optional[int] = None) -> None:
    if wandb_run is None:
        return
    wandb_run.log(_flatten_metrics("", payload), step=step)


@dataclass(frozen=True)
class PairExample:
    query_text: str
    pos_text: str
    neg_text: str
    aspect: str
    teacher_margin: float
    teacher_pos_score: float
    teacher_neg_score: float
    pair_type: str


class PairDataset(Dataset):
    def __init__(self, rows: Sequence[PairExample]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> PairExample:
        return self.rows[idx]


class ListDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_path(value: Any) -> Path:
    path = Path(_clean_text(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _clamp_01(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


def _cluster_id_from_band_or_score(band: Any, score: Any, *, high_threshold: float, mid_threshold: float) -> int:
    b = _clean_text(band).lower()
    if b == "high":
        return 2
    if b == "mid":
        return 1
    if b == "low":
        return 0
    s = _clamp_01(score)
    if s >= float(high_threshold):
        return 2
    if s >= float(mid_threshold):
        return 1
    return 0


def _parse_pair_type_weight_map(value: Any) -> Tuple[Dict[str, float], float]:
    default = 1.0
    out: Dict[str, float] = {}
    for token in _clean_text(value).split(","):
        if "=" not in token:
            continue
        key, raw_val = token.split("=", 1)
        key = _clean_text(key).lower()
        if not key:
            continue
        try:
            parsed = max(0.0, min(10.0, float(raw_val)))
        except Exception:
            continue
        if key == "default":
            default = parsed
        else:
            out[key] = parsed
    return out, default


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_pair_rows(path: Path) -> List[PairExample]:
    out: List[PairExample] = []
    for row in _load_jsonl(path):
        query = _clean_text(row.get("query_text"))
        pos = _clean_text(row.get("pos_text"))
        neg = _clean_text(row.get("neg_text"))
        aspect = _clean_text(row.get("aspect")) or aspect_from_prefixed_query(query)
        if not query or not pos or not neg or aspect not in ASPECTS:
            continue
        out.append(
            PairExample(
                query_text=query,
                pos_text=pos,
                neg_text=neg,
                aspect=aspect,
                teacher_margin=max(0.0, float(row.get("teacher_margin", 0.0) or 0.0)),
                teacher_pos_score=_clamp_01(row.get("teacher_pos_score")),
                teacher_neg_score=_clamp_01(row.get("teacher_neg_score")),
                pair_type=_clean_text(row.get("pair_type")) or "unknown",
            )
        )
    return out


class PairCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_length: int,
        *,
        pair_type_weights: Dict[str, float],
        default_pair_weight: float,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.pair_type_weights = dict(pair_type_weights)
        self.default_pair_weight = float(default_pair_weight)

    def __call__(self, batch: Sequence[PairExample]) -> Dict[str, Any]:
        pos_q: List[str] = []
        pos_d: List[str] = []
        neg_q: List[str] = []
        neg_d: List[str] = []
        aspect_ids: List[int] = []
        weights: List[float] = []
        for row in batch:
            q1, d1 = format_aspect_pair(row.query_text, row.pos_text)
            q2, d2 = format_aspect_pair(row.query_text, row.neg_text)
            pos_q.append(q1)
            pos_d.append(d1)
            neg_q.append(q2)
            neg_d.append(d2)
            aspect_ids.append(aspect_id_from_name(row.aspect))
            weights.append(float(self.pair_type_weights.get(row.pair_type.lower(), self.default_pair_weight)))

        pos_enc = self.tokenizer(pos_q, pos_d, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt")
        neg_enc = self.tokenizer(neg_q, neg_d, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt")
        return {
            "pos": pos_enc,
            "neg": neg_enc,
            "margins": torch.tensor([row.teacher_margin for row in batch], dtype=torch.float32),
            "pair_weights": torch.tensor(weights, dtype=torch.float32),
            "aspect_ids": torch.tensor(aspect_ids, dtype=torch.long),
        }


class ListCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_length: int,
        *,
        high_threshold: float,
        mid_threshold: float,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.high_threshold = float(high_threshold)
        self.mid_threshold = float(mid_threshold)

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        queries: List[str] = []
        docs: List[str] = []
        scores: List[float] = []
        cluster_ids: List[int] = []
        aspect_ids: List[int] = []
        list_sizes: List[int] = []
        for row in batch:
            query = _clean_text(row.get("query_text"))
            aspect = _clean_text(row.get("aspect")) or aspect_from_prefixed_query(query)
            if not query or aspect not in ASPECTS:
                continue
            used = 0
            for doc in list(row.get("docs") or row.get("ranked_docs") or row.get("candidates") or []):
                if not isinstance(doc, dict):
                    continue
                doc_text = _clean_text(doc.get("text"))
                if not doc_text:
                    continue
                q_fmt, d_fmt = format_aspect_pair(query, doc_text)
                score = _clamp_01(doc.get("teacher_score", doc.get("score")))
                queries.append(q_fmt)
                docs.append(d_fmt)
                scores.append(score)
                cluster_ids.append(
                    _cluster_id_from_band_or_score(
                        doc.get("target_cluster", doc.get("band")),
                        score,
                        high_threshold=self.high_threshold,
                        mid_threshold=self.mid_threshold,
                    )
                )
                aspect_ids.append(aspect_id_from_name(aspect))
                used += 1
            if used > 0:
                list_sizes.append(used)

        if not queries:
            return {"enc": None, "scores": None, "cluster_ids": None, "aspect_ids": None, "list_sizes": []}
        enc = self.tokenizer(queries, docs, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt")
        return {
            "enc": enc,
            "scores": torch.tensor(scores, dtype=torch.float32),
            "cluster_ids": torch.tensor(cluster_ids, dtype=torch.long),
            "aspect_ids": torch.tensor(aspect_ids, dtype=torch.long),
            "list_sizes": list_sizes,
        }


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=(device.type == "cuda"))
    if hasattr(obj, "to"):
        try:
            return obj.to(device)
        except Exception:
            return obj
    return obj


def _concat_encoder_batches(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in a.keys():
        if key not in b:
            continue
        av, bv = a[key], b[key]
        if av.dim() >= 2 and bv.dim() >= 2 and int(av.shape[1]) != int(bv.shape[1]):
            target = max(int(av.shape[1]), int(bv.shape[1]))
            av = F.pad(av, (0, target - int(av.shape[1])), value=0)
            bv = F.pad(bv, (0, target - int(bv.shape[1])), value=0)
        out[key] = torch.cat([av, bv], dim=0)
    return out


def _split_pair_logits(logits: torch.Tensor, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = logits.view(-1)
    return flat[:n], flat[n : n * 2]


def variable_margin_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor, margins: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    raw = F.relu(margins - (pos_logits - neg_logits))
    weights = weights.clamp(min=0.0)
    return (raw * weights).sum() / weights.sum().clamp(min=1e-6)


def listwise_kl_mse_loss(
    logits_flat: torch.Tensor,
    scores_flat: torch.Tensor,
    list_sizes: Sequence[int],
    *,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kl_parts: List[torch.Tensor] = []
    mse_parts: List[torch.Tensor] = []
    cursor = 0
    temp = max(1e-6, float(temperature))
    for size in list_sizes:
        n = int(size)
        if n <= 0:
            continue
        logits = logits_flat[cursor : cursor + n]
        scores = scores_flat[cursor : cursor + n].clamp(0.0, 1.0)
        cursor += n
        if n > 1:
            teacher = torch.softmax(scores / temp, dim=0)
            student_log = torch.log_softmax(logits / temp, dim=0)
            kl_parts.append(F.kl_div(student_log, teacher, reduction="sum") * (temp * temp))
        mse_parts.append(F.mse_loss(torch.sigmoid(logits), scores, reduction="mean"))
    zero = logits_flat.sum() * 0.0
    kl = torch.stack(kl_parts).mean() if kl_parts else zero
    mse = torch.stack(mse_parts).mean() if mse_parts else zero
    return kl, mse


def cluster_margin_loss(
    logits_flat: torch.Tensor,
    cluster_ids_flat: torch.Tensor,
    list_sizes: Sequence[int],
    *,
    margin_hm: float,
    margin_ml: float,
    margin_hl: float,
) -> torch.Tensor:
    probs = torch.sigmoid(logits_flat)
    parts: List[torch.Tensor] = []
    cursor = 0
    for size in list_sizes:
        n = int(size)
        s = probs[cursor : cursor + n]
        c = cluster_ids_flat[cursor : cursor + n]
        cursor += n
        high = s[c == 2]
        mid = s[c == 1]
        low = s[c == 0]
        if high.numel() and mid.numel():
            parts.append(F.relu(torch.tensor(float(margin_hm), device=s.device, dtype=s.dtype) - (high.mean() - mid.mean())))
        if mid.numel() and low.numel():
            parts.append(F.relu(torch.tensor(float(margin_ml), device=s.device, dtype=s.dtype) - (mid.mean() - low.mean())))
        if high.numel() and low.numel():
            parts.append(F.relu(torch.tensor(float(margin_hl), device=s.device, dtype=s.dtype) - (high.mean() - low.mean())))
    return torch.stack(parts).mean() if parts else logits_flat.sum() * 0.0


def calibration_band_loss(
    logits_flat: torch.Tensor,
    cluster_ids_flat: torch.Tensor,
    *,
    high_floor: float,
    mid_low: float,
    mid_high: float,
    low_ceil: float,
) -> torch.Tensor:
    probs = torch.sigmoid(logits_flat)
    parts: List[torch.Tensor] = []
    high = probs[cluster_ids_flat == 2]
    mid = probs[cluster_ids_flat == 1]
    low = probs[cluster_ids_flat == 0]
    if high.numel():
        parts.append(F.relu(float(high_floor) - high).mean())
    if mid.numel():
        parts.append((F.relu(float(mid_low) - mid) + F.relu(mid - float(mid_high))).mean())
    if low.numel():
        parts.append(F.relu(low - float(low_ceil)).mean())
    return torch.stack(parts).mean() if parts else logits_flat.sum() * 0.0


def pair_loss_from_batch(model: nn.Module, batch: Dict[str, Any], device: torch.device, *, margin_min: float, margin_max: float) -> torch.Tensor:
    pos = _to_device(batch["pos"], device)
    neg = _to_device(batch["neg"], device)
    margins = batch["margins"].to(device).clamp(min=float(margin_min), max=float(margin_max))
    weights = batch["pair_weights"].to(device)
    aspect_ids = batch["aspect_ids"].to(device)
    pair_aspects = torch.cat([aspect_ids, aspect_ids], dim=0)
    logits = model_logits(model, _concat_encoder_batches(pos, neg), aspect_ids=pair_aspects)
    pos_logits, neg_logits = _split_pair_logits(logits, int(margins.shape[0]))
    return variable_margin_loss(pos_logits, neg_logits, margins, weights)


def list_losses_from_batch(model: nn.Module, batch: Dict[str, Any], device: torch.device, args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    if batch.get("enc") is None:
        zero = torch.zeros((), device=device)
        return {"kl": zero, "mse": zero, "cluster": zero, "calibration": zero}
    enc = _to_device(batch["enc"], device)
    scores = batch["scores"].to(device)
    clusters = batch["cluster_ids"].to(device)
    aspect_ids = batch["aspect_ids"].to(device)
    logits = model_logits(model, enc, aspect_ids=aspect_ids).view(-1)
    kl, mse = listwise_kl_mse_loss(logits, scores, batch["list_sizes"], temperature=args.teacher_temperature)
    cluster = cluster_margin_loss(
        logits,
        clusters,
        batch["list_sizes"],
        margin_hm=args.cluster_margin_hm,
        margin_ml=args.cluster_margin_ml,
        margin_hl=args.cluster_margin_hl,
    )
    calib = calibration_band_loss(
        logits,
        clusters,
        high_floor=args.high_threshold,
        mid_low=args.mid_threshold,
        mid_high=args.high_threshold,
        low_ceil=args.mid_threshold,
    )
    return {"kl": kl, "mse": mse, "cluster": cluster, "calibration": calib}


def total_loss(pair_loss: torch.Tensor, list_losses: Dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    return (
        float(args.loss_pair_weight) * pair_loss
        + float(args.loss_kl_weight) * list_losses["kl"]
        + float(args.loss_mse_weight) * list_losses["mse"]
        + float(args.loss_cluster_margin_weight) * list_losses["cluster"]
        + float(args.loss_calibration_weight) * list_losses["calibration"]
    )


def cycle_loader(loader: DataLoader) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


def oob_summary(probs: torch.Tensor, clusters: torch.Tensor, *, high_threshold: float, mid_threshold: float) -> Dict[str, float]:
    p = probs.detach().cpu()
    c = clusters.detach().cpu()
    low = c == 0
    mid = c == 1
    high = c == 2
    low_total = int(low.sum().item())
    mid_total = int(mid.sum().item())
    high_total = int(high.sum().item())
    low_out = int(((p >= mid_threshold) & low).sum().item())
    mid_low_out = int(((p < mid_threshold) & mid).sum().item())
    mid_high_out = int(((p >= high_threshold) & mid).sum().item())
    high_out = int(((p < high_threshold) & high).sum().item())
    return {
        "oob_low_rate": low_out / max(1, low_total),
        "oob_mid_rate": (mid_low_out + mid_high_out) / max(1, mid_total),
        "oob_mid_low_rate": mid_low_out / max(1, mid_total),
        "oob_mid_high_rate": mid_high_out / max(1, mid_total),
        "oob_high_rate": high_out / max(1, high_total),
        "oob_low_out": float(low_out),
        "oob_mid_out": float(mid_low_out + mid_high_out),
        "oob_high_out": float(high_out),
        "oob_low_total": float(low_total),
        "oob_mid_total": float(mid_total),
        "oob_high_total": float(high_total),
    }


def _dcg(rels: Sequence[float]) -> float:
    return sum((float(rel) / math.log2(i + 2.0)) for i, rel in enumerate(rels))


@torch.no_grad()
def evaluate(model: nn.Module, pair_loader: DataLoader, list_loader: DataLoader, device: torch.device, args: argparse.Namespace) -> Dict[str, float]:
    model.eval()
    pair_vals: List[float] = []
    for batch in pair_loader:
        loss = pair_loss_from_batch(model, batch, device, margin_min=args.margin_min, margin_max=args.margin_max)
        pair_vals.append(float(loss.detach().cpu().item()))

    kl_vals: List[float] = []
    mse_vals: List[float] = []
    cluster_vals: List[float] = []
    calib_vals: List[float] = []
    all_probs: List[torch.Tensor] = []
    all_clusters: List[torch.Tensor] = []
    ndcg_vals: List[float] = []
    mrr_vals: List[float] = []
    recall_vals: List[float] = []
    for batch in list_loader:
        if batch.get("enc") is None:
            continue
        losses = list_losses_from_batch(model, batch, device, args)
        kl_vals.append(float(losses["kl"].detach().cpu().item()))
        mse_vals.append(float(losses["mse"].detach().cpu().item()))
        cluster_vals.append(float(losses["cluster"].detach().cpu().item()))
        calib_vals.append(float(losses["calibration"].detach().cpu().item()))

        enc = _to_device(batch["enc"], device)
        aspect_ids = batch["aspect_ids"].to(device)
        logits = model_logits(model, enc, aspect_ids=aspect_ids).view(-1)
        probs = torch.sigmoid(logits).detach().cpu()
        scores = batch["scores"].detach().cpu()
        clusters = batch["cluster_ids"].detach().cpu()
        all_probs.append(probs)
        all_clusters.append(clusters)

        cursor = 0
        for size in batch["list_sizes"]:
            n = int(size)
            pred = probs[cursor : cursor + n]
            rel = scores[cursor : cursor + n]
            cursor += n
            if n <= 0:
                continue
            order = torch.argsort(pred, descending=True)
            rel_sorted = [float(rel[i].item()) for i in order[:10]]
            ideal = sorted([float(x.item()) for x in rel], reverse=True)[:10]
            denom = _dcg(ideal)
            ndcg_vals.append(_dcg(rel_sorted) / denom if denom > 0.0 else 0.0)
            relevant = (rel >= float(args.high_threshold)).nonzero().view(-1)
            if relevant.numel() > 0:
                ranks = {int(idx.item()): rank + 1 for rank, idx in enumerate(order[:10])}
                hits = [ranks[int(idx.item())] for idx in relevant if int(idx.item()) in ranks]
                mrr_vals.append(1.0 / min(hits) if hits else 0.0)
                top50 = set(int(i.item()) for i in order[:50])
                recall_vals.append(sum(1 for idx in relevant if int(idx.item()) in top50) / float(relevant.numel()))

    probs_all = torch.cat(all_probs) if all_probs else torch.empty(0)
    clusters_all = torch.cat(all_clusters) if all_clusters else torch.empty(0, dtype=torch.long)
    oob = oob_summary(probs_all, clusters_all, high_threshold=args.high_threshold, mid_threshold=args.mid_threshold) if probs_all.numel() else {}
    oob_objective = (
        2.0 * float(oob.get("oob_high_rate", 0.0))
        + float(oob.get("oob_mid_rate", 0.0))
        + float(oob.get("oob_low_rate", 0.0))
    ) / 4.0
    return {
        "pair_loss": sum(pair_vals) / max(1, len(pair_vals)),
        "kl_loss": sum(kl_vals) / max(1, len(kl_vals)),
        "mse_loss": sum(mse_vals) / max(1, len(mse_vals)),
        "cluster_margin_loss": sum(cluster_vals) / max(1, len(cluster_vals)),
        "calibration_loss": sum(calib_vals) / max(1, len(calib_vals)),
        "ndcg@10": sum(ndcg_vals) / max(1, len(ndcg_vals)),
        "mrr@10": sum(mrr_vals) / max(1, len(mrr_vals)),
        "recall@50": sum(recall_vals) / max(1, len(recall_vals)),
        "oob_objective": float(oob_objective),
        **{k: float(v) for k, v in oob.items()},
    }


def save_checkpoint(model: nn.Module, tokenizer: Any, path: Path, meta: Dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    (path / "trainer_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train_stage(
    *,
    stage: int,
    model: nn.Module,
    optimizer: AdamW,
    tokenizer: Any,
    primary_loader: DataLoader,
    secondary_loader: DataLoader,
    val_pair_loader: DataLoader,
    val_list_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
    global_step: int,
    wandb_run: Any = None,
) -> Tuple[int, Dict[str, Any]]:
    secondary_iter = cycle_loader(secondary_loader)
    best_selection = float("-inf")
    best_meta: Dict[str, Any] = {}
    use_amp = device.type == "cuda" and bool(args.fp16)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
    train_log_every_steps = max(0, int(getattr(args, "train_log_every_steps", 1)))
    eval_every_steps = max(0, int(getattr(args, "eval_every_steps", 100)))

    for epoch in range(1, int(args.stage1_epochs if stage == 1 else args.stage2_epochs) + 1):
        model.train()
        hist: Dict[str, List[float]] = {k: [] for k in ("total", "pair", "kl", "mse", "cluster", "calibration")}
        step_hist: Dict[str, List[float]] = {k: [] for k in hist}
        optimizer.zero_grad(set_to_none=True)
        accum = 0

        def mean_hist(values: Dict[str, List[float]]) -> Dict[str, float]:
            return {k: float(sum(v) / max(1, len(v))) for k, v in values.items()}

        def log_after_optimizer_step() -> None:
            nonlocal step_hist
            if train_log_every_steps > 0 and global_step % train_log_every_steps == 0:
                _wandb_log(
                    wandb_run,
                    {
                        "stage": float(stage),
                        "epoch": float(epoch),
                        "train": mean_hist(step_hist),
                    },
                    step=global_step,
                )
            if eval_every_steps > 0 and global_step % eval_every_steps == 0:
                step_metrics = evaluate(model, val_pair_loader, val_list_loader, device, args)
                step_ranking = (
                    float(step_metrics.get("ndcg@10", 0.0))
                    + float(step_metrics.get("mrr@10", 0.0))
                    + float(step_metrics.get("recall@50", 0.0))
                )
                step_selection = step_ranking - float(step_metrics.get("oob_objective", 0.0))
                print(
                    json.dumps(
                        {
                            "stage": int(stage),
                            "epoch": int(epoch),
                            "global_step": int(global_step),
                            "event": "step_eval",
                            "selection_score": float(step_selection),
                            "ranking_sum": float(step_ranking),
                            "val": step_metrics,
                        },
                        ensure_ascii=False,
                    )
                )
                _wandb_log(
                    wandb_run,
                    {
                        "stage": float(stage),
                        "epoch": float(epoch),
                        "selection_score": float(step_selection),
                        "ranking_sum": float(step_ranking),
                        "val": step_metrics,
                    },
                    step=global_step,
                )
                model.train()
            step_hist = {k: [] for k in hist}

        for primary_batch in primary_loader:
            secondary_batch = next(secondary_iter)
            pair_batch = primary_batch if stage == 1 else secondary_batch
            list_batch = secondary_batch if stage == 1 else primary_batch
            with amp_ctx:
                p_loss = pair_loss_from_batch(model, pair_batch, device, margin_min=args.margin_min, margin_max=args.margin_max)
                l_losses = list_losses_from_batch(model, list_batch, device, args)
                loss = total_loss(p_loss, l_losses, args) / float(args.grad_accum_steps)
            loss.backward()
            accum += 1
            hist["total"].append(float(loss.detach().cpu().item() * float(args.grad_accum_steps)))
            hist["pair"].append(float(p_loss.detach().cpu().item()))
            hist["kl"].append(float(l_losses["kl"].detach().cpu().item()))
            hist["mse"].append(float(l_losses["mse"].detach().cpu().item()))
            hist["cluster"].append(float(l_losses["cluster"].detach().cpu().item()))
            hist["calibration"].append(float(l_losses["calibration"].detach().cpu().item()))
            step_hist["total"].append(float(loss.detach().cpu().item() * float(args.grad_accum_steps)))
            step_hist["pair"].append(float(p_loss.detach().cpu().item()))
            step_hist["kl"].append(float(l_losses["kl"].detach().cpu().item()))
            step_hist["mse"].append(float(l_losses["mse"].detach().cpu().item()))
            step_hist["cluster"].append(float(l_losses["cluster"].detach().cpu().item()))
            step_hist["calibration"].append(float(l_losses["calibration"].detach().cpu().item()))
            if accum >= int(args.grad_accum_steps):
                if float(args.max_grad_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accum = 0
                log_after_optimizer_step()
        if accum > 0:
            if float(args.max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            log_after_optimizer_step()

        metrics = evaluate(model, val_pair_loader, val_list_loader, device, args)
        ranking = float(metrics.get("ndcg@10", 0.0)) + float(metrics.get("mrr@10", 0.0)) + float(metrics.get("recall@50", 0.0))
        selection = ranking - float(metrics.get("oob_objective", 0.0))
        meta = {
            "stage": int(stage),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "selection_score": float(selection),
            "ranking_sum": float(ranking),
            "train": {k: sum(v) / max(1, len(v)) for k, v in hist.items()},
            "val": metrics,
        }
        print(json.dumps(meta, ensure_ascii=False))
        _wandb_log(
            wandb_run,
            {
                "stage": float(stage),
                "epoch": float(epoch),
                "selection_score": float(selection),
                "ranking_sum": float(ranking),
                "train": meta["train"],
                "val": metrics,
            },
            step=global_step,
        )
        save_checkpoint(model, tokenizer, output_dir / f"stage{stage}_epoch_{epoch}", meta)
        if selection > best_selection:
            best_selection = selection
            best_meta = meta
            best_dir = output_dir / f"best_stage{stage}_selected"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            save_checkpoint(model, tokenizer, best_dir, meta)
    return global_step, best_meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CE3 aspect-conditioned multi-head cross-encoder trainer.")
    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    p.add_argument("--split-dir", type=str, default=SPLIT_DIR_DEFAULT)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--train-listwise", type=str, default="")
    p.add_argument("--val-listwise", type=str, default="")
    p.add_argument("--test-listwise", type=str, default="")
    p.add_argument("--train-pairwise", type=str, default="")
    p.add_argument("--val-pairwise", type=str, default="")
    p.add_argument("--test-pairwise", type=str, default="")
    p.add_argument("--stage1-epochs", type=int, default=1)
    p.add_argument("--stage2-epochs", type=int, default=3)
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--stage1-learning-rate", type=float, default=0.0)
    p.add_argument("--stage2-learning-rate", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--teacher-temperature", type=float, default=1.0)
    p.add_argument("--loss-pair-weight", type=float, default=0.5)
    p.add_argument("--loss-kl-weight", type=float, default=1.0)
    p.add_argument("--loss-mse-weight", type=float, default=0.2)
    p.add_argument("--loss-cluster-margin-weight", type=float, default=0.1)
    p.add_argument("--loss-calibration-weight", type=float, default=0.1)
    p.add_argument("--cluster-margin-hm", type=float, default=0.12)
    p.add_argument("--cluster-margin-ml", type=float, default=0.12)
    p.add_argument("--cluster-margin-hl", type=float, default=0.30)
    p.add_argument("--high-threshold", type=float, default=0.70)
    p.add_argument("--mid-threshold", type=float, default=0.30)
    p.add_argument("--margin-min", type=float, default=0.02)
    p.add_argument("--margin-max", type=float, default=0.60)
    p.add_argument("--pair-type-weight-map", type=str, default=PAIR_TYPE_WEIGHT_MAP_DEFAULT)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--train-log-every-steps", type=int, default=1, help="Log train losses to W&B every N optimizer steps. 0 disables step train logs.")
    p.add_argument("--eval-every-steps", type=int, default=100, help="Run validation and log val metrics every N optimizer steps. 0 disables step validation.")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--no-multihead", action="store_true")
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-project", type=str, default="")
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--wandb-run-name", type=str, default="")
    p.add_argument("--wandb-mode", type=str, default="online", help="online, offline, dryrun, or disabled.")
    p.add_argument("--wandb-tags", type=str, default="ce3,aspect-conditioned,multihead")
    return p.parse_args()


def _path_arg(args: argparse.Namespace, attr: str, default_name: str) -> Path:
    value = _clean_text(getattr(args, attr))
    if value:
        return _resolve_path(value)
    return _resolve_path(Path(args.split_dir) / default_name)


def main() -> int:
    started = time.time()
    args = parse_args()
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_list_path = _path_arg(args, "train_listwise", "llm_distill_all_listwise_train.jsonl")
    val_list_path = _path_arg(args, "val_listwise", "llm_distill_all_listwise_val.jsonl")
    test_list_path = _path_arg(args, "test_listwise", "llm_distill_all_listwise_test.jsonl")
    train_pair_path = _path_arg(args, "train_pairwise", "llm_distill_all_pairwise_train.jsonl")
    val_pair_path = _path_arg(args, "val_pairwise", "llm_distill_all_pairwise_val.jsonl")
    test_pair_path = _path_arg(args, "test_pairwise", "llm_distill_all_pairwise_test.jsonl")

    train_list_rows = _load_jsonl(train_list_path)
    val_list_rows = _load_jsonl(val_list_path)
    test_list_rows = _load_jsonl(test_list_path)
    train_pairs = load_pair_rows(train_pair_path)
    val_pairs = load_pair_rows(val_pair_path)
    test_pairs = load_pair_rows(test_pair_path)
    if not train_list_rows or not train_pairs:
        raise RuntimeError("Training requires non-empty train listwise and pairwise split files.")
    if not val_list_rows or not val_pairs:
        raise RuntimeError("Validation requires non-empty val listwise and pairwise split files.")

    pair_weights, pair_default = _parse_pair_type_weight_map(args.pair_type_weight_map)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=bool(args.trust_remote_code))
    model = load_sequence_classifier_model(
        args.model_id,
        num_labels=1,
        multi_aspect_heads=not bool(args.no_multihead),
        trust_remote_code=bool(args.trust_remote_code),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    pair_collator = PairCollator(tokenizer, args.max_length, pair_type_weights=pair_weights, default_pair_weight=pair_default)
    list_collator = ListCollator(tokenizer, args.max_length, high_threshold=args.high_threshold, mid_threshold=args.mid_threshold)
    loader_kwargs = {"num_workers": max(0, int(args.num_workers)), "pin_memory": device.type == "cuda"}

    pair_train_loader = DataLoader(PairDataset(train_pairs), batch_size=args.train_batch_size, shuffle=True, collate_fn=pair_collator, **loader_kwargs)
    list_train_loader = DataLoader(ListDataset(train_list_rows), batch_size=args.train_batch_size, shuffle=True, collate_fn=list_collator, **loader_kwargs)
    pair_val_loader = DataLoader(PairDataset(val_pairs), batch_size=args.eval_batch_size, shuffle=False, collate_fn=pair_collator, **loader_kwargs)
    list_val_loader = DataLoader(ListDataset(val_list_rows), batch_size=args.eval_batch_size, shuffle=False, collate_fn=list_collator, **loader_kwargs)
    pair_test_loader = DataLoader(PairDataset(test_pairs or val_pairs), batch_size=args.eval_batch_size, shuffle=False, collate_fn=pair_collator, **loader_kwargs)
    list_test_loader = DataLoader(ListDataset(test_list_rows or val_list_rows), batch_size=args.eval_batch_size, shuffle=False, collate_fn=list_collator, **loader_kwargs)

    setup_meta = {
        "stage": "ce3_train_setup",
        "model_id": args.model_id,
        "multihead": not bool(args.no_multihead),
        "device": str(device),
        "train_list_rows": len(train_list_rows),
        "train_pair_rows": len(train_pairs),
        "val_list_rows": len(val_list_rows),
        "val_pair_rows": len(val_pairs),
        "test_list_rows": len(test_list_rows),
        "test_pair_rows": len(test_pairs),
        "long_prefixes": True,
        "posthoc_calibration": False,
        "wandb_enabled": bool(_wandb_enabled(args)),
    }
    print(json.dumps(setup_meta, ensure_ascii=False))
    wandb_run = _init_wandb(
        args,
        run_config={
            **vars(args),
            "train_list_rows": len(train_list_rows),
            "train_pair_rows": len(train_pairs),
            "val_list_rows": len(val_list_rows),
            "val_pair_rows": len(val_pairs),
            "test_list_rows": len(test_list_rows),
            "test_pair_rows": len(test_pairs),
            "multihead": not bool(args.no_multihead),
            "long_prefixes": True,
            "posthoc_calibration": False,
        },
    )
    _wandb_log(wandb_run, {"setup": setup_meta}, step=0)

    global_step = 0
    best: Dict[str, Any] = {}
    try:
        if int(args.stage1_epochs) > 0:
            opt = AdamW(model.parameters(), lr=float(args.stage1_learning_rate or args.learning_rate), weight_decay=float(args.weight_decay))
            global_step, best["stage1"] = train_stage(
                stage=1,
                model=model,
                optimizer=opt,
                tokenizer=tokenizer,
                primary_loader=pair_train_loader,
                secondary_loader=list_train_loader,
                val_pair_loader=pair_val_loader,
                val_list_loader=list_val_loader,
                device=device,
                output_dir=output_dir,
                args=args,
                global_step=global_step,
                wandb_run=wandb_run,
            )
        if int(args.stage2_epochs) > 0:
            opt = AdamW(model.parameters(), lr=float(args.stage2_learning_rate or args.learning_rate), weight_decay=float(args.weight_decay))
            global_step, best["stage2"] = train_stage(
                stage=2,
                model=model,
                optimizer=opt,
                tokenizer=tokenizer,
                primary_loader=list_train_loader,
                secondary_loader=pair_train_loader,
                val_pair_loader=pair_val_loader,
                val_list_loader=list_val_loader,
                device=device,
                output_dir=output_dir,
                args=args,
                global_step=global_step,
                wandb_run=wandb_run,
            )

        test_metrics = evaluate(model, pair_test_loader, list_test_loader, device, args)
        _wandb_log(wandb_run, {"test": test_metrics}, step=global_step)
        final_meta = {
            "elapsed_sec": time.time() - started,
            "global_step": int(global_step),
            "best": best,
            "test": test_metrics,
            "args": vars(args),
        }
        save_checkpoint(model, tokenizer, output_dir / "final", final_meta)
        (output_dir / "train_manifest.json").write_text(json.dumps(final_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"stage": "ce3_train_done", **final_meta}, ensure_ascii=False))
        return 0
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
