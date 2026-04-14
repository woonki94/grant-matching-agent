from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


DEFAULT_MODEL_NAME = "BAAI/bge-reranker-base"
DEFAULT_VAL_RATIO = 0.1
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 2.0
DEFAULT_BATCH_SIZE = 8  # queries per step
DEFAULT_GRAD_ACCUM = 1
DEFAULT_LR = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.06
DEFAULT_MAX_LENGTH = 256
DEFAULT_LOGGING_STEPS = 20
DEFAULT_EVAL_STEPS = 200
DEFAULT_SAVE_STEPS = 500
DEFAULT_MAX_PAIRS_PER_QUERY = 24
DEFAULT_TOP_FOCUS = 2

WANDB_ENV_API_KEY = "WANDB_API_KEY"
WANDB_ENV_PROJECT = "WANDB_PROJECT"
WANDB_ENV_ENTITY = "WANDB_ENTITY"
WANDB_API_KEY_MIN_LEN = 30


@dataclass
class QueryExample:
    query: str
    query_group: str
    docs: List[str]
    target_scores: List[float]


@dataclass
class LossBreakdown:
    total: float
    listwise: float
    pairwise: float
    query_count: int


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
        out = float(value)
    except Exception:
        out = float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return float(out)


def _safe_unit_float(value: Any, *, default: float = 0.0) -> float:
    out = _safe_float(value, default=default)
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


def _load_selected_env_from_file(env_path: Path, keys: Sequence[str]) -> Dict[str, str]:
    wanted = set(str(k) for k in list(keys or []))
    if not wanted:
        return {}
    if not env_path.exists() or not env_path.is_file():
        return {}

    out: Dict[str, str] = {}
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {}

    for raw in lines:
        line = str(raw or "").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        k = _clean_text(key)
        if k not in wanted:
            continue
        v = str(val).strip()
        if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            v = v[1:-1]
        out[k] = v
    return out


def _resolve_latest_listwise(dataset_dir: Path) -> Path:
    cands = sorted(dataset_dir.glob("*_listwise_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No listwise dataset found under {dataset_dir}")
    return cands[0].resolve()


def _resolve_latest_pairs(dataset_dir: Path) -> Path:
    cands = sorted(dataset_dir.glob("*_pairs_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No pair dataset found under {dataset_dir}")
    return cands[0].resolve()


def _load_listwise_examples(path: Path, *, max_queries: int) -> Tuple[List[QueryExample], Dict[str, Any]]:
    safe_max = _safe_limit(max_queries, default=0, minimum=0, maximum=10_000_000)
    examples: List[QueryExample] = []
    skipped = 0
    source_counts: Dict[str, int] = {}

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if safe_max > 0 and len(examples) >= safe_max:
                break
            try:
                item = dict(json.loads(line) or {})
            except Exception:
                skipped += 1
                continue

            query = _clean_text(item.get("query"))
            query_group = _clean_text(item.get("query_group")) or f"q::{len(examples)}"
            candidates = list(item.get("candidates") or [])
            ranking = [int(x) for x in list(item.get("ranking") or []) if int(x) > 0]
            label_source = _clean_text(item.get("label_source")) or "unknown"
            source_counts[label_source] = source_counts.get(label_source, 0) + 1

            if not query or len(candidates) < 2 or len(ranking) < 2:
                skipped += 1
                continue

            by_i: Dict[int, str] = {}
            for c in candidates:
                idx = int(c.get("i") or 0)
                text = _clean_text(c.get("t"))
                if idx <= 0 or not text:
                    continue
                by_i[idx] = text

            ordered_docs: List[str] = []
            rank_positions: List[int] = []
            n = len(ranking)
            if n < 2:
                skipped += 1
                continue

            # Keep docs in candidate-index order for deterministic encoding order.
            valid_indices = sorted([k for k in by_i.keys() if k in set(ranking)])
            pos_map = {idx: pos for pos, idx in enumerate(ranking)}
            for idx in valid_indices:
                ordered_docs.append(by_i[idx])
                rank_positions.append(int(pos_map[idx]))

            if len(ordered_docs) < 2:
                skipped += 1
                continue

            denom = max(1, len(ranking) - 1)
            targets = [1.0 - (float(pos) / float(denom)) for pos in rank_positions]
            examples.append(
                QueryExample(
                    query=query,
                    query_group=query_group,
                    docs=ordered_docs,
                    target_scores=targets,
                )
            )

    meta = {
        "input_path": str(path),
        "examples_loaded": int(len(examples)),
        "rows_skipped": int(skipped),
        "label_source_counts": dict(source_counts),
    }
    return examples, meta


def _load_pairs_as_examples(path: Path, *, max_queries: int, min_candidates: int) -> Tuple[List[QueryExample], Dict[str, Any]]:
    safe_max = _safe_limit(max_queries, default=0, minimum=0, maximum=10_000_000)
    safe_min_cands = _safe_limit(min_candidates, default=2, minimum=2, maximum=256)
    grouped: Dict[str, Dict[str, Any]] = {}
    skipped = 0

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                item = dict(json.loads(line) or {})
            except Exception:
                skipped += 1
                continue
            query = _clean_text(item.get("query"))
            doc = _clean_text(item.get("doc"))
            group = _clean_text(item.get("query_group"))
            score = _safe_unit_float(item.get("label_score"), default=0.0)
            if not query or not doc or not group:
                skipped += 1
                continue

            payload = grouped.setdefault(
                group,
                {
                    "query": query,
                    "docs": {},
                },
            )
            docs: Dict[str, float] = payload["docs"]
            # Keep max score if duplicate doc appears.
            prev = docs.get(doc)
            if prev is None or score > prev:
                docs[doc] = float(score)

    examples: List[QueryExample] = []
    for group, payload in grouped.items():
        query = _clean_text(payload.get("query"))
        docs_map = dict(payload.get("docs") or {})
        if len(docs_map) < safe_min_cands:
            continue
        ranked = sorted(docs_map.items(), key=lambda x: float(x[1]), reverse=True)
        docs = [str(x[0]) for x in ranked]
        targets = [float(x[1]) for x in ranked]
        examples.append(QueryExample(query=query, query_group=group, docs=docs, target_scores=targets))
        if safe_max > 0 and len(examples) >= safe_max:
            break

    meta = {
        "input_path": str(path),
        "examples_loaded": int(len(examples)),
        "rows_skipped": int(skipped),
    }
    return examples, meta


def _split_by_query_group(
    examples: Sequence[QueryExample],
    *,
    val_ratio: float,
    seed: int,
) -> Tuple[List[QueryExample], List[QueryExample]]:
    rows = list(examples or [])
    if not rows:
        return [], []

    by_group: Dict[str, List[QueryExample]] = defaultdict(list)
    for ex in rows:
        by_group[_clean_text(ex.query_group) or "unknown"].append(ex)

    groups = list(by_group.keys())
    if len(groups) < 2:
        return rows, []

    rng = random.Random(int(seed))
    rng.shuffle(groups)

    target_val_groups = int(round(len(groups) * float(max(0.0, min(0.5, val_ratio)))))
    target_val_groups = max(1, min(len(groups) - 1, target_val_groups))
    val_groups = set(groups[:target_val_groups])

    train: List[QueryExample] = []
    val: List[QueryExample] = []
    for gid, items in by_group.items():
        if gid in val_groups:
            val.extend(items)
        else:
            train.extend(items)

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_tokenizer_stable(model_name_or_path: str):
    from transformers import AutoConfig, AutoTokenizer

    target = _clean_text(model_name_or_path)
    if not target:
        raise RuntimeError("Tokenizer load target is empty.")

    last_error: Optional[Exception] = None
    attempts = (
        {"use_fast": False, "fix_mistral_regex": True},
        {"use_fast": True, "fix_mistral_regex": True},
        {"use_fast": False},
        {"use_fast": True},
    )
    for kwargs in attempts:
        try:
            return AutoTokenizer.from_pretrained(target, trust_remote_code=True, **kwargs)
        except Exception as e:
            last_error = e

    base_name = ""
    try:
        cfg = AutoConfig.from_pretrained(target, trust_remote_code=True, local_files_only=True)
        base_name = _clean_text(getattr(cfg, "_name_or_path", ""))
    except Exception:
        base_name = ""

    if base_name and base_name != target:
        for kwargs in attempts:
            try:
                return AutoTokenizer.from_pretrained(base_name, trust_remote_code=True, **kwargs)
            except Exception as e:
                last_error = e

    raise RuntimeError(f"Failed to load tokenizer stably. target={target}, last_error={last_error}")


def _iter_batches(rows: Sequence[QueryExample], *, batch_size: int, shuffle: bool, seed: int) -> Iterable[List[QueryExample]]:
    items = list(rows or [])
    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(items)
    bs = _safe_limit(batch_size, default=DEFAULT_BATCH_SIZE, minimum=1, maximum=4096)
    for i in range(0, len(items), bs):
        yield items[i : i + bs]


def _build_pair_indices(
    n_items: int,
    *,
    max_pairs: int,
    top_focus: int,
    rng: random.Random,
) -> List[Tuple[int, int, float]]:
    if n_items < 2:
        return []
    safe_max_pairs = _safe_limit(max_pairs, default=DEFAULT_MAX_PAIRS_PER_QUERY, minimum=1, maximum=4096)
    safe_top_focus = _safe_limit(top_focus, default=DEFAULT_TOP_FOCUS, minimum=1, maximum=128)
    top_focus_n = min(n_items - 1, safe_top_focus)

    pairs: List[Tuple[int, int, float]] = []
    used = set()

    # Always prioritize top-ranked separation.
    for i in range(top_focus_n):
        for j in range(i + 1, n_items):
            gap = float(j - i)
            w = 1.0 + 0.25 * gap
            key = (i, j)
            if key in used:
                continue
            used.add(key)
            pairs.append((i, j, w))
            if len(pairs) >= safe_max_pairs:
                return pairs

    # Fill with random additional pair constraints.
    all_pairs = [(i, j) for i in range(0, n_items - 1) for j in range(i + 1, n_items)]
    rng.shuffle(all_pairs)
    for i, j in all_pairs:
        if len(pairs) >= safe_max_pairs:
            break
        key = (i, j)
        if key in used:
            continue
        used.add(key)
        gap = float(j - i)
        w = 1.0 + 0.15 * gap
        pairs.append((i, j, w))
    return pairs


def _batch_forward_and_loss(
    *,
    batch: Sequence[QueryExample],
    tokenizer,
    model,
    device,
    max_length: int,
    supervision_mode: str,
    listwise_weight: float,
    pairwise_weight: float,
    max_pairs_per_query: int,
    top_focus_pairs: int,
    rng: random.Random,
):
    import torch
    import torch.nn.functional as F

    mode = _clean_text(supervision_mode).lower() or "hybrid"
    list_w = float(max(0.0, listwise_weight))
    pair_w = float(max(0.0, pairwise_weight))

    q_texts: List[str] = []
    d_texts: List[str] = []
    spans: List[Tuple[int, int, QueryExample]] = []
    for ex in list(batch or []):
        start = len(q_texts)
        for d in list(ex.docs or []):
            q_texts.append(_clean_text(ex.query))
            d_texts.append(_clean_text(d))
        end = len(q_texts)
        if end - start >= 2:
            spans.append((start, end, ex))

    if not spans:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return z, LossBreakdown(total=0.0, listwise=0.0, pairwise=0.0, query_count=0)

    enc = tokenizer(
        q_texts,
        d_texts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
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
    pair_losses: List[float] = []

    for start, end, ex in spans:
        scores = flat_scores[start:end]
        target = torch.tensor([float(x) for x in ex.target_scores], dtype=torch.float32, device=device)
        if target.numel() != scores.numel() or target.numel() < 2:
            continue

        # Listwise (ListNet-style): fit the full ranking distribution.
        list_loss = -(F.softmax(target, dim=0) * F.log_softmax(scores, dim=0)).sum()

        # Pairwise (RankNet): enforce margin/order between ranked positions.
        pair_idx = _build_pair_indices(
            int(scores.numel()),
            max_pairs=int(max_pairs_per_query),
            top_focus=int(top_focus_pairs),
            rng=rng,
        )
        pair_terms: List[torch.Tensor] = []
        pair_weight_sum = 0.0
        for i, j, w in pair_idx:
            # i is better rank than j (by target ordering construction).
            pair_terms.append(F.softplus(-(scores[i] - scores[j])) * float(w))
            pair_weight_sum += float(w)
        if pair_terms and pair_weight_sum > 0.0:
            pair_loss = torch.stack(pair_terms).sum() / float(pair_weight_sum)
        else:
            pair_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        if mode == "listwise":
            q_loss = list_w * list_loss
        elif mode == "pairwise":
            q_loss = pair_w * pair_loss
        else:  # hybrid
            q_loss = (list_w * list_loss) + (pair_w * pair_loss)

        losses.append(q_loss)
        list_losses.append(float(list_loss.detach().item()))
        pair_losses.append(float(pair_loss.detach().item()))

    if not losses:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return z, LossBreakdown(total=0.0, listwise=0.0, pairwise=0.0, query_count=0)

    batch_loss = torch.stack(losses).mean()
    info = LossBreakdown(
        total=float(batch_loss.detach().item()),
        listwise=float(np.mean(list_losses)) if list_losses else 0.0,
        pairwise=float(np.mean(pair_losses)) if pair_losses else 0.0,
        query_count=int(len(losses)),
    )
    return batch_loss, info


def _compute_ranking_metrics(examples: Sequence[QueryExample], score_lists: Sequence[List[float]]) -> Dict[str, float]:
    def _dcg(rels: Sequence[float], k: int) -> float:
        out = 0.0
        for i, r in enumerate(list(rels or [])[: int(max(1, k))], start=1):
            out += (float(2.0 ** float(r)) - 1.0) / math.log2(float(i + 1))
        return float(out)

    top1 = 0
    mrr_sum = 0.0
    ndcg5 = 0.0
    ndcg10 = 0.0
    n = 0

    for ex, scores in zip(list(examples or []), list(score_lists or [])):
        targets = list(float(x) for x in list(ex.target_scores or []))
        if len(targets) < 2 or len(targets) != len(scores):
            continue

        pred_order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        true_best = int(max(range(len(targets)), key=lambda i: float(targets[i])))

        if int(pred_order[0]) == true_best:
            top1 += 1

        rr = 0.0
        for rank_pos, idx in enumerate(pred_order, start=1):
            if int(idx) == true_best:
                rr = 1.0 / float(rank_pos)
                break
        mrr_sum += float(rr)

        pred_rels = [targets[i] for i in pred_order]
        ideal_rels = sorted(targets, reverse=True)
        dcg5 = _dcg(pred_rels, 5)
        idcg5 = _dcg(ideal_rels, 5)
        dcg10 = _dcg(pred_rels, 10)
        idcg10 = _dcg(ideal_rels, 10)
        ndcg5 += (dcg5 / idcg5) if idcg5 > 0.0 else 0.0
        ndcg10 += (dcg10 / idcg10) if idcg10 > 0.0 else 0.0

        n += 1

    if n <= 0:
        return {
            "top1_accuracy": 0.0,
            "mrr": 0.0,
            "ndcg@5": 0.0,
            "ndcg@10": 0.0,
            "query_count": 0.0,
        }

    return {
        "top1_accuracy": float(top1) / float(n),
        "mrr": float(mrr_sum) / float(n),
        "ndcg@5": float(ndcg5) / float(n),
        "ndcg@10": float(ndcg10) / float(n),
        "query_count": float(n),
    }


def _evaluate(
    *,
    examples: Sequence[QueryExample],
    tokenizer,
    model,
    device,
    max_length: int,
    eval_batch_size: int,
    supervision_mode: str,
    listwise_weight: float,
    pairwise_weight: float,
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
            "eval/pairwise_loss": 0.0,
            "eval/top1_accuracy": 0.0,
            "eval/mrr": 0.0,
            "eval/ndcg@5": 0.0,
            "eval/ndcg@10": 0.0,
            "eval/query_count": 0.0,
        }

    model.eval()
    rng = random.Random(int(seed) + 9991)

    sum_loss = 0.0
    sum_list = 0.0
    sum_pair = 0.0
    sum_q = 0

    score_lists: List[List[float]] = []
    with torch.no_grad():
        for step, batch in enumerate(_iter_batches(rows, batch_size=int(eval_batch_size), shuffle=False, seed=int(seed))):
            loss, info = _batch_forward_and_loss(
                batch=batch,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=int(max_length),
                supervision_mode=supervision_mode,
                listwise_weight=float(listwise_weight),
                pairwise_weight=float(pairwise_weight),
                max_pairs_per_query=int(max_pairs_per_query),
                top_focus_pairs=int(top_focus_pairs),
                rng=rng,
            )
            _ = loss
            sum_loss += float(info.total) * int(max(1, info.query_count))
            sum_list += float(info.listwise) * int(max(1, info.query_count))
            sum_pair += float(info.pairwise) * int(max(1, info.query_count))
            sum_q += int(max(1, info.query_count))

            # Separate pass for exact per-query prediction list.
            q_texts: List[str] = []
            d_texts: List[str] = []
            spans: List[Tuple[int, int]] = []
            for ex in list(batch or []):
                s = len(q_texts)
                for d in list(ex.docs or []):
                    q_texts.append(_clean_text(ex.query))
                    d_texts.append(_clean_text(d))
                e = len(q_texts)
                spans.append((s, e))

            enc = tokenizer(
                q_texts,
                d_texts,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
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
    ranking = _compute_ranking_metrics(rows, score_lists)
    metrics = {
        "eval/loss": float(sum_loss) / float(denom),
        "eval/listwise_loss": float(sum_list) / float(denom),
        "eval/pairwise_loss": float(sum_pair) / float(denom),
        "eval/top1_accuracy": float(ranking.get("top1_accuracy") or 0.0),
        "eval/mrr": float(ranking.get("mrr") or 0.0),
        "eval/ndcg@5": float(ranking.get("ndcg@5") or 0.0),
        "eval/ndcg@10": float(ranking.get("ndcg@10") or 0.0),
        "eval/query_count": float(ranking.get("query_count") or 0.0),
    }
    return metrics


def train(args) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup, set_seed
    from transformers.utils import logging as hf_logging

    # Keep terminal output focused on training/eval signals.
    hf_logging.set_verbosity_error()

    dataset_dir = Path(_clean_text(args.dataset_dir)).expanduser().resolve()
    listwise_path = Path(_clean_text(args.listwise_jsonl)).expanduser().resolve() if _clean_text(args.listwise_jsonl) else None
    pairs_path = Path(_clean_text(args.pairs_jsonl)).expanduser().resolve() if _clean_text(args.pairs_jsonl) else None

    if listwise_path is None and pairs_path is None:
        listwise_path = _resolve_latest_listwise(dataset_dir)

    if listwise_path is not None:
        examples, data_meta = _load_listwise_examples(listwise_path, max_queries=int(args.max_queries))
    else:
        examples, data_meta = _load_pairs_as_examples(
            pairs_path,
            max_queries=int(args.max_queries),
            min_candidates=int(args.min_candidates_per_query),
        )

    if len(examples) < 10:
        raise RuntimeError(f"Not enough examples to train. examples={len(examples)}")

    train_rows, val_rows = _split_by_query_group(
        examples,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    if not train_rows:
        raise RuntimeError("Training split is empty.")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(_clean_text(args.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))
    device = _pick_device()

    tokenizer = _load_tokenizer_stable(_clean_text(args.model_name) or DEFAULT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        _clean_text(args.model_name) or DEFAULT_MODEL_NAME,
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
        loaded_env = _load_selected_env_from_file(
            env_file,
            keys=[WANDB_ENV_API_KEY, WANDB_ENV_PROJECT, WANDB_ENV_ENTITY],
        )
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
            raise RuntimeError("W&B requested but `wandb` is not installed. Run: pip install wandb") from e

        wandb = _wandb
        run_name = _clean_text(args.wandb_run_name) or f"spec-chunk-ranker-{run_ts}"
        wandb.init(
            project=_clean_text(os.getenv(WANDB_ENV_PROJECT)) or None,
            entity=_clean_text(os.getenv(WANDB_ENV_ENTITY)) or None,
            name=run_name,
            config={
                "model_name": _clean_text(args.model_name) or DEFAULT_MODEL_NAME,
                "supervision_mode": _clean_text(args.supervision_mode) or "hybrid",
                "epochs": float(args.epochs),
                "batch_size": int(args.batch_size),
                "grad_accum": int(args.grad_accum),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "warmup_ratio": float(args.warmup_ratio),
                "max_length": int(args.max_length),
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
    train_pair_running = 0.0
    train_count_running = 0

    epoch_count = max(1, int(math.ceil(float(args.epochs))))
    for epoch_idx in range(epoch_count):
        epoch_seed = int(args.seed) + epoch_idx * 1003
        for batch in _iter_batches(train_rows, batch_size=int(args.batch_size), shuffle=True, seed=epoch_seed):
            global_step += 1

            loss, info = _batch_forward_and_loss(
                batch=batch,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=int(args.max_length),
                supervision_mode=_clean_text(args.supervision_mode) or "hybrid",
                listwise_weight=float(args.listwise_weight),
                pairwise_weight=float(args.pairwise_weight),
                max_pairs_per_query=int(args.max_pairs_per_query),
                top_focus_pairs=int(args.top_focus_pairs),
                rng=rng,
            )

            loss_to_backprop = loss / float(max(1, int(args.grad_accum)))
            loss_to_backprop.backward()

            train_loss_running += float(info.total)
            train_list_running += float(info.listwise)
            train_pair_running += float(info.pairwise)
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
                    "train/pairwise_loss": float(train_pair_running) / float(denom),
                    "train/lr": float(scheduler.get_last_lr()[0]) if scheduler.get_last_lr() else float(args.learning_rate),
                    "train/epoch": float(epoch_idx + (global_step / max(1, steps_per_epoch))),
                    "train/global_step": float(global_step),
                    "train/opt_step": float(opt_step),
                }
                print(
                    f"[train] step={global_step} opt_step={opt_step} "
                    f"loss={payload['train/loss']:.6f} "
                    f"list={payload['train/listwise_loss']:.6f} "
                    f"pair={payload['train/pairwise_loss']:.6f}"
                )
                if wandb is not None and getattr(wandb, "run", None) is not None:
                    wandb.log(payload, step=int(global_step))
                train_loss_running = 0.0
                train_list_running = 0.0
                train_pair_running = 0.0
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
                    supervision_mode=_clean_text(args.supervision_mode) or "hybrid",
                    listwise_weight=float(args.listwise_weight),
                    pairwise_weight=float(args.pairwise_weight),
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

                metric_key = _clean_text(args.select_best_metric) or "eval/ndcg@10"
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

    # Final evaluation and save.
    final_eval = _evaluate(
        examples=val_rows,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=int(args.max_length),
        eval_batch_size=int(args.eval_batch_size),
        supervision_mode=_clean_text(args.supervision_mode) or "hybrid",
        listwise_weight=float(args.listwise_weight),
        pairwise_weight=float(args.pairwise_weight),
        max_pairs_per_query=int(args.max_pairs_per_query),
        top_focus_pairs=int(args.top_focus_pairs),
        seed=int(args.seed) + 777,
    ) if val_rows else {
        "eval/loss": 0.0,
        "eval/listwise_loss": 0.0,
        "eval/pairwise_loss": 0.0,
        "eval/top1_accuracy": 0.0,
        "eval/mrr": 0.0,
        "eval/ndcg@5": 0.0,
        "eval/ndcg@10": 0.0,
        "eval/query_count": 0.0,
    }

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Promote best if we never evaluated or never improved.
    if not best_checkpoint_dir.exists():
        best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(best_checkpoint_dir))
        tokenizer.save_pretrained(str(best_checkpoint_dir))
        best_eval = dict(final_eval)
        best_metric = float(best_eval.get(_clean_text(args.select_best_metric) or "eval/ndcg@10") or 0.0)

    promoted_dir = output_dir / "promoted_best"
    if promoted_dir.exists():
        shutil.rmtree(promoted_dir)
    shutil.copytree(best_checkpoint_dir, promoted_dir)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": _clean_text(args.model_name) or DEFAULT_MODEL_NAME,
        "supervision_mode": _clean_text(args.supervision_mode) or "hybrid",
        "dataset": str((listwise_path or pairs_path) or ""),
        "dataset_meta": data_meta,
        "train_query_count": int(len(train_rows)),
        "val_query_count": int(len(val_rows)),
        "device": str(device),
        "global_steps": int(global_step),
        "optimizer_steps": int(opt_step),
        "best_metric_name": _clean_text(args.select_best_metric) or "eval/ndcg@10",
        "best_metric_value": float(best_metric),
        "best_eval": best_eval,
        "final_eval": final_eval,
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
    default_output = Path(__file__).resolve().parent / "models" / f"spec_chunk_ranker_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    default_dataset_dir = Path(__file__).resolve().parent / "dataset"

    p = argparse.ArgumentParser(
        description=(
            "Strong cross-encoder supervision trainer for spec->chunk ranking. "
            "Default objective is hybrid listwise+pairwise with query-group split and optional W&B."
        )
    )
    p.add_argument("--dataset-dir", type=str, default=str(default_dataset_dir), help="Directory with generated dataset files.")
    p.add_argument("--listwise-jsonl", type=str, default="", help="Optional explicit listwise JSONL path.")
    p.add_argument("--pairs-jsonl", type=str, default="", help="Optional explicit pairs JSONL path (used if listwise not given).")
    p.add_argument("--max-queries", type=int, default=0, help="Max queries to load (0=all).")
    p.add_argument("--min-candidates-per-query", type=int, default=2)

    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--output-dir", type=str, default=str(default_output))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)

    p.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Queries per training step.")
    p.add_argument("--eval-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)

    p.add_argument("--supervision-mode", type=str, default="hybrid", choices=["hybrid", "listwise", "pairwise"])
    p.add_argument("--listwise-weight", type=float, default=1.0)
    p.add_argument("--pairwise-weight", type=float, default=1.0)
    p.add_argument("--max-pairs-per-query", type=int, default=DEFAULT_MAX_PAIRS_PER_QUERY)
    p.add_argument("--top-focus-pairs", type=int, default=DEFAULT_TOP_FOCUS)

    p.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    p.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS)
    p.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    p.add_argument("--select-best-metric", type=str, default="eval/ndcg@10")

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-run-name", type=str, default="")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    summary = train(args)

    print("Spec->chunk ranker training complete.")
    print(f"  output dir        : {summary.get('paths', {}).get('output_dir', '')}")
    print(f"  train queries     : {summary.get('train_query_count', 0)}")
    print(f"  val queries       : {summary.get('val_query_count', 0)}")
    print(f"  best metric       : {summary.get('best_metric_name', '')}={summary.get('best_metric_value', 0.0):.6f}")
    print(f"  final top1        : {summary.get('final_eval', {}).get('eval/top1_accuracy', 0.0):.4f}")
    print(f"  final ndcg@10     : {summary.get('final_eval', {}).get('eval/ndcg@10', 0.0):.4f}")
    print(f"  promoted best dir : {summary.get('paths', {}).get('promoted_best_dir', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
