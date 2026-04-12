from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import shutil
import sys
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
DEFAULT_ROUNDS = 2
DEFAULT_HARD_NEGATIVE_WEIGHT = 3.0
DEFAULT_POINTWISE_WEIGHT = 0.15
DEFAULT_PAIRWISE_WEIGHT = 1.0
DEFAULT_MARGIN_WEIGHT = 0.7
DEFAULT_MARGIN_VALUE = 0.2
DEFAULT_VAL_RATIO = 0.1
DEFAULT_EPOCHS = 1.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_GRAD_ACCUM = 1
DEFAULT_LR = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_MAX_LENGTH = 256
DEFAULT_LOGGING_STEPS = 50
DEFAULT_SAVE_STEPS = 500
DEFAULT_SAVE_TOTAL_LIMIT = 2
DEFAULT_INFER_BATCH_SIZE = 128
DEFAULT_MINE_PER_QUERY = 2
DEFAULT_MINE_MARGIN_WINDOW = 0.05
DEFAULT_GATE_HARD_NEG_MAX = 0.25
DEFAULT_GATE_FP_MAX = 0.25
DEFAULT_GATE_MARGIN_MIN = 0.10
DEFAULT_LLM_EVAL_CASES = 40

WANDB_REPORT_TARGET = "wandb"
WANDB_ENV_WATCH = "WANDB_WATCH"
WANDB_WATCH_DISABLED = "false"
WANDB_ENV_API_KEY = "WANDB_API_KEY"
WANDB_ENV_PROJECT = "WANDB_PROJECT"
WANDB_ENV_ENTITY = "WANDB_ENTITY"
WANDB_API_KEY_MIN_LEN = 30


@dataclass
class TripletRow:
    query: str
    positive: str
    negative: str
    query_group: str
    negative_candidate_type: str
    sample_weight: float
    pos_label_score: float
    neg_label_score: float
    teacher_margin: float
    label_source: str
    pair_id: str


def _default_llm_eval_model() -> str:
    try:
        from config import settings  # local import to avoid hard dependency during --help

        return _clean_text(getattr(settings, "haiku", ""))
    except Exception:
        return ""


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
    if np.isnan(parsed) or np.isinf(parsed):
        return float(default)
    return float(parsed)


def _safe_unit_float(value: Any, *, default: float = 0.0) -> float:
    out = _safe_float(value, default=default)
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


def _load_tokenizer_stable(model_name_or_path: str):
    try:
        from transformers import AutoConfig, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Missing tokenizer dependencies. Install: pip install transformers sentencepiece tiktoken"
        ) from e

    target = _clean_text(model_name_or_path)
    if not target:
        raise RuntimeError("Tokenizer load target is empty.")

    last_error: Optional[Exception] = None
    # Prefer slow tokenizer first to avoid broken fast-tokenizer regex patterns.
    attempts = (
        {"use_fast": False, "fix_mistral_regex": True},
        {"use_fast": False},
    )
    for kwargs in attempts:
        try:
            return AutoTokenizer.from_pretrained(target, trust_remote_code=True, **kwargs)
        except Exception as e:
            last_error = e

    # Fallback: tokenizer from base model id in local config.
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

    raise RuntimeError(
        "Failed to load a stable tokenizer. "
        f"target={target}, last_error={last_error}"
    )


def _stable_hash(value: str) -> str:
    return hashlib.sha1(_clean_text(value).encode("utf-8")).hexdigest()[:16]


def _resolve_dataset_path(path_str: str) -> Path:
    if _clean_text(path_str):
        p = Path(_clean_text(path_str)).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Dataset file not found: {p}")
        return p

    base = Path(__file__).resolve().parent / "dataset"
    cands = sorted(base.glob("*.triplets.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No triplet dataset found in {base}. Run dataset generation first.")
    return cands[0].resolve()


def _derive_query_group(item: Dict[str, Any], *, query_text: str) -> str:
    qid = _clean_text(item.get("query_spec_id"))
    if qid:
        return qid
    gid = _clean_text(item.get("grant_id"))
    if gid:
        return f"grant:{gid}"
    return f"query:{_stable_hash(query_text)}"


def _is_hard_negative_type(negative_type: str) -> bool:
    t = _clean_text(negative_type).lower()
    return ("hard" in t) or ("false_positive" in t) or ("mined" in t)


def load_triplet_rows(
    *,
    dataset_jsonl: Path,
    max_rows: int,
    hard_negative_weight: float,
) -> Tuple[List[TripletRow], Dict[str, Any]]:
    safe_max = _safe_limit(max_rows, default=0, minimum=0, maximum=50_000_000)
    hard_w = max(1.0, float(hard_negative_weight))
    rows: List[TripletRow] = []
    pair_seen = set()
    neg_type_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    skipped = 0

    with dataset_jsonl.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if safe_max > 0 and len(rows) >= safe_max:
                break
            try:
                item = dict(json.loads(line) or {})
            except Exception:
                skipped += 1
                continue
            query = _clean_text(item.get("query"))
            pos = _clean_text(item.get("positive"))
            neg = _clean_text(item.get("negative"))
            if not query or not pos or not neg:
                skipped += 1
                continue

            q_group = _derive_query_group(item, query_text=query)
            neg_type = _clean_text(item.get("negative_candidate_type")) or "unknown"
            source = _clean_text(item.get("label_source")) or "unknown"

            pair_id = f"{q_group}::{_stable_hash(pos)}::{_stable_hash(neg)}"
            if pair_id in pair_seen:
                continue
            pair_seen.add(pair_id)

            sample_weight = hard_w if _is_hard_negative_type(neg_type) else 1.0
            pos_score = _safe_unit_float(item.get("positive_teacher_score"), default=1.0)
            neg_score = _safe_unit_float(item.get("negative_teacher_score"), default=0.0)
            margin = _safe_float(item.get("teacher_margin"), default=(pos_score - neg_score))

            rows.append(
                TripletRow(
                    query=query,
                    positive=pos,
                    negative=neg,
                    query_group=q_group,
                    negative_candidate_type=neg_type,
                    sample_weight=float(sample_weight),
                    pos_label_score=float(pos_score),
                    neg_label_score=float(neg_score),
                    teacher_margin=float(margin),
                    label_source=source,
                    pair_id=pair_id,
                )
            )
            neg_type_counts[neg_type] = neg_type_counts.get(neg_type, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1

    stats = {
        "rows_loaded": int(len(rows)),
        "rows_skipped_bad_json_or_empty": int(skipped),
        "negative_type_counts": dict(neg_type_counts),
        "label_source_counts": dict(source_counts),
    }
    return rows, stats


def split_rows_by_query_group(
    rows: Sequence[TripletRow],
    *,
    val_ratio: float,
    seed: int,
) -> Tuple[List[TripletRow], List[TripletRow]]:
    all_rows = list(rows or [])
    if not all_rows:
        return [], []

    groups: Dict[str, List[TripletRow]] = {}
    for r in all_rows:
        groups.setdefault(r.query_group, []).append(r)
    group_ids = list(groups.keys())
    if len(group_ids) < 2:
        return all_rows, []

    rng = random.Random(int(seed))
    rng.shuffle(group_ids)

    target_groups = int(round(len(group_ids) * float(max(0.0, min(0.5, val_ratio)))))
    target_groups = max(1, min(len(group_ids) - 1, target_groups))
    val_set = set(group_ids[:target_groups])

    train_rows: List[TripletRow] = []
    val_rows: List[TripletRow] = []
    for gid, vals in groups.items():
        if gid in val_set:
            val_rows.extend(vals)
        else:
            train_rows.extend(vals)

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def _rows_to_dataset_dicts(rows: Sequence[TripletRow]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "query": r.query,
                "positive": r.positive,
                "negative": r.negative,
                "sample_weight": float(r.sample_weight),
                "pos_label_score": float(r.pos_label_score),
                "neg_label_score": float(r.neg_label_score),
                "query_group": r.query_group,
                "negative_candidate_type": r.negative_candidate_type,
                "pair_id": r.pair_id,
            }
        )
    return out


def _build_query_pools(rows: Sequence[TripletRow]) -> Dict[str, Dict[str, Any]]:
    pools: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        q = pools.setdefault(
            r.query_group,
            {
                "query": r.query,
                "docs": {},
            },
        )
        docs: Dict[str, Dict[str, Any]] = q["docs"]
        pos_key = _normalize_doc_key(r.positive)
        neg_key = _normalize_doc_key(r.negative)
        if pos_key:
            docs[pos_key] = {
                "text": r.positive,
                "is_positive": 1,
                "kind": "positive",
            }
        if neg_key:
            cur = docs.get(neg_key)
            if cur is None or int(cur.get("is_positive") or 0) != 1:
                docs[neg_key] = {
                    "text": r.negative,
                    "is_positive": 0,
                    "kind": _clean_text(r.negative_candidate_type) or "negative",
                }
    return pools


def _normalize_doc_key(text: str) -> str:
    return " ".join(_clean_text(text).lower().split())


def _score_pairs(
    *,
    tokenizer,
    model,
    device,
    pairs: Sequence[Tuple[str, str]],
    batch_size: int,
    max_length: int,
) -> List[float]:
    import torch

    bs = _safe_limit(batch_size, default=DEFAULT_INFER_BATCH_SIZE, minimum=1, maximum=8192)
    mx = _safe_limit(max_length, default=DEFAULT_MAX_LENGTH, minimum=32, maximum=4096)
    out: List[float] = []
    pair_list = list(pairs or [])
    for s in range(0, len(pair_list), bs):
        batch = pair_list[s : s + bs]
        q = [_clean_text(x[0]) for x in batch]
        d = [_clean_text(x[1]) for x in batch]
        enc = tokenizer(
            q,
            d,
            padding=True,
            truncation=True,
            max_length=mx,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            if logits.ndim == 2 and logits.shape[-1] > 1:
                score = logits[:, -1]
            else:
                score = logits.squeeze(-1)
            out.extend([float(x) for x in score.detach().cpu().tolist()])
    return out


def evaluate_ranking_metrics(
    *,
    tokenizer,
    model,
    device,
    rows: Sequence[TripletRow],
    batch_size: int,
    max_length: int,
) -> Dict[str, Any]:
    pools = _build_query_pools(rows)
    if not pools:
        return {
            "query_count": 0,
            "top1_accuracy": 0.0,
            "false_positive_rate": 0.0,
            "hard_negative_top1_rate": 0.0,
            "mean_top1_top2_margin": 0.0,
            "mrr": 0.0,
        }

    query_items = list(pools.items())
    pair_rows: List[Tuple[str, str]] = []
    row_meta: List[Tuple[str, str, int, str]] = []
    for qid, payload in query_items:
        query = _clean_text(payload.get("query"))
        docs = list(dict(payload.get("docs") or {}).values())
        for d in docs:
            text = _clean_text(d.get("text"))
            if not query or not text:
                continue
            pair_rows.append((query, text))
            row_meta.append((qid, text, int(d.get("is_positive") or 0), _clean_text(d.get("kind"))))

    scores = _score_pairs(
        tokenizer=tokenizer,
        model=model,
        device=device,
        pairs=pair_rows,
        batch_size=batch_size,
        max_length=max_length,
    )

    by_query: Dict[str, List[Dict[str, Any]]] = {}
    for meta, score in zip(row_meta, scores):
        qid, text, is_pos, kind = meta
        by_query.setdefault(qid, []).append(
            {
                "text": text,
                "is_positive": int(is_pos),
                "kind": kind,
                "score": float(score),
            }
        )

    valid_q = 0
    top1_correct = 0
    fp_top1 = 0
    hard_top1 = 0
    mrr_sum = 0.0
    margin_sum = 0.0
    margin_count = 0

    for qid, docs in by_query.items():
        ranked = sorted(docs, key=lambda x: float(x.get("score") or 0.0), reverse=True)
        if not ranked:
            continue
        if not any(int(x.get("is_positive") or 0) == 1 for x in ranked):
            continue
        valid_q += 1
        top1 = ranked[0]
        top1_pos = int(top1.get("is_positive") or 0) == 1
        if top1_pos:
            top1_correct += 1
        else:
            fp_top1 += 1
            if _is_hard_negative_type(_clean_text(top1.get("kind"))):
                hard_top1 += 1
        first_pos_rank = None
        for i, row in enumerate(ranked, start=1):
            if int(row.get("is_positive") or 0) == 1:
                first_pos_rank = i
                break
        if first_pos_rank is not None:
            mrr_sum += 1.0 / float(first_pos_rank)
        if len(ranked) >= 2:
            margin = float(ranked[0].get("score") or 0.0) - float(ranked[1].get("score") or 0.0)
            margin_sum += float(margin)
            margin_count += 1

    if valid_q <= 0:
        return {
            "query_count": 0,
            "top1_accuracy": 0.0,
            "false_positive_rate": 0.0,
            "hard_negative_top1_rate": 0.0,
            "mean_top1_top2_margin": 0.0,
            "mrr": 0.0,
        }

    return {
        "query_count": int(valid_q),
        "top1_accuracy": float(top1_correct) / float(valid_q),
        "false_positive_rate": float(fp_top1) / float(valid_q),
        "hard_negative_top1_rate": float(hard_top1) / float(valid_q),
        "mean_top1_top2_margin": float(margin_sum) / float(max(1, margin_count)),
        "mrr": float(mrr_sum) / float(valid_q),
    }


def evaluate_validation_loss(
    *,
    tokenizer,
    model,
    device,
    rows: Sequence[TripletRow],
    batch_size: int,
    max_length: int,
    pair_weight: float,
    margin_weight: float,
    pointwise_weight: float,
    margin_value: float,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    eval_rows = list(rows or [])
    if not eval_rows:
        return {
            "row_count": 0,
            "val_total_loss": 0.0,
            "val_pairwise_loss": 0.0,
            "val_margin_loss": 0.0,
            "val_pointwise_loss": 0.0,
            "val_pairwise_term": 0.0,
            "val_margin_term": 0.0,
            "val_pointwise_term": 0.0,
        }

    bs = _safe_limit(batch_size, default=DEFAULT_INFER_BATCH_SIZE, minimum=1, maximum=8192)
    mx = _safe_limit(max_length, default=DEFAULT_MAX_LENGTH, minimum=32, maximum=4096)
    pair_w = float(max(0.0, pair_weight))
    margin_w = float(max(0.0, margin_weight))
    point_w = float(max(0.0, pointwise_weight))
    margin_v = float(max(0.0, margin_value))

    model_input_names = set(str(x) for x in list(getattr(tokenizer, "model_input_names", []) or []))
    if not model_input_names:
        model_input_names = {"input_ids", "attention_mask", "token_type_ids"}

    def _to_score(logits):
        if logits.ndim == 2 and logits.shape[-1] > 1:
            return logits[:, -1].float().view(-1)
        return logits.squeeze(-1).float().view(-1)

    total_weight = 0.0
    total_sum = 0.0
    pair_sum = 0.0
    margin_sum = 0.0
    point_sum = 0.0

    model.eval()
    for s in range(0, len(eval_rows), bs):
        batch_rows = eval_rows[s : s + bs]
        q = [_clean_text(x.query) for x in batch_rows]
        p = [_clean_text(x.positive) for x in batch_rows]
        n = [_clean_text(x.negative) for x in batch_rows]

        enc_pos = tokenizer(
            q,
            p,
            padding=True,
            truncation=True,
            max_length=mx,
            return_tensors="pt",
        )
        enc_neg = tokenizer(
            q,
            n,
            padding=True,
            truncation=True,
            max_length=mx,
            return_tensors="pt",
        )
        pos_inputs = {k: v.to(device) for k, v in enc_pos.items() if str(k) in model_input_names}
        neg_inputs = {k: v.to(device) for k, v in enc_neg.items() if str(k) in model_input_names}
        sample_w = torch.tensor([float(max(0.0, x.sample_weight)) for x in batch_rows], dtype=torch.float32, device=device)
        pos_label = torch.tensor([_safe_unit_float(x.pos_label_score, default=1.0) for x in batch_rows], dtype=torch.float32, device=device)
        neg_label = torch.tensor([_safe_unit_float(x.neg_label_score, default=0.0) for x in batch_rows], dtype=torch.float32, device=device)

        with torch.no_grad():
            pos_outputs = model(**pos_inputs)
            neg_outputs = model(**neg_inputs)
            s_pos = _to_score(pos_outputs.logits)
            s_neg = _to_score(neg_outputs.logits)
            delta = s_pos - s_neg

            l_pair = F.softplus(-delta)
            l_margin = F.relu(margin_v - delta)
            l_point = 0.5 * ((s_pos - pos_label) ** 2 + (s_neg - neg_label) ** 2)

            per_item = (pair_w * l_pair) + (margin_w * l_margin)
            if point_w > 0.0:
                per_item = per_item + (point_w * l_point)

            w_sum = float(sample_w.sum().detach().cpu().item())
            total_weight += w_sum
            total_sum += float(torch.sum(sample_w * per_item).detach().cpu().item())
            pair_sum += float(torch.sum(sample_w * l_pair).detach().cpu().item())
            margin_sum += float(torch.sum(sample_w * l_margin).detach().cpu().item())
            point_sum += float(torch.sum(sample_w * l_point).detach().cpu().item())

    denom = max(1e-8, float(total_weight))
    val_pairwise_loss = float(pair_sum) / denom
    val_margin_loss = float(margin_sum) / denom
    val_pointwise_loss = float(point_sum) / denom
    val_pairwise_term = float(pair_w) * val_pairwise_loss
    val_margin_term = float(margin_w) * val_margin_loss
    val_pointwise_term = float(point_w) * val_pointwise_loss
    return {
        "row_count": int(len(eval_rows)),
        "val_total_loss": float(total_sum) / denom,
        "val_pairwise_loss": val_pairwise_loss,
        "val_margin_loss": val_margin_loss,
        "val_pointwise_loss": val_pointwise_loss,
        "val_pairwise_term": val_pairwise_term,
        "val_margin_term": val_margin_term,
        "val_pointwise_term": val_pointwise_term,
    }


def _dedupe_triplets(rows: Sequence[TripletRow]) -> List[TripletRow]:
    out: List[TripletRow] = []
    seen = set()
    for r in rows:
        key = (r.query_group, _normalize_doc_key(r.positive), _normalize_doc_key(r.negative))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def mine_false_positive_hard_negatives(
    *,
    tokenizer,
    model,
    device,
    rows: Sequence[TripletRow],
    max_per_query: int,
    margin_window: float,
    hard_negative_weight: float,
    batch_size: int,
    max_length: int,
) -> Tuple[List[TripletRow], Dict[str, Any]]:
    pools = _build_query_pools(rows)
    safe_per_q = _safe_limit(max_per_query, default=DEFAULT_MINE_PER_QUERY, minimum=0, maximum=64)
    if safe_per_q <= 0:
        return [], {"queries_seen": int(len(pools)), "queries_mined": 0, "triplets_mined": 0}

    mined: List[TripletRow] = []
    queries_mined = 0
    pair_seen = {(r.query_group, _normalize_doc_key(r.positive), _normalize_doc_key(r.negative)) for r in rows}

    for qid, payload in pools.items():
        query = _clean_text(payload.get("query"))
        docs = list(dict(payload.get("docs") or {}).values())
        positives = [d for d in docs if int(d.get("is_positive") or 0) == 1]
        negatives = [d for d in docs if int(d.get("is_positive") or 0) == 0]
        if not query or not positives or not negatives:
            continue

        pairs = [(query, _clean_text(d.get("text"))) for d in docs if _clean_text(d.get("text"))]
        if not pairs:
            continue
        scores = _score_pairs(
            tokenizer=tokenizer,
            model=model,
            device=device,
            pairs=pairs,
            batch_size=batch_size,
            max_length=max_length,
        )
        scored_docs: List[Dict[str, Any]] = []
        for d, s in zip(docs, scores):
            scored_docs.append(
                {
                    "text": _clean_text(d.get("text")),
                    "is_positive": int(d.get("is_positive") or 0),
                    "kind": _clean_text(d.get("kind")),
                    "score": float(s),
                }
            )
        scored_docs.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        scored_pos = [x for x in scored_docs if int(x.get("is_positive") or 0) == 1]
        scored_neg = [x for x in scored_docs if int(x.get("is_positive") or 0) == 0]
        if not scored_pos or not scored_neg:
            continue

        anchor_pos = scored_pos[0]
        pos_text = _clean_text(anchor_pos.get("text"))
        pos_score = float(anchor_pos.get("score") or 0.0)
        if not pos_text:
            continue

        mined_local = 0
        for neg in scored_neg:
            if mined_local >= safe_per_q:
                break
            neg_text = _clean_text(neg.get("text"))
            neg_score = float(neg.get("score") or 0.0)
            if not neg_text:
                continue
            if neg_score < (pos_score - float(max(0.0, margin_window))):
                continue
            key = (qid, _normalize_doc_key(pos_text), _normalize_doc_key(neg_text))
            if key in pair_seen:
                continue
            pair_seen.add(key)
            mined_local += 1
            mined.append(
                TripletRow(
                    query=query,
                    positive=pos_text,
                    negative=neg_text,
                    query_group=qid,
                    negative_candidate_type="mined_false_positive",
                    sample_weight=float(max(1.0, hard_negative_weight)),
                    pos_label_score=0.95,
                    neg_label_score=0.10,
                    teacher_margin=0.85,
                    label_source="mined_false_positive",
                    pair_id=f"{qid}::{_stable_hash(pos_text)}::{_stable_hash(neg_text)}",
                )
            )
        if mined_local > 0:
            queries_mined += 1

    return mined, {
        "queries_seen": int(len(pools)),
        "queries_mined": int(queries_mined),
        "triplets_mined": int(len(mined)),
    }


class PairwiseTripletCollator:
    def __init__(self, tokenizer, *, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = _safe_limit(max_length, default=DEFAULT_MAX_LENGTH, minimum=32, maximum=4096)

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        rows = list(features or [])
        q = [_clean_text(x.get("query")) for x in rows]
        p = [_clean_text(x.get("positive")) for x in rows]
        n = [_clean_text(x.get("negative")) for x in rows]

        enc_pos = self.tokenizer(
            q,
            p,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc_neg = self.tokenizer(
            q,
            n,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        out: Dict[str, Any] = {}
        for k, v in enc_pos.items():
            out[f"pos_{k}"] = v
        for k, v in enc_neg.items():
            out[f"neg_{k}"] = v
        out["sample_weight"] = torch.tensor([_safe_float(x.get("sample_weight"), default=1.0) for x in rows], dtype=torch.float32)
        out["pos_label_score"] = torch.tensor(
            [_safe_unit_float(x.get("pos_label_score"), default=1.0) for x in rows],
            dtype=torch.float32,
        )
        out["neg_label_score"] = torch.tensor(
            [_safe_unit_float(x.get("neg_label_score"), default=0.0) for x in rows],
            dtype=torch.float32,
        )
        return out


def _setup_training_components(
    *,
    model_name_or_path: str,
    max_length: int,
    pair_weight: float,
    margin_weight: float,
    pointwise_weight: float,
    margin_value: float,
):
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForSequenceClassification, Trainer
    except Exception as e:
        raise RuntimeError(
            "Missing training dependencies. Install: pip install torch transformers datasets accelerate"
        ) from e

    tokenizer = _load_tokenizer_stable(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=1,
        trust_remote_code=True,
    )

    class PairwiseMarginTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pair_weight = float(max(0.0, pair_weight))
            self.margin_weight = float(max(0.0, margin_weight))
            self.pointwise_weight = float(max(0.0, pointwise_weight))
            self.margin_value = float(max(0.0, margin_value))
            model_input_names = list(getattr(tokenizer, "model_input_names", []) or [])
            if not model_input_names:
                model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
            self.model_input_names = set(str(x) for x in model_input_names)

        @staticmethod
        def _extract_prefix_inputs(
            inputs: Dict[str, Any],
            prefix: str,
            *,
            allowed_names: Sequence[str],
        ) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            allowed = set(str(x) for x in list(allowed_names or []))
            for k, v in inputs.items():
                if k.startswith(prefix):
                    model_key = k[len(prefix) :]
                    if model_key in allowed:
                        out[model_key] = v
            return out

        @staticmethod
        def _to_score(logits):
            if logits.ndim == 2 and logits.shape[-1] > 1:
                return logits[:, -1].float().view(-1)
            return logits.squeeze(-1).float().view(-1)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
            pos_inputs = self._extract_prefix_inputs(
                inputs,
                "pos_",
                allowed_names=self.model_input_names,
            )
            neg_inputs = self._extract_prefix_inputs(
                inputs,
                "neg_",
                allowed_names=self.model_input_names,
            )
            pos_outputs = model(**pos_inputs)
            neg_outputs = model(**neg_inputs)
            s_pos = self._to_score(pos_outputs.logits)
            s_neg = self._to_score(neg_outputs.logits)
            delta = s_pos - s_neg

            l_pair = F.softplus(-delta)
            l_margin = F.relu(float(self.margin_value) - delta)

            sample_w = inputs.get("sample_weight")
            if sample_w is None:
                sample_w = torch.ones_like(delta)
            sample_w = sample_w.float().view(-1)

            per_item = (self.pair_weight * l_pair) + (self.margin_weight * l_margin)
            if self.pointwise_weight > 0.0:
                pos_label = inputs.get("pos_label_score")
                neg_label = inputs.get("neg_label_score")
                if pos_label is None:
                    pos_label = torch.ones_like(s_pos)
                if neg_label is None:
                    neg_label = torch.zeros_like(s_neg)
                l_point = 0.5 * ((s_pos - pos_label.float().view(-1)) ** 2 + (s_neg - neg_label.float().view(-1)) ** 2)
                per_item = per_item + (self.pointwise_weight * l_point)

            denom = torch.clamp(sample_w.sum(), min=1e-8)
            loss = torch.sum(sample_w * per_item) / denom
            if return_outputs:
                return loss, {"delta": delta.detach()}
            return loss

    return tokenizer, model, PairwiseMarginTrainer


def train_one_round(
    *,
    model_name_or_path: str,
    train_rows: Sequence[TripletRow],
    output_dir: Path,
    seed: int,
    epochs: float,
    batch_size: int,
    grad_accum_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    max_length: int,
    logging_steps: int,
    save_steps: int,
    save_total_limit: int,
    fp16: bool,
    bf16: bool,
    num_workers: int,
    pair_weight: float,
    margin_weight: float,
    pointwise_weight: float,
    margin_value: float,
    use_wandb: bool,
) -> Dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from transformers import Trainer, TrainingArguments, set_seed
    except Exception as e:
        raise RuntimeError(
            "Missing training dependencies. Install: pip install torch transformers datasets accelerate"
        ) from e

    tokenizer, model, PairwiseMarginTrainer = _setup_training_components(
        model_name_or_path=model_name_or_path,
        max_length=max_length,
        pair_weight=pair_weight,
        margin_weight=margin_weight,
        pointwise_weight=pointwise_weight,
        margin_value=margin_value,
    )
    set_seed(int(seed))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = Dataset.from_list(_rows_to_dataset_dicts(train_rows))
    collator = PairwiseTripletCollator(tokenizer, max_length=max_length)

    ta_sig = inspect.signature(TrainingArguments.__init__)
    ta_params = set(ta_sig.parameters.keys())
    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    use_cuda_device = bool(cuda_available)
    use_mps_device = bool((not use_cuda_device) and mps_available)

    ta_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": float(max(0.1, epochs)),
        "per_device_train_batch_size": _safe_limit(batch_size, default=DEFAULT_BATCH_SIZE, minimum=1, maximum=1024),
        "gradient_accumulation_steps": _safe_limit(grad_accum_steps, default=DEFAULT_GRAD_ACCUM, minimum=1, maximum=1024),
        "learning_rate": float(max(1e-8, learning_rate)),
        "weight_decay": float(max(0.0, weight_decay)),
        "warmup_ratio": float(max(0.0, min(1.0, warmup_ratio))),
        "logging_steps": _safe_limit(logging_steps, default=DEFAULT_LOGGING_STEPS, minimum=1, maximum=100000),
        "save_steps": _safe_limit(save_steps, default=DEFAULT_SAVE_STEPS, minimum=1, maximum=10000000),
        "save_total_limit": _safe_limit(save_total_limit, default=DEFAULT_SAVE_TOTAL_LIMIT, minimum=1, maximum=100),
        "dataloader_num_workers": _safe_limit(num_workers, default=0, minimum=0, maximum=64),
        "dataloader_pin_memory": False,
        "report_to": [WANDB_REPORT_TARGET] if bool(use_wandb) else [],
        "remove_unused_columns": False,
        "seed": int(seed),
        "fp16": bool(fp16 and use_cuda_device),
        "bf16": bool(bf16 and use_cuda_device),
    }
    eval_mode = "no"
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
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in ta_params and v is not None}
    training_args = TrainingArguments(**ta_kwargs)

    base_trainer_sig = inspect.signature(Trainer.__init__)
    base_trainer_params = set(base_trainer_sig.parameters.keys())
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "data_collator": collator,
    }
    if "tokenizer" in base_trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in base_trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in base_trainer_params}
    trainer = PairwiseMarginTrainer(**trainer_kwargs)

    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return {
        "output_dir": str(output_dir),
        "train_metrics": dict(getattr(train_result, "metrics", {}) or {}),
        "device": str(getattr(trainer.args, "device", "")),
        "cuda_available": bool(cuda_available),
        "mps_available": bool(mps_available),
        "wandb_enabled": bool(use_wandb),
    }


def _gate_result(
    *,
    metrics: Dict[str, Any],
    gate_hard_neg_max: float,
    gate_fp_max: float,
    gate_margin_min: float,
) -> Dict[str, Any]:
    hard = float(metrics.get("hard_negative_top1_rate") or 0.0)
    fp = float(metrics.get("false_positive_rate") or 0.0)
    margin = float(metrics.get("mean_top1_top2_margin") or 0.0)
    hard_pass = hard <= float(gate_hard_neg_max)
    fp_pass = fp <= float(gate_fp_max)
    margin_pass = margin >= float(gate_margin_min)
    return {
        "all_pass": bool(hard_pass and fp_pass and margin_pass),
        "hard_negative_top1_rate": {
            "actual": float(hard),
            "threshold_max": float(gate_hard_neg_max),
            "pass": bool(hard_pass),
        },
        "false_positive_rate": {
            "actual": float(fp),
            "threshold_max": float(gate_fp_max),
            "pass": bool(fp_pass),
        },
        "mean_top1_top2_margin": {
            "actual": float(margin),
            "threshold_min": float(gate_margin_min),
            "pass": bool(margin_pass),
        },
    }


def _select_promoted_round(round_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(round_summaries or [])
    if not rows:
        raise RuntimeError("No round summaries to select promoted checkpoint.")

    def _score(item: Dict[str, Any]) -> Tuple[int, float, float, float]:
        gate = dict(item.get("acceptance_gates") or {})
        metrics = dict(item.get("ranking_metrics") or {})
        pass_flag = bool(gate.get("all_pass"))
        hard = float(metrics.get("hard_negative_top1_rate") or 1.0)
        fp = float(metrics.get("false_positive_rate") or 1.0)
        margin = float(metrics.get("mean_top1_top2_margin") or 0.0)
        return (0 if pass_flag else 1, hard, fp, -margin)

    ranked = sorted(rows, key=_score)
    return dict(ranked[0])


def run_finetune(args) -> Dict[str, Any]:
    dataset_jsonl = _resolve_dataset_path(_clean_text(args.dataset_jsonl))
    output_root = Path(_clean_text(args.output_dir)).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_rows, load_stats = load_triplet_rows(
        dataset_jsonl=dataset_jsonl,
        max_rows=int(args.max_rows),
        hard_negative_weight=float(args.hard_negative_weight),
    )
    if len(base_rows) < 20:
        raise RuntimeError(f"Not enough rows to train. rows={len(base_rows)}")

    train_rows_base, val_rows = split_rows_by_query_group(
        base_rows,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    if not train_rows_base:
        raise RuntimeError("Training split is empty after query-group split.")
    if not val_rows:
        raise RuntimeError("Validation split is empty after query-group split. Increase dataset size or val_ratio.")

    wandb_enabled = bool(args.wandb)
    wandb = None
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

        try:
            import wandb as _wandb  # type: ignore
        except Exception as e:
            raise RuntimeError("W&B requested but package not installed. Run: pip install wandb") from e

        os.environ.setdefault(WANDB_ENV_WATCH, WANDB_WATCH_DISABLED)
        run_name = _clean_text(args.wandb_run_name) or f"finetune-cross-encoder-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        _wandb.init(
            project=_clean_text(os.getenv(WANDB_ENV_PROJECT)) or None,
            entity=_clean_text(os.getenv(WANDB_ENV_ENTITY)) or None,
            name=run_name,
            config={
                "dataset_jsonl": str(dataset_jsonl),
                "output_dir": str(output_root),
                "model_name": _clean_text(args.model_name) or DEFAULT_MODEL_NAME,
                "rounds": int(args.rounds),
                "epochs": float(args.epochs),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "pair_weight": float(args.pair_weight),
                "margin_weight": float(args.margin_weight),
                "pointwise_weight": float(args.pointwise_weight),
                "margin_value": float(args.margin_value),
                "hard_negative_weight": float(args.hard_negative_weight),
                "gate_hard_negative_top1_rate_max": float(args.gate_hard_negative_top1_rate_max),
                "gate_false_positive_rate_max": float(args.gate_false_positive_rate_max),
                "gate_mean_top1_top2_margin_min": float(args.gate_mean_top1_top2_margin_min),
            },
        )
        wandb = _wandb

    round_summaries: List[Dict[str, Any]] = []
    mined_rows_all: List[TripletRow] = []
    model_path_for_round = _clean_text(args.model_name) or DEFAULT_MODEL_NAME

    for round_idx in range(1, int(args.rounds) + 1):
        round_dir = output_root / f"round_{round_idx:02d}"
        train_rows = _dedupe_triplets(list(train_rows_base) + list(mined_rows_all))

        train_info = train_one_round(
            model_name_or_path=model_path_for_round,
            train_rows=train_rows,
            output_dir=round_dir,
            seed=int(args.seed) + int(round_idx),
            epochs=float(args.epochs),
            batch_size=int(args.batch_size),
            grad_accum_steps=int(args.grad_accum_steps),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            warmup_ratio=float(args.warmup_ratio),
            max_length=int(args.max_length),
            logging_steps=int(args.logging_steps),
            save_steps=int(args.save_steps),
            save_total_limit=int(args.save_total_limit),
            fp16=bool(args.fp16),
            bf16=bool(args.bf16),
            num_workers=int(args.num_workers),
            pair_weight=float(args.pair_weight),
            margin_weight=float(args.margin_weight),
            pointwise_weight=float(args.pointwise_weight),
            margin_value=float(args.margin_value),
            use_wandb=bool(wandb_enabled),
        )

        try:
            import torch
            from transformers import AutoModelForSequenceClassification
        except Exception as e:
            raise RuntimeError("Missing dependencies for evaluation. Install: pip install torch transformers") from e

        tokenizer = _load_tokenizer_stable(str(round_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(round_dir), trust_remote_code=True)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model.to(device)
        model.eval()

        ranking_metrics = evaluate_ranking_metrics(
            tokenizer=tokenizer,
            model=model,
            device=device,
            rows=val_rows,
            batch_size=int(args.infer_batch_size),
            max_length=int(args.max_length),
        )
        validation_loss = evaluate_validation_loss(
            tokenizer=tokenizer,
            model=model,
            device=device,
            rows=val_rows,
            batch_size=int(args.infer_batch_size),
            max_length=int(args.max_length),
            pair_weight=float(args.pair_weight),
            margin_weight=float(args.margin_weight),
            pointwise_weight=float(args.pointwise_weight),
            margin_value=float(args.margin_value),
        )

        llm_eval_summary: Dict[str, Any] = {}
        if int(args.llm_eval_cases) > 0:
            try:
                from train_cross_encoder.eval_bge_reranker_llm_quality import run_llm_quality_eval
            except Exception as e:
                raise RuntimeError(
                    "LLM eval requested, but eval module dependencies are unavailable. "
                    "Install dependencies in your project venv."
                ) from e
            eval_llm_model = _clean_text(args.llm_eval_model) or _default_llm_eval_model()
            if not eval_llm_model:
                raise RuntimeError(
                    "LLM eval requested, but no model id is configured. "
                    "Pass --llm-eval-model explicitly or set BEDROCK_CLAUDE_HAIKU in .env."
                )
            eval_kwargs = {
                "model_dir": str(round_dir),
                "llm_model": eval_llm_model,
                "num_cases": int(args.llm_eval_cases),
                "output_dir": output_root / "eval_runs",
                "cpu_only": bool(args.cpu_only_eval),
                "seed": int(args.seed) + int(round_idx),
                "gate_hard_negative_top1_rate_max": float(args.gate_hard_negative_top1_rate_max),
                "gate_false_positive_rate_max": float(args.gate_false_positive_rate_max),
                "gate_mean_top1_top2_margin_min": float(args.gate_mean_top1_top2_margin_min),
            }
            eval_sig = inspect.signature(run_llm_quality_eval)
            eval_supported = set(eval_sig.parameters.keys())
            eval_call_kwargs = {k: v for k, v in eval_kwargs.items() if k in eval_supported}
            eval_dropped = [k for k in eval_kwargs.keys() if k not in eval_supported]
            llm_eval_summary = run_llm_quality_eval(**eval_call_kwargs)
            if eval_dropped:
                llm_eval_summary["compat_note"] = (
                    "Evaluator version does not support some optional args; ignored: "
                    + ", ".join(eval_dropped)
                )
            acceptance_gates = dict(llm_eval_summary.get("acceptance_gates") or {})
            if not acceptance_gates:
                acceptance_gates = _gate_result(
                    metrics=ranking_metrics,
                    gate_hard_neg_max=float(args.gate_hard_negative_top1_rate_max),
                    gate_fp_max=float(args.gate_false_positive_rate_max),
                    gate_margin_min=float(args.gate_mean_top1_top2_margin_min),
                )
        else:
            acceptance_gates = _gate_result(
                metrics=ranking_metrics,
                gate_hard_neg_max=float(args.gate_hard_negative_top1_rate_max),
                gate_fp_max=float(args.gate_false_positive_rate_max),
                gate_margin_min=float(args.gate_mean_top1_top2_margin_min),
            )

        mine_meta: Dict[str, Any] = {"queries_seen": 0, "queries_mined": 0, "triplets_mined": 0}
        if round_idx < int(args.rounds):
            mined_rows, mine_meta = mine_false_positive_hard_negatives(
                tokenizer=tokenizer,
                model=model,
                device=device,
                rows=train_rows,
                max_per_query=int(args.mine_per_query),
                margin_window=float(args.mine_margin_window),
                hard_negative_weight=float(args.hard_negative_weight),
                batch_size=int(args.infer_batch_size),
                max_length=int(args.max_length),
            )
            if mined_rows:
                mined_rows_all.extend(mined_rows)

        round_summary = {
            "round": int(round_idx),
            "model_dir": str(round_dir),
            "train_rows": int(len(train_rows)),
            "val_rows": int(len(val_rows)),
            "train_info": dict(train_info or {}),
            "ranking_metrics": dict(ranking_metrics or {}),
            "validation_loss": dict(validation_loss or {}),
            "acceptance_gates": dict(acceptance_gates or {}),
            "llm_eval": dict(llm_eval_summary or {}),
            "mined_refresh": dict(mine_meta or {}),
            "mined_rows_cumulative": int(len(mined_rows_all)),
        }
        if wandb is not None and getattr(wandb, "run", None) is not None:
            wandb.log(
                {
                    "round/index": int(round_idx),
                    "round/train_rows": int(len(train_rows)),
                    "round/val_rows": int(len(val_rows)),
                    "round/ranking_top1_accuracy": float(ranking_metrics.get("top1_accuracy") or 0.0),
                    "round/ranking_false_positive_rate": float(ranking_metrics.get("false_positive_rate") or 0.0),
                    "round/ranking_hard_negative_top1_rate": float(ranking_metrics.get("hard_negative_top1_rate") or 0.0),
                    "round/ranking_mean_top1_top2_margin": float(ranking_metrics.get("mean_top1_top2_margin") or 0.0),
                    "round/ranking_mrr": float(ranking_metrics.get("mrr") or 0.0),
                    "round/val_total_loss": float(validation_loss.get("val_total_loss") or 0.0),
                    "round/val_pairwise_loss": float(validation_loss.get("val_pairwise_loss") or 0.0),
                    "round/val_margin_loss": float(validation_loss.get("val_margin_loss") or 0.0),
                    "round/val_pointwise_loss": float(validation_loss.get("val_pointwise_loss") or 0.0),
                    "round/val_pairwise_term": float(validation_loss.get("val_pairwise_term") or 0.0),
                    "round/val_margin_term": float(validation_loss.get("val_margin_term") or 0.0),
                    "round/val_pointwise_term": float(validation_loss.get("val_pointwise_term") or 0.0),
                    "round/gates_all_pass": int(bool(acceptance_gates.get("all_pass"))),
                    "round/mined_queries": int(mine_meta.get("queries_mined") or 0),
                    "round/mined_triplets": int(mine_meta.get("triplets_mined") or 0),
                    "round/mined_triplets_cumulative": int(len(mined_rows_all)),
                },
                step=int(round_idx),
            )
        (round_dir / "round_summary.json").write_text(
            json.dumps(round_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        round_summaries.append(round_summary)
        model_path_for_round = str(round_dir)

        # release GPU/MPS memory for next round
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    promoted = _select_promoted_round(round_summaries)
    promoted_src = Path(_clean_text(promoted.get("model_dir"))).resolve()
    promoted_dst = output_root / "promoted_best"
    if promoted_dst.exists():
        shutil.rmtree(promoted_dst)
    shutil.copytree(promoted_src, promoted_dst)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_jsonl": str(dataset_jsonl),
        "wandb_enabled": bool(wandb_enabled),
        "wandb_run_name": (_clean_text(args.wandb_run_name) if bool(wandb_enabled) else ""),
        "load_stats": dict(load_stats or {}),
        "split": {
            "train_rows_base": int(len(train_rows_base)),
            "val_rows": int(len(val_rows)),
            "val_ratio": float(args.val_ratio),
        },
        "rounds": round_summaries,
        "promoted": {
            "round": int(promoted.get("round") or 0),
            "source_model_dir": str(promoted_src),
            "promoted_model_dir": str(promoted_dst),
            "ranking_metrics": dict(promoted.get("ranking_metrics") or {}),
            "acceptance_gates": dict(promoted.get("acceptance_gates") or {}),
        },
    }
    if wandb is not None and getattr(wandb, "run", None) is not None:
        promoted_metrics = dict(summary.get("promoted", {}).get("ranking_metrics") or {})
        promoted_gates = dict(summary.get("promoted", {}).get("acceptance_gates") or {})
        wandb.summary["promoted_round"] = int(summary.get("promoted", {}).get("round") or 0)
        wandb.summary["promoted_top1_accuracy"] = float(promoted_metrics.get("top1_accuracy") or 0.0)
        wandb.summary["promoted_false_positive_rate"] = float(promoted_metrics.get("false_positive_rate") or 0.0)
        wandb.summary["promoted_hard_negative_top1_rate"] = float(promoted_metrics.get("hard_negative_top1_rate") or 0.0)
        wandb.summary["promoted_mean_top1_top2_margin"] = float(promoted_metrics.get("mean_top1_top2_margin") or 0.0)
        wandb.summary["promoted_gate_pass"] = bool(promoted_gates.get("all_pass"))
    summary_path = output_root / "finetune_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    if wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.finish()
    return summary


def _build_parser() -> argparse.ArgumentParser:
    default_out = Path(__file__).resolve().parent / "models" / f"pairwise_margin_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    p = argparse.ArgumentParser(
        description=(
            "Pairwise+margin cross-encoder fine-tuning with query-group split, "
            "iterative hard-negative refresh, and gate-based checkpoint promotion."
        )
    )
    p.add_argument("--dataset-jsonl", type=str, default="", help="Triplet dataset path. Default: latest finetune_cross_encoder/dataset/*.triplets.jsonl")
    p.add_argument("--output-dir", type=str, default=str(default_out), help="Training output root directory.")
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Base model name/path for round 1.")
    p.add_argument("--max-rows", type=int, default=0, help="Cap rows loaded from dataset (0 = all).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="Training rounds with iterative hard-negative refresh.")
    p.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation ratio by query groups.")
    p.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS, help="Epochs per round.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Per-device train batch size.")
    p.add_argument("--grad-accum-steps", type=int, default=DEFAULT_GRAD_ACCUM, help="Gradient accumulation steps.")
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LR, help="Learning rate.")
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay.")
    p.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO, help="Warmup ratio.")
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Tokenizer max length.")
    p.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS, help="Logging steps.")
    p.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS, help="Checkpoint save steps.")
    p.add_argument("--save-total-limit", type=int, default=DEFAULT_SAVE_TOTAL_LIMIT, help="Checkpoint retention.")
    p.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    p.add_argument("--fp16", action="store_true", help="Enable fp16 on CUDA.")
    p.add_argument("--bf16", action="store_true", help="Enable bf16 on CUDA.")

    p.add_argument("--pair-weight", type=float, default=DEFAULT_PAIRWISE_WEIGHT, help="Weight for pairwise logistic loss.")
    p.add_argument("--margin-weight", type=float, default=DEFAULT_MARGIN_WEIGHT, help="Weight for margin hinge loss.")
    p.add_argument("--pointwise-weight", type=float, default=DEFAULT_POINTWISE_WEIGHT, help="Optional pointwise calibration loss weight.")
    p.add_argument("--margin-value", type=float, default=DEFAULT_MARGIN_VALUE, help="Margin value for hinge term.")
    p.add_argument("--hard-negative-weight", type=float, default=DEFAULT_HARD_NEGATIVE_WEIGHT, help="Sample weight multiplier for hard negatives.")

    p.add_argument("--infer-batch-size", type=int, default=DEFAULT_INFER_BATCH_SIZE, help="Batch size for ranking eval/mining inference.")
    p.add_argument("--mine-per-query", type=int, default=DEFAULT_MINE_PER_QUERY, help="Max mined false-positive hard negatives per query per round.")
    p.add_argument("--mine-margin-window", type=float, default=DEFAULT_MINE_MARGIN_WINDOW, help="Mine negatives with score >= (best_positive_score - window).")

    p.add_argument("--llm-eval-model", type=str, default="", help="LLM model id for probe-based quality eval. Default: settings.haiku if available.")
    p.add_argument("--llm-eval-cases", type=int, default=DEFAULT_LLM_EVAL_CASES, help="Probe cases per round for LLM quality eval. 0 = disable.")
    p.add_argument("--cpu-only-eval", action="store_true", help="Force CPU for LLM quality eval inference.")

    p.add_argument("--gate-hard-negative-top1-rate-max", type=float, default=DEFAULT_GATE_HARD_NEG_MAX, help="Gate threshold: hard_negative_top1_rate <= x")
    p.add_argument("--gate-false-positive-rate-max", type=float, default=DEFAULT_GATE_FP_MAX, help="Gate threshold: false_positive_rate <= x")
    p.add_argument("--gate-mean-top1-top2-margin-min", type=float, default=DEFAULT_GATE_MARGIN_MIN, help="Gate threshold: mean_top1_top2_margin >= x")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-run-name", type=str, default="", help="Optional W&B run name.")
    p.add_argument("--json-only", action="store_true", help="Print JSON summary only.")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    summary = run_finetune(args)

    if not bool(args.json_only):
        promoted = dict(summary.get("promoted") or {})
        metrics = dict(promoted.get("ranking_metrics") or {})
        gates = dict(promoted.get("acceptance_gates") or {})
        print("Pairwise+margin finetuning complete.")
        print(f"  dataset                : {summary.get('dataset_jsonl', '')}")
        print(f"  promoted round         : {promoted.get('round', 0)}")
        print(f"  promoted model dir     : {promoted.get('promoted_model_dir', '')}")
        print(f"  top1 accuracy          : {metrics.get('top1_accuracy', 0.0):.4f}")
        print(f"  false positive rate    : {metrics.get('false_positive_rate', 0.0):.4f}")
        print(f"  hard negative top1 rate: {metrics.get('hard_negative_top1_rate', 0.0):.4f}")
        print(f"  mean top1-top2 margin  : {metrics.get('mean_top1_top2_margin', 0.0):.4f}")
        print(f"  acceptance gates pass  : {bool(gates.get('all_pass'))}")
        print(f"  summary                : {summary.get('summary_path', '')}")
        print()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
