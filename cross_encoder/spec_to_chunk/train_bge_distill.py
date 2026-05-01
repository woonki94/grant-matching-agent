from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # type: ignore[assignment]


# Ensure project root on sys.path for direct script execution.
def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "cross_encoder").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_ID_DEFAULT = "BAAI/bge-reranker-base"
RAW_INPUT_DEFAULT = "cross_encoder/spec_to_chunk/dataset/llm_distill_raw_scores.jsonl"
PAIRWISE_INPUT_DEFAULT = "cross_encoder/spec_to_chunk/dataset/llm_distill_pairwise.jsonl"
SPLIT_DIR_DEFAULT = "cross_encoder/spec_to_chunk/dataset/splits"
RAW_TRAIN_INPUT_DEFAULT = f"{SPLIT_DIR_DEFAULT}/llm_distill_raw_train.jsonl"
RAW_VAL_INPUT_DEFAULT = f"{SPLIT_DIR_DEFAULT}/llm_distill_raw_val.jsonl"
RAW_TEST_INPUT_DEFAULT = f"{SPLIT_DIR_DEFAULT}/llm_distill_raw_test.jsonl"
PAIRWISE_TRAIN_INPUT_DEFAULT = f"{SPLIT_DIR_DEFAULT}/llm_distill_pairwise_train.jsonl"
PAIRWISE_VAL_INPUT_DEFAULT = f"{SPLIT_DIR_DEFAULT}/llm_distill_pairwise_val.jsonl"
PAIRWISE_TEST_INPUT_DEFAULT = f"{SPLIT_DIR_DEFAULT}/llm_distill_pairwise_test.jsonl"
OUTPUT_DIR_DEFAULT = "cross_encoder/spec_to_chunk/models/bge_reranker_distill"
WANDB_PROJECT_DEFAULT = "cross_encoder_distill"


@dataclass
class Candidate:
    text: str
    score: float
    fac_id: int
    chunk_id: int
    chunk_index: int
    source_type: str


@dataclass
class QueryGroup:
    grant_id: str
    spec_idx: int
    query_text: str
    candidates: List[Candidate]


@dataclass
class PairExample:
    grant_id: str
    spec_idx: int
    query_text: str
    pos_text: str
    neg_text: str
    teacher_pos_score: float
    teacher_neg_score: float
    teacher_margin: float
    pair_type: str


class PairwiseDataset(Dataset):
    def __init__(self, rows: Sequence[PairExample]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> PairExample:
        return self.rows[idx]


class ListwiseDataset(Dataset):
    def __init__(self, rows: Sequence[QueryGroup], sampled_lists: Sequence[Dict[str, Any]]) -> None:
        self.rows = list(rows)
        self.sampled_lists = list(sampled_lists)

    def __len__(self) -> int:
        return len(self.sampled_lists)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.sampled_lists[idx]


class PairCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, batch: Sequence[PairExample]) -> Dict[str, Any]:
        queries = [x.query_text for x in batch]
        pos_docs = [x.pos_text for x in batch]
        neg_docs = [x.neg_text for x in batch]

        pos_enc = self.tokenizer(
            queries,
            pos_docs,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        neg_enc = self.tokenizer(
            queries,
            neg_docs,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        margins = torch.tensor(
            [float(x.teacher_margin) for x in batch],
            dtype=torch.float32,
        )
        pos_teacher = torch.tensor([float(x.teacher_pos_score) for x in batch], dtype=torch.float32)
        neg_teacher = torch.tensor([float(x.teacher_neg_score) for x in batch], dtype=torch.float32)

        return {
            "pos": pos_enc,
            "neg": neg_enc,
            "margins": margins,
            "teacher_pos": pos_teacher,
            "teacher_neg": neg_teacher,
        }


class ListCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        queries_flat: List[str] = []
        docs_flat: List[str] = []
        teacher_scores: List[float] = []
        list_sizes: List[int] = []

        for item in batch:
            query = str(item.get("query_text") or "").strip()
            docs = list(item.get("docs") or [])
            if not query or not docs:
                continue
            list_sizes.append(len(docs))
            for d in docs:
                queries_flat.append(query)
                docs_flat.append(str(d.get("text") or "").strip())
                teacher_scores.append(_clamp_01(d.get("teacher_score")))

        if not list_sizes:
            return {
                "enc": None,
                "teacher_scores": None,
                "list_sizes": [],
            }

        enc = self.tokenizer(
            queries_flat,
            docs_flat,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        return {
            "enc": enc,
            "teacher_scores": torch.tensor(teacher_scores, dtype=torch.float32),
            "list_sizes": list_sizes,
        }


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_path(value: Any) -> Path:
    path = Path(_clean_text(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _safe_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    if out < minimum:
        return minimum
    if out > maximum:
        return maximum
    return out


def _safe_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out < minimum:
        return minimum
    if out > maximum:
        return maximum
    return out


def _clamp_01(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        x = 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _parse_csv_items(value: Any) -> List[str]:
    raw = _clean_text(value)
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _float_token(value: float) -> str:
    s = f"{float(value):.6g}".lower()
    return s.replace("-", "m").replace("+", "").replace(".", "p")


def _build_output_suffix(
    *,
    seed: int,
    stage1_epochs: int,
    stage2_epochs: int,
    train_batch_size: int,
    grad_accum_steps: int,
    candidate_pool_size: int,
    mini_list_size: int,
    learning_rate: float,
    teacher_temperature: float,
    loss_kl_weight: float,
    loss_pair_weight: float,
    loss_mse_weight: float,
) -> str:
    parts = [
        f"sd{int(seed)}",
        f"s1{int(stage1_epochs)}",
        f"s2{int(stage2_epochs)}",
        f"bs{int(train_batch_size)}",
        f"ga{int(grad_accum_steps)}",
        f"cp{int(candidate_pool_size)}",
        f"ml{int(mini_list_size)}",
        f"lr{_float_token(float(learning_rate))}",
        f"t{_float_token(float(teacher_temperature))}",
        f"kl{_float_token(float(loss_kl_weight))}",
        f"pw{_float_token(float(loss_pair_weight))}",
        f"mse{_float_token(float(loss_mse_weight))}",
    ]
    return "_".join(parts)


def _wandb_log(run: Any, metrics: Dict[str, Any], *, step: Optional[int] = None) -> None:
    if run is None:
        return
    try:
        if step is None:
            run.log(metrics)
        else:
            run.log(metrics, step=int(step))
    except Exception:
        # Logging failures should not break training.
        return


def _hash_to_unit_interval(text: str) -> float:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return int(h, 16) / float(0xFFFFFFFF)


def _split_grants(
    grant_ids: Sequence[str],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[set[str], set[str], set[str]]:
    uniq = sorted({_clean_text(x) for x in grant_ids if _clean_text(x)})
    if not uniq:
        return set(), set(), set()

    vr = max(0.0, min(0.49, float(val_ratio)))
    tr = max(0.0, min(0.49, float(test_ratio)))
    if vr + tr >= 0.99:
        tr = max(0.0, 0.99 - vr)

    tagged: List[Tuple[float, str]] = []
    for gid in uniq:
        roll = _hash_to_unit_interval(f"{int(seed)}::{gid}")
        tagged.append((roll, gid))
    tagged.sort(key=lambda x: (x[0], x[1]))

    n = len(tagged)
    n_test = int(round(float(n) * tr))
    n_val = int(round(float(n) * vr))
    if n >= 3:
        if tr > 0.0:
            n_test = max(1, n_test)
        if vr > 0.0:
            n_val = max(1, n_val)
    if n_test + n_val >= n:
        overflow = n_test + n_val - (n - 1)
        if overflow > 0:
            reduce_test = min(overflow, n_test)
            n_test -= reduce_test
            overflow -= reduce_test
            if overflow > 0:
                n_val = max(0, n_val - overflow)

    test_ids = [gid for _, gid in tagged[:n_test]]
    val_ids = [gid for _, gid in tagged[n_test : n_test + n_val]]
    train_ids = [gid for _, gid in tagged[n_test + n_val :]]

    if not train_ids:
        if val_ids:
            train_ids.append(val_ids.pop())
        elif test_ids:
            train_ids.append(test_ids.pop())

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    val_set.difference_update(train_set)
    test_set.difference_update(train_set)
    test_set.difference_update(val_set)
    return train_set, val_set, test_set


def _load_raw_groups(path: Path, *, max_queries: int = 0) -> List[QueryGroup]:
    out: List[QueryGroup] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = _clean_text(line)
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            grant_id = _clean_text(obj.get("grant_id"))
            spec_idx = _safe_int(obj.get("spec_idx"), default=0, minimum=0, maximum=50_000_000)
            query_text = _clean_text(obj.get("spec_text") or obj.get("query_text"))
            cand_raw = list(obj.get("candidates") or [])
            if not grant_id or not query_text or not cand_raw:
                continue

            cands: List[Candidate] = []
            for c in cand_raw:
                if not isinstance(c, dict):
                    continue
                doc_text = _clean_text(c.get("chunk_text") or c.get("text"))
                if not doc_text:
                    continue
                cands.append(
                    Candidate(
                        text=doc_text,
                        score=_clamp_01(c.get("score") if "score" in c else c.get("teacher_score")),
                        fac_id=_safe_int(c.get("fac_id"), default=0, minimum=0, maximum=2_147_483_647),
                        chunk_id=_safe_int(c.get("chunk_id"), default=0, minimum=0, maximum=2_147_483_647),
                        chunk_index=_safe_int(c.get("chunk_index"), default=0, minimum=0, maximum=10_000_000),
                        source_type=_clean_text(c.get("source_type")) or "unknown",
                    )
                )

            if not cands:
                continue

            cands.sort(key=lambda x: float(x.score), reverse=True)
            out.append(
                QueryGroup(
                    grant_id=grant_id,
                    spec_idx=int(spec_idx),
                    query_text=query_text,
                    candidates=cands,
                )
            )
            if max_queries > 0 and len(out) >= int(max_queries):
                break

    return out


def _load_pairwise_rows(path: Optional[Path], *, max_rows: int = 0) -> List[PairExample]:
    if path is None or (not path.exists()):
        return []

    out: List[PairExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = _clean_text(line)
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            grant_id = _clean_text(obj.get("grant_id"))
            query_text = _clean_text(obj.get("query_text"))
            pos_text = _clean_text(obj.get("pos_text"))
            neg_text = _clean_text(obj.get("neg_text"))
            if not grant_id or not query_text or not pos_text or not neg_text:
                continue

            out.append(
                PairExample(
                    grant_id=grant_id,
                    spec_idx=_safe_int(obj.get("spec_idx"), default=0, minimum=0, maximum=50_000_000),
                    query_text=query_text,
                    pos_text=pos_text,
                    neg_text=neg_text,
                    teacher_pos_score=_clamp_01(obj.get("teacher_pos_score")),
                    teacher_neg_score=_clamp_01(obj.get("teacher_neg_score")),
                    teacher_margin=_safe_float(obj.get("teacher_margin"), default=0.0, minimum=-1.0, maximum=1.0),
                    pair_type=_clean_text(obj.get("pair_type")) or "unknown",
                )
            )

            if max_rows > 0 and len(out) >= int(max_rows):
                break

    return out


def _derive_pairwise_from_groups(
    groups: Sequence[QueryGroup],
    *,
    per_query_pos_k: int,
    per_query_hard_k: int,
    per_query_weak_k: int,
    per_query_cap: int,
) -> List[PairExample]:
    out: List[PairExample] = []
    pos_k = max(1, int(per_query_pos_k))
    hard_k = max(0, int(per_query_hard_k))
    weak_k = max(1, int(per_query_weak_k))
    cap = max(1, int(per_query_cap))

    for g in groups:
        if not g.candidates:
            continue
        cands = g.candidates
        pos = cands[: min(pos_k, len(cands))]
        hard = cands[min(len(cands), len(pos)) : min(len(cands), len(pos) + hard_k)]
        weak = cands[-min(weak_k, len(cands)) :]

        row_count = 0
        for p in pos:
            for n in weak:
                if p.chunk_id == n.chunk_id and p.fac_id == n.fac_id:
                    continue
                m = float(p.score - n.score)
                if m <= 0:
                    continue
                out.append(
                    PairExample(
                        grant_id=g.grant_id,
                        spec_idx=g.spec_idx,
                        query_text=g.query_text,
                        pos_text=p.text,
                        neg_text=n.text,
                        teacher_pos_score=float(p.score),
                        teacher_neg_score=float(n.score),
                        teacher_margin=float(m),
                        pair_type="derived_strong_vs_weak",
                    )
                )
                row_count += 1
                if row_count >= cap:
                    break
            if row_count >= cap:
                break

            for n in hard:
                if p.chunk_id == n.chunk_id and p.fac_id == n.fac_id:
                    continue
                m = float(p.score - n.score)
                if m <= 0:
                    continue
                out.append(
                    PairExample(
                        grant_id=g.grant_id,
                        spec_idx=g.spec_idx,
                        query_text=g.query_text,
                        pos_text=p.text,
                        neg_text=n.text,
                        teacher_pos_score=float(p.score),
                        teacher_neg_score=float(n.score),
                        teacher_margin=float(m),
                        pair_type="derived_strong_vs_hard",
                    )
                )
                row_count += 1
                if row_count >= cap:
                    break
            if row_count >= cap:
                break

    return out


def _derive_mid_lower_mid_pairs_from_groups(
    groups: Sequence[QueryGroup],
    *,
    per_query_cap: int,
    pos_score_min: float,
    pos_score_max: float,
    neg_score_min: float,
    neg_score_max: float,
    margin_min: float,
    margin_max: float,
    add_easy_contrast: bool,
) -> List[PairExample]:
    """
    Optional boundary-focused augmentation:
    for each query, build pairs where a mid-ranked doc should outrank a lower-mid doc.
    """
    out: List[PairExample] = []
    cap = max(1, int(per_query_cap))

    for g in groups:
        cands = list(g.candidates or [])
        n = len(cands)
        if n < 6:
            continue

        # Candidate list is already sorted high->low by teacher score.
        mid_start = max(1, int(0.30 * n))
        mid_end = min(n, max(mid_start + 1, int(0.55 * n)))
        lower_start = min(n, max(mid_end, int(0.55 * n)))
        lower_end = min(n, max(lower_start + 1, int(0.80 * n)))

        mid_group = cands[mid_start:mid_end]
        lower_mid_group = cands[lower_start:lower_end]
        if not mid_group or not lower_mid_group:
            continue

        row_count = 0
        for p in mid_group:
            p_score = float(p.score)
            if p_score < float(pos_score_min) or p_score > float(pos_score_max):
                continue
            viable = []
            for n_item in lower_mid_group:
                if p.chunk_id == n_item.chunk_id and p.fac_id == n_item.fac_id:
                    continue
                n_score = float(n_item.score)
                if n_score < float(neg_score_min) or n_score > float(neg_score_max):
                    continue
                margin = float(p_score - n_score)
                if margin <= 0.0:
                    continue
                if margin < float(margin_min) or margin > float(margin_max):
                    continue
                viable.append((margin, n_item))
            if not viable:
                continue

            # Pick the closest boundary candidate from lower-mid (harder pair).
            viable.sort(key=lambda x: x[0])
            margin, neg = viable[0]
            out.append(
                PairExample(
                    grant_id=g.grant_id,
                    spec_idx=g.spec_idx,
                    query_text=g.query_text,
                    pos_text=p.text,
                    neg_text=neg.text,
                    teacher_pos_score=float(p.score),
                    teacher_neg_score=float(neg.score),
                        teacher_margin=float(margin),
                        pair_type="derived_mid_vs_lower_mid",
                    )
                )
            row_count += 1
            if row_count >= cap:
                break

            # Optional: add a clear contrast pair as a second signal.
            if add_easy_contrast and len(viable) > 1:
                margin2, neg2 = viable[-1]
                if margin2 != margin or neg2.chunk_id != neg.chunk_id or neg2.fac_id != neg.fac_id:
                    out.append(
                        PairExample(
                            grant_id=g.grant_id,
                            spec_idx=g.spec_idx,
                            query_text=g.query_text,
                            pos_text=p.text,
                            neg_text=neg2.text,
                            teacher_pos_score=float(p_score),
                            teacher_neg_score=float(neg2.score),
                            teacher_margin=float(margin2),
                            pair_type="derived_mid_vs_lower_mid_easy",
                        )
                    )
                    row_count += 1
            if row_count >= cap:
                break

    return out


def _compute_score_stats(scores: Sequence[float]) -> Tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    mx = max(scores)
    mu = sum(scores) / float(len(scores))
    var = sum((x - mu) ** 2 for x in scores) / float(len(scores))
    return float(mx), float(math.sqrt(max(0.0, var)))


def _sample_docs_for_query(
    group: QueryGroup,
    *,
    candidate_pool_size: int,
    mini_list_size: int,
    boundary_center: float,
    boundary_bandwidth: float,
    top_ratio: float,
    boundary_ratio: float,
    random_ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    pool = list(group.candidates[: min(len(group.candidates), max(1, int(candidate_pool_size)))])
    if not pool:
        return []

    k = min(len(pool), max(2, int(mini_list_size)))
    rng_seed = f"{seed}::{group.grant_id}::{group.spec_idx}::{len(pool)}"
    rng = random.Random(int(hashlib.sha1(rng_seed.encode("utf-8")).hexdigest()[:16], 16))

    n_top = int(round(k * float(max(0.0, top_ratio))))
    n_boundary = int(round(k * float(max(0.0, boundary_ratio))))
    n_rand = int(round(k * float(max(0.0, random_ratio))))

    total_n = n_top + n_boundary + n_rand
    if total_n <= 0:
        n_top, n_boundary, n_rand = k // 3, k // 3, k - 2 * (k // 3)
    elif total_n < k:
        n_rand += (k - total_n)
    elif total_n > k:
        over = total_n - k
        cut = min(over, n_rand)
        n_rand -= cut
        over -= cut
        if over > 0:
            cut = min(over, n_boundary)
            n_boundary -= cut
            over -= cut
        if over > 0:
            n_top = max(0, n_top - over)

    selected: List[int] = []
    used = set()

    def add_idx(i: int) -> None:
        if i < 0 or i >= len(pool):
            return
        if i in used:
            return
        used.add(i)
        selected.append(i)

    for i in range(min(n_top, len(pool))):
        add_idx(i)

    if n_boundary > 0:
        center = float(boundary_center)
        width = max(1e-6, float(boundary_bandwidth))
        boundary_scored = []
        for i, c in enumerate(pool):
            if i in used:
                continue
            d = abs(float(c.score) - center)
            in_band_penalty = 0.0 if d <= width else (d - width)
            boundary_scored.append((in_band_penalty, d, i))
        boundary_scored.sort(key=lambda x: (x[0], x[1]))
        candidate_idxs = [i for _, _, i in boundary_scored[: max(n_boundary * 3, n_boundary)]]
        if candidate_idxs:
            pick = min(n_boundary, len(candidate_idxs))
            if pick == len(candidate_idxs):
                for i in candidate_idxs:
                    add_idx(i)
            else:
                for i in rng.sample(candidate_idxs, k=pick):
                    add_idx(i)

    if n_rand > 0:
        remain = [i for i in range(len(pool)) if i not in used]
        if remain:
            pick = min(n_rand, len(remain))
            if pick == len(remain):
                for i in remain:
                    add_idx(i)
            else:
                for i in rng.sample(remain, k=pick):
                    add_idx(i)

    if len(selected) < k:
        remain = [i for i in range(len(pool)) if i not in used]
        rng.shuffle(remain)
        for i in remain:
            if len(selected) >= k:
                break
            add_idx(i)

    docs = [pool[i] for i in selected[:k]]
    docs.sort(key=lambda x: float(x.score), reverse=True)
    return [
        {
            "text": d.text,
            "teacher_score": float(d.score),
            "fac_id": int(d.fac_id),
            "chunk_id": int(d.chunk_id),
            "chunk_index": int(d.chunk_index),
            "source_type": d.source_type,
        }
        for d in docs
    ]


def _build_sampled_list_rows(
    groups: Sequence[QueryGroup],
    *,
    candidate_pool_size: int,
    mini_list_size: int,
    boundary_center: float,
    boundary_bandwidth: float,
    top_ratio: float,
    boundary_ratio: float,
    random_ratio: float,
    low_signal_max_score_threshold: float,
    low_signal_std_threshold: float,
    low_signal_keep_prob: float,
    seed: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for g in groups:
        pool = list(g.candidates[: min(len(g.candidates), max(2, int(candidate_pool_size)))])
        scores = [float(c.score) for c in pool]
        mx, std = _compute_score_stats(scores)

        # Drop low-signal lists except a small retained slice for calibration.
        if mx < float(low_signal_max_score_threshold) and std < float(low_signal_std_threshold):
            keep_key = f"keep::{seed}::{g.grant_id}::{g.spec_idx}"
            keep_roll = _hash_to_unit_interval(keep_key)
            if keep_roll > float(low_signal_keep_prob):
                continue

        docs = _sample_docs_for_query(
            g,
            candidate_pool_size=candidate_pool_size,
            mini_list_size=mini_list_size,
            boundary_center=boundary_center,
            boundary_bandwidth=boundary_bandwidth,
            top_ratio=top_ratio,
            boundary_ratio=boundary_ratio,
            random_ratio=random_ratio,
            seed=seed,
        )
        if len(docs) < 2:
            continue

        out.append(
            {
                "grant_id": g.grant_id,
                "spec_idx": g.spec_idx,
                "query_text": g.query_text,
                "docs": docs,
            }
        )
    return out


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _make_pair_iterator(loader: DataLoader) -> Iterable[Dict[str, Any]]:
    while True:
        any_yield = False
        for b in loader:
            any_yield = True
            yield b
        if not any_yield:
            return


def _variable_margin_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor, margins: torch.Tensor) -> torch.Tensor:
    # max(0, margin - (pos-neg)) with per-example clipped margin target.
    return F.relu(margins - (pos_logits - neg_logits)).mean()


def _compute_listwise_kl_and_mse(
    *,
    logits_flat: torch.Tensor,
    teacher_scores_flat: torch.Tensor,
    list_sizes: Sequence[int],
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kl_parts: List[torch.Tensor] = []
    mse_parts: List[torch.Tensor] = []

    cursor = 0
    t = max(1e-6, float(temperature))

    for sz in list_sizes:
        n = int(sz)
        if n <= 1:
            cursor += max(0, n)
            continue
        s = logits_flat[cursor : cursor + n]
        y = teacher_scores_flat[cursor : cursor + n].clamp(0.0, 1.0)
        cursor += n

        teacher_probs = torch.softmax(y / t, dim=-1)
        student_log_probs = torch.log_softmax(s / t, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (t * t)
        mse = F.mse_loss(torch.sigmoid(s), y)
        kl_parts.append(kl)
        mse_parts.append(mse)

    if not kl_parts:
        zero = torch.zeros((), device=logits_flat.device, dtype=logits_flat.dtype)
        return zero, zero

    kl_loss = torch.stack(kl_parts).mean()
    mse_loss = torch.stack(mse_parts).mean()
    return kl_loss, mse_loss


def _dcg(scores: Sequence[float], k: int) -> float:
    lim = min(len(scores), max(0, int(k)))
    if lim <= 0:
        return 0.0
    out = 0.0
    for i in range(lim):
        rel = float(scores[i])
        out += (2.0**rel - 1.0) / math.log2(i + 2.0)
    return out


def _evaluate(
    *,
    model: nn.Module,
    tokenizer: Any,
    groups: Sequence[QueryGroup],
    device: torch.device,
    max_length: int,
    eval_batch_size: int,
    candidate_pool_size: int,
    mrr_rel_threshold: float,
    recall_rel_threshold: float,
) -> Dict[str, float]:
    model.eval()

    ndcg10_vals: List[float] = []
    mrr10_vals: List[float] = []
    recall50_vals: List[float] = []

    with torch.no_grad():
        for g in groups:
            pool = list(g.candidates[: min(len(g.candidates), max(2, int(candidate_pool_size)))])
            if len(pool) < 2:
                continue

            queries = [g.query_text] * len(pool)
            docs = [c.text for c in pool]

            logits_parts: List[torch.Tensor] = []
            for i in range(0, len(pool), max(1, int(eval_batch_size))):
                q_b = queries[i : i + int(eval_batch_size)]
                d_b = docs[i : i + int(eval_batch_size)]
                enc = tokenizer(
                    q_b,
                    d_b,
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                enc = _to_device(enc, device)
                logits = model(**enc).logits.squeeze(-1)
                logits_parts.append(logits.detach().cpu())

            student_logits = torch.cat(logits_parts, dim=0).tolist()
            teacher_scores = [float(c.score) for c in pool]

            ranked_idx = sorted(range(len(pool)), key=lambda i: float(student_logits[i]), reverse=True)
            student_ranked_teacher = [teacher_scores[i] for i in ranked_idx]
            ideal_teacher = sorted(teacher_scores, reverse=True)

            dcg10 = _dcg(student_ranked_teacher, 10)
            idcg10 = _dcg(ideal_teacher, 10)
            ndcg10 = (dcg10 / idcg10) if idcg10 > 0 else 0.0
            ndcg10_vals.append(float(ndcg10))

            # MRR@10 using teacher score threshold for relevance.
            mrr = 0.0
            for rank, idx in enumerate(ranked_idx[:10], start=1):
                if teacher_scores[idx] >= float(mrr_rel_threshold):
                    mrr = 1.0 / float(rank)
                    break
            mrr10_vals.append(float(mrr))

            # Recall@50 with teacher threshold for relevance.
            relevant_all = {i for i, s in enumerate(teacher_scores) if s >= float(recall_rel_threshold)}
            if relevant_all:
                top50 = set(ranked_idx[:50])
                recall = len(top50.intersection(relevant_all)) / float(len(relevant_all))
                recall50_vals.append(float(recall))

    return {
        "ndcg@10": float(sum(ndcg10_vals) / max(1, len(ndcg10_vals))),
        "mrr@10": float(sum(mrr10_vals) / max(1, len(mrr10_vals))),
        "recall@50": float(sum(recall50_vals) / max(1, len(recall50_vals))) if recall50_vals else 0.0,
        "eval_queries": int(len(ndcg10_vals)),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune BGE cross-encoder with staged distillation (pairwise warm start + listwise KD)."
    )

    p.add_argument("--raw-input", type=str, default=RAW_INPUT_DEFAULT, help="Raw distillation JSONL path.")
    p.add_argument(
        "--pairwise-input",
        type=str,
        default=PAIRWISE_INPUT_DEFAULT,
        help="Pairwise JSONL path (optional; if missing, pairs are derived from raw).",
    )
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT, help="Output model/checkpoint directory.")
    p.add_argument(
        "--append-args-to-output-dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append key hyperparameter args to output-dir name (default: true).",
    )

    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT, help="Cross-encoder base model id.")
    p.add_argument("--max-length", type=int, default=512, help="Tokenizer max sequence length.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument(
        "--split-dir",
        type=str,
        default=SPLIT_DIR_DEFAULT,
        help="Directory for deterministic split files (train/val/test JSONL).",
    )
    p.add_argument(
        "--use-prepared-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pre-generated split files. If missing, generate automatically.",
    )
    p.add_argument(
        "--regenerate-splits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force regeneration of split files before training.",
    )
    p.add_argument("--raw-train-input", type=str, default=RAW_TRAIN_INPUT_DEFAULT, help="Prepared raw train JSONL path.")
    p.add_argument("--raw-val-input", type=str, default=RAW_VAL_INPUT_DEFAULT, help="Prepared raw val/eval JSONL path.")
    p.add_argument("--raw-test-input", type=str, default=RAW_TEST_INPUT_DEFAULT, help="Prepared raw test JSONL path.")
    p.add_argument(
        "--pairwise-train-input",
        type=str,
        default=PAIRWISE_TRAIN_INPUT_DEFAULT,
        help="Prepared pairwise train JSONL path.",
    )
    p.add_argument(
        "--pairwise-val-input",
        type=str,
        default=PAIRWISE_VAL_INPUT_DEFAULT,
        help="Prepared pairwise val/eval JSONL path.",
    )
    p.add_argument(
        "--pairwise-test-input",
        type=str,
        default=PAIRWISE_TEST_INPUT_DEFAULT,
        help="Prepared pairwise test JSONL path.",
    )
    p.add_argument("--max-train-queries", type=int, default=0, help="Cap raw query groups loaded (0=all).")
    p.add_argument("--max-pairwise-rows", type=int, default=0, help="Cap pairwise rows loaded (0=all).")

    p.add_argument("--candidate-pool-size", type=int, default=64, help="Candidate pool per query before mini-list sampling.")
    p.add_argument("--mini-list-size", type=int, default=10, help="Mini-list docs per query for listwise loss.")

    p.add_argument("--boundary-center", type=float, default=0.6, help="Boundary score center for informative mids.")
    p.add_argument("--boundary-bandwidth", type=float, default=0.12, help="Boundary bandwidth around center.")
    p.add_argument("--top-ratio", type=float, default=0.4)
    p.add_argument("--boundary-ratio", type=float, default=0.4)
    p.add_argument("--random-ratio", type=float, default=0.2)

    p.add_argument("--low-signal-max-score-threshold", type=float, default=0.3)
    p.add_argument("--low-signal-std-threshold", type=float, default=0.05)
    p.add_argument("--low-signal-keep-prob", type=float, default=0.1)

    p.add_argument("--teacher-temperature", type=float, default=2.0)

    p.add_argument("--stage1-epochs", type=int, default=1)
    p.add_argument("--stage2-epochs", type=int, default=3)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    p.add_argument("--loss-kl-weight", type=float, default=1.0)
    p.add_argument("--loss-pair-weight", type=float, default=0.5)
    p.add_argument("--loss-mse-weight", type=float, default=0.05)

    p.add_argument("--margin-min", type=float, default=0.1)
    p.add_argument("--margin-max", type=float, default=0.7)

    p.add_argument("--pair-derive-pos-k", type=int, default=4)
    p.add_argument("--pair-derive-hard-k", type=int, default=4)
    p.add_argument("--pair-derive-weak-k", type=int, default=4)
    p.add_argument("--pair-derive-cap", type=int, default=64)
    p.add_argument(
        "--pair-add-mid-lower-mid",
        action="store_true",
        help=(
            "Augment pairwise rows with boundary-focused pairs derived from raw ranking: "
            "mid-group > lower-mid-group."
        ),
    )
    p.add_argument(
        "--pair-mid-pos-score-min",
        type=float,
        default=0.4,
        help="Min teacher score for mid-group positive in mid>lower-mid augmentation.",
    )
    p.add_argument(
        "--pair-mid-pos-score-max",
        type=float,
        default=0.7,
        help="Max teacher score for mid-group positive in mid>lower-mid augmentation.",
    )
    p.add_argument(
        "--pair-mid-neg-score-min",
        type=float,
        default=0.2,
        help="Min teacher score for lower-mid negative in mid>lower-mid augmentation.",
    )
    p.add_argument(
        "--pair-mid-neg-score-max",
        type=float,
        default=0.5,
        help="Max teacher score for lower-mid negative in mid>lower-mid augmentation.",
    )
    p.add_argument(
        "--pair-mid-margin-min",
        type=float,
        default=0.05,
        help="Min teacher margin for mid>lower-mid augmentation pairs.",
    )
    p.add_argument(
        "--pair-mid-margin-max",
        type=float,
        default=0.4,
        help="Max teacher margin for mid>lower-mid augmentation pairs.",
    )
    p.add_argument(
        "--pair-mid-add-easy-contrast",
        action="store_true",
        help="For each mid item, also add one larger-margin lower-mid pair (if available).",
    )

    p.add_argument("--mrr-rel-threshold", type=float, default=0.7)
    p.add_argument("--recall-rel-threshold", type=float, default=0.7)

    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument(
        "--eval-every-steps",
        type=int,
        default=100,
        help="Run validation losses every N optimizer steps (0 disables step validation).",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 autocast on CUDA.")
    p.add_argument("--fp16", action="store_true", help="Use float16 autocast on CUDA.")
    p.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars.")
    p.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    p.add_argument("--wandb-project", type=str, default=WANDB_PROJECT_DEFAULT, help="W&B project name.")
    p.add_argument("--wandb-entity", type=str, default="", help="W&B entity/team (optional).")
    p.add_argument("--wandb-run-name", type=str, default="", help="W&B run name (optional).")
    p.add_argument("--wandb-tags", type=str, default="cross-encoder,distill,bge", help="Comma-separated W&B tags.")
    p.add_argument("--wandb-group", type=str, default="", help="W&B group name (optional).")
    p.add_argument("--wandb-dir", type=str, default="", help="W&B local directory (default: output-dir).")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    raw_path = _resolve_path(args.raw_input)
    pairwise_path = _resolve_path(args.pairwise_input)
    output_dir_base = _resolve_path(args.output_dir)

    if not raw_path.exists():
        raise RuntimeError(f"raw_input not found: {raw_path}")

    seed = _safe_int(args.seed, default=42, minimum=0, maximum=2_147_483_647)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    max_train_queries = _safe_int(args.max_train_queries, default=0, minimum=0, maximum=100_000_000)
    max_pairwise_rows = _safe_int(args.max_pairwise_rows, default=0, minimum=0, maximum=2_000_000_000)

    candidate_pool_size = _safe_int(args.candidate_pool_size, default=64, minimum=2, maximum=10_000)
    mini_list_size = _safe_int(args.mini_list_size, default=10, minimum=2, maximum=256)

    stage1_epochs = _safe_int(args.stage1_epochs, default=1, minimum=0, maximum=100)
    stage2_epochs = _safe_int(args.stage2_epochs, default=3, minimum=0, maximum=100)
    train_batch_size = _safe_int(args.train_batch_size, default=8, minimum=1, maximum=4096)
    eval_batch_size = _safe_int(args.eval_batch_size, default=32, minimum=1, maximum=4096)
    grad_accum_steps = _safe_int(args.grad_accum_steps, default=8, minimum=1, maximum=1024)
    learning_rate = _safe_float(args.learning_rate, default=2e-5, minimum=1e-8, maximum=1.0)
    weight_decay = _safe_float(args.weight_decay, default=0.01, minimum=0.0, maximum=10.0)
    max_grad_norm = _safe_float(args.max_grad_norm, default=1.0, minimum=0.0, maximum=100.0)

    margin_min = _safe_float(args.margin_min, default=0.1, minimum=0.0, maximum=1.0)
    margin_max = _safe_float(args.margin_max, default=0.7, minimum=0.0, maximum=2.0)
    if margin_max < margin_min:
        margin_max = margin_min

    loss_kl_weight = _safe_float(args.loss_kl_weight, default=1.0, minimum=0.0, maximum=100.0)
    loss_pair_weight = _safe_float(args.loss_pair_weight, default=0.5, minimum=0.0, maximum=100.0)
    loss_mse_weight = _safe_float(args.loss_mse_weight, default=0.05, minimum=0.0, maximum=100.0)

    teacher_temperature = _safe_float(args.teacher_temperature, default=2.0, minimum=1e-6, maximum=100.0)

    low_signal_max_score_threshold = _safe_float(
        args.low_signal_max_score_threshold,
        default=0.3,
        minimum=0.0,
        maximum=1.0,
    )
    low_signal_std_threshold = _safe_float(args.low_signal_std_threshold, default=0.05, minimum=0.0, maximum=1.0)
    low_signal_keep_prob = _safe_float(args.low_signal_keep_prob, default=0.1, minimum=0.0, maximum=1.0)
    pair_mid_pos_score_min = _safe_float(args.pair_mid_pos_score_min, default=0.4, minimum=0.0, maximum=1.0)
    pair_mid_pos_score_max = _safe_float(args.pair_mid_pos_score_max, default=0.7, minimum=0.0, maximum=1.0)
    pair_mid_neg_score_min = _safe_float(args.pair_mid_neg_score_min, default=0.2, minimum=0.0, maximum=1.0)
    pair_mid_neg_score_max = _safe_float(args.pair_mid_neg_score_max, default=0.5, minimum=0.0, maximum=1.0)
    pair_mid_margin_min = _safe_float(args.pair_mid_margin_min, default=0.05, minimum=0.0, maximum=1.0)
    pair_mid_margin_max = _safe_float(args.pair_mid_margin_max, default=0.4, minimum=0.0, maximum=1.0)
    pair_mid_add_easy_contrast = bool(args.pair_mid_add_easy_contrast)

    if pair_mid_pos_score_max < pair_mid_pos_score_min:
        pair_mid_pos_score_max = pair_mid_pos_score_min
    if pair_mid_neg_score_max < pair_mid_neg_score_min:
        pair_mid_neg_score_max = pair_mid_neg_score_min
    if pair_mid_margin_max < pair_mid_margin_min:
        pair_mid_margin_max = pair_mid_margin_min

    max_length = _safe_int(args.max_length, default=512, minimum=16, maximum=8192)
    val_ratio = _safe_float(args.val_ratio, default=0.1, minimum=0.0, maximum=0.5)
    test_ratio = _safe_float(args.test_ratio, default=0.1, minimum=0.0, maximum=0.5)
    if val_ratio + test_ratio >= 0.99:
        test_ratio = max(0.0, 0.99 - val_ratio)
    log_every_steps = _safe_int(args.log_every_steps, default=50, minimum=1, maximum=1_000_000)
    eval_every_steps = _safe_int(args.eval_every_steps, default=100, minimum=0, maximum=1_000_000)
    append_args_to_output_dir = bool(args.append_args_to_output_dir)
    use_tqdm = not bool(args.no_tqdm)
    if use_tqdm and tqdm is None:
        print("tqdm_unavailable=true")
        use_tqdm = False

    output_suffix = ""
    if append_args_to_output_dir:
        output_suffix = _build_output_suffix(
            seed=seed,
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
            train_batch_size=train_batch_size,
            grad_accum_steps=grad_accum_steps,
            candidate_pool_size=candidate_pool_size,
            mini_list_size=mini_list_size,
            learning_rate=learning_rate,
            teacher_temperature=teacher_temperature,
            loss_kl_weight=loss_kl_weight,
            loss_pair_weight=loss_pair_weight,
            loss_mse_weight=loss_mse_weight,
        )
        output_dir = (output_dir_base.parent / f"{output_dir_base.name}__{output_suffix}").resolve()
    else:
        output_dir = output_dir_base
    output_dir.mkdir(parents=True, exist_ok=True)

    use_prepared_splits = bool(args.use_prepared_splits)
    regenerate_splits = bool(args.regenerate_splits)
    split_dir = _resolve_path(args.split_dir)
    raw_train_override_path = _resolve_path(args.raw_train_input)
    raw_val_override_path = _resolve_path(args.raw_val_input)
    raw_test_override_path = _resolve_path(args.raw_test_input)
    pair_train_override_path = _resolve_path(args.pairwise_train_input)
    pair_val_override_path = _resolve_path(args.pairwise_val_input)
    pair_test_override_path = _resolve_path(args.pairwise_test_input)
    split_policy = "deterministic_hash_by_grant_id"
    split_generated = False
    split_result: Optional[Dict[str, Any]] = None
    split_manifest_path: Optional[Path] = None
    split_raw_train_path = raw_train_override_path
    split_raw_val_path = raw_val_override_path
    split_raw_test_path = raw_test_override_path
    split_pair_train_path = pair_train_override_path
    split_pair_val_path = pair_val_override_path
    split_pair_test_path = pair_test_override_path

    if use_prepared_splits:
        split_raw_ready = (
            split_raw_train_path.exists()
            and split_raw_val_path.exists()
            and split_raw_test_path.exists()
        )
        split_pair_ready = (
            split_pair_train_path.exists()
            and split_pair_val_path.exists()
            and split_pair_test_path.exists()
        )
        if split_raw_ready and (split_pair_ready or not pairwise_path.exists()) and not regenerate_splits:
            split_policy = "prepared_split_files_manual_paths"
        else:
            from cross_encoder.spec_to_chunk.data_preparation.split_distill_dataset import ensure_split_files

            split_result = ensure_split_files(
                raw_input=raw_path,
                pairwise_input=pairwise_path if pairwise_path.exists() else None,
                split_dir=split_dir,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                overwrite=regenerate_splits,
            )
            split_paths = split_result["paths"]
            split_generated = bool(split_result.get("generated"))
            split_manifest_path = split_paths.manifest
            split_policy = "prepared_split_files"

            split_raw_train_path = split_paths.raw_train
            split_raw_val_path = split_paths.raw_val
            split_raw_test_path = split_paths.raw_test
            split_pair_train_path = split_paths.pair_train
            split_pair_val_path = split_paths.pair_val
            split_pair_test_path = split_paths.pair_test

    wandb_run: Any = None
    wandb_enabled = not bool(args.no_wandb)
    if wandb_enabled:
        try:
            import wandb  # type: ignore

            wandb_config = dict(vars(args))
            wandb_config["raw_input"] = str(raw_path)
            wandb_config["pairwise_input"] = str(pairwise_path)
            wandb_config["output_dir_base"] = str(output_dir_base)
            wandb_config["output_dir"] = str(output_dir)
            wandb_config["output_suffix"] = str(output_suffix)
            wandb_config["split_dir"] = str(split_dir)
            wandb_config["split_policy"] = str(split_policy)
            wandb_config["use_prepared_splits"] = bool(use_prepared_splits)
            wandb_config["regenerate_splits"] = bool(regenerate_splits)
            wandb_config["split_generated"] = bool(split_generated)

            init_kwargs: Dict[str, Any] = {
                "project": _clean_text(args.wandb_project) or WANDB_PROJECT_DEFAULT,
                "config": wandb_config,
                "tags": _parse_csv_items(args.wandb_tags),
                "dir": _clean_text(args.wandb_dir) or str(output_dir),
            }
            entity = _clean_text(args.wandb_entity)
            run_name = _clean_text(args.wandb_run_name)
            group = _clean_text(args.wandb_group)
            if entity:
                init_kwargs["entity"] = entity
            if run_name:
                init_kwargs["name"] = run_name
            if group:
                init_kwargs["group"] = group

            wandb_run = wandb.init(**init_kwargs)
            if wandb_run is not None:
                print(f"wandb_enabled=true project={init_kwargs.get('project')}")
                if _clean_text(getattr(wandb_run, "id", "")):
                    print(f"wandb_run_id={_clean_text(getattr(wandb_run, 'id', ''))}")
                if _clean_text(getattr(wandb_run, "url", "")):
                    print(f"wandb_run_url={_clean_text(getattr(wandb_run, 'url', ''))}")
            else:
                print("wandb_enabled=false reason=wandb.init_returned_none")
        except Exception as e:
            print(f"wandb_enabled=false reason={type(e).__name__}:{e}")
            wandb_run = None
    else:
        print("wandb_enabled=false reason=no_wandb_flag")

    print(f"raw_input={raw_path}")
    print(f"pairwise_input={pairwise_path} exists={pairwise_path.exists()}")
    print(f"output_dir_base={output_dir_base}")
    print(f"append_args_to_output_dir={append_args_to_output_dir}")
    if output_suffix:
        print(f"output_suffix={output_suffix}")
    print(f"output_dir={output_dir}")
    print(f"use_tqdm={use_tqdm}")
    print(f"use_prepared_splits={use_prepared_splits}")
    print(f"split_policy={split_policy}")
    print(f"split_dir={split_dir}")
    if use_prepared_splits:
        print(f"split_generated={split_generated}")
        print(f"split_manifest={split_manifest_path}")
        print(f"raw_split_train={split_raw_train_path}")
        print(f"raw_split_val={split_raw_val_path}")
        print(f"raw_split_test={split_raw_test_path}")
        print(f"pair_split_train={split_pair_train_path} exists={split_pair_train_path.exists()}")
        print(f"pair_split_val={split_pair_val_path} exists={split_pair_val_path.exists()}")
        print(f"pair_split_test={split_pair_test_path} exists={split_pair_test_path.exists()}")
        if split_result is not None:
            raw_band_counts = split_result.get("raw_file_band_counts") or {}
            if isinstance(raw_band_counts, dict):
                print("raw_file_band_counts:")
                for name in ("train", "val", "test"):
                    c = raw_band_counts.get(name) or {}
                    print(
                        f"  {name}: total={int(c.get('total', 0))} "
                        f"high={int(c.get('high', 0))} mid={int(c.get('mid', 0))} low={int(c.get('low', 0))}"
                    )
            pair_band_counts = split_result.get("pair_file_band_counts") or {}
            if isinstance(pair_band_counts, dict) and pair_band_counts:
                print("pair_file_band_counts:")
                for name in ("train", "val", "test"):
                    c = pair_band_counts.get(name) or {}
                    print(
                        f"  {name}: total={int(c.get('total', 0))} "
                        f"high={int(c.get('high', 0))} mid={int(c.get('mid', 0))} low={int(c.get('low', 0))}"
                    )

    if use_prepared_splits:
        train_groups = _load_raw_groups(split_raw_train_path, max_queries=max_train_queries)
        val_groups = _load_raw_groups(split_raw_val_path, max_queries=0)
        test_groups = _load_raw_groups(split_raw_test_path, max_queries=0)
        groups_all = list(train_groups) + list(val_groups) + list(test_groups)
    else:
        groups_all = _load_raw_groups(raw_path, max_queries=max_train_queries)
        if not groups_all:
            raise RuntimeError("No valid query groups loaded from raw_input.")

        train_grants, val_grants, test_grants = _split_grants(
            [g.grant_id for g in groups_all],
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        train_groups = [g for g in groups_all if g.grant_id in train_grants]
        val_groups = [g for g in groups_all if g.grant_id in val_grants]
        test_groups = [g for g in groups_all if g.grant_id in test_grants]

    if not train_groups:
        raise RuntimeError("Training split is empty.")

    train_grants = {g.grant_id for g in train_groups}
    val_grants = {g.grant_id for g in val_groups}
    test_grants = {g.grant_id for g in test_groups}

    pair_rows_source = "pairwise_file"
    train_pairs: List[PairExample] = []
    val_pairs: List[PairExample] = []
    test_pairs: List[PairExample] = []

    if use_prepared_splits and split_pair_train_path.exists() and split_pair_val_path.exists() and split_pair_test_path.exists():
        train_pairs = _load_pairwise_rows(split_pair_train_path, max_rows=max_pairwise_rows)
        val_pairs = _load_pairwise_rows(split_pair_val_path, max_rows=0)
        test_pairs = _load_pairwise_rows(split_pair_test_path, max_rows=0)
        pair_rows_source = "pairwise_split_files"
    else:
        pair_rows_loaded = _load_pairwise_rows(pairwise_path, max_rows=max_pairwise_rows)
        if pair_rows_loaded:
            train_pairs = [r for r in pair_rows_loaded if r.grant_id in train_grants]
            val_pairs = [r for r in pair_rows_loaded if r.grant_id in val_grants]
            test_pairs = [r for r in pair_rows_loaded if r.grant_id in test_grants]

    if not train_pairs:
        print("pairwise_rows_loaded=0; deriving pairwise rows from raw groups")
        train_pairs = _derive_pairwise_from_groups(
            train_groups,
            per_query_pos_k=_safe_int(args.pair_derive_pos_k, default=4, minimum=1, maximum=10_000),
            per_query_hard_k=_safe_int(args.pair_derive_hard_k, default=4, minimum=0, maximum=10_000),
            per_query_weak_k=_safe_int(args.pair_derive_weak_k, default=4, minimum=1, maximum=10_000),
            per_query_cap=_safe_int(args.pair_derive_cap, default=64, minimum=1, maximum=100_000),
        )
        val_pairs = _derive_pairwise_from_groups(
            val_groups,
            per_query_pos_k=_safe_int(args.pair_derive_pos_k, default=4, minimum=1, maximum=10_000),
            per_query_hard_k=_safe_int(args.pair_derive_hard_k, default=4, minimum=0, maximum=10_000),
            per_query_weak_k=_safe_int(args.pair_derive_weak_k, default=4, minimum=1, maximum=10_000),
            per_query_cap=_safe_int(args.pair_derive_cap, default=64, minimum=1, maximum=100_000),
        )
        test_pairs = _derive_pairwise_from_groups(
            test_groups,
            per_query_pos_k=_safe_int(args.pair_derive_pos_k, default=4, minimum=1, maximum=10_000),
            per_query_hard_k=_safe_int(args.pair_derive_hard_k, default=4, minimum=0, maximum=10_000),
            per_query_weak_k=_safe_int(args.pair_derive_weak_k, default=4, minimum=1, maximum=10_000),
            per_query_cap=_safe_int(args.pair_derive_cap, default=64, minimum=1, maximum=100_000),
        )
        pair_rows_source = "derived_from_raw_splits" if use_prepared_splits else "derived_from_raw"

    pair_mid_lower_enabled = bool(args.pair_add_mid_lower_mid)
    pair_mid_lower_added = 0
    if pair_mid_lower_enabled:
        train_mid_rows = _derive_mid_lower_mid_pairs_from_groups(
            train_groups,
            per_query_cap=_safe_int(args.pair_derive_cap, default=64, minimum=1, maximum=100_000),
            pos_score_min=pair_mid_pos_score_min,
            pos_score_max=pair_mid_pos_score_max,
            neg_score_min=pair_mid_neg_score_min,
            neg_score_max=pair_mid_neg_score_max,
            margin_min=pair_mid_margin_min,
            margin_max=pair_mid_margin_max,
            add_easy_contrast=pair_mid_add_easy_contrast,
        )
        val_mid_rows = _derive_mid_lower_mid_pairs_from_groups(
            val_groups,
            per_query_cap=_safe_int(args.pair_derive_cap, default=64, minimum=1, maximum=100_000),
            pos_score_min=pair_mid_pos_score_min,
            pos_score_max=pair_mid_pos_score_max,
            neg_score_min=pair_mid_neg_score_min,
            neg_score_max=pair_mid_neg_score_max,
            margin_min=pair_mid_margin_min,
            margin_max=pair_mid_margin_max,
            add_easy_contrast=pair_mid_add_easy_contrast,
        )
        test_mid_rows = _derive_mid_lower_mid_pairs_from_groups(
            test_groups,
            per_query_cap=_safe_int(args.pair_derive_cap, default=64, minimum=1, maximum=100_000),
            pos_score_min=pair_mid_pos_score_min,
            pos_score_max=pair_mid_pos_score_max,
            neg_score_min=pair_mid_neg_score_min,
            neg_score_max=pair_mid_neg_score_max,
            margin_min=pair_mid_margin_min,
            margin_max=pair_mid_margin_max,
            add_easy_contrast=pair_mid_add_easy_contrast,
        )
        pair_mid_lower_added = int(len(train_mid_rows) + len(val_mid_rows) + len(test_mid_rows))
        if train_mid_rows:
            train_pairs = list(train_pairs) + train_mid_rows
        if val_mid_rows:
            val_pairs = list(val_pairs) + val_mid_rows
        if test_mid_rows:
            test_pairs = list(test_pairs) + test_mid_rows
        print(f"pair_mid_lower_enabled=true pair_mid_lower_added={pair_mid_lower_added}")
        print(
            "pair_mid_config="
            f"pos:[{pair_mid_pos_score_min:.3f},{pair_mid_pos_score_max:.3f}] "
            f"neg:[{pair_mid_neg_score_min:.3f},{pair_mid_neg_score_max:.3f}] "
            f"margin:[{pair_mid_margin_min:.3f},{pair_mid_margin_max:.3f}] "
            f"easy_contrast={pair_mid_add_easy_contrast}"
        )
    else:
        print("pair_mid_lower_enabled=false")

    if not train_pairs:
        raise RuntimeError("No training pairwise rows available after split.")
    pair_rows = list(train_pairs) + list(val_pairs) + list(test_pairs)

    print(f"queries_total={len(groups_all)}")
    print(f"queries_train={len(train_groups)}")
    print(f"queries_val={len(val_groups)}")
    print(f"queries_test={len(test_groups)}")
    print(f"pair_rows_source={pair_rows_source}")
    print(f"pairwise_total_rows={len(pair_rows)}")
    print(f"pairwise_train={len(train_pairs)}")
    print(f"pairwise_val={len(val_pairs)}")
    print(f"pairwise_test={len(test_pairs)}")

    train_list_rows = _build_sampled_list_rows(
        train_groups,
        candidate_pool_size=candidate_pool_size,
        mini_list_size=mini_list_size,
        boundary_center=_safe_float(args.boundary_center, default=0.6, minimum=0.0, maximum=1.0),
        boundary_bandwidth=_safe_float(args.boundary_bandwidth, default=0.12, minimum=1e-6, maximum=1.0),
        top_ratio=_safe_float(args.top_ratio, default=0.4, minimum=0.0, maximum=1.0),
        boundary_ratio=_safe_float(args.boundary_ratio, default=0.4, minimum=0.0, maximum=1.0),
        random_ratio=_safe_float(args.random_ratio, default=0.2, minimum=0.0, maximum=1.0),
        low_signal_max_score_threshold=low_signal_max_score_threshold,
        low_signal_std_threshold=low_signal_std_threshold,
        low_signal_keep_prob=low_signal_keep_prob,
        seed=seed,
    )
    val_list_rows = _build_sampled_list_rows(
        val_groups,
        candidate_pool_size=candidate_pool_size,
        mini_list_size=mini_list_size,
        boundary_center=_safe_float(args.boundary_center, default=0.6, minimum=0.0, maximum=1.0),
        boundary_bandwidth=_safe_float(args.boundary_bandwidth, default=0.12, minimum=1e-6, maximum=1.0),
        top_ratio=_safe_float(args.top_ratio, default=0.4, minimum=0.0, maximum=1.0),
        boundary_ratio=_safe_float(args.boundary_ratio, default=0.4, minimum=0.0, maximum=1.0),
        random_ratio=_safe_float(args.random_ratio, default=0.2, minimum=0.0, maximum=1.0),
        low_signal_max_score_threshold=low_signal_max_score_threshold,
        low_signal_std_threshold=low_signal_std_threshold,
        low_signal_keep_prob=1.0,  # keep all val rows for stable eval.
        seed=seed + 1,
    )
    test_list_rows = _build_sampled_list_rows(
        test_groups,
        candidate_pool_size=candidate_pool_size,
        mini_list_size=mini_list_size,
        boundary_center=_safe_float(args.boundary_center, default=0.6, minimum=0.0, maximum=1.0),
        boundary_bandwidth=_safe_float(args.boundary_bandwidth, default=0.12, minimum=1e-6, maximum=1.0),
        top_ratio=_safe_float(args.top_ratio, default=0.4, minimum=0.0, maximum=1.0),
        boundary_ratio=_safe_float(args.boundary_ratio, default=0.4, minimum=0.0, maximum=1.0),
        random_ratio=_safe_float(args.random_ratio, default=0.2, minimum=0.0, maximum=1.0),
        low_signal_max_score_threshold=low_signal_max_score_threshold,
        low_signal_std_threshold=low_signal_std_threshold,
        low_signal_keep_prob=1.0,
        seed=seed + 2,
    )

    if not train_list_rows:
        raise RuntimeError("No listwise rows after filtering/sampling for training.")

    print(f"listwise_train={len(train_list_rows)}")
    print(f"listwise_val={len(val_list_rows)}")
    print(f"listwise_test={len(test_list_rows)}")
    _wandb_log(
        wandb_run,
        {
            "data/queries_total": int(len(groups_all)),
            "data/queries_train": int(len(train_groups)),
            "data/queries_val": int(len(val_groups)),
            "data/queries_test": int(len(test_groups)),
            "data/pairwise_train": int(len(train_pairs)),
            "data/pairwise_val": int(len(val_pairs)),
            "data/pairwise_test": int(len(test_pairs)),
            "data/pairwise_total": int(len(pair_rows)),
            "data/pair_mid_lower_enabled": int(1 if pair_mid_lower_enabled else 0),
            "data/pair_mid_lower_added": int(pair_mid_lower_added),
            "data/pair_mid_pos_score_min": float(pair_mid_pos_score_min),
            "data/pair_mid_pos_score_max": float(pair_mid_pos_score_max),
            "data/pair_mid_neg_score_min": float(pair_mid_neg_score_min),
            "data/pair_mid_neg_score_max": float(pair_mid_neg_score_max),
            "data/pair_mid_margin_min": float(pair_mid_margin_min),
            "data/pair_mid_margin_max": float(pair_mid_margin_max),
            "data/pair_mid_add_easy_contrast": int(1 if pair_mid_add_easy_contrast else 0),
            "data/listwise_train": int(len(train_list_rows)),
            "data/listwise_val": int(len(val_list_rows)),
            "data/listwise_test": int(len(test_list_rows)),
        },
        step=0,
    )

    device: torch.device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    use_amp = device.type == "cuda" and (bool(args.fp16) or bool(args.bf16))
    amp_dtype = torch.bfloat16 if bool(args.bf16) else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(_clean_text(args.model_id) or MODEL_ID_DEFAULT)
    model = AutoModelForSequenceClassification.from_pretrained(_clean_text(args.model_id) or MODEL_ID_DEFAULT, num_labels=1)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    pair_train_loader = DataLoader(
        PairwiseDataset(train_pairs),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=_safe_int(args.num_workers, default=0, minimum=0, maximum=32),
        collate_fn=PairCollator(tokenizer, max_length=max_length),
        drop_last=False,
    )
    pair_val_loader = DataLoader(
        PairwiseDataset(val_pairs),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=_safe_int(args.num_workers, default=0, minimum=0, maximum=32),
        collate_fn=PairCollator(tokenizer, max_length=max_length),
        drop_last=False,
    )
    pair_test_loader = DataLoader(
        PairwiseDataset(test_pairs),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=_safe_int(args.num_workers, default=0, minimum=0, maximum=32),
        collate_fn=PairCollator(tokenizer, max_length=max_length),
        drop_last=False,
    )
    val_list_loader: Optional[DataLoader] = None
    if val_list_rows:
        val_list_loader = DataLoader(
            ListwiseDataset(val_groups, val_list_rows),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=_safe_int(args.num_workers, default=0, minimum=0, maximum=32),
            collate_fn=ListCollator(tokenizer, max_length=max_length),
            drop_last=False,
        )
    test_list_loader: Optional[DataLoader] = None
    if test_list_rows:
        test_list_loader = DataLoader(
            ListwiseDataset(test_groups, test_list_rows),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=_safe_int(args.num_workers, default=0, minimum=0, maximum=32),
            collate_fn=ListCollator(tokenizer, max_length=max_length),
            drop_last=False,
        )

    best_ndcg = -1.0
    history: List[Dict[str, Any]] = []
    global_step = 0
    start_time = time.time()

    def eval_pairwise_margin(loader: DataLoader) -> float:
        was_training = bool(model.training)
        model.eval()
        vals: List[float] = []
        with torch.no_grad():
            for batch in loader:
                pos = _to_device(batch["pos"], device)
                neg = _to_device(batch["neg"], device)
                margins = batch["margins"].to(device)
                margins = margins.clamp(min=margin_min, max=margin_max)

                pos_logits = model(**pos).logits.squeeze(-1)
                neg_logits = model(**neg).logits.squeeze(-1)
                loss = _variable_margin_loss(pos_logits, neg_logits, margins)
                vals.append(float(loss.detach().cpu().item()))
        if was_training:
            model.train()
        return float(sum(vals) / max(1, len(vals)))

    def eval_listwise_losses(loader: Optional[DataLoader]) -> Dict[str, float]:
        if loader is None:
            return {"val_kl_loss": 0.0, "val_mse_loss": 0.0, "val_list_batches": 0.0}
        was_training = bool(model.training)
        model.eval()
        kl_vals: List[float] = []
        mse_vals: List[float] = []
        with torch.no_grad():
            for batch in loader:
                if batch.get("enc") is None:
                    continue
                enc = _to_device(batch["enc"], device)
                teacher_scores = batch["teacher_scores"].to(device)
                list_sizes = batch["list_sizes"]
                logits_flat = model(**enc).logits.squeeze(-1)
                kl_loss, mse_loss = _compute_listwise_kl_and_mse(
                    logits_flat=logits_flat,
                    teacher_scores_flat=teacher_scores,
                    list_sizes=list_sizes,
                    temperature=teacher_temperature,
                )
                kl_vals.append(float(kl_loss.detach().cpu().item()))
                mse_vals.append(float(mse_loss.detach().cpu().item()))
        if was_training:
            model.train()
        return {
            "val_kl_loss": float(sum(kl_vals) / max(1, len(kl_vals))),
            "val_mse_loss": float(sum(mse_vals) / max(1, len(mse_vals))),
            "val_list_batches": float(len(kl_vals)),
        }

    def run_step_validation(*, stage: int, epoch: int, global_step_now: int) -> None:
        if int(eval_every_steps) <= 0:
            return
        if int(global_step_now) <= 0:
            return
        if int(global_step_now) % int(eval_every_steps) != 0:
            return

        val_pair_loss = eval_pairwise_margin(pair_val_loader) if val_pairs else 0.0
        if int(stage) == 1:
            print(
                f"stage=1 epoch={epoch}/{stage1_epochs} step={global_step_now} "
                f"val_pair_loss={val_pair_loss:.6f}"
            )
            _wandb_log(
                wandb_run,
                {
                    "stage": 1,
                    "eval/loss_pair_step": float(val_pair_loss),
                },
                step=int(global_step_now),
            )
            return

        val_list_loss = eval_listwise_losses(val_list_loader)
        val_kl_loss = float(val_list_loss.get("val_kl_loss", 0.0))
        val_mse_loss = float(val_list_loss.get("val_mse_loss", 0.0))
        val_total_loss = (
            float(loss_kl_weight) * float(val_kl_loss)
            + float(loss_pair_weight) * float(val_pair_loss)
            + float(loss_mse_weight) * float(val_mse_loss)
        )
        print(
            f"stage=2 epoch={epoch}/{stage2_epochs} step={global_step_now} "
            f"val_total_loss={val_total_loss:.6f} val_kl_loss={val_kl_loss:.6f} "
            f"val_pair_loss={val_pair_loss:.6f} val_mse_loss={val_mse_loss:.6f}"
        )
        _wandb_log(
            wandb_run,
            {
                "stage": 2,
                "eval/loss_total_step": float(val_total_loss),
                "eval/loss_kl_step": float(val_kl_loss),
                "eval/loss_pair_step": float(val_pair_loss),
                "eval/loss_mse_step": float(val_mse_loss),
            },
            step=int(global_step_now),
        )

    # Stage 1: pairwise warm start.
    if stage1_epochs > 0:
        print(f"stage1_start=true epochs={stage1_epochs}")
        for epoch in range(1, stage1_epochs + 1):
            model.train()
            epoch_losses: List[float] = []
            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0

            stage1_iter: Any = pair_train_loader
            stage1_bar: Any = None
            if use_tqdm:
                stage1_bar = tqdm(
                    pair_train_loader,
                    total=len(pair_train_loader),
                    desc=f"Stage1 {epoch}/{stage1_epochs}",
                    dynamic_ncols=True,
                    leave=False,
                )
                stage1_iter = stage1_bar

            for step, batch in enumerate(stage1_iter, start=1):
                pos = _to_device(batch["pos"], device)
                neg = _to_device(batch["neg"], device)
                margins = batch["margins"].to(device)
                margins = margins.clamp(min=margin_min, max=margin_max)

                amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
                with amp_ctx:
                    pos_logits = model(**pos).logits.squeeze(-1)
                    neg_logits = model(**neg).logits.squeeze(-1)
                    loss = _variable_margin_loss(pos_logits, neg_logits, margins)
                    loss = loss / float(grad_accum_steps)

                loss.backward()
                epoch_losses.append(float(loss.detach().cpu().item() * float(grad_accum_steps)))
                accum_counter += 1

                if accum_counter >= grad_accum_steps:
                    if max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    accum_counter = 0

                    if global_step % log_every_steps == 0:
                        recent_pair = sum(epoch_losses[-10:]) / max(1, min(10, len(epoch_losses)))
                        print(
                            f"stage=1 epoch={epoch}/{stage1_epochs} step={global_step} "
                            f"loss_pair={recent_pair:.6f}"
                        )
                        _wandb_log(
                            wandb_run,
                            {
                                "stage": 1,
                                "train/loss_pair_step": float(recent_pair),
                            },
                            step=global_step,
                        )
                    run_step_validation(stage=1, epoch=epoch, global_step_now=global_step)
                if stage1_bar is not None and (step % 10 == 0):
                    recent_pair = sum(epoch_losses[-10:]) / max(1, min(10, len(epoch_losses)))
                    stage1_bar.set_postfix(
                        {
                            "loss_pair": f"{recent_pair:.4f}",
                            "gs": int(global_step),
                        }
                    )

            if accum_counter > 0:
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                run_step_validation(stage=1, epoch=epoch, global_step_now=global_step)
            if stage1_bar is not None:
                stage1_bar.close()

            train_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
            val_pair_loss = eval_pairwise_margin(pair_val_loader) if val_pairs else 0.0

            metrics = {
                "stage": 1,
                "epoch": epoch,
                "train_pair_loss": train_loss,
                "val_pair_loss": val_pair_loss,
                "global_step": global_step,
            }
            history.append(metrics)
            print(json.dumps(metrics, ensure_ascii=False))
            _wandb_log(
                wandb_run,
                {
                    "stage": 1,
                    "epoch": int(epoch),
                    "train/loss_pair_epoch": float(train_loss),
                    "val/loss_pair_epoch": float(val_pair_loss),
                },
                step=global_step if global_step > 0 else None,
            )

            stage1_dir = output_dir / f"stage1_epoch_{epoch}"
            stage1_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(stage1_dir)
            tokenizer.save_pretrained(stage1_dir)

    # Stage 2: listwise KD + pairwise + MSE.
    if stage2_epochs > 0:
        print(f"stage2_start=true epochs={stage2_epochs}")

        pair_cycle = _make_pair_iterator(pair_train_loader)
        if pair_cycle is None:
            raise RuntimeError("Unable to initialize pairwise iterator for stage 2.")

        for epoch in range(1, stage2_epochs + 1):
            # Re-sample list rows each epoch for more variety.
            epoch_list_rows = _build_sampled_list_rows(
                train_groups,
                candidate_pool_size=candidate_pool_size,
                mini_list_size=mini_list_size,
                boundary_center=_safe_float(args.boundary_center, default=0.6, minimum=0.0, maximum=1.0),
                boundary_bandwidth=_safe_float(args.boundary_bandwidth, default=0.12, minimum=1e-6, maximum=1.0),
                top_ratio=_safe_float(args.top_ratio, default=0.4, minimum=0.0, maximum=1.0),
                boundary_ratio=_safe_float(args.boundary_ratio, default=0.4, minimum=0.0, maximum=1.0),
                random_ratio=_safe_float(args.random_ratio, default=0.2, minimum=0.0, maximum=1.0),
                low_signal_max_score_threshold=low_signal_max_score_threshold,
                low_signal_std_threshold=low_signal_std_threshold,
                low_signal_keep_prob=low_signal_keep_prob,
                seed=seed + epoch,
            )
            if not epoch_list_rows:
                raise RuntimeError("No list rows available in stage 2 epoch resampling.")

            epoch_loader = DataLoader(
                ListwiseDataset(train_groups, epoch_list_rows),
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=_safe_int(args.num_workers, default=0, minimum=0, maximum=32),
                collate_fn=ListCollator(tokenizer, max_length=max_length),
                drop_last=False,
            )

            model.train()
            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0

            loss_hist: Dict[str, List[float]] = {"total": [], "kl": [], "pair": [], "mse": []}

            stage2_iter: Any = epoch_loader
            stage2_bar: Any = None
            if use_tqdm:
                stage2_bar = tqdm(
                    epoch_loader,
                    total=len(epoch_loader),
                    desc=f"Stage2 {epoch}/{stage2_epochs}",
                    dynamic_ncols=True,
                    leave=False,
                )
                stage2_iter = stage2_bar

            for step, list_batch in enumerate(stage2_iter, start=1):
                if list_batch.get("enc") is None:
                    continue

                list_enc = _to_device(list_batch["enc"], device)
                teacher_scores = list_batch["teacher_scores"].to(device)
                list_sizes = list_batch["list_sizes"]

                pair_batch = next(pair_cycle)
                pos = _to_device(pair_batch["pos"], device)
                neg = _to_device(pair_batch["neg"], device)
                margins = pair_batch["margins"].to(device).clamp(min=margin_min, max=margin_max)

                amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
                with amp_ctx:
                    list_logits_flat = model(**list_enc).logits.squeeze(-1)
                    kl_loss, mse_loss = _compute_listwise_kl_and_mse(
                        logits_flat=list_logits_flat,
                        teacher_scores_flat=teacher_scores,
                        list_sizes=list_sizes,
                        temperature=teacher_temperature,
                    )

                    pos_logits = model(**pos).logits.squeeze(-1)
                    neg_logits = model(**neg).logits.squeeze(-1)
                    pair_loss = _variable_margin_loss(pos_logits, neg_logits, margins)

                    total_loss = (
                        float(loss_kl_weight) * kl_loss
                        + float(loss_pair_weight) * pair_loss
                        + float(loss_mse_weight) * mse_loss
                    )
                    total_loss = total_loss / float(grad_accum_steps)

                total_loss.backward()

                loss_hist["total"].append(float(total_loss.detach().cpu().item() * float(grad_accum_steps)))
                loss_hist["kl"].append(float(kl_loss.detach().cpu().item()))
                loss_hist["pair"].append(float(pair_loss.detach().cpu().item()))
                loss_hist["mse"].append(float(mse_loss.detach().cpu().item()))
                accum_counter += 1

                if accum_counter >= grad_accum_steps:
                    if max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    accum_counter = 0

                    if global_step % log_every_steps == 0:
                        recent_total = sum(loss_hist["total"][-10:]) / max(1, min(10, len(loss_hist["total"])))
                        recent_kl = sum(loss_hist["kl"][-10:]) / max(1, min(10, len(loss_hist["kl"])))
                        recent_pair = sum(loss_hist["pair"][-10:]) / max(1, min(10, len(loss_hist["pair"])))
                        recent_mse = sum(loss_hist["mse"][-10:]) / max(1, min(10, len(loss_hist["mse"])))
                        print(
                            f"stage=2 epoch={epoch}/{stage2_epochs} step={global_step} "
                            f"loss_total={recent_total:.6f} "
                            f"loss_kl={recent_kl:.6f} "
                            f"loss_pair={recent_pair:.6f} "
                            f"loss_mse={recent_mse:.6f}"
                        )
                        _wandb_log(
                            wandb_run,
                            {
                                "stage": 2,
                                "train/loss_total_step": float(recent_total),
                                "train/loss_kl_step": float(recent_kl),
                                "train/loss_pair_step": float(recent_pair),
                                "train/loss_mse_step": float(recent_mse),
                            },
                            step=global_step,
                        )
                    run_step_validation(stage=2, epoch=epoch, global_step_now=global_step)
                if stage2_bar is not None and (step % 10 == 0):
                    recent_total = sum(loss_hist["total"][-10:]) / max(1, min(10, len(loss_hist["total"])))
                    recent_kl = sum(loss_hist["kl"][-10:]) / max(1, min(10, len(loss_hist["kl"])))
                    recent_pair = sum(loss_hist["pair"][-10:]) / max(1, min(10, len(loss_hist["pair"])))
                    recent_mse = sum(loss_hist["mse"][-10:]) / max(1, min(10, len(loss_hist["mse"])))
                    stage2_bar.set_postfix(
                        {
                            "loss": f"{recent_total:.4f}",
                            "kl": f"{recent_kl:.4f}",
                            "pair": f"{recent_pair:.4f}",
                            "mse": f"{recent_mse:.4f}",
                            "gs": int(global_step),
                        }
                    )

            if accum_counter > 0:
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                run_step_validation(stage=2, epoch=epoch, global_step_now=global_step)
            if stage2_bar is not None:
                stage2_bar.close()

            val_pair_loss = eval_pairwise_margin(pair_val_loader) if val_pairs else 0.0
            val_list_loss = eval_listwise_losses(val_list_loader)
            val_kl_loss = float(val_list_loss.get("val_kl_loss", 0.0))
            val_mse_loss = float(val_list_loss.get("val_mse_loss", 0.0))
            val_total_loss = (
                float(loss_kl_weight) * float(val_kl_loss)
                + float(loss_pair_weight) * float(val_pair_loss)
                + float(loss_mse_weight) * float(val_mse_loss)
            )

            eval_metrics = _evaluate(
                model=model,
                tokenizer=tokenizer,
                groups=val_groups,
                device=device,
                max_length=max_length,
                eval_batch_size=eval_batch_size,
                candidate_pool_size=candidate_pool_size,
                mrr_rel_threshold=_safe_float(args.mrr_rel_threshold, default=0.7, minimum=0.0, maximum=1.0),
                recall_rel_threshold=_safe_float(args.recall_rel_threshold, default=0.7, minimum=0.0, maximum=1.0),
            )

            metrics = {
                "stage": 2,
                "epoch": epoch,
                "train_total_loss": float(sum(loss_hist["total"]) / max(1, len(loss_hist["total"]))),
                "train_kl_loss": float(sum(loss_hist["kl"]) / max(1, len(loss_hist["kl"]))),
                "train_pair_loss": float(sum(loss_hist["pair"]) / max(1, len(loss_hist["pair"]))),
                "train_mse_loss": float(sum(loss_hist["mse"]) / max(1, len(loss_hist["mse"]))),
                "val_total_loss": float(val_total_loss),
                "val_kl_loss": float(val_kl_loss),
                "val_pair_loss": float(val_pair_loss),
                "val_mse_loss": float(val_mse_loss),
                "val_list_batches": int(val_list_loss.get("val_list_batches", 0.0)),
                "global_step": global_step,
            }
            metrics.update(eval_metrics)
            history.append(metrics)
            print(json.dumps(metrics, ensure_ascii=False))
            _wandb_log(
                wandb_run,
                {
                    "stage": 2,
                    "epoch": int(epoch),
                    "train/loss_total_epoch": float(metrics["train_total_loss"]),
                    "train/loss_kl_epoch": float(metrics["train_kl_loss"]),
                    "train/loss_pair_epoch": float(metrics["train_pair_loss"]),
                    "train/loss_mse_epoch": float(metrics["train_mse_loss"]),
                    "eval/loss_total_epoch": float(metrics["val_total_loss"]),
                    "eval/loss_kl_epoch": float(metrics["val_kl_loss"]),
                    "eval/loss_pair_epoch": float(metrics["val_pair_loss"]),
                    "eval/loss_mse_epoch": float(metrics["val_mse_loss"]),
                    "eval/ndcg@10": float(metrics.get("ndcg@10", 0.0)),
                    "eval/mrr@10": float(metrics.get("mrr@10", 0.0)),
                    "eval/recall@50": float(metrics.get("recall@50", 0.0)),
                    "eval/queries": int(metrics.get("eval_queries", 0)),
                },
                step=global_step if global_step > 0 else None,
            )

            epoch_dir = output_dir / f"stage2_epoch_{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)

            if float(eval_metrics.get("ndcg@10", 0.0)) > best_ndcg:
                best_ndcg = float(eval_metrics.get("ndcg@10", 0.0))
                best_dir = output_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)

    test_pair_loss = eval_pairwise_margin(pair_test_loader) if test_pairs else 0.0
    test_list_loss = eval_listwise_losses(test_list_loader)
    test_kl_loss = float(test_list_loss.get("val_kl_loss", 0.0))
    test_mse_loss = float(test_list_loss.get("val_mse_loss", 0.0))
    test_total_loss = (
        float(loss_kl_weight) * float(test_kl_loss)
        + float(loss_pair_weight) * float(test_pair_loss)
        + float(loss_mse_weight) * float(test_mse_loss)
    )
    test_eval_metrics = _evaluate(
        model=model,
        tokenizer=tokenizer,
        groups=test_groups,
        device=device,
        max_length=max_length,
        eval_batch_size=eval_batch_size,
        candidate_pool_size=candidate_pool_size,
        mrr_rel_threshold=_safe_float(args.mrr_rel_threshold, default=0.7, minimum=0.0, maximum=1.0),
        recall_rel_threshold=_safe_float(args.recall_rel_threshold, default=0.7, minimum=0.0, maximum=1.0),
    ) if test_groups else {"ndcg@10": 0.0, "mrr@10": 0.0, "recall@50": 0.0, "eval_queries": 0}
    test_metrics = {
        "stage": "test",
        "test_total_loss": float(test_total_loss),
        "test_kl_loss": float(test_kl_loss),
        "test_pair_loss": float(test_pair_loss),
        "test_mse_loss": float(test_mse_loss),
        "test_list_batches": int(test_list_loss.get("val_list_batches", 0.0)),
        "global_step": global_step,
    }
    test_metrics.update(test_eval_metrics)
    history.append(test_metrics)
    print(json.dumps(test_metrics, ensure_ascii=False))
    _wandb_log(
        wandb_run,
        {
            "test/loss_total": float(test_metrics["test_total_loss"]),
            "test/loss_kl": float(test_metrics["test_kl_loss"]),
            "test/loss_pair": float(test_metrics["test_pair_loss"]),
            "test/loss_mse": float(test_metrics["test_mse_loss"]),
            "test/ndcg@10": float(test_metrics.get("ndcg@10", 0.0)),
            "test/mrr@10": float(test_metrics.get("mrr@10", 0.0)),
            "test/recall@50": float(test_metrics.get("recall@50", 0.0)),
            "test/queries": int(test_metrics.get("eval_queries", 0)),
        },
        step=global_step if global_step > 0 else None,
    )

    elapsed = max(1e-6, time.time() - start_time)

    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_input": str(raw_path),
        "pairwise_input": str(pairwise_path),
        "output_dir_base": str(output_dir_base),
        "output_dir": str(output_dir),
        "append_args_to_output_dir": bool(append_args_to_output_dir),
        "output_suffix": str(output_suffix),
        "model_id": _clean_text(args.model_id) or MODEL_ID_DEFAULT,
        "seed": int(seed),
        "split_policy": str(split_policy),
        "use_prepared_splits": bool(use_prepared_splits),
        "regenerate_splits": bool(regenerate_splits),
        "split_generated": bool(split_generated),
        "split_dir": str(split_dir),
        "split_manifest": str(split_manifest_path) if split_manifest_path is not None else "",
        "split_raw_train_path": str(split_raw_train_path),
        "split_raw_val_path": str(split_raw_val_path),
        "split_raw_test_path": str(split_raw_test_path),
        "split_pair_train_path": str(split_pair_train_path),
        "split_pair_val_path": str(split_pair_val_path),
        "split_pair_test_path": str(split_pair_test_path),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "train_queries": int(len(train_groups)),
        "val_queries": int(len(val_groups)),
        "test_queries": int(len(test_groups)),
        "pair_rows_source": str(pair_rows_source),
        "pair_total_rows": int(len(pair_rows)),
        "train_pair_rows": int(len(train_pairs)),
        "val_pair_rows": int(len(val_pairs)),
        "test_pair_rows": int(len(test_pairs)),
        "train_list_rows": int(len(train_list_rows)),
        "val_list_rows": int(len(val_list_rows)),
        "test_list_rows": int(len(test_list_rows)),
        "pair_mid_lower_enabled": bool(pair_mid_lower_enabled),
        "pair_mid_lower_added": int(pair_mid_lower_added),
        "pair_mid_pos_score_min": float(pair_mid_pos_score_min),
        "pair_mid_pos_score_max": float(pair_mid_pos_score_max),
        "pair_mid_neg_score_min": float(pair_mid_neg_score_min),
        "pair_mid_neg_score_max": float(pair_mid_neg_score_max),
        "pair_mid_margin_min": float(pair_mid_margin_min),
        "pair_mid_margin_max": float(pair_mid_margin_max),
        "pair_mid_add_easy_contrast": bool(pair_mid_add_easy_contrast),
        "candidate_pool_size": int(candidate_pool_size),
        "mini_list_size": int(mini_list_size),
        "teacher_temperature": float(teacher_temperature),
        "loss_weights": {
            "kl": float(loss_kl_weight),
            "pair": float(loss_pair_weight),
            "mse": float(loss_mse_weight),
        },
        "margin_clip": {"min": float(margin_min), "max": float(margin_max)},
        "stage1_epochs": int(stage1_epochs),
        "stage2_epochs": int(stage2_epochs),
        "train_batch_size": int(train_batch_size),
        "eval_batch_size": int(eval_batch_size),
        "eval_every_steps": int(eval_every_steps),
        "grad_accum_steps": int(grad_accum_steps),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "max_length": int(max_length),
        "device": str(device),
        "use_tqdm": bool(use_tqdm),
        "use_amp": bool(use_amp),
        "amp_dtype": str(amp_dtype) if use_amp else "",
        "best_ndcg@10": float(best_ndcg),
        "test_metrics": test_metrics,
        "elapsed_seconds": float(elapsed),
        "history": history,
        "wandb": {
            "enabled": bool(wandb_run is not None),
            "project": _clean_text(args.wandb_project),
            "entity": _clean_text(args.wandb_entity),
            "run_name": _clean_text(args.wandb_run_name),
            "run_id": _clean_text(getattr(wandb_run, "id", "")) if wandb_run is not None else "",
            "run_url": _clean_text(getattr(wandb_run, "url", "")) if wandb_run is not None else "",
        },
    }

    manifest_path = output_dir / "train_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _wandb_log(
        wandb_run,
        {
            "run/best_ndcg@10": float(best_ndcg),
            "run/elapsed_seconds": float(elapsed),
        },
        step=global_step if global_step > 0 else None,
    )
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    print("done=true")
    print(f"best_ndcg@10={best_ndcg:.6f}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
