from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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


MODEL_ID_DEFAULT = "dleemiller/ModernCE-base-sts"
RAW_INPUT_DEFAULT = "cross_encoder/dataset/distill/llm_distill2_raw_scores.jsonl"
PAIRWISE_INPUT_DEFAULT = "cross_encoder/dataset/distill/llm_distill2_pairwise.jsonl"
SPLIT_DIR_DEFAULT = "cross_encoder/dataset/splits"
RAW_TRAIN_INPUT_DEFAULT = f"{SPLIT_DIR_DEFAULT}/llm_distill_raw_train.jsonl"
RAW_VAL_INPUT_DEFAULT = f"{SPLIT_DIR_DEFAULT}/llm_distill_raw_val.jsonl"
RAW_TEST_INPUT_DEFAULT = f"{SPLIT_DIR_DEFAULT}/llm_distill_raw_test.jsonl"
PAIRWISE_TRAIN_INPUT_DEFAULT = f"{SPLIT_DIR_DEFAULT}/llm_distill_pairwise_train.jsonl"
PAIRWISE_VAL_INPUT_DEFAULT = f"{SPLIT_DIR_DEFAULT}/llm_distill_pairwise_val.jsonl"
PAIRWISE_TEST_INPUT_DEFAULT = f"{SPLIT_DIR_DEFAULT}/llm_distill_pairwise_test.jsonl"
OUTPUT_DIR_DEFAULT = "cross_encoder/models/bge_reranker_distill"
WANDB_PROJECT_DEFAULT = "cross_encoder_distill"

TRAIN_BAND_HIGH_THRESHOLD_DEFAULT = 0.70
TRAIN_BAND_MID_THRESHOLD_DEFAULT = 0.30
LIST_AUGMENTED_DOC_WEIGHT_DEFAULT = 0.70
PAIR_TYPE_MAX_SHARE_DEFAULT = 1.0
PAIR_MID_MIN_CANDIDATES_DEFAULT = 6
PAIR_MID_START_RATIO_DEFAULT = 0.30
PAIR_MID_END_RATIO_DEFAULT = 0.55
PAIR_LOWER_MID_START_RATIO_DEFAULT = 0.55
PAIR_LOWER_MID_END_RATIO_DEFAULT = 0.80
PAIR_TYPE_WEIGHT_MAP_DEFAULT = (
    "default=1.0,"
    "llm_disagreement=1.15,"
    "strong_vs_boundary=1.05,"
    "strong_vs_weak=0.95,"
    "strong_vs_hard=1.0"
)


@dataclass
class Candidate:
    text: str
    score: float
    fac_id: int
    chunk_id: int
    chunk_index: int
    source_type: str
    target_cluster: str = "unknown"
    selected_for_target: bool = False
    is_augmented: bool = False
    is_disagreement: bool = False


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
        self.pair_type_weights = {
            _clean_text(k).lower(): float(v) for k, v in dict(pair_type_weights or {}).items() if _clean_text(k)
        }
        self.default_pair_weight = max(0.0, float(default_pair_weight))

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
        pair_weights = torch.tensor(
            [
                float(
                    self.pair_type_weights.get(
                        _clean_text(x.pair_type).lower(),
                        self.default_pair_weight,
                    )
                )
                for x in batch
            ],
            dtype=torch.float32,
        ).clamp(min=0.0)

        return {
            "pos": pos_enc,
            "neg": neg_enc,
            "margins": margins,
            "teacher_pos": pos_teacher,
            "teacher_neg": neg_teacher,
            "pair_weights": pair_weights,
        }


class ListCollator:
    def __init__(self, tokenizer: Any, max_length: int, *, augmented_doc_weight: float) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.augmented_doc_weight = _safe_float(
            augmented_doc_weight,
            default=LIST_AUGMENTED_DOC_WEIGHT_DEFAULT,
            minimum=0.0,
            maximum=1.0,
        )

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        queries_flat: List[str] = []
        docs_flat: List[str] = []
        teacher_scores: List[float] = []
        doc_weights: List[float] = []
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
                is_augmented = bool(d.get("is_augmented", False))
                doc_weights.append(float(self.augmented_doc_weight) if is_augmented else 1.0)

        if not list_sizes:
            return {
                "enc": None,
                "teacher_scores": None,
                "doc_weights": None,
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
            "doc_weights": torch.tensor(doc_weights, dtype=torch.float32),
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


def _resolve_split_helper() -> Any:
    """
    Resolve split helper from the current cross_encoder layout first,
    then fallback to legacy spec_to_spec location.
    """
    try:
        from cross_encoder.data_preparation.split_distill_dataset import ensure_split_files

        return ensure_split_files
    except Exception:
        from cross_encoder.spec_to_spec.data_preparation.split_distill_dataset import ensure_split_files

        return ensure_split_files


def _extract_raw_candidate_rows(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Backward-compatible candidate extractor:
    - legacy raw: candidates
    - llm_distill2 raw: ranked_docs
    - listwise-like rows: docs
    """
    for key in ("candidates", "ranked_docs", "docs"):
        rows = obj.get(key)
        if isinstance(rows, list):
            out: List[Dict[str, Any]] = []
            for row in rows:
                if isinstance(row, dict):
                    out.append(row)
            if out:
                return out
    return []


def _extract_candidate_score(row: Dict[str, Any]) -> float:
    for key in ("score", "teacher_score", "score_raw", "teacher_score_raw"):
        if key in row:
            return _clamp_01(row.get(key))
    return 0.0


def _parse_csv_items(value: Any) -> List[str]:
    raw = _clean_text(value)
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_pair_type_weight_map(value: Any) -> Tuple[Dict[str, float], float]:
    default_weight = 1.0
    out: Dict[str, float] = {}
    for token in _parse_csv_items(value):
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        key = _clean_text(k).lower()
        if not key:
            continue
        try:
            w = float(v)
        except Exception:
            continue
        w = max(0.0, min(10.0, w))
        if key == "default":
            default_weight = w
        else:
            out[key] = w
    return out, float(default_weight)


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
            cand_raw = _extract_raw_candidate_rows(obj)
            if not grant_id or not query_text or not cand_raw:
                continue

            cands: List[Candidate] = []
            for c in cand_raw:
                if not isinstance(c, dict):
                    continue
                doc_text = _clean_text(c.get("chunk_text") or c.get("fac_spec_text") or c.get("text"))
                if not doc_text:
                    continue
                doc_id = _safe_int(
                    c.get("chunk_id") if ("chunk_id" in c) else c.get("fac_spec_id"),
                    default=0,
                    minimum=0,
                    maximum=2_147_483_647,
                )
                doc_index = _safe_int(
                    c.get("chunk_index") if ("chunk_index" in c) else c.get("fac_spec_idx"),
                    default=0,
                    minimum=0,
                    maximum=10_000_000,
                )
                cands.append(
                    Candidate(
                        text=doc_text,
                        score=_extract_candidate_score(c),
                        fac_id=_safe_int(c.get("fac_id"), default=0, minimum=0, maximum=2_147_483_647),
                        chunk_id=doc_id,
                        chunk_index=doc_index,
                        source_type=_clean_text(c.get("source_type") or c.get("section")) or "unknown",
                        target_cluster=_clean_text(c.get("target_cluster")) or "unknown",
                        selected_for_target=bool(c.get("selected_for_target", False)),
                        is_augmented=bool(c.get("is_augmented", False)),
                        is_disagreement=bool(c.get("is_disagreement", False)),
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
    mid_min_candidates: int,
    mid_start_ratio: float,
    mid_end_ratio: float,
    lower_mid_start_ratio: float,
    lower_mid_end_ratio: float,
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
    min_candidates = max(2, int(mid_min_candidates))

    for g in groups:
        cands = list(g.candidates or [])
        n = len(cands)
        if n < min_candidates:
            continue

        # Candidate list is already sorted high->low by teacher score.
        mid_start = max(1, int(float(mid_start_ratio) * n))
        mid_end = min(n, max(mid_start + 1, int(float(mid_end_ratio) * n)))
        lower_start = min(n, max(mid_end, int(float(lower_mid_start_ratio) * n)))
        lower_end = min(n, max(lower_start + 1, int(float(lower_mid_end_ratio) * n)))

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
    if int(candidate_pool_size) <= 0:
        pool = list(group.candidates)
    else:
        pool = list(group.candidates[: min(len(group.candidates), max(1, int(candidate_pool_size)))])
    if not pool:
        return []

    if int(mini_list_size) <= 0:
        k = int(len(pool))
    else:
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
            "target_cluster": _clean_text(d.target_cluster) or "unknown",
            "selected_for_target": bool(d.selected_for_target),
            "is_augmented": bool(d.is_augmented),
            "is_disagreement": bool(d.is_disagreement),
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
        if int(candidate_pool_size) <= 0:
            pool = list(g.candidates)
        else:
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


def _variable_margin_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    margins: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # max(0, margin - (pos-neg)) with per-example clipped margin target.
    raw = F.relu(margins - (pos_logits - neg_logits))
    if weights is None:
        return raw.mean()
    w = weights.clamp(min=0.0)
    w_sum = torch.clamp(w.sum(), min=1e-6)
    return (raw * w).sum() / w_sum


def _compute_listwise_kl_and_mse(
    *,
    logits_flat: torch.Tensor,
    teacher_scores_flat: torch.Tensor,
    doc_weights_flat: Optional[torch.Tensor],
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
        if doc_weights_flat is not None:
            w = doc_weights_flat[cursor : cursor + n].clamp(min=0.0)
        else:
            w = torch.ones_like(y)
        cursor += n

        teacher_probs = torch.softmax(y / t, dim=-1)
        student_log_probs = torch.log_softmax(s / t, dim=-1)
        w_sum = torch.clamp(w.sum(), min=1e-6)
        kl_per_doc = teacher_probs * (torch.log(torch.clamp(teacher_probs, min=1e-12)) - student_log_probs)
        kl = (kl_per_doc * w).sum() / w_sum
        kl = kl * (t * t)
        mse = ((torch.sigmoid(s) - y).pow(2) * w).sum() / w_sum
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


def _run_command(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    cmd_list = [str(x) for x in cmd]
    cmd_print = " ".join(cmd_list)
    print(f"subprocess_start={cmd_print}")
    proc = subprocess.run(cmd_list, cwd=str(cwd or PROJECT_ROOT), check=False)
    if int(proc.returncode) != 0:
        raise RuntimeError(f"Subprocess failed (code={proc.returncode}): {cmd_print}")


def _resolve_round_model_dir(*, round_output_dir: Path, stage1_epochs: int, stage2_epochs: int) -> Path:
    best_dir = round_output_dir / "best"
    if best_dir.exists():
        return best_dir

    if int(stage2_epochs) > 0:
        stage2_final = round_output_dir / f"stage2_epoch_{int(stage2_epochs)}"
        if stage2_final.exists():
            return stage2_final
        stage2_candidates = sorted(
            [p for p in round_output_dir.glob("stage2_epoch_*") if p.is_dir()],
            key=lambda x: float(x.stat().st_mtime),
            reverse=True,
        )
        if stage2_candidates:
            return stage2_candidates[0]

    if int(stage1_epochs) > 0:
        stage1_final = round_output_dir / f"stage1_epoch_{int(stage1_epochs)}"
        if stage1_final.exists():
            return stage1_final
        stage1_candidates = sorted(
            [p for p in round_output_dir.glob("stage1_epoch_*") if p.is_dir()],
            key=lambda x: float(x.stat().st_mtime),
            reverse=True,
        )
        if stage1_candidates:
            return stage1_candidates[0]

    raise RuntimeError(
        f"Unable to find trained model checkpoint directory under {round_output_dir}. "
        "Expected best/stage2_epoch_*/stage1_epoch_*."
    )


def _find_latest_mismatch_json(*, eval_output_dir: Path, mismatch_target: str) -> Path:
    pref = sorted(
        [p for p in eval_output_dir.glob(f"distill_cluster_mismatch_{mismatch_target}_*.json") if p.is_file()],
        key=lambda x: float(x.stat().st_mtime),
        reverse=True,
    )
    if pref:
        return pref[0]
    any_json = sorted(
        [p for p in eval_output_dir.glob("distill_cluster_mismatch_*.json") if p.is_file()],
        key=lambda x: float(x.stat().st_mtime),
        reverse=True,
    )
    if any_json:
        return any_json[0]
    raise RuntimeError(f"No mismatch eval JSON found in {eval_output_dir}")


def _load_mismatch_rows(eval_json_path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(eval_json_path.read_text(encoding="utf-8"))
    rows = obj.get("rows") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(row)
    return out


def _build_mismatch_distill_files(
    *,
    rows: Sequence[Dict[str, Any]],
    raw_out_path: Path,
    pair_out_path: Path,
    pair_derive_pos_k: int,
    pair_derive_hard_k: int,
    pair_derive_weak_k: int,
    pair_derive_cap: int,
    pair_add_mid_lower_mid: bool,
    pair_mid_pos_score_min: float,
    pair_mid_pos_score_max: float,
    pair_mid_neg_score_min: float,
    pair_mid_neg_score_max: float,
    pair_mid_margin_min: float,
    pair_mid_margin_max: float,
    pair_mid_add_easy_contrast: bool,
    pair_mid_min_candidates: int,
    pair_mid_start_ratio: float,
    pair_mid_end_ratio: float,
    pair_lower_mid_start_ratio: float,
    pair_lower_mid_end_ratio: float,
) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, int, str], QueryGroup] = {}
    seen_docs: Dict[Tuple[str, int, str], Set[str]] = {}
    dropped_invalid = 0

    for row in rows:
        grant_id = _clean_text(row.get("grant_id"))
        spec_idx = _safe_int(row.get("spec_idx"), default=0, minimum=0, maximum=50_000_000)
        query_text = _clean_text(row.get("query_text") or row.get("spec_text"))
        doc_text = _clean_text(row.get("doc_text") or row.get("fac_spec_text") or row.get("text"))
        if (not grant_id) or (not query_text) or (not doc_text):
            dropped_invalid += 1
            continue

        teacher_score = _clamp_01(
            row.get("teacher_score_used")
            if (row.get("teacher_score_used") is not None)
            else (row.get("teacher_score") if (row.get("teacher_score") is not None) else row.get("score"))
        )
        fac_id = _safe_int(row.get("fac_id"), default=0, minimum=0, maximum=2_147_483_647)
        chunk_id = _safe_int(
            row.get("fac_spec_id") if (row.get("fac_spec_id") is not None) else row.get("chunk_id"),
            default=0,
            minimum=0,
            maximum=2_147_483_647,
        )
        chunk_index = _safe_int(row.get("chunk_index"), default=0, minimum=0, maximum=10_000_000)
        source_type = _clean_text(row.get("source_band") or row.get("score_band") or row.get("source_type")) or "mismatch"

        key = (grant_id, int(spec_idx), query_text)
        if key not in grouped:
            grouped[key] = QueryGroup(grant_id=grant_id, spec_idx=int(spec_idx), query_text=query_text, candidates=[])
            seen_docs[key] = set()

        dedup_key = f"{fac_id}::{chunk_id}::{doc_text}"
        if dedup_key in seen_docs[key]:
            continue
        seen_docs[key].add(dedup_key)

        grouped[key].candidates.append(
            Candidate(
                text=doc_text,
                score=float(teacher_score),
                fac_id=int(fac_id),
                chunk_id=int(chunk_id),
                chunk_index=int(chunk_index),
                source_type=source_type,
            )
        )

    raw_rows: List[Dict[str, Any]] = []
    usable_groups: List[QueryGroup] = []
    dropped_singleton = 0
    for g in grouped.values():
        cands = sorted(list(g.candidates), key=lambda x: float(x.score), reverse=True)
        if len(cands) < 2:
            dropped_singleton += 1
            continue
        group = QueryGroup(grant_id=g.grant_id, spec_idx=int(g.spec_idx), query_text=g.query_text, candidates=cands)
        usable_groups.append(group)
        raw_rows.append(
            {
                "grant_id": group.grant_id,
                "spec_idx": int(group.spec_idx),
                "query_text": group.query_text,
                "spec_text": group.query_text,
                "candidates": [
                    {
                        "fac_id": int(c.fac_id),
                        "fac_spec_id": int(c.chunk_id),
                        "chunk_id": int(c.chunk_id),
                        "chunk_index": int(c.chunk_index),
                        "source_type": c.source_type,
                        "section": c.source_type,
                        "fac_spec_text": c.text,
                        "chunk_text": c.text,
                        "text": c.text,
                        "score": float(c.score),
                        "score_raw": float(c.score),
                        "teacher_score": float(c.score),
                        "teacher_score_raw": float(c.score),
                        "target_cluster": _clean_text(c.target_cluster) or "unknown",
                        "selected_for_target": bool(c.selected_for_target),
                        "is_augmented": bool(c.is_augmented),
                        "is_disagreement": bool(c.is_disagreement),
                    }
                    for c in cands
                ],
            }
        )

    raw_out_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_out_path.open("w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pair_rows = _derive_pairwise_from_groups(
        usable_groups,
        per_query_pos_k=int(pair_derive_pos_k),
        per_query_hard_k=int(pair_derive_hard_k),
        per_query_weak_k=int(pair_derive_weak_k),
        per_query_cap=int(pair_derive_cap),
    )
    if bool(pair_add_mid_lower_mid):
        pair_rows = list(pair_rows) + _derive_mid_lower_mid_pairs_from_groups(
            usable_groups,
            per_query_cap=int(pair_derive_cap),
            mid_min_candidates=int(pair_mid_min_candidates),
            mid_start_ratio=float(pair_mid_start_ratio),
            mid_end_ratio=float(pair_mid_end_ratio),
            lower_mid_start_ratio=float(pair_lower_mid_start_ratio),
            lower_mid_end_ratio=float(pair_lower_mid_end_ratio),
            pos_score_min=float(pair_mid_pos_score_min),
            pos_score_max=float(pair_mid_pos_score_max),
            neg_score_min=float(pair_mid_neg_score_min),
            neg_score_max=float(pair_mid_neg_score_max),
            margin_min=float(pair_mid_margin_min),
            margin_max=float(pair_mid_margin_max),
            add_easy_contrast=bool(pair_mid_add_easy_contrast),
        )

    pair_out_path.parent.mkdir(parents=True, exist_ok=True)
    with pair_out_path.open("w", encoding="utf-8") as f:
        for p in pair_rows:
            f.write(
                json.dumps(
                    {
                        "grant_id": p.grant_id,
                        "spec_idx": int(p.spec_idx),
                        "query_text": p.query_text,
                        "pos_text": p.pos_text,
                        "neg_text": p.neg_text,
                        "teacher_pos_score": float(p.teacher_pos_score),
                        "teacher_neg_score": float(p.teacher_neg_score),
                        "teacher_margin": float(p.teacher_margin),
                        "pair_type": p.pair_type,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {
        "input_rows": int(len(rows)),
        "dropped_invalid_rows": int(dropped_invalid),
        "raw_query_groups_total": int(len(grouped)),
        "raw_query_groups_kept": int(len(usable_groups)),
        "raw_query_groups_dropped_singleton": int(dropped_singleton),
        "raw_rows_written": int(len(raw_rows)),
        "pair_rows_written": int(len(pair_rows)),
        "raw_out_path": str(raw_out_path),
        "pair_out_path": str(pair_out_path),
    }


def _score_band_from_value(score: float, *, high_threshold: float, mid_threshold: float) -> str:
    s = _clamp_01(score)
    if s >= float(high_threshold):
        return "high"
    if s >= float(mid_threshold):
        return "mid"
    return "low"


def _query_group_key(group: QueryGroup) -> str:
    return f"{_clean_text(group.grant_id)}::spec#{int(group.spec_idx)}"


def _group_band(group: QueryGroup, *, high_threshold: float, mid_threshold: float) -> str:
    if not group.candidates:
        return "low"
    top_score = float(max(float(c.score) for c in group.candidates))
    return _score_band_from_value(top_score, high_threshold=high_threshold, mid_threshold=mid_threshold)


def _query_key_from_row(row: Dict[str, Any]) -> str:
    return f"{_clean_text(row.get('grant_id'))}::spec#{_safe_int(row.get('spec_idx'), default=0, minimum=0, maximum=50_000_000)}"


def _pair_key(pair: PairExample) -> str:
    body = "||".join(
        [
            _clean_text(pair.grant_id),
            str(int(pair.spec_idx)),
            _clean_text(pair.query_text),
            _clean_text(pair.pos_text),
            _clean_text(pair.neg_text),
        ]
    )
    return hashlib.sha1(body.encode("utf-8")).hexdigest()


def _cap_pair_rows_by_type(
    rows: Sequence[PairExample],
    *,
    max_share: float,
    seed: int,
) -> Tuple[List[PairExample], Dict[str, Any]]:
    items = list(rows)
    total = int(len(items))
    share = max(0.0, min(1.0, float(max_share)))
    before_by_type: Dict[str, int] = {}
    for p in items:
        t = _clean_text(p.pair_type).lower() or "unknown"
        before_by_type[t] = int(before_by_type.get(t, 0)) + 1

    if total <= 0 or share <= 0.0 or share >= 1.0:
        return items, {
            "applied": False,
            "reason": "disabled_or_trivial",
            "max_share": float(share),
            "before_total": int(total),
            "after_total": int(total),
            "before_by_type": before_by_type,
            "after_by_type": dict(before_by_type),
            "dropped": 0,
        }

    cap_per_type = max(1, int(math.floor(float(total) * share)))
    by_type_rows: Dict[str, List[Tuple[int, PairExample]]] = {}
    for idx, p in enumerate(items):
        t = _clean_text(p.pair_type).lower() or "unknown"
        by_type_rows.setdefault(t, []).append((int(idx), p))

    keep_indices: Set[int] = set()
    for t, bucket in by_type_rows.items():
        if len(bucket) <= cap_per_type:
            for idx, _ in bucket:
                keep_indices.add(int(idx))
            continue
        ordered = sorted(
            bucket,
            key=lambda item: (
                _hash_to_unit_interval(f"{int(seed)}::{t}::{_pair_key(item[1])}::{int(item[0])}"),
                _pair_key(item[1]),
                int(item[0]),
            ),
        )
        for idx, _ in ordered[:cap_per_type]:
            keep_indices.add(int(idx))

    kept: List[PairExample] = []
    after_by_type: Dict[str, int] = {}
    for idx, p in enumerate(items):
        if int(idx) not in keep_indices:
            continue
        kept.append(p)
        t = _clean_text(p.pair_type).lower() or "unknown"
        after_by_type[t] = int(after_by_type.get(t, 0)) + 1

    return kept, {
        "applied": True,
        "reason": "capped_by_type_share",
        "max_share": float(share),
        "cap_per_type": int(cap_per_type),
        "before_total": int(total),
        "after_total": int(len(kept)),
        "before_by_type": before_by_type,
        "after_by_type": after_by_type,
        "dropped": int(total - len(kept)),
    }


def _pair_query_key(pair: PairExample) -> str:
    return f"{_clean_text(pair.grant_id)}::spec#{int(pair.spec_idx)}"


def _filter_pairs_for_groups(
    pairs: Sequence[PairExample],
    groups: Sequence[QueryGroup],
) -> List[PairExample]:
    keep_keys = {_query_group_key(g) for g in groups}
    return [p for p in pairs if _pair_query_key(p) in keep_keys]


def _count_group_bands(
    groups: Sequence[QueryGroup],
    *,
    high_threshold: float,
    mid_threshold: float,
) -> Dict[str, int]:
    out = {"high": 0, "mid": 0, "low": 0, "total": 0}
    for g in groups:
        b = _group_band(g, high_threshold=high_threshold, mid_threshold=mid_threshold)
        out[b] = int(out.get(b, 0)) + 1
        out["total"] += 1
    return out


def _cap_low_band_groups(
    groups: Sequence[QueryGroup],
    *,
    high_threshold: float,
    mid_threshold: float,
    low_cap_ratio_to_high: float,
    seed: int,
) -> Tuple[List[QueryGroup], Dict[str, Any]]:
    ratio = max(0.0, float(low_cap_ratio_to_high))
    before_counts = _count_group_bands(
        groups,
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
    )
    if ratio <= 0.0:
        return list(groups), {
            "applied": False,
            "reason": "ratio_disabled",
            "ratio": float(ratio),
            "low_cap_target": -1,
            "before": before_counts,
            "after": before_counts,
        }

    highs: List[QueryGroup] = []
    mids: List[QueryGroup] = []
    lows: List[QueryGroup] = []
    for g in groups:
        b = _group_band(g, high_threshold=high_threshold, mid_threshold=mid_threshold)
        if b == "high":
            highs.append(g)
        elif b == "mid":
            mids.append(g)
        else:
            lows.append(g)

    low_cap_target = max(0, int(math.floor(float(len(highs)) * float(ratio))))
    if len(lows) <= low_cap_target:
        return list(groups), {
            "applied": False,
            "reason": "already_within_cap",
            "ratio": float(ratio),
            "low_cap_target": int(low_cap_target),
            "before": before_counts,
            "after": before_counts,
        }

    kept_lows = _sample_groups_diverse(
        lows,
        target=int(low_cap_target),
        seed=int(seed + 909),
    )
    keep_low_keys = {_query_group_key(g) for g in kept_lows}
    out_groups: List[QueryGroup] = []
    for g in groups:
        b = _group_band(g, high_threshold=high_threshold, mid_threshold=mid_threshold)
        if b != "low":
            out_groups.append(g)
            continue
        if _query_group_key(g) in keep_low_keys:
            out_groups.append(g)

    after_counts = _count_group_bands(
        out_groups,
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
    )
    return out_groups, {
        "applied": True,
        "reason": "capped_low_to_high_ratio",
        "ratio": float(ratio),
        "low_cap_target": int(low_cap_target),
        "before": before_counts,
        "after": after_counts,
    }


def _merge_groups_unique(*parts: Sequence[QueryGroup]) -> List[QueryGroup]:
    seen: Set[str] = set()
    out: List[QueryGroup] = []
    for chunk in parts:
        for g in chunk:
            k = _query_group_key(g)
            if k in seen:
                continue
            seen.add(k)
            out.append(g)
    return out


def _merge_pairs_unique(*parts: Sequence[PairExample]) -> List[PairExample]:
    seen: Set[str] = set()
    out: List[PairExample] = []
    for chunk in parts:
        for p in chunk:
            k = _pair_key(p)
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
    return out


def _query_group_to_raw_obj(group: QueryGroup) -> Dict[str, Any]:
    cands = sorted(list(group.candidates), key=lambda x: float(x.score), reverse=True)
    return {
        "grant_id": _clean_text(group.grant_id),
        "spec_idx": int(group.spec_idx),
        "query_text": _clean_text(group.query_text),
        "spec_text": _clean_text(group.query_text),
        "candidates": [
            {
                "fac_id": int(c.fac_id),
                "fac_spec_id": int(c.chunk_id),
                "chunk_id": int(c.chunk_id),
                "chunk_index": int(c.chunk_index),
                "source_type": _clean_text(c.source_type) or "unknown",
                "section": _clean_text(c.source_type) or "unknown",
                "fac_spec_text": _clean_text(c.text),
                "chunk_text": _clean_text(c.text),
                "text": _clean_text(c.text),
                "score": float(c.score),
                "score_raw": float(c.score),
                "teacher_score": float(c.score),
                "teacher_score_raw": float(c.score),
                "target_cluster": _clean_text(c.target_cluster) or "unknown",
                "selected_for_target": bool(c.selected_for_target),
                "is_augmented": bool(c.is_augmented),
                "is_disagreement": bool(c.is_disagreement),
            }
            for c in cands
        ],
    }


def _write_raw_groups_jsonl(path: Path, groups: Sequence[QueryGroup]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for g in groups:
            f.write(json.dumps(_query_group_to_raw_obj(g), ensure_ascii=False) + "\n")


def _write_pair_rows_jsonl(path: Path, pairs: Sequence[PairExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(
                json.dumps(
                    {
                        "grant_id": _clean_text(p.grant_id),
                        "spec_idx": int(p.spec_idx),
                        "query_text": _clean_text(p.query_text),
                        "pos_text": _clean_text(p.pos_text),
                        "neg_text": _clean_text(p.neg_text),
                        "teacher_pos_score": float(p.teacher_pos_score),
                        "teacher_neg_score": float(p.teacher_neg_score),
                        "teacher_margin": float(p.teacher_margin),
                        "pair_type": _clean_text(p.pair_type) or "unknown",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _derive_pairs_for_groups(
    groups: Sequence[QueryGroup],
    *,
    pair_derive_pos_k: int,
    pair_derive_hard_k: int,
    pair_derive_weak_k: int,
    pair_derive_cap: int,
    pair_add_mid_lower_mid: bool,
    pair_mid_pos_score_min: float,
    pair_mid_pos_score_max: float,
    pair_mid_neg_score_min: float,
    pair_mid_neg_score_max: float,
    pair_mid_margin_min: float,
    pair_mid_margin_max: float,
    pair_mid_add_easy_contrast: bool,
    pair_mid_min_candidates: int,
    pair_mid_start_ratio: float,
    pair_mid_end_ratio: float,
    pair_lower_mid_start_ratio: float,
    pair_lower_mid_end_ratio: float,
) -> List[PairExample]:
    base_pairs = _derive_pairwise_from_groups(
        groups,
        per_query_pos_k=int(pair_derive_pos_k),
        per_query_hard_k=int(pair_derive_hard_k),
        per_query_weak_k=int(pair_derive_weak_k),
        per_query_cap=int(pair_derive_cap),
    )
    if not bool(pair_add_mid_lower_mid):
        return base_pairs
    mid_pairs = _derive_mid_lower_mid_pairs_from_groups(
        groups,
        per_query_cap=int(pair_derive_cap),
        mid_min_candidates=int(pair_mid_min_candidates),
        mid_start_ratio=float(pair_mid_start_ratio),
        mid_end_ratio=float(pair_mid_end_ratio),
        lower_mid_start_ratio=float(pair_lower_mid_start_ratio),
        lower_mid_end_ratio=float(pair_lower_mid_end_ratio),
        pos_score_min=float(pair_mid_pos_score_min),
        pos_score_max=float(pair_mid_pos_score_max),
        neg_score_min=float(pair_mid_neg_score_min),
        neg_score_max=float(pair_mid_neg_score_max),
        margin_min=float(pair_mid_margin_min),
        margin_max=float(pair_mid_margin_max),
        add_easy_contrast=bool(pair_mid_add_easy_contrast),
    )
    return list(base_pairs) + list(mid_pairs)


def _sample_groups_diverse(groups: Sequence[QueryGroup], *, target: int, seed: int) -> List[QueryGroup]:
    k = max(0, int(target))
    if k <= 0 or (not groups):
        return []

    by_grant: Dict[str, List[QueryGroup]] = {}
    for g in groups:
        gid = _clean_text(g.grant_id) or "unknown"
        by_grant.setdefault(gid, []).append(g)
    for gid, items in by_grant.items():
        items.sort(key=lambda x: _hash_to_unit_interval(f"{int(seed)}::{gid}::{_query_group_key(x)}"))

    grant_order = sorted(
        list(by_grant.keys()),
        key=lambda gid: _hash_to_unit_interval(f"{int(seed)}::grant::{gid}"),
    )

    out: List[QueryGroup] = []
    active = list(grant_order)
    while active and len(out) < k:
        next_active: List[str] = []
        for gid in active:
            bucket = by_grant.get(gid) or []
            if not bucket:
                continue
            out.append(bucket.pop(0))
            if bucket:
                next_active.append(gid)
            if len(out) >= k:
                break
        active = next_active
    return out


def _sample_replay_groups(
    *,
    original_train_groups: Sequence[QueryGroup],
    replay_target: int,
    high_threshold: float,
    mid_threshold: float,
    high_ratio: float,
    mid_ratio: float,
    low_ratio: float,
    seed: int,
) -> List[QueryGroup]:
    target = max(0, int(replay_target))
    if target <= 0:
        return []

    ratios = [max(0.0, float(high_ratio)), max(0.0, float(mid_ratio)), max(0.0, float(low_ratio))]
    ratio_sum = float(sum(ratios))
    if ratio_sum <= 0.0:
        ratios = [0.3, 0.4, 0.3]
        ratio_sum = 1.0
    ratios = [x / ratio_sum for x in ratios]

    by_band: Dict[str, List[QueryGroup]] = {"high": [], "mid": [], "low": []}
    for g in original_train_groups:
        b = _group_band(g, high_threshold=high_threshold, mid_threshold=mid_threshold)
        by_band[b].append(g)

    plan = {
        "high": int(round(float(target) * float(ratios[0]))),
        "mid": int(round(float(target) * float(ratios[1]))),
        "low": int(round(float(target) * float(ratios[2]))),
    }
    planned_sum = int(plan["high"] + plan["mid"] + plan["low"])
    if planned_sum != target:
        plan["mid"] += int(target - planned_sum)
    if plan["mid"] < 0:
        plan["mid"] = 0

    selected: List[QueryGroup] = []
    for band, offset in (("high", 101), ("mid", 202), ("low", 303)):
        picked = _sample_groups_diverse(by_band[band], target=int(plan[band]), seed=int(seed + offset))
        selected.extend(picked)

    if len(selected) < target:
        used = {_query_group_key(x) for x in selected}
        remain = [g for g in original_train_groups if _query_group_key(g) not in used]
        fill = _sample_groups_diverse(remain, target=int(target - len(selected)), seed=int(seed + 404))
        selected.extend(fill)

    return selected[:target]


def _collect_ranking_mistake_query_keys(
    rows: Sequence[Dict[str, Any]],
    *,
    model_score_key: str,
    high_threshold: float,
    mid_threshold: float,
) -> Set[str]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_query_key_from_row(row), []).append(row)

    bad: Set[str] = set()
    for qk, items in grouped.items():
        if len(items) < 2:
            continue
        teacher = [float(x.get("teacher_score_used") or 0.0) for x in items]
        pred = [float(x.get(model_score_key) or 0.0) for x in items]
        important = [i for i, t in enumerate(teacher) if t >= float(high_threshold)]
        if not important:
            continue
        top_pred = max(range(len(items)), key=lambda i: float(pred[i]))
        if top_pred not in set(important):
            bad.add(qk)
            continue
        if any(float(pred[i]) < float(mid_threshold) for i in important):
            bad.add(qk)
    return bad


def _filter_mined_rows_policy(
    *,
    rows: Sequence[Dict[str, Any]],
    mismatch_target: str,
    high_threshold: float,
    mid_threshold: float,
    gap_min: float,
    boundary_bandwidth: float,
    low_max_ratio: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    target = _clean_text(mismatch_target).lower() or "finetuned"
    if target not in {"finetuned", "base", "either", "both"}:
        target = "finetuned"

    ranking_ft = _collect_ranking_mistake_query_keys(
        rows,
        model_score_key="finetuned_score",
        high_threshold=float(high_threshold),
        mid_threshold=float(mid_threshold),
    )
    ranking_base = _collect_ranking_mistake_query_keys(
        rows,
        model_score_key="base_score",
        high_threshold=float(high_threshold),
        mid_threshold=float(mid_threshold),
    )
    if target == "base":
        ranking_bad_keys = set(ranking_base)
    elif target == "both":
        ranking_bad_keys = set(ranking_ft).intersection(ranking_base)
    elif target == "either":
        ranking_bad_keys = set(ranking_ft).union(ranking_base)
    else:
        ranking_bad_keys = set(ranking_ft)

    selected: List[Dict[str, Any]] = []
    for row in rows:
        gt = float(row.get("teacher_score_used") or 0.0)
        ft = float(row.get("finetuned_score") or 0.0)
        b = float(row.get("base_score") or 0.0)
        gt_band = _clean_text(row.get("gt_cluster")) or _score_band_from_value(
            gt,
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
        )
        ft_band = _clean_text(row.get("finetuned_cluster")) or _score_band_from_value(
            ft,
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
        )
        b_band = _clean_text(row.get("base_cluster")) or _score_band_from_value(
            b,
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
        )

        ft_mis = bool(row.get("finetuned_cluster_mismatch")) or (gt_band != ft_band)
        b_mis = bool(row.get("base_cluster_mismatch")) or (gt_band != b_band)
        ft_gap = abs(float(ft - gt))
        b_gap = abs(float(b - gt))

        if target == "base":
            model_pred_band = b_band
            cluster_mismatch = b_mis
            gap = float(b_gap)
        elif target == "both":
            model_pred_band = ft_band
            cluster_mismatch = bool(ft_mis and b_mis)
            gap = float(max(ft_gap, b_gap))
        elif target == "either":
            model_pred_band = ft_band
            cluster_mismatch = bool(ft_mis or b_mis)
            gap = float(max(ft_gap, b_gap))
        else:
            model_pred_band = ft_band
            cluster_mismatch = ft_mis
            gap = float(ft_gap)

        qk = _query_key_from_row(row)

        wrong_band = bool(gt_band != model_pred_band)
        rank_bad = bool(qk in ranking_bad_keys)

        if not cluster_mismatch:
            continue
        if not (gap >= float(gap_min) or wrong_band or rank_bad):
            continue

        near_boundary = min(abs(float(gt) - float(high_threshold)), abs(float(gt) - float(mid_threshold))) <= float(boundary_bandwidth)
        priority = 0.0
        if gt_band == "high" and model_pred_band in {"mid", "low"}:
            priority += 5.0
        elif gt_band == "mid" and model_pred_band in {"high", "low"}:
            priority += 4.0
        elif gt_band == "low" and model_pred_band in {"high", "mid"}:
            priority += 2.0
        if near_boundary:
            priority += 2.0
        if rank_bad:
            priority += 3.0
        priority += float(gap)

        copied = dict(row)
        copied["_mine_priority"] = float(priority)
        copied["_mine_gap"] = float(gap)
        copied["_mine_rank_bad"] = bool(rank_bad)
        copied["_mine_near_boundary"] = bool(near_boundary)
        copied["_mine_gt_band"] = gt_band
        copied["_mine_pred_band"] = model_pred_band
        selected.append(copied)

    selected.sort(key=lambda x: (float(x.get("_mine_priority") or 0.0), float(x.get("_mine_gap") or 0.0)), reverse=True)

    low_ratio_cap = max(0.0, min(1.0, float(low_max_ratio)))
    low_rows = [r for r in selected if _clean_text(r.get("_mine_gt_band")) == "low"]
    non_low_rows = [r for r in selected if _clean_text(r.get("_mine_gt_band")) != "low"]
    max_low = int(round(float(len(selected)) * low_ratio_cap))
    if max_low < 0:
        max_low = 0
    kept_low = low_rows[:max_low] if len(low_rows) > max_low else low_rows
    final_rows = list(non_low_rows) + list(kept_low)
    final_rows.sort(key=lambda x: (float(x.get("_mine_priority") or 0.0), float(x.get("_mine_gap") or 0.0)), reverse=True)

    stats = {
        "mismatch_target": target,
        "input_rows": int(len(rows)),
        "selected_before_low_cap": int(len(selected)),
        "selected_after_low_cap": int(len(final_rows)),
        "ranking_mistake_queries": int(len(ranking_bad_keys)),
        "low_cap_ratio": float(low_ratio_cap),
        "low_rows_before_cap": int(len(low_rows)),
        "low_rows_after_cap": int(len(kept_low)),
    }
    return final_rows, stats


def _run_iterative_orchestration(args: argparse.Namespace, *, raw_argv: Sequence[str]) -> int:
    ensure_split_files = _resolve_split_helper()

    iterative_max_rounds = _safe_int(args.iterative_max_rounds, default=1, minimum=1, maximum=100)
    if iterative_max_rounds <= 1:
        return 0

    train_script_path = Path(__file__).resolve()
    eval_script_path = (PROJECT_ROOT / "cross_encoder" / "spec_to_spec" / "eval" / "eval_faculty_grant_spec_compare_cluster_mismatch.py").resolve()
    if not eval_script_path.exists():
        raise RuntimeError(f"Mismatch eval script not found: {eval_script_path}")

    seed = _safe_int(args.seed, default=42, minimum=0, maximum=2_147_483_647)
    val_ratio = _safe_float(args.val_ratio, default=0.1, minimum=0.0, maximum=0.5)
    test_ratio = _safe_float(args.test_ratio, default=0.1, minimum=0.0, maximum=0.5)
    if (val_ratio + test_ratio) >= 0.99:
        test_ratio = max(0.0, 0.99 - val_ratio)

    stage1_epochs = _safe_int(args.stage1_epochs, default=1, minimum=0, maximum=100)
    stage2_epochs = _safe_int(args.stage2_epochs, default=3, minimum=0, maximum=100)

    output_dir_base = _resolve_path(args.output_dir)
    iterative_root = (output_dir_base / "iterative_rounds").resolve()
    iterative_root.mkdir(parents=True, exist_ok=True)

    original_raw_input = _resolve_path(args.raw_input)
    original_pairwise_input = _resolve_path(args.pairwise_input)
    original_split_dir = _resolve_path(args.split_dir)

    fixed_eval_test_input = _resolve_path(args.iterative_eval_test_input) if _clean_text(args.iterative_eval_test_input) else None
    eval_base_model = _clean_text(args.iterative_eval_base_model) or (_clean_text(args.model_id) or MODEL_ID_DEFAULT)
    current_model_id = _clean_text(args.model_id) or MODEL_ID_DEFAULT

    mismatch_target = _clean_text(args.iterative_mismatch_target).lower() or "finetuned"
    mismatch_ground_truth = _clean_text(args.iterative_mismatch_ground_truth).lower() or "raw"
    mismatch_high_threshold = _safe_float(
        args.iterative_mismatch_high_threshold,
        default=TRAIN_BAND_HIGH_THRESHOLD_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    mismatch_mid_threshold = _safe_float(
        args.iterative_mismatch_mid_threshold,
        default=TRAIN_BAND_MID_THRESHOLD_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    if mismatch_mid_threshold > mismatch_high_threshold:
        mismatch_mid_threshold = mismatch_high_threshold
    mismatch_pred_high_threshold = _safe_float(args.iterative_mismatch_pred_high_threshold, default=-1.0, minimum=-1.0, maximum=1.0)
    mismatch_pred_mid_threshold = _safe_float(args.iterative_mismatch_pred_mid_threshold, default=-1.0, minimum=-1.0, maximum=1.0)
    mismatch_sample_high = _safe_int(args.iterative_mismatch_sample_high, default=0, minimum=0, maximum=50_000_000)
    mismatch_sample_mid = _safe_int(args.iterative_mismatch_sample_mid, default=0, minimum=0, maximum=50_000_000)
    mismatch_sample_low = _safe_int(args.iterative_mismatch_sample_low, default=0, minimum=0, maximum=50_000_000)
    mismatch_max_pairs_scan = _safe_int(args.iterative_mismatch_max_pairs_scan, default=0, minimum=0, maximum=200_000_000)
    mismatch_min_rows = _safe_int(args.iterative_mismatch_min_rows, default=1, minimum=1, maximum=10_000_000)
    mismatch_max_rows = _safe_int(args.iterative_mismatch_max_rows, default=0, minimum=0, maximum=200_000_000)
    iterative_stop_on_empty = bool(args.iterative_stop_on_empty)
    mined_gap_min = _safe_float(args.iterative_mined_gap_min, default=0.15, minimum=0.0, maximum=1.0)
    mined_boundary_bandwidth = _safe_float(args.iterative_mined_boundary_bandwidth, default=0.08, minimum=0.0, maximum=1.0)
    mined_low_max_ratio = _safe_float(args.iterative_mined_low_max_ratio, default=0.35, minimum=0.0, maximum=1.0)
    train_band_high_threshold = _safe_float(
        args.train_band_high_threshold,
        default=TRAIN_BAND_HIGH_THRESHOLD_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    train_band_mid_threshold = _safe_float(
        args.train_band_mid_threshold,
        default=TRAIN_BAND_MID_THRESHOLD_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    if train_band_mid_threshold > train_band_high_threshold:
        train_band_mid_threshold = train_band_high_threshold
    train_low_band_cap_ratio = _safe_float(args.train_low_band_cap_ratio, default=1.0, minimum=0.0, maximum=100.0)

    pair_derive_pos_k = _safe_int(args.pair_derive_pos_k, default=4, minimum=1, maximum=10_000)
    pair_derive_hard_k = _safe_int(args.pair_derive_hard_k, default=4, minimum=0, maximum=10_000)
    pair_derive_weak_k = _safe_int(args.pair_derive_weak_k, default=4, minimum=1, maximum=10_000)
    pair_derive_cap = _safe_int(args.pair_derive_cap, default=64, minimum=1, maximum=100_000)
    pair_add_mid_lower_mid = bool(args.pair_add_mid_lower_mid)
    pair_mid_pos_score_min = _safe_float(args.pair_mid_pos_score_min, default=0.4, minimum=0.0, maximum=1.0)
    pair_mid_pos_score_max = _safe_float(args.pair_mid_pos_score_max, default=0.7, minimum=0.0, maximum=1.0)
    pair_mid_neg_score_min = _safe_float(args.pair_mid_neg_score_min, default=0.2, minimum=0.0, maximum=1.0)
    pair_mid_neg_score_max = _safe_float(args.pair_mid_neg_score_max, default=0.5, minimum=0.0, maximum=1.0)
    pair_mid_margin_min = _safe_float(args.pair_mid_margin_min, default=0.05, minimum=0.0, maximum=1.0)
    pair_mid_margin_max = _safe_float(args.pair_mid_margin_max, default=0.4, minimum=0.0, maximum=1.0)
    pair_mid_add_easy_contrast = bool(args.pair_mid_add_easy_contrast)
    pair_mid_min_candidates = _safe_int(
        args.pair_mid_min_candidates,
        default=PAIR_MID_MIN_CANDIDATES_DEFAULT,
        minimum=2,
        maximum=100_000,
    )
    pair_mid_start_ratio = _safe_float(
        args.pair_mid_start_ratio,
        default=PAIR_MID_START_RATIO_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    pair_mid_end_ratio = _safe_float(
        args.pair_mid_end_ratio,
        default=PAIR_MID_END_RATIO_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    pair_lower_mid_start_ratio = _safe_float(
        args.pair_lower_mid_start_ratio,
        default=PAIR_LOWER_MID_START_RATIO_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    pair_lower_mid_end_ratio = _safe_float(
        args.pair_lower_mid_end_ratio,
        default=PAIR_LOWER_MID_END_RATIO_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    if pair_mid_pos_score_max < pair_mid_pos_score_min:
        pair_mid_pos_score_max = pair_mid_pos_score_min
    if pair_mid_neg_score_max < pair_mid_neg_score_min:
        pair_mid_neg_score_max = pair_mid_neg_score_min
    if pair_mid_margin_max < pair_mid_margin_min:
        pair_mid_margin_max = pair_mid_margin_min
    if pair_mid_end_ratio < pair_mid_start_ratio:
        pair_mid_end_ratio = pair_mid_start_ratio
    if pair_lower_mid_end_ratio < pair_lower_mid_start_ratio:
        pair_lower_mid_end_ratio = pair_lower_mid_start_ratio

    replay_multiplier = _safe_float(args.iterative_replay_multiplier, default=0.25, minimum=0.0, maximum=10.0)
    replay_min_groups = _safe_int(args.iterative_replay_min_groups, default=0, minimum=0, maximum=10_000_000)
    replay_max_multiplier = _safe_float(args.iterative_replay_max_multiplier, default=0.5, minimum=0.0, maximum=10.0)
    replay_high_ratio = _safe_float(args.iterative_replay_high_ratio, default=0.3, minimum=0.0, maximum=1.0)
    replay_mid_ratio = _safe_float(args.iterative_replay_mid_ratio, default=0.4, minimum=0.0, maximum=1.0)
    replay_low_ratio = _safe_float(args.iterative_replay_low_ratio, default=0.3, minimum=0.0, maximum=1.0)
    iterative_stop_patience = _safe_int(args.iterative_stop_patience, default=2, minimum=1, maximum=20)

    if not original_raw_input.exists():
        raise RuntimeError(f"raw_input not found: {original_raw_input}")

    original_split_result = ensure_split_files(
        raw_input=original_raw_input,
        pairwise_input=original_pairwise_input if original_pairwise_input.exists() else None,
        split_dir=original_split_dir,
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        seed=int(seed),
        overwrite=bool(args.regenerate_splits),
        high_threshold=float(mismatch_high_threshold),
        mid_threshold=float(mismatch_mid_threshold),
    )
    original_paths = original_split_result["paths"]

    original_train_groups = _load_raw_groups(original_paths.raw_train, max_queries=0)
    original_val_groups = _load_raw_groups(original_paths.raw_val, max_queries=0)
    original_test_groups = _load_raw_groups(original_paths.raw_test, max_queries=0)
    if not original_train_groups:
        raise RuntimeError("Original train split is empty.")
    original_train_groups, original_train_low_cap_stats = _cap_low_band_groups(
        original_train_groups,
        high_threshold=float(train_band_high_threshold),
        mid_threshold=float(train_band_mid_threshold),
        low_cap_ratio_to_high=float(train_low_band_cap_ratio),
        seed=int(seed),
    )
    if not original_train_groups:
        raise RuntimeError("Original train split became empty after low-band cap.")

    mining_dev_raw_input = _resolve_path(args.iterative_mining_input) if _clean_text(args.iterative_mining_input) else None
    final_test_raw_input = _resolve_path(args.iterative_final_test_input) if _clean_text(args.iterative_final_test_input) else None
    if mining_dev_raw_input is None and fixed_eval_test_input is not None:
        mining_dev_raw_input = fixed_eval_test_input
    if mining_dev_raw_input is None:
        mining_dev_raw_input = original_paths.raw_val
    if final_test_raw_input is None:
        final_test_raw_input = original_paths.raw_test
    if not mining_dev_raw_input.exists():
        raise RuntimeError(f"iterative_mining_input not found: {mining_dev_raw_input}")
    if not final_test_raw_input.exists():
        raise RuntimeError(f"iterative_final_test_input not found: {final_test_raw_input}")
    if mining_dev_raw_input.resolve() == final_test_raw_input.resolve():
        raise RuntimeError(
            "Mining-dev input and final-test input are identical. "
            "Use separate datasets to avoid test leakage."
        )

    fixed_mining_dev_groups = _load_raw_groups(mining_dev_raw_input, max_queries=0)
    fixed_final_test_groups = _load_raw_groups(final_test_raw_input, max_queries=0)
    if not fixed_mining_dev_groups:
        raise RuntimeError("Mining-dev split is empty.")
    if not fixed_final_test_groups:
        raise RuntimeError("Final held-out test split is empty.")
    fixed_eval_test_input = mining_dev_raw_input

    if original_paths.pair_train.exists() and original_paths.pair_val.exists():
        original_train_pairs = _load_pairwise_rows(original_paths.pair_train, max_rows=0)
        original_val_pairs = _load_pairwise_rows(original_paths.pair_val, max_rows=0)
    else:
        original_train_pairs = _derive_pairs_for_groups(
            original_train_groups,
            pair_derive_pos_k=pair_derive_pos_k,
            pair_derive_hard_k=pair_derive_hard_k,
            pair_derive_weak_k=pair_derive_weak_k,
            pair_derive_cap=pair_derive_cap,
            pair_add_mid_lower_mid=pair_add_mid_lower_mid,
            pair_mid_pos_score_min=pair_mid_pos_score_min,
            pair_mid_pos_score_max=pair_mid_pos_score_max,
            pair_mid_neg_score_min=pair_mid_neg_score_min,
            pair_mid_neg_score_max=pair_mid_neg_score_max,
            pair_mid_margin_min=pair_mid_margin_min,
            pair_mid_margin_max=pair_mid_margin_max,
            pair_mid_add_easy_contrast=pair_mid_add_easy_contrast,
            pair_mid_min_candidates=pair_mid_min_candidates,
            pair_mid_start_ratio=pair_mid_start_ratio,
            pair_mid_end_ratio=pair_mid_end_ratio,
            pair_lower_mid_start_ratio=pair_lower_mid_start_ratio,
            pair_lower_mid_end_ratio=pair_lower_mid_end_ratio,
        )
        original_val_pairs = _derive_pairs_for_groups(
            original_val_groups,
            pair_derive_pos_k=pair_derive_pos_k,
            pair_derive_hard_k=pair_derive_hard_k,
            pair_derive_weak_k=pair_derive_weak_k,
            pair_derive_cap=pair_derive_cap,
            pair_add_mid_lower_mid=pair_add_mid_lower_mid,
            pair_mid_pos_score_min=pair_mid_pos_score_min,
            pair_mid_pos_score_max=pair_mid_pos_score_max,
            pair_mid_neg_score_min=pair_mid_neg_score_min,
            pair_mid_neg_score_max=pair_mid_neg_score_max,
            pair_mid_margin_min=pair_mid_margin_min,
            pair_mid_margin_max=pair_mid_margin_max,
            pair_mid_add_easy_contrast=pair_mid_add_easy_contrast,
            pair_mid_min_candidates=pair_mid_min_candidates,
            pair_mid_start_ratio=pair_mid_start_ratio,
            pair_mid_end_ratio=pair_mid_end_ratio,
            pair_lower_mid_start_ratio=pair_lower_mid_start_ratio,
            pair_lower_mid_end_ratio=pair_lower_mid_end_ratio,
        )

    original_train_pairs_before_low_cap_filter = int(len(original_train_pairs))
    original_train_pairs = _filter_pairs_for_groups(original_train_pairs, original_train_groups)
    original_train_pairs_after_low_cap_filter = int(len(original_train_pairs))
    original_train_pairs_low_cap_filtered = (
        original_train_pairs_before_low_cap_filter - original_train_pairs_after_low_cap_filter
    )
    if not original_train_pairs:
        original_train_pairs = _derive_pairs_for_groups(
            original_train_groups,
            pair_derive_pos_k=pair_derive_pos_k,
            pair_derive_hard_k=pair_derive_hard_k,
            pair_derive_weak_k=pair_derive_weak_k,
            pair_derive_cap=pair_derive_cap,
            pair_add_mid_lower_mid=pair_add_mid_lower_mid,
            pair_mid_pos_score_min=pair_mid_pos_score_min,
            pair_mid_pos_score_max=pair_mid_pos_score_max,
            pair_mid_neg_score_min=pair_mid_neg_score_min,
            pair_mid_neg_score_max=pair_mid_neg_score_max,
            pair_mid_margin_min=pair_mid_margin_min,
            pair_mid_margin_max=pair_mid_margin_max,
            pair_mid_add_easy_contrast=pair_mid_add_easy_contrast,
            pair_mid_min_candidates=pair_mid_min_candidates,
            pair_mid_start_ratio=pair_mid_start_ratio,
            pair_mid_end_ratio=pair_mid_end_ratio,
            pair_lower_mid_start_ratio=pair_lower_mid_start_ratio,
            pair_lower_mid_end_ratio=pair_lower_mid_end_ratio,
        )

    fixed_mining_dev_pairs = _derive_pairs_for_groups(
        fixed_mining_dev_groups,
        pair_derive_pos_k=pair_derive_pos_k,
        pair_derive_hard_k=pair_derive_hard_k,
        pair_derive_weak_k=pair_derive_weak_k,
        pair_derive_cap=pair_derive_cap,
        pair_add_mid_lower_mid=pair_add_mid_lower_mid,
        pair_mid_pos_score_min=pair_mid_pos_score_min,
        pair_mid_pos_score_max=pair_mid_pos_score_max,
        pair_mid_neg_score_min=pair_mid_neg_score_min,
        pair_mid_neg_score_max=pair_mid_neg_score_max,
        pair_mid_margin_min=pair_mid_margin_min,
        pair_mid_margin_max=pair_mid_margin_max,
        pair_mid_add_easy_contrast=pair_mid_add_easy_contrast,
        pair_mid_min_candidates=pair_mid_min_candidates,
        pair_mid_start_ratio=pair_mid_start_ratio,
        pair_mid_end_ratio=pair_mid_end_ratio,
        pair_lower_mid_start_ratio=pair_lower_mid_start_ratio,
        pair_lower_mid_end_ratio=pair_lower_mid_end_ratio,
    )

    current_train_groups = list(original_train_groups)
    current_val_groups = list(original_val_groups)
    current_test_groups = list(fixed_mining_dev_groups)
    current_train_pairs = list(original_train_pairs)
    current_val_pairs = list(original_val_pairs)
    current_test_pairs = list(fixed_mining_dev_pairs)

    round_summaries: List[Dict[str, Any]] = []
    final_model_dir: Optional[Path] = None
    best_mining_dev_ndcg = -1.0
    rounds_without_improve = 0

    print("iterative_mode=true")
    print(f"iterative_max_rounds={iterative_max_rounds}")
    print(f"iterative_root={iterative_root}")
    print(f"fixed_original_split_dir={original_paths.split_dir}")
    print(f"fixed_mining_dev_input={mining_dev_raw_input}")
    print(f"fixed_final_test_input={final_test_raw_input}")
    print(f"fixed_original_train_queries={len(original_train_groups)}")
    print(f"fixed_original_val_queries={len(original_val_groups)}")
    print(f"fixed_original_test_queries={len(original_test_groups)}")
    low_cap_before = original_train_low_cap_stats.get("before") if isinstance(original_train_low_cap_stats, dict) else {}
    low_cap_after = original_train_low_cap_stats.get("after") if isinstance(original_train_low_cap_stats, dict) else {}
    print(
        "fixed_original_train_low_cap="
        f"applied={bool(original_train_low_cap_stats.get('applied'))} "
        f"reason={_clean_text(original_train_low_cap_stats.get('reason')) or 'unknown'} "
        f"ratio={float(original_train_low_cap_stats.get('ratio', 0.0)):.4f} "
        f"target={int(original_train_low_cap_stats.get('low_cap_target', -1))} "
        f"before(high={int((low_cap_before or {}).get('high', 0))},"
        f"mid={int((low_cap_before or {}).get('mid', 0))},"
        f"low={int((low_cap_before or {}).get('low', 0))},"
        f"total={int((low_cap_before or {}).get('total', 0))}) "
        f"after(high={int((low_cap_after or {}).get('high', 0))},"
        f"mid={int((low_cap_after or {}).get('mid', 0))},"
        f"low={int((low_cap_after or {}).get('low', 0))},"
        f"total={int((low_cap_after or {}).get('total', 0))})"
    )
    print(
        "fixed_original_train_pairs_low_cap_filter="
        f"before={original_train_pairs_before_low_cap_filter} "
        f"after={original_train_pairs_after_low_cap_filter} "
        f"dropped={original_train_pairs_low_cap_filtered}"
    )
    print(f"fixed_mining_dev_queries={len(fixed_mining_dev_groups)}")
    print(f"fixed_final_test_queries={len(fixed_final_test_groups)}")

    for round_idx in range(iterative_max_rounds):
        round_dir = (iterative_root / f"round_{round_idx:02d}").resolve()
        round_output_dir = (round_dir / "model").resolve()
        round_output_dir.mkdir(parents=True, exist_ok=True)
        round_split_dir = (round_dir / "train_splits").resolve()
        round_split_dir.mkdir(parents=True, exist_ok=True)

        round_raw_train = round_split_dir / "llm_distill_raw_train.jsonl"
        round_raw_val = round_split_dir / "llm_distill_raw_val.jsonl"
        round_raw_test = round_split_dir / "llm_distill_raw_test.jsonl"
        round_pair_train = round_split_dir / "llm_distill_pairwise_train.jsonl"
        round_pair_val = round_split_dir / "llm_distill_pairwise_val.jsonl"
        round_pair_test = round_split_dir / "llm_distill_pairwise_test.jsonl"

        _write_raw_groups_jsonl(round_raw_train, current_train_groups)
        _write_raw_groups_jsonl(round_raw_val, current_val_groups)
        _write_raw_groups_jsonl(round_raw_test, current_test_groups)
        _write_pair_rows_jsonl(round_pair_train, current_train_pairs)
        _write_pair_rows_jsonl(round_pair_val, current_val_pairs)
        _write_pair_rows_jsonl(round_pair_test, current_test_pairs)

        print(f"iterative_round_start={round_idx}/{iterative_max_rounds - 1}")
        print(f"round_train_queries={len(current_train_groups)}")
        print(f"round_val_queries={len(current_val_groups)}")
        print(f"round_test_queries={len(current_test_groups)}")
        print(f"round_pair_train={len(current_train_pairs)}")
        print(f"round_pair_val={len(current_val_pairs)}")
        print(f"round_pair_test={len(current_test_pairs)}")
        print(f"round_training_model_id={current_model_id}")
        pairwise_hint_path = original_pairwise_input if original_pairwise_input.exists() else round_pair_train

        child_cmd: List[str] = [
            sys.executable,
            str(train_script_path),
            *[str(x) for x in raw_argv],
            "--iterative-inner-run",
            "--iterative-max-rounds",
            "1",
            "--output-dir",
            str(round_output_dir),
            "--no-append-args-to-output-dir",
            "--model-id",
            str(current_model_id),
            "--raw-input",
            str(original_raw_input),
            "--pairwise-input",
            str(pairwise_hint_path),
            "--split-dir",
            str(round_split_dir),
            "--use-prepared-splits",
            "--no-regenerate-splits",
            "--raw-train-input",
            str(round_raw_train),
            "--raw-val-input",
            str(round_raw_val),
            "--raw-test-input",
            str(round_raw_test),
            "--pairwise-train-input",
            str(round_pair_train),
            "--pairwise-val-input",
            str(round_pair_val),
            "--pairwise-test-input",
            str(round_pair_test),
        ]
        _run_command(child_cmd, cwd=PROJECT_ROOT)

        round_manifest_path = round_output_dir / "train_manifest.json"
        if not round_manifest_path.exists():
            raise RuntimeError(f"Expected train manifest not found: {round_manifest_path}")
        round_manifest = json.loads(round_manifest_path.read_text(encoding="utf-8"))

        trained_model_dir = _resolve_round_model_dir(
            round_output_dir=round_output_dir,
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
        )
        final_model_dir = trained_model_dir
        current_model_id = str(trained_model_dir)
        mining_dev_ndcg = float((round_manifest.get("test_metrics") or {}).get("ndcg@10", 0.0))
        mining_dev_mrr = float((round_manifest.get("test_metrics") or {}).get("mrr@10", 0.0))
        val_ndcg = float(round_manifest.get("best_ndcg@10", 0.0))

        if mining_dev_ndcg > best_mining_dev_ndcg:
            best_mining_dev_ndcg = mining_dev_ndcg
            rounds_without_improve = 0
        else:
            rounds_without_improve += 1

        round_summary: Dict[str, Any] = {
            "round_idx": int(round_idx),
            "round_output_dir": str(round_output_dir),
            "round_manifest_path": str(round_manifest_path),
            "trained_model_dir": str(trained_model_dir),
            "train_manifest_best_ndcg@10": float(round_manifest.get("best_ndcg@10", 0.0)),
            "fixed_val_ndcg@10": float(val_ndcg),
            "mining_dev_ndcg@10": float(mining_dev_ndcg),
            "mining_dev_mrr@10": float(mining_dev_mrr),
            "rounds_without_mining_dev_improve": int(rounds_without_improve),
            "best_mining_dev_ndcg@10": float(best_mining_dev_ndcg),
        }

        if round_idx >= (iterative_max_rounds - 1):
            round_summaries.append(round_summary)
            print(f"iterative_round_end={round_idx}/{iterative_max_rounds - 1} status=last_round_no_mismatch_mining")
            break
        if rounds_without_improve >= int(iterative_stop_patience):
            round_summary["stop_reason"] = f"mining_dev_no_improve_patience_{int(iterative_stop_patience)}"
            round_summaries.append(round_summary)
            print(f"iterative_round_end={round_idx}/{iterative_max_rounds - 1} status={round_summary['stop_reason']}")
            break

        mismatch_eval_out_dir = (round_dir / "mismatch_eval").resolve()
        mismatch_eval_out_dir.mkdir(parents=True, exist_ok=True)
        eval_cmd: List[str] = [
            sys.executable,
            str(eval_script_path),
            "--finetuned-model",
            str(trained_model_dir),
            "--base-model",
            str(eval_base_model),
            "--distill-test-input",
            str(fixed_eval_test_input),
            "--distill-input",
            str(fixed_eval_test_input),
            "--distill-ground-truth",
            str(mismatch_ground_truth),
            "--distill-high-threshold",
            str(mismatch_high_threshold),
            "--distill-mid-threshold",
            str(mismatch_mid_threshold),
            "--distill-sample-high",
            str(mismatch_sample_high),
            "--distill-sample-mid",
            str(mismatch_sample_mid),
            "--distill-sample-low",
            str(mismatch_sample_low),
            "--distill-max-pairs-scan",
            str(mismatch_max_pairs_scan),
            "--mismatch-target",
            str(mismatch_target),
            "--pred-high-threshold",
            str(mismatch_pred_high_threshold),
            "--pred-mid-threshold",
            str(mismatch_pred_mid_threshold),
            "--output-dir",
            str(mismatch_eval_out_dir),
            "--save",
            "--no-print",
            "--no-distill-print-pair-tables",
            "--no-distill-save-pair-tables",
        ]
        _run_command(eval_cmd, cwd=PROJECT_ROOT)

        mismatch_eval_json = _find_latest_mismatch_json(
            eval_output_dir=mismatch_eval_out_dir,
            mismatch_target=mismatch_target,
        )
        mismatch_rows_raw = _load_mismatch_rows(mismatch_eval_json)
        mismatch_rows, mined_filter_stats = _filter_mined_rows_policy(
            rows=mismatch_rows_raw,
            mismatch_target=mismatch_target,
            high_threshold=float(mismatch_high_threshold),
            mid_threshold=float(mismatch_mid_threshold),
            gap_min=float(mined_gap_min),
            boundary_bandwidth=float(mined_boundary_bandwidth),
            low_max_ratio=float(mined_low_max_ratio),
        )
        if mismatch_max_rows > 0:
            mismatch_rows = mismatch_rows[: int(mismatch_max_rows)]

        mismatch_row_count = int(len(mismatch_rows))
        round_summary["mismatch_eval_json"] = str(mismatch_eval_json)
        round_summary["mismatch_rows_raw"] = int(len(mismatch_rows_raw))
        round_summary["mismatch_rows"] = int(mismatch_row_count)
        round_summary["mined_filter_stats"] = mined_filter_stats
        print(f"iterative_round_mismatch_rows={mismatch_row_count}")

        if mismatch_row_count < int(mismatch_min_rows):
            round_summary["stop_reason"] = (
                f"mismatch_rows_below_min({mismatch_row_count}<{int(mismatch_min_rows)})"
            )
            round_summaries.append(round_summary)
            print(f"iterative_round_end={round_idx}/{iterative_max_rounds - 1} status={round_summary['stop_reason']}")
            if iterative_stop_on_empty:
                break
            continue

        mismatch_dataset_dir = (round_dir / "mismatch_dataset").resolve()
        mismatch_dataset_dir.mkdir(parents=True, exist_ok=True)
        mismatch_raw_path = mismatch_dataset_dir / "llm_distill_raw_scores.jsonl"
        mismatch_pairwise_path = mismatch_dataset_dir / "llm_distill_pairwise.jsonl"

        build_stats = _build_mismatch_distill_files(
            rows=mismatch_rows,
            raw_out_path=mismatch_raw_path,
            pair_out_path=mismatch_pairwise_path,
            pair_derive_pos_k=pair_derive_pos_k,
            pair_derive_hard_k=pair_derive_hard_k,
            pair_derive_weak_k=pair_derive_weak_k,
            pair_derive_cap=pair_derive_cap,
            pair_add_mid_lower_mid=pair_add_mid_lower_mid,
            pair_mid_pos_score_min=pair_mid_pos_score_min,
            pair_mid_pos_score_max=pair_mid_pos_score_max,
            pair_mid_neg_score_min=pair_mid_neg_score_min,
            pair_mid_neg_score_max=pair_mid_neg_score_max,
            pair_mid_margin_min=pair_mid_margin_min,
            pair_mid_margin_max=pair_mid_margin_max,
            pair_mid_add_easy_contrast=pair_mid_add_easy_contrast,
            pair_mid_min_candidates=pair_mid_min_candidates,
            pair_mid_start_ratio=pair_mid_start_ratio,
            pair_mid_end_ratio=pair_mid_end_ratio,
            pair_lower_mid_start_ratio=pair_lower_mid_start_ratio,
            pair_lower_mid_end_ratio=pair_lower_mid_end_ratio,
        )
        round_summary["mismatch_build_stats"] = build_stats
        print(json.dumps({"iterative_mismatch_build": build_stats}, ensure_ascii=False))

        kept_groups = int(build_stats.get("raw_query_groups_kept", 0))
        if kept_groups <= 0:
            round_summary["stop_reason"] = "no_usable_mismatch_query_groups_after_conversion"
            round_summaries.append(round_summary)
            print(f"iterative_round_end={round_idx}/{iterative_max_rounds - 1} status={round_summary['stop_reason']}")
            if iterative_stop_on_empty:
                break
            continue

        mismatch_split_dir = (mismatch_dataset_dir / "splits").resolve()
        split_result = ensure_split_files(
            raw_input=mismatch_raw_path,
            pairwise_input=mismatch_pairwise_path,
            split_dir=mismatch_split_dir,
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
            seed=int(seed + round_idx),
            overwrite=True,
            high_threshold=float(mismatch_high_threshold),
            mid_threshold=float(mismatch_mid_threshold),
        )
        split_paths = split_result["paths"]
        round_summary["mismatch_split_dir"] = str(mismatch_split_dir)
        round_summary["mismatch_split_manifest"] = str(split_paths.manifest)

        mined_train_groups = _load_raw_groups(split_paths.raw_train, max_queries=0)
        mined_val_groups = _load_raw_groups(split_paths.raw_val, max_queries=0)
        if split_paths.pair_train.exists() and split_paths.pair_val.exists():
            mined_train_pairs = _load_pairwise_rows(split_paths.pair_train, max_rows=0)
            mined_val_pairs = _load_pairwise_rows(split_paths.pair_val, max_rows=0)
        else:
            mined_train_pairs = _derive_pairs_for_groups(
                mined_train_groups,
                pair_derive_pos_k=pair_derive_pos_k,
                pair_derive_hard_k=pair_derive_hard_k,
                pair_derive_weak_k=pair_derive_weak_k,
                pair_derive_cap=pair_derive_cap,
                pair_add_mid_lower_mid=pair_add_mid_lower_mid,
                pair_mid_pos_score_min=pair_mid_pos_score_min,
                pair_mid_pos_score_max=pair_mid_pos_score_max,
                pair_mid_neg_score_min=pair_mid_neg_score_min,
                pair_mid_neg_score_max=pair_mid_neg_score_max,
                pair_mid_margin_min=pair_mid_margin_min,
                pair_mid_margin_max=pair_mid_margin_max,
                pair_mid_add_easy_contrast=pair_mid_add_easy_contrast,
                pair_mid_min_candidates=pair_mid_min_candidates,
                pair_mid_start_ratio=pair_mid_start_ratio,
                pair_mid_end_ratio=pair_mid_end_ratio,
                pair_lower_mid_start_ratio=pair_lower_mid_start_ratio,
                pair_lower_mid_end_ratio=pair_lower_mid_end_ratio,
            )
            mined_val_pairs = _derive_pairs_for_groups(
                mined_val_groups,
                pair_derive_pos_k=pair_derive_pos_k,
                pair_derive_hard_k=pair_derive_hard_k,
                pair_derive_weak_k=pair_derive_weak_k,
                pair_derive_cap=pair_derive_cap,
                pair_add_mid_lower_mid=pair_add_mid_lower_mid,
                pair_mid_pos_score_min=pair_mid_pos_score_min,
                pair_mid_pos_score_max=pair_mid_pos_score_max,
                pair_mid_neg_score_min=pair_mid_neg_score_min,
                pair_mid_neg_score_max=pair_mid_neg_score_max,
                pair_mid_margin_min=pair_mid_margin_min,
                pair_mid_margin_max=pair_mid_margin_max,
                pair_mid_add_easy_contrast=pair_mid_add_easy_contrast,
                pair_mid_min_candidates=pair_mid_min_candidates,
                pair_mid_start_ratio=pair_mid_start_ratio,
                pair_mid_end_ratio=pair_mid_end_ratio,
                pair_lower_mid_start_ratio=pair_lower_mid_start_ratio,
                pair_lower_mid_end_ratio=pair_lower_mid_end_ratio,
            )

        mined_case_count = int(len(mined_train_groups))
        replay_target = int(round(float(mined_case_count) * float(replay_multiplier)))
        replay_target = max(int(replay_min_groups), replay_target)
        replay_cap = int(round(float(mined_case_count) * float(replay_max_multiplier)))
        if replay_cap > 0:
            replay_target = min(replay_target, replay_cap)
        replay_target = min(replay_target, int(len(original_train_groups)))

        replay_groups = _sample_replay_groups(
            original_train_groups=original_train_groups,
            replay_target=int(replay_target),
            high_threshold=float(mismatch_high_threshold),
            mid_threshold=float(mismatch_mid_threshold),
            high_ratio=float(replay_high_ratio),
            mid_ratio=float(replay_mid_ratio),
            low_ratio=float(replay_low_ratio),
            seed=int(seed + round_idx + 1),
        )
        replay_pairs = _derive_pairs_for_groups(
            replay_groups,
            pair_derive_pos_k=pair_derive_pos_k,
            pair_derive_hard_k=pair_derive_hard_k,
            pair_derive_weak_k=pair_derive_weak_k,
            pair_derive_cap=pair_derive_cap,
            pair_add_mid_lower_mid=pair_add_mid_lower_mid,
            pair_mid_pos_score_min=pair_mid_pos_score_min,
            pair_mid_pos_score_max=pair_mid_pos_score_max,
            pair_mid_neg_score_min=pair_mid_neg_score_min,
            pair_mid_neg_score_max=pair_mid_neg_score_max,
            pair_mid_margin_min=pair_mid_margin_min,
            pair_mid_margin_max=pair_mid_margin_max,
            pair_mid_add_easy_contrast=pair_mid_add_easy_contrast,
            pair_mid_min_candidates=pair_mid_min_candidates,
            pair_mid_start_ratio=pair_mid_start_ratio,
            pair_mid_end_ratio=pair_mid_end_ratio,
            pair_lower_mid_start_ratio=pair_lower_mid_start_ratio,
            pair_lower_mid_end_ratio=pair_lower_mid_end_ratio,
        )

        next_train_groups = _merge_groups_unique(mined_train_groups, replay_groups)
        next_val_groups = list(original_val_groups)
        next_test_groups = list(fixed_mining_dev_groups)

        next_train_pairs = _merge_pairs_unique(mined_train_pairs, replay_pairs)
        next_val_pairs = list(original_val_pairs)
        next_test_pairs = list(fixed_mining_dev_pairs)

        next_train_groups, next_train_low_cap_stats = _cap_low_band_groups(
            next_train_groups,
            high_threshold=float(train_band_high_threshold),
            mid_threshold=float(train_band_mid_threshold),
            low_cap_ratio_to_high=float(train_low_band_cap_ratio),
            seed=int(seed + round_idx + 11),
        )
        next_train_pairs_before_low_cap_filter = int(len(next_train_pairs))
        next_train_pairs = _filter_pairs_for_groups(next_train_pairs, next_train_groups)
        next_train_pairs_after_low_cap_filter = int(len(next_train_pairs))
        next_train_pairs_low_cap_filtered = (
            next_train_pairs_before_low_cap_filter - next_train_pairs_after_low_cap_filter
        )
        round_low_cap_before = next_train_low_cap_stats.get("before") if isinstance(next_train_low_cap_stats, dict) else {}
        round_low_cap_after = next_train_low_cap_stats.get("after") if isinstance(next_train_low_cap_stats, dict) else {}
        print(
            "round_train_low_cap="
            f"applied={bool(next_train_low_cap_stats.get('applied'))} "
            f"reason={_clean_text(next_train_low_cap_stats.get('reason')) or 'unknown'} "
            f"ratio={float(next_train_low_cap_stats.get('ratio', 0.0)):.4f} "
            f"target={int(next_train_low_cap_stats.get('low_cap_target', -1))} "
            f"before(high={int((round_low_cap_before or {}).get('high', 0))},"
            f"mid={int((round_low_cap_before or {}).get('mid', 0))},"
            f"low={int((round_low_cap_before or {}).get('low', 0))},"
            f"total={int((round_low_cap_before or {}).get('total', 0))}) "
            f"after(high={int((round_low_cap_after or {}).get('high', 0))},"
            f"mid={int((round_low_cap_after or {}).get('mid', 0))},"
            f"low={int((round_low_cap_after or {}).get('low', 0))},"
            f"total={int((round_low_cap_after or {}).get('total', 0))})"
        )
        print(
            "round_train_pairs_low_cap_filter="
            f"before={next_train_pairs_before_low_cap_filter} "
            f"after={next_train_pairs_after_low_cap_filter} "
            f"dropped={next_train_pairs_low_cap_filtered}"
        )

        if not next_train_groups:
            round_summary["stop_reason"] = "next_train_groups_empty_after_mining_replay"
            round_summaries.append(round_summary)
            print(f"iterative_round_end={round_idx}/{iterative_max_rounds - 1} status={round_summary['stop_reason']}")
            break
        if not next_train_pairs:
            next_train_pairs = _derive_pairs_for_groups(
                next_train_groups,
                pair_derive_pos_k=pair_derive_pos_k,
                pair_derive_hard_k=pair_derive_hard_k,
                pair_derive_weak_k=pair_derive_weak_k,
                pair_derive_cap=pair_derive_cap,
                pair_add_mid_lower_mid=pair_add_mid_lower_mid,
                pair_mid_pos_score_min=pair_mid_pos_score_min,
                pair_mid_pos_score_max=pair_mid_pos_score_max,
                pair_mid_neg_score_min=pair_mid_neg_score_min,
                pair_mid_neg_score_max=pair_mid_neg_score_max,
                pair_mid_margin_min=pair_mid_margin_min,
                pair_mid_margin_max=pair_mid_margin_max,
                pair_mid_add_easy_contrast=pair_mid_add_easy_contrast,
                pair_mid_min_candidates=pair_mid_min_candidates,
                pair_mid_start_ratio=pair_mid_start_ratio,
                pair_mid_end_ratio=pair_mid_end_ratio,
                pair_lower_mid_start_ratio=pair_lower_mid_start_ratio,
                pair_lower_mid_end_ratio=pair_lower_mid_end_ratio,
            )
        if not next_train_pairs:
            round_summary["stop_reason"] = "next_train_pairs_empty_after_mining_replay"
            round_summaries.append(round_summary)
            print(f"iterative_round_end={round_idx}/{iterative_max_rounds - 1} status={round_summary['stop_reason']}")
            break
        if not next_val_pairs:
            next_val_pairs = _derive_pairs_for_groups(
                next_val_groups,
                pair_derive_pos_k=pair_derive_pos_k,
                pair_derive_hard_k=pair_derive_hard_k,
                pair_derive_weak_k=pair_derive_weak_k,
                pair_derive_cap=pair_derive_cap,
                pair_add_mid_lower_mid=pair_add_mid_lower_mid,
                pair_mid_pos_score_min=pair_mid_pos_score_min,
                pair_mid_pos_score_max=pair_mid_pos_score_max,
                pair_mid_neg_score_min=pair_mid_neg_score_min,
                pair_mid_neg_score_max=pair_mid_neg_score_max,
                pair_mid_margin_min=pair_mid_margin_min,
                pair_mid_margin_max=pair_mid_margin_max,
                pair_mid_add_easy_contrast=pair_mid_add_easy_contrast,
                pair_mid_min_candidates=pair_mid_min_candidates,
                pair_mid_start_ratio=pair_mid_start_ratio,
                pair_mid_end_ratio=pair_mid_end_ratio,
                pair_lower_mid_start_ratio=pair_lower_mid_start_ratio,
                pair_lower_mid_end_ratio=pair_lower_mid_end_ratio,
            )
        if not next_test_pairs:
            next_test_pairs = _derive_pairs_for_groups(
                next_test_groups,
                pair_derive_pos_k=pair_derive_pos_k,
                pair_derive_hard_k=pair_derive_hard_k,
                pair_derive_weak_k=pair_derive_weak_k,
                pair_derive_cap=pair_derive_cap,
                pair_add_mid_lower_mid=pair_add_mid_lower_mid,
                pair_mid_pos_score_min=pair_mid_pos_score_min,
                pair_mid_pos_score_max=pair_mid_pos_score_max,
                pair_mid_neg_score_min=pair_mid_neg_score_min,
                pair_mid_neg_score_max=pair_mid_neg_score_max,
                pair_mid_margin_min=pair_mid_margin_min,
                pair_mid_margin_max=pair_mid_margin_max,
                pair_mid_add_easy_contrast=pair_mid_add_easy_contrast,
                pair_mid_min_candidates=pair_mid_min_candidates,
                pair_mid_start_ratio=pair_mid_start_ratio,
                pair_mid_end_ratio=pair_mid_end_ratio,
                pair_lower_mid_start_ratio=pair_lower_mid_start_ratio,
                pair_lower_mid_end_ratio=pair_lower_mid_end_ratio,
            )

        current_train_groups = next_train_groups
        current_val_groups = next_val_groups
        current_test_groups = next_test_groups
        current_train_pairs = next_train_pairs
        current_val_pairs = next_val_pairs
        current_test_pairs = next_test_pairs

        round_summary["replay_target"] = int(replay_target)
        round_summary["replay_groups"] = int(len(replay_groups))
        round_summary["mined_val_groups"] = int(len(mined_val_groups))
        round_summary["mined_val_pairs"] = int(len(mined_val_pairs))
        round_summary["next_train_low_cap"] = next_train_low_cap_stats
        round_summary["next_train_pairs_low_cap_filter"] = {
            "before": int(next_train_pairs_before_low_cap_filter),
            "after": int(next_train_pairs_after_low_cap_filter),
            "dropped": int(next_train_pairs_low_cap_filtered),
        }
        round_summary["next_train_groups"] = int(len(current_train_groups))
        round_summary["next_val_groups"] = int(len(current_val_groups))
        round_summary["next_test_groups"] = int(len(current_test_groups))
        round_summary["next_train_pairs"] = int(len(current_train_pairs))
        round_summary["next_val_pairs"] = int(len(current_val_pairs))
        round_summary["next_test_pairs"] = int(len(current_test_pairs))
        round_summary["next_round_model_id"] = str(current_model_id)
        round_summaries.append(round_summary)

        print(f"iterative_round_end={round_idx}/{iterative_max_rounds - 1} status=ok")

    final_test_metrics: Dict[str, Any] = {}
    if final_model_dir is not None and fixed_final_test_groups:
        if torch.cuda.is_available():
            eval_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            eval_device = torch.device("mps")
        else:
            eval_device = torch.device("cpu")
        eval_batch_size = _safe_int(args.eval_batch_size, default=32, minimum=1, maximum=4096)
        candidate_pool_size = _safe_int(args.candidate_pool_size, default=64, minimum=2, maximum=10_000)
        max_length = _safe_int(args.max_length, default=512, minimum=16, maximum=8192)
        mrr_rel_threshold = _safe_float(args.mrr_rel_threshold, default=0.7, minimum=0.0, maximum=1.0)
        recall_rel_threshold = _safe_float(args.recall_rel_threshold, default=0.7, minimum=0.0, maximum=1.0)

        tok_final = AutoTokenizer.from_pretrained(str(final_model_dir))
        model_final = AutoModelForSequenceClassification.from_pretrained(str(final_model_dir), num_labels=1)
        model_final.to(eval_device)
        final_test_metrics = _evaluate(
            model=model_final,
            tokenizer=tok_final,
            groups=fixed_final_test_groups,
            device=eval_device,
            max_length=max_length,
            eval_batch_size=eval_batch_size,
            candidate_pool_size=candidate_pool_size,
            mrr_rel_threshold=mrr_rel_threshold,
            recall_rel_threshold=recall_rel_threshold,
        )
        final_test_metrics["device"] = str(eval_device)
        print(json.dumps({"final_test_metrics": final_test_metrics}, ensure_ascii=False))

    iterative_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "iterative_max_rounds": int(iterative_max_rounds),
        "iterative_root": str(iterative_root),
        "fixed_mining_dev_input": str(mining_dev_raw_input),
        "fixed_final_test_input": str(final_test_raw_input),
        "original_raw_input": str(original_raw_input),
        "original_pairwise_input": str(original_pairwise_input),
        "original_split_dir": str(original_paths.split_dir),
        "eval_base_model": str(eval_base_model),
        "mismatch_target": str(mismatch_target),
        "mismatch_ground_truth": str(mismatch_ground_truth),
        "mismatch_thresholds": {
            "teacher_high": float(mismatch_high_threshold),
            "teacher_mid": float(mismatch_mid_threshold),
            "pred_high": float(mismatch_pred_high_threshold),
            "pred_mid": float(mismatch_pred_mid_threshold),
        },
        "mismatch_sampling": {
            "sample_high": int(mismatch_sample_high),
            "sample_mid": int(mismatch_sample_mid),
            "sample_low": int(mismatch_sample_low),
            "max_pairs_scan": int(mismatch_max_pairs_scan),
            "min_rows": int(mismatch_min_rows),
            "max_rows": int(mismatch_max_rows),
        },
        "mined_filter_policy": {
            "gap_min": float(mined_gap_min),
            "boundary_bandwidth": float(mined_boundary_bandwidth),
            "low_max_ratio": float(mined_low_max_ratio),
        },
        "train_low_cap_policy": {
            "train_band_high_threshold": float(train_band_high_threshold),
            "train_band_mid_threshold": float(train_band_mid_threshold),
            "train_low_band_cap_ratio": float(train_low_band_cap_ratio),
            "original_train_low_cap": original_train_low_cap_stats,
            "original_train_pairs_low_cap_filter": {
                "before": int(original_train_pairs_before_low_cap_filter),
                "after": int(original_train_pairs_after_low_cap_filter),
                "dropped": int(original_train_pairs_low_cap_filtered),
            },
        },
        "replay_policy": {
            "replay_multiplier": float(replay_multiplier),
            "replay_min_groups": int(replay_min_groups),
            "replay_max_multiplier": float(replay_max_multiplier),
            "replay_high_ratio": float(replay_high_ratio),
            "replay_mid_ratio": float(replay_mid_ratio),
            "replay_low_ratio": float(replay_low_ratio),
        },
        "stop_policy": {
            "mining_dev_no_improve_patience": int(iterative_stop_patience),
        },
        "rounds": round_summaries,
        "final_model_dir": str(final_model_dir) if final_model_dir is not None else "",
        "best_mining_dev_ndcg@10": float(best_mining_dev_ndcg),
        "final_test_metrics": final_test_metrics,
    }
    iterative_manifest_path = iterative_root / "iterative_manifest.json"
    iterative_manifest_path.write_text(json.dumps(iterative_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("iterative_done=true")
    print(f"iterative_rounds_completed={len(round_summaries)}")
    if final_model_dir is not None:
        print(f"iterative_final_model_dir={final_model_dir}")
    print(f"iterative_manifest={iterative_manifest_path}")
    return 0


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

    p.add_argument("--candidate-pool-size", type=int, default=0, help="Candidate pool per query before mini-list sampling (0=all docs).")
    p.add_argument("--mini-list-size", type=int, default=0, help="Mini-list docs per query for listwise loss (0=all docs in pool).")

    p.add_argument("--boundary-center", type=float, default=0.6, help="Boundary score center for informative mids.")
    p.add_argument("--boundary-bandwidth", type=float, default=0.12, help="Boundary bandwidth around center.")
    p.add_argument("--top-ratio", type=float, default=0.4)
    p.add_argument("--boundary-ratio", type=float, default=0.4)
    p.add_argument("--random-ratio", type=float, default=0.2)

    p.add_argument("--low-signal-max-score-threshold", type=float, default=0.3)
    p.add_argument("--low-signal-std-threshold", type=float, default=0.05)
    p.add_argument("--low-signal-keep-prob", type=float, default=1.0)

    p.add_argument("--teacher-temperature", type=float, default=2.0)

    p.add_argument("--stage1-epochs", type=int, default=1)
    p.add_argument("--stage2-epochs", type=int, default=3)
    p.add_argument(
        "--stage1-early-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Stage1 early stopping using val_pair_loss (epoch-level patience).",
    )
    p.add_argument(
        "--stage1-early-stop-patience",
        type=int,
        default=1,
        help="Stage1 early-stop patience in epochs without val_pair_loss improvement.",
    )
    p.add_argument(
        "--stage2-start-from-best-stage1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Before Stage2, reload checkpoint from best_stage1_val_pair_loss if available.",
    )
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
        "--train-band-high-threshold",
        type=float,
        default=TRAIN_BAND_HIGH_THRESHOLD_DEFAULT,
        help="Teacher-score threshold used to classify query groups as high.",
    )
    p.add_argument(
        "--train-band-mid-threshold",
        type=float,
        default=TRAIN_BAND_MID_THRESHOLD_DEFAULT,
        help="Teacher-score threshold used to classify query groups as mid (below high is low).",
    )
    p.add_argument(
        "--train-low-band-cap-ratio",
        type=float,
        default=0.0,
        help=(
            "Cap low-band train query groups to <= high_count * ratio. "
            "Set 0 to disable. Default 0.0 keeps all low-band groups."
        ),
    )
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
    p.add_argument(
        "--pair-mid-min-candidates",
        type=int,
        default=PAIR_MID_MIN_CANDIDATES_DEFAULT,
        help="Minimum candidates required in a query list before mid>lower-mid augmentation is attempted.",
    )
    p.add_argument(
        "--pair-mid-start-ratio",
        type=float,
        default=PAIR_MID_START_RATIO_DEFAULT,
        help="Start ratio for mid slice (index ~= ratio * n).",
    )
    p.add_argument(
        "--pair-mid-end-ratio",
        type=float,
        default=PAIR_MID_END_RATIO_DEFAULT,
        help="End ratio for mid slice (exclusive index ~= ratio * n).",
    )
    p.add_argument(
        "--pair-lower-mid-start-ratio",
        type=float,
        default=PAIR_LOWER_MID_START_RATIO_DEFAULT,
        help="Start ratio for lower-mid slice (index ~= ratio * n).",
    )
    p.add_argument(
        "--pair-lower-mid-end-ratio",
        type=float,
        default=PAIR_LOWER_MID_END_RATIO_DEFAULT,
        help="End ratio for lower-mid slice (exclusive index ~= ratio * n).",
    )
    p.add_argument(
        "--pair-type-max-share",
        type=float,
        default=PAIR_TYPE_MAX_SHARE_DEFAULT,
        help=(
            "Train split only: cap each pair_type to at most this share of total train pairs. "
            "Set <=0 or >=1 to disable."
        ),
    )
    p.add_argument(
        "--pair-type-weight-map",
        type=str,
        default=PAIR_TYPE_WEIGHT_MAP_DEFAULT,
        help="Comma map: pair_type=weight (plus optional default=...). Example: llm_disagreement=1.15,strong_vs_weak=0.95,default=1.0",
    )
    p.add_argument(
        "--list-augmented-doc-weight",
        type=float,
        default=LIST_AUGMENTED_DOC_WEIGHT_DEFAULT,
        help="Listwise loss per-doc weight for synthetic/augmented docs (real docs use 1.0).",
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

    p.add_argument(
        "--iterative-max-rounds",
        type=int,
        default=1,
        help=(
            "Iterative hard-case rounds. 1 = current single run behavior. "
            "If >1, each round trains stage1+stage2, mines cluster mismatches on mining-dev split, "
            "rebuilds mismatch distill splits, then trains the next round."
        ),
    )
    p.add_argument(
        "--iterative-eval-test-input",
        type=str,
        default="",
        help="Deprecated alias for --iterative-mining-input.",
    )
    p.add_argument(
        "--iterative-mining-input",
        type=str,
        default="",
        help="Raw JSONL used for hard-case mining each round (mining-dev role).",
    )
    p.add_argument(
        "--iterative-final-test-input",
        type=str,
        default="",
        help="Final held-out raw JSONL. Never used for mining or round-wise stopping.",
    )
    p.add_argument(
        "--iterative-eval-base-model",
        type=str,
        default="",
        help="Base model id/path for mismatch eval comparison (default: --model-id).",
    )
    p.add_argument(
        "--iterative-mismatch-target",
        type=str,
        default="finetuned",
        choices=["finetuned", "base", "either", "both"],
        help="Mismatch condition to mine from eval output.",
    )
    p.add_argument(
        "--iterative-mismatch-ground-truth",
        type=str,
        default="raw",
        choices=["normalized", "raw"],
        help="Teacher score mode used in mismatch eval.",
    )
    p.add_argument("--iterative-mismatch-high-threshold", type=float, default=TRAIN_BAND_HIGH_THRESHOLD_DEFAULT)
    p.add_argument("--iterative-mismatch-mid-threshold", type=float, default=TRAIN_BAND_MID_THRESHOLD_DEFAULT)
    p.add_argument(
        "--iterative-mismatch-pred-high-threshold",
        type=float,
        default=-1.0,
        help="Prediction high threshold for mismatch eval (-1 uses teacher high threshold).",
    )
    p.add_argument(
        "--iterative-mismatch-pred-mid-threshold",
        type=float,
        default=-1.0,
        help="Prediction mid threshold for mismatch eval (-1 uses teacher mid threshold).",
    )
    p.add_argument("--iterative-mismatch-sample-high", type=int, default=0)
    p.add_argument("--iterative-mismatch-sample-mid", type=int, default=0)
    p.add_argument("--iterative-mismatch-sample-low", type=int, default=0)
    p.add_argument("--iterative-mismatch-max-pairs-scan", type=int, default=0)
    p.add_argument(
        "--iterative-mismatch-min-rows",
        type=int,
        default=1,
        help="Stop iterative loop when mined mismatch rows are below this count.",
    )
    p.add_argument(
        "--iterative-mismatch-max-rows",
        type=int,
        default=0,
        help="Optional cap on mined mismatch rows used to build next-round dataset (0=all).",
    )
    p.add_argument(
        "--iterative-stop-on-empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop loop early if mismatch mining yields no usable rows.",
    )
    p.add_argument(
        "--iterative-mined-gap-min",
        type=float,
        default=0.15,
        help="Minimum |teacher-student| gap used when selecting mined hard cases.",
    )
    p.add_argument(
        "--iterative-mined-boundary-bandwidth",
        type=float,
        default=0.08,
        help="Boundary proximity width around high/mid thresholds for mined-case prioritization.",
    )
    p.add_argument(
        "--iterative-mined-low-max-ratio",
        type=float,
        default=0.35,
        help="Max share of low-band cases in mined hard set.",
    )
    p.add_argument(
        "--iterative-replay-multiplier",
        type=float,
        default=0.25,
        help="Replay size as multiplier of mined-train query count (0.25 ~= 20% replay in mixed train).",
    )
    p.add_argument(
        "--iterative-replay-min-groups",
        type=int,
        default=0,
        help="Minimum replay query groups each iterative round.",
    )
    p.add_argument(
        "--iterative-replay-max-multiplier",
        type=float,
        default=0.5,
        help="Upper cap for replay size relative to mined-train query count.",
    )
    p.add_argument("--iterative-replay-high-ratio", type=float, default=0.3)
    p.add_argument("--iterative-replay-mid-ratio", type=float, default=0.4)
    p.add_argument("--iterative-replay-low-ratio", type=float, default=0.3)
    p.add_argument(
        "--iterative-stop-patience",
        type=int,
        default=2,
        help="Early stop rounds without improvement on fixed held-out test ndcg@10.",
    )
    p.add_argument("--iterative-inner-run", action="store_true", help=argparse.SUPPRESS)
    return p


def main() -> int:
    args = _build_parser().parse_args()

    iterative_max_rounds = _safe_int(args.iterative_max_rounds, default=1, minimum=1, maximum=100)
    if iterative_max_rounds > 1 and (not bool(args.iterative_inner_run)):
        return _run_iterative_orchestration(args, raw_argv=sys.argv[1:])

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

    candidate_pool_size = _safe_int(args.candidate_pool_size, default=0, minimum=0, maximum=100_000)
    mini_list_size = _safe_int(args.mini_list_size, default=0, minimum=0, maximum=100_000)

    stage1_epochs = _safe_int(args.stage1_epochs, default=1, minimum=0, maximum=100)
    stage2_epochs = _safe_int(args.stage2_epochs, default=3, minimum=0, maximum=100)
    stage1_early_stop = bool(args.stage1_early_stop)
    stage1_early_stop_patience = _safe_int(args.stage1_early_stop_patience, default=1, minimum=0, maximum=1000)
    stage2_start_from_best_stage1 = bool(args.stage2_start_from_best_stage1)
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
    low_signal_keep_prob = _safe_float(args.low_signal_keep_prob, default=1.0, minimum=0.0, maximum=1.0)
    train_band_high_threshold = _safe_float(
        args.train_band_high_threshold,
        default=TRAIN_BAND_HIGH_THRESHOLD_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    train_band_mid_threshold = _safe_float(
        args.train_band_mid_threshold,
        default=TRAIN_BAND_MID_THRESHOLD_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    if train_band_mid_threshold > train_band_high_threshold:
        train_band_mid_threshold = train_band_high_threshold
    train_low_band_cap_ratio = _safe_float(args.train_low_band_cap_ratio, default=0.0, minimum=0.0, maximum=100.0)
    pair_type_max_share = _safe_float(
        args.pair_type_max_share,
        default=PAIR_TYPE_MAX_SHARE_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    pair_type_weights, pair_default_weight = _parse_pair_type_weight_map(args.pair_type_weight_map)
    list_augmented_doc_weight = _safe_float(
        args.list_augmented_doc_weight,
        default=LIST_AUGMENTED_DOC_WEIGHT_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    pair_mid_pos_score_min = _safe_float(args.pair_mid_pos_score_min, default=0.4, minimum=0.0, maximum=1.0)
    pair_mid_pos_score_max = _safe_float(args.pair_mid_pos_score_max, default=0.7, minimum=0.0, maximum=1.0)
    pair_mid_neg_score_min = _safe_float(args.pair_mid_neg_score_min, default=0.2, minimum=0.0, maximum=1.0)
    pair_mid_neg_score_max = _safe_float(args.pair_mid_neg_score_max, default=0.5, minimum=0.0, maximum=1.0)
    pair_mid_margin_min = _safe_float(args.pair_mid_margin_min, default=0.05, minimum=0.0, maximum=1.0)
    pair_mid_margin_max = _safe_float(args.pair_mid_margin_max, default=0.4, minimum=0.0, maximum=1.0)
    pair_mid_add_easy_contrast = bool(args.pair_mid_add_easy_contrast)
    pair_mid_min_candidates = _safe_int(
        args.pair_mid_min_candidates,
        default=PAIR_MID_MIN_CANDIDATES_DEFAULT,
        minimum=2,
        maximum=100_000,
    )
    pair_mid_start_ratio = _safe_float(
        args.pair_mid_start_ratio,
        default=PAIR_MID_START_RATIO_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    pair_mid_end_ratio = _safe_float(
        args.pair_mid_end_ratio,
        default=PAIR_MID_END_RATIO_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    pair_lower_mid_start_ratio = _safe_float(
        args.pair_lower_mid_start_ratio,
        default=PAIR_LOWER_MID_START_RATIO_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    pair_lower_mid_end_ratio = _safe_float(
        args.pair_lower_mid_end_ratio,
        default=PAIR_LOWER_MID_END_RATIO_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )

    if pair_mid_pos_score_max < pair_mid_pos_score_min:
        pair_mid_pos_score_max = pair_mid_pos_score_min
    if pair_mid_neg_score_max < pair_mid_neg_score_min:
        pair_mid_neg_score_max = pair_mid_neg_score_min
    if pair_mid_margin_max < pair_mid_margin_min:
        pair_mid_margin_max = pair_mid_margin_min
    if pair_mid_end_ratio < pair_mid_start_ratio:
        pair_mid_end_ratio = pair_mid_start_ratio
    if pair_lower_mid_end_ratio < pair_lower_mid_start_ratio:
        pair_lower_mid_end_ratio = pair_lower_mid_start_ratio

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
            ensure_split_files = _resolve_split_helper()
            try:
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
            except Exception as split_exc:
                # Keep training runnable with newer raw schemas by falling back
                # to in-script deterministic grant-id split.
                print(
                    "prepared_split_fallback=true "
                    f"reason={type(split_exc).__name__}:{split_exc}"
                )
                use_prepared_splits = False
                split_policy = "deterministic_hash_by_grant_id"

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
    print(
        "stage1_control="
        f"early_stop:{stage1_early_stop},"
        f"patience:{stage1_early_stop_patience},"
        f"stage2_start_from_best_stage1:{stage2_start_from_best_stage1}"
    )
    print(f"train_band_thresholds=high:{train_band_high_threshold:.2f},mid:{train_band_mid_threshold:.2f}")
    print(
        "pair_type_weighting="
        f"default:{pair_default_weight:.4f} "
        f"map:{json.dumps(pair_type_weights, ensure_ascii=False)}"
    )
    print(f"list_augmented_doc_weight={list_augmented_doc_weight:.4f}")
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

    train_groups, train_low_cap_stats = _cap_low_band_groups(
        train_groups,
        high_threshold=float(train_band_high_threshold),
        mid_threshold=float(train_band_mid_threshold),
        low_cap_ratio_to_high=float(train_low_band_cap_ratio),
        seed=int(seed),
    )
    if not train_groups:
        raise RuntimeError("Training split became empty after low-band cap.")
    low_cap_before = train_low_cap_stats.get("before") if isinstance(train_low_cap_stats, dict) else {}
    low_cap_after = train_low_cap_stats.get("after") if isinstance(train_low_cap_stats, dict) else {}
    print(
        "train_low_band_cap="
        f"applied={bool(train_low_cap_stats.get('applied'))} "
        f"reason={_clean_text(train_low_cap_stats.get('reason')) or 'unknown'} "
        f"ratio={float(train_low_cap_stats.get('ratio', 0.0)):.4f} "
        f"target={int(train_low_cap_stats.get('low_cap_target', -1))} "
        f"before(high={int((low_cap_before or {}).get('high', 0))},"
        f"mid={int((low_cap_before or {}).get('mid', 0))},"
        f"low={int((low_cap_before or {}).get('low', 0))},"
        f"total={int((low_cap_before or {}).get('total', 0))}) "
        f"after(high={int((low_cap_after or {}).get('high', 0))},"
        f"mid={int((low_cap_after or {}).get('mid', 0))},"
        f"low={int((low_cap_after or {}).get('low', 0))},"
        f"total={int((low_cap_after or {}).get('total', 0))})"
    )
    groups_all = list(train_groups) + list(val_groups) + list(test_groups)

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

    train_pairs_before_low_cap_filter = int(len(train_pairs))
    train_pairs = _filter_pairs_for_groups(train_pairs, train_groups)
    train_pairs_after_low_cap_filter = int(len(train_pairs))
    train_pairs_low_cap_filtered = train_pairs_before_low_cap_filter - train_pairs_after_low_cap_filter
    if train_pairs_low_cap_filtered > 0:
        print(
            "train_pairs_low_cap_filter="
            f"before={train_pairs_before_low_cap_filter} "
            f"after={train_pairs_after_low_cap_filter} "
            f"dropped={train_pairs_low_cap_filtered}"
        )

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
            mid_min_candidates=pair_mid_min_candidates,
            mid_start_ratio=pair_mid_start_ratio,
            mid_end_ratio=pair_mid_end_ratio,
            lower_mid_start_ratio=pair_lower_mid_start_ratio,
            lower_mid_end_ratio=pair_lower_mid_end_ratio,
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
            mid_min_candidates=pair_mid_min_candidates,
            mid_start_ratio=pair_mid_start_ratio,
            mid_end_ratio=pair_mid_end_ratio,
            lower_mid_start_ratio=pair_lower_mid_start_ratio,
            lower_mid_end_ratio=pair_lower_mid_end_ratio,
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
            mid_min_candidates=pair_mid_min_candidates,
            mid_start_ratio=pair_mid_start_ratio,
            mid_end_ratio=pair_mid_end_ratio,
            lower_mid_start_ratio=pair_lower_mid_start_ratio,
            lower_mid_end_ratio=pair_lower_mid_end_ratio,
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
            f"min_candidates={pair_mid_min_candidates} "
            f"mid_ratio:[{pair_mid_start_ratio:.3f},{pair_mid_end_ratio:.3f}] "
            f"lower_mid_ratio:[{pair_lower_mid_start_ratio:.3f},{pair_lower_mid_end_ratio:.3f}] "
            f"easy_contrast={pair_mid_add_easy_contrast}"
        )
    else:
        print("pair_mid_lower_enabled=false")

    train_pairs, pair_type_cap_stats = _cap_pair_rows_by_type(
        train_pairs,
        max_share=float(pair_type_max_share),
        seed=int(seed + 1111),
    )
    print(
        "pair_type_cap="
        f"applied={bool((pair_type_cap_stats or {}).get('applied'))} "
        f"reason={_clean_text((pair_type_cap_stats or {}).get('reason')) or 'unknown'} "
        f"max_share={float((pair_type_cap_stats or {}).get('max_share', 0.0)):.4f} "
        f"before={int((pair_type_cap_stats or {}).get('before_total', 0))} "
        f"after={int((pair_type_cap_stats or {}).get('after_total', 0))} "
        f"dropped={int((pair_type_cap_stats or {}).get('dropped', 0))}"
    )

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
        low_signal_keep_prob=1.0,  # keep all test rows for stable final eval.
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
            "dataset/queries_total": int(len(groups_all)),
            "dataset/queries_train": int(len(train_groups)),
            "dataset/queries_val": int(len(val_groups)),
            "dataset/queries_test": int(len(test_groups)),
            "dataset/pairwise_train": int(len(train_pairs)),
            "dataset/pairwise_val": int(len(val_pairs)),
            "dataset/pairwise_test": int(len(test_pairs)),
            "dataset/pairwise_total": int(len(pair_rows)),
            "dataset/pair_mid_lower_enabled": int(1 if pair_mid_lower_enabled else 0),
            "dataset/pair_mid_lower_added": int(pair_mid_lower_added),
            "dataset/pair_mid_pos_score_min": float(pair_mid_pos_score_min),
            "dataset/pair_mid_pos_score_max": float(pair_mid_pos_score_max),
            "dataset/pair_mid_neg_score_min": float(pair_mid_neg_score_min),
            "dataset/pair_mid_neg_score_max": float(pair_mid_neg_score_max),
            "dataset/pair_mid_margin_min": float(pair_mid_margin_min),
            "dataset/pair_mid_margin_max": float(pair_mid_margin_max),
            "dataset/pair_mid_add_easy_contrast": int(1 if pair_mid_add_easy_contrast else 0),
            "dataset/pair_mid_min_candidates": int(pair_mid_min_candidates),
            "dataset/pair_mid_start_ratio": float(pair_mid_start_ratio),
            "dataset/pair_mid_end_ratio": float(pair_mid_end_ratio),
            "dataset/pair_lower_mid_start_ratio": float(pair_lower_mid_start_ratio),
            "dataset/pair_lower_mid_end_ratio": float(pair_lower_mid_end_ratio),
            "dataset/pair_type_max_share": float(pair_type_max_share),
            "dataset/pair_type_cap_applied": int(1 if bool((pair_type_cap_stats or {}).get("applied")) else 0),
            "dataset/pair_type_cap_dropped": int((pair_type_cap_stats or {}).get("dropped", 0)),
            "dataset/list_augmented_doc_weight": float(list_augmented_doc_weight),
            "dataset/train_low_cap_applied": int(1 if bool(train_low_cap_stats.get("applied")) else 0),
            "dataset/train_low_cap_ratio": float(train_low_cap_stats.get("ratio", 0.0)),
            "dataset/train_low_cap_target": int(train_low_cap_stats.get("low_cap_target", -1)),
            "dataset/train_high_before_low_cap": int((low_cap_before or {}).get("high", 0)),
            "dataset/train_mid_before_low_cap": int((low_cap_before or {}).get("mid", 0)),
            "dataset/train_low_before_low_cap": int((low_cap_before or {}).get("low", 0)),
            "dataset/train_total_before_low_cap": int((low_cap_before or {}).get("total", 0)),
            "dataset/train_high_after_low_cap": int((low_cap_after or {}).get("high", 0)),
            "dataset/train_mid_after_low_cap": int((low_cap_after or {}).get("mid", 0)),
            "dataset/train_low_after_low_cap": int((low_cap_after or {}).get("low", 0)),
            "dataset/train_total_after_low_cap": int((low_cap_after or {}).get("total", 0)),
            "dataset/train_pairs_before_low_cap_filter": int(train_pairs_before_low_cap_filter),
            "dataset/train_pairs_after_low_cap_filter": int(train_pairs_after_low_cap_filter),
            "dataset/train_pairs_low_cap_filtered": int(train_pairs_low_cap_filtered),
            "dataset/listwise_train": int(len(train_list_rows)),
            "dataset/listwise_val": int(len(val_list_rows)),
            "dataset/listwise_test": int(len(test_list_rows)),
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
    pair_collator = PairCollator(
        tokenizer,
        max_length=max_length,
        pair_type_weights=pair_type_weights,
        default_pair_weight=float(pair_default_weight),
    )
    list_collator = ListCollator(
        tokenizer,
        max_length=max_length,
        augmented_doc_weight=float(list_augmented_doc_weight),
    )

    pair_train_loader = DataLoader(
        PairwiseDataset(train_pairs),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=_safe_int(args.num_workers, default=0, minimum=0, maximum=32),
        collate_fn=pair_collator,
        drop_last=False,
    )
    pair_val_loader = DataLoader(
        PairwiseDataset(val_pairs),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=_safe_int(args.num_workers, default=0, minimum=0, maximum=32),
        collate_fn=pair_collator,
        drop_last=False,
    )
    pair_test_loader = DataLoader(
        PairwiseDataset(test_pairs),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=_safe_int(args.num_workers, default=0, minimum=0, maximum=32),
        collate_fn=pair_collator,
        drop_last=False,
    )
    val_list_loader: Optional[DataLoader] = None
    if val_list_rows:
        val_list_loader = DataLoader(
            ListwiseDataset(val_groups, val_list_rows),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=_safe_int(args.num_workers, default=0, minimum=0, maximum=32),
            collate_fn=list_collator,
            drop_last=False,
        )
    test_list_loader: Optional[DataLoader] = None
    if test_list_rows:
        test_list_loader = DataLoader(
            ListwiseDataset(test_groups, test_list_rows),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=_safe_int(args.num_workers, default=0, minimum=0, maximum=32),
            collate_fn=list_collator,
            drop_last=False,
        )

    best_ndcg = -1.0
    best_stage1_val_pair_loss = float("inf")
    best_stage1_val_pair_loss_step = -1
    best_stage1_val_pair_loss_epoch = -1
    best_stage1_val_pair_loss_source = ""
    best_val_total_loss = float("inf")
    best_val_total_loss_step = -1
    best_val_total_loss_epoch = -1
    best_val_total_loss_source = ""
    best_val_total_loss_components: Dict[str, float] = {
        "val_kl_loss": 0.0,
        "val_pair_loss": 0.0,
        "val_mse_loss": 0.0,
    }
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
                pair_weights = batch.get("pair_weights")
                if isinstance(pair_weights, torch.Tensor):
                    pair_weights = pair_weights.to(device)
                else:
                    pair_weights = None

                pos_logits = model(**pos).logits.squeeze(-1)
                neg_logits = model(**neg).logits.squeeze(-1)
                loss = _variable_margin_loss(pos_logits, neg_logits, margins, pair_weights)
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
                doc_weights = batch.get("doc_weights")
                if isinstance(doc_weights, torch.Tensor):
                    doc_weights = doc_weights.to(device)
                else:
                    doc_weights = None
                list_sizes = batch["list_sizes"]
                logits_flat = model(**enc).logits.squeeze(-1)
                kl_loss, mse_loss = _compute_listwise_kl_and_mse(
                    logits_flat=logits_flat,
                    teacher_scores_flat=teacher_scores,
                    doc_weights_flat=doc_weights,
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

    def maybe_save_best_val_total_loss(
        *,
        epoch: int,
        global_step_now: int,
        val_total_loss: float,
        val_kl_loss: float,
        val_pair_loss: float,
        val_mse_loss: float,
        source: str,
    ) -> None:
        nonlocal best_val_total_loss
        nonlocal best_val_total_loss_step
        nonlocal best_val_total_loss_epoch
        nonlocal best_val_total_loss_source
        nonlocal best_val_total_loss_components

        current = float(val_total_loss)
        if not math.isfinite(current):
            return
        if current >= float(best_val_total_loss):
            return

        best_val_total_loss = float(current)
        best_val_total_loss_step = int(global_step_now)
        best_val_total_loss_epoch = int(epoch)
        best_val_total_loss_source = _clean_text(source) or "unknown"
        best_val_total_loss_components = {
            "val_kl_loss": float(val_kl_loss),
            "val_pair_loss": float(val_pair_loss),
            "val_mse_loss": float(val_mse_loss),
        }

        best_loss_dir = output_dir / "best_val_loss"
        best_loss_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(best_loss_dir)
        tokenizer.save_pretrained(best_loss_dir)
        (best_loss_dir / "best_val_loss_meta.json").write_text(
            json.dumps(
                {
                    "best_val_total_loss": float(best_val_total_loss),
                    "epoch": int(best_val_total_loss_epoch),
                    "global_step": int(best_val_total_loss_step),
                    "source": str(best_val_total_loss_source),
                    "components": dict(best_val_total_loss_components),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "best_val_total_loss_update="
            f"{best_val_total_loss:.6f} "
            f"epoch={best_val_total_loss_epoch} "
            f"step={best_val_total_loss_step} "
            f"source={best_val_total_loss_source}"
        )

    def maybe_save_best_stage1_val_pair_loss(
        *,
        epoch: int,
        global_step_now: int,
        val_pair_loss: float,
        source: str,
    ) -> None:
        nonlocal best_stage1_val_pair_loss
        nonlocal best_stage1_val_pair_loss_step
        nonlocal best_stage1_val_pair_loss_epoch
        nonlocal best_stage1_val_pair_loss_source

        current = float(val_pair_loss)
        if not math.isfinite(current):
            return
        if current >= float(best_stage1_val_pair_loss):
            return

        best_stage1_val_pair_loss = float(current)
        best_stage1_val_pair_loss_step = int(global_step_now)
        best_stage1_val_pair_loss_epoch = int(epoch)
        best_stage1_val_pair_loss_source = _clean_text(source) or "unknown"

        best_stage1_dir = output_dir / "best_stage1_val_pair_loss"
        best_stage1_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(best_stage1_dir)
        tokenizer.save_pretrained(best_stage1_dir)
        (best_stage1_dir / "best_stage1_val_pair_loss_meta.json").write_text(
            json.dumps(
                {
                    "best_stage1_val_pair_loss": float(best_stage1_val_pair_loss),
                    "epoch": int(best_stage1_val_pair_loss_epoch),
                    "global_step": int(best_stage1_val_pair_loss_step),
                    "source": str(best_stage1_val_pair_loss_source),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "best_stage1_val_pair_loss_update="
            f"{best_stage1_val_pair_loss:.6f} "
            f"epoch={best_stage1_val_pair_loss_epoch} "
            f"step={best_stage1_val_pair_loss_step} "
            f"source={best_stage1_val_pair_loss_source}"
        )

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
            maybe_save_best_stage1_val_pair_loss(
                epoch=int(epoch),
                global_step_now=int(global_step_now),
                val_pair_loss=float(val_pair_loss),
                source="step_eval",
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
        maybe_save_best_val_total_loss(
            epoch=int(epoch),
            global_step_now=int(global_step_now),
            val_total_loss=float(val_total_loss),
            val_kl_loss=float(val_kl_loss),
            val_pair_loss=float(val_pair_loss),
            val_mse_loss=float(val_mse_loss),
            source="step_eval",
        )

    # Stage 1: pairwise warm start.
    if stage1_epochs > 0:
        print(f"stage1_start=true epochs={stage1_epochs}")
        stage1_no_improve_epochs = 0
        stage1_best_epoch_val_pair_loss = float("inf")
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
                pair_weights = batch.get("pair_weights")
                if isinstance(pair_weights, torch.Tensor):
                    pair_weights = pair_weights.to(device)
                else:
                    pair_weights = None

                amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
                with amp_ctx:
                    pos_logits = model(**pos).logits.squeeze(-1)
                    neg_logits = model(**neg).logits.squeeze(-1)
                    loss = _variable_margin_loss(pos_logits, neg_logits, margins, pair_weights)
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
            maybe_save_best_stage1_val_pair_loss(
                epoch=int(epoch),
                global_step_now=int(global_step),
                val_pair_loss=float(val_pair_loss),
                source="epoch_eval",
            )

            stage1_dir = output_dir / f"stage1_epoch_{epoch}"
            stage1_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(stage1_dir)
            tokenizer.save_pretrained(stage1_dir)

            improved_epoch = float(val_pair_loss) < float(stage1_best_epoch_val_pair_loss)
            if improved_epoch:
                stage1_best_epoch_val_pair_loss = float(val_pair_loss)
                stage1_no_improve_epochs = 0
            else:
                stage1_no_improve_epochs += 1

            if bool(stage1_early_stop) and int(stage1_no_improve_epochs) >= int(stage1_early_stop_patience):
                print(
                    "stage1_early_stop=true "
                    f"epoch={epoch} "
                    f"no_improve_epochs={stage1_no_improve_epochs} "
                    f"patience={stage1_early_stop_patience} "
                    f"best_epoch_val_pair_loss={stage1_best_epoch_val_pair_loss:.6f}"
                )
                break

    # Stage 2: listwise KD + pairwise + MSE.
    if stage2_epochs > 0:
        if bool(stage2_start_from_best_stage1) and int(stage1_epochs) > 0:
            best_stage1_dir = output_dir / "best_stage1_val_pair_loss"
            if best_stage1_dir.exists():
                print(f"stage2_init_from_best_stage1=true ckpt={best_stage1_dir}")
                model = AutoModelForSequenceClassification.from_pretrained(str(best_stage1_dir), num_labels=1).to(device)
                optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            else:
                print(
                    "stage2_init_from_best_stage1=false "
                    f"reason=missing_checkpoint path={best_stage1_dir}"
                )
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
                collate_fn=list_collator,
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
                doc_weights = list_batch.get("doc_weights")
                if isinstance(doc_weights, torch.Tensor):
                    doc_weights = doc_weights.to(device)
                else:
                    doc_weights = None
                list_sizes = list_batch["list_sizes"]

                pair_batch = next(pair_cycle)
                pos = _to_device(pair_batch["pos"], device)
                neg = _to_device(pair_batch["neg"], device)
                margins = pair_batch["margins"].to(device).clamp(min=margin_min, max=margin_max)
                pair_weights = pair_batch.get("pair_weights")
                if isinstance(pair_weights, torch.Tensor):
                    pair_weights = pair_weights.to(device)
                else:
                    pair_weights = None

                amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
                with amp_ctx:
                    list_logits_flat = model(**list_enc).logits.squeeze(-1)
                    kl_loss, mse_loss = _compute_listwise_kl_and_mse(
                        logits_flat=list_logits_flat,
                        teacher_scores_flat=teacher_scores,
                        doc_weights_flat=doc_weights,
                        list_sizes=list_sizes,
                        temperature=teacher_temperature,
                    )

                    pos_logits = model(**pos).logits.squeeze(-1)
                    neg_logits = model(**neg).logits.squeeze(-1)
                    pair_loss = _variable_margin_loss(pos_logits, neg_logits, margins, pair_weights)

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
            maybe_save_best_val_total_loss(
                epoch=int(epoch),
                global_step_now=int(global_step),
                val_total_loss=float(val_total_loss),
                val_kl_loss=float(val_kl_loss),
                val_pair_loss=float(val_pair_loss),
                val_mse_loss=float(val_mse_loss),
                source="epoch_eval",
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
        "pair_mid_min_candidates": int(pair_mid_min_candidates),
        "pair_mid_start_ratio": float(pair_mid_start_ratio),
        "pair_mid_end_ratio": float(pair_mid_end_ratio),
        "pair_lower_mid_start_ratio": float(pair_lower_mid_start_ratio),
        "pair_lower_mid_end_ratio": float(pair_lower_mid_end_ratio),
        "pair_type_max_share": float(pair_type_max_share),
        "pair_type_weight_default": float(pair_default_weight),
        "pair_type_weights": dict(pair_type_weights),
        "pair_type_cap_stats": pair_type_cap_stats,
        "list_augmented_doc_weight": float(list_augmented_doc_weight),
        "train_low_cap": {
            "train_band_high_threshold": float(train_band_high_threshold),
            "train_band_mid_threshold": float(train_band_mid_threshold),
            "train_low_band_cap_ratio": float(train_low_band_cap_ratio),
            "stats": train_low_cap_stats,
            "train_pairs_before_low_cap_filter": int(train_pairs_before_low_cap_filter),
            "train_pairs_after_low_cap_filter": int(train_pairs_after_low_cap_filter),
            "train_pairs_low_cap_filtered": int(train_pairs_low_cap_filtered),
        },
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
        "stage1_early_stop": bool(stage1_early_stop),
        "stage1_early_stop_patience": int(stage1_early_stop_patience),
        "stage2_start_from_best_stage1": bool(stage2_start_from_best_stage1),
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
        "best_stage1_val_pair_loss": (
            float(best_stage1_val_pair_loss) if math.isfinite(best_stage1_val_pair_loss) else -1.0
        ),
        "best_stage1_val_pair_loss_step": int(best_stage1_val_pair_loss_step),
        "best_stage1_val_pair_loss_epoch": int(best_stage1_val_pair_loss_epoch),
        "best_stage1_val_pair_loss_source": str(best_stage1_val_pair_loss_source),
        "best_stage1_val_pair_loss_checkpoint": str((output_dir / "best_stage1_val_pair_loss").resolve()),
        "best_val_total_loss": float(best_val_total_loss) if math.isfinite(best_val_total_loss) else -1.0,
        "best_val_total_loss_step": int(best_val_total_loss_step),
        "best_val_total_loss_epoch": int(best_val_total_loss_epoch),
        "best_val_total_loss_source": str(best_val_total_loss_source),
        "best_val_total_loss_components": dict(best_val_total_loss_components),
        "best_val_total_loss_checkpoint": str((output_dir / "best_val_loss").resolve()),
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
            "run/best_stage1_val_pair_loss": (
                float(best_stage1_val_pair_loss) if math.isfinite(best_stage1_val_pair_loss) else -1.0
            ),
            "run/best_val_total_loss": float(best_val_total_loss) if math.isfinite(best_val_total_loss) else -1.0,
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
    if math.isfinite(best_stage1_val_pair_loss):
        print(f"best_stage1_val_pair_loss={best_stage1_val_pair_loss:.6f}")
        print(f"best_stage1_val_pair_loss_step={best_stage1_val_pair_loss_step}")
        print(f"best_stage1_val_pair_loss_epoch={best_stage1_val_pair_loss_epoch}")
        print(f"best_stage1_val_pair_loss_ckpt={output_dir / 'best_stage1_val_pair_loss'}")
    if math.isfinite(best_val_total_loss):
        print(f"best_val_total_loss={best_val_total_loss:.6f}")
        print(f"best_val_total_loss_step={best_val_total_loss_step}")
        print(f"best_val_total_loss_epoch={best_val_total_loss_epoch}")
        print(f"best_val_total_loss_ckpt={output_dir / 'best_val_loss'}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
