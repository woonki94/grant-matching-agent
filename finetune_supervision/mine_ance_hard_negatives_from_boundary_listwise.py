from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


DEFAULT_MODEL_NAME = "BAAI/bge-reranker-base"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 64
DEFAULT_MINE_PER_QUERY = 4
DEFAULT_MARGIN_WINDOW = 0.20
DEFAULT_HARD_WEIGHT = 3.0
DEFAULT_TOP_NEGATIVES = 8
DEFAULT_SEED = 42


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


def _to_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _resolve_latest_boundary_listwise(dataset_dir: Path) -> Path:
    cands = sorted(
        dataset_dir.glob("*_listwise_boundary_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"No boundary listwise dataset found under {dataset_dir}")
    return cands[0].resolve()


def _resolve_model_dir(model_dir: str) -> Path:
    def _has_model_artifacts(p: Path) -> bool:
        if not p.exists() or not p.is_dir():
            return False
        has_cfg = (p / "config.json").exists()
        has_model = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
        has_tok = (p / "tokenizer_config.json").exists() or (p / "tokenizer.json").exists()
        return bool(has_cfg and has_model and has_tok)

    def _pick_best_checkpoint_dir(p: Path) -> Optional[Path]:
        cands = [x for x in p.glob("checkpoint-*") if x.is_dir()]
        if not cands:
            return None

        def _score(x: Path) -> Tuple[int, float]:
            step = -1
            try:
                step = int(x.name.split("-", 1)[1])
            except Exception:
                step = -1
            return (int(step), float(x.stat().st_mtime))

        cands.sort(key=_score, reverse=True)
        for c in cands:
            if _has_model_artifacts(c):
                return c.resolve()
        return None

    target = _clean_text(model_dir)
    if target:
        p = Path(target).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Model directory not found: {p}")
        if _has_model_artifacts(p):
            return p
        ckpt = _pick_best_checkpoint_dir(p)
        if ckpt is not None:
            return ckpt
        raise FileNotFoundError(
            f"No loadable model artifacts found in {p}. "
            "Expected config/model/tokenizer in directory or checkpoint-* subdirs."
        )
    return Path(DEFAULT_MODEL_NAME)


def _pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def _load_tokenizer_stable(model_name_or_path: str):
    from transformers import AutoConfig, AutoTokenizer

    target = _clean_text(model_name_or_path)
    if not target:
        raise RuntimeError("Tokenizer load target is empty.")

    last_error: Optional[Exception] = None
    saw_sentencepiece_error = False
    local_attempts = (
        {"use_fast": False, "fix_mistral_regex": True, "local_files_only": True},
        {"use_fast": True, "fix_mistral_regex": True, "local_files_only": True},
        {"use_fast": False, "local_files_only": True},
        {"use_fast": True, "local_files_only": True},
    )
    remote_attempts = (
        {"use_fast": False, "fix_mistral_regex": True},
        {"use_fast": True, "fix_mistral_regex": True},
        {"use_fast": False},
        {"use_fast": True},
    )
    for kwargs in local_attempts:
        try:
            return AutoTokenizer.from_pretrained(target, trust_remote_code=True, **kwargs)
        except Exception as e:
            last_error = e
            if "sentencepiece" in _clean_text(e).lower():
                saw_sentencepiece_error = True

    is_local_target = Path(target).expanduser().exists()
    if not is_local_target:
        for kwargs in remote_attempts:
            try:
                return AutoTokenizer.from_pretrained(target, trust_remote_code=True, **kwargs)
            except Exception as e:
                last_error = e
                if "sentencepiece" in _clean_text(e).lower():
                    saw_sentencepiece_error = True

    base_name = ""
    try:
        cfg = AutoConfig.from_pretrained(target, trust_remote_code=True, local_files_only=True)
        base_name = _clean_text(getattr(cfg, "_name_or_path", ""))
    except Exception:
        base_name = ""
    if base_name and base_name != target:
        base_is_local = Path(base_name).expanduser().exists()
        if base_is_local:
            for kwargs in local_attempts:
                try:
                    return AutoTokenizer.from_pretrained(base_name, trust_remote_code=True, **kwargs)
                except Exception as e:
                    last_error = e
                    if "sentencepiece" in _clean_text(e).lower():
                        saw_sentencepiece_error = True
        if (not is_local_target) and (not base_is_local):
            for kwargs in remote_attempts:
                try:
                    return AutoTokenizer.from_pretrained(base_name, trust_remote_code=True, **kwargs)
                except Exception as e:
                    last_error = e
                    if "sentencepiece" in _clean_text(e).lower():
                        saw_sentencepiece_error = True

    # Final fallback: known base tokenizer for this reranker family.
    if (not is_local_target) and _clean_text(DEFAULT_MODEL_NAME) and _clean_text(DEFAULT_MODEL_NAME) != target:
        for kwargs in remote_attempts:
            try:
                return AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, trust_remote_code=True, **kwargs)
            except Exception as e:
                last_error = e
    elif _clean_text(DEFAULT_MODEL_NAME) and _clean_text(DEFAULT_MODEL_NAME) != target:
        default_local = Path(_clean_text(DEFAULT_MODEL_NAME)).expanduser()
        if default_local.exists():
            for kwargs in local_attempts:
                try:
                    return AutoTokenizer.from_pretrained(str(default_local), trust_remote_code=True, **kwargs)
                except Exception as e:
                    last_error = e
                    if "sentencepiece" in _clean_text(e).lower():
                        saw_sentencepiece_error = True
        if not is_local_target:
            for kwargs in remote_attempts:
                try:
                    return AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, trust_remote_code=True, **kwargs)
                except Exception as e:
                    last_error = e
                    if "sentencepiece" in _clean_text(e).lower():
                        saw_sentencepiece_error = True

    if saw_sentencepiece_error:
        raise RuntimeError(
            "Failed to load tokenizer: sentencepiece is required for this tokenizer. "
            "Install in your environment (`pip install sentencepiece`) and retry."
        ) from last_error

    raise RuntimeError(f"Failed to load tokenizer stably. target={target}, last_error={last_error}")


def _load_model(model_name_or_path: str):
    from transformers import AutoModelForSequenceClassification

    tokenizer = _load_tokenizer_stable(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    device, device_name = _pick_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device, device_name


def _score_pairs(
    *,
    tokenizer,
    model,
    device,
    pairs: Sequence[Tuple[str, str]],
    batch_size: int,
    max_length: int,
    show_progress: bool,
) -> List[float]:
    import torch

    safe_batch = _safe_limit(batch_size, default=DEFAULT_BATCH_SIZE, minimum=1, maximum=8192)
    safe_max_len = _safe_limit(max_length, default=DEFAULT_MAX_LENGTH, minimum=32, maximum=4096)
    pair_list = list(pairs or [])
    if not pair_list:
        return []

    starts = list(range(0, len(pair_list), safe_batch))
    out: List[float] = []
    iterator = starts
    if bool(show_progress) and _tqdm is not None and len(starts) > 1:
        iterator = _tqdm(starts, desc="Scoring pairs", leave=False)

    for s in iterator:
        batch = pair_list[s : s + safe_batch]
        q = [_clean_text(x[0]) for x in batch]
        d = [_clean_text(x[1]) for x in batch]
        enc = tokenizer(
            q,
            d,
            padding=True,
            truncation=True,
            max_length=safe_max_len,
            return_tensors="pt",
            verbose=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            if logits.ndim == 2 and logits.shape[-1] > 1:
                s_batch = logits[:, -1]
            else:
                s_batch = logits.squeeze(-1)
            out.extend([float(x) for x in s_batch.detach().cpu().tolist()])
    return out


def _load_boundary_rows(
    path: Path,
    *,
    max_queries: int,
    min_candidates: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    safe_max = _safe_limit(max_queries, default=0, minimum=0, maximum=10_000_000)
    safe_min_candidates = _safe_limit(min_candidates, default=4, minimum=2, maximum=256)
    out: List[Dict[str, Any]] = []
    skipped = Counter()

    with path.open("r", encoding="utf-8") as f:
        for row_idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            if safe_max > 0 and len(out) >= safe_max:
                break
            try:
                item = dict(json.loads(line) or {})
            except Exception:
                skipped["parse_error"] += 1
                continue

            query = _clean_text(item.get("query"))
            query_group = _clean_text(item.get("query_group")) or f"query::{row_idx}"
            ranking = [int(x) for x in list(item.get("ranking") or []) if int(x) > 0]
            if not query or len(ranking) < safe_min_candidates:
                skipped["bad_shape"] += 1
                continue

            by_i: Dict[int, str] = {}
            for c in list(item.get("candidates") or []):
                i = int(c.get("i") or 0)
                t = _clean_text(c.get("t") or c.get("text"))
                if i > 0 and t:
                    by_i[i] = t

            ordered = [i for i in ranking if i in by_i]
            if len(ordered) < safe_min_candidates:
                skipped["too_few_candidates"] += 1
                continue

            positives = [int(x) for x in list(item.get("positive_indices") or []) if int(x) in by_i]
            negatives = [int(x) for x in list(item.get("negative_indices") or []) if int(x) in by_i]
            uncertain = [int(x) for x in list(item.get("uncertain_indices") or []) if int(x) in by_i]
            if not positives:
                skipped["no_positive"] += 1
                continue
            if not negatives:
                skipped["no_negative"] += 1
                continue

            out.append(
                {
                    "row_idx": int(item.get("row_idx") or row_idx),
                    "query": query,
                    "query_group": query_group,
                    "ranking": ordered,
                    "by_i": by_i,
                    "positive_indices": positives,
                    "negative_indices": negatives,
                    "uncertain_indices": uncertain,
                    "first_negative_rank": int(item.get("first_negative_rank") or 0),
                    "label_source": _clean_text(item.get("label_source")) or "llm_boundary_labels",
                }
            )

    meta = {
        "input_path": str(path),
        "queries_loaded": int(len(out)),
        "rows_skipped": int(sum(skipped.values())),
        "skip_counts": dict(skipped),
    }
    return out, meta


def _mine_ance_pairs(
    *,
    rows: Sequence[Dict[str, Any]],
    tokenizer,
    model,
    device,
    batch_size: int,
    max_length: int,
    mine_per_query: int,
    margin_window: float,
    top_negatives: int,
    hard_weight: float,
    easy_weight: float,
    include_easy_fallback: bool,
    seed: int,
    show_progress: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    safe_per_q = _safe_limit(mine_per_query, default=DEFAULT_MINE_PER_QUERY, minimum=1, maximum=128)
    safe_top_neg = _safe_limit(top_negatives, default=DEFAULT_TOP_NEGATIVES, minimum=1, maximum=512)
    safe_margin = float(max(0.0, _safe_float(margin_window, default=DEFAULT_MARGIN_WINDOW)))
    safe_hard_w = float(max(0.0, _safe_float(hard_weight, default=DEFAULT_HARD_WEIGHT)))
    safe_easy_w = float(max(0.0, _safe_float(easy_weight, default=1.0)))
    rng = random.Random(int(seed))

    pair_inputs: List[Tuple[str, str]] = []
    backrefs: List[Tuple[int, int]] = []
    scored_rows: List[List[Dict[str, Any]]] = []

    for ridx, row in enumerate(list(rows or [])):
        query = _clean_text(row.get("query"))
        ranking = [int(x) for x in list(row.get("ranking") or []) if int(x) > 0]
        by_i = dict(row.get("by_i") or {})
        docs_for_row: List[Dict[str, Any]] = []
        for rank_pos, idx in enumerate(ranking, start=1):
            text = _clean_text(by_i.get(int(idx)))
            if not text:
                continue
            docs_for_row.append(
                {
                    "idx": int(idx),
                    "rank": int(rank_pos),
                    "text": text,
                    "is_positive": int(idx in set(row.get("positive_indices") or [])),
                    "is_negative": int(idx in set(row.get("negative_indices") or [])),
                    "is_uncertain": int(idx in set(row.get("uncertain_indices") or [])),
                    "score": 0.0,
                }
            )
            pair_inputs.append((query, text))
            backrefs.append((int(ridx), len(docs_for_row) - 1))
        scored_rows.append(docs_for_row)

    scores = _score_pairs(
        tokenizer=tokenizer,
        model=model,
        device=device,
        pairs=pair_inputs,
        batch_size=batch_size,
        max_length=max_length,
        show_progress=bool(show_progress),
    )

    for score, (ridx, didx) in zip(scores, backrefs):
        if 0 <= ridx < len(scored_rows) and 0 <= didx < len(scored_rows[ridx]):
            scored_rows[ridx][didx]["score"] = float(score)

    out_rows: List[Dict[str, Any]] = []
    stats = Counter()
    score_gaps: List[float] = []

    iterator = list(enumerate(rows))
    if bool(show_progress) and _tqdm is not None and len(iterator) > 1:
        iterator = list(_tqdm(iterator, desc="Mining ANCE negatives", leave=False))

    for ridx, row in iterator:
        docs = list(scored_rows[ridx] if ridx < len(scored_rows) else [])
        query = _clean_text(row.get("query"))
        query_group = _clean_text(row.get("query_group"))
        label_source = _clean_text(row.get("label_source")) or "llm_boundary_labels"
        if not query or not docs:
            continue

        pos_docs = [d for d in docs if int(d.get("is_positive") or 0) == 1]
        neg_docs = [d for d in docs if int(d.get("is_negative") or 0) == 1]
        if not pos_docs or not neg_docs:
            stats["rows_without_pos_or_neg"] += 1
            continue

        pos_docs.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        neg_docs.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        anchor_pos = pos_docs[0]
        pos_score = float(anchor_pos.get("score") or 0.0)
        pos_text = _clean_text(anchor_pos.get("text"))
        pos_idx = int(anchor_pos.get("idx") or 0)
        pos_rank = int(anchor_pos.get("rank") or 0)
        if not pos_text or pos_idx <= 0:
            stats["rows_missing_anchor_positive"] += 1
            continue

        threshold = float(pos_score - safe_margin)
        candidate_negs = [n for n in neg_docs[:safe_top_neg] if float(n.get("score") or 0.0) >= threshold]
        if not candidate_negs and bool(include_easy_fallback) and neg_docs:
            candidate_negs = [neg_docs[0]]
            stats["fallback_easy_used"] += 1

        if not candidate_negs:
            stats["rows_with_no_mined_negatives"] += 1
            continue

        if len(candidate_negs) > safe_per_q:
            rng.shuffle(candidate_negs)
            candidate_negs = sorted(candidate_negs[:safe_per_q], key=lambda x: float(x.get("score") or 0.0), reverse=True)

        mined_for_row = 0
        for neg in candidate_negs:
            neg_text = _clean_text(neg.get("text"))
            neg_idx = int(neg.get("idx") or 0)
            neg_rank = int(neg.get("rank") or 0)
            neg_score = float(neg.get("score") or 0.0)
            if not neg_text or neg_idx <= 0 or neg_text == pos_text:
                continue

            gap = float(pos_score - neg_score)
            neg_type = "hard_negative"
            pair_weight = float(safe_hard_w)
            neg_teacher = 0.2
            if gap > safe_margin and bool(include_easy_fallback):
                neg_type = "easy_negative"
                pair_weight = float(max(0.0, safe_easy_w))
                neg_teacher = 0.0

            out_rows.append(
                {
                    "query": query,
                    "positive": pos_text,
                    "negative": neg_text,
                    "negative_type": neg_type,
                    "pair_weight": float(pair_weight),
                    "query_group": query_group,
                    "label_source": "ance_mined_from_boundary",
                    "base_label_source": label_source,
                    "positive_teacher_score": 1.0,
                    "negative_teacher_score": float(neg_teacher),
                    "teacher_margin": float(max(0.0, 1.0 - neg_teacher)),
                    "positive_index": int(pos_idx),
                    "negative_index": int(neg_idx),
                    "positive_rank": int(pos_rank),
                    "negative_rank": int(neg_rank),
                    "positive_model_score": float(pos_score),
                    "negative_model_score": float(neg_score),
                    "model_score_gap": float(gap),
                }
            )
            score_gaps.append(float(gap))
            mined_for_row += 1
            if neg_type == "hard_negative":
                stats["hard_pairs"] += 1
            else:
                stats["easy_pairs"] += 1
        if mined_for_row > 0:
            stats["queries_mined"] += 1
            if any(float(n.get("score") or -1e9) >= pos_score for n in candidate_negs):
                stats["queries_with_false_top_neg"] += 1
        stats["queries_processed"] += 1

    gap_mean = 0.0
    gap_min = 0.0
    gap_max = 0.0
    if score_gaps:
        gap_mean = float(sum(score_gaps) / float(len(score_gaps)))
        gap_min = float(min(score_gaps))
        gap_max = float(max(score_gaps))

    meta = {
        "queries_processed": int(stats.get("queries_processed", 0)),
        "queries_mined": int(stats.get("queries_mined", 0)),
        "queries_with_false_top_neg": int(stats.get("queries_with_false_top_neg", 0)),
        "rows_without_pos_or_neg": int(stats.get("rows_without_pos_or_neg", 0)),
        "rows_missing_anchor_positive": int(stats.get("rows_missing_anchor_positive", 0)),
        "rows_with_no_mined_negatives": int(stats.get("rows_with_no_mined_negatives", 0)),
        "fallback_easy_used": int(stats.get("fallback_easy_used", 0)),
        "pairs_output": int(len(out_rows)),
        "hard_pairs": int(stats.get("hard_pairs", 0)),
        "easy_pairs": int(stats.get("easy_pairs", 0)),
        "model_score_gap": {
            "mean": float(gap_mean),
            "min": float(gap_min),
            "max": float(gap_max),
        },
    }
    return out_rows, meta


def _save_jsonl(
    rows: Sequence[Dict[str, Any]],
    *,
    output_dir: Path,
    stem: str,
    meta: Dict[str, Any],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{stem}.jsonl"
    meta_path = output_dir / f"{stem}.meta.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in list(rows or []):
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    payload = dict(meta or {})
    payload["row_count"] = int(len(list(rows or [])))
    payload["jsonl_path"] = str(jsonl_path)
    payload["meta_path"] = str(meta_path)
    meta_path.write_text(json.dumps(_to_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"jsonl_path": str(jsonl_path), "meta_path": str(meta_path)}


def build_dataset(
    *,
    input_listwise_jsonl: str,
    model_name_or_path: str,
    output_dir: str,
    output_prefix: str,
    max_queries: int,
    min_candidates: int,
    batch_size: int,
    max_length: int,
    mine_per_query: int,
    margin_window: float,
    top_negatives: int,
    hard_negative_weight: float,
    easy_negative_weight: float,
    include_easy_fallback: bool,
    seed: int,
    show_progress: bool,
) -> Dict[str, Any]:
    input_path = Path(_clean_text(input_listwise_jsonl)).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(_clean_text(output_dir)).expanduser().resolve()
    safe_prefix = _clean_text(output_prefix) or "spec_chunk_ance_mined"
    safe_seed = _safe_limit(seed, default=DEFAULT_SEED, minimum=1, maximum=2_147_483_647)

    model_target = _resolve_model_dir(model_name_or_path)
    rows, load_meta = _load_boundary_rows(
        input_path,
        max_queries=int(max_queries),
        min_candidates=int(min_candidates),
    )
    if not rows:
        raise RuntimeError("No valid boundary listwise rows loaded.")

    tokenizer, model, device, device_name = _load_model(str(model_target))
    mined_rows, mine_meta = _mine_ance_pairs(
        rows=rows,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=int(batch_size),
        max_length=int(max_length),
        mine_per_query=int(mine_per_query),
        margin_window=float(margin_window),
        top_negatives=int(top_negatives),
        hard_weight=float(hard_negative_weight),
        easy_weight=float(easy_negative_weight),
        include_easy_fallback=bool(include_easy_fallback),
        seed=int(safe_seed),
        show_progress=bool(show_progress),
    )

    save_meta = {
        "kind": "spec_chunk_ance_hard_negative_pairs",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_listwise_jsonl": str(input_path),
        "model_name_or_path": str(model_target),
        "device": str(device_name),
        "params": {
            "max_queries": int(max_queries),
            "min_candidates": int(min_candidates),
            "batch_size": int(batch_size),
            "max_length": int(max_length),
            "mine_per_query": int(mine_per_query),
            "margin_window": float(margin_window),
            "top_negatives": int(top_negatives),
            "hard_negative_weight": float(hard_negative_weight),
            "easy_negative_weight": float(easy_negative_weight),
            "include_easy_fallback": bool(include_easy_fallback),
            "seed": int(safe_seed),
        },
        "load": load_meta,
        "mine": mine_meta,
    }
    paths = _save_jsonl(
        mined_rows,
        output_dir=output_path,
        stem=f"{safe_prefix}_pairs_ance_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        meta=save_meta,
    )

    return {
        "input": str(input_path),
        "model": str(model_target),
        "device": str(device_name),
        "load": load_meta,
        "mine": mine_meta,
        "output": paths,
    }


def _build_parser() -> argparse.ArgumentParser:
    dataset_dir = Path(__file__).resolve().parent / "dataset"
    default_input = ""
    try:
        default_input = str(_resolve_latest_boundary_listwise(dataset_dir))
    except Exception:
        default_input = str(dataset_dir / "spec_chunk_boundary_listwise.jsonl")

    p = argparse.ArgumentParser(
        description=(
            "ANCE-style hard-negative mining from LLM boundary listwise rows "
            "(query + candidates + positive/negative indices)."
        )
    )
    p.add_argument("--input-listwise-jsonl", type=str, default=default_input)
    p.add_argument("--model-name-or-path", type=str, default=str((Path(__file__).resolve().parent / "models" / "promoted_best").resolve()))
    p.add_argument("--output-dir", type=str, default=str(dataset_dir.resolve()))
    p.add_argument("--output-prefix", type=str, default="spec_chunk_boundary_ance")
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--min-candidates", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--mine-per-query", type=int, default=DEFAULT_MINE_PER_QUERY)
    p.add_argument("--margin-window", type=float, default=DEFAULT_MARGIN_WINDOW)
    p.add_argument("--top-negatives", type=int, default=DEFAULT_TOP_NEGATIVES)
    p.add_argument("--hard-negative-weight", type=float, default=DEFAULT_HARD_WEIGHT)
    p.add_argument("--easy-negative-weight", type=float, default=1.0)
    p.add_argument("--include-easy-fallback", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--no-tqdm", action="store_true")
    p.add_argument("--json-only", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    payload = build_dataset(
        input_listwise_jsonl=_clean_text(args.input_listwise_jsonl),
        model_name_or_path=_clean_text(args.model_name_or_path),
        output_dir=_clean_text(args.output_dir),
        output_prefix=_clean_text(args.output_prefix),
        max_queries=int(args.max_queries),
        min_candidates=int(args.min_candidates),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        mine_per_query=int(args.mine_per_query),
        margin_window=float(args.margin_window),
        top_negatives=int(args.top_negatives),
        hard_negative_weight=float(args.hard_negative_weight),
        easy_negative_weight=float(args.easy_negative_weight),
        include_easy_fallback=bool(args.include_easy_fallback),
        seed=int(args.seed),
        show_progress=not bool(args.no_tqdm),
    )

    if not bool(args.json_only):
        print("ANCE-style hard negative mining complete.")
        print(f"  input listwise jsonl : {payload.get('input', '')}")
        print(f"  model                : {payload.get('model', '')}")
        print(f"  device               : {payload.get('device', '')}")
        print(f"  loaded queries       : {payload.get('load', {}).get('queries_loaded', 0)}")
        print(f"  queries mined        : {payload.get('mine', {}).get('queries_mined', 0)}")
        print(f"  output pairs rows    : {payload.get('mine', {}).get('pairs_output', 0)}")
        print(f"  hard pairs           : {payload.get('mine', {}).get('hard_pairs', 0)}")
        print(f"  easy pairs           : {payload.get('mine', {}).get('easy_pairs', 0)}")
        print(f"  output pairs jsonl   : {payload.get('output', {}).get('jsonl_path', '')}")
        print()

    print(json.dumps(_to_json_ready(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


