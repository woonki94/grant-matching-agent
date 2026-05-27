from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ce2.data_preparation.llm_runtime import (
    batched,
    build_prompt,
    clean_text,
    coerce_score,
    extract_json_object,
    generate_responses_batch,
    normalize_ws,
    score_to_band,
)
from ce2.data_preparation.prompt.decomposition import DECOMPOSE_SYSTEM_PROMPTS_BY_ASPECT, DECOMPOSE_USER_PROMPT_TEMPLATE
from ce2.data_preparation.prompt.distillation import SCORE_SYSTEM_PROMPTS_BY_ASPECT, SCORE_USER_PROMPT_TEMPLATE

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # type: ignore[assignment]


MODEL_ID_DEFAULT = "Qwen/Qwen3-14B"
GRANT_DB_DEFAULT = "ce/dataset/source/grant_keywords_spec_keywords_db.json"
FAC_DB_DEFAULT = "ce/dataset/source/fac_specs_db.json"
SOURCE_DIR_DEFAULT = "ce2/dataset/source"
DECOMPOSITION_DIR_DEFAULT = "ce2/dataset/decomposed"
DISTILL_DIR_DEFAULT = "ce2/dataset/distill"
OUTPUT_DIR_DEFAULT = DISTILL_DIR_DEFAULT
DECOMPOSITION_OUTPUT_DEFAULT = f"{DECOMPOSITION_DIR_DEFAULT}/spec_decompositions_3aspect_shortform.jsonl"
DISTILLATION_OUTPUT_DEFAULT = f"{DISTILL_DIR_DEFAULT}/llm_distillation.jsonl"
SUMMARY_OUTPUT_DEFAULT = f"{DISTILL_DIR_DEFAULT}/llm_distillation_summary.json"
PREFILTER_CACHE_OUTPUT_DEFAULT = f"{SOURCE_DIR_DEFAULT}/prefilter_cache.jsonl"
PREFILTER_CACHE_MANIFEST_DEFAULT = f"{SOURCE_DIR_DEFAULT}/prefilter_cache.manifest.json"

SEED_DEFAULT = 42
MAX_GRANT_SPECS_DEFAULT = 80
MAX_FAC_SPECS_DEFAULT = 2500
DECOMPOSE_BATCH_SIZE_DEFAULT = 16
DISTILL_BATCH_SIZE_DEFAULT = 24
DECOMPOSE_MAX_NEW_TOKENS_DEFAULT = 512
DISTILL_MAX_NEW_TOKENS_DEFAULT = 32
MAX_MODEL_LEN_DEFAULT = 4096
TEMPERATURE_DEFAULT = 0.0
TOP_P_DEFAULT = 0.9
MAX_ATTEMPTS_DEFAULT = 2

ASPECTS = ("domain", "method", "target")
ASPECT_WORD_LIMITS = {"domain": 6, "method": 8, "target": 6}
ASPECT_MAX_ITEMS = {"domain": 4, "method": 4, "target": 4}
DISTILL_TARGET_HIGH_PER_ASPECT_DEFAULT = 2
DISTILL_TARGET_MID_PER_ASPECT_DEFAULT = 2
DISTILL_TARGET_LOW_PER_ASPECT_DEFAULT = 2
PREFILTER_MULTIPLIER_HIGH_DEFAULT = 8.0
PREFILTER_MULTIPLIER_MID_DEFAULT = 8.0
PREFILTER_MULTIPLIER_LOW_DEFAULT = 4.0
PREFILTER_HIGH_THRESHOLD_DEFAULT = 0.70
PREFILTER_LOW_THRESHOLD_DEFAULT = 0.30
ASPECT_PREFILTER_HIGH_PER_ASPECT_DEFAULT = 2
ASPECT_PREFILTER_MID_PER_ASPECT_DEFAULT = 2
ASPECT_PREFILTER_LOW_PER_ASPECT_DEFAULT = 2
ASPECT_PREFILTER_HIGH_POOL_SIZE_DEFAULT = 24
ASPECT_PREFILTER_MID_RANK_START_DEFAULT = 24
ASPECT_PREFILTER_MID_RANK_END_DEFAULT = 220
STS_PREFILTER_MODEL_ID_DEFAULT = "dleemiller/ModernCE-base-sts"
STS_PREFILTER_BATCH_SIZE_DEFAULT = 256
STS_PREFILTER_MAX_LENGTH_DEFAULT = 64
STS_PREFILTER_GRANT_BLOCK_SIZE_DEFAULT = 256
STS_PREFILTER_LOW_TAIL_POOL_DEFAULT = 400
PREFILTER_CACHE_MODEL_ID_DEFAULT = "dleemiller/ModernCE-base-sts"
PREFILTER_CACHE_BATCH_SIZE_DEFAULT = 64
PREFILTER_CACHE_MAX_LENGTH_DEFAULT = 256
PREFILTER_CACHE_TOP_K_PER_ASPECT_DEFAULT = 256


@dataclass(frozen=True)
class SpecItem:
    item_id: str
    kind: str
    text: str
    meta: Dict[str, Any]


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(clean_text(value)).expanduser()
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_jsonl_by_key(path: Path, key: str) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = clean_text(raw)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                k = clean_text(obj.get(key))
                if k:
                    out[k] = obj
    return out


def load_distilled_pair_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = clean_text(raw)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pair_id = clean_text(obj.get("pair_id"))
            if pair_id:
                keys.add(pair_id)
    return keys


def prefilter_cache_paths(output_base_path: Path) -> Dict[str, Path]:
    base_name = output_base_path.name
    base_stem = output_base_path.stem if output_base_path.suffix else base_name
    base_dir = output_base_path.parent
    return {
        "domain": (base_dir / f"{base_stem}_domain.jsonl").resolve(),
        "method": (base_dir / f"{base_stem}_method.jsonl").resolve(),
        "target": (base_dir / f"{base_stem}_target.jsonl").resolve(),
    }


def row_needs_redecompose(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return True
    if not bool(row.get("parse_ok")):
        return True
    decomp = row.get("decomposition")
    if not isinstance(decomp, dict):
        return True
    if any(aspect not in decomp for aspect in ASPECTS):
        return True
    has_any = False
    for aspect in ASPECTS:
        value = decomp.get(aspect)
        if isinstance(value, list) and any(normalize_ws(x) for x in value):
            has_any = True
            break
    return not has_any


def refresh_decomposition_cache(
    existing: Dict[str, Dict[str, Any]],
    *,
    refresh_failed_only: bool,
    refresh_all: bool,
) -> Dict[str, Dict[str, Any]]:
    if refresh_all:
        return {}
    if not refresh_failed_only:
        return existing
    return {item_id: row for item_id, row in existing.items() if not row_needs_redecompose(row)}


def load_grant_specs(path: Path, *, max_items: int, seed: int) -> List[SpecItem]:
    db = read_json(path)
    grants = db.get("grants") if isinstance(db, dict) else []
    items: List[SpecItem] = []
    for grant in grants or []:
        grant_id = clean_text(grant.get("grant_id"))
        specs = grant.get("grant_spec_keywords") if isinstance(grant, dict) else []
        for idx, text in enumerate(specs or []):
            norm = normalize_ws(text)
            if not norm:
                continue
            items.append(
                SpecItem(
                    item_id=f"grant:{grant_id}:{idx}",
                    kind="grant",
                    text=norm,
                    meta={"grant_id": grant_id, "grant_spec_idx": int(idx)},
                )
            )
    rng = random.Random(int(seed))
    rng.shuffle(items)
    if int(max_items) > 0:
        items = items[: int(max_items)]
    return items


def load_fac_specs(path: Path, *, max_items: int, seed: int) -> List[SpecItem]:
    db = read_json(path)
    rows = db.get("fac_specs") if isinstance(db, dict) else []
    items: List[SpecItem] = []
    for row in rows or []:
        text = normalize_ws(row.get("text"))
        if not text:
            continue
        fac_id = row.get("fac_id")
        fac_spec_id = row.get("fac_spec_id")
        fac_spec_idx = row.get("fac_spec_idx")
        items.append(
            SpecItem(
                item_id=f"fac:{fac_id}:{fac_spec_id}:{fac_spec_idx}",
                kind="faculty",
                text=text,
                meta={
                    "fac_id": fac_id,
                    "fac_spec_id": fac_spec_id,
                    "fac_spec_idx": fac_spec_idx,
                    "section": row.get("section"),
                },
            )
        )
    rng = random.Random(int(seed) + 13)
    rng.shuffle(items)
    if int(max_items) > 0:
        items = items[: int(max_items)]
    return items


def _pick_device() -> Any:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch: Dict[str, Any], device: Any) -> Dict[str, Any]:
    return {k: v.to(device) for k, v in batch.items()}


def _encode_texts_sts(
    *,
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    device: Any,
    batch_size: int,
    max_length: int,
) -> Any:
    import torch
    import torch.nn.functional as F

    outputs: List[Any] = []
    step = max(1, int(batch_size))
    with torch.no_grad():
        for i in range(0, len(texts), step):
            batch = list(texts[i : i + step])
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = F.normalize(pooled, p=2, dim=1)
            outputs.append(pooled)
    return torch.cat(outputs, dim=0) if outputs else torch.empty((0, 0), device=device)


def _kind_from_row(item_id: str, row_kind: Any) -> str:
    kind = clean_text(row_kind).lower()
    if kind in {"grant", "faculty"}:
        return kind
    if item_id.startswith("grant:"):
        return "grant"
    if item_id.startswith("fac:"):
        return "faculty"
    return kind or "unknown"


def load_specs_from_decompositions(
    *,
    decompositions: Dict[str, Dict[str, Any]],
    max_grant_specs: int,
    max_fac_specs: int,
    seed: int,
) -> tuple[List[SpecItem], List[SpecItem]]:
    grant_specs: List[SpecItem] = []
    fac_specs: List[SpecItem] = []

    for item_id, row in decompositions.items():
        if not isinstance(row, dict):
            continue
        iid = clean_text(item_id) or clean_text(row.get("item_id"))
        if not iid:
            continue
        kind = _kind_from_row(iid, row.get("kind"))
        text = normalize_ws(row.get("text"))
        if not text:
            continue
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        item = SpecItem(item_id=iid, kind=kind, text=text, meta=meta)
        if kind == "grant":
            grant_specs.append(item)
        elif kind == "faculty":
            fac_specs.append(item)

    rng = random.Random(int(seed))
    rng.shuffle(grant_specs)
    rng = random.Random(int(seed) + 13)
    rng.shuffle(fac_specs)

    if int(max_grant_specs) > 0:
        grant_specs = grant_specs[: int(max_grant_specs)]
    if int(max_fac_specs) > 0:
        fac_specs = fac_specs[: int(max_fac_specs)]
    return grant_specs, fac_specs


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def score_query_against_docs(
    *,
    model: Any,
    tokenizer: Any,
    query_text: str,
    docs: Sequence[str],
    device: Any,
    batch_size: int,
    max_length: int,
) -> List[float]:
    import torch

    scores: List[float] = []
    step = max(1, int(batch_size))
    with torch.no_grad():
        for i in range(0, len(docs), step):
            docs_batch = list(docs[i : i + step])
            q_batch = [query_text] * len(docs_batch)
            enc = tokenizer(
                q_batch,
                docs_batch,
                max_length=int(max_length),
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = _to_device(enc, device)
            logits = model(**enc).logits
            if logits.ndim == 2 and logits.shape[-1] > 1:
                vals = logits[:, -1]
            else:
                vals = logits.squeeze(-1)
            scores.extend(float(x) for x in vals.detach().cpu().tolist())
    return scores


def iter_topk_candidates(
    *,
    fac_specs: Sequence[SpecItem],
    logits: Sequence[float],
    top_k: int,
) -> Iterable[Dict[str, Any]]:
    ranked = sorted(range(len(logits)), key=lambda i: float(logits[i]), reverse=True)
    keep = ranked[: max(0, int(top_k))] if int(top_k) > 0 else ranked
    for rank, idx in enumerate(keep, start=1):
        fac = fac_specs[idx]
        logit = float(logits[idx])
        yield {
            "rank": int(rank),
            "fac_item_id": fac.item_id,
            "fac_id": fac.meta.get("fac_id"),
            "fac_spec_id": fac.meta.get("fac_spec_id"),
            "fac_spec_idx": fac.meta.get("fac_spec_idx"),
            "section": clean_text(fac.meta.get("section")),
            "ce_logit": float(logit),
            "ce_score": float(_sigmoid(logit)),
        }

def _aspect_text_for_item(
    item: SpecItem,
    *,
    aspect: str,
    decompositions: Dict[str, Dict[str, Any]],
) -> str:
    row = decompositions.get(item.item_id, {})
    decomp = row.get("decomposition") if isinstance(row, dict) else {}
    values = decomp.get(aspect) if isinstance(decomp, dict) else []
    if isinstance(values, list):
        phrases = [normalize_ws(v) for v in values if normalize_ws(v)]
        if phrases:
            return " ".join(phrases)
    return item.text


def _safe_decomposition(value: Any) -> Dict[str, List[str]]:
    if not isinstance(value, dict):
        return {aspect: [] for aspect in ASPECTS}
    return {
        aspect: _as_list(value.get(aspect))
        for aspect in ASPECTS
    }


def _has_aspect_items(decomp: Dict[str, List[str]], aspect: str) -> bool:
    return any(normalize_ws(x) for x in decomp.get(aspect, []))


def select_pairs_with_sts_prefilter(
    grant_specs: Sequence[SpecItem],
    fac_specs: Sequence[SpecItem],
    *,
    decompositions: Dict[str, Dict[str, Any]],
    seed: int,
    high_per_aspect: int = ASPECT_PREFILTER_HIGH_PER_ASPECT_DEFAULT,
    mid_per_aspect: int = ASPECT_PREFILTER_MID_PER_ASPECT_DEFAULT,
    low_per_aspect: int = ASPECT_PREFILTER_LOW_PER_ASPECT_DEFAULT,
    high_pool_size: int = ASPECT_PREFILTER_HIGH_POOL_SIZE_DEFAULT,
    mid_rank_start: int = ASPECT_PREFILTER_MID_RANK_START_DEFAULT,
    mid_rank_end: int = ASPECT_PREFILTER_MID_RANK_END_DEFAULT,
    sts_model_id: str = STS_PREFILTER_MODEL_ID_DEFAULT,
    sts_batch_size: int = STS_PREFILTER_BATCH_SIZE_DEFAULT,
    sts_max_length: int = STS_PREFILTER_MAX_LENGTH_DEFAULT,
    grant_block_size: int = STS_PREFILTER_GRANT_BLOCK_SIZE_DEFAULT,
) -> List[Tuple[SpecItem, SpecItem, float, str]]:
    try:
        import torch
    except Exception as e:
        raise RuntimeError("STS prefilter requires torch to be installed.") from e
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "STS prefilter requires transformers to be installed. "
            "Install `transformers` in your runtime."
        ) from e

    rng = random.Random(int(seed) + 911)
    n_fac = len(fac_specs)
    selected: Dict[str, Dict[str, Any]] = {}
    device = _pick_device()
    model_ref = clean_text(sts_model_id) or STS_PREFILTER_MODEL_ID_DEFAULT
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_ref, trust_remote_code=True)
    model.to(device)
    model.eval()
    print(
        f"prefilter_method=sts_embed_cosine "
        f"prefilter_model={model_ref} prefilter_device={device}"
    )

    def _add_pair(grant: SpecItem, fac: SpecItem, score: float, source: str) -> None:
        pair_id = f"{grant.item_id}::{fac.item_id}"
        aspects = _source_aspects(source)
        aspect_key = aspects[0] if aspects else "unknown"
        selected_id = f"{pair_id}::{aspect_key}"
        bucket = selected.get(selected_id)
        if bucket is None:
            selected[selected_id] = {
                "grant": grant,
                "fac": fac,
                "score": float(score),
                "sources": {source},
            }
            return
        bucket["score"] = max(float(bucket["score"]), float(score))
        bucket["sources"].add(source)

    for aspect in ASPECTS:
        grant_aspect_texts = [
            _aspect_text_for_item(g, aspect=aspect, decompositions=decompositions)
            for g in grant_specs
        ]
        fac_aspect_texts = [
            _aspect_text_for_item(f, aspect=aspect, decompositions=decompositions)
            for f in fac_specs
        ]
        grant_emb = _encode_texts_sts(
            texts=grant_aspect_texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=int(sts_batch_size),
            max_length=int(sts_max_length),
        )
        fac_emb = _encode_texts_sts(
            texts=fac_aspect_texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=int(sts_batch_size),
            max_length=int(sts_max_length),
        )
        fac_emb_t = fac_emb.transpose(0, 1)
        block = max(1, int(grant_block_size))
        mid_start = max(0, int(mid_rank_start))
        mid_end = max(mid_start, int(mid_rank_end))
        low_tail = max(32, int(STS_PREFILTER_LOW_TAIL_POOL_DEFAULT))

        for start in range(0, len(grant_specs), block):
            end = min(len(grant_specs), start + block)
            sims = torch.matmul(grant_emb[start:end], fac_emb_t)
            for local_i, grant in enumerate(grant_specs[start:end]):
                row = sims[local_i]
                order = torch.argsort(row, descending=True)
                sorted_idx = order.detach().cpu().tolist()

                high_pool = sorted_idx[: max(0, int(high_pool_size))]
                high_k = min(max(0, int(high_per_aspect)), len(high_pool))
                high_pick = rng.sample(high_pool, high_k) if high_k > 0 else []

                mid_pool = sorted_idx[mid_start:mid_end]
                mid_k = min(max(0, int(mid_per_aspect)), len(mid_pool))
                mid_pick = rng.sample(mid_pool, mid_k) if mid_k > 0 else []

                low_start = max(mid_end, n_fac - low_tail)
                low_pool = sorted_idx[low_start:]
                if not low_pool:
                    low_pool = sorted_idx[mid_end:] if mid_end < len(sorted_idx) else sorted_idx
                low_k = min(max(0, int(low_per_aspect)), len(low_pool))
                low_pick = rng.sample(low_pool, low_k) if low_k > 0 else []

                for idx in high_pick:
                    fac = fac_specs[idx]
                    _add_pair(grant, fac, (float(row[idx].item()) + 1.0) / 2.0, f"apf_{aspect}_high")
                for idx in mid_pick:
                    fac = fac_specs[idx]
                    _add_pair(grant, fac, (float(row[idx].item()) + 1.0) / 2.0, f"apf_{aspect}_mid")
                for idx in low_pick:
                    fac = fac_specs[idx]
                    _add_pair(grant, fac, (float(row[idx].item()) + 1.0) / 2.0, f"apf_{aspect}_low")

    del model

    pairs: List[Tuple[SpecItem, SpecItem, float, str]] = []
    for item in selected.values():
        grant = item["grant"]
        fac = item["fac"]
        score = float(item["score"])
        sources = "|".join(sorted(item["sources"]))
        pairs.append((grant, fac, score, sources))
    pairs.sort(key=lambda x: (x[0].item_id, -float(x[2]), x[1].item_id))
    return pairs


def select_pairs_from_prefilter_cache(
    grant_specs: Sequence[SpecItem],
    fac_specs: Sequence[SpecItem],
    *,
    cache_base_path: Path,
    seed: int,
    high_per_aspect: int = ASPECT_PREFILTER_HIGH_PER_ASPECT_DEFAULT,
    mid_per_aspect: int = ASPECT_PREFILTER_MID_PER_ASPECT_DEFAULT,
    low_per_aspect: int = ASPECT_PREFILTER_LOW_PER_ASPECT_DEFAULT,
    high_threshold: float = PREFILTER_HIGH_THRESHOLD_DEFAULT,
    low_threshold: float = PREFILTER_LOW_THRESHOLD_DEFAULT,
) -> List[Tuple[SpecItem, SpecItem, float, str]]:
    grant_by_id = {g.item_id: g for g in grant_specs}
    fac_by_id = {f.item_id: f for f in fac_specs}
    cache_paths = prefilter_cache_paths(cache_base_path)
    missing = [str(path) for path in cache_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prefilter cache files:\n"
            + "\n".join(missing)
            + "\nRun ce2/data_preparation/build_prefilter_cache.py first."
        )

    selected: Dict[str, Dict[str, Any]] = {}

    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _candidate_score(cand: Dict[str, Any]) -> float:
        if cand.get("ce_score") is not None:
            return _safe_float(cand.get("ce_score"), 0.0)
        if cand.get("score") is not None:
            return _safe_float(cand.get("score"), 0.0)
        return float(_sigmoid(_safe_float(cand.get("ce_logit"), 0.0)))

    def _dedupe_ranked(items: Iterable[Tuple[int, SpecItem, float]]) -> List[Tuple[int, SpecItem, float]]:
        out: List[Tuple[int, SpecItem, float]] = []
        seen: set[str] = set()
        for item in items:
            fac_id = item[1].item_id
            if fac_id in seen:
                continue
            seen.add(fac_id)
            out.append(item)
        return out

    def _window_right(items: Sequence[Tuple[int, SpecItem, float]], start: int, count: int) -> List[Tuple[int, SpecItem, float]]:
        if not items or int(count) <= 0:
            return []
        s = min(max(0, int(start)), len(items) - 1)
        return list(items[s : min(len(items), s + int(count))])

    def _window_left(items: Sequence[Tuple[int, SpecItem, float]], end: int, count: int) -> List[Tuple[int, SpecItem, float]]:
        if not items or int(count) <= 0:
            return []
        e = min(max(0, int(end)), len(items) - 1)
        s = max(0, e - int(count) + 1)
        return list(items[s : e + 1])

    def _center_window(items: Sequence[Tuple[int, SpecItem, float]], count: int) -> List[Tuple[int, SpecItem, float]]:
        if not items or int(count) <= 0:
            return []
        k = min(int(count), len(items))
        center = len(items) // 2
        start = max(0, center - (k // 2))
        end = min(len(items), start + k)
        start = max(0, end - k)
        return list(items[start:end])

    def _pick_mid_candidates(ranked: Sequence[Tuple[int, SpecItem, float]], count: int) -> List[Tuple[int, SpecItem, float]]:
        if not ranked or int(count) <= 0:
            return []
        k = min(int(count), len(ranked))
        right_count = (k + 1) // 2
        left_count = k - right_count

        first_under_high = next(
            (i for i, (_, _, score) in enumerate(ranked) if float(score) < float(high_threshold)),
            len(ranked) // 2,
        )
        last_over_low = next(
            (i for i in range(len(ranked) - 1, -1, -1) if float(ranked[i][2]) > float(low_threshold)),
            len(ranked) // 2,
        )
        picks = _dedupe_ranked(
            [
                *_window_right(ranked, first_under_high, right_count),
                *_window_left(ranked, last_over_low, left_count),
            ]
        )
        if len(picks) < k:
            picks = _dedupe_ranked([*picks, *_center_window(ranked, k - len(picks))])
        if len(picks) < k:
            picks = _dedupe_ranked([*picks, *ranked])
        return picks[:k]

    def _add_pair(grant: SpecItem, fac: SpecItem, score: float, source: str) -> None:
        pair_id = f"{grant.item_id}::{fac.item_id}"
        aspects = _source_aspects(source)
        aspect_key = aspects[0] if aspects else "unknown"
        selected_id = f"{pair_id}::{aspect_key}"
        bucket = selected.get(selected_id)
        if bucket is None:
            selected[selected_id] = {
                "grant": grant,
                "fac": fac,
                "score": float(score),
                "sources": {source},
            }
            return
        bucket["score"] = max(float(bucket["score"]), float(score))
        bucket["sources"].add(source)

    for aspect in ASPECTS:
        for row in iter_jsonl(cache_paths[aspect]):
            grant = grant_by_id.get(clean_text(row.get("grant_item_id")))
            if grant is None:
                continue
            raw_candidates = row.get("candidates")
            if not isinstance(raw_candidates, list):
                continue

            ranked: List[Tuple[int, SpecItem, float]] = []
            for idx, cand in enumerate(raw_candidates, start=1):
                if not isinstance(cand, dict):
                    continue
                fac = fac_by_id.get(clean_text(cand.get("fac_item_id")))
                if fac is None:
                    continue
                rank = _safe_int(cand.get("rank"), idx)
                score = _candidate_score(cand)
                ranked.append((rank, fac, score))

            if not ranked:
                continue
            ranked.sort(key=lambda x: (x[0], -float(x[2]), x[1].item_id))

            high_k = min(max(0, int(high_per_aspect)), len(ranked))
            mid_k = min(max(0, int(mid_per_aspect)), len(ranked))
            low_k = min(max(0, int(low_per_aspect)), len(ranked))
            high_pick = ranked[:high_k]
            mid_pick = _pick_mid_candidates(ranked, mid_k)
            low_pick = ranked[len(ranked) - low_k :] if low_k > 0 else []

            for _, fac, score in high_pick:
                _add_pair(grant, fac, score, f"cecache_{aspect}_high")
            for _, fac, score in mid_pick:
                _add_pair(grant, fac, score, f"cecache_{aspect}_mid")
            for _, fac, score in low_pick:
                _add_pair(grant, fac, score, f"cecache_{aspect}_low")

    pairs: List[Tuple[SpecItem, SpecItem, float, str]] = []
    for item in selected.values():
        grant = item["grant"]
        fac = item["fac"]
        score = float(item["score"])
        sources = "|".join(sorted(item["sources"]))
        pairs.append((grant, fac, score, sources))
    pairs.sort(key=lambda x: (x[0].item_id, -float(x[2]), x[1].item_id))
    return pairs


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = clean_text(raw)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    idx = min(len(vals) - 1, max(0, int(round(float(q) * (len(vals) - 1)))))
    return float(vals[idx])


def _score_stats(values: Sequence[float]) -> Dict[str, Any]:
    vals = [float(v) for v in values]
    if not vals:
        return {"count": 0}
    avg = sum(vals) / len(vals)
    var = sum((x - avg) ** 2 for x in vals) / max(1, len(vals))
    return {
        "count": len(vals),
        "mean": avg,
        "std": math.sqrt(var),
        "min": min(vals),
        "p10": _percentile(vals, 0.10),
        "p25": _percentile(vals, 0.25),
        "p50": _percentile(vals, 0.50),
        "p75": _percentile(vals, 0.75),
        "p90": _percentile(vals, 0.90),
        "max": max(vals),
        "bands": dict(Counter(score_to_band(x) for x in vals)),
    }


def build_summary(*, distillation_path: Path, decomposition_path: Path, started_at: float, config: Dict[str, Any]) -> Dict[str, Any]:
    rows = list(iter_jsonl(distillation_path))
    decomps = list(iter_jsonl(decomposition_path))
    score_values: Dict[str, List[float]] = {aspect: [] for aspect in ASPECTS}
    for row in rows:
        row_aspect = normalize_ws(row.get("aspect")).lower()
        if row_aspect in ASPECTS and row.get("score") is not None:
            score_values[row_aspect].append(coerce_score(row.get("score")))
            continue
        scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
        for aspect in ASPECTS:
            if aspect in scores:
                score_values[aspect].append(coerce_score(scores.get(aspect)))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - started_at,
        "config": config,
        "decompositions": {
            "count": len(decomps),
            "parse_ok": sum(1 for row in decomps if bool(row.get("parse_ok"))),
            "parse_failed": sum(1 for row in decomps if not bool(row.get("parse_ok"))),
            "by_kind": dict(Counter(clean_text(row.get("kind")) for row in decomps)),
        },
        "pairs": {
            "count": len(rows),
            "by_source": dict(Counter(clean_text(row.get("pair_source")) for row in rows)),
        },
        "score_stats": {aspect: _score_stats(values) for aspect, values in score_values.items()},
    }


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [normalize_ws(x) for x in value if normalize_ws(x)]
    if isinstance(value, str) and normalize_ws(value):
        return [normalize_ws(value)]
    return []


def _parse_decomposition_items(obj: Optional[Dict[str, Any]], aspect: str) -> Tuple[List[str], bool]:
    if not isinstance(obj, dict):
        return [], False
    if "items" in obj:
        return _as_list(obj.get("items")), True
    if aspect in obj:
        return _as_list(obj.get(aspect)), True
    return [], False


_TRAILING_DROP_WORDS = {
    "and",
    "or",
    "for",
    "with",
    "to",
    "in",
    "on",
    "via",
    "through",
    "including",
    "involving",
}

_METHOD_SINGLETON_WHITELIST = {
    "screening",
    "surveying",
    "triage",
    "simulation",
    "optimization",
    "modeling",
    "implementation",
    "management",
    "evaluation",
    "assessment",
    "analysis",
    "monitoring",
    "design",
    "development",
    "training",
}

def _dedupe_phrases(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        phrase = normalize_ws(v)
        if not phrase:
            continue
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    return out


def _strip_capability_prefix(phrase: str) -> str:
    p = normalize_ws(phrase)
    p = re.sub(
        r"^(experience|expertise|skill|ability|proficiency|competence)\s+(with|in|for)\s+",
        "",
        p,
        flags=re.IGNORECASE,
    )
    p = re.sub(r"^(experience|expertise|skill|ability)\s+", "", p, flags=re.IGNORECASE)
    return normalize_ws(p)


def _limit_words(phrase: str, *, max_words: int) -> str:
    text = normalize_ws(phrase)
    if not text:
        return ""
    words = text.split()
    while words and words[-1].lower() in _TRAILING_DROP_WORDS:
        words = words[:-1]
    if len(words) <= int(max_words):
        return " ".join(words)
    return " ".join(words[: int(max_words)])


def _normalize_aspect_items(aspect: str, values: Sequence[str]) -> List[str]:
    max_words = int(ASPECT_WORD_LIMITS.get(aspect, 4))
    max_items = int(ASPECT_MAX_ITEMS.get(aspect, 4))
    out: List[str] = []
    for raw in values:
        phrase = normalize_ws(raw)
        if not phrase:
            continue
        phrase = re.sub(r"^[\-\u2022\*\d\.\)\(]+\s*", "", phrase)
        if aspect == "method":
            phrase = _strip_capability_prefix(phrase)
            phrase = _limit_words(phrase, max_words=max_words)
        else:
            phrase = _limit_words(phrase, max_words=max_words)
        if not phrase:
            continue
        if aspect == "method":
            w = phrase.split()
            if len(w) < 2 and phrase.lower() not in _METHOD_SINGLETON_WHITELIST:
                continue
        out.append(phrase)
    return _dedupe_phrases(out)[:max_items]


def _clean_decomposition(text: str, decomp: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {aspect: _normalize_aspect_items(aspect, decomp.get(aspect, [])) for aspect in ASPECTS}


def decompose_specs(
    *,
    llm_bundle: Dict[str, Any],
    model_id: str,
    items: Sequence[SpecItem],
    existing: Dict[str, Dict[str, Any]],
    output_path: Path,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_attempts: int,
) -> Dict[str, Dict[str, Any]]:
    tokenizer = llm_bundle["tokenizer"]
    pending = [item for item in items if item.item_id not in existing]
    print(f"decompose_existing={len(existing)} decompose_pending={len(pending)}")
    for attempt in range(max(1, int(max_attempts))):
        if not pending:
            break
        next_pending: List[SpecItem] = []
        chunks = list(batched(pending, int(batch_size)))
        chunk_iter: Iterable[Sequence[SpecItem]] = chunks
        bar = None
        if tqdm is not None:
            bar = tqdm(
                chunks,
                total=len(chunks),
                desc=f"Decompose {attempt + 1}/{max(1, int(max_attempts))}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )
            chunk_iter = bar
        written_this_attempt = 0
        for chunk in chunk_iter:
            prompts: List[str] = []
            task_items: List[Tuple[str, str, SpecItem]] = []
            for aspect in ASPECTS:
                for item in chunk:
                    prompts.append(
                        build_prompt(
                            tokenizer,
                            model_id=model_id,
                            system_prompt=DECOMPOSE_SYSTEM_PROMPTS_BY_ASPECT[aspect],
                            user_prompt=DECOMPOSE_USER_PROMPT_TEMPLATE.format(aspect=aspect, text=item.text),
                        )
                    )
                    task_items.append((item.item_id, aspect, item))
            responses = generate_responses_batch(
                llm_bundle=llm_bundle,
                prompts=prompts,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            )
            rows: List[Dict[str, Any]] = []
            grouped: Dict[str, Dict[str, Any]] = {}
            for task, response in zip(task_items, responses):
                item_id, aspect, item = task
                parsed = extract_json_object(response)
                items_out, ok = _parse_decomposition_items(parsed, aspect)
                bucket = grouped.setdefault(
                    item_id,
                    {"item": item, "decomposition": {a: [] for a in ASPECTS}, "ok": {}, "raw_responses": {}},
                )
                bucket["decomposition"][aspect] = items_out
                bucket["ok"][aspect] = bool(ok)
                bucket["raw_responses"][aspect] = response

            for item_id, bucket in grouped.items():
                item = bucket["item"]
                decomp = _clean_decomposition(item.text, bucket["decomposition"])
                ok_map = bucket["ok"]
                parsed_aspects = [aspect for aspect in ASPECTS if bool(ok_map.get(aspect))]
                nonempty_aspects = [aspect for aspect in ASPECTS if len(decomp.get(aspect, [])) > 0]
                parse_ok = len(parsed_aspects) >= max(1, len(ASPECTS) - 1) and len(nonempty_aspects) >= 1
                row = {
                    "item_id": item.item_id,
                    "kind": item.kind,
                    "text": item.text,
                    "meta": item.meta,
                    "decomposition": decomp,
                    "parse_ok": bool(parse_ok),
                    "decomposition_parse": {
                        "parsed_aspects_count": int(len(parsed_aspects)),
                        "nonempty_aspects_count": int(len(nonempty_aspects)),
                        "parsed_aspects": parsed_aspects,
                        "nonempty_aspects": nonempty_aspects,
                    },
                    "attempt": int(attempt + 1),
                    "model_id": model_id,
                    "raw_responses": bucket["raw_responses"],
                }
                if len(parsed_aspects) == 0:
                    next_pending.append(item)
                else:
                    existing[item.item_id] = row
                    rows.append(row)
            written_this_attempt += len(rows)
            append_jsonl(output_path, rows)
            if bar is not None:
                bar.set_postfix(written=int(written_this_attempt), retry_pending=int(len(next_pending)), refresh=False)
        if bar is not None:
            bar.close()
        pending = next_pending
        if pending:
            print(f"decompose_retry_pending={len(pending)} attempt={attempt + 1}")

    if pending:
        rows = []
        for item in pending:
            decomp = {aspect: [] for aspect in ASPECTS}
            row = {
                "item_id": item.item_id,
                "kind": item.kind,
                "text": item.text,
                "meta": item.meta,
                "decomposition": decomp,
                "parse_ok": False,
                "attempt": int(max(1, int(max_attempts))),
                "model_id": model_id,
                "raw_responses": {},
            }
            existing[item.item_id] = row
            rows.append(row)
        append_jsonl(output_path, rows)
        print(f"decompose_failed_written={len(rows)}")
    return existing


def _parse_single_score(obj: Optional[Dict[str, Any]]) -> Tuple[float, bool]:
    if not isinstance(obj, dict):
        return 0.0, False
    if "score" not in obj:
        return 0.0, False
    return coerce_score(obj.get("score")), True


def _source_aspects(pair_source: str) -> List[str]:
    aspects: List[str] = []
    for raw in normalize_ws(pair_source).split("|"):
        parts = raw.split("_")
        if len(parts) >= 3 and parts[0] in {"cecache", "apf"} and parts[1] in ASPECTS:
            aspects.append(parts[1])
    out: List[str] = []
    seen: set[str] = set()
    for aspect in aspects:
        if aspect in seen:
            continue
        seen.add(aspect)
        out.append(aspect)
    return out


def _quota_targets(
    *,
    high: int,
    mid: int,
    low: int,
) -> Dict[str, int]:
    return {
        "high": max(0, int(high)),
        "mid": max(0, int(mid)),
        "low": max(0, int(low)),
    }


def _select_distilled_row(
    *,
    row: Dict[str, Any],
    pair_source: str,
    counts: Dict[Tuple[str, str, str], int],
    targets: Dict[str, int],
) -> Tuple[bool, List[Dict[str, Any]]]:
    if not any(int(v) > 0 for v in targets.values()):
        return True, []

    grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
    grant_item_id = normalize_ws(grant.get("item_id"))
    if not grant_item_id:
        return False, []

    bands = row.get("bands") if isinstance(row.get("bands"), dict) else {}
    row_aspect = normalize_ws(row.get("aspect")).lower()
    source_aspects = [row_aspect] if row_aspect in ASPECTS else (_source_aspects(pair_source) or list(ASPECTS))
    selected_clusters: List[Dict[str, Any]] = []
    for aspect in source_aspects:
        band = normalize_ws(row.get("band")).lower() if row_aspect == aspect else normalize_ws(bands.get(aspect)).lower()
        if band not in targets:
            continue
        target = int(targets.get(band, 0))
        if target <= 0:
            continue
        key = (grant_item_id, aspect, band)
        current = int(counts.get(key, 0))
        if current >= target:
            continue
        selected_clusters.append(
            {
                "grant_item_id": grant_item_id,
                "aspect": aspect,
                "band": band,
                "slot": current + 1,
                "target": target,
            }
        )

    if not selected_clusters:
        return False, []
    for cluster in selected_clusters:
        key = (
            normalize_ws(cluster.get("grant_item_id")),
            normalize_ws(cluster.get("aspect")),
            normalize_ws(cluster.get("band")),
        )
        counts[key] = int(counts.get(key, 0) + 1)
    return True, selected_clusters


def distill_pairs(
    *,
    llm_bundle: Dict[str, Any],
    model_id: str,
    pairs: Sequence[Tuple[SpecItem, SpecItem, float, str]],
    decompositions: Dict[str, Dict[str, Any]],
    existing_pair_ids: set[str],
    output_path: Path,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_attempts: int,
    target_high_per_aspect: int = DISTILL_TARGET_HIGH_PER_ASPECT_DEFAULT,
    target_mid_per_aspect: int = DISTILL_TARGET_MID_PER_ASPECT_DEFAULT,
    target_low_per_aspect: int = DISTILL_TARGET_LOW_PER_ASPECT_DEFAULT,
) -> List[Dict[str, Any]]:
    tokenizer = llm_bundle["tokenizer"]
    def _pair_aspect_key(grant: SpecItem, fac: SpecItem, pair_source: str) -> str:
        aspect = (_source_aspects(pair_source) or ["unknown"])[0]
        return f"{grant.item_id}::{fac.item_id}::{aspect}"

    pending = [p for p in pairs if _pair_aspect_key(p[0], p[1], p[3]) not in existing_pair_ids]
    print(f"distill_existing={len(existing_pair_ids)} distill_pending={len(pending)}")
    written_rows: List[Dict[str, Any]] = []
    selected_counts: Dict[Tuple[str, str, str], int] = {}
    target_counts = _quota_targets(
        high=target_high_per_aspect,
        mid=target_mid_per_aspect,
        low=target_low_per_aspect,
    )
    for attempt in range(max(1, int(max_attempts))):
        if not pending:
            break
        next_pending: List[Tuple[SpecItem, SpecItem, float, str]] = []
        chunks = list(batched(pending, int(batch_size)))
        chunk_iter: Iterable[Sequence[Tuple[SpecItem, SpecItem, float, str]]] = chunks
        bar = None
        if tqdm is not None:
            bar = tqdm(
                chunks,
                total=len(chunks),
                desc=f"Distill {attempt + 1}/{max(1, int(max_attempts))}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )
            chunk_iter = bar
        written_this_attempt = 0
        for chunk in chunk_iter:
            prompts: List[str] = []
            task_items: List[Tuple[str, str, SpecItem, SpecItem, float, str, Dict[str, Any], Dict[str, Any]]] = []
            for grant, fac, lexical_score, pair_source in chunk:
                source_aspects = _source_aspects(pair_source)
                if not source_aspects:
                    source_aspects = list(ASPECTS)
                g_dec_row = decompositions.get(grant.item_id, {})
                f_dec_row = decompositions.get(fac.item_id, {})
                g_dec = _safe_decomposition(g_dec_row.get("decomposition") if isinstance(g_dec_row, dict) else {})
                f_dec = _safe_decomposition(f_dec_row.get("decomposition") if isinstance(f_dec_row, dict) else {})
                for aspect in source_aspects:
                    if not (_has_aspect_items(g_dec, aspect) or _has_aspect_items(f_dec, aspect)):
                        continue
                    pair_id = f"{grant.item_id}::{fac.item_id}::{aspect}"
                    user_prompt = SCORE_USER_PROMPT_TEMPLATE.format(
                        aspect=aspect,
                        grant_text=grant.text,
                        grant_aspect_items_json=json.dumps(g_dec.get(aspect, []), ensure_ascii=False),
                        grant_decomposition_json=json.dumps(g_dec, ensure_ascii=False),
                        fac_text=fac.text,
                        fac_aspect_items_json=json.dumps(f_dec.get(aspect, []), ensure_ascii=False),
                        fac_decomposition_json=json.dumps(f_dec, ensure_ascii=False),
                    )
                    prompts.append(
                        build_prompt(
                            tokenizer,
                            model_id=model_id,
                            system_prompt=SCORE_SYSTEM_PROMPTS_BY_ASPECT[aspect],
                            user_prompt=user_prompt,
                        )
                    )
                    task_items.append((pair_id, aspect, grant, fac, lexical_score, pair_source, g_dec_row, f_dec_row))

            if not prompts:
                if bar is not None:
                    bar.set_postfix(written=int(written_this_attempt), retry_pending=int(len(next_pending)), refresh=False)
                continue

            responses = generate_responses_batch(
                llm_bundle=llm_bundle,
                prompts=prompts,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            )

            rows: List[Dict[str, Any]] = []
            for item, response in zip(task_items, responses):
                pair_id, aspect, grant, fac, lexical_score, pair_source, g_dec_row, f_dec_row = item
                parsed = extract_json_object(response)
                score, ok = _parse_single_score(parsed)
                if not ok:
                    next_pending.append((grant, fac, lexical_score, pair_source))
                    continue
                band = score_to_band(float(score))
                row = {
                    "pair_id": pair_id,
                    "aspect": aspect,
                    "score": float(score),
                    "band": band,
                    "grant": {
                        "item_id": grant.item_id,
                        "text": grant.text,
                        "meta": grant.meta,
                        "decomposition": g_dec,
                    },
                    "faculty": {
                        "item_id": fac.item_id,
                        "text": fac.text,
                        "meta": fac.meta,
                        "decomposition": f_dec,
                    },
                    "scores": {aspect: float(score)},
                    "bands": {aspect: band},
                    "lexical_prefilter_score": float(lexical_score),
                    "pair_source": pair_source,
                    "parse_ok": True,
                    "attempt": int(attempt + 1),
                    "model_id": model_id,
                    "raw_response": response,
                    "raw_responses": {aspect: response},
                }
                keep_row, selected_clusters = _select_distilled_row(
                    row=row,
                    pair_source=pair_source,
                    counts=selected_counts,
                    targets=target_counts,
                )
                if not keep_row:
                    existing_pair_ids.add(pair_id)
                    continue
                row["distill_selected_clusters"] = selected_clusters
                existing_pair_ids.add(pair_id)
                rows.append(row)
                written_rows.append(row)

            written_this_attempt += len(rows)
            append_jsonl(output_path, rows)
            if bar is not None:
                bar.set_postfix(written=int(written_this_attempt), retry_pending=int(len(next_pending)), refresh=False)
        if bar is not None:
            bar.close()
        pending = next_pending
        if pending:
            print(f"distill_retry_pending={len(pending)} attempt={attempt + 1}")
    if pending:
        print(f"distill_failed={len(pending)}")
    return written_rows
