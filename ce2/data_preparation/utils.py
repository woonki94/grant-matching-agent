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
DISTILL_MAX_NEW_TOKENS_DEFAULT = 300
MAX_MODEL_LEN_DEFAULT = 4096
TEMPERATURE_DEFAULT = 0.0
TOP_P_DEFAULT = 0.9
MAX_ATTEMPTS_DEFAULT = 2

ASPECTS = ("domain", "method", "target")
ASPECT_WORD_LIMITS = {"domain": 3, "method": 4, "target": 4}
ASPECT_MAX_ITEMS = {"domain": 4, "method": 4, "target": 4}
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
        bucket = selected.get(pair_id)
        if bucket is None:
            selected[pair_id] = {
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
    high_pool_size: int = ASPECT_PREFILTER_HIGH_POOL_SIZE_DEFAULT,
    mid_rank_start: int = ASPECT_PREFILTER_MID_RANK_START_DEFAULT,
    mid_rank_end: int = ASPECT_PREFILTER_MID_RANK_END_DEFAULT,
) -> List[Tuple[SpecItem, SpecItem, float, str]]:
    rng = random.Random(int(seed) + 911)
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

    def _add_pair(grant: SpecItem, fac: SpecItem, score: float, source: str) -> None:
        pair_id = f"{grant.item_id}::{fac.item_id}"
        bucket = selected.get(pair_id)
        if bucket is None:
            selected[pair_id] = {
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
                score = _safe_float(cand.get("ce_score"), _safe_float(cand.get("ce_logit"), 0.0))
                ranked.append((rank, fac, score))

            if not ranked:
                continue
            ranked.sort(key=lambda x: (x[0], -float(x[2]), x[1].item_id))

            high_pool = ranked[: max(0, int(high_pool_size))]
            high_k = min(max(0, int(high_per_aspect)), len(high_pool))
            high_pick = rng.sample(high_pool, high_k) if high_k > 0 else []

            mid_start = max(0, int(mid_rank_start))
            mid_end = max(mid_start, int(mid_rank_end))
            mid_pool = ranked[mid_start:mid_end]
            mid_k = min(max(0, int(mid_per_aspect)), len(mid_pool))
            mid_pick = rng.sample(mid_pool, mid_k) if mid_k > 0 else []

            low_start = max(mid_end, len(ranked) - max(32, int(STS_PREFILTER_LOW_TAIL_POOL_DEFAULT)))
            low_pool = ranked[low_start:]
            if not low_pool:
                low_pool = ranked[mid_end:] if mid_end < len(ranked) else ranked
            low_k = min(max(0, int(low_per_aspect)), len(low_pool))
            low_pick = rng.sample(low_pool, low_k) if low_k > 0 else []

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
    score_values = {
        aspect: [float(row.get("scores", {}).get(aspect, 0.0)) for row in rows if isinstance(row.get("scores"), dict)]
        for aspect in (*ASPECTS, "overall")
    }
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


_METHOD_CUE_RE = re.compile(
    r"\b("
    r"manag(?:e|es|ed|ing|ment)?|"
    r"identif(?:y|ies|ied|ying|ication)?|"
    r"evaluat(?:e|es|ed|ing|ion)?|"
    r"measur(?:e|es|ed|ing|ement)?|"
    r"assess(?:ment|e|es|ed|ing)?|"
    r"screen(?:ing|ed|s)?|survey(?:ing|ed|s)?|"
    r"model(?:ing|led|s)?|analy(?:sis|ze|zes|zed|zing|tical)?|"
    r"map(?:ping|ped|s)?|monitor(?:ing|ed|s)?|"
    r"implement(?:ation|ing|ed|s)?|"
    r"develop(?:ing|ed|s)?|design(?:ing|ed|s)?|"
    r"creat(?:e|es|ed|ing|ion)?|distribut(?:e|es|ed|ing|ion)?|"
    r"train(?:ing|ed|s)?|optimiz(?:e|es|ed|ing|ation)?|simulate(?:d|s|ing)?|validate(?:d|s|ing)?|"
    r"case management|legal aid|workflow|protocol|algorithm|method\w*|technique\w*"
    r")\b"
)

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

_NEAR_DUP_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "for",
    "with",
    "to",
    "in",
    "on",
    "and",
    "or",
    "by",
    "via",
    "through",
}

_TARGET_LIKE_RE = re.compile(
    r"\b("
    r"patient\w*|famil\w*|youth|children|child|student\w*|participant\w*|"
    r"institution\w*|university|college|school|institute\w*|entity|entities|"
    r"community|communities|clinic\w*|hospital\w*|agency|agencies|"
    r"service\w*|program\w*|initiative\w*|platform\w*|resource\w*|textbook\w*|"
    r"dataset\w*|tool\w*|fish|species|system\w*|infrastructure"
    r")\b"
)

_DELIVERABLE_LIKE_RE = re.compile(
    r"\b("
    r"textbook\w*|resource\w*|platform\w*|dataset\w*|tool\w*|software|"
    r"report\w*|protocol\w*|model\w*|content|materials?"
    r")\b"
)


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
            words = phrase.split()
            if len(words) > max_words and words and words[0].lower().endswith("ing"):
                phrase = " ".join([words[0], *words[-(max_words - 1) :]])
            else:
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


def _token_set_loose(phrase: str) -> set[str]:
    toks = []
    for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", normalize_ws(phrase).lower()):
        if t in _NEAR_DUP_STOPWORDS:
            continue
        toks.append(t)
    return set(toks)


def _is_near_duplicate(a: str, b: str) -> bool:
    aa = a.casefold()
    bb = b.casefold()
    if aa == bb:
        return True
    ta = _token_set_loose(a)
    tb = _token_set_loose(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    uni = len(ta | tb)
    if uni == 0:
        return False
    j = inter / uni
    if j >= 0.80:
        return True
    return ta.issubset(tb) or tb.issubset(ta)


def _extract_method_fallbacks(text: str) -> List[str]:
    parts = re.split(r",|;", normalize_ws(text), flags=re.IGNORECASE)
    out: List[str] = []
    for part in parts:
        phrase = _strip_capability_prefix(part)
        if not phrase:
            continue
        if not _METHOD_CUE_RE.search(phrase.lower()):
            continue
        words = phrase.split()
        if len(words) < 1:
            continue
        out.append(_limit_words(phrase, max_words=int(ASPECT_WORD_LIMITS["method"])))
    return _normalize_aspect_items("method", out)


def _extract_domain_fallbacks(text: str) -> List[str]:
    base = _strip_capability_prefix(text)
    if not base:
        return []
    preferred = base
    m = re.search(r"\b(?:for|in|on)\b\s+(.+)$", base, flags=re.IGNORECASE)
    if m:
        preferred = normalize_ws(m.group(1))
    preferred = re.split(r",|;|\bwith\b|\busing\b|\bvia\b|\bthrough\b", preferred, maxsplit=1, flags=re.IGNORECASE)[0]
    preferred = _strip_capability_prefix(preferred)
    preferred = re.sub(
        r"^(manag(?:e|es|ed|ing)|identif(?:y|ies|ied|ying)|evaluat(?:e|es|ed|ing)|"
        r"measur(?:e|es|ed|ing)|implement(?:ation|ing|ed|s)|design(?:ing|ed|s)|"
        r"develop(?:ing|ed|s)|creat(?:e|es|ed|ing)|distribut(?:e|es|ed|ing))\s+",
        "",
        preferred,
        flags=re.IGNORECASE,
    )
    cleaned = _normalize_aspect_items("domain", [preferred])
    if cleaned:
        return cleaned
    return _normalize_aspect_items("domain", [_limit_words(base, max_words=int(ASPECT_WORD_LIMITS["domain"]))])


def _extract_target_fallbacks(text: str) -> List[str]:
    base = _strip_capability_prefix(text)
    if not base:
        return []
    candidates: List[str] = []
    for m in re.finditer(
        r"\b(?:for|involving|among|serving|targeting|toward|towards|within)\b\s+([^,;]+)",
        base,
        flags=re.IGNORECASE,
    ):
        seg = normalize_ws(m.group(1))
        if not seg:
            continue
        parts = re.split(r"\band\b|,|;", seg, flags=re.IGNORECASE)
        candidates.extend(normalize_ws(p) for p in parts if normalize_ws(p))
    targetish: List[str] = []
    for c in candidates:
        if _TARGET_LIKE_RE.search(c.lower()):
            targetish.append(c)
    if targetish:
        return _normalize_aspect_items("target", targetish)
    return _normalize_aspect_items("target", candidates)


def _ensure_dense_decomposition(text: str, decomp: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out = {aspect: list(decomp.get(aspect, [])) for aspect in ASPECTS}
    if not out.get("domain"):
        out["domain"] = _extract_domain_fallbacks(text)
    if not out.get("method"):
        out["method"] = _extract_method_fallbacks(text)
    if not out.get("target"):
        out["target"] = _extract_target_fallbacks(text)
    if not out["domain"] and out["target"]:
        out["domain"] = _normalize_aspect_items("domain", [out["target"][0]])
    if not out["target"] and out["domain"]:
        out["target"] = _normalize_aspect_items("target", [out["domain"][0]])
    if not out["method"]:
        seed = out["domain"][0] if out["domain"] else (out["target"][0] if out["target"] else "")
        if not seed:
            seed = _limit_words(_strip_capability_prefix(text), max_words=int(ASPECT_WORD_LIMITS["method"]))
        out["method"] = _normalize_aspect_items("method", [seed])
        if not out["method"]:
            out["method"] = _normalize_aspect_items("method", [f"{seed} method"])
    return {aspect: _normalize_aspect_items(aspect, out.get(aspect, [])) for aspect in ASPECTS}


def _clean_decomposition(text: str, decomp: Dict[str, List[str]]) -> Dict[str, List[str]]:
    cleaned: Dict[str, List[str]] = {aspect: _normalize_aspect_items(aspect, decomp.get(aspect, [])) for aspect in ASPECTS}
    original_domain = list(cleaned.get("domain", []))
    cleaned["method"] = [p for p in cleaned["method"] if _METHOD_CUE_RE.search(p.lower())]
    if not cleaned["method"]:
        cleaned["method"] = _extract_method_fallbacks(text)

    method_keys = {p.casefold() for p in cleaned["method"]}
    cleaned["target"] = [p for p in cleaned["target"] if p.casefold() not in method_keys]
    cleaned["domain"] = [p for p in cleaned["domain"] if p.casefold() not in method_keys]
    cleaned["domain"] = [p for p in cleaned["domain"] if not _METHOD_CUE_RE.search(p.lower())]

    target_vals = list(cleaned["target"])
    cleaned["domain"] = [d for d in cleaned["domain"] if not any(_is_near_duplicate(d, t) for t in target_vals)]
    if target_vals:
        cleaned["domain"] = [d for d in cleaned["domain"] if not _TARGET_LIKE_RE.search(d.lower())]
    cleaned["domain"] = [d for d in cleaned["domain"] if not _DELIVERABLE_LIKE_RE.search(d.lower())]

    if not cleaned["domain"]:
        for d in original_domain:
            dl = d.lower()
            if _METHOD_CUE_RE.search(dl):
                continue
            if _DELIVERABLE_LIKE_RE.search(dl):
                continue
            cleaned["domain"] = [d]
            break

    if len(cleaned["target"]) > 1 and cleaned["domain"]:
        pruned_target = [t for t in cleaned["target"] if not any(_is_near_duplicate(t, d) for d in cleaned["domain"])]
        if pruned_target:
            cleaned["target"] = pruned_target
    cleaned = _ensure_dense_decomposition(text, cleaned)
    for aspect in ASPECTS:
        cleaned[aspect] = _normalize_aspect_items(aspect, cleaned[aspect])
    return cleaned


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
            decomp = _ensure_dense_decomposition(item.text, {aspect: [] for aspect in ASPECTS})
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


def _parse_single_score(obj: Optional[Dict[str, Any]]) -> Tuple[float, str, bool]:
    if not isinstance(obj, dict):
        return 0.0, "", False
    if "score" not in obj:
        return 0.0, "", False
    return coerce_score(obj.get("score")), normalize_ws(obj.get("reason")), True


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
) -> List[Dict[str, Any]]:
    tokenizer = llm_bundle["tokenizer"]
    pending = [p for p in pairs if f"{p[0].item_id}::{p[1].item_id}" not in existing_pair_ids]
    print(f"distill_existing={len(existing_pair_ids)} distill_pending={len(pending)}")
    written_rows: List[Dict[str, Any]] = []
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
            for aspect in ASPECTS:
                for grant, fac, lexical_score, pair_source in chunk:
                    g_dec_row = decompositions.get(grant.item_id, {})
                    f_dec_row = decompositions.get(fac.item_id, {})
                    g_dec = g_dec_row.get("decomposition") if isinstance(g_dec_row, dict) else {}
                    f_dec = f_dec_row.get("decomposition") if isinstance(f_dec_row, dict) else {}
                    pair_id = f"{grant.item_id}::{fac.item_id}"
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

            responses = generate_responses_batch(
                llm_bundle=llm_bundle,
                prompts=prompts,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            )

            rows: List[Dict[str, Any]] = []
            grouped: Dict[str, Dict[str, Any]] = {}
            for item, response in zip(task_items, responses):
                pair_id, aspect, grant, fac, lexical_score, pair_source, g_dec_row, f_dec_row = item
                parsed = extract_json_object(response)
                score, reason, ok = _parse_single_score(parsed)
                bucket = grouped.setdefault(
                    pair_id,
                    {
                        "grant": grant,
                        "fac": fac,
                        "lexical_score": lexical_score,
                        "pair_source": pair_source,
                        "g_dec_row": g_dec_row,
                        "f_dec_row": f_dec_row,
                        "scores": {},
                        "reasons": {},
                        "raw_responses": {},
                        "ok": {},
                    },
                )
                bucket["scores"][aspect] = float(score)
                bucket["reasons"][aspect] = reason
                bucket["raw_responses"][aspect] = response
                bucket["ok"][aspect] = bool(ok)

            for pair_id, bucket in grouped.items():
                grant = bucket["grant"]
                fac = bucket["fac"]
                lexical_score = bucket["lexical_score"]
                pair_source = bucket["pair_source"]
                g_dec_row = bucket["g_dec_row"]
                f_dec_row = bucket["f_dec_row"]
                score_map = bucket["scores"]
                reason_map = bucket["reasons"]
                ok_map = bucket["ok"]
                parse_ok = all(bool(ok_map.get(aspect)) and aspect in score_map for aspect in ASPECTS)
                if not parse_ok:
                    next_pending.append((grant, fac, lexical_score, pair_source))
                    continue
                aspect_scores = {aspect: float(score_map[aspect]) for aspect in ASPECTS}
                overall_score = float(sum(aspect_scores.values()) / max(1, len(ASPECTS)))
                row = {
                    "pair_id": pair_id,
                    "grant": {
                        "item_id": grant.item_id,
                        "text": grant.text,
                        "meta": grant.meta,
                        "decomposition": g_dec_row.get("decomposition", {}),
                    },
                    "faculty": {
                        "item_id": fac.item_id,
                        "text": fac.text,
                        "meta": fac.meta,
                        "decomposition": f_dec_row.get("decomposition", {}),
                    },
                    "scores": {**aspect_scores, "overall": overall_score},
                    "bands": {
                        **{aspect: score_to_band(score) for aspect, score in aspect_scores.items()},
                        "overall": score_to_band(overall_score),
                    },
                    "reasons": {aspect: reason_map.get(aspect, "") for aspect in ASPECTS},
                    "lexical_prefilter_score": float(lexical_score),
                    "pair_source": pair_source,
                    "parse_ok": True,
                    "attempt": int(attempt + 1),
                    "model_id": model_id,
                    "raw_responses": bucket["raw_responses"],
                }
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
