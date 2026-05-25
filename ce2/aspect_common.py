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
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ce2.llm_runtime import clean_text, normalize_ws, score_to_band


MODEL_ID_DEFAULT = "Qwen/Qwen3-14B"
GRANT_DB_DEFAULT = "ce/dataset/source/grant_keywords_spec_keywords_db.json"
FAC_DB_DEFAULT = "ce/dataset/source/fac_specs_db.json"
OUTPUT_DIR_DEFAULT = "ce2/dataset/distill"
DECOMPOSITION_OUTPUT_DEFAULT = "ce2/dataset/distill/spec_decompositions_3aspect_shortform.jsonl"
SCORES_OUTPUT_DEFAULT = "ce2/dataset/distill/decomposed_3aspect_shortform_pair_scores.jsonl"
SUMMARY_OUTPUT_DEFAULT = "ce2/dataset/distill/decomposed_3aspect_shortform_pair_scores_summary.json"

SEED_DEFAULT = 42
MAX_GRANT_SPECS_DEFAULT = 80
MAX_FAC_SPECS_DEFAULT = 2500
DECOMPOSE_BATCH_SIZE_DEFAULT = 16
SCORE_BATCH_SIZE_DEFAULT = 24
DECOMPOSE_MAX_NEW_TOKENS_DEFAULT = 512
SCORE_MAX_NEW_TOKENS_DEFAULT = 300
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


def load_scored_pair_keys(path: Path) -> set[str]:
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


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "including",
    "into",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "using",
    "with",
}


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", clean_text(text).lower()) if t not in _STOPWORDS]


def _weighted_overlap_score(query_tokens: Sequence[str], doc_tokens: Sequence[str], idf: Dict[str, float]) -> float:
    q = set(query_tokens)
    d = set(doc_tokens)
    if not q or not d:
        return 0.0
    inter = q & d
    if not inter:
        return 0.0
    inter_w = sum(idf.get(t, 1.0) for t in inter)
    denom = math.sqrt(sum(idf.get(t, 1.0) ** 2 for t in q)) * math.sqrt(sum(idf.get(t, 1.0) ** 2 for t in d))
    return float(inter_w / max(denom, 1e-9))

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


def select_pairs_aspect_prefilter(
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
) -> List[Tuple[SpecItem, SpecItem, float, str]]:
    rng = random.Random(int(seed) + 911)
    n_fac = len(fac_specs)
    selected: Dict[str, Dict[str, Any]] = {}

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

    for aspect_idx, aspect in enumerate(ASPECTS):
        grant_aspect_texts = [
            _aspect_text_for_item(g, aspect=aspect, decompositions=decompositions)
            for g in grant_specs
        ]
        fac_aspect_texts = [
            _aspect_text_for_item(f, aspect=aspect, decompositions=decompositions)
            for f in fac_specs
        ]
        grant_tokens = [_tokens(t) for t in grant_aspect_texts]
        fac_tokens = [_tokens(t) for t in fac_aspect_texts]

        df: Counter[str] = Counter()
        for toks in (*grant_tokens, *fac_tokens):
            df.update(set(toks))
        n_docs = max(1, len(grant_tokens) + len(fac_tokens))
        idf = {tok: math.log((1.0 + n_docs) / (1.0 + cnt)) + 1.0 for tok, cnt in df.items()}

        posting: Dict[str, List[int]] = {}
        for fac_idx, toks in enumerate(fac_tokens):
            for tok in set(toks):
                posting.setdefault(tok, []).append(fac_idx)

        for grant_idx, grant in enumerate(grant_specs):
            gtoks = grant_tokens[grant_idx]
            candidate_idx: set[int] = set()
            for tok in set(gtoks):
                for idx in posting.get(tok, []):
                    candidate_idx.add(idx)

            scored: List[Tuple[float, int]] = []
            for idx in candidate_idx:
                score = _weighted_overlap_score(gtoks, fac_tokens[idx], idf)
                scored.append((float(score), idx))
            scored.sort(key=lambda x: x[0], reverse=True)

            high_pool = [idx for _, idx in scored[: max(0, int(high_pool_size))]]
            high_k = min(max(0, int(high_per_aspect)), len(high_pool))
            high_pick = rng.sample(high_pool, high_k) if high_k > 0 else []

            mid_start = max(0, int(mid_rank_start))
            mid_end = max(mid_start, int(mid_rank_end))
            mid_pool = [idx for _, idx in scored[mid_start:mid_end]]
            mid_k = min(max(0, int(mid_per_aspect)), len(mid_pool))
            mid_pick = rng.sample(mid_pool, mid_k) if mid_k > 0 else []

            tail_from_scored = [idx for _, idx in scored[mid_end:]]
            zero_pool = [idx for idx in range(n_fac) if idx not in candidate_idx]
            low_pool = zero_pool if zero_pool else tail_from_scored
            if not low_pool:
                low_pool = list(range(n_fac))
            low_k = min(max(0, int(low_per_aspect)), len(low_pool))
            low_pick = rng.sample(low_pool, low_k) if low_k > 0 else []

            score_lookup = {idx: s for s, idx in scored}

            for idx in high_pick:
                fac = fac_specs[idx]
                _add_pair(grant, fac, score_lookup.get(idx, 0.0), f"apf_{aspect}_high")
            for idx in mid_pick:
                fac = fac_specs[idx]
                _add_pair(grant, fac, score_lookup.get(idx, 0.0), f"apf_{aspect}_mid")
            for idx in low_pick:
                fac = fac_specs[idx]
                _add_pair(grant, fac, score_lookup.get(idx, 0.0), f"apf_{aspect}_low")

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


def build_summary(*, scores_path: Path, decomposition_path: Path, started_at: float, config: Dict[str, Any]) -> Dict[str, Any]:
    rows = list(iter_jsonl(scores_path))
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
