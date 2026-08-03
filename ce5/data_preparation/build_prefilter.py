"""Build CE5 high/mid/low candidate sets using CE-STS scoring.

For every grant specialization, ModernCE scores every loaded faculty
specialization.  The scores are divided into per-grant quantile bands and a
small, faculty-diverse sample is retained from each band for later LLM judging.

After the grant-faculty pass completes, the script also generates every unique
pair of specialization keywords belonging to the same faculty owner.  Those
pairs are scored in flat H100-friendly batches and assigned global high/mid/low
rank bands.  They are written separately because they have no grant record.

The bands are candidate-selection strata, not training labels.  A ``high``
ModernCE score still needs an LLM or human capability-coverage judgment.

Both scoring phases are independently resumable.  Use ``--overwrite`` to start
again with a changed configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SOURCE_DIR = REPO_ROOT / "ce5" / "dataset" / "source"
DEFAULT_GRANT_DB = SOURCE_DIR / "grant_specialization_keyword_db.json"
DEFAULT_FACULTY_DB = SOURCE_DIR / "faculty_specialization_keywords_db.json"
DEFAULT_OUTPUT = SOURCE_DIR / "prefilter_candidates.jsonl"
DEFAULT_MANIFEST = SOURCE_DIR / "prefilter_candidates.manifest.json"
DEFAULT_FACULTY_PAIR_OUTPUT = SOURCE_DIR / "faculty_pair_candidates.jsonl"
DEFAULT_FACULTY_PAIR_SCORE_CACHE = SOURCE_DIR / "faculty_pair_scores.jsonl"
DEFAULT_FACULTY_PAIR_MANIFEST = SOURCE_DIR / "faculty_pair_candidates.manifest.json"
DEFAULT_CE_MODEL = "dleemiller/ModernCE-base-sts"
SCHEMA_VERSION = 2
FACULTY_PAIR_SCHEMA_VERSION = "ce5.faculty-pair-prefilter.v1"
FACULTY_PAIR_SCORE_SCHEMA_VERSION = "ce5.faculty-pair-score.v1"
BANDS = ("high", "mid", "low")


@dataclass(frozen=True)
class Specialization:
    item_id: str
    owner_id: str | int
    keyword_index: int
    text: str


@dataclass(frozen=True)
class ScoredResult:
    doc_index: int
    logit: float
    score: float
    rank: int


@dataclass(frozen=True)
class FacultyKeywordPair:
    pair_id: str
    faculty_id: str | int
    left: Specialization
    right: Specialization


class CrossEncoderScorer:
    """Batched ModernCE scorer that returns raw logits and sigmoid scores."""

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        batch_size: int,
        max_length: int,
        local_files_only: bool,
        dtype: str,
        attn_implementation: str,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("CE-STS requires torch and transformers.") from exc

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        model_dtype = _resolve_model_dtype(torch, device=device, requested=dtype)
        if device.startswith("cuda"):
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "torch_dtype": model_dtype,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, **model_kwargs
        )
        self.model.to(device)
        self.model.eval()

    def score_pairs(
        self,
        left_texts: Sequence[str],
        right_texts: Sequence[str],
    ) -> list[tuple[float, float]]:
        if len(left_texts) != len(right_texts):
            raise ValueError("left_texts and right_texts must have equal lengths")
        raw_logits: list[float] = []
        normalized_scores: list[float] = []
        for start in range(0, len(left_texts), self.batch_size):
            left_batch = left_texts[start : start + self.batch_size]
            right_batch = right_texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                list(left_batch),
                list(right_batch),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits

            if logits.ndim == 1:
                batch_logits = logits
                batch_scores = self.torch.sigmoid(batch_logits)
            elif logits.shape[-1] == 1:
                batch_logits = logits[:, 0]
                batch_scores = self.torch.sigmoid(batch_logits)
            elif logits.shape[-1] == 2:
                batch_logits = logits[:, 1] - logits[:, 0]
                batch_scores = self.torch.softmax(logits, dim=-1)[:, 1]
            else:
                raise RuntimeError(
                    f"Unsupported CE output shape: {tuple(logits.shape)}"
                )

            raw_logits.extend(
                float(value) for value in batch_logits.detach().cpu().tolist()
            )
            normalized_scores.extend(
                float(value) for value in batch_scores.detach().cpu().tolist()
            )
        return list(zip(raw_logits, normalized_scores, strict=True))

    def score_all(self, query: str, documents: Sequence[str]) -> list[ScoredResult]:
        pair_scores = self.score_pairs([query] * len(documents), documents)
        raw_logits = [result[0] for result in pair_scores]
        normalized_scores = [result[1] for result in pair_scores]

        ordered = sorted(
            range(len(documents)),
            key=lambda index: (-normalized_scores[index], index),
        )
        rank_by_index = {
            doc_index: rank for rank, doc_index in enumerate(ordered, start=1)
        }
        return [
            ScoredResult(
                doc_index=index,
                logit=raw_logits[index],
                score=normalized_scores[index],
                rank=rank_by_index[index],
            )
            for index in range(len(documents))
        ]


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _resolve_path(value: Path) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_fingerprint(configuration: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_id(*parts: Any, prefix: str) -> str:
    raw = "\x1f".join(_clean_text(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:24]}"


def _load_specializations(
    path: Path,
    *,
    collection_name: str,
    owner_id_key: str,
    item_prefix: str,
) -> list[Specialization]:
    payload = _read_json(path)
    rows = payload.get(collection_name) if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError(f"Expected a '{collection_name}' list in {path}")

    output: list[Specialization] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        owner_id = row.get(owner_id_key)
        if owner_id is None or (isinstance(owner_id, str) and not owner_id.strip()):
            continue
        keywords = row.get("specialization_keywords")
        if not isinstance(keywords, list):
            continue
        for keyword_index, raw_text in enumerate(keywords):
            text = _clean_text(raw_text)
            if not text:
                continue
            output.append(
                Specialization(
                    item_id=f"{item_prefix}:{owner_id}:{keyword_index}",
                    owner_id=owner_id,
                    keyword_index=keyword_index,
                    text=text,
                )
            )
    return output


def _sample_items(
    items: Sequence[Specialization], *, maximum: int, seed: int
) -> list[Specialization]:
    if maximum <= 0 or maximum >= len(items):
        return list(items)
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(items)), maximum))
    return [items[index] for index in selected_indices]


def _build_faculty_keyword_pairs(
    faculty: Sequence[Specialization],
) -> list[FacultyKeywordPair]:
    by_owner: dict[str | int, list[Specialization]] = defaultdict(list)
    for item in faculty:
        by_owner[item.owner_id].append(item)

    pairs: list[FacultyKeywordPair] = []
    for owner_id in sorted(by_owner, key=lambda value: str(value)):
        owner_items = sorted(
            by_owner[owner_id],
            key=lambda item: (item.keyword_index, item.item_id),
        )
        for left_index, left in enumerate(owner_items):
            for right in owner_items[left_index + 1 :]:
                pairs.append(
                    FacultyKeywordPair(
                        pair_id=_stable_id(
                            owner_id,
                            left.item_id,
                            right.item_id,
                            prefix="faculty_pair",
                        ),
                        faculty_id=owner_id,
                        left=left,
                        right=right,
                    )
                )
    return pairs


def _pick_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _resolve_model_dtype(torch_module: Any, *, device: str, requested: str) -> Any:
    value = requested.strip().lower()
    if value == "auto":
        if device.startswith("cuda"):
            return (
                torch_module.bfloat16
                if torch_module.cuda.is_bf16_supported()
                else torch_module.float16
            )
        if device == "mps":
            return torch_module.float16
        return torch_module.float32
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    return mapping[value]


def _stable_random(seed: int, grant_item_id: str, band: str) -> random.Random:
    digest = hashlib.sha256(
        f"{seed}:{grant_item_id}:{band}".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], byteorder="big"))


def _partition_by_quantile(
    results: Sequence[ScoredResult],
    *,
    low_quantile: float,
    high_quantile: float,
) -> tuple[dict[str, list[ScoredResult]], dict[str, float]]:
    if not results:
        return {band: [] for band in BANDS}, {"low_max": 0.0, "high_min": 0.0}
    ordered = sorted(results, key=lambda result: (-result.score, result.doc_index))
    item_count = len(ordered)
    high_count = max(1, item_count - int(math.floor(item_count * high_quantile)))
    low_count = max(1, int(math.floor(item_count * low_quantile)))
    high_end = min(item_count, high_count)
    low_start = max(high_end, item_count - low_count)
    bands = {
        "high": ordered[:high_end],
        "mid": ordered[high_end:low_start],
        "low": ordered[low_start:],
    }
    high_min = float(bands["high"][-1].score)
    low_max = float(bands["low"][0].score) if bands["low"] else high_min
    return bands, {"low_max": low_max, "high_min": high_min}


def _select_from_band(
    pool: Sequence[ScoredResult],
    *,
    count: int,
    band: str,
    grant_item_id: str,
    seed: int,
    faculty: Sequence[Specialization],
    owner_counts: Counter[str | int],
    max_per_faculty: int,
) -> list[ScoredResult]:
    if count <= 0 or not pool:
        return []

    ordered = list(pool)
    if band != "high":
        _stable_random(seed, grant_item_id, band).shuffle(ordered)

    selected: list[ScoredResult] = []
    selected_indices: set[int] = set()
    for enforce_cap in (True, False):
        for result in ordered:
            if result.doc_index in selected_indices:
                continue
            owner_id = faculty[result.doc_index].owner_id
            if (
                enforce_cap
                and max_per_faculty > 0
                and owner_counts[owner_id] >= max_per_faculty
            ):
                continue
            selected.append(result)
            selected_indices.add(result.doc_index)
            owner_counts[owner_id] += 1
            if len(selected) >= count:
                return sorted(selected, key=lambda item: (-item.score, item.doc_index))
    return sorted(selected, key=lambda item: (-item.score, item.doc_index))


def _candidate_json(
    result: ScoredResult,
    *,
    faculty_item: Specialization,
    faculty_count: int,
) -> dict[str, Any]:
    rank_percentile = (
        float(result.rank - 1) / float(faculty_count - 1)
        if faculty_count > 1
        else 0.0
    )
    return {
        "faculty_item_id": faculty_item.item_id,
        "faculty_id": faculty_item.owner_id,
        "faculty_keyword_index": faculty_item.keyword_index,
        "faculty_text": faculty_item.text,
        "ce_sts_score": float(result.score),
        "ce_sts_logit": float(result.logit),
        "ce_sts_rank": int(result.rank),
        "ce_sts_rank_percentile": rank_percentile,
    }


def _read_existing_rows(
    output_path: Path,
    *,
    expected_fingerprint: str,
) -> tuple[set[str], int, Counter[str], list[int]]:
    completed: set[str] = set()
    candidate_count = 0
    band_counts: Counter[str] = Counter()
    pool_sizes: list[int] = []
    if not output_path.exists():
        return completed, candidate_count, band_counts, pool_sizes

    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid partial JSONL at {output_path}:{line_number}. "
                    "Repair it or rerun with --overwrite."
                ) from exc
            if row.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Existing output uses schema {row.get('schema_version')}; "
                    "rerun with --overwrite."
                )
            if row.get("config_fingerprint") != expected_fingerprint:
                raise RuntimeError(
                    "Existing output was built with a different configuration; "
                    "rerun with --overwrite."
                )
            grant_item_id = _clean_text(row.get("grant_item_id"))
            if grant_item_id:
                completed.add(grant_item_id)
            pool_sizes.append(int(row.get("faculty_pool_size", 0) or 0))
            relevance_sets = row.get("relevance_sets")
            if isinstance(relevance_sets, Mapping):
                for band in BANDS:
                    values = relevance_sets.get(band)
                    if isinstance(values, list):
                        band_counts[band] += len(values)
                        candidate_count += len(values)
    return completed, candidate_count, band_counts, pool_sizes


def _read_faculty_pair_score_cache(
    cache_path: Path,
    *,
    expected_fingerprint: str,
) -> dict[str, tuple[float, float]]:
    scores: dict[str, tuple[float, float]] = {}
    if not cache_path.exists():
        return scores
    with cache_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid faculty-pair score cache at {cache_path}:{line_number}. "
                    "Repair it or rerun with --overwrite."
                ) from exc
            if row.get("schema_version") != FACULTY_PAIR_SCORE_SCHEMA_VERSION:
                raise RuntimeError(
                    "Existing faculty-pair score cache has an incompatible schema; "
                    "rerun with --overwrite."
                )
            if row.get("config_fingerprint") != expected_fingerprint:
                raise RuntimeError(
                    "Existing faculty-pair score cache was built with a different "
                    "configuration; rerun with --overwrite."
                )
            pair_id = _clean_text(row.get("pair_id"))
            if not pair_id:
                continue
            scores[pair_id] = (
                float(row.get("ce_sts_logit", 0.0)),
                float(row.get("ce_sts_score", 0.0)),
            )
    return scores


def _faculty_pair_candidate_rows(
    pairs: Sequence[FacultyKeywordPair],
    *,
    scores_by_pair_id: Mapping[str, tuple[float, float]],
    config_fingerprint: str,
    low_quantile: float,
    high_quantile: float,
) -> tuple[list[dict[str, Any]], dict[str, float], Counter[str]]:
    missing = [pair.pair_id for pair in pairs if pair.pair_id not in scores_by_pair_id]
    if missing:
        raise RuntimeError(
            f"Cannot finalize faculty pairs: {len(missing)} pairs have no CE-STS score"
        )

    scored_results = [
        ScoredResult(
            doc_index=index,
            logit=scores_by_pair_id[pair.pair_id][0],
            score=scores_by_pair_id[pair.pair_id][1],
            rank=0,
        )
        for index, pair in enumerate(pairs)
    ]
    ordered_indices = sorted(
        range(len(scored_results)),
        key=lambda index: (-scored_results[index].score, index),
    )
    rank_by_index = {
        pair_index: rank
        for rank, pair_index in enumerate(ordered_indices, start=1)
    }
    ranked_results = [
        ScoredResult(
            doc_index=result.doc_index,
            logit=result.logit,
            score=result.score,
            rank=rank_by_index[result.doc_index],
        )
        for result in scored_results
    ]
    band_pools, thresholds = _partition_by_quantile(
        ranked_results,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
    )
    band_by_pair_index = {
        result.doc_index: band
        for band, results in band_pools.items()
        for result in results
    }
    band_counts: Counter[str] = Counter(band_by_pair_index.values())
    pair_count = len(pairs)
    rows: list[dict[str, Any]] = []
    for pair_index in ordered_indices:
        pair = pairs[pair_index]
        result = ranked_results[pair_index]
        rank_percentile = (
            float(result.rank - 1) / float(pair_count - 1)
            if pair_count > 1
            else 0.0
        )
        rows.append(
            {
                "schema_version": FACULTY_PAIR_SCHEMA_VERSION,
                "config_fingerprint": config_fingerprint,
                "pair_id": pair.pair_id,
                "pair_type": "faculty_faculty",
                "faculty_id": pair.faculty_id,
                "left_item_id": pair.left.item_id,
                "left_keyword_index": pair.left.keyword_index,
                "left_text": pair.left.text,
                "right_item_id": pair.right.item_id,
                "right_keyword_index": pair.right.keyword_index,
                "right_text": pair.right.text,
                "ce_sts_score": float(result.score),
                "ce_sts_logit": float(result.logit),
                "ce_sts_global_rank": int(result.rank),
                "ce_sts_rank_percentile": rank_percentile,
                "prefilter_band": band_by_pair_index[pair_index],
            }
        )
    return rows, thresholds, band_counts


def _write_jsonl_atomically(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _try_progress(total: int, *, description: str, unit: str) -> Any:
    try:
        from tqdm.auto import tqdm

        return tqdm(
            total=total,
            desc=description,
            unit=unit,
            dynamic_ncols=True,
        )
    except Exception:
        return None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _quantile(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("quantile must be strictly between 0 and 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score every grant/faculty specialization pair with CE-STS and "
            "retain high, mid, and low relevance candidate sets, then score "
            "all within-faculty keyword pairs into global relevance bands."
        )
    )
    parser.add_argument("--grant-db", type=Path, default=DEFAULT_GRANT_DB)
    parser.add_argument("--faculty-db", type=Path, default=DEFAULT_FACULTY_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--faculty-pair-output",
        type=Path,
        default=DEFAULT_FACULTY_PAIR_OUTPUT,
    )
    parser.add_argument(
        "--faculty-pair-score-cache",
        type=Path,
        default=DEFAULT_FACULTY_PAIR_SCORE_CACHE,
    )
    parser.add_argument(
        "--faculty-pair-manifest",
        type=Path,
        default=DEFAULT_FACULTY_PAIR_MANIFEST,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grant-keywords", type=_nonnegative_int, default=0)
    parser.add_argument("--max-faculty-keywords", type=_nonnegative_int, default=0)

    parser.add_argument("--high-count", type=_nonnegative_int, default=8)
    parser.add_argument("--mid-count", type=_nonnegative_int, default=8)
    parser.add_argument("--low-count", type=_nonnegative_int, default=6)
    parser.add_argument("--low-quantile", type=_quantile, default=1.0 / 3.0)
    parser.add_argument("--high-quantile", type=_quantile, default=2.0 / 3.0)
    parser.add_argument(
        "--max-candidates-per-faculty",
        type=_nonnegative_int,
        default=2,
        help="Maximum selected keyword pairs from one faculty owner; 0 disables the cap.",
    )

    parser.add_argument("--ce-model", default=DEFAULT_CE_MODEL)
    parser.add_argument(
        "--ce-batch-size",
        type=_positive_int,
        default=512,
        help="Pair batch size; 512 is an H100-oriented starting point.",
    )
    parser.add_argument("--ce-max-length", type=_positive_int, default=128)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="auto selects BF16 on an H100-compatible CUDA device.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("", "eager", "sdpa", "flash_attention_2"),
        default="",
        help=(
            "Optional Transformers attention backend. Leave empty for the model "
            "default, or use flash_attention_2 when installed on the HPC server."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the CE model to already exist in the Hugging Face cache.",
    )
    parser.add_argument(
        "--skip-faculty-pairs",
        action="store_true",
        help="Run only the existing grant-faculty prefilter phase.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output instead of safely resuming it.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    if args.low_quantile >= args.high_quantile:
        raise ValueError("--low-quantile must be smaller than --high-quantile")
    if args.high_count + args.mid_count + args.low_count <= 0:
        raise ValueError("At least one of --high-count/--mid-count/--low-count must be positive")

    grant_db = _resolve_path(args.grant_db)
    faculty_db = _resolve_path(args.faculty_db)
    output_path = _resolve_path(args.output)
    manifest_path = _resolve_path(args.manifest)
    faculty_pair_output_path = _resolve_path(args.faculty_pair_output)
    faculty_pair_score_cache_path = _resolve_path(args.faculty_pair_score_cache)
    faculty_pair_manifest_path = _resolve_path(args.faculty_pair_manifest)
    if not args.skip_faculty_pairs:
        distinct_paths = {
            output_path,
            manifest_path,
            faculty_pair_output_path,
            faculty_pair_score_cache_path,
            faculty_pair_manifest_path,
        }
        if len(distinct_paths) != 5:
            raise ValueError("Grant-faculty and faculty-pair output paths must be distinct")
    for required in (grant_db, faculty_db):
        if not required.exists():
            raise FileNotFoundError(f"Input file not found: {required}")

    grants = _load_specializations(
        grant_db,
        collection_name="grants",
        owner_id_key="grant_id",
        item_prefix="grant",
    )
    faculty = _load_specializations(
        faculty_db,
        collection_name="faculty",
        owner_id_key="faculty_id",
        item_prefix="faculty",
    )
    grants = _sample_items(grants, maximum=args.max_grant_keywords, seed=args.seed)
    faculty = _sample_items(
        faculty,
        maximum=args.max_faculty_keywords,
        seed=args.seed + 1,
    )
    if not grants:
        raise RuntimeError("No grant specialization keywords were loaded.")
    if not faculty:
        raise RuntimeError("No faculty specialization keywords were loaded.")

    grant_sha256 = _file_sha256(grant_db)
    faculty_sha256 = _file_sha256(faculty_db)
    device = _pick_device(args.device)
    configuration = {
        "schema_version": SCHEMA_VERSION,
        "grant_db_sha256": grant_sha256,
        "faculty_db_sha256": faculty_sha256,
        "seed": args.seed,
        "max_grant_keywords": args.max_grant_keywords,
        "max_faculty_keywords": args.max_faculty_keywords,
        "high_count": args.high_count,
        "mid_count": args.mid_count,
        "low_count": args.low_count,
        "low_quantile": args.low_quantile,
        "high_quantile": args.high_quantile,
        "max_candidates_per_faculty": args.max_candidates_per_faculty,
        "ce_model": args.ce_model,
        "ce_max_length": args.ce_max_length,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation or "model_default",
    }
    fingerprint = _config_fingerprint(configuration)

    faculty_pair_configuration = {
        "schema_version": FACULTY_PAIR_SCHEMA_VERSION,
        "score_schema_version": FACULTY_PAIR_SCORE_SCHEMA_VERSION,
        "faculty_db_sha256": faculty_sha256,
        "seed": args.seed,
        "max_faculty_keywords": args.max_faculty_keywords,
        "low_quantile": args.low_quantile,
        "high_quantile": args.high_quantile,
        "ce_model": args.ce_model,
        "ce_max_length": args.ce_max_length,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation or "model_default",
    }
    faculty_pair_fingerprint = _config_fingerprint(faculty_pair_configuration)
    faculty_pairs = (
        [] if args.skip_faculty_pairs else _build_faculty_keyword_pairs(faculty)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        completed: set[str] = set()
        candidates_written = 0
        band_counts: Counter[str] = Counter()
        pool_sizes: list[int] = []
        output_mode = "w"
    else:
        completed, candidates_written, band_counts, pool_sizes = _read_existing_rows(
            output_path,
            expected_fingerprint=fingerprint,
        )
        output_mode = "a"

    if args.skip_faculty_pairs:
        faculty_pair_scores: dict[str, tuple[float, float]] = {}
        pending_faculty_pairs: list[FacultyKeywordPair] = []
        faculty_pair_cache_mode = "a"
    elif args.overwrite:
        faculty_pair_scores = {}
        pending_faculty_pairs = list(faculty_pairs)
        faculty_pair_score_cache_path.parent.mkdir(parents=True, exist_ok=True)
        faculty_pair_score_cache_path.write_text("", encoding="utf-8")
        faculty_pair_cache_mode = "a"
    else:
        faculty_pair_scores = _read_faculty_pair_score_cache(
            faculty_pair_score_cache_path,
            expected_fingerprint=faculty_pair_fingerprint,
        )
        pending_faculty_pairs = [
            pair for pair in faculty_pairs if pair.pair_id not in faculty_pair_scores
        ]
        faculty_pair_cache_mode = "a"

    remaining_grants = [grant for grant in grants if grant.item_id not in completed]
    faculty_texts = [item.text for item in faculty]
    progress = _try_progress(
        len(remaining_grants),
        description="CE5 grant-faculty CE-STS",
        unit="grant item",
    )
    scorer = None
    if remaining_grants or pending_faculty_pairs:
        scorer = CrossEncoderScorer(
            model_id=args.ce_model,
            device=device,
            batch_size=args.ce_batch_size,
            max_length=args.ce_max_length,
            local_files_only=args.local_files_only,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )

    with output_path.open(output_mode, encoding="utf-8") as handle:
        for grant in remaining_grants:
            if scorer is None:
                raise RuntimeError("CE-STS scorer was not initialized")
            scored = scorer.score_all(grant.text, faculty_texts)
            band_pools, thresholds = _partition_by_quantile(
                scored,
                low_quantile=args.low_quantile,
                high_quantile=args.high_quantile,
            )
            owner_counts: Counter[str | int] = Counter()
            relevance_sets: dict[str, list[dict[str, Any]]] = {}
            requested_counts = {
                "high": args.high_count,
                "mid": args.mid_count,
                "low": args.low_count,
            }
            for band in BANDS:
                selected = _select_from_band(
                    band_pools[band],
                    count=requested_counts[band],
                    band=band,
                    grant_item_id=grant.item_id,
                    seed=args.seed,
                    faculty=faculty,
                    owner_counts=owner_counts,
                    max_per_faculty=args.max_candidates_per_faculty,
                )
                relevance_sets[band] = [
                    _candidate_json(
                        result,
                        faculty_item=faculty[result.doc_index],
                        faculty_count=len(faculty),
                    )
                    for result in selected
                ]
                band_counts[band] += len(selected)
                candidates_written += len(selected)

            row = {
                "schema_version": SCHEMA_VERSION,
                "config_fingerprint": fingerprint,
                "grant_item_id": grant.item_id,
                "grant_id": grant.owner_id,
                "grant_keyword_index": grant.keyword_index,
                "grant_text": grant.text,
                "faculty_pool_size": len(faculty),
                "all_faculty_scored": True,
                "band_method": "per_grant_ce_sts_rank_quantiles",
                "band_thresholds": thresholds,
                "band_pool_sizes": {
                    band: len(band_pools[band]) for band in BANDS
                },
                "relevance_sets": relevance_sets,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            completed.add(grant.item_id)
            pool_sizes.append(len(faculty))
            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    grant_faculty_elapsed = time.time() - started
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": len(completed) == len(grants),
        "config_fingerprint": fingerprint,
        "grant_db": str(grant_db),
        "grant_db_sha256": grant_sha256,
        "faculty_db": str(faculty_db),
        "faculty_db_sha256": faculty_sha256,
        "output": str(output_path),
        "device": device,
        "configuration": configuration,
        "grant_keywords_loaded": len(grants),
        "grant_keywords_completed": len(completed),
        "faculty_keywords_loaded": len(faculty),
        "cartesian_pairs_scored_when_complete": len(grants) * len(faculty),
        "candidates_written": candidates_written,
        "selection_bucket_counts": dict(sorted(band_counts.items())),
        "candidate_pool_size": {
            "minimum": min(pool_sizes) if pool_sizes else 0,
            "maximum": max(pool_sizes) if pool_sizes else 0,
            "mean": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        },
        "elapsed_seconds_this_run": grant_faculty_elapsed,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    faculty_pair_elapsed = 0.0
    faculty_pair_scored_this_run = 0
    if not args.skip_faculty_pairs:
        faculty_pair_started = time.time()
        faculty_pair_score_cache_path.parent.mkdir(parents=True, exist_ok=True)
        faculty_progress = _try_progress(
            len(pending_faculty_pairs),
            description="CE5 within-faculty CE-STS",
            unit="pair",
        )
        with faculty_pair_score_cache_path.open(
            faculty_pair_cache_mode,
            encoding="utf-8",
        ) as cache_handle:
            for batch in _chunks(pending_faculty_pairs, args.ce_batch_size):
                if scorer is None:
                    raise RuntimeError("CE-STS scorer was not initialized")
                batch_scores = scorer.score_pairs(
                    [pair.left.text for pair in batch],
                    [pair.right.text for pair in batch],
                )
                for pair, (logit, score) in zip(batch, batch_scores, strict=True):
                    cache_handle.write(
                        json.dumps(
                            {
                                "schema_version": FACULTY_PAIR_SCORE_SCHEMA_VERSION,
                                "config_fingerprint": faculty_pair_fingerprint,
                                "pair_id": pair.pair_id,
                                "ce_sts_logit": float(logit),
                                "ce_sts_score": float(score),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    faculty_pair_scores[pair.pair_id] = (
                        float(logit),
                        float(score),
                    )
                cache_handle.flush()
                faculty_pair_scored_this_run += len(batch)
                if faculty_progress is not None:
                    faculty_progress.update(len(batch))
        if faculty_progress is not None:
            faculty_progress.close()

        faculty_pair_rows, faculty_pair_thresholds, faculty_pair_band_counts = (
            _faculty_pair_candidate_rows(
                faculty_pairs,
                scores_by_pair_id=faculty_pair_scores,
                config_fingerprint=faculty_pair_fingerprint,
                low_quantile=args.low_quantile,
                high_quantile=args.high_quantile,
            )
        )
        _write_jsonl_atomically(faculty_pair_output_path, faculty_pair_rows)
        faculty_pair_elapsed = time.time() - faculty_pair_started
        faculty_pair_manifest = {
            "schema_version": FACULTY_PAIR_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "complete": len(faculty_pair_scores) >= len(faculty_pairs),
            "config_fingerprint": faculty_pair_fingerprint,
            "faculty_db": str(faculty_db),
            "faculty_db_sha256": faculty_sha256,
            "score_cache": str(faculty_pair_score_cache_path),
            "output": str(faculty_pair_output_path),
            "device": device,
            "configuration": faculty_pair_configuration,
            "faculty_owners_loaded": len({item.owner_id for item in faculty}),
            "faculty_keywords_loaded": len(faculty),
            "faculty_pairs_generated": len(faculty_pairs),
            "faculty_pairs_scored": sum(
                pair.pair_id in faculty_pair_scores for pair in faculty_pairs
            ),
            "faculty_pairs_scored_this_run": faculty_pair_scored_this_run,
            "band_method": "global_ce_sts_rank_quantiles",
            "band_thresholds": faculty_pair_thresholds,
            "band_counts": dict(sorted(faculty_pair_band_counts.items())),
            "elapsed_seconds_this_run": faculty_pair_elapsed,
        }
        faculty_pair_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        faculty_pair_manifest_path.write_text(
            json.dumps(faculty_pair_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    total_elapsed = time.time() - started
    print(f"grant_faculty_output={output_path}")
    print(f"grant_faculty_manifest={manifest_path}")
    print(f"grant_keywords_completed={len(completed)}/{len(grants)}")
    print(f"faculty_keywords={len(faculty)}")
    print(f"grant_faculty_cartesian_pairs={len(grants) * len(faculty)}")
    print(f"grant_faculty_candidates_written={candidates_written}")
    if not args.skip_faculty_pairs:
        print(f"faculty_pair_output={faculty_pair_output_path}")
        print(f"faculty_pair_manifest={faculty_pair_manifest_path}")
        print(f"faculty_pairs_generated={len(faculty_pairs)}")
        print(f"faculty_pairs_scored_this_run={faculty_pair_scored_this_run}")
        print(f"faculty_pair_elapsed_seconds={faculty_pair_elapsed:.2f}")
    print(f"total_elapsed_seconds_this_run={total_elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
