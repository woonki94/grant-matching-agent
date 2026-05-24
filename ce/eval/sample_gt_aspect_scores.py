from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce.aspect_modeling import (  # noqa: E402
    aspect_id_from_name,
    clean_aspect_condition_mode,
    format_aspect_pair,
    load_sequence_classifier_model,
    model_logits,
)

INPUT_DEFAULT = "ce/dataset/distill/llm_ground_truth_requirement_common_test_listwise.jsonl"
OUTPUT_DIR_DEFAULT = "ce/eval/results"
MODEL_DEFAULT_CANDIDATES = [
    "ce/models/stage2_epoch_6"
]
HIGH_THRESHOLD_DEFAULT = 0.70
MID_THRESHOLD_DEFAULT = 0.30
SAMPLES_PER_BAND_DEFAULT = 5
BATCH_SIZE_DEFAULT = 32
MAX_LENGTH_DEFAULT = 512
SEED_DEFAULT = 42

ASPECT_PREFIXES = {
    "domain": "[DOMAIN]",
    "method": "[METHOD]",
    "constraint": "[CONSTRAINT]",
}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_ws(value: Any) -> str:
    return " ".join(_clean_text(value).split())


def _safe_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _safe_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _resolve_path(value: str) -> Path:
    p = Path(_clean_text(value)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _resolve_model_ref(value: str) -> str:
    raw = _clean_text(value)
    if not raw:
        return raw
    p = Path(raw).expanduser()
    if p.exists():
        return str(p.resolve())
    p2 = _resolve_path(raw)
    if p2.exists():
        return str(p2)
    return raw


def _default_model_ref() -> str:
    for candidate in MODEL_DEFAULT_CANDIDATES:
        resolved = _resolve_model_ref(candidate)
        if Path(resolved).exists():
            return resolved
    return _resolve_model_ref(MODEL_DEFAULT_CANDIDATES[0])


def _pick_device(device_arg: str) -> torch.device:
    requested = _clean_text(device_arg).lower()
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = _clean_text(raw)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no} ({type(exc).__name__}: {exc})") from exc
            if isinstance(obj, dict):
                yield obj


def _score_band(score: float, *, high_threshold: float, mid_threshold: float) -> str:
    s = float(score)
    if s >= float(high_threshold):
        return "high"
    if s >= float(mid_threshold):
        return "mid"
    return "low"


def _load_ground_truth_rows(
    *,
    path: Path,
    high_threshold: float,
    mid_threshold: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row_idx, obj in enumerate(_iter_jsonl(path), start=1):
        query = _normalize_ws(obj.get("query") or obj.get("query_text"))
        doc = _normalize_ws(obj.get("doc") or obj.get("text"))
        if not query or not doc:
            continue
        gt = _safe_float(
            obj.get("score", obj.get("gt_score", obj.get("teacher_score_raw", obj.get("teacher_score", 0.0)))),
            default=0.0,
            minimum=0.0,
            maximum=1.0,
        )
        rows.append(
            {
                "row_idx": int(row_idx),
                "query": query,
                "doc": doc,
                "gt_score": float(gt),
                "gt_band": _score_band(gt, high_threshold=high_threshold, mid_threshold=mid_threshold),
            }
        )
    return rows


def _sample_rows_by_band(
    *,
    rows: Sequence[Dict[str, Any]],
    samples_per_band: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    by_band: Dict[str, List[Dict[str, Any]]] = {"low": [], "mid": [], "high": []}
    for row in rows:
        band = _clean_text(row.get("gt_band")).lower()
        if band in by_band:
            by_band[band].append(dict(row))

    rng = random.Random(int(seed))
    sampled: List[Dict[str, Any]] = []
    available: Dict[str, int] = {}
    for band in ("low", "mid", "high"):
        bucket = list(by_band[band])
        available[band] = int(len(bucket))
        rng.shuffle(bucket)
        chosen = bucket[: max(0, int(samples_per_band))]
        for i, row in enumerate(chosen, start=1):
            row["sample_band_rank"] = int(i)
            row["pair_id"] = f"{band}_{i}"
            sampled.append(row)
    return sampled, available


def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _resolve_score_calibration_json(*, model_ref: str, explicit_path: str, enabled: bool) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    if not enabled:
        return None, None
    raw = _clean_text(explicit_path)
    if raw:
        p = _resolve_path(raw)
        return p, _load_json_file(p)

    p_model = Path(_clean_text(model_ref)).expanduser()
    if not p_model.is_absolute():
        p_model = _resolve_path(str(p_model))
    if p_model.is_dir():
        p = p_model / "posthoc_calibration_affine.json"
        return p, _load_json_file(p)
    return None, None


def _sigmoid_scalar(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return float(1.0 / (1.0 + z))
    z = math.exp(x)
    return float(z / (1.0 + z))


def _score_to_logit(score: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, float(score)))
    return float(math.log(p / (1.0 - p)))


def _calibration_params_for_aspect(payload: Optional[Dict[str, Any]], aspect: str) -> Tuple[float, float]:
    if not isinstance(payload, dict):
        return 1.0, 0.0
    aspect_key = _clean_text(aspect).lower()
    aspects = payload.get("aspects")
    if isinstance(aspects, dict):
        obj = aspects.get(aspect_key)
        if isinstance(obj, dict):
            return _safe_float(obj.get("scale"), default=1.0, minimum=1e-6, maximum=100.0), _safe_float(
                obj.get("bias"), default=0.0, minimum=-100.0, maximum=100.0
            )
    global_obj = payload.get("global")
    if isinstance(global_obj, dict):
        return _safe_float(global_obj.get("scale"), default=1.0, minimum=1e-6, maximum=100.0), _safe_float(
            global_obj.get("bias"), default=0.0, minimum=-100.0, maximum=100.0
        )
    return _safe_float(payload.get("scale"), default=1.0, minimum=1e-6, maximum=100.0), _safe_float(
        payload.get("bias"), default=0.0, minimum=-100.0, maximum=100.0
    )


def _apply_score_calibration(score: float, *, aspect: str, payload: Optional[Dict[str, Any]]) -> float:
    if not isinstance(payload, dict):
        return float(score)
    scale, bias = _calibration_params_for_aspect(payload, aspect)
    return float(_sigmoid_scalar(float(scale) * _score_to_logit(float(score)) + float(bias)))


def _score_pairs(
    *,
    model: Any,
    tokenizer: Any,
    queries: Sequence[str],
    docs: Sequence[str],
    aspect: str,
    aspect_condition_mode: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> List[float]:
    if len(queries) != len(docs):
        raise RuntimeError("queries/docs length mismatch")
    out: List[float] = []
    step = max(1, int(batch_size))
    aspect_id = int(aspect_id_from_name(aspect))
    with torch.no_grad():
        for start in range(0, len(queries), step):
            formatted = [
                format_aspect_pair(
                    q,
                    d,
                    aspect_condition_mode=aspect_condition_mode,
                )
                for q, d in zip(queries[start : start + step], docs[start : start + step])
            ]
            q_chunk = [p[0] for p in formatted]
            d_chunk = [p[1] for p in formatted]
            enc = tokenizer(
                q_chunk,
                d_chunk,
                max_length=int(max_length),
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = _to_device(enc, device)
            aspect_ids = torch.full((len(q_chunk),), aspect_id, dtype=torch.int64, device=device)
            logits = model_logits(model, enc, aspect_ids=aspect_ids).squeeze(-1)
            probs = torch.sigmoid(logits)
            out.extend(float(x) for x in probs.detach().cpu().tolist())
    return out


def _score_aspects(
    *,
    rows: List[Dict[str, Any]],
    model_ref: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
    calibration_payload: Optional[Dict[str, Any]],
    aspect_condition_mode: str,
) -> None:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Failed to import transformers. Run this script in the same env where training/eval dependencies are installed."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    model = load_sequence_classifier_model(model_ref, num_labels=1, multi_aspect_heads=False, trust_remote_code=True)
    model.to(device)
    model.eval()

    try:
        for aspect, prefix in ASPECT_PREFIXES.items():
            queries = [f"{prefix} {_clean_text(row.get('query'))}" for row in rows]
            docs = [f"{prefix} {_clean_text(row.get('doc'))}" for row in rows]
            scores = _score_pairs(
                model=model,
                tokenizer=tokenizer,
                queries=queries,
                docs=docs,
                aspect=aspect,
                aspect_condition_mode=aspect_condition_mode,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
            )
            for row, score in zip(rows, scores):
                row[f"{aspect}_score"] = float(
                    _apply_score_calibration(score, aspect=aspect, payload=calibration_payload)
                )
    finally:
        try:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass


def _default_output_path(output_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"gt_aspect_score_samples_{ts}.csv"


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_id",
        "gt_band",
        "gt_score",
        "domain_score",
        "method_score",
        "constraint_score",
        "query",
        "doc",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample low/mid/high requirement GT pairs and score them with DOMAIN/METHOD/CONSTRAINT prefixes."
    )
    p.add_argument("--input", type=str, default=INPUT_DEFAULT, help="Requirement GT JSONL path.")
    p.add_argument("--model", type=str, default="", help="Finetuned CE model path/ref. Defaults to local/HPC known checkpoints.")
    p.add_argument("--output", type=str, default="", help="Output CSV path. Default writes to ce/eval/results.")
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--samples-per-band", type=int, default=SAMPLES_PER_BAND_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH_DEFAULT)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--high-threshold", type=float, default=HIGH_THRESHOLD_DEFAULT)
    p.add_argument("--mid-threshold", type=float, default=MID_THRESHOLD_DEFAULT)
    p.add_argument("--score-calibration-json", type=str, default="", help="Optional posthoc_calibration_affine.json path.")
    p.add_argument("--no-score-calibration", action="store_true", help="Disable auto/apply posthoc score calibration.")
    p.add_argument(
        "--aspect-condition-mode",
        type=str,
        default="legacy",
        choices=("legacy", "long_prefix", "none"),
        help="How to present aspect conditioning to the finetuned model.",
    )
    return p.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()

    input_path = _resolve_path(args.input)
    if not input_path.exists():
        raise RuntimeError(f"Input GT file not found: {input_path}")

    output_path = _resolve_path(args.output) if _clean_text(args.output) else _default_output_path(_resolve_path(args.output_dir))
    model_ref = _resolve_model_ref(_clean_text(args.model) or _default_model_ref())
    device = _pick_device(args.device)
    aspect_condition_mode = clean_aspect_condition_mode(args.aspect_condition_mode)

    high_threshold = _safe_float(args.high_threshold, default=HIGH_THRESHOLD_DEFAULT, minimum=0.0, maximum=1.0)
    mid_threshold = _safe_float(args.mid_threshold, default=MID_THRESHOLD_DEFAULT, minimum=0.0, maximum=1.0)
    if mid_threshold > high_threshold:
        mid_threshold = high_threshold

    rows_all = _load_ground_truth_rows(
        path=input_path,
        high_threshold=high_threshold,
        mid_threshold=mid_threshold,
    )
    if not rows_all:
        raise RuntimeError(f"No usable query/doc/score rows found in {input_path}")

    rows, available = _sample_rows_by_band(
        rows=rows_all,
        samples_per_band=_safe_int(args.samples_per_band, default=SAMPLES_PER_BAND_DEFAULT, minimum=1, maximum=10_000),
        seed=_safe_int(args.seed, default=SEED_DEFAULT, minimum=0, maximum=2_147_483_647),
    )
    if not rows:
        raise RuntimeError("Sampling produced no rows. Check GT score thresholds and input file.")

    calibration_path, calibration_payload = _resolve_score_calibration_json(
        model_ref=model_ref,
        explicit_path=args.score_calibration_json,
        enabled=not bool(args.no_score_calibration),
    )

    print(f"input={input_path}")
    print(f"model={model_ref}")
    print(f"device={device}")
    print(f"aspect_condition_mode={aspect_condition_mode}")
    print(f"available_bands={available}")
    print(f"sampled_rows={len(rows)}")
    if calibration_path is not None:
        print(f"score_calibration_json={calibration_path} loaded={isinstance(calibration_payload, dict)}")

    _score_aspects(
        rows=rows,
        model_ref=model_ref,
        device=device,
        batch_size=_safe_int(args.batch_size, default=BATCH_SIZE_DEFAULT, minimum=1, maximum=4096),
        max_length=_safe_int(args.max_length, default=MAX_LENGTH_DEFAULT, minimum=8, maximum=100_000),
        calibration_payload=calibration_payload,
        aspect_condition_mode=aspect_condition_mode,
    )

    _write_csv(output_path, rows)
    elapsed = time.time() - started
    print(f"saved_csv={output_path}")
    print(f"elapsed_sec={elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
