from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List, Sequence, Tuple


DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "train_cross_encoder"
    / "models"
    / "bge-reranker-base-finetuned"
    / "checkpoint-47022"
)


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _safe_limit(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _to_bool(value: object) -> bool:
    v = _clean_text(value).lower()
    return v in {"1", "true", "yes", "y", "on"}


def _clamp_unit(value: object) -> float:
    try:
        s = float(value)
    except Exception:
        return 0.0
    if s < 0.0:
        return 0.0
    if s > 1.0:
        return 1.0
    return s


class SpecializationCrossEncoderScorer:
    """
    Lazy-loaded scorer for specialization-pair relevance using a fine-tuned cross-encoder.

    The model is loaded on first use and reused for subsequent calls.
    """

    def __init__(
        self,
        *,
        model_dir: str = "",
        batch_size: int = 64,
        max_length: int = 256,
        cpu_only: bool = False,
    ):
        env_model_dir = _clean_text(os.getenv("MATCH_CROSS_ENCODER_MODEL_DIR"))
        self.model_dir = _clean_text(model_dir) or env_model_dir or str(DEFAULT_MODEL_DIR)
        self.batch_size = _safe_limit(
            os.getenv("MATCH_CROSS_ENCODER_BATCH_SIZE", batch_size),
            default=int(batch_size),
            minimum=1,
            maximum=4096,
        )
        self.max_length = _safe_limit(
            os.getenv("MATCH_CROSS_ENCODER_MAX_LENGTH", max_length),
            default=int(max_length),
            minimum=32,
            maximum=4096,
        )
        self.cpu_only = bool(cpu_only) or _to_bool(os.getenv("MATCH_CROSS_ENCODER_CPU_ONLY"))

        self._load_lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._device = None
        self._device_name = ""
        self._resolved_model_dir = ""

    @property
    def resolved_model_dir(self) -> str:
        return self._resolved_model_dir or self.model_dir

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._device is not None:
            return
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None and self._device is not None:
                return

            from train_cross_encoder.infer_bge_reranker import (
                _load_model,
                _resolve_model_dir,
                _select_device,
            )

            resolved = _resolve_model_dir(self.model_dir)
            device, device_name = _select_device(cpu_only=self.cpu_only)
            tokenizer, model = _load_model(resolved, device, device_name)

            self._resolved_model_dir = str(resolved)
            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            self._device_name = device_name

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """
        Score (query, doc) pairs and return unit-clamped relevance scores in [0, 1].
        """
        pair_list = list(pairs or [])
        if not pair_list:
            return []

        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None and self._device is not None

        rows = []
        row_idx = []
        out = [0.0] * len(pair_list)
        for i, (query, doc) in enumerate(pair_list):
            q = _clean_text(query)
            d = _clean_text(doc)
            if not q or not d:
                continue
            row_idx.append(i)
            rows.append({"query": q, "doc": d})

        if not rows:
            return out

        from train_cross_encoder.infer_bge_reranker import _score_pairs

        raw_scores = _score_pairs(
            tokenizer=self._tokenizer,
            model=self._model,
            device=self._device,
            pairs=rows,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        for i, score in zip(row_idx, raw_scores):
            out[i] = _clamp_unit(score)
        return out
