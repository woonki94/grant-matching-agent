"""Utilities for preserving the two sides of a tokenized text pair."""

from __future__ import annotations

from typing import Any, MutableMapping

import torch
from torch import Tensor


TARGET_MASK_KEY = "target_mask"
CANDIDATE_MASK_KEY = "candidate_mask"


def add_pair_sequence_masks(encoded: MutableMapping[str, Tensor], tokenizer: Any) -> None:
    """Add boolean masks for the first and second input sequences in-place.

    Fast Hugging Face tokenizers retain a sequence id (0, 1, or ``None``) for
    every encoded token. Using those ids avoids treating separators and padding
    as evidence and works for tokenizer-specific pair layouts.
    """

    input_ids = encoded.get("input_ids")
    attention_mask = encoded.get("attention_mask")
    if input_ids is None or attention_mask is None:
        raise ValueError("Pair encoding must contain input_ids and attention_mask")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("Pair encoding tensors must have shape [batch, sequence]")

    sequence_ids_method = getattr(encoded, "sequence_ids", None)
    if not callable(sequence_ids_method):
        raise RuntimeError(
            "The directional latent matcher requires a fast tokenizer that "
            "exposes BatchEncoding.sequence_ids()."
        )

    target_rows: list[list[bool]] = []
    candidate_rows: list[list[bool]] = []
    sequence_length = int(input_ids.shape[1])
    for batch_index in range(int(input_ids.shape[0])):
        sequence_ids = sequence_ids_method(batch_index)
        if sequence_ids is None or len(sequence_ids) != sequence_length:
            raise RuntimeError(
                "Tokenizer sequence ids are missing or do not match the encoded length"
            )
        target = [sequence_id == 0 for sequence_id in sequence_ids]
        candidate = [sequence_id == 1 for sequence_id in sequence_ids]
        if not any(target) or not any(candidate):
            raise RuntimeError(
                "Tokenization removed an entire side of a text pair; increase --max-length"
            )
        target_rows.append(target)
        candidate_rows.append(candidate)

    target_mask = torch.tensor(target_rows, dtype=torch.bool)
    candidate_mask = torch.tensor(candidate_rows, dtype=torch.bool)
    valid_tokens = attention_mask.bool()
    encoded[TARGET_MASK_KEY] = target_mask & valid_tokens
    encoded[CANDIDATE_MASK_KEY] = candidate_mask & valid_tokens


def tokenize_pairs_with_masks(
    tokenizer: Any,
    first_texts: list[str],
    second_texts: list[str],
    *,
    max_length: int,
) -> MutableMapping[str, Tensor]:
    """Tokenize paired texts and attach directional sequence masks."""

    encoded = tokenizer(
        first_texts,
        second_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    add_pair_sequence_masks(encoded, tokenizer)
    return encoded
