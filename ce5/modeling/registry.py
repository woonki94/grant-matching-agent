"""Checkpoint-based CE5 architecture selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ce5.modeling.directional_latent_matcher import (
    ARCHITECTURE_TYPE as DIRECTIONAL_ARCHITECTURE_TYPE,
    ModernCEDirectionalLatentMatcher,
)
from ce5.modeling.independent_latent_heads import ModernCELatentHeadModel
from ce5.modeling.independent_pair_aware_heads import (
    ARCHITECTURE_TYPE as INDEPENDENT_PAIR_AWARE_ARCHITECTURE_TYPE,
    ModernCEIndependentPairAwareModel,
)
from ce5.modeling.directional_private_experts import (
    ARCHITECTURE_TYPE as DIRECTIONAL_PRIVATE_ARCHITECTURE_TYPE,
    ModernCEDirectionalPrivateExperts,
)


INDEPENDENT_ARCHITECTURE_TYPE = "independent_latent_heads"


def checkpoint_architecture_type(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> str:
    """Read only enough checkpoint metadata to identify the model class."""

    payload = torch.load(
        Path(checkpoint_path).expanduser(),
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid CE5 checkpoint: expected a dictionary")
    # Original CE5 checkpoints predate explicit architecture metadata.
    return str(payload.get("architecture_type") or INDEPENDENT_ARCHITECTURE_TYPE)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    **pretrained_kwargs: Any,
) -> Any:
    """Restore the correct CE5 model implementation from checkpoint metadata."""

    architecture_type = checkpoint_architecture_type(
        checkpoint_path,
        map_location=map_location,
    )
    if architecture_type == DIRECTIONAL_ARCHITECTURE_TYPE:
        return ModernCEDirectionalLatentMatcher.from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            **pretrained_kwargs,
        )
    if architecture_type == DIRECTIONAL_PRIVATE_ARCHITECTURE_TYPE:
        return ModernCEDirectionalPrivateExperts.from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            **pretrained_kwargs,
        )
    if architecture_type == INDEPENDENT_PAIR_AWARE_ARCHITECTURE_TYPE:
        return ModernCEIndependentPairAwareModel.from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            **pretrained_kwargs,
        )
    if architecture_type == INDEPENDENT_ARCHITECTURE_TYPE:
        return ModernCELatentHeadModel.from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            **pretrained_kwargs,
        )
    raise RuntimeError(f"Unsupported CE5 architecture type: {architecture_type!r}")
