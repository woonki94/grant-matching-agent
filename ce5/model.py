"""Compatibility imports for the original CE5 model.

New model variants live under :mod:`ce5.modeling`.  This module intentionally
keeps the historical import path stable so existing training scripts,
evaluation scripts, tests, and checkpoints continue to work unchanged.
"""

from ce5.modeling.independent_latent_heads import (
    CHECKPOINT_FORMAT_VERSION,
    DEFAULT_MODEL_ID,
    SUPPORTED_CHECKPOINT_FORMAT_VERSIONS,
    LatentAttentionHead,
    LatentHeadConfig,
    LatentHeadOutput,
    ModernCELatentHeadModel,
)

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "DEFAULT_MODEL_ID",
    "SUPPORTED_CHECKPOINT_FORMAT_VERSIONS",
    "LatentAttentionHead",
    "LatentHeadConfig",
    "LatentHeadOutput",
    "ModernCELatentHeadModel",
]
