"""Source model architectures for CE5 experiments.

This package is deliberately named ``modeling``.  The sibling ``ce5/models``
directory is reserved for generated checkpoints and is ignored by Git.
"""

from ce5.modeling.independent_latent_heads import (
    DEFAULT_MODEL_ID,
    LatentHeadConfig,
    LatentHeadOutput,
    ModernCELatentHeadModel,
)
from ce5.modeling.directional_latent_matcher import (
    DirectionalLatentMatcherConfig,
    DirectionalLatentMatcherOutput,
    ModernCEDirectionalLatentMatcher,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "LatentHeadConfig",
    "LatentHeadOutput",
    "ModernCELatentHeadModel",
    "DirectionalLatentMatcherConfig",
    "DirectionalLatentMatcherOutput",
    "ModernCEDirectionalLatentMatcher",
]
