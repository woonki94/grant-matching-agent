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
from ce5.modeling.directional_private_experts import (
    ModernCEDirectionalPrivateExperts,
    PrivateDeltaScorer,
)
from ce5.modeling.independent_pair_aware_heads import (
    IndependentPairAwareConfig,
    IndependentPairAwareExpert,
    IndependentPairAwareOutput,
    ModernCEIndependentPairAwareModel,
    PrivateSidePooler,
)
from ce5.modeling.logit_aware_router import (
    LogitAwareRouterConfig,
    ModernCELogitAwareRouterModel,
)
from ce5.modeling.reliability_aware_router import (
    ModernCEReliabilityAwareRouterModel,
    ReliabilityAwareRouterConfig,
    ReliabilityAwareRouterOutput,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "LatentHeadConfig",
    "LatentHeadOutput",
    "ModernCELatentHeadModel",
    "DirectionalLatentMatcherConfig",
    "DirectionalLatentMatcherOutput",
    "ModernCEDirectionalLatentMatcher",
    "ModernCEDirectionalPrivateExperts",
    "PrivateDeltaScorer",
    "IndependentPairAwareConfig",
    "IndependentPairAwareExpert",
    "IndependentPairAwareOutput",
    "ModernCEIndependentPairAwareModel",
    "PrivateSidePooler",
    "LogitAwareRouterConfig",
    "ModernCELogitAwareRouterModel",
    "ReliabilityAwareRouterConfig",
    "ReliabilityAwareRouterOutput",
    "ModernCEReliabilityAwareRouterModel",
]
