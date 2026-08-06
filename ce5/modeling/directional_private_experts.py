"""Directional latent matcher with a private scoring function per expert."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ce5.modeling.directional_latent_matcher import (
    DirectionalLatentMatcherConfig,
    DirectionalLatentMatcherOutput,
    ModernCEDirectionalLatentMatcher,
)


ARCHITECTURE_TYPE = "directional_private_experts"


class PrivateDeltaScorer(nn.Module):
    """A latent-specific output MLP with no parameters shared across experts."""

    def __init__(self, *, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )
        final_layer = self.layers[-1]
        if isinstance(final_layer, nn.Linear):
            # Stay close to the STS prior while propagating gradients through
            # every private scorer from the first update.
            nn.init.normal_(final_layer.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(final_layer.bias)

    def forward(self, latent: Tensor) -> Tensor:
        return self.layers(latent).squeeze(-1)


class ModernCEDirectionalPrivateExperts(ModernCEDirectionalLatentMatcher):
    """Directional refinement followed by six independent scoring paths.

    Only the output scoring functions differ from directional v1. The encoder,
    token masks, cross-attention blocks, latent interaction, router, residual
    STS path, and training objective remain unchanged.
    """

    architecture_type = ARCHITECTURE_TYPE

    def __init__(
        self,
        *,
        encoder: nn.Module,
        architecture_config: DirectionalLatentMatcherConfig,
        base_sts_expert: nn.Module | None = None,
    ) -> None:
        super().__init__(
            encoder=encoder,
            architecture_config=architecture_config,
            base_sts_expert=base_sts_expert,
        )
        self.delta_scorer = nn.ModuleList(
            PrivateDeltaScorer(
                latent_dim=architecture_config.latent_dim,
                dropout=architecture_config.dropout,
            )
            for _ in range(architecture_config.num_latent_heads)
        )

    def _score_latent_deltas(self, latents: Tensor) -> Tensor:
        if latents.shape[1] != len(self.delta_scorer):
            raise RuntimeError("Latent count does not match private scorer count")
        return torch.stack(
            [
                scorer(latents[:, index, :])
                for index, scorer in enumerate(self.delta_scorer)
            ],
            dim=-1,
        )


__all__ = [
    "ARCHITECTURE_TYPE",
    "DirectionalLatentMatcherConfig",
    "DirectionalLatentMatcherOutput",
    "ModernCEDirectionalPrivateExperts",
    "PrivateDeltaScorer",
]
