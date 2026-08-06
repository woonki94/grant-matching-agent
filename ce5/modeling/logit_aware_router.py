"""Logit-aware dense router layered over CE5 independent-v2 experts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn
from transformers import AutoModelForSequenceClassification

from ce5.modeling.independent_pair_aware_heads import (
    IndependentPairAwareConfig,
    ModernCEIndependentPairAwareModel,
)


ARCHITECTURE_TYPE = "independent_logit_aware_router"
CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class LogitAwareRouterConfig(IndependentPairAwareConfig):
    """Independent-v2 configuration plus the router-only extension."""

    router_hidden_dim: int = 128
    router_dropout: float = 0.1
    router_logit_clip: float = 12.0
    detach_expert_logits: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.router_hidden_dim < 1:
            raise ValueError("router_hidden_dim must be positive")
        if not 0.0 <= self.router_dropout < 1.0:
            raise ValueError("router_dropout must be in [0, 1)")
        if self.router_logit_clip <= 0.0:
            raise ValueError("router_logit_clip must be positive")


class ModernCELogitAwareRouterModel(ModernCEIndependentPairAwareModel):
    """Independent-v2 with a feature-aware residual correction to its gate.

    The existing CLS-linear router remains intact.  A small MLP observes CLS,
    detached expert logits, probabilities, expert/base disagreements, and
    global disagreement statistics.  Its final layer starts at zero, so a new
    instance initialized from an independent-v2 checkpoint produces exactly
    the same scores before router training.
    """

    architecture_type = ARCHITECTURE_TYPE

    def __init__(
        self,
        *,
        encoder: nn.Module,
        architecture_config: LogitAwareRouterConfig,
        base_sts_expert: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(
            encoder=encoder,
            architecture_config=architecture_config,
            base_sts_expert=base_sts_expert,
        )
        hidden_size = int(getattr(encoder.config, "hidden_size"))
        num_experts = self.num_experts
        self.router_cls_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, architecture_config.router_hidden_dim),
            nn.GELU(),
        )
        # raw logits + probabilities + distance from base/reference + four stats
        router_feature_dim = architecture_config.router_hidden_dim + 3 * num_experts + 4
        self.router_feature_norm = nn.LayerNorm(router_feature_dim)
        self.router_correction = nn.Sequential(
            nn.Linear(router_feature_dim, architecture_config.router_hidden_dim),
            nn.GELU(),
            nn.Dropout(architecture_config.router_dropout),
            nn.Linear(architecture_config.router_hidden_dim, num_experts),
        )
        final_layer = self.router_correction[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
        self._router_only_training = False

    def _compute_gate_logits(
        self,
        cls_state: Tensor,
        expert_logits: Tensor,
    ) -> Tensor:
        base_gate_logits = self.gate(self.gate_norm(cls_state))
        routing_logits = (
            expert_logits.detach()
            if self.architecture_config.detach_expert_logits
            else expert_logits
        )
        routing_logits = routing_logits.clamp(
            min=-self.architecture_config.router_logit_clip,
            max=self.architecture_config.router_logit_clip,
        )
        probabilities = torch.sigmoid(routing_logits)
        if self.architecture_config.use_base_sts_expert:
            reference = routing_logits[:, :1]
        else:
            reference = routing_logits.mean(dim=-1, keepdim=True)
        disagreements = routing_logits - reference
        statistics = torch.cat(
            (
                routing_logits.mean(dim=-1, keepdim=True),
                routing_logits.std(dim=-1, keepdim=True, unbiased=False),
                routing_logits.amin(dim=-1, keepdim=True),
                routing_logits.amax(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        features = torch.cat(
            (
                self.router_cls_projection(cls_state),
                routing_logits,
                probabilities,
                disagreements,
                statistics,
            ),
            dim=-1,
        )
        correction = self.router_correction(self.router_feature_norm(features))
        return base_gate_logits + correction

    def set_router_trainable_only(self) -> None:
        """Freeze the backbone and experts while retaining both router stages."""

        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in (
            self.gate_norm,
            self.gate,
            self.router_cls_projection,
            self.router_feature_norm,
            self.router_correction,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True
        self._router_only_training = True

    def train(self, mode: bool = True) -> "ModernCELogitAwareRouterModel":
        """Keep frozen feature producers deterministic during router-only fitting."""

        super().train(mode)
        if mode and self._router_only_training:
            self.encoder.eval()
            self.latent_heads.eval()
            if self.base_sts_expert is not None:
                self.base_sts_expert.eval()
        return self

    def _apply_head_dropout(self, gate_logits: Tensor) -> Tensor:
        if self._router_only_training:
            return gate_logits
        return super()._apply_head_dropout(gate_logits)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        num_latent_heads: int = 6,
        num_queries_per_side: int = 2,
        attention_dim: int = 128,
        head_dim: int = 192,
        expert_ffn_dim: int = 384,
        dropout: float = 0.1,
        head_dropout: float = 0.0,
        use_base_sts_expert: bool = True,
        base_sts_gate_bias: float = 4.0,
        router_hidden_dim: int = 128,
        router_dropout: float = 0.1,
        router_logit_clip: float = 12.0,
        detach_expert_logits: bool = True,
        **pretrained_kwargs: Any,
    ) -> "ModernCELogitAwareRouterModel":
        config = LogitAwareRouterConfig(
            backbone_model_id=model_id,
            num_latent_heads=num_latent_heads,
            num_queries_per_side=num_queries_per_side,
            attention_dim=attention_dim,
            head_dim=head_dim,
            expert_ffn_dim=expert_ffn_dim,
            dropout=dropout,
            head_dropout=head_dropout,
            use_base_sts_expert=use_base_sts_expert,
            base_sts_gate_bias=base_sts_gate_bias,
            router_hidden_dim=router_hidden_dim,
            router_dropout=router_dropout,
            router_logit_clip=router_logit_clip,
            detach_expert_logits=detach_expert_logits,
        )
        source_model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            **pretrained_kwargs,
        )
        encoder = source_model.base_model
        base_expert: Optional[nn.Module] = None
        if use_base_sts_expert:
            required = ("head", "drop", "classifier")
            missing = [
                name
                for name in required
                if not isinstance(getattr(source_model, name, None), nn.Module)
            ]
            if missing:
                raise RuntimeError(
                    "The requested checkpoint does not expose ModernCE's expected "
                    f"classification modules: {', '.join(missing)}"
                )
            base_expert = nn.Sequential(
                source_model.head,
                source_model.drop,
                source_model.classifier,
            )
        return cls(
            encoder=encoder,
            architecture_config=config,
            base_sts_expert=base_expert,
        )

    @classmethod
    def from_independent_v2_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        router_hidden_dim: int = 128,
        router_dropout: float = 0.1,
        router_logit_clip: float = 12.0,
        detach_expert_logits: bool = True,
        **pretrained_kwargs: Any,
    ) -> "ModernCELogitAwareRouterModel":
        source = ModernCEIndependentPairAwareModel.from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            **pretrained_kwargs,
        )
        config = LogitAwareRouterConfig(
            **asdict(source.architecture_config),
            router_hidden_dim=router_hidden_dim,
            router_dropout=router_dropout,
            router_logit_clip=router_logit_clip,
            detach_expert_logits=detach_expert_logits,
        )
        model = cls(
            encoder=source.encoder,
            architecture_config=config,
            base_sts_expert=source.base_sts_expert,
        )
        incompatible = model.load_state_dict(source.state_dict(), strict=False)
        unexpected = tuple(incompatible.unexpected_keys)
        invalid_missing = tuple(
            name
            for name in incompatible.missing_keys
            if not name.startswith("router_")
        )
        if unexpected or invalid_missing:
            raise RuntimeError(
                "Could not initialize logit-aware router from independent-v2: "
                f"missing={invalid_missing}, unexpected={unexpected}"
            )
        return model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        **pretrained_kwargs: Any,
    ) -> "ModernCELogitAwareRouterModel":
        payload = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location=map_location,
            weights_only=True,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid logit-aware router checkpoint")
        if payload.get("architecture_type") != cls.architecture_type:
            raise RuntimeError(
                f"Expected {cls.architecture_type!r}, "
                f"found {payload.get('architecture_type')!r}"
            )
        version = int(payload.get("format_version", 0))
        if version != CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError(f"Unsupported logit-router checkpoint version {version}")
        raw_config = payload.get("architecture_config")
        state_dict = payload.get("state_dict")
        if not isinstance(raw_config, dict) or not isinstance(state_dict, dict):
            raise RuntimeError("Logit-router checkpoint config or state_dict is missing")
        config = LogitAwareRouterConfig(**raw_config)
        constructor_args = asdict(config)
        model_id = str(constructor_args.pop("backbone_model_id"))
        model = cls.from_pretrained(
            model_id,
            **constructor_args,
            **pretrained_kwargs,
        )
        model.load_state_dict(state_dict, strict=True)
        return model


__all__ = [
    "ARCHITECTURE_TYPE",
    "LogitAwareRouterConfig",
    "ModernCELogitAwareRouterModel",
]
