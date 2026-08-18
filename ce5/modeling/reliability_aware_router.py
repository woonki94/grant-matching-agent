"""Reliability-aware residual routing over frozen CE5 independent-v2 experts.

This experiment preserves an independent-v2 checkpoint as its exact epoch-zero
predictor.  Each expert exposes an evidence token, predicts its own absolute
error, and participates in a small expert-set attention block.  The resulting
router correction is zero-initialized and added to the frozen v2 gate logits.

Teacher labels are deliberately absent from this module.  They are used only
by the separate training entry point to supervise predicted reliability and
routing.  Inference depends exclusively on the input pair and model outputs.
"""

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


ARCHITECTURE_TYPE = "independent_reliability_aware_router"
CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ReliabilityAwareRouterConfig(IndependentPairAwareConfig):
    """Independent-v2 configuration plus evidence-aware routing dimensions."""

    router_hidden_dim: int = 128
    router_attention_heads: int = 4
    router_ffn_dim: int = 256
    router_dropout: float = 0.1
    router_logit_clip: float = 12.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.router_hidden_dim < 1:
            raise ValueError("router_hidden_dim must be positive")
        if self.router_attention_heads < 1:
            raise ValueError("router_attention_heads must be positive")
        if self.router_hidden_dim % self.router_attention_heads != 0:
            raise ValueError(
                "router_hidden_dim must be divisible by router_attention_heads"
            )
        if self.router_ffn_dim < 1:
            raise ValueError("router_ffn_dim must be positive")
        if not 0.0 <= self.router_dropout < 1.0:
            raise ValueError("router_dropout must be in [0, 1)")
        if self.router_logit_clip <= 0.0:
            raise ValueError("router_logit_clip must be positive")


@dataclass
class ReliabilityAwareRouterOutput:
    """Predictions plus the diagnostics needed for supervised routing."""

    scores: Tensor
    logits: Tensor
    head_scores: Tensor
    head_logits: Tensor
    gate_weights: Tensor
    attention_weights: Tensor
    target_attention_weights: Tensor
    candidate_attention_weights: Tensor
    base_gate_weights: Tensor
    predicted_expert_errors: Tensor
    routing_correction_logits: Tensor
    expert_evidence_tokens: Tensor


class ModernCEReliabilityAwareRouterModel(ModernCEIndependentPairAwareModel):
    """Independent-v2 with explicit expert evidence and reliability routing."""

    architecture_type = ARCHITECTURE_TYPE

    def __init__(
        self,
        *,
        encoder: nn.Module,
        architecture_config: ReliabilityAwareRouterConfig,
        base_sts_expert: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(
            encoder=encoder,
            architecture_config=architecture_config,
            base_sts_expert=base_sts_expert,
        )
        hidden_size = int(getattr(encoder.config, "hidden_size"))
        router_dim = architecture_config.router_hidden_dim

        self.base_evidence_projection: Optional[nn.Module] = None
        if architecture_config.use_base_sts_expert:
            self.base_evidence_projection = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, router_dim),
                nn.GELU(),
            )
        self.latent_evidence_projections = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(architecture_config.expert_ffn_dim),
                nn.Linear(architecture_config.expert_ffn_dim, router_dim),
                nn.GELU(),
            )
            for _ in range(architecture_config.num_latent_heads)
        )
        self.expert_identity = nn.Parameter(torch.empty(self.num_experts, router_dim))
        nn.init.normal_(self.expert_identity, mean=0.0, std=0.02)

        # Evidence + raw logit + probability + original-v2 gate probability.
        reliability_feature_dim = router_dim + 3
        self.reliability_feature_norm = nn.LayerNorm(reliability_feature_dim)
        self.reliability_predictor = nn.Sequential(
            nn.Linear(reliability_feature_dim, router_dim),
            nn.GELU(),
            nn.Dropout(architecture_config.router_dropout),
            nn.Linear(router_dim, 1),
        )

        # The predicted error is appended only after the reliability estimate is
        # formed, preventing a circular dependency inside the reliability head.
        router_feature_dim = router_dim + 4
        self.router_token_projection = nn.Sequential(
            nn.LayerNorm(router_feature_dim),
            nn.Linear(router_feature_dim, router_dim),
            nn.GELU(),
        )
        self.expert_set_attention = nn.TransformerEncoderLayer(
            d_model=router_dim,
            nhead=architecture_config.router_attention_heads,
            dim_feedforward=architecture_config.router_ffn_dim,
            dropout=architecture_config.router_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.router_output_norm = nn.LayerNorm(router_dim)
        self.router_correction = nn.Linear(router_dim, 1)
        nn.init.zeros_(self.router_correction.weight)
        nn.init.zeros_(self.router_correction.bias)
        self._reliability_router_only_training = False

    def _expert_evidence(
        self,
        cls_state: Tensor,
        latent_private_evidence: Tensor,
    ) -> Tensor:
        """Project each expert's private pre-logit representation."""

        latent_tokens = torch.stack(
            [
                projection(latent_private_evidence[:, index])
                for index, projection in enumerate(
                    self.latent_evidence_projections
                )
            ],
            dim=1,
        )
        if not self.architecture_config.use_base_sts_expert:
            return latent_tokens
        if self.base_evidence_projection is None:
            raise RuntimeError("The configured base evidence projection is missing")
        base_token = self.base_evidence_projection(cls_state).unsqueeze(1)
        return torch.cat((base_token, latent_tokens), dim=1)

    @staticmethod
    def _run_latent_expert(
        expert: nn.Module,
        hidden_states: Tensor,
        *,
        target_mask: Tensor,
        candidate_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run an independent-v2 expert while retaining its private evidence."""

        target, target_attention = expert.target_pooler(hidden_states, target_mask)
        candidate, candidate_attention = expert.candidate_pooler(
            hidden_states,
            candidate_mask,
        )
        interaction = torch.cat(
            (target, candidate, torch.abs(target - candidate), target * candidate),
            dim=-1,
        ).flatten(start_dim=1)
        private_evidence = expert.interaction_norm(interaction)
        scorer_layers = tuple(expert.scorer.children())
        for layer in scorer_layers[:-1]:
            private_evidence = layer(private_evidence)
        logit = scorer_layers[-1](private_evidence).squeeze(-1)
        target_attention_mean = target_attention.mean(dim=1)
        candidate_attention_mean = candidate_attention.mean(dim=1)
        combined_attention = 0.5 * (
            target_attention_mean + candidate_attention_mean
        )
        return (
            logit,
            combined_attention,
            target_attention_mean,
            candidate_attention_mean,
            private_evidence,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
        candidate_mask: Optional[Tensor] = None,
        **encoder_kwargs: Any,
    ) -> ReliabilityAwareRouterOutput:
        forbidden_teacher_inputs = {
            key
            for key in ("labels", "targets", "teacher_scores", "score_bins")
            if key in encoder_kwargs
        }
        if forbidden_teacher_inputs:
            raise ValueError(
                "Reliability-aware inference does not accept teacher-derived "
                f"inputs: {sorted(forbidden_teacher_inputs)}"
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if target_mask is None or candidate_mask is None:
            raise ValueError(
                "Reliability-aware routing requires target_mask and candidate_mask"
            )
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [batch, sequence]")
        if target_mask.shape != input_ids.shape or candidate_mask.shape != input_ids.shape:
            raise ValueError("Pair masks must match input_ids shape")
        target_mask = target_mask.bool() & attention_mask.bool()
        candidate_mask = candidate_mask.bool() & attention_mask.bool()
        if torch.any(target_mask & candidate_mask):
            raise ValueError("target_mask and candidate_mask must not overlap")
        if torch.any(target_mask.sum(dim=-1) == 0):
            raise ValueError("Every example must retain at least one target token")
        if torch.any(candidate_mask.sum(dim=-1) == 0):
            raise ValueError("Every example must retain at least one candidate token")

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **encoder_kwargs,
        )
        hidden_states = encoder_outputs.last_hidden_state
        cls_state = hidden_states[:, 0]
        expert_results = [
            self._run_latent_expert(
                expert,
                hidden_states,
                target_mask=target_mask,
                candidate_mask=candidate_mask,
            )
            for expert in self.latent_heads
        ]
        latent_logits = torch.stack([result[0] for result in expert_results], dim=-1)
        attention_weights = torch.stack([result[1] for result in expert_results], dim=1)
        target_attention_weights = torch.stack(
            [result[2] for result in expert_results],
            dim=1,
        )
        candidate_attention_weights = torch.stack(
            [result[3] for result in expert_results],
            dim=1,
        )
        latent_private_evidence = torch.stack(
            [result[4] for result in expert_results],
            dim=1,
        )

        expert_logits = latent_logits
        if self.architecture_config.use_base_sts_expert:
            if self.base_sts_expert is None:
                raise RuntimeError("The configured base STS expert is missing")
            base_logit = self.base_sts_expert(cls_state).reshape(-1, 1)
            expert_logits = torch.cat((base_logit, latent_logits), dim=-1)

        evidence_tokens = self._expert_evidence(
            cls_state,
            latent_private_evidence,
        )
        routing_logits = expert_logits.detach().clamp(
            min=-self.architecture_config.router_logit_clip,
            max=self.architecture_config.router_logit_clip,
        )
        expert_scores = torch.sigmoid(routing_logits)
        base_gate_logits = self.gate(self.gate_norm(cls_state)).detach()
        base_gate_weights = torch.softmax(base_gate_logits, dim=-1)
        scalar_features = torch.stack(
            (routing_logits, expert_scores, base_gate_weights),
            dim=-1,
        )

        reliability_features = torch.cat((evidence_tokens, scalar_features), dim=-1)
        predicted_errors = torch.sigmoid(
            self.reliability_predictor(
                self.reliability_feature_norm(reliability_features)
            ).squeeze(-1)
        )
        router_features = torch.cat(
            # Reliability is a separately supervised quantity, not a hidden
            # side-channel for the routing loss.
            (reliability_features, predicted_errors.detach().unsqueeze(-1)),
            dim=-1,
        )
        router_tokens = (
            self.router_token_projection(router_features)
            + self.expert_identity.unsqueeze(0)
        )
        contextual_tokens = self.expert_set_attention(router_tokens)
        correction_logits = self.router_correction(
            self.router_output_norm(contextual_tokens)
        ).squeeze(-1)

        gate_logits = self._apply_head_dropout(base_gate_logits + correction_logits)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        logits = torch.sum(gate_weights * expert_logits, dim=-1)
        return ReliabilityAwareRouterOutput(
            scores=torch.sigmoid(logits),
            logits=logits,
            head_scores=torch.sigmoid(expert_logits),
            head_logits=expert_logits,
            gate_weights=gate_weights,
            attention_weights=attention_weights,
            target_attention_weights=target_attention_weights,
            candidate_attention_weights=candidate_attention_weights,
            base_gate_weights=base_gate_weights,
            predicted_expert_errors=predicted_errors,
            routing_correction_logits=correction_logits,
            expert_evidence_tokens=evidence_tokens,
        )

    def set_reliability_router_trainable_only(self) -> None:
        """Freeze independent-v2 and train only new reliability/router modules."""

        for parameter in self.parameters():
            parameter.requires_grad = False
        trainable_modules = (
            self.base_evidence_projection,
            self.latent_evidence_projections,
            self.reliability_feature_norm,
            self.reliability_predictor,
            self.router_token_projection,
            self.expert_set_attention,
            self.router_output_norm,
            self.router_correction,
        )
        for module in trainable_modules:
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.expert_identity.requires_grad = True
        self._reliability_router_only_training = True

    def train(self, mode: bool = True) -> "ModernCEReliabilityAwareRouterModel":
        super().train(mode)
        if mode and self._reliability_router_only_training:
            self.encoder.eval()
            self.latent_heads.eval()
            self.gate_norm.eval()
            self.gate.eval()
            if self.base_sts_expert is not None:
                self.base_sts_expert.eval()
        return self

    def _apply_head_dropout(self, gate_logits: Tensor) -> Tensor:
        if self._reliability_router_only_training:
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
        router_attention_heads: int = 4,
        router_ffn_dim: int = 256,
        router_dropout: float = 0.1,
        router_logit_clip: float = 12.0,
        **pretrained_kwargs: Any,
    ) -> "ModernCEReliabilityAwareRouterModel":
        config = ReliabilityAwareRouterConfig(
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
            router_attention_heads=router_attention_heads,
            router_ffn_dim=router_ffn_dim,
            router_dropout=router_dropout,
            router_logit_clip=router_logit_clip,
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
        router_attention_heads: int = 4,
        router_ffn_dim: int = 256,
        router_dropout: float = 0.1,
        router_logit_clip: float = 12.0,
        **pretrained_kwargs: Any,
    ) -> "ModernCEReliabilityAwareRouterModel":
        source = ModernCEIndependentPairAwareModel.from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            **pretrained_kwargs,
        )
        config = ReliabilityAwareRouterConfig(
            **asdict(source.architecture_config),
            router_hidden_dim=router_hidden_dim,
            router_attention_heads=router_attention_heads,
            router_ffn_dim=router_ffn_dim,
            router_dropout=router_dropout,
            router_logit_clip=router_logit_clip,
        )
        model = cls(
            encoder=source.encoder,
            architecture_config=config,
            base_sts_expert=source.base_sts_expert,
        )
        incompatible = model.load_state_dict(source.state_dict(), strict=False)
        unexpected = tuple(incompatible.unexpected_keys)
        allowed_prefixes = (
            "base_evidence_projection.",
            "latent_evidence_projections.",
            "expert_identity",
            "reliability_feature_norm.",
            "reliability_predictor.",
            "router_token_projection.",
            "expert_set_attention.",
            "router_output_norm.",
            "router_correction.",
        )
        invalid_missing = tuple(
            name
            for name in incompatible.missing_keys
            if not name.startswith(allowed_prefixes)
        )
        if unexpected or invalid_missing:
            raise RuntimeError(
                "Could not initialize reliability-aware router from independent-v2: "
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
    ) -> "ModernCEReliabilityAwareRouterModel":
        payload = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location=map_location,
            weights_only=True,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid reliability-aware router checkpoint")
        if payload.get("architecture_type") != cls.architecture_type:
            raise RuntimeError(
                f"Expected {cls.architecture_type!r}, "
                f"found {payload.get('architecture_type')!r}"
            )
        version = int(payload.get("format_version", 0))
        if version != CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError(
                f"Unsupported reliability-router checkpoint version {version}"
            )
        raw_config = payload.get("architecture_config")
        state_dict = payload.get("state_dict")
        if not isinstance(raw_config, dict) or not isinstance(state_dict, dict):
            raise RuntimeError(
                "Reliability-router checkpoint config or state_dict is missing"
            )
        config = ReliabilityAwareRouterConfig(**raw_config)
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
    "ReliabilityAwareRouterConfig",
    "ReliabilityAwareRouterOutput",
    "ModernCEReliabilityAwareRouterModel",
]
