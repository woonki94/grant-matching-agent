"""Directional interacting-latent matcher for capability coverage scoring.

The pretrained cross encoder still runs once. A small latent module then
decomposes the first sequence into learned requirement slots, lets those slots
communicate, and uses them to retrieve evidence from the second sequence.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn
from transformers import AutoModelForSequenceClassification


DEFAULT_MODEL_ID = "dleemiller/ModernCE-base-sts"
ARCHITECTURE_TYPE = "directional_latent_matcher"
CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class DirectionalLatentMatcherConfig:
    """Inference architecture saved in a directional matcher checkpoint."""

    backbone_model_id: str = DEFAULT_MODEL_ID
    num_latent_heads: int = 6
    latent_dim: int = 256
    num_refinement_blocks: int = 2
    cross_attention_heads: int = 8
    self_attention_heads: int = 4
    ffn_dim: int = 768
    dropout: float = 0.1
    latent_dropout: float = 0.1
    use_base_sts_expert: bool = True
    initial_latent_contribution: float = 0.1

    def __post_init__(self) -> None:
        if self.num_latent_heads < 1:
            raise ValueError("num_latent_heads must be at least 1")
        if self.latent_dim < 1 or self.ffn_dim < 1:
            raise ValueError("latent_dim and ffn_dim must be positive")
        if self.num_refinement_blocks < 1:
            raise ValueError("num_refinement_blocks must be at least 1")
        if self.cross_attention_heads < 1 or self.self_attention_heads < 1:
            raise ValueError("attention head counts must be positive")
        if self.latent_dim % self.cross_attention_heads:
            raise ValueError("latent_dim must be divisible by cross_attention_heads")
        if self.latent_dim % self.self_attention_heads:
            raise ValueError("latent_dim must be divisible by self_attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.latent_dropout < 1.0:
            raise ValueError("latent_dropout must be in [0, 1)")
        if not 0.0 < self.initial_latent_contribution < 1.0:
            raise ValueError("initial_latent_contribution must be in (0, 1)")


@dataclass
class DirectionalLatentMatcherOutput:
    """Predictions plus routing and attention diagnostics."""

    scores: Tensor
    logits: Tensor
    head_scores: Tensor
    head_logits: Tensor
    gate_weights: Tensor
    attention_weights: Tensor
    target_attention_weights: Tensor
    candidate_attention_weights: Tensor
    routing_weights: Tensor
    latent_contribution: Tensor


class DirectionalRefinementBlock(nn.Module):
    """One target -> latent interaction -> candidate refinement round."""

    def __init__(
        self,
        *,
        latent_dim: int,
        cross_attention_heads: int,
        self_attention_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.target_query_norm = nn.LayerNorm(latent_dim)
        self.target_memory_norm = nn.LayerNorm(latent_dim)
        self.target_attention = nn.MultiheadAttention(
            latent_dim,
            cross_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_query_norm = nn.LayerNorm(latent_dim)
        self.self_attention = nn.MultiheadAttention(
            latent_dim,
            self_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.candidate_query_norm = nn.LayerNorm(latent_dim)
        self.candidate_memory_norm = nn.LayerNorm(latent_dim)
        self.candidate_attention = nn.MultiheadAttention(
            latent_dim,
            cross_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, latent_dim),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        latents: Tensor,
        target_memory: Tensor,
        candidate_memory: Tensor,
        *,
        target_mask: Tensor,
        candidate_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        target_update, target_attention = self.target_attention(
            self.target_query_norm(latents),
            self.target_memory_norm(target_memory),
            self.target_memory_norm(target_memory),
            key_padding_mask=~target_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        latents = latents + self.residual_dropout(target_update)

        normalized_latents = self.self_query_norm(latents)
        interaction_update, _ = self.self_attention(
            normalized_latents,
            normalized_latents,
            normalized_latents,
            need_weights=False,
        )
        latents = latents + self.residual_dropout(interaction_update)

        candidate_update, candidate_attention = self.candidate_attention(
            self.candidate_query_norm(latents),
            self.candidate_memory_norm(candidate_memory),
            self.candidate_memory_norm(candidate_memory),
            key_padding_mask=~candidate_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        latents = latents + self.residual_dropout(candidate_update)
        latents = latents + self.residual_dropout(self.ffn(self.ffn_norm(latents)))

        # MultiheadAttention returns [batch, attention heads, latents, tokens].
        return (
            latents,
            target_attention.mean(dim=1),
            candidate_attention.mean(dim=1),
        )


class ModernCEDirectionalLatentMatcher(nn.Module):
    """ModernCE plus a lightweight directional latent reasoning module."""

    architecture_type = ARCHITECTURE_TYPE
    requires_pair_masks = True

    def __init__(
        self,
        *,
        encoder: nn.Module,
        architecture_config: DirectionalLatentMatcherConfig,
        base_sts_expert: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.architecture_config = architecture_config
        self.base_sts_expert = base_sts_expert

        encoder_config = getattr(encoder, "config", None)
        hidden_size = int(getattr(encoder_config, "hidden_size", 0) or 0)
        if hidden_size <= 0:
            raise ValueError("The encoder config must expose a positive hidden_size")
        if architecture_config.use_base_sts_expert and base_sts_expert is None:
            raise ValueError("use_base_sts_expert=True requires a base STS expert")

        latent_dim = architecture_config.latent_dim
        self.target_projection = nn.Linear(hidden_size, latent_dim)
        self.candidate_projection = nn.Linear(hidden_size, latent_dim)
        self.latent_queries = nn.Parameter(
            torch.empty(architecture_config.num_latent_heads, latent_dim)
        )
        self.refinement_blocks = nn.ModuleList(
            DirectionalRefinementBlock(
                latent_dim=latent_dim,
                cross_attention_heads=architecture_config.cross_attention_heads,
                self_attention_heads=architecture_config.self_attention_heads,
                ffn_dim=architecture_config.ffn_dim,
                dropout=architecture_config.dropout,
            )
            for _ in range(architecture_config.num_refinement_blocks)
        )
        self.output_norm = nn.LayerNorm(latent_dim)
        self.delta_scorer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(architecture_config.dropout),
            nn.Linear(latent_dim, 1),
        )
        self.importance_scorer = nn.Linear(latent_dim, 1)

        if architecture_config.use_base_sts_expert:
            self.residual_router = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.residual_router = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)
        nn.init.zeros_(self.importance_scorer.weight)
        nn.init.zeros_(self.importance_scorer.bias)
        final_delta_layer = self.delta_scorer[-1]
        if isinstance(final_delta_layer, nn.Linear):
            # A tiny non-zero projection keeps the initial prediction extremely
            # close to the STS prior while allowing gradients to reach every
            # refinement block on the first optimizer step.
            nn.init.normal_(final_delta_layer.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(final_delta_layer.bias)
        if self.residual_router is not None:
            router = self.residual_router[-1]
            if isinstance(router, nn.Linear):
                nn.init.zeros_(router.weight)
                probability = self.architecture_config.initial_latent_contribution
                nn.init.constant_(router.bias, math.log(probability / (1.0 - probability)))

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        num_latent_heads: int = 6,
        latent_dim: int = 256,
        num_refinement_blocks: int = 2,
        cross_attention_heads: int = 8,
        self_attention_heads: int = 4,
        ffn_dim: int = 768,
        dropout: float = 0.1,
        latent_dropout: float = 0.1,
        use_base_sts_expert: bool = True,
        initial_latent_contribution: float = 0.1,
        **pretrained_kwargs: Any,
    ) -> "ModernCEDirectionalLatentMatcher":
        config = DirectionalLatentMatcherConfig(
            backbone_model_id=model_id,
            num_latent_heads=num_latent_heads,
            latent_dim=latent_dim,
            num_refinement_blocks=num_refinement_blocks,
            cross_attention_heads=cross_attention_heads,
            self_attention_heads=self_attention_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            latent_dropout=latent_dropout,
            use_base_sts_expert=use_base_sts_expert,
            initial_latent_contribution=initial_latent_contribution,
        )
        source_model = AutoModelForSequenceClassification.from_pretrained(
            model_id, **pretrained_kwargs
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

    def _apply_latent_dropout(self, importance_logits: Tensor) -> Tensor:
        probability = self.architecture_config.latent_dropout
        if not self.training or probability <= 0.0:
            return importance_logits
        keep = torch.rand_like(importance_logits) >= probability
        fallback = torch.randint(
            low=0,
            high=importance_logits.shape[-1],
            size=(importance_logits.shape[0],),
            device=importance_logits.device,
        )
        keep[
            torch.arange(importance_logits.shape[0], device=importance_logits.device),
            fallback,
        ] = True
        return importance_logits.masked_fill(
            ~keep, torch.finfo(importance_logits.dtype).min
        )

    def _score_latent_deltas(self, latents: Tensor) -> Tensor:
        """Map refined latent states to per-latent logit corrections."""

        return self.delta_scorer(latents).squeeze(-1)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
        candidate_mask: Optional[Tensor] = None,
        **encoder_kwargs: Any,
    ) -> DirectionalLatentMatcherOutput:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if target_mask is None or candidate_mask is None:
            raise ValueError(
                "Directional matcher requires target_mask and candidate_mask from "
                "paired tokenization"
            )
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [batch, sequence]")
        if target_mask.shape != input_ids.shape or candidate_mask.shape != input_ids.shape:
            raise ValueError("Directional masks must match input_ids shape")
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
        target_memory = self.target_projection(hidden_states)
        candidate_memory = self.candidate_projection(hidden_states)
        latents = self.latent_queries.unsqueeze(0).expand(input_ids.shape[0], -1, -1)

        target_attentions: list[Tensor] = []
        candidate_attentions: list[Tensor] = []
        for block in self.refinement_blocks:
            latents, target_attention, candidate_attention = block(
                latents,
                target_memory,
                candidate_memory,
                target_mask=target_mask,
                candidate_mask=candidate_mask,
            )
            target_attentions.append(target_attention)
            candidate_attentions.append(candidate_attention)

        latents = self.output_norm(latents)
        delta_logits = self._score_latent_deltas(latents)
        importance_logits = self.importance_scorer(latents).squeeze(-1)
        importance_logits = self._apply_latent_dropout(importance_logits)
        routing_weights = torch.softmax(importance_logits, dim=-1)

        if self.architecture_config.use_base_sts_expert:
            if self.base_sts_expert is None or self.residual_router is None:
                raise RuntimeError("The configured base STS expert is missing")
            base_logit = self.base_sts_expert(cls_state).reshape(-1, 1)
            latent_expert_logits = base_logit + delta_logits
            latent_contribution = torch.sigmoid(self.residual_router(cls_state))
            gate_weights = torch.cat(
                (
                    1.0 - latent_contribution,
                    latent_contribution * routing_weights,
                ),
                dim=-1,
            )
            expert_logits = torch.cat((base_logit, latent_expert_logits), dim=-1)
        else:
            latent_contribution = torch.ones(
                (input_ids.shape[0], 1),
                dtype=delta_logits.dtype,
                device=delta_logits.device,
            )
            gate_weights = routing_weights
            expert_logits = delta_logits

        logits = torch.sum(gate_weights * expert_logits, dim=-1)
        scores = torch.sigmoid(logits)
        target_attention_weights = torch.stack(target_attentions, dim=0).mean(dim=0)
        candidate_attention_weights = torch.stack(candidate_attentions, dim=0).mean(dim=0)
        attention_weights = 0.5 * (
            target_attention_weights + candidate_attention_weights
        )
        return DirectionalLatentMatcherOutput(
            scores=scores,
            logits=logits,
            head_scores=torch.sigmoid(expert_logits),
            head_logits=expert_logits,
            gate_weights=gate_weights,
            attention_weights=attention_weights,
            target_attention_weights=target_attention_weights,
            candidate_attention_weights=candidate_attention_weights,
            routing_weights=routing_weights,
            latent_contribution=latent_contribution.squeeze(-1),
        )

    def architecture_dict(self) -> dict[str, Any]:
        return {"architecture_type": self.architecture_type, **asdict(self.architecture_config)}

    @property
    def expert_names(self) -> tuple[str, ...]:
        latent_names = tuple(
            f"latent_{index}"
            for index in range(self.architecture_config.num_latent_heads)
        )
        if self.architecture_config.use_base_sts_expert:
            return ("base_sts", *latent_names)
        return latent_names

    def save_checkpoint(self, checkpoint_path: str | Path) -> Path:
        path = Path(checkpoint_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture_type": self.architecture_type,
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "architecture_config": asdict(self.architecture_config),
                "state_dict": self.state_dict(),
            },
            path,
        )
        return path

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        **pretrained_kwargs: Any,
    ) -> "ModernCEDirectionalLatentMatcher":
        payload = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location=map_location,
            weights_only=True,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid directional matcher checkpoint")
        architecture_type = payload.get("architecture_type")
        if architecture_type != cls.architecture_type:
            raise RuntimeError(
                f"Expected {cls.architecture_type!r}, found {architecture_type!r}"
            )
        version = int(payload.get("format_version", 0))
        if version != CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError(f"Unsupported directional checkpoint version {version}")
        raw_config = payload.get("architecture_config")
        state_dict = payload.get("state_dict")
        if not isinstance(raw_config, dict) or not isinstance(state_dict, dict):
            raise RuntimeError("Directional checkpoint config or state_dict is missing")
        config = DirectionalLatentMatcherConfig(**raw_config)
        constructor_args = asdict(config)
        model_id = str(constructor_args.pop("backbone_model_id"))
        model = cls.from_pretrained(
            model_id,
            **constructor_args,
            **pretrained_kwargs,
        )
        model.load_state_dict(state_dict, strict=True)
        return model
