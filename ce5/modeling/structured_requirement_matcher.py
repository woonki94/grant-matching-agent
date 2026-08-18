"""Structured requirement-slot matcher for grant-faculty capability coverage.

The cross encoder runs once over the raw grant/faculty pair. Learned slots then
locate grant requirements, retrieve supporting faculty evidence, and predict
per-requirement coverage. A permutation-invariant aggregator produces the
overall score. Decomposed claims are training supervision only; inference needs
the two original texts and their tokenizer-derived sequence masks.
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
ARCHITECTURE_TYPE = "structured_requirement_matcher"
CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class StructuredRequirementMatcherConfig:
    backbone_model_id: str = DEFAULT_MODEL_ID
    num_requirement_slots: int = 6
    latent_dim: int = 256
    num_refinement_blocks: int = 2
    cross_attention_heads: int = 8
    self_attention_heads: int = 4
    ffn_dim: int = 768
    dropout: float = 0.1
    use_base_sts_expert: bool = True
    initial_structured_contribution: float = 0.1

    def __post_init__(self) -> None:
        if self.num_requirement_slots < 1:
            raise ValueError("num_requirement_slots must be at least 1")
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
            raise ValueError("dropout must be in [0,1)")
        if not 0.0 < self.initial_structured_contribution <= 1.0:
            raise ValueError("initial_structured_contribution must be in (0,1]")


@dataclass
class StructuredRequirementMatcherOutput:
    scores: Tensor
    logits: Tensor
    slot_coverage_scores: Tensor
    slot_coverage_logits: Tensor
    slot_active_probabilities: Tensor
    slot_active_logits: Tensor
    slot_confidence_scores: Tensor
    slot_confidence_logits: Tensor
    overall_confidence_score: Tensor
    overall_confidence_logit: Tensor
    slot_importance_weights: Tensor
    target_attention_weights: Tensor
    candidate_attention_weights: Tensor
    structured_delta_logits: Tensor
    structured_contribution: Tensor
    base_sts_scores: Optional[Tensor]
    base_sts_logits: Optional[Tensor]


class StructuredRefinementBlock(nn.Module):
    """Grant localization, slot interaction, and faculty evidence retrieval."""

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
        slots: Tensor,
        target_memory: Tensor,
        candidate_memory: Tensor,
        *,
        target_mask: Tensor,
        candidate_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        target_values = self.target_memory_norm(target_memory)
        target_update, target_attention = self.target_attention(
            self.target_query_norm(slots),
            target_values,
            target_values,
            key_padding_mask=~target_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        slots = slots + self.residual_dropout(target_update)

        normalized_slots = self.self_query_norm(slots)
        interaction_update, _ = self.self_attention(
            normalized_slots,
            normalized_slots,
            normalized_slots,
            need_weights=False,
        )
        slots = slots + self.residual_dropout(interaction_update)

        candidate_values = self.candidate_memory_norm(candidate_memory)
        candidate_update, candidate_attention = self.candidate_attention(
            self.candidate_query_norm(slots),
            candidate_values,
            candidate_values,
            key_padding_mask=~candidate_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        slots = slots + self.residual_dropout(candidate_update)
        slots = slots + self.residual_dropout(self.ffn(self.ffn_norm(slots)))
        return (
            slots,
            target_attention.mean(dim=1),
            candidate_attention.mean(dim=1),
        )


class ModernCEStructuredRequirementMatcher(nn.Module):
    architecture_type = ARCHITECTURE_TYPE
    requires_pair_masks = True

    def __init__(
        self,
        *,
        encoder: nn.Module,
        architecture_config: StructuredRequirementMatcherConfig,
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
        self.cls_projection = nn.Linear(hidden_size, latent_dim)
        self.requirement_queries = nn.Parameter(
            torch.empty(architecture_config.num_requirement_slots, latent_dim)
        )
        self.refinement_blocks = nn.ModuleList(
            StructuredRefinementBlock(
                latent_dim=latent_dim,
                cross_attention_heads=architecture_config.cross_attention_heads,
                self_attention_heads=architecture_config.self_attention_heads,
                ffn_dim=architecture_config.ffn_dim,
                dropout=architecture_config.dropout,
            )
            for _ in range(architecture_config.num_refinement_blocks)
        )
        self.slot_norm = nn.LayerNorm(latent_dim)
        self.slot_coverage_scorer = nn.Linear(latent_dim, 1)
        self.slot_active_scorer = nn.Linear(latent_dim, 1)
        self.slot_confidence_scorer = nn.Linear(latent_dim, 1)
        self.slot_importance_scorer = nn.Linear(latent_dim, 1)

        aggregate_dim = latent_dim * 2 + 2
        self.structured_delta_scorer = nn.Sequential(
            nn.LayerNorm(aggregate_dim),
            nn.Linear(aggregate_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(architecture_config.dropout),
            nn.Linear(latent_dim, 1),
        )
        self.overall_confidence_scorer = nn.Sequential(
            nn.LayerNorm(aggregate_dim),
            nn.Linear(aggregate_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(architecture_config.dropout),
            nn.Linear(latent_dim // 2, 1),
        )
        self.structured_router = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.requirement_queries, mean=0.0, std=0.02)
        for scorer in (
            self.slot_coverage_scorer,
            self.slot_active_scorer,
            self.slot_confidence_scorer,
            self.slot_importance_scorer,
        ):
            nn.init.normal_(scorer.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(scorer.bias)
        final_delta = self.structured_delta_scorer[-1]
        if isinstance(final_delta, nn.Linear):
            nn.init.zeros_(final_delta.weight)
            nn.init.zeros_(final_delta.bias)
        final_confidence = self.overall_confidence_scorer[-1]
        if isinstance(final_confidence, nn.Linear):
            nn.init.normal_(final_confidence.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(final_confidence.bias)
        router = self.structured_router[-1]
        if isinstance(router, nn.Linear):
            nn.init.zeros_(router.weight)
            probability = self.architecture_config.initial_structured_contribution
            if probability >= 1.0:
                nn.init.constant_(router.bias, 20.0)
            else:
                nn.init.constant_(
                    router.bias,
                    math.log(probability / (1.0 - probability)),
                )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        num_requirement_slots: int = 6,
        latent_dim: int = 256,
        num_refinement_blocks: int = 2,
        cross_attention_heads: int = 8,
        self_attention_heads: int = 4,
        ffn_dim: int = 768,
        dropout: float = 0.1,
        use_base_sts_expert: bool = True,
        initial_structured_contribution: float = 0.1,
        **pretrained_kwargs: Any,
    ) -> "ModernCEStructuredRequirementMatcher":
        config = StructuredRequirementMatcherConfig(
            backbone_model_id=model_id,
            num_requirement_slots=num_requirement_slots,
            latent_dim=latent_dim,
            num_refinement_blocks=num_refinement_blocks,
            cross_attention_heads=cross_attention_heads,
            self_attention_heads=self_attention_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_base_sts_expert=use_base_sts_expert,
            initial_structured_contribution=initial_structured_contribution,
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

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
        candidate_mask: Optional[Tensor] = None,
        **encoder_kwargs: Any,
    ) -> StructuredRequirementMatcherOutput:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if target_mask is None or candidate_mask is None:
            raise ValueError("target_mask and candidate_mask are required")
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [batch, sequence]")
        if target_mask.shape != input_ids.shape or candidate_mask.shape != input_ids.shape:
            raise ValueError("Pair masks must match input_ids")
        target_mask = target_mask.bool() & attention_mask.bool()
        candidate_mask = candidate_mask.bool() & attention_mask.bool()
        if torch.any(target_mask.sum(dim=-1) == 0):
            raise ValueError("Every example must retain at least one grant token")
        if torch.any(candidate_mask.sum(dim=-1) == 0):
            raise ValueError("Every example must retain at least one faculty token")

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
        slots = self.requirement_queries.unsqueeze(0).expand(
            input_ids.shape[0],
            -1,
            -1,
        )

        target_attentions: list[Tensor] = []
        candidate_attentions: list[Tensor] = []
        for block in self.refinement_blocks:
            slots, target_attention, candidate_attention = block(
                slots,
                target_memory,
                candidate_memory,
                target_mask=target_mask,
                candidate_mask=candidate_mask,
            )
            target_attentions.append(target_attention)
            candidate_attentions.append(candidate_attention)
        target_attention_weights = torch.stack(target_attentions, dim=0).mean(dim=0)
        candidate_attention_weights = torch.stack(
            candidate_attentions,
            dim=0,
        ).mean(dim=0)

        slots = self.slot_norm(slots)
        slot_coverage_logits = self.slot_coverage_scorer(slots).squeeze(-1)
        slot_active_logits = self.slot_active_scorer(slots).squeeze(-1)
        slot_confidence_logits = self.slot_confidence_scorer(slots).squeeze(-1)
        slot_coverage_scores = torch.sigmoid(slot_coverage_logits)
        slot_active_probabilities = torch.sigmoid(slot_active_logits)
        slot_confidence_scores = torch.sigmoid(slot_confidence_logits)

        importance_logits = self.slot_importance_scorer(slots).squeeze(-1)
        routing_logits = importance_logits + torch.log(
            slot_active_probabilities.clamp_min(1e-6)
        )
        slot_importance_weights = torch.softmax(routing_logits, dim=-1)
        pooled_slots = torch.einsum("bs,bsd->bd", slot_importance_weights, slots)
        weighted_coverage_logit = torch.sum(
            slot_importance_weights * slot_coverage_logits,
            dim=-1,
            keepdim=True,
        )
        active_fraction = slot_active_probabilities.mean(dim=-1, keepdim=True)
        aggregate = torch.cat(
            (
                pooled_slots,
                self.cls_projection(cls_state),
                weighted_coverage_logit,
                active_fraction,
            ),
            dim=-1,
        )
        structured_delta_logits = self.structured_delta_scorer(aggregate).squeeze(-1)
        overall_confidence_logit = self.overall_confidence_scorer(aggregate).squeeze(-1)
        overall_confidence_score = torch.sigmoid(overall_confidence_logit)

        structured_contribution = torch.sigmoid(
            self.structured_router(cls_state).squeeze(-1)
        )
        base_sts_logits: Optional[Tensor] = None
        base_sts_scores: Optional[Tensor] = None
        if self.architecture_config.use_base_sts_expert:
            if self.base_sts_expert is None:
                raise RuntimeError("The configured base STS expert is missing")
            base_sts_logits = self.base_sts_expert(cls_state).reshape(-1)
            base_sts_scores = torch.sigmoid(base_sts_logits)
            logits = base_sts_logits + structured_contribution * structured_delta_logits
        else:
            logits = structured_delta_logits
            structured_contribution = torch.ones_like(logits)
        scores = torch.sigmoid(logits)
        return StructuredRequirementMatcherOutput(
            scores=scores,
            logits=logits,
            slot_coverage_scores=slot_coverage_scores,
            slot_coverage_logits=slot_coverage_logits,
            slot_active_probabilities=slot_active_probabilities,
            slot_active_logits=slot_active_logits,
            slot_confidence_scores=slot_confidence_scores,
            slot_confidence_logits=slot_confidence_logits,
            overall_confidence_score=overall_confidence_score,
            overall_confidence_logit=overall_confidence_logit,
            slot_importance_weights=slot_importance_weights,
            target_attention_weights=target_attention_weights,
            candidate_attention_weights=candidate_attention_weights,
            structured_delta_logits=structured_delta_logits,
            structured_contribution=structured_contribution,
            base_sts_scores=base_sts_scores,
            base_sts_logits=base_sts_logits,
        )

    def architecture_dict(self) -> dict[str, Any]:
        return asdict(self.architecture_config)

    def save_checkpoint(self, checkpoint_path: str | Path) -> Path:
        path = Path(checkpoint_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture_type": ARCHITECTURE_TYPE,
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "architecture_config": self.architecture_dict(),
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
    ) -> "ModernCEStructuredRequirementMatcher":
        payload = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location=map_location,
            weights_only=True,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid CE5 checkpoint: expected a dictionary")
        if int(payload.get("format_version", 0)) != CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError(
                "Unsupported structured requirement matcher checkpoint version"
            )
        raw_config = payload.get("architecture_config")
        state_dict = payload.get("state_dict")
        if not isinstance(raw_config, dict) or not isinstance(state_dict, dict):
            raise RuntimeError("Invalid CE5 checkpoint: config or state_dict is missing")
        config = StructuredRequirementMatcherConfig(**raw_config)
        constructor_args = asdict(config)
        model_id = str(constructor_args.pop("backbone_model_id"))
        model = cls.from_pretrained(
            model_id,
            **constructor_args,
            **pretrained_kwargs,
        )
        model.load_state_dict(state_dict, strict=True)
        return model
