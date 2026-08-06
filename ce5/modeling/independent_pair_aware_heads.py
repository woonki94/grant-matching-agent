"""Independent-v2 pair-aware latent experts for CE5.

The ModernCE backbone still runs exactly once.  Unlike the original latent
heads, every v2 expert independently pools the target and candidate sequences,
constructs explicit pair interactions, and scores that private representation.
The original dense soft router and absolute-logit mixture are intentionally
preserved so this architecture isolates the effect of a more expressive expert.
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
ARCHITECTURE_TYPE = "independent_pair_aware_heads"
CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class IndependentPairAwareConfig:
    """Inference architecture stored with an independent-v2 checkpoint."""

    backbone_model_id: str = DEFAULT_MODEL_ID
    num_latent_heads: int = 6
    num_queries_per_side: int = 2
    attention_dim: int = 128
    head_dim: int = 192
    expert_ffn_dim: int = 384
    dropout: float = 0.1
    head_dropout: float = 0.0
    use_base_sts_expert: bool = True
    base_sts_gate_bias: float = 4.0

    def __post_init__(self) -> None:
        if self.num_latent_heads < 1:
            raise ValueError("num_latent_heads must be at least 1")
        if self.num_queries_per_side < 1:
            raise ValueError("num_queries_per_side must be at least 1")
        if self.attention_dim < 1 or self.head_dim < 1 or self.expert_ffn_dim < 1:
            raise ValueError("attention_dim, head_dim, and expert_ffn_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.head_dropout < 1.0:
            raise ValueError("head_dropout must be in [0, 1)")


@dataclass
class IndependentPairAwareOutput:
    """Predictions and diagnostics emitted by independent-v2."""

    scores: Tensor
    logits: Tensor
    head_scores: Tensor
    head_logits: Tensor
    gate_weights: Tensor
    attention_weights: Tensor
    target_attention_weights: Tensor
    candidate_attention_weights: Tensor


class PrivateSidePooler(nn.Module):
    """One expert's private multi-query attention pooler for one pair side."""

    def __init__(
        self,
        *,
        hidden_size: int,
        attention_dim: int,
        head_dim: int,
        num_queries: int,
    ) -> None:
        super().__init__()
        self.key_projection = nn.Linear(hidden_size, attention_dim, bias=False)
        self.value_projection = nn.Linear(hidden_size, head_dim, bias=False)
        self.queries = nn.Parameter(torch.empty(num_queries, attention_dim))
        self.value_norm = nn.LayerNorm(head_dim)
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

    def forward(self, hidden_states: Tensor, side_mask: Tensor) -> tuple[Tensor, Tensor]:
        keys = torch.tanh(self.key_projection(hidden_states))
        attention_logits = torch.einsum("bld,qd->bql", keys, self.queries)
        attention_logits = attention_logits / math.sqrt(float(self.queries.shape[-1]))
        attention_logits = attention_logits.masked_fill(
            ~side_mask[:, None, :],
            torch.finfo(attention_logits.dtype).min,
        )
        attention_weights = torch.softmax(attention_logits, dim=-1)
        values = self.value_projection(hidden_states)
        pooled = torch.einsum("bql,bld->bqd", attention_weights, values)
        return self.value_norm(pooled), attention_weights


class IndependentPairAwareExpert(nn.Module):
    """A fully private target/candidate pooler, interaction block, and scorer."""

    def __init__(
        self,
        *,
        hidden_size: int,
        attention_dim: int,
        head_dim: int,
        num_queries_per_side: int,
        expert_ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.target_pooler = PrivateSidePooler(
            hidden_size=hidden_size,
            attention_dim=attention_dim,
            head_dim=head_dim,
            num_queries=num_queries_per_side,
        )
        self.candidate_pooler = PrivateSidePooler(
            hidden_size=hidden_size,
            attention_dim=attention_dim,
            head_dim=head_dim,
            num_queries=num_queries_per_side,
        )
        interaction_dim = 4 * num_queries_per_side * head_dim
        self.interaction_norm = nn.LayerNorm(interaction_dim)
        self.scorer = nn.Sequential(
            nn.Linear(interaction_dim, expert_ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_ffn_dim, expert_ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_ffn_dim, 1),
        )
        final_layer = self.scorer[-1]
        if isinstance(final_layer, nn.Linear):
            # Match independent-v1 initialization exactly: new experts begin at
            # zero logit and the pretrained STS expert supplies the initial prior.
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        target_mask: Tensor,
        candidate_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        target, target_attention = self.target_pooler(hidden_states, target_mask)
        candidate, candidate_attention = self.candidate_pooler(
            hidden_states,
            candidate_mask,
        )
        interaction = torch.cat(
            (target, candidate, torch.abs(target - candidate), target * candidate),
            dim=-1,
        ).flatten(start_dim=1)
        logit = self.scorer(self.interaction_norm(interaction)).squeeze(-1)

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
        )


class ModernCEIndependentPairAwareModel(nn.Module):
    """One ModernCE pass followed by independent pair-aware expert paths."""

    architecture_type = ARCHITECTURE_TYPE
    requires_pair_masks = True

    def __init__(
        self,
        *,
        encoder: nn.Module,
        architecture_config: IndependentPairAwareConfig,
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

        self.latent_heads = nn.ModuleList(
            IndependentPairAwareExpert(
                hidden_size=hidden_size,
                attention_dim=architecture_config.attention_dim,
                head_dim=architecture_config.head_dim,
                num_queries_per_side=architecture_config.num_queries_per_side,
                expert_ffn_dim=architecture_config.expert_ffn_dim,
                dropout=architecture_config.dropout,
            )
            for _ in range(architecture_config.num_latent_heads)
        )
        self.num_experts = architecture_config.num_latent_heads + int(
            architecture_config.use_base_sts_expert
        )
        # This is intentionally the original v1 router for a controlled test.
        self.gate_norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Linear(hidden_size, self.num_experts)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        if architecture_config.use_base_sts_expert:
            with torch.no_grad():
                self.gate.bias[0] = architecture_config.base_sts_gate_bias

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
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
        **pretrained_kwargs: Any,
    ) -> "ModernCEIndependentPairAwareModel":
        config = IndependentPairAwareConfig(
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
    ) -> IndependentPairAwareOutput:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if target_mask is None or candidate_mask is None:
            raise ValueError(
                "Independent-v2 requires target_mask and candidate_mask from paired "
                "tokenization"
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
            expert(
                hidden_states,
                target_mask=target_mask,
                candidate_mask=candidate_mask,
            )
            for expert in self.latent_heads
        ]
        latent_logits = torch.stack([result[0] for result in expert_results], dim=-1)
        attention_weights = torch.stack(
            [result[1] for result in expert_results],
            dim=1,
        )
        target_attention_weights = torch.stack(
            [result[2] for result in expert_results],
            dim=1,
        )
        candidate_attention_weights = torch.stack(
            [result[3] for result in expert_results],
            dim=1,
        )

        expert_logits = latent_logits
        if self.architecture_config.use_base_sts_expert:
            if self.base_sts_expert is None:
                raise RuntimeError("The configured base STS expert is missing")
            base_logit = self.base_sts_expert(cls_state).reshape(-1, 1)
            expert_logits = torch.cat((base_logit, latent_logits), dim=-1)

        gate_logits = self.gate(self.gate_norm(cls_state))
        gate_logits = self._apply_head_dropout(gate_logits)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        logits = torch.sum(gate_weights * expert_logits, dim=-1)
        return IndependentPairAwareOutput(
            scores=torch.sigmoid(logits),
            logits=logits,
            head_scores=torch.sigmoid(expert_logits),
            head_logits=expert_logits,
            gate_weights=gate_weights,
            attention_weights=attention_weights,
            target_attention_weights=target_attention_weights,
            candidate_attention_weights=candidate_attention_weights,
        )

    def _apply_head_dropout(self, gate_logits: Tensor) -> Tensor:
        probability = self.architecture_config.head_dropout
        if not self.training or probability <= 0.0:
            return gate_logits
        keep = torch.rand_like(gate_logits) >= probability
        fallback = torch.randint(
            low=0,
            high=self.num_experts,
            size=(gate_logits.shape[0],),
            device=gate_logits.device,
        )
        keep[
            torch.arange(gate_logits.shape[0], device=gate_logits.device),
            fallback,
        ] = True
        return gate_logits.masked_fill(
            ~keep,
            torch.finfo(gate_logits.dtype).min,
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
    ) -> "ModernCEIndependentPairAwareModel":
        payload = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location=map_location,
            weights_only=True,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid independent-v2 checkpoint")
        architecture_type = payload.get("architecture_type")
        if architecture_type != cls.architecture_type:
            raise RuntimeError(
                f"Expected {cls.architecture_type!r}, found {architecture_type!r}"
            )
        version = int(payload.get("format_version", 0))
        if version != CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError(f"Unsupported independent-v2 checkpoint version {version}")
        raw_config = payload.get("architecture_config")
        state_dict = payload.get("state_dict")
        if not isinstance(raw_config, dict) or not isinstance(state_dict, dict):
            raise RuntimeError("Independent-v2 checkpoint config or state_dict is missing")
        config = IndependentPairAwareConfig(**raw_config)
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
    "IndependentPairAwareConfig",
    "IndependentPairAwareExpert",
    "IndependentPairAwareOutput",
    "ModernCEIndependentPairAwareModel",
    "PrivateSidePooler",
]
