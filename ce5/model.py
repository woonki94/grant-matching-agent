"""Multi-latent-head ModernCE model for grant-to-faculty capability scoring.

The model intentionally performs one transformer pass.  Several lightweight
latent heads then attend to different parts of the shared token representation,
and a learned gate combines their scores into one directional 0-1 score.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModelForSequenceClassification


DEFAULT_MODEL_ID = "dleemiller/ModernCE-base-sts"
CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class LatentHeadConfig:
    """Architecture and regularization settings saved with a CE5 checkpoint."""

    backbone_model_id: str = DEFAULT_MODEL_ID
    num_latent_heads: int = 6
    attention_dim: int = 128
    head_dim: int = 192
    dropout: float = 0.1
    head_dropout: float = 0.0
    use_base_sts_expert: bool = True
    base_sts_gate_bias: float = 4.0
    score_loss: Literal["smooth_l1", "mse", "bce"] = "smooth_l1"
    diversity_loss_weight: float = 0.01
    gate_balance_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.num_latent_heads < 1:
            raise ValueError("num_latent_heads must be at least 1")
        if self.attention_dim < 1 or self.head_dim < 1:
            raise ValueError("attention_dim and head_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.head_dropout < 1.0:
            raise ValueError("head_dropout must be in [0, 1)")
        if self.score_loss not in {"smooth_l1", "mse", "bce"}:
            raise ValueError(f"Unsupported score_loss: {self.score_loss}")
        if self.diversity_loss_weight < 0.0:
            raise ValueError("diversity_loss_weight cannot be negative")
        if self.gate_balance_loss_weight < 0.0:
            raise ValueError("gate_balance_loss_weight cannot be negative")


@dataclass
class LatentHeadOutput:
    """Model outputs, including diagnostics useful during CE5 experiments."""

    scores: Tensor
    logits: Tensor
    head_scores: Tensor
    head_logits: Tensor
    gate_weights: Tensor
    attention_weights: Tensor
    loss: Optional[Tensor] = None
    score_loss: Optional[Tensor] = None
    diversity_loss: Optional[Tensor] = None
    gate_balance_loss: Optional[Tensor] = None


class LatentAttentionHead(nn.Module):
    """A learned attention pooler and scalar scorer for one latent view."""

    def __init__(
        self,
        *,
        hidden_size: int,
        attention_dim: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.key_projection = nn.Linear(hidden_size, attention_dim, bias=False)
        self.query = nn.Parameter(torch.empty(attention_dim))
        self.value_projection = nn.Linear(hidden_size, head_dim, bias=False)
        self.value_norm = nn.LayerNorm(head_dim)
        self.scorer = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        # Starting close to zero lets the pretrained STS expert provide a stable
        # initial score while the latent heads begin to specialize.
        output_layer = self.scorer[-1]
        if isinstance(output_layer, nn.Linear):
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        keys = torch.tanh(self.key_projection(hidden_states))
        attention_logits = torch.einsum("bld,d->bl", keys, self.query)
        attention_logits = attention_logits / math.sqrt(float(self.query.numel()))
        attention_logits = attention_logits.masked_fill(
            ~attention_mask.bool(), torch.finfo(attention_logits.dtype).min
        )
        attention_weights = torch.softmax(attention_logits, dim=-1)

        values = self.value_projection(hidden_states)
        pooled = torch.einsum("bl,bld->bd", attention_weights, values)
        pooled = self.value_norm(pooled)
        logit = self.scorer(pooled).squeeze(-1)
        return logit, attention_weights


class ModernCELatentHeadModel(nn.Module):
    """One ModernCE encoder with learned latent experts and dynamic gating.

    The first expert can preserve ModernCE's original STS scoring head.  The
    remaining experts independently pool token representations.  The original
    expert is strongly favored at initialization so a newly created model starts
    near the pretrained checkpoint rather than near a random ensemble.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        architecture_config: LatentHeadConfig,
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
            LatentAttentionHead(
                hidden_size=hidden_size,
                attention_dim=architecture_config.attention_dim,
                head_dim=architecture_config.head_dim,
                dropout=architecture_config.dropout,
            )
            for _ in range(architecture_config.num_latent_heads)
        )

        self.num_experts = architecture_config.num_latent_heads + int(
            architecture_config.use_base_sts_expert
        )
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
        attention_dim: int = 128,
        head_dim: int = 192,
        dropout: float = 0.1,
        head_dropout: float = 0.0,
        use_base_sts_expert: bool = True,
        base_sts_gate_bias: float = 4.0,
        score_loss: Literal["smooth_l1", "mse", "bce"] = "smooth_l1",
        diversity_loss_weight: float = 0.01,
        gate_balance_loss_weight: float = 0.0,
        **pretrained_kwargs: Any,
    ) -> "ModernCELatentHeadModel":
        """Load ModernCE and attach new latent heads.

        ``pretrained_kwargs`` are passed to Hugging Face, so callers can use
        options such as ``local_files_only=True``, ``torch_dtype=...``, or a
        cache directory without changing this class.
        """

        config = LatentHeadConfig(
            backbone_model_id=model_id,
            num_latent_heads=num_latent_heads,
            attention_dim=attention_dim,
            head_dim=head_dim,
            dropout=dropout,
            head_dropout=head_dropout,
            use_base_sts_expert=use_base_sts_expert,
            base_sts_gate_bias=base_sts_gate_bias,
            score_loss=score_loss,
            diversity_loss_weight=diversity_loss_weight,
            gate_balance_loss_weight=gate_balance_loss_weight,
        )
        source_model = AutoModelForSequenceClassification.from_pretrained(
            model_id, **pretrained_kwargs
        )
        encoder = source_model.base_model
        base_expert: Optional[nn.Module] = None
        if use_base_sts_expert:
            required = ("head", "drop", "classifier")
            missing = [name for name in required if not isinstance(getattr(source_model, name, None), nn.Module)]
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
        *,
        labels: Optional[Tensor] = None,
        **encoder_kwargs: Any,
    ) -> LatentHeadOutput:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.ndim != 2 or input_ids.ndim != 2:
            raise ValueError("input_ids and attention_mask must both have shape [batch, sequence]")
        if torch.any(attention_mask.sum(dim=-1) == 0):
            raise ValueError("Every example must contain at least one unmasked token")

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **encoder_kwargs,
        )
        hidden_states = encoder_outputs.last_hidden_state
        cls_state = hidden_states[:, 0]

        latent_results = [
            head(hidden_states, attention_mask) for head in self.latent_heads
        ]
        latent_logits = torch.stack([result[0] for result in latent_results], dim=-1)
        attention_weights = torch.stack(
            [result[1] for result in latent_results], dim=1
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
        scores = torch.sigmoid(logits)

        diversity_loss = self._attention_diversity_loss(attention_weights)
        gate_balance_loss = self._gate_balance_loss(gate_weights)
        score_loss: Optional[Tensor] = None
        total_loss: Optional[Tensor] = None
        if labels is not None:
            targets = labels.to(device=scores.device, dtype=scores.dtype).reshape(-1)
            if targets.shape != scores.shape:
                raise ValueError(
                    f"labels must contain {scores.numel()} values; got {targets.numel()}"
                )
            if torch.any((targets < 0.0) | (targets > 1.0)):
                raise ValueError("Continuous teacher labels must be in [0, 1]")
            score_loss = self._score_loss(logits, scores, targets)
            total_loss = (
                score_loss
                + self.architecture_config.diversity_loss_weight * diversity_loss
                + self.architecture_config.gate_balance_loss_weight * gate_balance_loss
            )

        return LatentHeadOutput(
            scores=scores,
            logits=logits,
            head_scores=torch.sigmoid(expert_logits),
            head_logits=expert_logits,
            gate_weights=gate_weights,
            attention_weights=attention_weights,
            loss=total_loss,
            score_loss=score_loss,
            diversity_loss=diversity_loss,
            gate_balance_loss=gate_balance_loss,
        )

    def _apply_head_dropout(self, gate_logits: Tensor) -> Tensor:
        probability = self.architecture_config.head_dropout
        if not self.training or probability <= 0.0:
            return gate_logits
        keep = torch.rand_like(gate_logits) >= probability
        # Always retain one randomly selected expert per example.
        fallback = torch.randint(
            low=0,
            high=self.num_experts,
            size=(gate_logits.shape[0],),
            device=gate_logits.device,
        )
        keep[torch.arange(gate_logits.shape[0], device=gate_logits.device), fallback] = True
        return gate_logits.masked_fill(~keep, torch.finfo(gate_logits.dtype).min)

    def _score_loss(self, logits: Tensor, scores: Tensor, targets: Tensor) -> Tensor:
        loss_name = self.architecture_config.score_loss
        if loss_name == "smooth_l1":
            return F.smooth_l1_loss(scores, targets)
        if loss_name == "mse":
            return F.mse_loss(scores, targets)
        if loss_name == "bce":
            return F.binary_cross_entropy_with_logits(logits, targets)
        raise RuntimeError(f"Unsupported score loss: {loss_name}")

    @staticmethod
    def _attention_diversity_loss(attention_weights: Tensor) -> Tensor:
        num_heads = attention_weights.shape[1]
        if num_heads < 2:
            return attention_weights.new_zeros(())
        normalized = F.normalize(attention_weights, p=2, dim=-1, eps=1e-8)
        similarities = torch.bmm(normalized, normalized.transpose(1, 2))
        off_diagonal = ~torch.eye(
            num_heads, dtype=torch.bool, device=attention_weights.device
        ).unsqueeze(0)
        return similarities.masked_select(off_diagonal).square().mean()

    @staticmethod
    def _gate_balance_loss(gate_weights: Tensor) -> Tensor:
        average_usage = gate_weights.mean(dim=0).clamp_min(1e-8)
        num_experts = gate_weights.shape[-1]
        return torch.sum(average_usage * torch.log(average_usage * num_experts))

    def architecture_dict(self) -> dict[str, Any]:
        return asdict(self.architecture_config)

    @property
    def expert_names(self) -> tuple[str, ...]:
        latent_names = tuple(
            f"latent_{index}" for index in range(self.architecture_config.num_latent_heads)
        )
        if self.architecture_config.use_base_sts_expert:
            return ("base_sts", *latent_names)
        return latent_names

    def save_checkpoint(self, checkpoint_path: str | Path) -> Path:
        """Save the complete encoder and latent-head state as one checkpoint."""

        path = Path(checkpoint_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
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
    ) -> "ModernCELatentHeadModel":
        """Restore a CE5 checkpoint using its recorded backbone architecture."""

        payload = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location=map_location,
            weights_only=True,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid CE5 checkpoint: expected a dictionary")
        version = int(payload.get("format_version", 0))
        if version != CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError(
                f"Unsupported CE5 checkpoint version {version}; "
                f"expected {CHECKPOINT_FORMAT_VERSION}"
            )
        raw_config = payload.get("architecture_config")
        state_dict = payload.get("state_dict")
        if not isinstance(raw_config, dict) or not isinstance(state_dict, dict):
            raise RuntimeError("Invalid CE5 checkpoint: config or state_dict is missing")

        config = LatentHeadConfig(**raw_config)
        constructor_args = asdict(config)
        model_id = str(constructor_args.pop("backbone_model_id"))
        model = cls.from_pretrained(
            model_id,
            **constructor_args,
            **pretrained_kwargs,
        )
        model.load_state_dict(state_dict, strict=True)
        return model

    def freeze_encoder(self) -> None:
        """Freeze the shared transformer while warming up the new heads."""

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def parameter_counts(self) -> dict[str, int]:
        parameters: Sequence[nn.Parameter] = tuple(self.parameters())
        return {
            "total": sum(parameter.numel() for parameter in parameters),
            "trainable": sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            ),
        }


def pairwise_ranking_loss(
    positive_logits: Tensor,
    negative_logits: Tensor,
    *,
    margin: float = 0.0,
    sample_weights: Optional[Tensor] = None,
) -> Tensor:
    """Logistic ranking loss for grant-matched positive/negative capabilities."""

    positive = positive_logits.reshape(-1)
    negative = negative_logits.reshape(-1)
    if positive.shape != negative.shape:
        raise ValueError("positive_logits and negative_logits must have equal sizes")
    losses = F.softplus(-(positive - negative - margin))
    if sample_weights is None:
        return losses.mean()
    weights = sample_weights.to(device=losses.device, dtype=losses.dtype).reshape(-1)
    if weights.shape != losses.shape:
        raise ValueError("sample_weights must match the number of positive/negative pairs")
    return torch.sum(losses * weights) / weights.sum().clamp_min(1e-8)
