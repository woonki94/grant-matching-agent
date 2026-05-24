from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification


ASPECT_HEADS_FILE = "aspect_heads.pt"
ASPECT_HEADS_CONFIG_FILE = "aspect_heads_config.json"

ASPECT_PREFIX_BY_NAME: Dict[str, str] = {
    "domain": "[DOMAIN]",
    "method": "[METHOD]",
    "constraint": "[CONSTRAINT]",
}

ASPECT_CONDITION_TEXTS: Dict[str, str] = {
    "domain": (
        "[DOMAIN] Compare whether the pair belongs to the same research domain, "
        "subject area, application area, or scientific field."
    ),
    "method": (
        "[METHOD] Compare whether the pair shares the same methods, techniques, "
        "procedures, or implementation approach."
    ),
    "constraint": (
        "[CONSTRAINT] Compare whether the document satisfies the concrete requirement, "
        "constraint, qualification, deliverable, or capability requested by the query."
    ),
}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def clean_aspect_condition_mode(value: Any) -> str:
    mode = _clean_text(value).lower()
    if mode in {"legacy", "long_prefix", "none"}:
        return mode
    return "legacy"


def aspect_from_prefixed_query(query_text: Any) -> str:
    q = _clean_text(query_text).lstrip().upper()
    for aspect, prefix in ASPECT_PREFIX_BY_NAME.items():
        if q.startswith(prefix):
            return aspect
    return "unknown"


def aspect_id_from_name(aspect: Any) -> int:
    a = _clean_text(aspect).lower()
    if a == "domain":
        return 1
    if a == "method":
        return 2
    if a == "constraint":
        return 3
    return 0


def aspect_name_from_id(aspect_id: Any) -> str:
    try:
        parsed = int(aspect_id)
    except Exception:
        parsed = 0
    if parsed == 1:
        return "domain"
    if parsed == 2:
        return "method"
    if parsed == 3:
        return "constraint"
    return "domain"


def strip_aspect_prefix(text: Any) -> str:
    raw = _clean_text(text).lstrip()
    upper = raw.upper()
    for prefix in ASPECT_PREFIX_BY_NAME.values():
        if upper.startswith(prefix):
            return raw[len(prefix) :].strip()
    return raw


def format_aspect_pair(query_text: Any, doc_text: Any, *, aspect_condition_mode: str) -> Tuple[str, str]:
    mode = clean_aspect_condition_mode(aspect_condition_mode)
    if mode == "legacy":
        return _clean_text(query_text), _clean_text(doc_text)

    aspect = aspect_from_prefixed_query(query_text)
    query = strip_aspect_prefix(query_text)
    doc = strip_aspect_prefix(doc_text)
    if mode == "none":
        return query, doc

    condition = ASPECT_CONDITION_TEXTS.get(aspect) or "[MATCH] Compare the semantic match between the query and document."
    return f"{condition}\nQuery: {query}", f"Document: {doc}"


def _infer_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    for attr in ("hidden_size", "d_model", "dim"):
        value = getattr(config, attr, None)
        if value:
            return int(value)
    for param in model.parameters():
        if param.dim() >= 2:
            return int(param.shape[-1])
    raise ValueError("Unable to infer hidden size for aspect-head wrapper.")


def _find_linear_classifier(model: nn.Module) -> Optional[nn.Linear]:
    for attr in ("classifier", "score", "regressor"):
        head = getattr(model, attr, None)
        if isinstance(head, nn.Linear) and int(head.out_features) == 1:
            return head
        if isinstance(head, nn.Sequential):
            linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
            if linears and int(linears[-1].out_features) == 1:
                return linears[-1]
    return None


class AspectHeadSequenceClassifier(nn.Module):
    """Shared transformer encoder with separate scalar heads per training aspect."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = getattr(backbone, "config", None)
        hidden_size = _infer_hidden_size(backbone)
        self.heads = nn.ModuleDict(
            {
                "domain": nn.Linear(hidden_size, 1),
                "method": nn.Linear(hidden_size, 1),
                "constraint": nn.Linear(hidden_size, 1),
            }
        )
        self._init_from_backbone_classifier()

    def _init_from_backbone_classifier(self) -> None:
        source = _find_linear_classifier(self.backbone)
        if source is None:
            return
        for head in self.heads.values():
            if head.weight.shape == source.weight.shape:
                head.weight.data.copy_(source.weight.data)
            if head.bias is not None and source.bias is not None and head.bias.shape == source.bias.shape:
                head.bias.data.copy_(source.bias.data)

    def _encoder_forward(self, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        kwargs.pop("labels", None)
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        return self.backbone(**kwargs)

    def forward(self, *args: Any, aspect_ids: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        if args:
            raise TypeError("AspectHeadSequenceClassifier expects keyword inputs.")
        outputs = self._encoder_forward(**kwargs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states:
                    last_hidden = hidden_states[-1]
            if last_hidden is None:
                raise RuntimeError("Backbone did not return hidden states for aspect heads.")
            pooled = last_hidden[:, 0, :]

        batch_size = int(pooled.shape[0])
        if aspect_ids is None:
            aspect_ids = torch.zeros(batch_size, device=pooled.device, dtype=torch.long)
        else:
            aspect_ids = aspect_ids.to(device=pooled.device, dtype=torch.long).view(-1)
            if int(aspect_ids.numel()) != batch_size:
                aspect_ids = torch.zeros(batch_size, device=pooled.device, dtype=torch.long)

        logits = pooled.new_empty((batch_size, 1))
        handled = torch.zeros(batch_size, device=pooled.device, dtype=torch.bool)
        for aspect_id, aspect_name in ((0, "domain"), (1, "domain"), (2, "method"), (3, "constraint")):
            mask = aspect_ids == int(aspect_id)
            if bool(mask.any().item()):
                logits[mask] = self.heads[aspect_name](pooled[mask])
                handled = handled | mask
        if bool((~handled).any().item()):
            logits[~handled] = self.heads["domain"](pooled[~handled])
        return SimpleNamespace(logits=logits)

    def save_pretrained(self, save_directory: Any, **kwargs: Any) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path, **kwargs)
        torch.save(self.heads.state_dict(), path / ASPECT_HEADS_FILE)
        (path / ASPECT_HEADS_CONFIG_FILE).write_text(
            json.dumps(
                {
                    "architecture": self.__class__.__name__,
                    "heads": list(self.heads.keys()),
                    "head_file": ASPECT_HEADS_FILE,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_aspect_heads(self, model_dir: Any, *, map_location: Any = "cpu") -> bool:
        path = Path(model_dir) / ASPECT_HEADS_FILE
        if not path.exists():
            return False
        state = torch.load(path, map_location=map_location)
        self.heads.load_state_dict(state, strict=True)
        return True


def load_sequence_classifier_model(
    model_ref: str,
    *,
    num_labels: int = 1,
    multi_aspect_heads: bool = False,
    trust_remote_code: bool = False,
) -> nn.Module:
    backbone = AutoModelForSequenceClassification.from_pretrained(
        model_ref,
        num_labels=num_labels,
        trust_remote_code=trust_remote_code,
    )
    model_path = Path(_clean_text(model_ref)).expanduser()
    has_saved_heads = model_path.is_dir() and (model_path / ASPECT_HEADS_FILE).exists()
    if not (multi_aspect_heads or has_saved_heads):
        return backbone
    model = AspectHeadSequenceClassifier(backbone)
    if has_saved_heads:
        model.load_aspect_heads(model_path, map_location="cpu")
    return model


def model_logits(model: nn.Module, enc: Dict[str, torch.Tensor], *, aspect_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    if aspect_ids is not None and isinstance(model, AspectHeadSequenceClassifier):
        return model(**enc, aspect_ids=aspect_ids).logits
    return model(**enc).logits
