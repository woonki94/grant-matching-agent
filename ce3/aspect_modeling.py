from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification


ASPECTS = ("topic", "approach", "objective")
ASPECT_HEADS_FILE = "aspect_heads.pt"
ASPECT_HEADS_CONFIG_FILE = "aspect_heads_config.json"

ASPECT_PREFIX_BY_NAME: Dict[str, str] = {
    "topic": "[TOPIC]",
    "approach": "[APPROACH]",
    "objective": "[OBJECTIVE]",
}

ASPECT_CONDITION_TEXTS: Dict[str, str] = {
    "topic": (
        "Attention lens: TOPIC. Compare whether the grant specialization and faculty "
        "specialization share the same subject area, problem context, field, technical "
        "domain, application area, or service area. Ignore similarity that is only about "
        "the method used or the beneficiary/outcome unless it supports the topic match."
    ),
    "approach": (
        "Attention lens: APPROACH. Compare whether the grant specialization and faculty "
        "specialization share the same capability, method, technique, workflow, action, "
        "analysis procedure, design process, intervention, or work performed. Do not give "
        "credit for topic overlap alone."
    ),
    "objective": (
        "Attention lens: OBJECTIVE. Compare whether the grant specialization and faculty "
        "specialization share the same target object, beneficiary, system, material, "
        "dataset, outcome, condition, use case, deployment setting, requirement, or "
        "intended purpose. Do not give credit for method overlap alone."
    ),
}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def aspect_from_prefixed_query(query_text: Any) -> str:
    raw = _clean_text(query_text).lstrip()
    upper = raw.upper()
    for aspect, prefix in ASPECT_PREFIX_BY_NAME.items():
        if upper.startswith(prefix):
            return aspect
    lowered = raw.lower()
    for aspect in ASPECTS:
        if lowered.startswith(f"{aspect}:"):
            return aspect
    return "topic"


def aspect_id_from_name(aspect: Any) -> int:
    a = _clean_text(aspect).lower()
    if a == "topic":
        return 1
    if a == "approach":
        return 2
    if a == "objective":
        return 3
    return 0


def aspect_name_from_id(aspect_id: Any) -> str:
    try:
        parsed = int(aspect_id)
    except Exception:
        parsed = 0
    if parsed == 1:
        return "topic"
    if parsed == 2:
        return "approach"
    if parsed == 3:
        return "objective"
    return "topic"


def strip_aspect_prefix(text: Any) -> str:
    raw = _clean_text(text).lstrip()
    upper = raw.upper()
    for prefix in ASPECT_PREFIX_BY_NAME.values():
        if upper.startswith(prefix):
            return raw[len(prefix) :].strip()
    for aspect in ASPECTS:
        label = f"{aspect}:"
        if raw.lower().startswith(label):
            return raw[len(label) :].strip()
    return raw


def format_aspect_pair(query_text: Any, doc_text: Any) -> Tuple[str, str]:
    aspect = aspect_from_prefixed_query(query_text)
    query = strip_aspect_prefix(query_text)
    doc = strip_aspect_prefix(doc_text)
    condition = ASPECT_CONDITION_TEXTS.get(aspect) or ASPECT_CONDITION_TEXTS["topic"]
    return f"{condition}\nGrant specialization: {query}", f"Faculty specialization: {doc}"


def _find_classifier_attr_name(model: nn.Module) -> Optional[str]:
    for attr in ("classifier", "score", "regressor"):
        if isinstance(getattr(model, attr, None), nn.Module):
            return attr
    return None


def _slice_batch_kwargs(kwargs: Dict[str, Any], mask: torch.Tensor, batch_size: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if key == "labels":
            continue
        if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == int(batch_size):
            out[key] = value[mask]
        else:
            out[key] = value
    out["return_dict"] = True
    return out


class AspectHeadSequenceClassifier(nn.Module):
    """Shared cross-encoder backbone with one scalar head per CE3 aspect."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = getattr(backbone, "config", None)
        self.classifier_attr_name = _find_classifier_attr_name(backbone)
        source_head = getattr(backbone, self.classifier_attr_name, None) if self.classifier_attr_name else None
        if not isinstance(source_head, nn.Module):
            raise RuntimeError("CE3 multi-head mode expects a sequence-classification backbone with a classifier head.")
        self.heads = nn.ModuleDict({aspect: copy.deepcopy(source_head) for aspect in ASPECTS})

    def _clean_aspect_ids(self, aspect_ids: Optional[torch.Tensor], *, batch_size: int, device: torch.device) -> torch.Tensor:
        if aspect_ids is None:
            return torch.ones(batch_size, device=device, dtype=torch.long)
        out = aspect_ids.to(device=device, dtype=torch.long).view(-1)
        if int(out.numel()) != batch_size:
            return torch.ones(batch_size, device=device, dtype=torch.long)
        return out

    def forward(self, *args: Any, aspect_ids: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        if args:
            raise TypeError("AspectHeadSequenceClassifier expects keyword inputs.")
        input_ids = kwargs.get("input_ids")
        if not torch.is_tensor(input_ids):
            raise RuntimeError("Aspect-head routing requires tensor input_ids.")
        if self.classifier_attr_name is None:
            raise RuntimeError("Missing classifier attribute for aspect-head routing.")

        batch_size = int(input_ids.shape[0])
        aspect_ids = self._clean_aspect_ids(aspect_ids, batch_size=batch_size, device=input_ids.device)
        original_head = getattr(self.backbone, self.classifier_attr_name)
        logits: Optional[torch.Tensor] = None
        handled = torch.zeros(batch_size, device=input_ids.device, dtype=torch.bool)
        try:
            for aspect_id, aspect_name in ((0, "topic"), (1, "topic"), (2, "approach"), (3, "objective")):
                mask = aspect_ids == int(aspect_id)
                if not bool(mask.any().item()):
                    continue
                setattr(self.backbone, self.classifier_attr_name, self.heads[aspect_name])
                sub_kwargs = _slice_batch_kwargs(kwargs, mask, batch_size)
                head_logits = self.backbone(**sub_kwargs).logits
                if logits is None:
                    logits = head_logits.new_empty((batch_size, int(head_logits.shape[-1])))
                logits[mask] = head_logits.to(dtype=logits.dtype)
                handled = handled | mask
            if bool((~handled).any().item()):
                setattr(self.backbone, self.classifier_attr_name, self.heads["topic"])
                sub_kwargs = _slice_batch_kwargs(kwargs, ~handled, batch_size)
                head_logits = self.backbone(**sub_kwargs).logits
                if logits is None:
                    logits = head_logits.new_empty((batch_size, int(head_logits.shape[-1])))
                logits[~handled] = head_logits.to(dtype=logits.dtype)
        finally:
            setattr(self.backbone, self.classifier_attr_name, original_head)

        if logits is None:
            logits = input_ids.new_zeros((batch_size, 1), dtype=torch.float32)
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
                    "aspects": list(ASPECTS),
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
    multi_aspect_heads: bool = True,
    trust_remote_code: bool = True,
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
