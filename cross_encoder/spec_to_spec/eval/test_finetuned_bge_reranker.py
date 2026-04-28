from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure project root on sys.path for direct script execution.
def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "cross_encoder").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Edit these three fields directly.
FINETUNED_MODEL_PATH = "cross_encoder/spec_to_spec/models/spec_to_spec_finetuned_ce"
PURE_BGE_MODEL_ID = "BAAI/bge-reranker-base"

QUERY_TEXT = "Methods for sim-to-real transfer enabling deployment of learned robot controllers on physical hardware platforms."

DOC_TEXTS = [
    # PERFECT
    "Sim-to-real transfer learning for deploying robot controllers from simulation to physical hardware",

    # Strong (related but less explicit)
    "Vision-based bipedal and humanoid locomotion via sim-to-real transfer learning",

    # Partial (robot learning but no transfer)
    "Offline reinforcement learning with safety constraints and policy adaptation",

    # Partial (control but no sim-to-real)
    "Reinforcement learning for sequential decision-making with visual perception and planning",

    # Trap (general ML)
    "Deep learning combining neural networks with AI planning and reasoning engines",

    # Negative
    "Agricultural decision support systems for optimizing crop management and irrigation using ML"
]

# Optional settings.
MAX_LENGTH = 512
BATCH_SIZE = 16
SORT_DESCENDING = True


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_path(value: Any) -> Path:
    path = Path(_clean_text(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_model_ref(*, model_ref: str, use_pure_bge: bool) -> str:
    selected = _clean_text(model_ref)
    if not selected:
        selected = PURE_BGE_MODEL_ID if bool(use_pure_bge) else _clean_text(FINETUNED_MODEL_PATH)

    if not selected:
        raise RuntimeError("Model reference is empty.")

    path_candidate = Path(selected).expanduser()
    if path_candidate.exists():
        return str(path_candidate.resolve())

    project_candidate = _resolve_path(selected)
    if project_candidate.exists():
        return str(project_candidate)

    # Fallback to Hugging Face model id.
    return selected


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _score_docs(
    *,
    model: Any,
    tokenizer: Any,
    query_text: str,
    docs: List[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for i in range(0, len(docs), max(1, int(batch_size))):
            docs_batch = docs[i : i + int(batch_size)]
            queries_batch = [query_text] * len(docs_batch)

            enc = tokenizer(
                queries_batch,
                docs_batch,
                max_length=int(max_length),
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = _to_device(enc, device)

            logits = model(**enc).logits.squeeze(-1)
            probs = torch.sigmoid(logits)

            for j, doc_text in enumerate(docs_batch):
                rows.append(
                    {
                        "doc_index": int(i + j),
                        "raw_logit": float(logits[j].detach().cpu().item()),
                        "score": float(probs[j].detach().cpu().item()),
                        "doc_text": str(doc_text),
                    }
                )

    return rows


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Test CE scoring with either a local finetuned checkpoint or a pure BGE reranker model ID."
    )
    p.add_argument(
        "--use-pure-bge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=f"Use pure BGE CE model ({PURE_BGE_MODEL_ID}) instead of local finetuned checkpoint.",
    )
    p.add_argument(
        "--model-ref",
        type=str,
        default="",
        help="Optional explicit model ref (local path or Hugging Face model id). Overrides --use-pure-bge.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    model_ref = _resolve_model_ref(model_ref=str(args.model_ref), use_pure_bge=bool(args.use_pure_bge))
    query = _clean_text(QUERY_TEXT)
    docs = [_clean_text(x) for x in DOC_TEXTS if _clean_text(x)]

    if not query:
        raise RuntimeError("QUERY_TEXT is empty. Paste one query string in the file.")
    if not docs:
        raise RuntimeError("DOC_TEXTS is empty. Paste one or more doc strings in the file.")

    device = _pick_device()

    tokenizer = AutoTokenizer.from_pretrained(str(model_ref))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_ref), num_labels=1)
    model.to(device)
    model.eval()

    rows = _score_docs(
        model=model,
        tokenizer=tokenizer,
        query_text=query,
        docs=docs,
        device=device,
        max_length=int(MAX_LENGTH),
        batch_size=int(BATCH_SIZE),
    )

    rows.sort(key=lambda x: float(x["score"]), reverse=bool(SORT_DESCENDING))

    print(f"device={device}")
    print(f"model_ref={model_ref}")
    print(f"query={query}")
    print(f"doc_count={len(rows)}")
    print("\nScores:")

    for rank, row in enumerate(rows, start=1):
        preview = row["doc_text"][:180].replace("\n", " ")
        print(
            f"{rank:02d}. doc_index={row['doc_index']} "
            f"score={row['score']:.6f} "
            f"raw_logit={row['raw_logit']:.6f} "
            f"text_preview={preview}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
