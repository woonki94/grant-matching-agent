from __future__ import annotations

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
FINETUNED_MODEL_PATH = "cross_encoder/spec_to_chunk/models/bge_reranker_distill/best"
QUERY_TEXT = "Reinforcement learning combined with symbolic planning and reasoning for long-horizon decision-making in dynamic, uncertain real-world environments."

DOC_TEXTS = [
    # 🟢 Strong match (~0.9–1.0)
    "We present an integrated framework for long-horizon decision-making in dynamic and uncertain environments by combining reinforcement learning with symbolic planning and reasoning. The approach uses reinforcement learning to acquire low-level control policies while a symbolic planner encodes high-level task structure, constraints, and goals. The system supports temporal abstraction, hierarchical planning, and adaptive decision-making through interaction between learned policies and structured reasoning components. Experimental results in robotic navigation and manipulation tasks demonstrate improved performance in multi-stage environments requiring sequential decision-making, adaptability, and robustness under uncertainty. The proposed framework enables effective coordination between learning-based control and symbolic reasoning for complex real-world scenarios.",

    # 🟢 Good match (~0.75–0.85)
    "This work explores hybrid approaches that integrate reinforcement learning with planning techniques for multi-step decision-making in dynamic environments. The proposed method combines learned policies with structured representations of task sequences, allowing improved long-horizon performance and adaptability under uncertainty. The system incorporates elements of hierarchical planning and temporal abstraction, enabling efficient reasoning over extended decision horizons. Evaluations in simulated control and navigation tasks show that the hybrid approach improves stability and scalability compared to purely learning-based methods, particularly in environments with complex state transitions and delayed rewards.",

    # 🟡 Partial match (~0.6–0.7)
    "We study reinforcement learning methods for long-horizon decision-making in dynamic environments using hierarchical policies and temporal abstraction. The approach focuses on learning scalable control strategies that can handle complex sequential tasks over extended time horizons. Experimental results demonstrate improved policy efficiency and adaptability across a range of simulated environments. The work emphasizes reinforcement learning techniques for planning-like behavior through learned abstractions, enabling more effective decision-making in uncertain and changing conditions.",

    # 🟡 Weak / domain overlap (~0.4–0.55)
    "This paper investigates reinforcement learning techniques for optimizing decision-making policies in dynamic environments, with an emphasis on cost efficiency and reward shaping. The proposed methods focus on scalable policy optimization and efficient exploration in large state spaces. Experimental evaluation demonstrates improved performance in resource allocation and control tasks, highlighting the effectiveness of reinforcement learning for decision-making under uncertainty in practical applications.",

    # 🔴 Weak / partial domain (~0.3–0.4)
    "We present a framework for symbolic planning and reasoning in dynamic environments using structured representations and search-based methods. The approach focuses on constraint-driven task decomposition and efficient planning over long horizons. Experimental results demonstrate improved performance in planning tasks requiring logical reasoning and structured decision-making. The system enables effective handling of complex task dependencies and goal hierarchies in dynamic settings.",

    # 🔴 Negative (~0.0–0.1)
    "This paper studies statistical forecasting and optimization methods for supply chain systems, including demand prediction, inventory control, and logistics planning. The approach uses probabilistic models and operations research techniques to improve decision-making efficiency in large-scale systems. Experimental results demonstrate improved forecasting accuracy and resource allocation performance across multiple real-world datasets."
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


def main() -> int:
    model_path = _resolve_path(FINETUNED_MODEL_PATH)
    query = _clean_text(QUERY_TEXT)
    docs = [_clean_text(x) for x in DOC_TEXTS if _clean_text(x)]

    if not model_path.exists():
        raise RuntimeError(f"Model path not found: {model_path}")
    if not query:
        raise RuntimeError("QUERY_TEXT is empty. Paste one query string in the file.")
    if not docs:
        raise RuntimeError("DOC_TEXTS is empty. Paste one or more doc strings in the file.")

    device = _pick_device()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), num_labels=1)
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
    print(f"model_path={model_path.resolve()}")
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
