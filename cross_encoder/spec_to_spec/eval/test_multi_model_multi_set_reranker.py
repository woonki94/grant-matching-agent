from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

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


# -----------------------------------------------------------------------------
# Manual config: edit here only.
# -----------------------------------------------------------------------------
MODEL_SPECS: List[Dict[str, str]] = [
    {
        "name": "finetuned_spec_to_spec",
        "model_ref": "cross_encoder/spec_to_spec/models/spec_to_spec_finetuned_ce",
    },
    {
        "name": "pure_bge_base",
        "model_ref": "BAAI/bge-reranker-base",
    },
]

TEST_SETS: List[Dict[str, Any]] = [
    {
        "name": "robotics_sim2real",
        "query_text": "Methods for sim-to-real transfer enabling deployment of learned robot controllers on physical hardware platforms.",
        "doc_texts": [
            "Sim-to-real transfer learning for deploying robot controllers from simulation to physical hardware",
            "Vision-based bipedal and humanoid locomotion via sim-to-real transfer learning",
            "Offline reinforcement learning with safety constraints and policy adaptation",
            "Reinforcement learning for sequential decision-making with visual perception and planning",
            "Deep learning combining neural networks with AI planning and reasoning engines",
            "Agricultural decision support systems for optimizing crop management and irrigation using ML",
        ],
    },
    {
        "name": "materials_energy",
        "query_text": "Advanced materials for sustainable energy storage and electrochemical conversion with robust long-cycle performance.",
        "doc_texts": [
            "Novel battery electrode materials and interfaces for high-capacity, long-cycle lithium-ion storage",
            "Catalyst design for electrochemical CO2 conversion and hydrogen production",
            "Machine learning methods for materials discovery in energy applications",
            "Clinical informatics for patient risk prediction in hospital settings",
            "Sustainable polymer membranes for fuel cell durability and ion transport",
            "Robotics path planning and multi-agent coordination in unknown environments",
        ],
    },
]

MAX_LENGTH = 512
BATCH_SIZE = 16
SORT_DESCENDING = True
PREVIEW_CHARS = 160


# -----------------------------------------------------------------------------
# Helpers.
# -----------------------------------------------------------------------------
def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_path(value: Any) -> Path:
    path = Path(_clean_text(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_model_ref(model_ref: str) -> str:
    selected = _clean_text(model_ref)
    if not selected:
        raise RuntimeError("Model reference is empty in MODEL_SPECS.")

    path_candidate = Path(selected).expanduser()
    if path_candidate.exists():
        return str(path_candidate.resolve())

    project_candidate = _resolve_path(selected)
    if project_candidate.exists():
        return str(project_candidate)

    # Fallback to Hugging Face model ID.
    return selected


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _shorten(text: str, max_chars: int) -> str:
    compact = " ".join(_clean_text(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)] + "..."


def _score_docs(
    *,
    model: Any,
    tokenizer: Any,
    query_text: str,
    docs: Sequence[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        step = max(1, int(batch_size))
        for i in range(0, len(docs), step):
            docs_batch = list(docs[i : i + step])
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


def _validate_config() -> None:
    if not MODEL_SPECS:
        raise RuntimeError("MODEL_SPECS is empty. Add at least one model.")
    if not TEST_SETS:
        raise RuntimeError("TEST_SETS is empty. Add at least one query/doc set.")

    for spec in MODEL_SPECS:
        if not _clean_text(spec.get("name")) or not _clean_text(spec.get("model_ref")):
            raise RuntimeError(f"Invalid model spec: {spec}")

    for test in TEST_SETS:
        name = _clean_text(test.get("name"))
        query = _clean_text(test.get("query_text"))
        docs = [_clean_text(x) for x in list(test.get("doc_texts") or []) if _clean_text(x)]
        if not name:
            raise RuntimeError(f"A test set has empty name: {test}")
        if not query:
            raise RuntimeError(f"Test set '{name}' has empty query_text.")
        if not docs:
            raise RuntimeError(f"Test set '{name}' has empty doc_texts.")


def main() -> int:
    _validate_config()
    device = _pick_device()

    # case_name -> model_name -> rows
    all_scores: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    case_meta: Dict[str, Dict[str, Any]] = {}

    for test in TEST_SETS:
        case_name = _clean_text(test["name"])
        case_meta[case_name] = {
            "query_text": _clean_text(test["query_text"]),
            "doc_texts": [_clean_text(x) for x in list(test["doc_texts"]) if _clean_text(x)],
        }
        all_scores[case_name] = {}

    print(f"device={device}")
    print(f"test_set_count={len(TEST_SETS)}")
    print(f"model_count={len(MODEL_SPECS)}")

    for model_spec in MODEL_SPECS:
        model_name = _clean_text(model_spec["name"])
        model_ref = _resolve_model_ref(_clean_text(model_spec["model_ref"]))

        print("\n" + "=" * 92)
        print(f"Loading model: {model_name}")
        print(f"model_ref={model_ref}")

        tokenizer = AutoTokenizer.from_pretrained(model_ref)
        model = AutoModelForSequenceClassification.from_pretrained(model_ref, num_labels=1)
        model.to(device)
        model.eval()

        for case_name, meta in case_meta.items():
            rows = _score_docs(
                model=model,
                tokenizer=tokenizer,
                query_text=str(meta["query_text"]),
                docs=list(meta["doc_texts"]),
                device=device,
                max_length=int(MAX_LENGTH),
                batch_size=int(BATCH_SIZE),
            )
            rows.sort(key=lambda x: float(x["score"]), reverse=bool(SORT_DESCENDING))
            all_scores[case_name][model_name] = rows

        # Release model before loading next model.
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Print per-case reports.
    for case_name, model_rows in all_scores.items():
        query_text = str(case_meta[case_name]["query_text"])
        doc_texts = list(case_meta[case_name]["doc_texts"])

        print("\n" + "#" * 110)
        print(f"CASE: {case_name}")
        print(f"QUERY: {query_text}")
        print(f"DOC_COUNT: {len(doc_texts)}")

        # Table 1: one row per doc with model score columns (original doc order).
        print("\n[Combined Scores by Original doc_index]")
        header = "doc_index"
        for spec in MODEL_SPECS:
            header += f" | {_clean_text(spec['name'])[:22]:>22}"
        header += " | doc_preview"
        print(header)
        print("-" * min(200, len(header) + 40))

        for doc_index, doc_text in enumerate(doc_texts):
            line = f"{doc_index:>9}"
            for spec in MODEL_SPECS:
                model_name = _clean_text(spec["name"])
                rows = model_rows.get(model_name, [])
                score_map = {int(r["doc_index"]): float(r["score"]) for r in rows}
                line += f" | {score_map.get(doc_index, 0.0):>22.6f}"
            line += f" | {_shorten(doc_text, PREVIEW_CHARS)}"
            print(line)

        # Table 2+: ranking per model.
        for spec in MODEL_SPECS:
            model_name = _clean_text(spec["name"])
            rows = list(model_rows.get(model_name, []))

            print(f"\n[Ranking: {model_name}]")
            for rank, row in enumerate(rows, start=1):
                print(
                    f"{rank:02d}. doc_index={int(row['doc_index']):>2} "
                    f"score={float(row['score']):.6f} "
                    f"raw_logit={float(row['raw_logit']):.6f} "
                    f"text={_shorten(str(row['doc_text']), PREVIEW_CHARS)}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
