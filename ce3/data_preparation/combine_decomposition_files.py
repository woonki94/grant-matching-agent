from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce3").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce3.data_preparation.llm_runtime import clean_text, normalize_ws  # noqa: E402
from ce3.data_preparation.utils import resolve_path  # noqa: E402


ORIGINAL_DECOMPOSITION_DEFAULT = "ce3/dataset/decomposed/spec_decompositions_topic_approach_objective.jsonl"
AUGMENTED_DECOMPOSITION_DEFAULT = "ce3/dataset/decomposed/spec_decompositions_augmented.jsonl"
COMBINED_OUTPUT_DEFAULT = "ce3/dataset/decomposed/spec_decompositions_combined.jsonl"


def _iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            item_id = clean_text(row.get("item_id"))
            kind = clean_text(row.get("kind"))
            text = normalize_ws(row.get("text"))
            decomp = row.get("decomposition")
            if not item_id or not kind or not text or not isinstance(decomp, dict):
                continue
            yield row


def combine_files(paths: List[Path], output_path: Path) -> Dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    kind_counts: Dict[str, int] = {}
    written = 0
    skipped_duplicates = 0
    with output_path.open("w", encoding="utf-8") as out:
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Missing decomposition input: {path}")
            for row in _iter_rows(path):
                item_id = clean_text(row.get("item_id"))
                if item_id in seen:
                    skipped_duplicates += 1
                    continue
                seen.add(item_id)
                kind = clean_text(row.get("kind"))
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    return {"written_rows": written, "skipped_duplicates": skipped_duplicates, **{f"kind_{k}": v for k, v in kind_counts.items()}}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine CE3 original and augmented decomposition JSONL files.")
    p.add_argument("--original-decomposition", type=str, default=ORIGINAL_DECOMPOSITION_DEFAULT)
    p.add_argument("--augmented-decomposition", type=str, default=AUGMENTED_DECOMPOSITION_DEFAULT)
    p.add_argument("--combined-output", type=str, default=COMBINED_OUTPUT_DEFAULT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    original = resolve_path(args.original_decomposition)
    augmented = resolve_path(args.augmented_decomposition)
    output = resolve_path(args.combined_output)
    stats = combine_files([original, augmented], output)
    print(json.dumps({"combined_output": str(output), **stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
