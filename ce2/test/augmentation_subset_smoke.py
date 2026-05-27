from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce2").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce2.data_preparation.utils import (  # noqa: E402
    DISTILL_BATCH_SIZE_DEFAULT,
    DISTILL_MAX_NEW_TOKENS_DEFAULT,
    MAX_MODEL_LEN_DEFAULT,
    SEED_DEFAULT,
    resolve_path,
)


DISTILLATION_INPUT_DEFAULT = "ce2/test/output/llm_distillation_subset.jsonl"
DECOMPOSITION_OUTPUT_DEFAULT = "ce2/test/output/spec_decompositions_subset.jsonl"
AUGMENT_OUTPUT_DEFAULT = "ce2/test/output/augmentation_subset.jsonl"
AUGMENT_SUMMARY_DEFAULT = "ce2/test/output/augmentation_subset_summary.json"
PREVIEW_OUTPUT_DEFAULT = "ce2/test/output/augmentation_subset_preview.txt"
SAFE_TEST_OUTPUT_ROOT = "ce2/test/output"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_ws(value: Any) -> str:
    return " ".join(_clean_text(value).split())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = _clean_text(raw)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {type(exc).__name__}: {exc}") from exc
            if isinstance(obj, dict):
                yield obj


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _assert_safe_output_path(path: Path, *, safe_root: Path, allow_non_test_output: bool) -> None:
    if bool(allow_non_test_output):
        return
    if not _is_relative_to(path, safe_root):
        raise RuntimeError(
            f"Refusing to write outside test output root.\n"
            f"path={path}\n"
            f"allowed_root={safe_root}\n"
            "Set --allow-non-test-output to bypass intentionally."
        )


def _truncate(text: str, limit: int) -> str:
    raw = _normalize_ws(text)
    if len(raw) <= int(limit):
        return raw
    return raw[: max(0, int(limit) - 3)] + "..."


def _aspect_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    c = Counter()
    for row in rows:
        c[_normalize_ws(row.get("aspect")).lower() or "unknown"] += 1
    return {k: int(c[k]) for k in sorted(c)}


def _band_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    c = Counter()
    for row in rows:
        c[_normalize_ws(row.get("band")).lower() or "unknown"] += 1
    return {k: int(c[k]) for k in sorted(c)}


def _write_preview(
    *,
    rows: List[Dict[str, Any]],
    preview_path: Path,
    preview_count: int,
) -> None:
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            _normalize_ws(r.get("augment_target_aspect")),
            _normalize_ws(r.get("augment_target_band")),
            _normalize_ws(r.get("aspect")),
            -_safe_float(r.get("score"), default=0.0),
            _normalize_ws(r.get("pair_id")),
        ),
    )
    keep = sorted_rows[: max(1, int(preview_count))]

    lines: List[str] = []
    lines.append("CE2 Augmentation Subset Preview")
    lines.append(
        f"rows_total={len(rows)} rows_shown={len(keep)} "
        f"aspect_counts={json.dumps(_aspect_counts(rows), ensure_ascii=False)} "
        f"band_counts={json.dumps(_band_counts(rows), ensure_ascii=False)}"
    )
    lines.append("")

    for i, row in enumerate(keep, start=1):
        grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
        fac = row.get("faculty") if isinstance(row.get("faculty"), dict) else {}
        lines.append(f"[{i}] pair_id={_normalize_ws(row.get('pair_id'))}")
        lines.append(
            f"aspect={_normalize_ws(row.get('aspect'))} "
            f"score={_safe_float(row.get('score'), 0.0):.4f} "
            f"band={_normalize_ws(row.get('band')).lower()} "
            f"target={_normalize_ws(row.get('augment_target_aspect'))}/{_normalize_ws(row.get('augment_target_band')).lower()}"
        )
        lines.append(
            f"pair_source={_normalize_ws(row.get('pair_source'))} "
            f"is_augmented={bool(row.get('is_augmented', False))} "
            f"generation_note={_truncate(_normalize_ws(row.get('augment_generation_note')), 140)}"
        )
        lines.append(f"grant_item_id={_normalize_ws(grant.get('item_id'))}")
        lines.append(f"grant_text={_truncate(_normalize_ws(grant.get('text')), 220)}")
        lines.append(f"faculty_item_id={_normalize_ws(fac.get('item_id'))}")
        lines.append(f"faculty_text={_truncate(_normalize_ws(fac.get('text')), 220)}")
        lines.append("")

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run a tiny augmentation smoke pass using the exact CE2 augmentation script "
            "(same prompts and logic), then write a readable preview."
        )
    )
    p.add_argument("--python-bin", type=str, default=sys.executable or "python")
    p.add_argument("--model-id", type=str, default="auto")
    p.add_argument("--distillation-input", type=str, default=DISTILLATION_INPUT_DEFAULT)
    p.add_argument("--decomposition-output", type=str, default=DECOMPOSITION_OUTPUT_DEFAULT)
    p.add_argument("--output", type=str, default=AUGMENT_OUTPUT_DEFAULT)
    p.add_argument("--summary-output", type=str, default=AUGMENT_SUMMARY_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--target-policy", type=str, choices=("median", "explicit"), default="explicit")
    p.add_argument("--target-high", type=int, default=2)
    p.add_argument("--target-mid", type=int, default=2)
    p.add_argument("--target-low", type=int, default=2)
    p.add_argument("--max-add-per-band", type=int, default=48)
    p.add_argument("--gen-batch-size", type=int, default=24)
    p.add_argument("--gen-max-new-tokens", type=int, default=512)
    p.add_argument("--max-tries-per-missing", type=int, default=6)
    p.add_argument("--distill-batch-size", type=int, default=DISTILL_BATCH_SIZE_DEFAULT)
    p.add_argument("--distill-max-new-tokens", type=int, default=DISTILL_MAX_NEW_TOKENS_DEFAULT)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN_DEFAULT)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--preview-output", type=str, default=PREVIEW_OUTPUT_DEFAULT)
    p.add_argument("--preview-count", type=int, default=30)
    p.add_argument("--write-preview", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--write-summary", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--allow-non-test-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow writing outputs outside ce2/test/output (disabled by default for safety).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    distillation_input = resolve_path(PROJECT_ROOT, args.distillation_input)
    decomposition_output = resolve_path(PROJECT_ROOT, args.decomposition_output)
    augment_output = resolve_path(PROJECT_ROOT, args.output)
    summary_output = resolve_path(PROJECT_ROOT, args.summary_output)
    summary_tmp_output = summary_output.parent / f"{summary_output.stem}.tmp.json"
    preview_output = resolve_path(PROJECT_ROOT, args.preview_output)
    safe_root = resolve_path(PROJECT_ROOT, SAFE_TEST_OUTPUT_ROOT)

    if not distillation_input.exists():
        raise FileNotFoundError(
            f"Missing distillation input: {distillation_input}\n"
            "Run ce2/test/run_llm_distillation_subset.sh first, or pass --distillation-input."
        )
    if not decomposition_output.exists():
        raise FileNotFoundError(
            f"Missing decomposition output: {decomposition_output}\n"
            "Run ce2/test/run_decomposition_subset.sh first, or pass --decomposition-output."
        )
    _assert_safe_output_path(
        augment_output,
        safe_root=safe_root,
        allow_non_test_output=bool(args.allow_non_test_output),
    )
    _assert_safe_output_path(
        summary_output,
        safe_root=safe_root,
        allow_non_test_output=bool(args.allow_non_test_output),
    ) if bool(args.write_summary) else _assert_safe_output_path(
        summary_tmp_output,
        safe_root=safe_root,
        allow_non_test_output=bool(args.allow_non_test_output),
    )
    if bool(args.write_preview):
        _assert_safe_output_path(
            preview_output,
            safe_root=safe_root,
            allow_non_test_output=bool(args.allow_non_test_output),
        )

    summary_output_run = summary_output if bool(args.write_summary) else summary_tmp_output
    cmd = [
        str(args.python_bin),
        "ce2/data_preparation/augmentation.py",
        "--model-id",
        str(args.model_id),
        "--distillation-input",
        str(distillation_input),
        "--decomposition-output",
        str(decomposition_output),
        "--output",
        str(augment_output),
        "--summary-output",
        str(summary_output_run),
        "--seed",
        str(int(args.seed)),
        "--target-policy",
        str(args.target_policy),
        "--target-high",
        str(int(args.target_high)),
        "--target-mid",
        str(int(args.target_mid)),
        "--target-low",
        str(int(args.target_low)),
        "--max-add-per-band",
        str(int(args.max_add_per_band)),
        "--gen-batch-size",
        str(int(args.gen_batch_size)),
        "--gen-max-new-tokens",
        str(int(args.gen_max_new_tokens)),
        "--max-tries-per-missing",
        str(int(args.max_tries_per_missing)),
        "--distill-batch-size",
        str(int(args.distill_batch_size)),
        "--distill-max-new-tokens",
        str(int(args.distill_max_new_tokens)),
        "--temperature",
        str(float(args.temperature)),
        "--top-p",
        str(float(args.top_p)),
        "--max-model-len",
        str(int(args.max_model_len)),
        "--gpu-memory-utilization",
        str(float(args.gpu_memory_utilization)),
        "--tensor-parallel-size",
        str(int(args.tensor_parallel_size)),
    ]
    if bool(args.overwrite):
        cmd.append("--overwrite")

    print(f"running_cmd={' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

    if (not bool(args.write_summary)) and summary_output_run.exists():
        summary_output_run.unlink()

    rows = [row for row in _iter_jsonl(augment_output)] if augment_output.exists() else []
    if bool(args.write_preview):
        _write_preview(rows=rows, preview_path=preview_output, preview_count=max(1, int(args.preview_count)))

    print(f"distillation_input={distillation_input}")
    print(f"decomposition_output={decomposition_output}")
    print(f"augmentation_output={augment_output}")
    if bool(args.write_summary):
        print(f"summary_output={summary_output}")
    if bool(args.write_preview):
        print(f"preview_output={preview_output}")
    print(f"rows_total={len(rows)}")
    print(f"aspect_counts={json.dumps(_aspect_counts(rows), ensure_ascii=False)}")
    print(f"band_counts={json.dumps(_band_counts(rows), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
