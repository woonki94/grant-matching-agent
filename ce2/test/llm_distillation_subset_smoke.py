from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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
    FAC_DB_DEFAULT,
    GRANT_DB_DEFAULT,
    MAX_ATTEMPTS_DEFAULT,
    MAX_MODEL_LEN_DEFAULT,
    MODEL_ID_DEFAULT,
    PREFILTER_CACHE_OUTPUT_DEFAULT,
    PREFILTER_HIGH_THRESHOLD_DEFAULT,
    PREFILTER_LOW_THRESHOLD_DEFAULT,
    PREFILTER_MULTIPLIER_HIGH_DEFAULT,
    PREFILTER_MULTIPLIER_LOW_DEFAULT,
    PREFILTER_MULTIPLIER_MID_DEFAULT,
    SEED_DEFAULT,
    TEMPERATURE_DEFAULT,
    TOP_P_DEFAULT,
    prefilter_cache_paths,
    resolve_path,
)


DECOMPOSITION_OUTPUT_DEFAULT = "ce2/test/output/spec_decompositions_subset.jsonl"
DISTILLATION_OUTPUT_DEFAULT = "ce2/test/output/llm_distillation_subset.jsonl"
SUMMARY_OUTPUT_DEFAULT = "ce2/test/output/llm_distillation_subset_summary.json"
PREVIEW_OUTPUT_DEFAULT = "ce2/test/output/llm_distillation_subset_preview.txt"
SAFE_TEST_OUTPUT_ROOT = "ce2/test/output"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_ws(value: Any) -> str:
    return " ".join(_clean_text(value).split())


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


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


def _resolve_prefilter_source(*, requested: str, prefilter_cache: Path) -> Tuple[str, str]:
    choice = _normalize_ws(requested).lower()
    if choice in {"ce-cache", "sts"}:
        return choice, "explicit"

    paths = prefilter_cache_paths(prefilter_cache)
    missing = [name for name, path in paths.items() if not path.exists()]
    if not missing:
        return "ce-cache", "auto_detected_cache_available"
    return "sts", f"auto_fallback_missing_cache({','.join(missing)})"


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


def _write_preview(*, rows: List[Dict[str, Any]], preview_path: Path, preview_count: int) -> None:
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            _normalize_ws(r.get("aspect")),
            -_safe_float(r.get("score"), default=0.0),
            _normalize_ws(r.get("pair_id")),
        ),
    )
    keep = sorted_rows[: max(1, int(preview_count))]

    lines: List[str] = []
    lines.append("CE2 LLM Distillation Subset Preview")
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
            f"source={_normalize_ws(row.get('pair_source'))} "
            f"lexical_prefilter={_safe_float(row.get('lexical_prefilter_score'), 0.0):.4f}"
        )
        lines.append(
            "selected_clusters="
            + json.dumps(row.get("distill_selected_clusters") or [], ensure_ascii=False)
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
            "Run a tiny llm_distillation smoke pass using the exact CE2 distillation script "
            "(same prompts and logic), then write a readable preview."
        )
    )
    p.add_argument("--python-bin", type=str, default=sys.executable or "python")
    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    p.add_argument("--grant-db", type=str, default=GRANT_DB_DEFAULT)
    p.add_argument("--fac-db", type=str, default=FAC_DB_DEFAULT)
    p.add_argument("--decomposition-output", type=str, default=DECOMPOSITION_OUTPUT_DEFAULT)
    p.add_argument("--prefilter-source", type=str, choices=("auto", "ce-cache", "sts"), default="auto")
    p.add_argument("--prefilter-cache", type=str, default=PREFILTER_CACHE_OUTPUT_DEFAULT)
    p.add_argument("--distillation-output", type=str, default=DISTILLATION_OUTPUT_DEFAULT)
    p.add_argument("--summary-output", type=str, default=SUMMARY_OUTPUT_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--max-grant-specs", type=int, default=6)
    p.add_argument("--max-fac-specs", type=int, default=6)
    p.add_argument("--target-high-per-aspect", type=int, default=1)
    p.add_argument("--target-mid-per-aspect", type=int, default=1)
    p.add_argument("--target-low-per-aspect", type=int, default=1)
    p.add_argument("--prefilter-multiplier-high", type=float, default=PREFILTER_MULTIPLIER_HIGH_DEFAULT)
    p.add_argument("--prefilter-multiplier-mid", type=float, default=PREFILTER_MULTIPLIER_MID_DEFAULT)
    p.add_argument("--prefilter-multiplier-low", type=float, default=PREFILTER_MULTIPLIER_LOW_DEFAULT)
    p.add_argument("--prefilter-high-threshold", type=float, default=PREFILTER_HIGH_THRESHOLD_DEFAULT)
    p.add_argument("--prefilter-low-threshold", type=float, default=PREFILTER_LOW_THRESHOLD_DEFAULT)
    p.add_argument("--distill-batch-size", type=int, default=DISTILL_BATCH_SIZE_DEFAULT)
    p.add_argument("--distill-max-new-tokens", type=int, default=DISTILL_MAX_NEW_TOKENS_DEFAULT)
    p.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    p.add_argument("--top-p", type=float, default=TOP_P_DEFAULT)
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN_DEFAULT)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--preview-output", type=str, default=PREVIEW_OUTPUT_DEFAULT)
    p.add_argument("--preview-count", type=int, default=36)
    p.add_argument(
        "--allow-non-test-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow writing outputs outside ce2/test/output (disabled by default for safety).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    decomposition_output = resolve_path(PROJECT_ROOT, args.decomposition_output)
    prefilter_cache = resolve_path(PROJECT_ROOT, args.prefilter_cache)
    distillation_output = resolve_path(PROJECT_ROOT, args.distillation_output)
    summary_output = resolve_path(PROJECT_ROOT, args.summary_output)
    preview_output = resolve_path(PROJECT_ROOT, args.preview_output)
    safe_root = resolve_path(PROJECT_ROOT, SAFE_TEST_OUTPUT_ROOT)

    if not decomposition_output.exists():
        raise FileNotFoundError(
            f"Missing decomposition file: {decomposition_output}\n"
            "Run ce2/test/decomposition_subset_smoke.py first, or pass --decomposition-output."
        )
    _assert_safe_output_path(
        distillation_output,
        safe_root=safe_root,
        allow_non_test_output=bool(args.allow_non_test_output),
    )
    _assert_safe_output_path(
        summary_output,
        safe_root=safe_root,
        allow_non_test_output=bool(args.allow_non_test_output),
    )
    _assert_safe_output_path(
        preview_output,
        safe_root=safe_root,
        allow_non_test_output=bool(args.allow_non_test_output),
    )

    resolved_prefilter, prefilter_reason = _resolve_prefilter_source(
        requested=str(args.prefilter_source),
        prefilter_cache=prefilter_cache,
    )

    out_dir = distillation_output.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.python_bin),
        "ce2/data_preparation/llm_distillation.py",
        "--model-id",
        str(args.model_id),
        "--grant-db",
        str(args.grant_db),
        "--fac-db",
        str(args.fac_db),
        "--output-dir",
        str(out_dir),
        "--decomposition-output",
        str(decomposition_output),
        "--prefilter-source",
        str(resolved_prefilter),
        "--prefilter-cache",
        str(prefilter_cache),
        "--distillation-output",
        str(distillation_output),
        "--summary-output",
        str(summary_output),
        "--seed",
        str(int(args.seed)),
        "--max-grant-specs",
        str(int(args.max_grant_specs)),
        "--max-fac-specs",
        str(int(args.max_fac_specs)),
        "--target-high-per-aspect",
        str(int(args.target_high_per_aspect)),
        "--target-mid-per-aspect",
        str(int(args.target_mid_per_aspect)),
        "--target-low-per-aspect",
        str(int(args.target_low_per_aspect)),
        "--prefilter-multiplier-high",
        str(float(args.prefilter_multiplier_high)),
        "--prefilter-multiplier-mid",
        str(float(args.prefilter_multiplier_mid)),
        "--prefilter-multiplier-low",
        str(float(args.prefilter_multiplier_low)),
        "--prefilter-high-threshold",
        str(float(args.prefilter_high_threshold)),
        "--prefilter-low-threshold",
        str(float(args.prefilter_low_threshold)),
        "--distill-batch-size",
        str(int(args.distill_batch_size)),
        "--distill-max-new-tokens",
        str(int(args.distill_max_new_tokens)),
        "--temperature",
        str(float(args.temperature)),
        "--top-p",
        str(float(args.top_p)),
        "--max-attempts",
        str(int(args.max_attempts)),
        "--max-model-len",
        str(int(args.max_model_len)),
        "--gpu-memory-utilization",
        str(float(args.gpu_memory_utilization)),
        "--tensor-parallel-size",
        str(int(args.tensor_parallel_size)),
    ]
    if bool(args.overwrite):
        cmd.append("--overwrite")

    print(f"prefilter_source_resolved={resolved_prefilter} reason={prefilter_reason}")
    print(f"running_cmd={' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

    rows = [row for row in _iter_jsonl(distillation_output)]
    _write_preview(rows=rows, preview_path=preview_output, preview_count=max(1, int(args.preview_count)))

    print(f"decomposition_output={decomposition_output}")
    print(f"distillation_output={distillation_output}")
    print(f"summary_output={summary_output}")
    print(f"preview_output={preview_output}")
    print(f"rows_total={len(rows)}")
    print(f"aspect_counts={json.dumps(_aspect_counts(rows), ensure_ascii=False)}")
    print(f"band_counts={json.dumps(_band_counts(rows), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
