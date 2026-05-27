from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
    DECOMPOSE_BATCH_SIZE_DEFAULT,
    DECOMPOSE_MAX_NEW_TOKENS_DEFAULT,
    FAC_DB_DEFAULT,
    GRANT_DB_DEFAULT,
    MAX_ATTEMPTS_DEFAULT,
    MAX_MODEL_LEN_DEFAULT,
    MODEL_ID_DEFAULT,
    SEED_DEFAULT,
    TEMPERATURE_DEFAULT,
    TOP_P_DEFAULT,
    resolve_path,
)


DECOMPOSITION_OUTPUT_DEFAULT = "ce2/test/output/spec_decompositions_subset.jsonl"
PREVIEW_OUTPUT_DEFAULT = "ce2/test/output/spec_decompositions_subset_preview.txt"
SAFE_TEST_OUTPUT_ROOT = "ce2/test/output"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_ws(value: Any) -> str:
    return " ".join(_clean_text(value).split())


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


def _items(value: Any) -> List[str]:
    if isinstance(value, list):
        out = [_normalize_ws(v) for v in value if _normalize_ws(v)]
        return out
    return []


def _format_decomp_list(items: List[str]) -> str:
    if not items:
        return "[]"
    return "[" + ", ".join(items) + "]"


def _write_preview(
    *,
    rows: List[Dict[str, Any]],
    preview_path: Path,
    preview_count: int,
    preview_kind: str,
) -> None:
    selected: List[Dict[str, Any]] = []
    for row in rows:
        kind = _normalize_ws(row.get("kind")).lower()
        if preview_kind == "all":
            selected.append(row)
        elif preview_kind == "grant" and kind == "grant":
            selected.append(row)
        elif preview_kind == "faculty" and kind == "faculty":
            selected.append(row)
        if len(selected) >= int(preview_count):
            break

    lines: List[str] = []
    lines.append("CE2 Decomposition Subset Preview")
    lines.append(f"rows_total={len(rows)} rows_shown={len(selected)} filter_kind={preview_kind}")
    lines.append("")

    for i, row in enumerate(selected, start=1):
        decomp = row.get("decomposition") if isinstance(row.get("decomposition"), dict) else {}
        parse = row.get("decomposition_parse") if isinstance(row.get("decomposition_parse"), dict) else {}
        lines.append(f"[{i}] item_id={_normalize_ws(row.get('item_id'))}")
        lines.append(
            f"kind={_normalize_ws(row.get('kind'))} parse_ok={bool(row.get('parse_ok', False))} "
            f"attempt={int(row.get('attempt') or 0)}"
        )
        lines.append(f"text={_truncate(_normalize_ws(row.get('text')), 240)}")
        lines.append(f"domain={_format_decomp_list(_items(decomp.get('domain')))}")
        lines.append(f"method={_format_decomp_list(_items(decomp.get('method')))}")
        lines.append(f"target={_format_decomp_list(_items(decomp.get('target')))}")
        if parse:
            lines.append(
                "parse_summary="
                f"parsed={int(parse.get('parsed_aspects_count', 0))} "
                f"nonempty={int(parse.get('nonempty_aspects_count', 0))}"
            )
        lines.append("")

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run a tiny decomposition smoke pass using the exact same CE2 decomposition script "
            "(same prompts and logic), then write a readable preview."
        )
    )
    p.add_argument("--python-bin", type=str, default=sys.executable or "python")
    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    p.add_argument("--grant-db", type=str, default=GRANT_DB_DEFAULT)
    p.add_argument("--fac-db", type=str, default=FAC_DB_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--max-grant-specs", type=int, default=6)
    p.add_argument("--max-fac-specs", type=int, default=6)
    p.add_argument("--decompose-batch-size", type=int, default=DECOMPOSE_BATCH_SIZE_DEFAULT)
    p.add_argument("--decompose-max-new-tokens", type=int, default=DECOMPOSE_MAX_NEW_TOKENS_DEFAULT)
    p.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    p.add_argument("--top-p", type=float, default=TOP_P_DEFAULT)
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN_DEFAULT)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--refresh-failed-decompositions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--refresh-all-decompositions", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--decomposition-output", type=str, default=DECOMPOSITION_OUTPUT_DEFAULT)
    p.add_argument("--preview-output", type=str, default=PREVIEW_OUTPUT_DEFAULT)
    p.add_argument("--preview-count", type=int, default=12)
    p.add_argument("--preview-kind", type=str, choices=("all", "grant", "faculty"), default="all")
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
    preview_output = resolve_path(PROJECT_ROOT, args.preview_output)
    safe_root = resolve_path(PROJECT_ROOT, SAFE_TEST_OUTPUT_ROOT)
    _assert_safe_output_path(
        decomposition_output,
        safe_root=safe_root,
        allow_non_test_output=bool(args.allow_non_test_output),
    )
    _assert_safe_output_path(
        preview_output,
        safe_root=safe_root,
        allow_non_test_output=bool(args.allow_non_test_output),
    )
    output_dir = decomposition_output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.python_bin),
        "ce2/data_preparation/decompose_aspect_specs.py",
        "--model-id",
        str(args.model_id),
        "--grant-db",
        str(args.grant_db),
        "--fac-db",
        str(args.fac_db),
        "--output-dir",
        str(output_dir),
        "--decomposition-output",
        str(decomposition_output),
        "--seed",
        str(int(args.seed)),
        "--max-grant-specs",
        str(int(args.max_grant_specs)),
        "--max-fac-specs",
        str(int(args.max_fac_specs)),
        "--decompose-batch-size",
        str(int(args.decompose_batch_size)),
        "--decompose-max-new-tokens",
        str(int(args.decompose_max_new_tokens)),
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
    if bool(args.refresh_failed_decompositions):
        cmd.append("--refresh-failed-decompositions")
    else:
        cmd.append("--no-refresh-failed-decompositions")
    if bool(args.refresh_all_decompositions):
        cmd.append("--refresh-all-decompositions")
    else:
        cmd.append("--no-refresh-all-decompositions")

    print(f"running_cmd={' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

    rows = [row for row in _iter_jsonl(decomposition_output)]
    _write_preview(
        rows=rows,
        preview_path=preview_output,
        preview_count=max(1, int(args.preview_count)),
        preview_kind=_normalize_ws(args.preview_kind).lower() or "all",
    )

    print(f"decomposition_output={decomposition_output}")
    print(f"preview_output={preview_output}")
    print(f"rows_total={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
