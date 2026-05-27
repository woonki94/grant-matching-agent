from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


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
    PREFILTER_CACHE_OUTPUT_DEFAULT,
    PREFILTER_HIGH_THRESHOLD_DEFAULT,
    PREFILTER_LOW_THRESHOLD_DEFAULT,
    SEED_DEFAULT,
    load_fac_specs,
    load_grant_specs,
    prefilter_cache_paths,
    select_pairs_from_prefilter_cache,
    TEMPERATURE_DEFAULT,
    TOP_P_DEFAULT,
    resolve_path,
)


DECOMPOSITION_OUTPUT_DEFAULT = "ce2/test/output/spec_decompositions_subset.jsonl"
PREVIEW_OUTPUT_DEFAULT = "ce2/test/output/spec_decompositions_subset_preview.txt"
SUBSET_GRANT_DB_OUTPUT_DEFAULT = "ce2/test/output/grant_keywords_spec_keywords_db_subset.json"
SUBSET_FAC_DB_OUTPUT_DEFAULT = "ce2/test/output/fac_specs_db_subset.json"
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


def _resolve_subset_mode(*, requested: str, prefilter_cache: Path) -> tuple[str, str]:
    mode = _normalize_ws(requested).lower()
    if mode == "random":
        return "random", "explicit_random"
    if mode == "prefilter":
        paths = prefilter_cache_paths(prefilter_cache)
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Requested prefilter subset mode, but cache files are missing: "
                + ", ".join(missing)
            )
        return "prefilter", "explicit_prefilter"
    paths = prefilter_cache_paths(prefilter_cache)
    missing = [name for name, path in paths.items() if not path.exists()]
    if not missing:
        return "prefilter", "auto_prefilter_cache_available"
    return "random", f"auto_fallback_random_missing_cache({','.join(missing)})"


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


def _source_cluster(source: str) -> str:
    parts = [p for p in str(source or "").split("_") if p]
    if len(parts) >= 2:
        return f"{parts[-2]}:{parts[-1]}"
    if parts:
        return parts[-1]
    return "unknown"


def _parse_grant_item_id(item_id: str) -> Tuple[str, int]:
    raw = _normalize_ws(item_id)
    if not raw.startswith("grant:"):
        return "", -1
    rest = raw[len("grant:") :]
    if ":" not in rest:
        return "", -1
    grant_id, idx_raw = rest.rsplit(":", 1)
    try:
        idx = int(idx_raw)
    except Exception:
        return "", -1
    if idx < 0:
        return "", -1
    return grant_id, idx


def _parse_fac_item_id(item_id: str) -> Tuple[str, str, str]:
    raw = _normalize_ws(item_id)
    if not raw.startswith("fac:"):
        return "", "", ""
    rest = raw[len("fac:") :]
    parts = rest.split(":")
    if len(parts) < 3:
        return "", "", ""
    fac_id = _normalize_ws(parts[0])
    fac_spec_idx = _normalize_ws(parts[-1])
    fac_spec_id = _normalize_ws(":".join(parts[1:-1]))
    if not fac_id or not fac_spec_id or not fac_spec_idx:
        return "", "", ""
    return fac_id, fac_spec_id, fac_spec_idx


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rank_grants_by_coverage(pair_rows: Sequence[Tuple[str, str, str]]) -> List[str]:
    coverage: Dict[str, Set[str]] = {}
    for grant_item_id, _fac_item_id, cluster in pair_rows:
        coverage.setdefault(grant_item_id, set()).add(cluster)
    return sorted(coverage.keys(), key=lambda g: (-len(coverage.get(g, set())), g))


def _rank_facs_by_frequency(pair_rows: Sequence[Tuple[str, str, str]]) -> List[str]:
    freq: Dict[str, int] = {}
    for _grant_item_id, fac_item_id, _cluster in pair_rows:
        freq[fac_item_id] = int(freq.get(fac_item_id, 0) + 1)
    return sorted(freq.keys(), key=lambda f: (-int(freq.get(f, 0)), f))


def _build_prefilter_subset_dbs(
    *,
    grant_db_path: Path,
    fac_db_path: Path,
    subset_grant_db_output: Path,
    subset_fac_db_output: Path,
    prefilter_cache: Path,
    seed: int,
    max_grant_specs: int,
    max_fac_specs: int,
    high_per_aspect: int,
    mid_per_aspect: int,
    low_per_aspect: int,
    high_threshold: float,
    low_threshold: float,
) -> Dict[str, Any]:
    grant_specs_all = load_grant_specs(grant_db_path, max_items=0, seed=int(seed))
    fac_specs_all = load_fac_specs(fac_db_path, max_items=0, seed=int(seed))
    pairs = select_pairs_from_prefilter_cache(
        grant_specs_all,
        fac_specs_all,
        cache_base_path=prefilter_cache,
        seed=int(seed),
        high_per_aspect=max(0, int(high_per_aspect)),
        mid_per_aspect=max(0, int(mid_per_aspect)),
        low_per_aspect=max(0, int(low_per_aspect)),
        high_threshold=float(high_threshold),
        low_threshold=float(low_threshold),
    )
    pair_rows: List[Tuple[str, str, str]] = []
    for grant, fac, _score, source in pairs:
        pair_rows.append((str(grant.item_id), str(fac.item_id), _source_cluster(str(source))))

    if int(max_grant_specs) > 0 and pair_rows:
        grant_ranked = _rank_grants_by_coverage(pair_rows)
        keep_grants = set(grant_ranked[: int(max_grant_specs)])
        pair_rows = [row for row in pair_rows if row[0] in keep_grants]

    if int(max_fac_specs) > 0 and pair_rows:
        fac_ranked = _rank_facs_by_frequency(pair_rows)
        keep_facs = set(fac_ranked[: int(max_fac_specs)])
        pair_rows = [row for row in pair_rows if row[1] in keep_facs]

    grant_item_ids = {g for g, _f, _c in pair_rows}
    fac_item_ids = {f for _g, f, _c in pair_rows}
    if not grant_item_ids or not fac_item_ids:
        raise RuntimeError(
            "Prefilter subset selection produced zero items. "
            "Try increasing --max-grant-specs/--max-fac-specs or prefilter per-aspect counts."
        )

    selected_grant_indices: Dict[str, Set[int]] = {}
    for grant_item_id in grant_item_ids:
        grant_id, spec_idx = _parse_grant_item_id(grant_item_id)
        if not grant_id or spec_idx < 0:
            continue
        selected_grant_indices.setdefault(grant_id, set()).add(int(spec_idx))

    selected_fac_keys: Set[Tuple[str, str, str]] = set()
    for fac_item_id in fac_item_ids:
        key = _parse_fac_item_id(fac_item_id)
        if all(key):
            selected_fac_keys.add(key)

    grant_db_obj = json.loads(grant_db_path.read_text(encoding="utf-8"))
    grants = grant_db_obj.get("grants") if isinstance(grant_db_obj, dict) else []
    if not isinstance(grants, list):
        grants = []

    grant_subset_rows: List[Dict[str, Any]] = []
    grant_kept_specs = 0
    for row in grants:
        if not isinstance(row, dict):
            continue
        grant_id = _normalize_ws(row.get("grant_id"))
        selected_idxs = selected_grant_indices.get(grant_id)
        if not selected_idxs:
            continue
        src_specs = row.get("grant_spec_keywords")
        specs = list(src_specs) if isinstance(src_specs, list) else []
        if not specs:
            continue
        out_specs: List[str] = []
        kept_local = 0
        for idx, txt in enumerate(specs):
            text = str(txt or "")
            if idx in selected_idxs and _normalize_ws(text):
                out_specs.append(text)
                kept_local += 1
            else:
                out_specs.append("")
        if kept_local <= 0:
            continue
        new_row = dict(row)
        new_row["grant_spec_keywords"] = out_specs
        grant_subset_rows.append(new_row)
        grant_kept_specs += int(kept_local)

    fac_db_obj = json.loads(fac_db_path.read_text(encoding="utf-8"))
    fac_rows = fac_db_obj.get("fac_specs") if isinstance(fac_db_obj, dict) else []
    if not isinstance(fac_rows, list):
        fac_rows = []

    fac_subset_rows: List[Dict[str, Any]] = []
    for row in fac_rows:
        if not isinstance(row, dict):
            continue
        key = (
            _normalize_ws(row.get("fac_id")),
            _normalize_ws(row.get("fac_spec_id")),
            _normalize_ws(row.get("fac_spec_idx")),
        )
        if key in selected_fac_keys:
            fac_subset_rows.append(dict(row))

    if not grant_subset_rows or not fac_subset_rows:
        raise RuntimeError(
            "Prefilter subset DB build produced empty grant/fac rows. "
            "Try increasing subset caps or prefilter per-aspect counts."
        )

    grant_out_obj = dict(grant_db_obj) if isinstance(grant_db_obj, dict) else {}
    grant_out_obj["grants"] = grant_subset_rows
    fac_out_obj = dict(fac_db_obj) if isinstance(fac_db_obj, dict) else {}
    fac_out_obj["fac_specs"] = fac_subset_rows
    _write_json(subset_grant_db_output, grant_out_obj)
    _write_json(subset_fac_db_output, fac_out_obj)

    return {
        "candidate_pairs": int(len(pairs)),
        "candidate_pairs_after_caps": int(len(pair_rows)),
        "grant_item_ids": int(len(grant_item_ids)),
        "fac_item_ids": int(len(fac_item_ids)),
        "grant_rows": int(len(grant_subset_rows)),
        "grant_specs_kept": int(grant_kept_specs),
        "fac_rows": int(len(fac_subset_rows)),
        "subset_grant_db": str(subset_grant_db_output),
        "subset_fac_db": str(subset_fac_db_output),
    }


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
    p.add_argument("--subset-mode", type=str, choices=("auto", "prefilter", "random"), default="auto")
    p.add_argument("--prefilter-cache", type=str, default=PREFILTER_CACHE_OUTPUT_DEFAULT)
    p.add_argument("--prefilter-high-per-aspect", type=int, default=4)
    p.add_argument("--prefilter-mid-per-aspect", type=int, default=4)
    p.add_argument("--prefilter-low-per-aspect", type=int, default=4)
    p.add_argument("--prefilter-high-threshold", type=float, default=PREFILTER_HIGH_THRESHOLD_DEFAULT)
    p.add_argument("--prefilter-low-threshold", type=float, default=PREFILTER_LOW_THRESHOLD_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--max-grant-specs", type=int, default=6)
    p.add_argument("--max-fac-specs", type=int, default=24)
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
    p.add_argument("--subset-grant-db-output", type=str, default=SUBSET_GRANT_DB_OUTPUT_DEFAULT)
    p.add_argument("--subset-fac-db-output", type=str, default=SUBSET_FAC_DB_OUTPUT_DEFAULT)
    p.add_argument(
        "--allow-non-test-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow writing outputs outside ce2/test/output (disabled by default for safety).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    grant_db_input = resolve_path(PROJECT_ROOT, args.grant_db)
    fac_db_input = resolve_path(PROJECT_ROOT, args.fac_db)
    decomposition_output = resolve_path(PROJECT_ROOT, args.decomposition_output)
    preview_output = resolve_path(PROJECT_ROOT, args.preview_output)
    prefilter_cache = resolve_path(PROJECT_ROOT, args.prefilter_cache)
    subset_grant_db_output = resolve_path(PROJECT_ROOT, args.subset_grant_db_output)
    subset_fac_db_output = resolve_path(PROJECT_ROOT, args.subset_fac_db_output)
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
    _assert_safe_output_path(
        subset_grant_db_output,
        safe_root=safe_root,
        allow_non_test_output=bool(args.allow_non_test_output),
    )
    _assert_safe_output_path(
        subset_fac_db_output,
        safe_root=safe_root,
        allow_non_test_output=bool(args.allow_non_test_output),
    )

    output_dir = decomposition_output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_mode, subset_reason = _resolve_subset_mode(
        requested=str(args.subset_mode),
        prefilter_cache=prefilter_cache,
    )

    run_grant_db = grant_db_input
    run_fac_db = fac_db_input
    run_max_grant_specs = max(0, int(args.max_grant_specs))
    run_max_fac_specs = max(0, int(args.max_fac_specs))
    prefilter_stats: Dict[str, Any] = {}
    if subset_mode == "prefilter":
        prefilter_stats = _build_prefilter_subset_dbs(
            grant_db_path=grant_db_input,
            fac_db_path=fac_db_input,
            subset_grant_db_output=subset_grant_db_output,
            subset_fac_db_output=subset_fac_db_output,
            prefilter_cache=prefilter_cache,
            seed=int(args.seed),
            max_grant_specs=max(0, int(args.max_grant_specs)),
            max_fac_specs=max(0, int(args.max_fac_specs)),
            high_per_aspect=max(0, int(args.prefilter_high_per_aspect)),
            mid_per_aspect=max(0, int(args.prefilter_mid_per_aspect)),
            low_per_aspect=max(0, int(args.prefilter_low_per_aspect)),
            high_threshold=float(args.prefilter_high_threshold),
            low_threshold=float(args.prefilter_low_threshold),
        )
        run_grant_db = subset_grant_db_output
        run_fac_db = subset_fac_db_output
        run_max_grant_specs = 0
        run_max_fac_specs = 0

    cmd = [
        str(args.python_bin),
        "ce2/data_preparation/decompose_aspect_specs.py",
        "--model-id",
        str(args.model_id),
        "--grant-db",
        str(run_grant_db),
        "--fac-db",
        str(run_fac_db),
        "--output-dir",
        str(output_dir),
        "--decomposition-output",
        str(decomposition_output),
        "--seed",
        str(int(args.seed)),
        "--max-grant-specs",
        str(int(run_max_grant_specs)),
        "--max-fac-specs",
        str(int(run_max_fac_specs)),
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

    print(
        f"subset_mode_resolved={subset_mode} reason={subset_reason} "
        f"max_grant_specs={run_max_grant_specs} max_fac_specs={run_max_fac_specs}"
    )
    if prefilter_stats:
        print(f"prefilter_subset_stats={json.dumps(prefilter_stats, ensure_ascii=False)}")
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
