from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "cross_encoder").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPORT_SCRIPT = PROJECT_ROOT / "cross_encoder" / "spec_to_chunk" / "data_preparation" / "export_db.py"
PREFILTER_SCRIPT = PROJECT_ROOT / "cross_encoder" / "spec_to_chunk" / "data_preparation" / "build_prefilter_cache.py"
LLM_DISTILL_SCRIPT = PROJECT_ROOT / "cross_encoder" / "spec_to_chunk" / "llm_distillation" / "llm_distillation.py"

GRANT_DB_DEFAULT = "cross_encoder/spec_to_chunk/dataset/grant_keywords_spec_keywords_db.json"
FAC_DB_DEFAULT = "cross_encoder/spec_to_chunk/dataset/fac_chunks_db.json"
PREFILTER_OUTPUT_DEFAULT = "cross_encoder/spec_to_chunk/dataset/spec_chunk_bge_cache.jsonl"
PREFILTER_MANIFEST_DEFAULT = "cross_encoder/spec_to_chunk/dataset/spec_chunk_bge_cache.manifest.json"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_path(value: Any) -> Path:
    path = Path(_clean_text(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _safe_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _safe_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _run_command(cmd: List[str], *, dry_run: bool) -> None:
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"$ {printable}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _read_json_obj(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _file_size_mb(path: Path) -> float:
    try:
        size = float(path.stat().st_size)
    except Exception:
        return 0.0
    return size / (1024.0 * 1024.0)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Prepare all prerequisite artifacts for spec_to_chunk llm_distillation: "
            "JSON DB export + BGE prefilter cache."
        )
    )

    p.add_argument("--grant-db", type=str, default=GRANT_DB_DEFAULT)
    p.add_argument("--fac-db", type=str, default=FAC_DB_DEFAULT)
    p.add_argument("--prefilter-output", type=str, default=PREFILTER_OUTPUT_DEFAULT)
    p.add_argument("--prefilter-manifest", type=str, default=PREFILTER_MANIFEST_DEFAULT)

    p.add_argument("--skip-export", action="store_true", help="Skip export_db step.")
    p.add_argument("--skip-prefilter", action="store_true", help="Skip build_prefilter_cache step.")
    p.add_argument("--dry-run", action="store_true", help="Print commands only.")

    p.add_argument("--limit-grants", type=int, default=0)
    p.add_argument("--limit-faculties", type=int, default=0)
    p.add_argument("--min-spec-weight", type=float, default=0.0)
    p.add_argument("--batch-faculty-size", type=int, default=200)
    p.add_argument("--max-chars-per-chunk", type=int, default=6000)
    p.add_argument("--indent", type=int, default=2)

    p.add_argument("--prefilter-model-id", type=str, default="")
    p.add_argument("--prefilter-top-k-per-spec", type=int, default=0)
    p.add_argument("--prefilter-method", type=str, default="bge")
    p.add_argument("--prefilter-bge-model-id", type=str, default="BAAI/bge-reranker-base")
    p.add_argument("--prefilter-bge-batch-size", type=int, default=64)
    p.add_argument("--prefilter-bge-max-length", type=int, default=512)

    p.add_argument("--suggest-prefilter-top-k", type=int, default=40)
    p.add_argument("--suggest-prefilter-mid-k", type=int, default=12)
    p.add_argument("--suggest-prefilter-rand-k", type=int, default=12)
    p.add_argument("--suggest-batch-size", type=int, default=64)
    return p


def main() -> int:
    args = _build_parser().parse_args()

    grant_db = _resolve_path(args.grant_db)
    fac_db = _resolve_path(args.fac_db)
    prefilter_output = _resolve_path(args.prefilter_output)
    prefilter_manifest = _resolve_path(args.prefilter_manifest)

    limit_grants = _safe_int(args.limit_grants, default=0, minimum=0, maximum=5_000_000)
    limit_faculties = _safe_int(args.limit_faculties, default=0, minimum=0, maximum=5_000_000)
    min_spec_weight = _safe_float(args.min_spec_weight, default=0.0, minimum=0.0, maximum=1.0)
    batch_faculty_size = _safe_int(args.batch_faculty_size, default=200, minimum=1, maximum=10_000)
    max_chars_per_chunk = _safe_int(args.max_chars_per_chunk, default=6000, minimum=0, maximum=500_000)
    indent = _safe_int(args.indent, default=2, minimum=0, maximum=16)
    prefilter_top_k_per_spec = _safe_int(args.prefilter_top_k_per_spec, default=0, minimum=0, maximum=2_000_000)
    prefilter_method = _clean_text(args.prefilter_method) or "bge"
    prefilter_bge_model_id = _clean_text(args.prefilter_bge_model_id) or "BAAI/bge-reranker-base"
    prefilter_bge_batch_size = _safe_int(args.prefilter_bge_batch_size, default=64, minimum=1, maximum=4096)
    prefilter_bge_max_length = _safe_int(args.prefilter_bge_max_length, default=512, minimum=32, maximum=4096)

    suggest_top_k = _safe_int(args.suggest_prefilter_top_k, default=40, minimum=0, maximum=200_000)
    suggest_mid_k = _safe_int(args.suggest_prefilter_mid_k, default=12, minimum=0, maximum=200_000)
    suggest_rand_k = _safe_int(args.suggest_prefilter_rand_k, default=12, minimum=0, maximum=200_000)
    suggest_batch_size = _safe_int(args.suggest_batch_size, default=64, minimum=1, maximum=4096)

    dry_run = bool(args.dry_run)
    skip_export = bool(args.skip_export)
    skip_prefilter = bool(args.skip_prefilter)

    started_at = time.time()

    if not skip_export:
        grant_db.parent.mkdir(parents=True, exist_ok=True)
        fac_db.parent.mkdir(parents=True, exist_ok=True)
        export_cmd = [
            sys.executable,
            str(EXPORT_SCRIPT),
            "--grant-output",
            str(grant_db),
            "--fac-output",
            str(fac_db),
            "--limit-grants",
            str(limit_grants),
            "--limit-faculties",
            str(limit_faculties),
            "--min-spec-weight",
            str(min_spec_weight),
            "--batch-faculty-size",
            str(batch_faculty_size),
            "--max-chars-per-chunk",
            str(max_chars_per_chunk),
            "--indent",
            str(indent),
        ]
        _run_command(export_cmd, dry_run=dry_run)

    if not skip_prefilter:
        prefilter_output.parent.mkdir(parents=True, exist_ok=True)
        prefilter_manifest.parent.mkdir(parents=True, exist_ok=True)
        prefilter_cmd = [
            sys.executable,
            str(PREFILTER_SCRIPT),
            "--grant-db",
            str(grant_db),
            "--fac-db",
            str(fac_db),
            "--output",
            str(prefilter_output),
            "--manifest",
            str(prefilter_manifest),
            "--top-k-per-spec",
            str(prefilter_top_k_per_spec),
            "--prefilter-method",
            str(prefilter_method),
            "--bge-model-id",
            str(prefilter_bge_model_id),
            "--bge-batch-size",
            str(prefilter_bge_batch_size),
            "--bge-max-length",
            str(prefilter_bge_max_length),
        ]
        model_id = _clean_text(args.prefilter_model_id)
        if model_id:
            prefilter_cmd.extend(["--model-id", model_id])
        _run_command(prefilter_cmd, dry_run=dry_run)

    elapsed = time.time() - started_at

    print("")
    print("Preparation summary")
    print(f"project_root={PROJECT_ROOT}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"grant_db={grant_db} exists={grant_db.exists()}")
    print(f"fac_db={fac_db} exists={fac_db.exists()}")
    print(f"prefilter_output={prefilter_output} exists={prefilter_output.exists()}")
    print(f"prefilter_manifest={prefilter_manifest} exists={prefilter_manifest.exists()}")

    if grant_db.exists():
        grant_obj = _read_json_obj(grant_db)
        grant_meta = dict(grant_obj.get("meta") or {})
        print(f"grant_db_size_mb={_file_size_mb(grant_db):.2f}")
        print(f"grant_count={grant_meta.get('grant_count')}")
        print(f"grant_spec_keyword_count={grant_meta.get('grant_spec_keyword_count')}")

    if fac_db.exists():
        fac_obj = _read_json_obj(fac_db)
        fac_meta = dict(fac_obj.get("meta") or {})
        print(f"fac_db_size_mb={_file_size_mb(fac_db):.2f}")
        print(f"fac_chunk_count={fac_meta.get('fac_chunk_count')}")

    if prefilter_manifest.exists():
        manifest_obj = _read_json_obj(prefilter_manifest)
        print(f"prefilter_manifest_size_mb={_file_size_mb(prefilter_manifest):.2f}")
        print(f"prefilter_written_spec_count={manifest_obj.get('written_spec_count')}")
        print(f"prefilter_total_written_candidates={manifest_obj.get('total_written_candidates')}")

    llm_cmd = [
        sys.executable,
        str(LLM_DISTILL_SCRIPT),
        "--grant-db",
        str(grant_db),
        "--fac-db",
        str(fac_db),
        "--prefilter-score-cache",
        str(prefilter_output),
        "--prefilter-top-k",
        str(suggest_top_k),
        "--prefilter-mid-k",
        str(suggest_mid_k),
        "--prefilter-rand-k",
        str(suggest_rand_k),
        "--batch-size",
        str(suggest_batch_size),
    ]
    print("")
    print("Suggested next command (llm_distillation):")
    print(" ".join(shlex.quote(x) for x in llm_cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
