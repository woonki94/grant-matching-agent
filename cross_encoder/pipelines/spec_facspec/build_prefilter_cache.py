from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import bindparam, text

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from db.db_conn import SessionLocal


GRANT_DB_DEFAULT = "cross_encoder/dataset/spec_facspec/grant_keywords_spec_keywords_db.json"
FAC_DB_DEFAULT = "cross_encoder/dataset/spec_facspec/fac_specs_db.json"
OUTPUT_DEFAULT = "cross_encoder/dataset/spec_facspec/spec_facspec_cosine_cache.jsonl"
MANIFEST_DEFAULT = "cross_encoder/dataset/spec_facspec/spec_facspec_cosine_cache.manifest.json"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


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


def _normalize_text(text: Any) -> str:
    return " ".join(_clean_text(text).lower().split())


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON {path}: {type(e).__name__}: {e}") from e
    if not isinstance(obj, dict):
        raise RuntimeError(f"JSON root must be object: {path}")
    return obj


def _to_vec(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None

    try:
        if isinstance(value, np.ndarray):
            arr = value.astype(np.float32)
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value, dtype=np.float32)
        elif hasattr(value, "tolist"):
            arr = np.asarray(value.tolist(), dtype=np.float32)
        elif isinstance(value, str):
            s = value.strip()
            if s.startswith("[") and s.endswith("]"):
                arr = np.asarray(json.loads(s), dtype=np.float32)
            else:
                return None
        else:
            arr = np.asarray(list(value), dtype=np.float32)
    except Exception:
        return None

    if arr.ndim != 1 or arr.size <= 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _flatten_specs(grant_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for grant in list(grant_payload.get("grants") or []):
        if not isinstance(grant, dict):
            continue
        grant_id = _clean_text(grant.get("grant_id"))
        if not grant_id:
            continue
        for idx, spec in enumerate(list(grant.get("grant_spec_keywords") or [])):
            spec_text = _clean_text(spec)
            if not spec_text:
                continue
            out.append(
                {
                    "grant_id": grant_id,
                    "spec_idx": int(idx),
                    "spec_text": spec_text,
                    "spec_norm": _normalize_text(spec_text),
                }
            )
    return out


def _flatten_fac_specs(fac_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for fac_spec in list(fac_payload.get("fac_specs") or []):
        if not isinstance(fac_spec, dict):
            continue
        fac_id = _safe_int(fac_spec.get("fac_id"), default=0, minimum=0, maximum=2_147_483_647)
        fac_spec_id = _safe_int(fac_spec.get("fac_spec_id"), default=0, minimum=0, maximum=9_223_372_036_854_775_807)
        fac_spec_idx = _safe_int(fac_spec.get("fac_spec_idx"), default=0, minimum=0, maximum=50_000_000)
        section = _clean_text(fac_spec.get("section")) or "unknown"
        spec_text = _clean_text(fac_spec.get("text"))
        if fac_id <= 0 or fac_spec_id <= 0 or not spec_text:
            continue
        out.append(
            {
                "fac_id": fac_id,
                "fac_spec_id": fac_spec_id,
                "fac_spec_idx": fac_spec_idx,
                "section": section,
                "fac_spec_text": spec_text,
            }
        )
    return out


def _load_grant_spec_vec_map(
    *,
    specs: List[Dict[str, Any]],
    model_id: str,
) -> Dict[Tuple[str, str], np.ndarray]:
    if not specs:
        return {}

    grant_ids = sorted({_clean_text(x.get("grant_id")) for x in specs if _clean_text(x.get("grant_id"))})
    spec_norms = sorted({_normalize_text(x.get("spec_text")) for x in specs if _clean_text(x.get("spec_text"))})

    sql_text = """
        SELECT
            ose.opportunity_id AS grant_id,
            COALESCE(ose.spec_text, '') AS spec_text,
            COALESCE(ose.spec_weight, 1.0) AS spec_weight,
            ose.id AS spec_row_id,
            ose.spec_vec AS spec_vec
        FROM opportunity_specialization_embedding ose
        WHERE ose.opportunity_id IN :grant_ids
          AND COALESCE(ose.spec_text, '') <> ''
    """
    if _clean_text(model_id):
        sql_text += "\n  AND COALESCE(ose.model, '') = :model_id"
    sql_text += """
        ORDER BY
            ose.opportunity_id ASC,
            COALESCE(ose.spec_weight, 1.0) DESC,
            ose.id ASC
    """

    sql = text(sql_text).bindparams(bindparam("grant_ids", expanding=True))
    params: Dict[str, Any] = {"grant_ids": grant_ids}
    if _clean_text(model_id):
        params["model_id"] = _clean_text(model_id)

    with SessionLocal() as sess:
        rows = sess.execute(sql, params).mappings().all()

    out: Dict[Tuple[str, str], np.ndarray] = {}
    for row in rows:
        grant_id = _clean_text(row.get("grant_id"))
        spec_norm = _normalize_text(row.get("spec_text"))
        if not grant_id or not spec_norm or spec_norm not in spec_norms:
            continue
        key = (grant_id, spec_norm)
        if key in out:
            continue
        vec = _to_vec(row.get("spec_vec"))
        if vec is None:
            continue
        out[key] = vec
    return out


def _load_fac_spec_vec_map(
    *,
    fac_specs: List[Dict[str, Any]],
    model_id: str,
) -> Dict[int, np.ndarray]:
    if not fac_specs:
        return {}

    fac_spec_ids = sorted({int(x["fac_spec_id"]) for x in fac_specs if int(x.get("fac_spec_id") or 0) > 0})
    if not fac_spec_ids:
        return {}

    sql_text = """
        SELECT
            fse.id AS fac_spec_id,
            fse.spec_vec AS spec_vec
        FROM faculty_specialization_embedding fse
        WHERE fse.id IN :fac_spec_ids
          AND fse.spec_vec IS NOT NULL
    """
    if _clean_text(model_id):
        sql_text += "\n  AND COALESCE(fse.model, '') = :model_id"

    sql = text(sql_text).bindparams(bindparam("fac_spec_ids", expanding=True))
    params: Dict[str, Any] = {"fac_spec_ids": fac_spec_ids}
    if _clean_text(model_id):
        params["model_id"] = _clean_text(model_id)

    with SessionLocal() as sess:
        rows = sess.execute(sql, params).mappings().all()

    out: Dict[int, np.ndarray] = {}
    for row in rows:
        fac_spec_id = _safe_int(row.get("fac_spec_id"), default=0, minimum=0, maximum=9_223_372_036_854_775_807)
        if fac_spec_id <= 0 or fac_spec_id in out:
            continue
        vec = _to_vec(row.get("spec_vec"))
        if vec is None:
            continue
        out[fac_spec_id] = vec
    return out


def _try_tqdm(total: int) -> Any:
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, desc="Spec facspec cosine cache", unit="spec", dynamic_ncols=True)
    except Exception:
        return None


def build_cache(
    *,
    grant_db_path: Path,
    fac_db_path: Path,
    output_path: Path,
    manifest_path: Path,
    model_id: str,
    top_k_per_spec: int,
) -> Dict[str, Any]:
    grant_payload = _load_json(grant_db_path)
    fac_payload = _load_json(fac_db_path)

    specs = _flatten_specs(grant_payload)
    fac_specs_raw = _flatten_fac_specs(fac_payload)
    if not specs:
        raise RuntimeError("No specs found in grant DB JSON.")
    if not fac_specs_raw:
        raise RuntimeError("No fac specs found in faculty DB JSON.")

    grant_spec_vec_map = _load_grant_spec_vec_map(specs=specs, model_id=model_id)
    fac_spec_vec_map = _load_fac_spec_vec_map(fac_specs=fac_specs_raw, model_id=model_id)

    spec_rows: List[Dict[str, Any]] = []
    missing_spec_vec_count = 0
    dims_counter_spec: Dict[int, int] = {}
    for spec in specs:
        key = (_clean_text(spec.get("grant_id")), _normalize_text(spec.get("spec_text")))
        vec = grant_spec_vec_map.get(key)
        if vec is None:
            missing_spec_vec_count += 1
            continue
        dims_counter_spec[len(vec)] = int(dims_counter_spec.get(len(vec), 0) + 1)
        spec_rows.append({**spec, "spec_vec": vec})

    fac_spec_rows: List[Dict[str, Any]] = []
    missing_fac_spec_vec_count = 0
    dims_counter_fac_spec: Dict[int, int] = {}
    for fac_spec in fac_specs_raw:
        fac_spec_id = int(fac_spec["fac_spec_id"])
        vec = fac_spec_vec_map.get(fac_spec_id)
        if vec is None:
            missing_fac_spec_vec_count += 1
            continue
        dims_counter_fac_spec[len(vec)] = int(dims_counter_fac_spec.get(len(vec), 0) + 1)
        fac_spec_rows.append({**fac_spec, "fac_spec_vec": vec})

    if not spec_rows:
        raise RuntimeError("No grant specs with vectors after DB alignment.")
    if not fac_spec_rows:
        raise RuntimeError("No faculty specs with vectors after DB alignment.")

    common_dims = sorted(set(dims_counter_spec.keys()) & set(dims_counter_fac_spec.keys()))
    if not common_dims:
        raise RuntimeError(
            f"No shared embedding dimension. spec_dims={sorted(dims_counter_spec.keys())}, "
            f"fac_spec_dims={sorted(dims_counter_fac_spec.keys())}"
        )

    dim = int(common_dims[-1])
    spec_rows = [x for x in spec_rows if int(len(x["spec_vec"])) == dim]
    fac_spec_rows = [x for x in fac_spec_rows if int(len(x["fac_spec_vec"])) == dim]
    if not spec_rows or not fac_spec_rows:
        raise RuntimeError("No rows left after filtering to common embedding dimension.")

    fac_matrix = np.asarray([x["fac_spec_vec"] for x in fac_spec_rows], dtype=np.float32)
    fac_norms = np.linalg.norm(fac_matrix, axis=1, keepdims=True)
    fac_matrix = fac_matrix / np.clip(fac_norms, a_min=1e-12, a_max=None)

    safe_top_k = _safe_int(top_k_per_spec, default=0, minimum=0, maximum=2_000_000)
    if safe_top_k > 0:
        keep_k = min(safe_top_k, len(fac_spec_rows))
    else:
        keep_k = len(fac_spec_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    bar = _try_tqdm(total=len(spec_rows))
    spec_count_written = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for spec in spec_rows:
            vec = np.asarray(spec["spec_vec"], dtype=np.float32)
            vec = vec / max(1e-12, float(np.linalg.norm(vec)))
            sims = fac_matrix @ vec
            order = np.argsort(-sims)[:keep_k]

            candidates = []
            for idx in order.tolist():
                fac_spec = fac_spec_rows[int(idx)]
                candidates.append(
                    {
                        "fac_id": int(fac_spec["fac_id"]),
                        "fac_spec_id": int(fac_spec["fac_spec_id"]),
                        "fac_spec_idx": int(fac_spec["fac_spec_idx"]),
                        "section": _clean_text(fac_spec["section"]),
                        "cosine_score": float(sims[int(idx)]),
                    }
                )

            row = {
                "grant_id": _clean_text(spec["grant_id"]),
                "spec_idx": int(spec["spec_idx"]),
                "spec_text": _clean_text(spec["spec_text"]),
                "candidates": candidates,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            spec_count_written += 1
            if bar is not None:
                bar.update(1)

    if bar is not None:
        bar.close()

    elapsed = max(1e-6, time.time() - started)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "db_vectors",
        "grant_db_path": str(grant_db_path),
        "fac_db_path": str(fac_db_path),
        "output_path": str(output_path),
        "model_filter": _clean_text(model_id),
        "top_k_per_spec": int(safe_top_k),
        "written_spec_count": int(spec_count_written),
        "candidate_count_per_spec": int(keep_k),
        "total_written_candidates": int(spec_count_written * keep_k),
        "input_spec_count": int(len(specs)),
        "input_fac_spec_count": int(len(fac_specs_raw)),
        "aligned_spec_count": int(len(spec_rows)),
        "aligned_fac_spec_count": int(len(fac_spec_rows)),
        "missing_spec_vector_count": int(missing_spec_vec_count),
        "missing_fac_spec_vector_count": int(missing_fac_spec_vec_count),
        "spec_dim_counts": dims_counter_spec,
        "fac_spec_dim_counts": dims_counter_fac_spec,
        "common_dim": int(dim),
        "elapsed_seconds": float(elapsed),
        "specs_per_second": float(spec_count_written / elapsed),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build reusable spec x faculty-spec cosine score cache using DB embeddings "
            "(opportunity_specialization_embedding.spec_vec and faculty_specialization_embedding.spec_vec)."
        )
    )
    p.add_argument("--grant-db", type=str, default=GRANT_DB_DEFAULT, help="Grant JSON DB path.")
    p.add_argument("--fac-db", type=str, default=FAC_DB_DEFAULT, help="Faculty spec JSON DB path.")
    p.add_argument("--output", type=str, default=OUTPUT_DEFAULT, help="Output JSONL cache path.")
    p.add_argument("--manifest", type=str, default=MANIFEST_DEFAULT, help="Output manifest JSON path.")
    p.add_argument(
        "--model-id",
        type=str,
        default="",
        help="Optional embedding model filter for both specialization embedding tables.",
    )
    p.add_argument(
        "--top-k-per-spec",
        type=int,
        default=0,
        help="Keep only top-k candidates per spec in cache (0 = all aligned faculty specs).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    grant_db = Path(_clean_text(args.grant_db)).expanduser().resolve()
    fac_db = Path(_clean_text(args.fac_db)).expanduser().resolve()
    output = Path(_clean_text(args.output)).expanduser().resolve()
    manifest = Path(_clean_text(args.manifest)).expanduser().resolve()

    if not grant_db.exists():
        raise RuntimeError(f"Grant DB JSON not found: {grant_db}")
    if not fac_db.exists():
        raise RuntimeError(f"Faculty DB JSON not found: {fac_db}")

    result = build_cache(
        grant_db_path=grant_db,
        fac_db_path=fac_db,
        output_path=output,
        manifest_path=manifest,
        model_id=_clean_text(args.model_id),
        top_k_per_spec=_safe_int(args.top_k_per_spec, default=0, minimum=0, maximum=2_000_000),
    )

    print(f"output={output}")
    print(f"manifest={manifest}")
    print(f"written_spec_count={result.get('written_spec_count')}")
    print(f"candidate_count_per_spec={result.get('candidate_count_per_spec')}")
    print(f"total_written_candidates={result.get('total_written_candidates')}")
    print(f"aligned_spec_count={result.get('aligned_spec_count')}")
    print(f"aligned_fac_spec_count={result.get('aligned_fac_spec_count')}")
    print(f"common_dim={result.get('common_dim')}")
    print(f"elapsed_seconds={result.get('elapsed_seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
