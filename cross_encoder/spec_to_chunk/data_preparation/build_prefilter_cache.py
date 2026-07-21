from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sqlalchemy import bindparam, text
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

from db.db_conn import SessionLocal


GRANT_DB_DEFAULT = "cross_encoder/spec_to_chunk/dataset/grant_keywords_spec_keywords_db.json"
FAC_DB_DEFAULT = "cross_encoder/spec_to_chunk/dataset/fac_chunks_db.json"
OUTPUT_DEFAULT = "cross_encoder/spec_to_chunk/dataset/spec_chunk_bge_cache.jsonl"
MANIFEST_DEFAULT = "cross_encoder/spec_to_chunk/dataset/spec_chunk_bge_cache.manifest.json"
PREFILTER_METHOD_DEFAULT = "bge"
BGE_MODEL_ID_DEFAULT = "BAAI/bge-reranker-base"


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


def _flatten_chunks(fac_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for chunk in list(fac_payload.get("fac_chunks") or []):
        if not isinstance(chunk, dict):
            continue
        chunk_text = _clean_text(chunk.get("text"))
        if not chunk_text:
            continue
        fac_id = _safe_int(chunk.get("fac_id"), default=0, minimum=0, maximum=2_147_483_647)
        chunk_id = _safe_int(chunk.get("chunk_id"), default=0, minimum=0, maximum=2_147_483_647)
        chunk_index = _safe_int(chunk.get("chunk_index"), default=0, minimum=0, maximum=1_000_000)
        source_type = _clean_text(chunk.get("source_type")) or "unknown"
        if fac_id <= 0 or chunk_id <= 0:
            continue
        out.append(
            {
                "fac_id": fac_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "source_type": source_type,
                "chunk_text": chunk_text,
            }
        )
    return out


def _load_spec_vec_map(
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


def _load_chunk_vec_map(
    *,
    chunks: List[Dict[str, Any]],
) -> Dict[Tuple[int, str, int, int], np.ndarray]:
    additional_ids = sorted(
        {
            int(c["chunk_id"])
            for c in chunks
            if _clean_text(c.get("source_type")) == "additional_info"
        }
    )
    publication_ids = sorted(
        {
            int(c["chunk_id"])
            for c in chunks
            if _clean_text(c.get("source_type")) == "publication_abstract"
        }
    )

    out: Dict[Tuple[int, str, int, int], np.ndarray] = {}

    if additional_ids:
        sql = text(
            """
            SELECT
                fai.id AS chunk_id,
                fai.faculty_id AS fac_id,
                COALESCE(fai.chunk_index, 0) AS chunk_index,
                fai.content_embedding AS chunk_vec
            FROM faculty_additional_info fai
            WHERE fai.id IN :chunk_ids
              AND fai.content_embedding IS NOT NULL
            """
        ).bindparams(bindparam("chunk_ids", expanding=True))
        with SessionLocal() as sess:
            rows = sess.execute(sql, {"chunk_ids": additional_ids}).mappings().all()

        for row in rows:
            fac_id = _safe_int(row.get("fac_id"), default=0, minimum=0, maximum=2_147_483_647)
            chunk_id = _safe_int(row.get("chunk_id"), default=0, minimum=0, maximum=2_147_483_647)
            chunk_index = _safe_int(row.get("chunk_index"), default=0, minimum=0, maximum=1_000_000)
            vec = _to_vec(row.get("chunk_vec"))
            if fac_id <= 0 or chunk_id <= 0 or vec is None:
                continue
            out[(fac_id, "additional_info", chunk_id, chunk_index)] = vec

    if publication_ids:
        sql = text(
            """
            SELECT
                fp.id AS chunk_id,
                fp.faculty_id AS fac_id,
                fp.abstract_embedding AS chunk_vec
            FROM faculty_publication fp
            WHERE fp.id IN :chunk_ids
              AND fp.abstract_embedding IS NOT NULL
            """
        ).bindparams(bindparam("chunk_ids", expanding=True))
        with SessionLocal() as sess:
            rows = sess.execute(sql, {"chunk_ids": publication_ids}).mappings().all()

        for row in rows:
            fac_id = _safe_int(row.get("fac_id"), default=0, minimum=0, maximum=2_147_483_647)
            chunk_id = _safe_int(row.get("chunk_id"), default=0, minimum=0, maximum=2_147_483_647)
            vec = _to_vec(row.get("chunk_vec"))
            if fac_id <= 0 or chunk_id <= 0 or vec is None:
                continue
            out[(fac_id, "publication_abstract", chunk_id, 0)] = vec

    return out


def _try_tqdm(total: int) -> Any:
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, desc="Spec prefilter cache", unit="spec", dynamic_ncols=True)
    except Exception:
        return None


def _build_cache_cosine(
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
    chunks_raw = _flatten_chunks(fac_payload)
    if not specs:
        raise RuntimeError("No specs found in grant DB JSON.")
    if not chunks_raw:
        raise RuntimeError("No chunks found in fac DB JSON.")

    spec_vec_map = _load_spec_vec_map(specs=specs, model_id=model_id)
    chunk_vec_map = _load_chunk_vec_map(chunks=chunks_raw)

    # Align to rows with vectors in both spaces.
    spec_rows: List[Dict[str, Any]] = []
    missing_spec_vec_count = 0
    dims_counter_spec: Dict[int, int] = {}
    for spec in specs:
        key = (_clean_text(spec.get("grant_id")), _normalize_text(spec.get("spec_text")))
        vec = spec_vec_map.get(key)
        if vec is None:
            missing_spec_vec_count += 1
            continue
        dims_counter_spec[len(vec)] = int(dims_counter_spec.get(len(vec), 0) + 1)
        spec_rows.append({**spec, "spec_vec": vec})

    chunk_rows: List[Dict[str, Any]] = []
    missing_chunk_vec_count = 0
    dims_counter_chunk: Dict[int, int] = {}
    for chunk in chunks_raw:
        key = (
            int(chunk["fac_id"]),
            _clean_text(chunk["source_type"]),
            int(chunk["chunk_id"]),
            int(chunk["chunk_index"]),
        )
        vec = chunk_vec_map.get(key)
        if vec is None:
            missing_chunk_vec_count += 1
            continue
        dims_counter_chunk[len(vec)] = int(dims_counter_chunk.get(len(vec), 0) + 1)
        chunk_rows.append({**chunk, "chunk_vec": vec})

    if not spec_rows:
        raise RuntimeError("No specs with vectors after DB alignment.")
    if not chunk_rows:
        raise RuntimeError("No chunks with vectors after DB alignment.")

    # Pick common dim present in both tables to avoid mismatch rows.
    common_dims = sorted(set(dims_counter_spec.keys()) & set(dims_counter_chunk.keys()))
    if not common_dims:
        raise RuntimeError(
            f"No shared embedding dimension. spec_dims={sorted(dims_counter_spec.keys())}, "
            f"chunk_dims={sorted(dims_counter_chunk.keys())}"
        )
    dim = int(common_dims[-1])
    spec_rows = [x for x in spec_rows if int(len(x["spec_vec"])) == dim]
    chunk_rows = [x for x in chunk_rows if int(len(x["chunk_vec"])) == dim]
    if not spec_rows or not chunk_rows:
        raise RuntimeError("No rows left after filtering to common embedding dimension.")

    # Build normalized chunk matrix for fast cosine via dot-product.
    chunk_matrix = np.asarray([x["chunk_vec"] for x in chunk_rows], dtype=np.float32)
    chunk_norms = np.linalg.norm(chunk_matrix, axis=1, keepdims=True)
    chunk_matrix = chunk_matrix / np.clip(chunk_norms, a_min=1e-12, a_max=None)

    safe_top_k = _safe_int(top_k_per_spec, default=0, minimum=0, maximum=2_000_000)
    if safe_top_k > 0:
        keep_k = min(safe_top_k, len(chunk_rows))
    else:
        keep_k = len(chunk_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    bar = _try_tqdm(total=len(spec_rows))
    spec_count_written = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for spec in spec_rows:
            vec = np.asarray(spec["spec_vec"], dtype=np.float32)
            vec = vec / max(1e-12, float(np.linalg.norm(vec)))
            sims = chunk_matrix @ vec  # cosine since normalized
            order = np.argsort(-sims)[:keep_k]

            candidates = []
            for idx in order.tolist():
                chunk = chunk_rows[int(idx)]
                candidates.append(
                    {
                        "fac_id": int(chunk["fac_id"]),
                        "source_type": _clean_text(chunk["source_type"]),
                        "chunk_id": int(chunk["chunk_id"]),
                        "chunk_index": int(chunk["chunk_index"]),
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
        "prefilter_method": "cosine",
        "grant_db_path": str(grant_db_path),
        "fac_db_path": str(fac_db_path),
        "output_path": str(output_path),
        "model_filter": _clean_text(model_id),
        "top_k_per_spec": int(safe_top_k),
        "written_spec_count": int(spec_count_written),
        "candidate_count_per_spec": int(keep_k),
        "total_written_candidates": int(spec_count_written * keep_k),
        "input_spec_count": int(len(specs)),
        "input_chunk_count": int(len(chunks_raw)),
        "aligned_spec_count": int(len(spec_rows)),
        "aligned_chunk_count": int(len(chunk_rows)),
        "missing_spec_vector_count": int(missing_spec_vec_count),
        "missing_chunk_vector_count": int(missing_chunk_vec_count),
        "spec_dim_counts": dims_counter_spec,
        "chunk_dim_counts": dims_counter_chunk,
        "common_dim": int(dim),
        "elapsed_seconds": float(elapsed),
        "specs_per_second": float(spec_count_written / elapsed),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _score_docs_with_bge_ce(
    *,
    model: Any,
    tokenizer: Any,
    query_text: str,
    docs: List[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> List[float]:
    scores: List[float] = []
    step = max(1, int(batch_size))
    with torch.no_grad():
        for i in range(0, len(docs), step):
            docs_batch = docs[i : i + step]
            q_batch = [query_text] * len(docs_batch)
            enc = tokenizer(
                q_batch,
                docs_batch,
                max_length=int(max_length),
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = _to_device(enc, device)
            logits = model(**enc).logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            scores.extend(float(x) for x in probs.detach().cpu().tolist())
    return scores


def _build_cache_bge(
    *,
    grant_db_path: Path,
    fac_db_path: Path,
    output_path: Path,
    manifest_path: Path,
    bge_model_id: str,
    top_k_per_spec: int,
    batch_size: int,
    max_length: int,
) -> Dict[str, Any]:
    grant_payload = _load_json(grant_db_path)
    fac_payload = _load_json(fac_db_path)

    specs = _flatten_specs(grant_payload)
    chunks = _flatten_chunks(fac_payload)
    if not specs:
        raise RuntimeError("No specs found in grant DB JSON.")
    if not chunks:
        raise RuntimeError("No chunks found in fac DB JSON.")

    model_ref = _clean_text(bge_model_id) or BGE_MODEL_ID_DEFAULT
    device = _pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    model = AutoModelForSequenceClassification.from_pretrained(model_ref, num_labels=1)
    model.to(device)
    model.eval()

    safe_top_k = _safe_int(top_k_per_spec, default=0, minimum=0, maximum=2_000_000)
    if safe_top_k > 0:
        keep_k = min(safe_top_k, len(chunks))
    else:
        keep_k = len(chunks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    doc_texts = [_clean_text(x.get("chunk_text")) for x in chunks]
    started = time.time()
    bar = _try_tqdm(total=len(specs))
    spec_count_written = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for spec in specs:
            query_text = _clean_text(spec.get("spec_text"))
            if not query_text:
                continue
            scores = _score_docs_with_bge_ce(
                model=model,
                tokenizer=tokenizer,
                query_text=query_text,
                docs=doc_texts,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
            )
            order = np.argsort(-np.asarray(scores, dtype=np.float32))[:keep_k]

            candidates: List[Dict[str, Any]] = []
            for idx in order.tolist():
                chunk = chunks[int(idx)]
                candidates.append(
                    {
                        "fac_id": int(chunk["fac_id"]),
                        "source_type": _clean_text(chunk["source_type"]),
                        "chunk_id": int(chunk["chunk_id"]),
                        "chunk_index": int(chunk["chunk_index"]),
                        "bge_score": float(scores[int(idx)]),
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
        "source": "bge_cross_encoder",
        "prefilter_method": "bge",
        "grant_db_path": str(grant_db_path),
        "fac_db_path": str(fac_db_path),
        "output_path": str(output_path),
        "bge_model_id": model_ref,
        "top_k_per_spec": int(safe_top_k),
        "written_spec_count": int(spec_count_written),
        "candidate_count_per_spec": int(keep_k),
        "total_written_candidates": int(spec_count_written * keep_k),
        "input_spec_count": int(len(specs)),
        "input_chunk_count": int(len(chunks)),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "device": str(device),
        "elapsed_seconds": float(elapsed),
        "specs_per_second": float(spec_count_written / elapsed),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_cache(
    *,
    grant_db_path: Path,
    fac_db_path: Path,
    output_path: Path,
    manifest_path: Path,
    model_id: str,
    top_k_per_spec: int,
    prefilter_method: str,
    bge_model_id: str,
    bge_batch_size: int,
    bge_max_length: int,
) -> Dict[str, Any]:
    method = (_clean_text(prefilter_method) or PREFILTER_METHOD_DEFAULT).lower()
    if method in {"bge", "bge_ce", "ce"}:
        return _build_cache_bge(
            grant_db_path=grant_db_path,
            fac_db_path=fac_db_path,
            output_path=output_path,
            manifest_path=manifest_path,
            bge_model_id=bge_model_id,
            top_k_per_spec=top_k_per_spec,
            batch_size=bge_batch_size,
            max_length=bge_max_length,
        )
    if method == "cosine":
        return _build_cache_cosine(
            grant_db_path=grant_db_path,
            fac_db_path=fac_db_path,
            output_path=output_path,
            manifest_path=manifest_path,
            model_id=model_id,
            top_k_per_spec=top_k_per_spec,
        )
    raise RuntimeError(f"Unsupported prefilter method: {prefilter_method}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build reusable spec x faculty-chunk prefilter score cache. "
            "Default method is plain BGE cross-encoder reranker scoring."
        )
    )
    p.add_argument("--grant-db", type=str, default=GRANT_DB_DEFAULT, help="Grant JSON DB path.")
    p.add_argument("--fac-db", type=str, default=FAC_DB_DEFAULT, help="Faculty chunk JSON DB path.")
    p.add_argument("--output", type=str, default=OUTPUT_DEFAULT, help="Output JSONL cache path.")
    p.add_argument("--manifest", type=str, default=MANIFEST_DEFAULT, help="Output manifest JSON path.")
    p.add_argument(
        "--prefilter-method",
        type=str,
        default=PREFILTER_METHOD_DEFAULT,
        help="Prefilter method: bge (default) or cosine.",
    )
    p.add_argument(
        "--bge-model-id",
        type=str,
        default=BGE_MODEL_ID_DEFAULT,
        help="BGE cross-encoder model id/path when prefilter-method=bge.",
    )
    p.add_argument("--bge-batch-size", type=int, default=64, help="BGE CE scoring batch size.")
    p.add_argument("--bge-max-length", type=int, default=512, help="BGE CE tokenizer max length.")
    p.add_argument(
        "--model-id",
        type=str,
        default="",
        help="Optional embedding model filter used only when prefilter-method=cosine.",
    )
    p.add_argument(
        "--top-k-per-spec",
        type=int,
        default=0,
        help="Keep only top-k candidates per spec in cache (0 = all aligned chunks).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    grant_db = _resolve_path(args.grant_db)
    fac_db = _resolve_path(args.fac_db)
    output = _resolve_path(args.output)
    manifest = _resolve_path(args.manifest)

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
        prefilter_method=_clean_text(args.prefilter_method),
        bge_model_id=_clean_text(args.bge_model_id),
        bge_batch_size=_safe_int(args.bge_batch_size, default=64, minimum=1, maximum=4096),
        bge_max_length=_safe_int(args.bge_max_length, default=512, minimum=32, maximum=4096),
    )

    print(f"output={output}")
    print(f"manifest={manifest}")
    print(f"prefilter_method={result.get('prefilter_method', 'cosine')}")
    print(f"written_spec_count={result.get('written_spec_count')}")
    print(f"candidate_count_per_spec={result.get('candidate_count_per_spec')}")
    print(f"total_written_candidates={result.get('total_written_candidates')}")
    print(f"aligned_spec_count={result.get('aligned_spec_count')}")
    print(f"aligned_chunk_count={result.get('aligned_chunk_count')}")
    print(f"common_dim={result.get('common_dim')}")
    print(f"elapsed_seconds={result.get('elapsed_seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
