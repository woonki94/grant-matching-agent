from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Tuple


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "ce2").is_dir() and (parent / "ce").is_dir():
            return parent
    return here.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ce2.data_preparation.utils import resolve_path  # noqa: E402


INPUT_SPLIT_DIR_DEFAULT = "ce2/dataset/splits"
OUTPUT_DIR_DEFAULT = "ce2/dataset/splits_ce_train2"
MANIFEST_BASENAME = "ce_train2_split_manifest.json"
ASPECT_NAME_MAP = {
    "domain": "domain",
    "method": "method",
    "target": "constraint",
    "constraint": "constraint",
}
DEFAULT_SOURCE_ASPECTS = ("domain", "method", "target")
DEFAULT_SPLITS = ("train", "val", "test")


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


def _clamp01(value: Any, default: float = 0.0) -> float:
    out = _safe_float(value, default=default)
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


def _parse_csv(value: Any) -> List[str]:
    out: List[str] = []
    for part in str(value or "").split(","):
        name = _normalize_ws(part).lower()
        if name:
            out.append(name)
    return out


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


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    return [row for row in _iter_jsonl(path)]


def _write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _extract_doc_text(row: Dict[str, Any]) -> str:
    for key in ("doc_text", "fac_spec_text", "chunk_text", "text"):
        text = _normalize_ws(row.get(key))
        if text:
            return text
    return ""


def _extract_query_text(row: Dict[str, Any]) -> str:
    return _normalize_ws(row.get("query_text") or row.get("spec_text"))


def _extract_score_raw(row: Dict[str, Any]) -> float:
    for key in ("teacher_score_raw", "score_raw", "teacher_score_used", "teacher_score", "score"):
        if row.get(key) is not None:
            return _clamp01(row.get(key), default=0.0)
    return 0.0


def _extract_score_norm(row: Dict[str, Any]) -> float:
    for key in ("teacher_score_used", "teacher_score", "score", "score_norm", "teacher_score_raw", "score_raw"):
        if row.get(key) is not None:
            return _clamp01(row.get(key), default=0.0)
    return 0.0


def _candidate_from_flat_row(row: Dict[str, Any]) -> Dict[str, Any]:
    doc_text = _extract_doc_text(row)
    if not doc_text:
        return {}

    doc_id_text = _normalize_ws(row.get("doc_id") or row.get("fac_spec_id") or row.get("chunk_id"))
    doc_id = _safe_int(doc_id_text, default=_safe_int(row.get("fac_spec_id"), default=_safe_int(row.get("chunk_id"), default=0)))
    fac_id = _safe_int(row.get("fac_id"), default=doc_id)
    chunk_index = _safe_int(row.get("chunk_index"), default=_safe_int(row.get("fac_spec_idx"), default=0))
    source_type = _normalize_ws(row.get("source_type") or row.get("section") or row.get("source") or row.get("pair_source")) or "unknown"
    band = _normalize_ws(row.get("band") or row.get("score_band")).lower()
    target_cluster = band if band in {"high", "mid", "low"} else "unknown"
    score_raw = _extract_score_raw(row)
    score_norm = _extract_score_norm(row)
    is_augmented = bool(row.get("is_augmented", False)) or _normalize_ws(row.get("source")).lower() == "augmentation"
    selected_for_target = bool(row.get("selected_for_target", True))

    return {
        "fac_id": int(fac_id),
        "fac_spec_id": int(doc_id),
        "fac_spec_idx": int(chunk_index),
        "chunk_id": int(doc_id),
        "chunk_index": int(chunk_index),
        "source_type": source_type,
        "section": source_type,
        "target_cluster": target_cluster,
        "selected_for_target": selected_for_target,
        "is_augmented": bool(is_augmented),
        "is_disagreement": bool(row.get("is_disagreement", False)),
        "teacher_score": float(score_norm),
        "teacher_score_raw": float(score_raw),
        "score": float(score_norm),
        "score_raw": float(score_raw),
        "text": doc_text,
        "fac_spec_text": doc_text,
        "chunk_text": doc_text,
    }


def _convert_listwise_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    min_docs_per_query: int,
    max_docs_per_query: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    grouped: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

    for row in rows:
        grant_id = _normalize_ws(row.get("grant_id") or row.get("query_id"))
        spec_idx = _safe_int(row.get("spec_idx"), default=0)
        query_text = _extract_query_text(row)
        if (not grant_id) or (not query_text):
            continue

        key = (grant_id, int(spec_idx), query_text)
        if key not in grouped:
            grouped[key] = {
                "grant_id": grant_id,
                "spec_idx": int(spec_idx),
                "query_text": query_text,
                "docs": [],
                "_seen": set(),
            }

        cand = _candidate_from_flat_row(row)
        if not cand:
            continue
        dedup_key = f"{int(cand.get('fac_id', 0))}::{int(cand.get('fac_spec_id', 0))}::{_normalize_ws(cand.get('text'))}"
        if dedup_key in grouped[key]["_seen"]:
            continue
        grouped[key]["_seen"].add(dedup_key)
        grouped[key]["docs"].append(cand)

    min_docs = max(1, int(min_docs_per_query))
    max_docs = max(0, int(max_docs_per_query))
    out: List[Dict[str, Any]] = []
    dropped_too_small = 0

    for key in sorted(grouped):
        group = grouped[key]
        docs = list(group.get("docs") or [])
        docs.sort(
            key=lambda d: (
                -float(d.get("teacher_score") or 0.0),
                _safe_int(d.get("fac_spec_id"), default=0),
                _normalize_ws(d.get("text")),
            )
        )
        if max_docs > 0:
            docs = docs[:max_docs]
        if len(docs) < min_docs:
            dropped_too_small += 1
            continue

        for rank, doc in enumerate(docs, start=1):
            doc["rank"] = int(rank)
            doc["sts_rank"] = int(rank - 1)

        out.append(
            {
                "grant_id": str(group["grant_id"]),
                "spec_idx": int(group["spec_idx"]),
                "query_text": str(group["query_text"]),
                "spec_text": str(group["query_text"]),
                "docs": docs,
            }
        )

    stats = {
        "input_rows": int(len(rows)),
        "query_groups_before_filter": int(len(grouped)),
        "query_groups_after_filter": int(len(out)),
        "dropped_groups_too_small": int(dropped_too_small),
    }
    return out, stats


def _convert_pairwise_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        grant_id = _normalize_ws(row.get("grant_id") or row.get("query_id"))
        query_text = _extract_query_text(row)
        pos_text = _normalize_ws(row.get("pos_text"))
        neg_text = _normalize_ws(row.get("neg_text"))
        if (not grant_id) or (not query_text) or (not pos_text) or (not neg_text):
            continue

        pos_score = _clamp01(row.get("teacher_pos_score"), default=0.0)
        neg_score = _clamp01(row.get("teacher_neg_score"), default=0.0)
        teacher_margin = _safe_float(row.get("teacher_margin"), default=(pos_score - neg_score))
        if teacher_margin <= 0.0:
            teacher_margin = max(0.0, pos_score - neg_score)

        out.append(
            {
                "grant_id": grant_id,
                "spec_idx": _safe_int(row.get("spec_idx"), default=0),
                "query_id": _normalize_ws(row.get("query_id")),
                "query_text": query_text,
                "pos_doc_id": _normalize_ws(row.get("pos_doc_id")),
                "neg_doc_id": _normalize_ws(row.get("neg_doc_id")),
                "pos_pair_id": _normalize_ws(row.get("pos_pair_id")),
                "neg_pair_id": _normalize_ws(row.get("neg_pair_id")),
                "pos_text": pos_text,
                "neg_text": neg_text,
                "teacher_pos_score": float(pos_score),
                "teacher_neg_score": float(neg_score),
                "teacher_margin": float(teacher_margin),
                "pair_type": _normalize_ws(row.get("pair_type")) or "unknown",
            }
        )
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert CE2 split outputs into CE train2-compatible listwise/pairwise split files "
            "(maps target -> constraint and groups listwise rows by query)."
        )
    )
    p.add_argument("--input-split-dir", type=str, default=INPUT_SPLIT_DIR_DEFAULT)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--source-aspects", type=str, default=",".join(DEFAULT_SOURCE_ASPECTS))
    p.add_argument("--splits", type=str, default=",".join(DEFAULT_SPLITS))
    p.add_argument("--min-docs-per-query", type=int, default=2)
    p.add_argument("--max-docs-per-query", type=int, default=0, help="0 keeps all docs per query.")
    p.add_argument("--require-pairwise", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--write-aggregate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    input_dir = resolve_path(PROJECT_ROOT, args.input_split_dir)
    output_dir = resolve_path(PROJECT_ROOT, args.output_dir)
    source_aspects = _parse_csv(args.source_aspects) or list(DEFAULT_SOURCE_ASPECTS)
    splits = _parse_csv(args.splits) or list(DEFAULT_SPLITS)
    min_docs_per_query = max(1, int(args.min_docs_per_query))
    max_docs_per_query = max(0, int(args.max_docs_per_query))

    if not input_dir.exists():
        raise FileNotFoundError(f"Input split directory not found: {input_dir}")

    targets: List[str] = []
    for src_aspect in source_aspects:
        mapped = ASPECT_NAME_MAP.get(src_aspect, src_aspect)
        if mapped not in targets:
            targets.append(mapped)

    expected_outputs: List[Path] = []
    for target_aspect in targets:
        for split in splits:
            expected_outputs.append(output_dir / f"llm_distill_{target_aspect}_listwise_{split}.jsonl")
            expected_outputs.append(output_dir / f"llm_distill_{target_aspect}_pairwise_{split}.jsonl")
        if bool(args.write_aggregate):
            expected_outputs.append(output_dir / f"llm_distill_{target_aspect}_listwise.jsonl")
            expected_outputs.append(output_dir / f"llm_distill_{target_aspect}_pairwise.jsonl")
    expected_outputs.append(output_dir / MANIFEST_BASENAME)

    if not bool(args.overwrite):
        existing = [str(path) for path in expected_outputs if path.exists()]
        if existing:
            raise FileExistsError("Output files already exist. Use --overwrite.\n" + "\n".join(existing))

    listwise_by_target_split: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    pairwise_by_target_split: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    listwise_stats: Dict[str, Dict[str, Dict[str, int]]] = {}
    source_file_map: Dict[str, Dict[str, Dict[str, str]]] = {}

    for src_aspect in source_aspects:
        target_aspect = ASPECT_NAME_MAP.get(src_aspect, src_aspect)
        listwise_stats.setdefault(target_aspect, {})
        source_file_map.setdefault(target_aspect, {})
        for split in splits:
            in_list_path = input_dir / f"{src_aspect}_{split}.jsonl"
            in_pair_path = input_dir / f"{src_aspect}_pairwise_{split}.jsonl"
            if not in_list_path.exists():
                raise FileNotFoundError(f"Missing CE2 listwise split file: {in_list_path}")
            if bool(args.require_pairwise) and (not in_pair_path.exists()):
                raise FileNotFoundError(f"Missing CE2 pairwise split file: {in_pair_path}")

            list_rows_flat = _load_rows(in_list_path)
            list_rows_ce, stats = _convert_listwise_rows(
                list_rows_flat,
                min_docs_per_query=min_docs_per_query,
                max_docs_per_query=max_docs_per_query,
            )
            listwise_by_target_split[(target_aspect, split)].extend(list_rows_ce)
            listwise_stats[target_aspect][split] = stats
            source_file_map[target_aspect][split] = {
                "source_listwise": str(in_list_path),
                "source_pairwise": str(in_pair_path) if in_pair_path.exists() else "",
            }

            if in_pair_path.exists():
                pair_rows_flat = _load_rows(in_pair_path)
                pair_rows_ce = _convert_pairwise_rows(pair_rows_flat)
                pairwise_by_target_split[(target_aspect, split)].extend(pair_rows_ce)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    aggregate_counts: Dict[str, Dict[str, int]] = {}

    for target_aspect in targets:
        out_counts[target_aspect] = {}
        aggregate_listwise: List[Dict[str, Any]] = []
        aggregate_pairwise: List[Dict[str, Any]] = []
        for split in splits:
            key = (target_aspect, split)
            list_rows = listwise_by_target_split.get(key, [])
            pair_rows = pairwise_by_target_split.get(key, [])

            out_list_path = output_dir / f"llm_distill_{target_aspect}_listwise_{split}.jsonl"
            out_pair_path = output_dir / f"llm_distill_{target_aspect}_pairwise_{split}.jsonl"
            n_list = _write_rows(out_list_path, list_rows)
            n_pair = _write_rows(out_pair_path, pair_rows)

            out_counts[target_aspect][split] = {"listwise_rows": int(n_list), "pairwise_rows": int(n_pair)}
            aggregate_listwise.extend(list_rows)
            aggregate_pairwise.extend(pair_rows)

        if bool(args.write_aggregate):
            n_list_all = _write_rows(output_dir / f"llm_distill_{target_aspect}_listwise.jsonl", aggregate_listwise)
            n_pair_all = _write_rows(output_dir / f"llm_distill_{target_aspect}_pairwise.jsonl", aggregate_pairwise)
            aggregate_counts[target_aspect] = {"listwise_rows": int(n_list_all), "pairwise_rows": int(n_pair_all)}

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_split_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_aspects": source_aspects,
        "target_aspects": targets,
        "aspect_name_map": dict(ASPECT_NAME_MAP),
        "splits": splits,
        "min_docs_per_query": int(min_docs_per_query),
        "max_docs_per_query": int(max_docs_per_query),
        "require_pairwise": bool(args.require_pairwise),
        "write_aggregate": bool(args.write_aggregate),
        "source_files": source_file_map,
        "listwise_grouping_stats": listwise_stats,
        "output_counts": out_counts,
        "aggregate_counts": aggregate_counts,
    }
    manifest_path = output_dir / MANIFEST_BASENAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"input_split_dir={input_dir}")
    print(f"output_dir={output_dir}")
    print(f"manifest={manifest_path}")
    print(f"source_aspects={','.join(source_aspects)}")
    print(f"target_aspects={','.join(targets)}")
    print(f"splits={','.join(splits)}")
    print(f"output_counts={json.dumps(out_counts, ensure_ascii=False)}")
    if aggregate_counts:
        print(f"aggregate_counts={json.dumps(aggregate_counts, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
