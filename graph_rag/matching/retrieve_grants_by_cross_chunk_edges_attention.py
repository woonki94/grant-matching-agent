from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from neo4j import GraphDatabase

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from graph_rag.common import json_ready, load_dotenv_if_present, read_neo4j_settings
from graph_rag.matching.retrieve_grants_by_cross_chunk_edges import retrieve_grants_by_cross_chunk_edges


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_limit(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _safe_unit_float(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _truncate_text(value: Any, max_chars: int) -> str:
    text = _clean_text(value)
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class _CrossEncoderScorer:
    def __init__(self, *, model_name: str, batch_size: int):
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            raise RuntimeError(
                "Attention rerank requires sentence-transformers. "
                "Install with: pip install sentence-transformers torch"
            ) from exc

        self.batch_size = max(1, int(batch_size or 16))
        self.model_name = str(model_name or "").strip() or "BAAI/bge-reranker-v2-m3"
        self.model = CrossEncoder(self.model_name)

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        raw = self.model.predict(
            list(pairs),
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        vals = np.asarray(raw, dtype=np.float32).reshape(-1)
        out: List[float] = []
        for item in vals:
            out.append(_safe_unit_float(_sigmoid(float(item)), default=0.0))
        return out


def _fetch_grant_chunk_texts(
    *,
    driver,
    database: str,
    opportunity_id: str,
    chunk_ids: Sequence[str],
) -> Dict[str, str]:
    ids = [x for x in {_clean_text(v) for v in chunk_ids or []} if x]
    if not ids:
        return {}
    records, _, _ = driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(c:GrantTextChunk)
        WHERE c.chunk_id IN $chunk_ids
          AND c.text IS NOT NULL
          AND trim(toString(c.text)) <> ''
        RETURN c.chunk_id AS chunk_id, c.text AS text
        """,
        parameters_={
            "opportunity_id": _clean_text(opportunity_id),
            "chunk_ids": ids,
        },
        database_=database,
    )
    out: Dict[str, str] = {}
    for row in records:
        cid = _clean_text(row.get("chunk_id"))
        txt = _clean_text(row.get("text"))
        if cid and txt:
            out[cid] = txt
    return out


def _fetch_faculty_evidence_texts(
    *,
    driver,
    database: str,
    faculty_id: Optional[int],
    chunk_ids: Sequence[str],
    publication_ids: Sequence[int],
) -> Dict[Tuple[str, str], str]:
    out: Dict[Tuple[str, str], str] = {}

    cids = [x for x in {_clean_text(v) for v in chunk_ids or []} if x]
    if cids:
        records, _, _ = driver.execute_query(
            """
            MATCH (c:FacultyTextChunk)
            WHERE c.chunk_id IN $chunk_ids
              AND c.text IS NOT NULL
              AND trim(toString(c.text)) <> ''
            RETURN c.chunk_id AS evidence_id, c.text AS text
            """,
            parameters_={"chunk_ids": cids},
            database_=database,
        )
        for row in records:
            eid = _clean_text(row.get("evidence_id"))
            txt = _clean_text(row.get("text"))
            if eid and txt:
                out[("chunk", eid)] = txt

    pids = sorted({int(v) for v in publication_ids or [] if v is not None})
    if pids:
        if faculty_id is not None:
            records, _, _ = driver.execute_query(
                """
                MATCH (p:FacultyPublication)
                WHERE p.publication_id IN $publication_ids
                  AND p.faculty_id = $faculty_id
                  AND p.abstract IS NOT NULL
                  AND trim(toString(p.abstract)) <> ''
                RETURN toString(p.publication_id) AS evidence_id, p.abstract AS text
                """,
                parameters_={
                    "publication_ids": pids,
                    "faculty_id": int(faculty_id),
                },
                database_=database,
            )
        else:
            records, _, _ = driver.execute_query(
                """
                MATCH (p:FacultyPublication)
                WHERE p.publication_id IN $publication_ids
                  AND p.abstract IS NOT NULL
                  AND trim(toString(p.abstract)) <> ''
                RETURN toString(p.publication_id) AS evidence_id, p.abstract AS text
                """,
                parameters_={"publication_ids": pids},
                database_=database,
            )
        for row in records:
            eid = _clean_text(row.get("evidence_id"))
            txt = _clean_text(row.get("text"))
            if eid and txt:
                out[("publication", eid)] = txt

    return out


def _extract_pub_id(text_id: str) -> Optional[int]:
    token = _clean_text(text_id)
    if not token:
        return None
    try:
        return int(token)
    except Exception:
        return None


def _build_attention_pairs_for_grant(
    *,
    grant_row: Dict[str, Any],
    grant_chunk_texts: Dict[str, str],
    faculty_evidence_texts: Dict[Tuple[str, str], str],
    pair_limit_per_direction: int,
    max_pair_text_chars: int,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    safe_limit = _safe_limit(pair_limit_per_direction, default=10, minimum=1, maximum=200)
    safe_max_chars = _safe_limit(max_pair_text_chars, default=1800, minimum=200, maximum=12000)

    f2g_items = list(grant_row.get("fac_to_grant_chunk_pairs") or [])[:safe_limit]
    for item in f2g_items:
        left = _truncate_text(item.get("faculty_keyword_value"), safe_max_chars)
        chunk_id = _clean_text(item.get("grant_chunk_id"))
        right = _truncate_text(grant_chunk_texts.get(chunk_id, ""), safe_max_chars)
        if not left or not right:
            continue
        pairs.append(
            {
                "direction": "fac_to_grant_chunk",
                "left_text": left,
                "right_text": right,
                "edge_score": float(item.get("edge_score") or 0.0),
                "metadata": {
                    "faculty_keyword_value": _clean_text(item.get("faculty_keyword_value")),
                    "faculty_keyword_section": _clean_text(item.get("faculty_keyword_section")),
                    "grant_chunk_id": chunk_id,
                    "grant_chunk_source_type": _clean_text(item.get("grant_chunk_source_type")),
                },
            }
        )

    g2f_items = list(grant_row.get("grant_to_fac_evidence_pairs") or [])[:safe_limit]
    for item in g2f_items:
        left = _truncate_text(item.get("grant_keyword_value"), safe_max_chars)
        ev_kind = _clean_text(item.get("faculty_evidence_kind")).lower()
        ev_id = _clean_text(item.get("faculty_evidence_id"))
        right = _truncate_text(faculty_evidence_texts.get((ev_kind, ev_id), ""), safe_max_chars)
        if not left or not right:
            continue
        pairs.append(
            {
                "direction": "grant_to_fac_evidence",
                "left_text": left,
                "right_text": right,
                "edge_score": float(item.get("edge_score") or 0.0),
                "metadata": {
                    "grant_keyword_value": _clean_text(item.get("grant_keyword_value")),
                    "grant_keyword_section": _clean_text(item.get("grant_keyword_section")),
                    "faculty_evidence_kind": ev_kind,
                    "faculty_evidence_id": ev_id,
                    "faculty_evidence_source_type": _clean_text(item.get("faculty_evidence_source_type")),
                },
            }
        )
    return pairs


def retrieve_grants_by_cross_chunk_edges_attention(
    *,
    faculty_id: Optional[int],
    faculty_email: str,
    top_k: int,
    pairs_per_grant: int,
    min_edge_score: float,
    coverage_bonus: float,
    include_closed: bool,
    rerank_top_n: int,
    rerank_pairs_per_direction: int,
    attention_alpha: float,
    cross_encoder_model: str,
    cross_encoder_batch_size: int,
    max_pair_text_chars: int,
    uri: str = "",
    username: str = "",
    password: str = "",
    database: str = "",
) -> Dict[str, Any]:
    safe_rerank_top_n = _safe_limit(rerank_top_n, default=30, minimum=0, maximum=5000)
    safe_pairs_per_direction = _safe_limit(rerank_pairs_per_direction, default=10, minimum=1, maximum=200)
    safe_alpha = _safe_unit_float(attention_alpha, default=0.5)
    safe_cross_encoder_batch = _safe_limit(cross_encoder_batch_size, default=16, minimum=1, maximum=512)
    safe_max_pair_chars = _safe_limit(max_pair_text_chars, default=1800, minimum=200, maximum=12000)
    safe_model = _clean_text(cross_encoder_model) or "BAAI/bge-reranker-v2-m3"

    base_payload = retrieve_grants_by_cross_chunk_edges(
        faculty_id=faculty_id,
        faculty_email=faculty_email,
        top_k=top_k,
        pairs_per_grant=pairs_per_grant,
        min_edge_score=min_edge_score,
        coverage_bonus=coverage_bonus,
        include_closed=include_closed,
        uri=uri,
        username=username,
        password=password,
        database=database,
    )

    grants = list(base_payload.get("grants") or [])
    if not grants:
        out = dict(base_payload)
        out["rerank"] = {
            "enabled": bool(safe_rerank_top_n > 0),
            "reranked_grants": 0,
            "pairs_scored": 0,
            "model": safe_model,
            "alpha": safe_alpha,
        }
        return out

    max_graph_rank = max(float(x.get("rank_score") or 0.0) for x in grants) or 0.0
    for row in grants:
        graph_rank = float(row.get("rank_score") or 0.0)
        row["graph_rank_norm"] = (graph_rank / max_graph_rank) if max_graph_rank > 0 else 0.0
        row["attention_score"] = None
        row["attention_pairs_evaluated"] = 0
        row["attention_pair_details"] = []
        row["final_rank_score"] = float(row["graph_rank_norm"])

    if safe_rerank_top_n <= 0:
        grants.sort(key=lambda x: (float(x.get("final_rank_score") or 0.0), float(x.get("rank_score") or 0.0)), reverse=True)
        out = dict(base_payload)
        out["grants"] = grants
        out["rerank"] = {
            "enabled": False,
            "reranked_grants": 0,
            "pairs_scored": 0,
            "model": safe_model,
            "alpha": safe_alpha,
        }
        return out

    load_dotenv_if_present()
    settings = read_neo4j_settings(
        uri=uri,
        username=username,
        password=password,
        database=database,
    )

    scorer = _CrossEncoderScorer(
        model_name=safe_model,
        batch_size=safe_cross_encoder_batch,
    )

    rerank_candidates = grants[:safe_rerank_top_n]
    total_pairs_scored = 0

    with GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    ) as driver:
        driver.verify_connectivity()

        for grant_row in rerank_candidates:
            opp_id = _clean_text(grant_row.get("opportunity_id"))
            if not opp_id:
                continue

            f2g = list(grant_row.get("fac_to_grant_chunk_pairs") or [])[:safe_pairs_per_direction]
            g2f = list(grant_row.get("grant_to_fac_evidence_pairs") or [])[:safe_pairs_per_direction]

            grant_chunk_ids = [_clean_text(x.get("grant_chunk_id")) for x in f2g]
            grant_chunk_texts = _fetch_grant_chunk_texts(
                driver=driver,
                database=settings.database,
                opportunity_id=opp_id,
                chunk_ids=grant_chunk_ids,
            )

            ev_chunk_ids: List[str] = []
            ev_pub_ids: List[int] = []
            for x in g2f:
                kind = _clean_text(x.get("faculty_evidence_kind")).lower()
                eid = _clean_text(x.get("faculty_evidence_id"))
                if kind == "chunk":
                    if eid:
                        ev_chunk_ids.append(eid)
                elif kind == "publication":
                    pub_id = _extract_pub_id(eid)
                    if pub_id is not None:
                        ev_pub_ids.append(pub_id)

            faculty_evidence_texts = _fetch_faculty_evidence_texts(
                driver=driver,
                database=settings.database,
                faculty_id=faculty_id,
                chunk_ids=ev_chunk_ids,
                publication_ids=ev_pub_ids,
            )

            pair_rows = _build_attention_pairs_for_grant(
                grant_row=grant_row,
                grant_chunk_texts=grant_chunk_texts,
                faculty_evidence_texts=faculty_evidence_texts,
                pair_limit_per_direction=safe_pairs_per_direction,
                max_pair_text_chars=safe_max_pair_chars,
            )
            if not pair_rows:
                continue

            score_inputs = [(x["left_text"], x["right_text"]) for x in pair_rows]
            attention_scores = scorer.score_pairs(score_inputs)
            if not attention_scores:
                continue

            weighted_sum = 0.0
            weight_total = 0.0
            details: List[Dict[str, Any]] = []
            for idx, attn in enumerate(attention_scores):
                row = pair_rows[idx]
                edge_w = max(0.0001, float(row.get("edge_score") or 0.0))
                weighted_sum += float(attn) * edge_w
                weight_total += edge_w
                details.append(
                    {
                        "direction": _clean_text(row.get("direction")),
                        "attention_score": _safe_unit_float(attn, default=0.0),
                        "edge_score": float(row.get("edge_score") or 0.0),
                        "metadata": dict(row.get("metadata") or {}),
                    }
                )

            if weight_total <= 0.0:
                continue

            attention_score = max(0.0, min(1.0, weighted_sum / weight_total))
            graph_norm = float(grant_row.get("graph_rank_norm") or 0.0)
            final_rank = (1.0 - safe_alpha) * graph_norm + safe_alpha * attention_score

            grant_row["attention_score"] = attention_score
            grant_row["attention_pairs_evaluated"] = len(attention_scores)
            grant_row["attention_pair_details"] = details
            grant_row["final_rank_score"] = max(0.0, min(1.0, final_rank))
            total_pairs_scored += len(attention_scores)

    grants.sort(
        key=lambda x: (
            float(x.get("final_rank_score") or 0.0),
            float(x.get("rank_score") or 0.0),
            float(x.get("base_score") or 0.0),
        ),
        reverse=True,
    )

    out = dict(base_payload)
    out["grants"] = grants
    out["rerank"] = {
        "enabled": True,
        "reranked_grants": min(len(grants), safe_rerank_top_n),
        "pairs_scored": total_pairs_scored,
        "model": safe_model,
        "alpha": safe_alpha,
        "pairs_per_direction": safe_pairs_per_direction,
        "max_pair_text_chars": safe_max_pair_chars,
    }
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve grants by cross-chunk graph edges and rerank top-N results "
            "with cross-encoder attention scoring."
        )
    )
    parser.add_argument("--faculty-id", type=int, default=0, help="Faculty ID filter.")
    parser.add_argument("--faculty-email", type=str, default="", help="Faculty email filter.")
    parser.add_argument("--top-k", type=int, default=20, help="Top K ranked grants from graph retrieval.")
    parser.add_argument("--pairs-per-grant", type=int, default=30, help="Matched pair rows per direction returned by base retrieval.")
    parser.add_argument("--min-edge-score", type=float, default=0.0, help="Minimum edge score for base retrieval.")
    parser.add_argument("--coverage-bonus", type=float, default=0.0, help="Optional coverage bonus for base graph rank.")
    parser.add_argument("--include-closed", action="store_true", help="Include closed grants in output.")
    parser.add_argument("--rerank-top-n", type=int, default=30, help="Top-N grants to rerank with attention.")
    parser.add_argument("--rerank-pairs-per-direction", type=int, default=10, help="Max pairs per direction scored per grant.")
    parser.add_argument("--attention-alpha", type=float, default=0.5, help="Blend weight for attention rerank (0..1).")
    parser.add_argument("--cross-encoder-model", type=str, default="BAAI/bge-reranker-v2-m3", help="Cross-encoder model name.")
    parser.add_argument("--cross-encoder-batch-size", type=int, default=16, help="Cross-encoder batch size.")
    parser.add_argument("--max-pair-text-chars", type=int, default=1800, help="Max chars per side per pair for rerank scoring.")
    parser.add_argument("--json-only", action="store_true", help="Print only JSON output.")

    parser.add_argument("--uri", type=str, default="", help="Neo4j URI. Fallback: NEO4J_URI")
    parser.add_argument("--username", type=str, default="", help="Neo4j username. Fallback: NEO4J_USERNAME")
    parser.add_argument("--password", type=str, default="", help="Neo4j password. Fallback: NEO4J_PASSWORD")
    parser.add_argument("--database", type=str, default="", help="Neo4j database. Fallback: NEO4J_DATABASE or neo4j")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    fid = int(args.faculty_id or 0)
    faculty_id = fid if fid > 0 else None

    payload = retrieve_grants_by_cross_chunk_edges_attention(
        faculty_id=faculty_id,
        faculty_email=_clean_text(args.faculty_email).lower(),
        top_k=int(args.top_k or 20),
        pairs_per_grant=int(args.pairs_per_grant or 30),
        min_edge_score=float(args.min_edge_score),
        coverage_bonus=float(args.coverage_bonus),
        include_closed=bool(args.include_closed),
        rerank_top_n=int(args.rerank_top_n or 30),
        rerank_pairs_per_direction=int(args.rerank_pairs_per_direction or 10),
        attention_alpha=float(args.attention_alpha),
        cross_encoder_model=args.cross_encoder_model,
        cross_encoder_batch_size=int(args.cross_encoder_batch_size or 16),
        max_pair_text_chars=int(args.max_pair_text_chars or 1800),
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    if not args.json_only:
        print("Grant retrieval + attention rerank complete.")
        print(f"  grants returned   : {payload.get('totals', {}).get('grants', 0)}")
        print(f"  reranked grants   : {payload.get('rerank', {}).get('reranked_grants', 0)}")
        print(f"  pairs scored      : {payload.get('rerank', {}).get('pairs_scored', 0)}")
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

