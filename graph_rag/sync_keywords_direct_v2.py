from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import settings
from graph_rag.common import json_ready, load_dotenv_if_present, read_neo4j_settings, safe_text
from graph_rag.faculty.sync_neo4j import _clean_text as _clean_fac_text
from graph_rag.faculty.sync_neo4j import _embed_text_values as _embed_fac_values
from graph_rag.grant.sync_neo4j import _clean_text as _clean_grant_text
from graph_rag.grant.sync_neo4j import _embed_text_values as _embed_grant_values
from services.keywords.faculty_keyword_generator_v2 import FacultyKeywordGeneratorV2
from services.keywords.grant_keyword_generator_v2 import GrantKeywordGeneratorV2
from utils.embedder import embed_domain_bucket


FACULTY_RELATIONS = {
    "domain": "HAS_RESEARCH_DOMAIN",
    "specialization": "HAS_RESEARCH_SPECIALIZATION",
}

GRANT_RELATIONS = {
    "domain": "HAS_RESEARCH_DOMAIN",
    "specialization": "HAS_RESEARCH_SPECIALIZATION",
}


def _coerce_weight(value: Any, *, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _snippet_map(value: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(value, dict):
        for sid_raw, score_raw in value.items():
            sid = safe_text(sid_raw)
            if not sid:
                continue
            out[sid] = _coerce_weight(score_raw, default=0.8)
        return out
    if isinstance(value, list):
        for sid_raw in value:
            sid = safe_text(sid_raw)
            if sid:
                out[sid] = 0.8
    return out


def _faculty_keyword_rows_from_v2_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    if not isinstance(payload, dict):
        return out

    domains = payload.get("domains") or []
    if isinstance(domains, list):
        for item in domains:
            if isinstance(item, dict):
                value = _clean_fac_text(item.get("t") or item.get("text"), max_chars=300)
                weight = item.get("w")
                snippet_ids = _snippet_map(item.get("snippet_ids"))
            else:
                value = _clean_fac_text(item, max_chars=300)
                weight = None
                snippet_ids = {}
            if not value:
                continue
            key = ("domain", value.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "value": value,
                    "section": "research",
                    "bucket": "domain",
                    "relation": FACULTY_RELATIONS["domain"],
                    "weight": None if weight is None else _coerce_weight(weight),
                    "snippet_ids": snippet_ids,
                    "embedding": None,
                    "embedding_model": None,
                }
            )

    specs = payload.get("specializations") or []
    if isinstance(specs, list):
        for item in specs:
            if isinstance(item, dict):
                value = _clean_fac_text(item.get("t") or item.get("text"), max_chars=300)
                weight = item.get("w")
                snippet_ids = _snippet_map(item.get("snippet_ids"))
            else:
                value = _clean_fac_text(item, max_chars=300)
                weight = None
                snippet_ids = {}
            if not value:
                continue
            key = ("specialization", value.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "value": value,
                    "section": "research",
                    "bucket": "specialization",
                    "relation": FACULTY_RELATIONS["specialization"],
                    "weight": _coerce_weight(weight, default=0.5),
                    "snippet_ids": snippet_ids,
                    "embedding": None,
                    "embedding_model": None,
                }
            )

    return out


def _grant_keyword_rows_from_v2_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    if not isinstance(payload, dict):
        return out

    domains = payload.get("domains") or []
    if isinstance(domains, list):
        for item in domains:
            if isinstance(item, dict):
                value = _clean_grant_text(item.get("t") or item.get("text"), max_chars=300)
                weight = item.get("w")
                snippet_ids = _snippet_map(item.get("snippet_ids"))
            else:
                value = _clean_grant_text(item, max_chars=300)
                weight = None
                snippet_ids = {}
            if not value:
                continue
            key = ("domain", value.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "value": value,
                    "section": "research",
                    "bucket": "domain",
                    "relation": GRANT_RELATIONS["domain"],
                    "weight": None if weight is None else _coerce_weight(weight),
                    "snippet_ids": snippet_ids,
                    "embedding": None,
                    "embedding_model": None,
                }
            )

    specs = payload.get("specializations") or []
    if isinstance(specs, list):
        for item in specs:
            if isinstance(item, dict):
                value = _clean_grant_text(item.get("t") or item.get("text"), max_chars=300)
                weight = item.get("w")
                snippet_ids = _snippet_map(item.get("snippet_ids"))
            else:
                value = _clean_grant_text(item, max_chars=300)
                weight = None
                snippet_ids = {}
            if not value:
                continue
            key = ("specialization", value.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "value": value,
                    "section": "research",
                    "bucket": "specialization",
                    "relation": GRANT_RELATIONS["specialization"],
                    "weight": _coerce_weight(weight, default=0.5),
                    "snippet_ids": snippet_ids,
                    "embedding": None,
                    "embedding_model": None,
                }
            )

    return out


def _faculty_support_rows_from_keywords(keyword_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    chunk_rows: List[Dict[str, Any]] = []
    pub_rows: List[Dict[str, Any]] = []
    seen_chunk = set()
    seen_pub = set()

    for row in keyword_rows:
        if str(row.get("bucket") or "") != "specialization":
            continue
        snippet_ids = _snippet_map(row.get("snippet_ids"))
        for sid, score in snippet_ids.items():
            if sid.startswith("pub|"):
                parts = sid.split("|")
                if len(parts) >= 4:
                    try:
                        publication_id = int(parts[3])
                    except Exception:
                        publication_id = None
                else:
                    publication_id = None
                if publication_id is None:
                    key = (
                        row.get("value"),
                        row.get("section"),
                        row.get("bucket"),
                        sid,
                    )
                    if key in seen_chunk:
                        continue
                    seen_chunk.add(key)
                    chunk_rows.append(
                        {
                            "keyword_value": row.get("value"),
                            "keyword_section": row.get("section"),
                            "keyword_bucket": row.get("bucket"),
                            "chunk_id": sid,
                            "score": score,
                        }
                    )
                    continue
                key = (
                    row.get("value"),
                    row.get("section"),
                    row.get("bucket"),
                    publication_id,
                )
                if key in seen_pub:
                    continue
                seen_pub.add(key)
                pub_rows.append(
                    {
                        "keyword_value": row.get("value"),
                        "keyword_section": row.get("section"),
                        "keyword_bucket": row.get("bucket"),
                        "publication_id": publication_id,
                        "score": score,
                    }
                )
            else:
                key = (
                    row.get("value"),
                    row.get("section"),
                    row.get("bucket"),
                    sid,
                )
                if key in seen_chunk:
                    continue
                seen_chunk.add(key)
                chunk_rows.append(
                    {
                        "keyword_value": row.get("value"),
                        "keyword_section": row.get("section"),
                        "keyword_bucket": row.get("bucket"),
                        "chunk_id": sid,
                        "score": score,
                    }
                )
    return chunk_rows, pub_rows


def _grant_support_rows_from_keywords(keyword_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in keyword_rows:
        if str(row.get("bucket") or "") != "specialization":
            continue
        snippet_ids = _snippet_map(row.get("snippet_ids"))
        for sid, score in snippet_ids.items():
            key = (
                row.get("value"),
                row.get("section"),
                row.get("bucket"),
                sid,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "keyword_value": row.get("value"),
                    "keyword_section": row.get("section"),
                    "keyword_bucket": row.get("bucket"),
                    "chunk_id": sid,
                    "score": score,
                }
            )
    return out


def _attach_embeddings_to_faculty_keywords(
    keyword_rows: List[Dict[str, Any]],
    *,
    include_embeddings: bool,
    embedding_batch_size: int,
) -> None:
    if not include_embeddings or not keyword_rows:
        return
    embedding_model = (settings.bedrock_embed_model_id or "").strip()

    spec_texts = [str(x.get("value") or "") for x in keyword_rows if str(x.get("bucket") or "") == "specialization"]
    spec_embeddings = _embed_fac_values(spec_texts, batch_size=max(1, int(embedding_batch_size)))
    for row in keyword_rows:
        if str(row.get("bucket") or "") != "specialization":
            continue
        key_text = _clean_fac_text(row.get("value"))
        emb = spec_embeddings.get(key_text or "")
        if emb:
            row["embedding"] = emb
            row["embedding_model"] = embedding_model

    for row in keyword_rows:
        if str(row.get("bucket") or "") != "domain":
            continue
        value = str(row.get("value") or "").strip()
        if not value:
            continue
        vec = embed_domain_bucket([value])
        if vec:
            row["embedding"] = [float(x) for x in vec]
            row["embedding_model"] = embedding_model


def _attach_embeddings_to_grant_keywords(
    keyword_rows: List[Dict[str, Any]],
    *,
    include_embeddings: bool,
    embedding_batch_size: int,
) -> None:
    if not include_embeddings or not keyword_rows:
        return
    embedding_model = (settings.bedrock_embed_model_id or "").strip()

    spec_texts = [str(x.get("value") or "") for x in keyword_rows if str(x.get("bucket") or "") == "specialization"]
    spec_embeddings = _embed_grant_values(spec_texts, batch_size=max(1, int(embedding_batch_size)))
    for row in keyword_rows:
        if str(row.get("bucket") or "") != "specialization":
            continue
        key_text = _clean_grant_text(row.get("value"))
        emb = spec_embeddings.get(key_text or "")
        if emb:
            row["embedding"] = emb
            row["embedding_model"] = embedding_model

    for row in keyword_rows:
        if str(row.get("bucket") or "") != "domain":
            continue
        value = str(row.get("value") or "").strip()
        if not value:
            continue
        vec = embed_domain_bucket([value])
        if vec:
            row["embedding"] = [float(x) for x in vec]
            row["embedding_model"] = embedding_model


def sync_faculty_keywords_direct_to_neo4j(
    *,
    driver,
    database: str,
    faculty_id: int,
    keywords: Dict[str, Any],
    include_embeddings: bool,
    embedding_batch_size: int,
) -> Dict[str, Any]:
    fid = int(faculty_id)
    keyword_rows = _faculty_keyword_rows_from_v2_payload(keywords)
    _attach_embeddings_to_faculty_keywords(
        keyword_rows,
        include_embeddings=include_embeddings,
        embedding_batch_size=embedding_batch_size,
    )
    support_chunk_rows, support_pub_rows = _faculty_support_rows_from_keywords(keyword_rows)

    driver.execute_query(
        """
        MATCH (f:Faculty {faculty_id: $faculty_id})-[r]->(:FacultyKeyword)
        WHERE type(r) IN [
            'HAS_RESEARCH_DOMAIN',
            'HAS_RESEARCH_SPECIALIZATION',
            'HAS_APPLICATION_DOMAIN',
            'HAS_APPLICATION_SPECIALIZATION'
        ]
        DELETE r
        """,
        parameters_={"faculty_id": fid},
        database_=database,
    )
    driver.execute_query(
        """
        MATCH (:FacultyKeyword)-[r:SUPPORTED_BY_FACULTY_CHUNK {scope_faculty_id: $faculty_id}]->(:FacultyTextChunk)
        DELETE r
        """,
        parameters_={"faculty_id": fid},
        database_=database,
    )
    driver.execute_query(
        """
        MATCH (:FacultyKeyword)-[r:SUPPORTED_BY_FACULTY_PUBLICATION {scope_faculty_id: $faculty_id}]->(:FacultyPublication)
        DELETE r
        """,
        parameters_={"faculty_id": fid},
        database_=database,
    )

    for relation in (FACULTY_RELATIONS["domain"], FACULTY_RELATIONS["specialization"]):
        rows = [item for item in keyword_rows if item.get("relation") == relation]
        if not rows:
            continue
        driver.execute_query(
            f"""
            MATCH (f:Faculty {{faculty_id: $faculty_id}})
            UNWIND $rows AS row
            MERGE (k:FacultyKeyword {{
                value: row.value,
                section: row.section,
                bucket: row.bucket
            }})
            SET
                k.embedding = row.embedding,
                k.embedding_model = row.embedding_model,
                k.updated_at = datetime()
            MERGE (f)-[r:{relation}]->(k)
            SET
                r.weight = row.weight,
                r.updated_at = datetime()
            """,
            parameters_={"faculty_id": fid, "rows": rows},
            database_=database,
        )

    linked_chunk_rows = 0
    if support_chunk_rows:
        records, _, _ = driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (k:FacultyKeyword {
                value: row.keyword_value,
                section: row.keyword_section,
                bucket: row.keyword_bucket
            })
            MATCH (c:FacultyTextChunk {chunk_id: row.chunk_id})
            MERGE (k)-[r:SUPPORTED_BY_FACULTY_CHUNK {
                scope_faculty_id: $faculty_id,
                chunk_id: row.chunk_id,
                keyword_value: row.keyword_value,
                keyword_section: row.keyword_section,
                keyword_bucket: row.keyword_bucket
            }]->(c)
            SET
                r.score = row.score,
                r.method = 'snippet_ids',
                r.updated_at = datetime()
            RETURN count(r) AS linked_count
            """,
            parameters_={"faculty_id": fid, "rows": support_chunk_rows},
            database_=database,
        )
        if records:
            linked_chunk_rows = int(records[0].get("linked_count") or 0)

    linked_pub_rows = 0
    if support_pub_rows:
        records, _, _ = driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (k:FacultyKeyword {
                value: row.keyword_value,
                section: row.keyword_section,
                bucket: row.keyword_bucket
            })
            MATCH (p:FacultyPublication {publication_id: row.publication_id})
            WHERE p.faculty_id = $faculty_id
            MERGE (k)-[r:SUPPORTED_BY_FACULTY_PUBLICATION {
                scope_faculty_id: $faculty_id,
                publication_id: row.publication_id,
                keyword_value: row.keyword_value,
                keyword_section: row.keyword_section,
                keyword_bucket: row.keyword_bucket
            }]->(p)
            SET
                r.score = row.score,
                r.method = 'snippet_ids',
                r.updated_at = datetime()
            RETURN count(r) AS linked_count
            """,
            parameters_={"faculty_id": fid, "rows": support_pub_rows},
            database_=database,
        )
        if records:
            linked_pub_rows = int(records[0].get("linked_count") or 0)

    return {
        "faculty_id": fid,
        "counts": {
            "keywords": len(keyword_rows),
            "domain_keywords": len([x for x in keyword_rows if str(x.get("bucket") or "") == "domain"]),
            "specialization_keywords": len([x for x in keyword_rows if str(x.get("bucket") or "") == "specialization"]),
            "embedded_keywords": len([x for x in keyword_rows if x.get("embedding")]),
            "support_chunk_candidates": len(support_chunk_rows),
            "support_publication_candidates": len(support_pub_rows),
            "support_chunk_linked": int(linked_chunk_rows),
            "support_publication_linked": int(linked_pub_rows),
            "support_linked": int(linked_chunk_rows + linked_pub_rows),
        },
    }


def sync_grant_keywords_direct_to_neo4j(
    *,
    driver,
    database: str,
    opportunity_id: str,
    keywords: Dict[str, Any],
    include_embeddings: bool,
    embedding_batch_size: int,
) -> Dict[str, Any]:
    opp_id = str(opportunity_id or "").strip()
    keyword_rows = _grant_keyword_rows_from_v2_payload(keywords)
    _attach_embeddings_to_grant_keywords(
        keyword_rows,
        include_embeddings=include_embeddings,
        embedding_batch_size=embedding_batch_size,
    )
    support_rows = _grant_support_rows_from_keywords(keyword_rows)

    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:GrantKeyword)
        WHERE type(r) IN [
            'HAS_RESEARCH_DOMAIN',
            'HAS_RESEARCH_SPECIALIZATION',
            'HAS_APPLICATION_DOMAIN',
            'HAS_APPLICATION_SPECIALIZATION'
        ]
        DELETE r
        """,
        parameters_={"opportunity_id": opp_id},
        database_=database,
    )
    driver.execute_query(
        """
        MATCH (:GrantKeyword)-[r:SUPPORTED_BY_GRANT_CHUNK {scope_opportunity_id: $opportunity_id}]->(:GrantTextChunk)
        DELETE r
        """,
        parameters_={"opportunity_id": opp_id},
        database_=database,
    )

    for relation in (GRANT_RELATIONS["domain"], GRANT_RELATIONS["specialization"]):
        rows = [item for item in keyword_rows if item.get("relation") == relation]
        if not rows:
            continue
        driver.execute_query(
            f"""
            MATCH (g:Grant {{opportunity_id: $opportunity_id}})
            UNWIND $rows AS row
            MERGE (k:GrantKeyword {{
                value: row.value,
                section: row.section,
                bucket: row.bucket
            }})
            SET
                k.embedding = row.embedding,
                k.embedding_model = row.embedding_model,
                k.updated_at = datetime()
            MERGE (g)-[r:{relation}]->(k)
            SET
                r.weight = row.weight,
                r.updated_at = datetime()
            """,
            parameters_={"opportunity_id": opp_id, "rows": rows},
            database_=database,
        )

    linked_chunk_rows = 0
    if support_rows:
        records, _, _ = driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (k:GrantKeyword {
                value: row.keyword_value,
                section: row.keyword_section,
                bucket: row.keyword_bucket
            })
            MATCH (c:GrantTextChunk {chunk_id: row.chunk_id})
            WHERE c.opportunity_id = $opportunity_id
            MERGE (k)-[r:SUPPORTED_BY_GRANT_CHUNK {
                scope_opportunity_id: $opportunity_id,
                chunk_id: row.chunk_id,
                keyword_value: row.keyword_value,
                keyword_section: row.keyword_section,
                keyword_bucket: row.keyword_bucket
            }]->(c)
            SET
                r.score = row.score,
                r.method = 'snippet_ids',
                r.updated_at = datetime()
            RETURN count(r) AS linked_count
            """,
            parameters_={"opportunity_id": opp_id, "rows": support_rows},
            database_=database,
        )
        if records:
            linked_chunk_rows = int(records[0].get("linked_count") or 0)

    return {
        "opportunity_id": opp_id,
        "counts": {
            "keywords": len(keyword_rows),
            "domain_keywords": len([x for x in keyword_rows if str(x.get("bucket") or "") == "domain"]),
            "specialization_keywords": len([x for x in keyword_rows if str(x.get("bucket") or "") == "specialization"]),
            "embedded_keywords": len([x for x in keyword_rows if x.get("embedding")]),
            "support_chunk_candidates": len(support_rows),
            "support_chunk_linked": int(linked_chunk_rows),
            "support_linked": int(linked_chunk_rows),
        },
    }


class FacultyKeywordGeneratorV2Neo4jSink(FacultyKeywordGeneratorV2):
    def __init__(
        self,
        *,
        driver,
        database: str,
        include_embeddings: bool,
        embedding_batch_size: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._driver = driver
        self._database = str(database or "").strip() or "neo4j"
        self._include_embeddings = bool(include_embeddings)
        self._embedding_batch_size = max(1, int(embedding_batch_size or 1))
        self._lock = threading.Lock()
        self.synced: List[Dict[str, Any]] = []

    def save_faculty_keywords(
        self,
        *,
        faculty_id: int,
        keywords: Dict[str, Any],
        raw_json: Optional[Dict[str, Any]] = None,
        source_model: Optional[str] = None,
    ) -> None:
        _ = raw_json, source_model
        result = sync_faculty_keywords_direct_to_neo4j(
            driver=self._driver,
            database=self._database,
            faculty_id=int(faculty_id),
            keywords=keywords,
            include_embeddings=self._include_embeddings,
            embedding_batch_size=self._embedding_batch_size,
        )
        with self._lock:
            self.synced.append(result)


class GrantKeywordGeneratorV2Neo4jSink(GrantKeywordGeneratorV2):
    def __init__(
        self,
        *,
        driver,
        database: str,
        include_embeddings: bool,
        embedding_batch_size: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._driver = driver
        self._database = str(database or "").strip() or "neo4j"
        self._include_embeddings = bool(include_embeddings)
        self._embedding_batch_size = max(1, int(embedding_batch_size or 1))
        self._lock = threading.Lock()
        self.synced: List[Dict[str, Any]] = []

    def save_grant_keywords(
        self,
        *,
        opportunity_id: str,
        keywords: Dict[str, Any],
        raw_json: Optional[Dict[str, Any]] = None,
        source_model: Optional[str] = None,
    ) -> None:
        _ = raw_json, source_model
        result = sync_grant_keywords_direct_to_neo4j(
            driver=self._driver,
            database=self._database,
            opportunity_id=str(opportunity_id),
            keywords=keywords,
            include_embeddings=self._include_embeddings,
            embedding_batch_size=self._embedding_batch_size,
        )
        with self._lock:
            self.synced.append(result)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate keywords with v2 generators and sync directly to Neo4j (no keyword persistence to Postgres)."
    )
    parser.add_argument("--target", choices=["faculty", "grant", "both"], default="both")
    parser.add_argument("--all", action="store_true", help="Process all rows for selected target(s).")
    parser.add_argument("--faculty-id", type=int, default=0, help="Run one faculty_id.")
    parser.add_argument("--opportunity-id", type=str, default="", help="Run one opportunity_id.")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers for run_all pipelines.")
    parser.add_argument("--max-context-chars", type=int, default=40000)
    parser.add_argument("--max-neo4j-chunks", type=int, default=4000)
    parser.add_argument("--reserve-prompt-chars", type=int, default=3000)
    parser.add_argument("--embedding-batch-size", type=int, default=12)
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--uri", type=str, default="", help="Neo4j URI. Fallback: NEO4J_URI")
    parser.add_argument("--username", type=str, default="", help="Neo4j username. Fallback: NEO4J_USERNAME")
    parser.add_argument("--password", type=str, default="", help="Neo4j password. Fallback: NEO4J_PASSWORD")
    parser.add_argument("--database", type=str, default="", help="Neo4j database. Fallback: NEO4J_DATABASE or neo4j")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    load_dotenv_if_present()
    settings_neo4j = read_neo4j_settings(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    include_embeddings = not bool(args.skip_embeddings)
    embedding_batch_size = max(1, int(args.embedding_batch_size or 1))
    max_workers = max(1, int(args.max_workers or 1))
    max_context_chars = max(1000, int(args.max_context_chars or 40000))
    max_neo4j_chunks = max(10, int(args.max_neo4j_chunks or 4000))
    reserve_prompt_chars = max(0, int(args.reserve_prompt_chars or 0))

    payload: Dict[str, Any] = {
        "scope": {
            "target": str(args.target),
            "all": bool(args.all),
            "faculty_id": int(args.faculty_id or 0),
            "opportunity_id": str(args.opportunity_id or "").strip(),
            "max_workers": max_workers,
            "max_context_chars": max_context_chars,
            "max_neo4j_chunks": max_neo4j_chunks,
            "reserve_prompt_chars": reserve_prompt_chars,
            "include_embeddings": bool(include_embeddings),
            "embedding_batch_size": embedding_batch_size,
            "embedding_model": (settings.bedrock_embed_model_id or "").strip() if include_embeddings else None,
        },
        "faculty": None,
        "grant": None,
    }

    with GraphDatabase.driver(
        settings_neo4j.uri,
        auth=(settings_neo4j.username, settings_neo4j.password),
    ) as driver:
        driver.verify_connectivity()

        if args.target in {"faculty", "both"}:
            fac_gen = FacultyKeywordGeneratorV2Neo4jSink(
                driver=driver,
                database=settings_neo4j.database,
                include_embeddings=include_embeddings,
                embedding_batch_size=embedding_batch_size,
                max_context_chars=max_context_chars,
                max_neo4j_chunks=max_neo4j_chunks,
                reserve_prompt_chars=reserve_prompt_chars,
            )
            if int(args.faculty_id or 0) > 0:
                fac_gen.run_faculty_keyword_pipeline(
                    faculty_id=int(args.faculty_id),
                    max_context_chars=max_context_chars,
                    persist=True,
                )
                fac_summary = {
                    "total": 1,
                    "succeeded": 1 if fac_gen.synced else 0,
                    "failed": 0 if fac_gen.synced else 1,
                    "failed_faculty_ids": [] if fac_gen.synced else [int(args.faculty_id)],
                }
            elif bool(args.all):
                fac_summary = fac_gen.run_all_faculty_keyword_pipelines_parallel(
                    max_workers=max_workers,
                    max_context_chars=max_context_chars,
                    persist=True,
                )
            else:
                faculty_ids = fac_gen.list_all_faculty_ids()
                if not faculty_ids:
                    fac_summary = {"total": 0, "succeeded": 0, "failed": 0, "failed_faculty_ids": []}
                else:
                    one_fid = int(faculty_ids[0])
                    fac_gen.run_faculty_keyword_pipeline(
                        faculty_id=one_fid,
                        max_context_chars=max_context_chars,
                        persist=True,
                    )
                    fac_summary = {
                        "total": 1,
                        "succeeded": 1 if fac_gen.synced else 0,
                        "failed": 0 if fac_gen.synced else 1,
                        "failed_faculty_ids": [] if fac_gen.synced else [one_fid],
                    }
            payload["faculty"] = {
                "summary": fac_summary,
                "synced": fac_gen.synced,
                "totals": {
                    "keywords": sum(int(x.get("counts", {}).get("keywords", 0)) for x in fac_gen.synced),
                    "domain_keywords": sum(int(x.get("counts", {}).get("domain_keywords", 0)) for x in fac_gen.synced),
                    "specialization_keywords": sum(
                        int(x.get("counts", {}).get("specialization_keywords", 0)) for x in fac_gen.synced
                    ),
                    "embedded_keywords": sum(int(x.get("counts", {}).get("embedded_keywords", 0)) for x in fac_gen.synced),
                    "support_chunk_candidates": sum(
                        int(x.get("counts", {}).get("support_chunk_candidates", 0)) for x in fac_gen.synced
                    ),
                    "support_publication_candidates": sum(
                        int(x.get("counts", {}).get("support_publication_candidates", 0)) for x in fac_gen.synced
                    ),
                    "support_chunk_linked": sum(
                        int(x.get("counts", {}).get("support_chunk_linked", 0)) for x in fac_gen.synced
                    ),
                    "support_publication_linked": sum(
                        int(x.get("counts", {}).get("support_publication_linked", 0)) for x in fac_gen.synced
                    ),
                    "support_linked": sum(int(x.get("counts", {}).get("support_linked", 0)) for x in fac_gen.synced),
                },
            }

        if args.target in {"grant", "both"}:
            grant_gen = GrantKeywordGeneratorV2Neo4jSink(
                driver=driver,
                database=settings_neo4j.database,
                include_embeddings=include_embeddings,
                embedding_batch_size=embedding_batch_size,
                max_context_chars=max_context_chars,
                max_neo4j_chunks=max_neo4j_chunks,
                reserve_prompt_chars=reserve_prompt_chars,
            )
            if str(args.opportunity_id or "").strip():
                grant_gen.run_grant_keyword_pipeline(
                    opportunity_id=str(args.opportunity_id).strip(),
                    max_context_chars=max_context_chars,
                    persist=True,
                )
                grant_summary = {
                    "total": 1,
                    "succeeded": 1 if grant_gen.synced else 0,
                    "failed": 0 if grant_gen.synced else 1,
                    "failed_opportunity_ids": [] if grant_gen.synced else [str(args.opportunity_id).strip()],
                }
            elif bool(args.all):
                grant_summary = grant_gen.run_all_grant_keyword_pipelines_parallel(
                    max_workers=max_workers,
                    max_context_chars=max_context_chars,
                    persist=True,
                )
            else:
                opportunity_ids = grant_gen.list_all_opportunity_ids()
                if not opportunity_ids:
                    grant_summary = {"total": 0, "succeeded": 0, "failed": 0, "failed_opportunity_ids": []}
                else:
                    one_oid = str(opportunity_ids[0])
                    grant_gen.run_grant_keyword_pipeline(
                        opportunity_id=one_oid,
                        max_context_chars=max_context_chars,
                        persist=True,
                    )
                    grant_summary = {
                        "total": 1,
                        "succeeded": 1 if grant_gen.synced else 0,
                        "failed": 0 if grant_gen.synced else 1,
                        "failed_opportunity_ids": [] if grant_gen.synced else [one_oid],
                    }
            payload["grant"] = {
                "summary": grant_summary,
                "synced": grant_gen.synced,
                "totals": {
                    "keywords": sum(int(x.get("counts", {}).get("keywords", 0)) for x in grant_gen.synced),
                    "domain_keywords": sum(int(x.get("counts", {}).get("domain_keywords", 0)) for x in grant_gen.synced),
                    "specialization_keywords": sum(
                        int(x.get("counts", {}).get("specialization_keywords", 0)) for x in grant_gen.synced
                    ),
                    "embedded_keywords": sum(int(x.get("counts", {}).get("embedded_keywords", 0)) for x in grant_gen.synced),
                    "support_chunk_candidates": sum(
                        int(x.get("counts", {}).get("support_chunk_candidates", 0)) for x in grant_gen.synced
                    ),
                    "support_chunk_linked": sum(
                        int(x.get("counts", {}).get("support_chunk_linked", 0)) for x in grant_gen.synced
                    ),
                    "support_linked": sum(int(x.get("counts", {}).get("support_linked", 0)) for x in grant_gen.synced),
                },
            }

    if not args.json_only:
        if payload.get("faculty"):
            f = payload["faculty"]
            print("Faculty direct keyword sync complete.")
            print(f"  total                       : {f['summary']['total']}")
            print(f"  succeeded                   : {f['summary']['succeeded']}")
            print(f"  failed                      : {f['summary']['failed']}")
            print(f"  keyword edges               : {f['totals']['keywords']}")
            print(f"  domain keywords             : {f['totals']['domain_keywords']}")
            print(f"  specialization keywords     : {f['totals']['specialization_keywords']}")
            print(f"  embedded keywords           : {f['totals']['embedded_keywords']}")
            print(f"  support chunk linked        : {f['totals']['support_chunk_linked']}")
            print(f"  support publication linked  : {f['totals']['support_publication_linked']}")
            print()
        if payload.get("grant"):
            g = payload["grant"]
            print("Grant direct keyword sync complete.")
            print(f"  total                       : {g['summary']['total']}")
            print(f"  succeeded                   : {g['summary']['succeeded']}")
            print(f"  failed                      : {g['summary']['failed']}")
            print(f"  keyword edges               : {g['totals']['keywords']}")
            print(f"  domain keywords             : {g['totals']['domain_keywords']}")
            print(f"  specialization keywords     : {g['totals']['specialization_keywords']}")
            print(f"  embedded keywords           : {g['totals']['embedded_keywords']}")
            print(f"  support chunk linked        : {g['totals']['support_chunk_linked']}")
            print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
