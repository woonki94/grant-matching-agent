from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from neo4j import GraphDatabase

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from graph_rag.common import json_ready, load_dotenv_if_present, read_neo4j_settings, safe_text
from graph_rag.faculty.sync_neo4j import (
    KEYWORD_RELATIONS,
    _clean_text,
    _embed_text_values,
    _keyword_rows,
    _load_faculties,
    _safe_limit,
)
from config import settings
from utils.embedder import embed_domain_bucket


def _keyword_chunk_support_rows(*, keyword_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in keyword_rows or []:
        if str(row.get("bucket") or "") != "specialization":
            continue
        snippet_ids = row.get("snippet_ids")
        if not isinstance(snippet_ids, dict):
            continue
        for sid_raw, conf_raw in snippet_ids.items():
            chunk_id = safe_text(sid_raw)
            if not chunk_id:
                continue
            try:
                score = float(conf_raw)
            except Exception:
                score = 0.8
            score = max(0.0, min(1.0, score))
            key = (
                str(row.get("value") or "").lower(),
                str(row.get("section") or "").lower(),
                str(row.get("bucket") or "").lower(),
                chunk_id,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "keyword_value": row.get("value"),
                    "keyword_section": row.get("section"),
                    "keyword_bucket": row.get("bucket"),
                    "chunk_id": chunk_id,
                    "score": score,
                }
            )
    return out


def _support_rows_from_keyword_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        return out
    seen = set()
    for section in ("research", "application"):
        sec = payload.get(section)
        if not isinstance(sec, dict):
            continue
        specs = sec.get("specialization") or []
        if not isinstance(specs, list):
            continue
        for item in specs:
            if not isinstance(item, dict):
                continue
            keyword = safe_text(item.get("t") or item.get("text"))
            if not keyword:
                continue
            snippet_ids = item.get("snippet_ids")
            if not isinstance(snippet_ids, dict):
                continue
            for sid_raw, conf_raw in snippet_ids.items():
                chunk_id = safe_text(sid_raw)
                if not chunk_id:
                    continue
                try:
                    score = float(conf_raw)
                except Exception:
                    score = 0.8
                score = max(0.0, min(1.0, score))
                key = (section, "specialization", keyword.lower(), chunk_id)
                if key in seen:
                    continue
                seen.add(key)
                row = {
                    "keyword_value": keyword,
                    "keyword_section": section,
                    "keyword_bucket": "specialization",
                    "score": score,
                }
                if chunk_id.startswith("pub|"):
                    parts = chunk_id.split("|")
                    if len(parts) >= 4:
                        try:
                            row["publication_id"] = int(parts[3])
                        except Exception:
                            row["chunk_id"] = chunk_id
                    else:
                        row["chunk_id"] = chunk_id
                else:
                    row["chunk_id"] = chunk_id
                out.append(row)
    return out


def sync_faculty_keywords_to_neo4j(
    *,
    driver,
    settings_neo4j,
    fac,
    include_embeddings: bool,
    embedding_batch_size: int,
) -> Dict[str, Any]:
    faculty_id = int(getattr(fac, "faculty_id", 0) or 0)
    email = str(getattr(fac, "email", "") or "").strip().lower()
    if faculty_id <= 0 or not email:
        raise ValueError("Faculty must have faculty_id and email.")

    keyword_payload = (getattr(fac, "keyword", None) and getattr(fac.keyword, "keywords", None)) or {}
    raw_keyword_rows = _keyword_rows(keyword_payload)

    # Domain: one aggregated node per section (research/application).
    # Specialization: one node per specialization phrase.
    domain_terms_by_section: Dict[str, List[str]] = {"research": [], "application": []}
    domain_seen = set()
    keyword_rows: List[Dict[str, Any]] = []

    for row in raw_keyword_rows:
        section = str(row.get("section") or "").strip().lower()
        bucket = str(row.get("bucket") or "").strip().lower()
        if section not in {"research", "application"}:
            continue
        if bucket == "domain":
            value = safe_text(row.get("value"))
            if not value:
                continue
            key = (section, value.lower())
            if key in domain_seen:
                continue
            domain_seen.add(key)
            domain_terms_by_section[section].append(value)
            continue
        if bucket == "specialization":
            keyword_rows.append(dict(row))

    section_to_relation = {
        "research": "HAS_RESEARCH_DOMAIN",
        "application": "HAS_APPLICATION_DOMAIN",
    }
    for section in ("research", "application"):
        terms = list(domain_terms_by_section.get(section) or [])
        if not terms:
            continue
        keyword_rows.append(
            {
                "value": " | ".join(terms),
                "section": section,
                "bucket": "domain",
                "relation": section_to_relation[section],
                "weight": None,
                "embedding": None,
                "embedding_model": None,
                "domain_terms": terms,
            }
        )

    if include_embeddings and keyword_rows:
        embedding_model = (settings.bedrock_embed_model_id or "").strip()

        # Specialization embeddings: one per specialization node text.
        spec_texts = [
            str(x.get("value") or "")
            for x in keyword_rows
            if str(x.get("bucket") or "") == "specialization"
        ]
        spec_embeddings = _embed_text_values(spec_texts, batch_size=embedding_batch_size)
        for row in keyword_rows:
            if str(row.get("bucket") or "") != "specialization":
                continue
            key_text = _clean_text(row.get("value"))
            emb = spec_embeddings.get(key_text or "")
            if emb:
                row["embedding"] = emb
                row["embedding_model"] = embedding_model

        # Domain embeddings: section-separated bucket embeddings (research/application).
        for row in keyword_rows:
            if str(row.get("bucket") or "") != "domain":
                continue
            terms = [str(x).strip() for x in (row.get("domain_terms") or []) if str(x).strip()]
            vec = embed_domain_bucket(terms)
            if vec:
                row["embedding"] = [float(x) for x in vec]
                row["embedding_model"] = embedding_model

    support_rows = _support_rows_from_keyword_payload(keyword_payload)
    if not support_rows:
        support_rows = _keyword_chunk_support_rows(keyword_rows=keyword_rows)

    # Refresh keyword edges for this faculty scope.
    driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})-[r]->(:FacultyKeyword)
        WHERE type(r) IN [
            'HAS_RESEARCH_DOMAIN',
            'HAS_RESEARCH_SPECIALIZATION',
            'HAS_APPLICATION_DOMAIN',
            'HAS_APPLICATION_SPECIALIZATION'
        ]
        DELETE r
        """,
        parameters_={"email": email},
        database_=settings_neo4j.database,
    )
    driver.execute_query(
        """
        MATCH (:FacultyKeyword)-[r:SUPPORTED_BY_FACULTY_CHUNK {scope_faculty_id: $faculty_id}]->(:FacultyTextChunk)
        DELETE r
        """,
        parameters_={"faculty_id": faculty_id},
        database_=settings_neo4j.database,
    )
    driver.execute_query(
        """
        MATCH (:FacultyKeyword)-[r:SUPPORTED_BY_FACULTY_PUBLICATION {scope_faculty_id: $faculty_id}]->(:FacultyPublication)
        DELETE r
        """,
        parameters_={"faculty_id": faculty_id},
        database_=settings_neo4j.database,
    )

    for relation in KEYWORD_RELATIONS:
        rows = [item for item in keyword_rows if item.get("relation") == relation]
        if not rows:
            continue
        driver.execute_query(
            f"""
            MATCH (f:Faculty {{email: $email}})
            UNWIND $rows AS row
            MERGE (k:FacultyKeyword {{
                value: row.value,
                section: row.section,
                bucket: row.bucket
            }})
            SET
                k.domain_terms = row.domain_terms,
                k.embedding = row.embedding,
                k.embedding_model = row.embedding_model,
                k.updated_at = datetime()
            MERGE (f)-[r:{relation}]->(k)
            SET
                r.weight = row.weight,
                r.updated_at = datetime()
            """,
            parameters_={"email": email, "rows": rows},
            database_=settings_neo4j.database,
        )

    support_chunk_rows = [x for x in support_rows if x.get("chunk_id")]
    support_pub_rows = [x for x in support_rows if x.get("publication_id") is not None]

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
            parameters_={"faculty_id": faculty_id, "rows": support_chunk_rows},
            database_=settings_neo4j.database,
        )
        if records:
            try:
                linked_chunk_rows = int(records[0].get("linked_count") or 0)
            except Exception:
                linked_chunk_rows = 0

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
            parameters_={"faculty_id": faculty_id, "rows": support_pub_rows},
            database_=settings_neo4j.database,
        )
        if records:
            try:
                linked_pub_rows = int(records[0].get("linked_count") or 0)
            except Exception:
                linked_pub_rows = 0

    return {
        "faculty_id": faculty_id,
        "email": email,
        "counts": {
            "keywords": len(keyword_rows),
            "domain_nodes": len([x for x in keyword_rows if str(x.get("bucket") or "") == "domain"]),
            "specialization_keywords": len([x for x in keyword_rows if str(x.get("bucket") or "") == "specialization"]),
            "embedded_keywords": len([x for x in keyword_rows if x.get("embedding")]),
            "embedded_domain_nodes": len(
                [x for x in keyword_rows if str(x.get("bucket") or "") == "domain" and x.get("embedding")]
            ),
            "embedded_specialization_keywords": len(
                [x for x in keyword_rows if str(x.get("bucket") or "") == "specialization" and x.get("embedding")]
            ),
            "support_candidates": len(support_rows),
            "support_chunk_candidates": len(support_chunk_rows),
            "support_publication_candidates": len(support_pub_rows),
            "support_chunk_linked": int(linked_chunk_rows),
            "support_publication_linked": int(linked_pub_rows),
            "support_linked": int(linked_chunk_rows + linked_pub_rows),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync faculty keywords + snippet-based keyword->chunk links into Neo4j.")
    parser.add_argument("--email", type=str, default="", help="Sync one faculty by email.")
    parser.add_argument("--all", action="store_true", help="Sync all faculty rows.")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows when using --all (0 = no limit).")
    parser.add_argument("--offset", type=int, default=0, help="Offset rows when using --all.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on first faculty sync error.")
    parser.add_argument("--embedding-batch-size", type=int, default=12, help="Embedding batch size for keyword nodes.")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip keyword embeddings on FacultyKeyword nodes.")
    parser.add_argument("--json-only", action="store_true", help="Print only JSON output.")
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

    rows = _load_faculties(
        email=args.email,
        sync_all=bool(args.all),
        limit=max(0, int(args.limit or 0)),
        offset=max(0, int(args.offset or 0)),
    )
    if not rows:
        raise RuntimeError("No faculty rows found for requested sync scope.")

    synced: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    include_embeddings = not bool(args.skip_embeddings)
    embedding_batch_size = _safe_limit(args.embedding_batch_size, default=12, minimum=1, maximum=128)

    with GraphDatabase.driver(
        settings_neo4j.uri,
        auth=(settings_neo4j.username, settings_neo4j.password),
    ) as driver:
        driver.verify_connectivity()

        for fac in rows:
            email = str(getattr(fac, "email", "") or "").strip().lower()
            try:
                synced.append(
                    sync_faculty_keywords_to_neo4j(
                        driver=driver,
                        settings_neo4j=settings_neo4j,
                        fac=fac,
                        include_embeddings=include_embeddings,
                        embedding_batch_size=embedding_batch_size,
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "faculty_id": int(getattr(fac, "faculty_id", 0) or 0),
                        "email": email,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if bool(args.stop_on_error):
                    break

    totals = {
        "faculties_synced": len(synced),
        "faculties_failed": len(errors),
        "keywords": sum(int(item.get("counts", {}).get("keywords", 0)) for item in synced),
        "domain_nodes": sum(int(item.get("counts", {}).get("domain_nodes", 0)) for item in synced),
        "specialization_keywords": sum(int(item.get("counts", {}).get("specialization_keywords", 0)) for item in synced),
        "embedded_keywords": sum(int(item.get("counts", {}).get("embedded_keywords", 0)) for item in synced),
        "embedded_domain_nodes": sum(int(item.get("counts", {}).get("embedded_domain_nodes", 0)) for item in synced),
        "embedded_specialization_keywords": sum(
            int(item.get("counts", {}).get("embedded_specialization_keywords", 0)) for item in synced
        ),
        "support_candidates": sum(int(item.get("counts", {}).get("support_candidates", 0)) for item in synced),
        "support_chunk_candidates": sum(int(item.get("counts", {}).get("support_chunk_candidates", 0)) for item in synced),
        "support_publication_candidates": sum(int(item.get("counts", {}).get("support_publication_candidates", 0)) for item in synced),
        "support_chunk_linked": sum(int(item.get("counts", {}).get("support_chunk_linked", 0)) for item in synced),
        "support_publication_linked": sum(int(item.get("counts", {}).get("support_publication_linked", 0)) for item in synced),
        "support_linked": sum(int(item.get("counts", {}).get("support_linked", 0)) for item in synced),
    }

    payload = {
        "scope": {
            "email": str(args.email or "").strip().lower(),
            "all": bool(args.all),
            "limit": max(0, int(args.limit or 0)),
            "offset": max(0, int(args.offset or 0)),
            "include_embeddings": bool(include_embeddings),
            "embedding_model": (settings.bedrock_embed_model_id or "").strip() if include_embeddings else None,
        },
        "totals": totals,
        "synced": synced,
        "errors": errors,
    }

    if not args.json_only:
        print("Faculty keyword sync complete.")
        print(f"  synced faculties             : {totals['faculties_synced']}")
        print(f"  failed faculties             : {totals['faculties_failed']}")
        print(f"  keyword edges                : {totals['keywords']}")
        print(f"  domain nodes                 : {totals['domain_nodes']}")
        print(f"  specialization keywords      : {totals['specialization_keywords']}")
        print(f"  embedded keyword nodes       : {totals['embedded_keywords']}")
        print(f"  embedded domain nodes        : {totals['embedded_domain_nodes']}")
        print(f"  embedded specialization kws  : {totals['embedded_specialization_keywords']}")
        print(f"  keyword->support candidates  : {totals['support_candidates']}")
        print(f"  -> chunk candidates          : {totals['support_chunk_candidates']}")
        print(f"  -> publication candidates    : {totals['support_publication_candidates']}")
        print(f"  keyword->chunk linked edges  : {totals['support_chunk_linked']}")
        print(f"  keyword->publication linked  : {totals['support_publication_linked']}")
        print(f"  keyword->support linked all  : {totals['support_linked']}")
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
