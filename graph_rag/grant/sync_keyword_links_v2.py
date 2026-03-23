from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from neo4j import GraphDatabase
from sqlalchemy.orm import selectinload

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import settings
from db.db_conn import SessionLocal
from db.models.opportunity import Opportunity
from graph_rag.common import json_ready, load_dotenv_if_present, read_neo4j_settings, safe_text
from graph_rag.grant.sync_neo4j import _clean_text as _clean_keyword_text
from graph_rag.grant.sync_neo4j import _embed_text_values
from utils.embedder import embed_domain_bucket

KEYWORD_RELATIONS: Dict[Tuple[str, str], str] = {
    ("general", "domain"): "HAS_DOMAIN_KEYWORD",
    ("general", "specialization"): "HAS_SPECIALIZATION_KEYWORD",
}

CUSTOM_EDGE_TYPES = [
    "DOMAIN_SUPPORTED_BY_GRANT_CHUNK",
    "GRANT_DOMAIN_HAS_SPECIALIZATION",
    "GRANT_CHUNK_SUPPORTS_SPECIALIZATION",
]


def _norm(value: Any) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def _safe_section(value: Any, *, default: str = "general") -> str:
    _ = value, default
    return "general"


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
        for sid_raw, conf_raw in value.items():
            sid = safe_text(sid_raw)
            if not sid:
                continue
            out[sid] = _coerce_weight(conf_raw, default=0.8)
        return out
    if isinstance(value, list):
        for sid_raw in value:
            sid = safe_text(sid_raw)
            if sid:
                out[sid] = 0.8
    return out


def _merge_snippet_scores(dst: Dict[str, float], src: Dict[str, float]) -> Dict[str, float]:
    out = dict(dst)
    for sid, score in src.items():
        sid_s = str(sid or "").strip()
        if not sid_s:
            continue
        out[sid_s] = max(float(out.get(sid_s, 0.0)), _coerce_weight(score, default=0.8))
    return out


def _parse_keywords_payload(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    domain_acc: Dict[Tuple[str, str], Dict[str, Any]] = {}
    spec_acc: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _upsert_domain(*, section: str, value: Any, weight: Any, snippet_ids: Any) -> None:
        section_s = _safe_section(section)
        value_s = _norm(value)
        if not value_s:
            return
        key = (section_s, value_s)
        row = domain_acc.get(key)
        if row is None:
            row = {
                "section": section_s,
                "bucket": "domain",
                "value": value_s,
                "weight": _coerce_weight(weight, default=0.5),
                "snippet_ids": {},
            }
            domain_acc[key] = row
        else:
            row["weight"] = max(float(row.get("weight") or 0.0), _coerce_weight(weight, default=0.5))
        row["snippet_ids"] = _merge_snippet_scores(dict(row.get("snippet_ids") or {}), _snippet_map(snippet_ids))

    def _upsert_spec(
        *,
        section: str,
        value: Any,
        weight: Any,
        snippet_ids: Any,
        domains: Any,
    ) -> None:
        section_s = _safe_section(section)
        value_s = _norm(value)
        if not value_s:
            return
        key = (section_s, value_s)
        row = spec_acc.get(key)
        if row is None:
            row = {
                "section": section_s,
                "bucket": "specialization",
                "value": value_s,
                "weight": _coerce_weight(weight, default=0.5),
                "snippet_ids": {},
                "domains": {},
            }
            spec_acc[key] = row
        else:
            row["weight"] = max(float(row.get("weight") or 0.0), _coerce_weight(weight, default=0.5))
        row["snippet_ids"] = _merge_snippet_scores(dict(row.get("snippet_ids") or {}), _snippet_map(snippet_ids))

        domain_map = dict(row.get("domains") or {})
        if isinstance(domains, dict):
            for d_raw, rel_raw in domains.items():
                d_name = _norm(d_raw)
                if not d_name:
                    continue
                rel = _coerce_weight(rel_raw, default=0.0)
                if rel <= 0.0:
                    continue
                domain_map[d_name] = max(float(domain_map.get(d_name, 0.0)), rel)
        row["domains"] = domain_map

    if not isinstance(payload, dict):
        return [], []

    # v2 shape: {"domains":[...], "specializations":[...]}
    domains_v2 = payload.get("domains")
    specs_v2 = payload.get("specializations")
    if isinstance(domains_v2, list) or isinstance(specs_v2, list):
        for item in list(domains_v2 or []):
            if isinstance(item, dict):
                _upsert_domain(
                    section=item.get("section") or "general",
                    value=item.get("t") or item.get("value") or item.get("text"),
                    weight=item.get("w") or item.get("weight"),
                    snippet_ids=item.get("snippet_ids"),
                )
            else:
                _upsert_domain(section="general", value=item, weight=None, snippet_ids={})

        for item in list(specs_v2 or []):
            if isinstance(item, dict):
                _upsert_spec(
                    section=item.get("section") or "general",
                    value=item.get("t") or item.get("value") or item.get("text"),
                    weight=item.get("w") or item.get("weight"),
                    snippet_ids=item.get("snippet_ids"),
                    domains=item.get("domains"),
                )
            else:
                _upsert_spec(section="general", value=item, weight=None, snippet_ids={}, domains={})

    # legacy shape: {"research": {...}, "application": {...}}
    for section in ("research", "application"):
        sec = payload.get(section)
        if not isinstance(sec, dict):
            continue
        for item in list(sec.get("domain") or []):
            if isinstance(item, dict):
                _upsert_domain(
                    section="general",
                    value=item.get("t") or item.get("value") or item.get("text"),
                    weight=item.get("w") or item.get("weight"),
                    snippet_ids=item.get("snippet_ids"),
                )
            else:
                _upsert_domain(section="general", value=item, weight=None, snippet_ids={})
        for item in list(sec.get("specialization") or []):
            if isinstance(item, dict):
                _upsert_spec(
                    section="general",
                    value=item.get("t") or item.get("value") or item.get("text"),
                    weight=item.get("w") or item.get("weight"),
                    snippet_ids=item.get("snippet_ids"),
                    domains=item.get("domains"),
                )
            else:
                _upsert_spec(section="general", value=item, weight=None, snippet_ids={}, domains={})

    # Ensure each specialization-linked domain exists.
    for spec in spec_acc.values():
        section_s = _safe_section(spec.get("section"))
        for d_name in dict(spec.get("domains") or {}).keys():
            key = (section_s, _norm(d_name))
            if not key[1]:
                continue
            if key not in domain_acc:
                domain_acc[key] = {
                    "section": section_s,
                    "bucket": "domain",
                    "value": key[1],
                    "weight": 0.5,
                    "snippet_ids": {},
                }

    domains_out = sorted(list(domain_acc.values()), key=lambda x: (str(x.get("section") or ""), str(x.get("value") or "")))
    specs_out = sorted(list(spec_acc.values()), key=lambda x: (str(x.get("section") or ""), str(x.get("value") or "")))
    return domains_out, specs_out


def _attach_keyword_embeddings(
    keyword_rows: List[Dict[str, Any]],
    *,
    embedding_batch_size: int,
) -> None:
    if not keyword_rows:
        return

    texts: List[str] = []
    for row in keyword_rows:
        value = _clean_keyword_text(row.get("value"), max_chars=500)
        if value:
            texts.append(value)
    if not texts:
        return

    embeddings = _embed_text_values(
        texts,
        batch_size=max(1, int(embedding_batch_size or 1)),
        max_workers=4,
    )
    embedding_model = (settings.bedrock_embed_model_id or "").strip()

    for row in keyword_rows:
        value = _clean_keyword_text(row.get("value"), max_chars=500)
        if not value:
            continue
        vec = embeddings.get(value)
        if not vec:
            continue
        row["embedding"] = [float(x) for x in vec]
        row["embedding_model"] = embedding_model


def _load_opportunities(
    *,
    opportunity_id: str,
    sync_all: bool,
    limit: int,
    offset: int,
) -> List[Dict[str, Any]]:
    with SessionLocal() as sess:
        q = (
            sess.query(Opportunity)
            .options(selectinload(Opportunity.keyword))
            .order_by(Opportunity.opportunity_id.asc())
        )
        oid = str(opportunity_id or "").strip()
        if oid:
            q = q.filter(Opportunity.opportunity_id == oid)
        elif bool(sync_all):
            if int(offset or 0) > 0:
                q = q.offset(max(0, int(offset or 0)))
            if int(limit or 0) > 0:
                q = q.limit(max(1, int(limit or 0)))
        else:
            raise ValueError("Set --opportunity-id or --all.")
        rows = q.all()

    out: List[Dict[str, Any]] = []
    for opp in rows:
        opp_id = str(getattr(opp, "opportunity_id", "") or "").strip()
        if not opp_id:
            continue
        payload = (getattr(opp, "keyword", None) and getattr(opp.keyword, "keywords", None)) or {}
        if not isinstance(payload, dict):
            payload = {}
        out.append(
            {
                "opportunity_id": opp_id,
                "keywords": payload,
            }
        )
    return out


def _link_count(records: List[Dict[str, Any]]) -> int:
    if not records:
        return 0
    try:
        return int(records[0].get("linked_count") or 0)
    except Exception:
        return 0


def sync_grant_keyword_links_to_neo4j(
    *,
    driver,
    database: str,
    opportunity_row: Dict[str, Any],
    embedding_batch_size: int = 12,
) -> Dict[str, Any]:
    opportunity_id = str(opportunity_row["opportunity_id"]).strip()
    keywords = dict(opportunity_row.get("keywords") or {})
    domain_rows, spec_rows = _parse_keywords_payload(keywords)

    # Clear uploader-managed edges for this grant scope.
    driver.execute_query(
        """
        MATCH ()-[r]->()
        WHERE r.scope_opportunity_id = $opportunity_id
          AND type(r) IN $types
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id, "types": CUSTOM_EDGE_TYPES},
        database_=database,
    )
    driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(:GrantKeyword)
        WHERE type(r) IN [
            'HAS_RESEARCH_DOMAIN',
            'HAS_RESEARCH_SPECIALIZATION',
            'HAS_APPLICATION_DOMAIN',
            'HAS_APPLICATION_SPECIALIZATION',
            'HAS_DOMAIN_KEYWORD',
            'HAS_SPECIALIZATION_KEYWORD'
        ]
        DELETE r
        """,
        parameters_={"opportunity_id": opportunity_id},
        database_=database,
    )

    keyword_rows = domain_rows + spec_rows
    _attach_keyword_embeddings(
        keyword_rows,
        embedding_batch_size=max(1, int(embedding_batch_size or 1)),
    )

    domain_terms = sorted(
        {
            str(row.get("value") or "").strip()
            for row in domain_rows
            if str(row.get("value") or "").strip()
        }
    )
    domain_text = " | ".join(domain_terms)
    domain_gate_embedding = None
    domain_gate_embedding_model = None
    if domain_terms:
        try:
            vec = embed_domain_bucket(domain_terms)
            if vec:
                domain_gate_embedding = [float(x) for x in vec]
                domain_gate_embedding_model = (settings.bedrock_embed_model_id or "").strip() or None
        except Exception:
            domain_gate_embedding = None
            domain_gate_embedding_model = None

        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            MERGE (dg:GrantDomainGate {opportunity_id: $opportunity_id})
            SET
                dg.section = 'general',
                dg.domain_terms = $domain_terms,
                dg.domain_text = $domain_text,
                dg.domain_count = size($domain_terms),
                dg.embedding = $embedding,
                dg.embedding_model = $embedding_model,
                dg.updated_at = datetime()
            MERGE (g)-[r:HAS_DOMAIN_GATE]->(dg)
            SET r.updated_at = datetime()
            """,
            parameters_={
                "opportunity_id": opportunity_id,
                "domain_terms": domain_terms,
                "domain_text": domain_text,
                "embedding": domain_gate_embedding,
                "embedding_model": domain_gate_embedding_model,
            },
            database_=database,
        )
    else:
        driver.execute_query(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})-[r:HAS_DOMAIN_GATE]->(dg:GrantDomainGate {opportunity_id: $opportunity_id})
            DELETE r
            WITH dg
            DETACH DELETE dg
            """,
            parameters_={"opportunity_id": opportunity_id},
            database_=database,
        )

    for (section, bucket), relation in KEYWORD_RELATIONS.items():
        rows = [
            {
                "value": str(row.get("value") or "").strip(),
                "section": section,
                "bucket": bucket,
                "weight": _coerce_weight(row.get("weight"), default=0.5),
                "embedding": row.get("embedding"),
                "embedding_model": row.get("embedding_model"),
            }
            for row in keyword_rows
            if _safe_section(row.get("section"), default="general") == section
            and _norm(row.get("value")) != ""
            and str(row.get("bucket") or "") == bucket
        ]
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
            parameters_={"opportunity_id": opportunity_id, "rows": rows},
            database_=database,
        )

    # Shared domain nodes across faculty/grant keyword spaces.
    shared_domain_rows = []
    seen_shared_domain = set()
    for row in domain_rows:
        value_raw = str(row.get("value") or "").strip()
        value_norm = _norm(value_raw)
        section = _safe_section(row.get("section"), default="general")
        if not value_norm:
            continue
        key = (value_norm, section)
        if key in seen_shared_domain:
            continue
        seen_shared_domain.add(key)
        shared_domain_rows.append(
            {
                "value": value_raw or value_norm,
                "value_norm": value_norm,
                "section": section,
            }
        )

    if shared_domain_rows:
        driver.execute_query(
            """
            UNWIND $rows AS row
            MERGE (sd:DomainKeywordShared {
                value_norm: row.value_norm,
                section: row.section,
                bucket: 'domain'
            })
            ON CREATE SET
                sd.value = row.value,
                sd.created_at = datetime()
            SET
                sd.updated_at = datetime()
            WITH row, sd
            MATCH (k:GrantKeyword {
                value: row.value,
                section: row.section,
                bucket: 'domain'
            })
            MERGE (k)-[r:MAPS_TO_SHARED_DOMAIN]->(sd)
            SET r.updated_at = datetime()
            """,
            parameters_={"rows": shared_domain_rows},
            database_=database,
        )

    domain_weight_by_key: Dict[Tuple[str, str], float] = {
        (_safe_section(row.get("section"), default="general"), _norm(row.get("value"))): _coerce_weight(row.get("weight"), default=0.5)
        for row in domain_rows
    }

    domain_chunk_rows: List[Dict[str, Any]] = []
    seen_domain_chunk = set()
    for row in domain_rows:
        section = _safe_section(row.get("section"), default="general")
        domain_value = _norm(row.get("value"))
        if not domain_value:
            continue
        domain_weight = _coerce_weight(row.get("weight"), default=0.5)
        for sid, score in dict(row.get("snippet_ids") or {}).items():
            sid_s = str(sid or "").strip()
            if not sid_s:
                continue
            key = (section, domain_value, sid_s)
            if key in seen_domain_chunk:
                continue
            seen_domain_chunk.add(key)
            domain_chunk_rows.append(
                {
                    "domain_value": domain_value,
                    "domain_section": section,
                    "chunk_id": sid_s,
                    "score": _coerce_weight(score, default=0.8),
                    "domain_weight": domain_weight,
                }
            )

    domain_spec_rows: List[Dict[str, Any]] = []
    seen_domain_spec = set()
    chunk_spec_rows: List[Dict[str, Any]] = []
    seen_chunk_spec = set()

    for row in spec_rows:
        spec_section = _safe_section(row.get("section"), default="general")
        spec_value = _norm(row.get("value"))
        if not spec_value:
            continue
        spec_weight = _coerce_weight(row.get("weight"), default=0.5)

        for domain_raw, rel_raw in dict(row.get("domains") or {}).items():
            domain_value = _norm(domain_raw)
            if not domain_value:
                continue
            rel = _coerce_weight(rel_raw, default=0.0)
            if rel <= 0.0:
                continue
            key = (spec_section, domain_value, spec_value)
            if key in seen_domain_spec:
                continue
            seen_domain_spec.add(key)
            domain_spec_rows.append(
                {
                    "domain_section": spec_section,
                    "domain_value": domain_value,
                    "specialization_section": spec_section,
                    "specialization_value": spec_value,
                    "score": rel,
                    "domain_weight": float(domain_weight_by_key.get((spec_section, domain_value), 0.5)),
                    "specialization_weight": spec_weight,
                }
            )

        for sid, score in dict(row.get("snippet_ids") or {}).items():
            sid_s = str(sid or "").strip()
            if not sid_s:
                continue
            key = (spec_section, spec_value, sid_s)
            if key in seen_chunk_spec:
                continue
            seen_chunk_spec.add(key)
            chunk_spec_rows.append(
                {
                    "specialization_section": spec_section,
                    "specialization_value": spec_value,
                    "chunk_id": sid_s,
                    "score": _coerce_weight(score, default=0.8),
                    "specialization_weight": spec_weight,
                }
            )

    domain_chunk_linked = 0
    if domain_chunk_rows:
        records, _, _ = driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (d:GrantKeyword {
                value: row.domain_value,
                section: row.domain_section,
                bucket: 'domain'
            })
            MATCH (c:GrantTextChunk {chunk_id: row.chunk_id})
            WHERE c.opportunity_id = $opportunity_id
            MERGE (d)-[r:DOMAIN_SUPPORTED_BY_GRANT_CHUNK {
                scope_opportunity_id: $opportunity_id,
                domain_value: row.domain_value,
                domain_section: row.domain_section,
                chunk_id: row.chunk_id
            }]->(c)
            SET
                r.score = row.score,
                r.domain_weight = row.domain_weight,
                r.updated_at = datetime()
            RETURN count(r) AS linked_count
            """,
            parameters_={"opportunity_id": opportunity_id, "rows": domain_chunk_rows},
            database_=database,
        )
        domain_chunk_linked = _link_count(records)

    domain_spec_linked = 0
    if domain_spec_rows:
        records, _, _ = driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (d:GrantKeyword {
                value: row.domain_value,
                section: row.domain_section,
                bucket: 'domain'
            })
            MATCH (s:GrantKeyword {
                value: row.specialization_value,
                section: row.specialization_section,
                bucket: 'specialization'
            })
            MERGE (d)-[r:GRANT_DOMAIN_HAS_SPECIALIZATION {
                scope_opportunity_id: $opportunity_id,
                domain_value: row.domain_value,
                domain_section: row.domain_section,
                specialization_value: row.specialization_value,
                specialization_section: row.specialization_section
            }]->(s)
            SET
                r.score = row.score,
                r.domain_weight = row.domain_weight,
                r.specialization_weight = row.specialization_weight,
                r.updated_at = datetime()
            RETURN count(r) AS linked_count
            """,
            parameters_={"opportunity_id": opportunity_id, "rows": domain_spec_rows},
            database_=database,
        )
        domain_spec_linked = _link_count(records)

    chunk_spec_linked = 0
    if chunk_spec_rows:
        records, _, _ = driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (c:GrantTextChunk {chunk_id: row.chunk_id})
            WHERE c.opportunity_id = $opportunity_id
            MATCH (s:GrantKeyword {
                value: row.specialization_value,
                section: row.specialization_section,
                bucket: 'specialization'
            })
            MERGE (c)-[r:GRANT_CHUNK_SUPPORTS_SPECIALIZATION {
                scope_opportunity_id: $opportunity_id,
                chunk_id: row.chunk_id,
                specialization_value: row.specialization_value,
                specialization_section: row.specialization_section
            }]->(s)
            SET
                r.score = row.score,
                r.specialization_weight = row.specialization_weight,
                r.updated_at = datetime()
            RETURN count(r) AS linked_count
            """,
            parameters_={"opportunity_id": opportunity_id, "rows": chunk_spec_rows},
            database_=database,
        )
        chunk_spec_linked = _link_count(records)

    return {
        "opportunity_id": opportunity_id,
        "counts": {
            "domain_keywords": len(domain_rows),
            "specialization_keywords": len(spec_rows),
            "embedded_keywords": len([x for x in keyword_rows if x.get("embedding")]),
            "domain_gate_node": 1 if domain_terms else 0,
            "domain_gate_terms": len(domain_terms),
            "domain_gate_embedded": 1 if domain_gate_embedding else 0,
            "domain_chunk_candidates": len(domain_chunk_rows),
            "domain_specialization_candidates": len(domain_spec_rows),
            "chunk_specialization_candidates": len(chunk_spec_rows),
            "domain_chunk_linked": int(domain_chunk_linked),
            "domain_specialization_linked": int(domain_spec_linked),
            "chunk_specialization_linked": int(chunk_spec_linked),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sync grant keyword linkage edges from relational keywords JSON into Neo4j. "
            "Creates: domain->chunk, domain->specialization, chunk->specialization."
        )
    )
    parser.add_argument("--opportunity-id", type=str, default="", help="Sync one grant by opportunity_id.")
    parser.add_argument("--all", action="store_true", help="Sync all grant rows.")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows when using --all (0 = no limit).")
    parser.add_argument("--offset", type=int, default=0, help="Offset rows when using --all.")
    parser.add_argument("--embedding-batch-size", type=int, default=12, help="Embedding batch size for keyword nodes.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on first error.")
    parser.add_argument("--json-only", action="store_true", help="Print only JSON payload.")
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

    rows = _load_opportunities(
        opportunity_id=str(args.opportunity_id or "").strip(),
        sync_all=bool(args.all),
        limit=max(0, int(args.limit or 0)),
        offset=max(0, int(args.offset or 0)),
    )
    if not rows:
        raise RuntimeError("No grant rows found for requested scope.")

    synced: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    embedding_batch_size = max(1, int(args.embedding_batch_size or 1))

    with GraphDatabase.driver(
        settings_neo4j.uri,
        auth=(settings_neo4j.username, settings_neo4j.password),
    ) as driver:
        driver.verify_connectivity()
        for row in rows:
            try:
                synced.append(
                    sync_grant_keyword_links_to_neo4j(
                        driver=driver,
                        database=settings_neo4j.database,
                        opportunity_row=row,
                        embedding_batch_size=embedding_batch_size,
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "opportunity_id": str(row.get("opportunity_id") or ""),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if bool(args.stop_on_error):
                    break

    totals = {
        "grants_synced": len(synced),
        "grants_failed": len(errors),
        "domain_keywords": sum(int(x.get("counts", {}).get("domain_keywords", 0)) for x in synced),
        "specialization_keywords": sum(int(x.get("counts", {}).get("specialization_keywords", 0)) for x in synced),
        "embedded_keywords": sum(int(x.get("counts", {}).get("embedded_keywords", 0)) for x in synced),
        "domain_gate_nodes": sum(int(x.get("counts", {}).get("domain_gate_node", 0)) for x in synced),
        "domain_gate_terms": sum(int(x.get("counts", {}).get("domain_gate_terms", 0)) for x in synced),
        "domain_gate_embedded": sum(int(x.get("counts", {}).get("domain_gate_embedded", 0)) for x in synced),
        "domain_chunk_candidates": sum(int(x.get("counts", {}).get("domain_chunk_candidates", 0)) for x in synced),
        "domain_specialization_candidates": sum(int(x.get("counts", {}).get("domain_specialization_candidates", 0)) for x in synced),
        "chunk_specialization_candidates": sum(int(x.get("counts", {}).get("chunk_specialization_candidates", 0)) for x in synced),
        "domain_chunk_linked": sum(int(x.get("counts", {}).get("domain_chunk_linked", 0)) for x in synced),
        "domain_specialization_linked": sum(int(x.get("counts", {}).get("domain_specialization_linked", 0)) for x in synced),
        "chunk_specialization_linked": sum(int(x.get("counts", {}).get("chunk_specialization_linked", 0)) for x in synced),
    }

    payload = {
        "scope": {
            "opportunity_id": str(args.opportunity_id or "").strip(),
            "all": bool(args.all),
            "limit": max(0, int(args.limit or 0)),
            "offset": max(0, int(args.offset or 0)),
            "embedding_batch_size": embedding_batch_size,
        },
        "totals": totals,
        "synced": synced,
        "errors": errors,
    }

    if not bool(args.json_only):
        print("Grant keyword link sync complete.")
        print(f"  synced grants                    : {totals['grants_synced']}")
        print(f"  failed grants                    : {totals['grants_failed']}")
        print(f"  domain keywords                  : {totals['domain_keywords']}")
        print(f"  specialization keywords          : {totals['specialization_keywords']}")
        print(f"  embedded keywords                : {totals['embedded_keywords']}")
        print(f"  domain gate nodes                : {totals['domain_gate_nodes']}")
        print(f"  domain gate terms                : {totals['domain_gate_terms']}")
        print(f"  domain gate embedded             : {totals['domain_gate_embedded']}")
        print(f"  domain->chunk linked             : {totals['domain_chunk_linked']}")
        print(f"  domain->specialization linked    : {totals['domain_specialization_linked']}")
        print(f"  chunk->specialization linked     : {totals['chunk_specialization_linked']}")
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
