from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from pydantic import BaseModel

try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:  # pragma: no cover - optional
    ChatPromptTemplate = None  # type: ignore[assignment]

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from graph_rag.common import json_ready, load_dotenv_if_present, read_neo4j_settings
from graph_rag.matching.retrieve_grants_by_domain_gate_spec_coverage import (
    retrieve_grants_by_domain_gate_spec_coverage,
)
from logging_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_unit_float(value: Any, *, default: float = 0.0) -> float:
    parsed = _safe_float(value, default=default)
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


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


def _clip_text(text: Any, *, max_chars: int = 320) -> str:
    s = _clean_text(text)
    cap = max(80, int(max_chars))
    if len(s) <= cap:
        return s
    return s[:cap].rstrip() + "..."


def _unique_rows(rows: List[Dict[str, Any]], *, key_fields: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in list(rows or []):
        key = tuple(_clean_text(row.get(k)).lower() for k in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


class _LLMJustificationOut(BaseModel):
    full_justification: str = ""


LLM_JUSTIFICATION_PROMPT = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grant-match justification writer. "
                "Use ONLY the provided source JSON. Do not fabricate facts. "
                "Return strict JSON only.",
            ),
            (
                "human",
                "Write a full, detailed, evidence-grounded top-1 match justification.\n"
                "Source bundle JSON:\n{source_json}\n\n"
                "Requirements:\n"
                "- full_justification: multi-paragraph narrative.\n"
                "- explain how each required capability area is matched by faculty capabilities.\n"
                "- explicitly mention related faculty publications tied to those specialization matches.\n"
                "- cite supporting evidence from both faculty-side and grant-side chunks.\n"
                "- avoid showing numeric scores, weights, similarities, or confidence values.\n"
                "- do not quote specialization keyword strings verbatim; synthesize broader capability themes.\n"
                "- avoid rigid bullet-list style and write fluent, varied prose.\n"
                "- write natural prose in this style: since the faculty has done X, they can perform Y for this grant.\n",
            ),
        ]
    )
    if ChatPromptTemplate is not None
    else None
)


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _build_llm_chain(*, llm_model: str):
    if ChatPromptTemplate is None:
        reason = "langchain_core_not_available: ChatPromptTemplate import failed"
        logger.warning("LLM chain unavailable: %s", reason)
        return None, reason
    if LLM_JUSTIFICATION_PROMPT is None:
        reason = "prompt_unavailable: LLM_JUSTIFICATION_PROMPT was not initialized"
        logger.warning("LLM chain unavailable: %s", reason)
        return None, reason
    try:
        from config import get_llm_client
    except Exception as e:
        reason = f"config_import_error: {type(e).__name__}: {e}"
        logger.warning("LLM chain unavailable: %s", reason)
        return None, reason

    try:
        model_id = _clean_text(llm_model) or None
        llm = get_llm_client(model_id).build()
    except Exception as e:
        reason = f"llm_client_build_error: {type(e).__name__}: {e}"
        logger.warning("LLM chain unavailable: %s", reason)
        return None, reason

    try:
        chain = LLM_JUSTIFICATION_PROMPT | llm.with_structured_output(_LLMJustificationOut)
        logger.info("Built LLM justification chain (model_override=%s)", _clean_text(llm_model) or "<default>")
        return chain, ""
    except Exception as e:
        reason = f"structured_output_error: {type(e).__name__}: {e}"
        logger.warning("LLM chain unavailable: %s", reason)
        return None, reason


def _trim_evidence_rows(rows: List[Dict[str, Any]], *, limit: int = 8) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in list(rows or [])[: max(1, int(limit))]:
        out.append(dict(row))
    return out


def _build_llm_source_bundle(
    *,
    faculty_identity: Dict[str, Any],
    top_grant: Dict[str, Any],
    matched_domains: List[Dict[str, Any]],
    matched_specs: List[Dict[str, Any]],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    trimmed_specs: List[Dict[str, Any]] = []
    for row in list(matched_specs or [])[:10]:
        trimmed_specs.append(
            {
                "grant_keyword_value": _clean_text(row.get("grant_keyword_value")),
                "grant_keyword_section": _clean_text(row.get("grant_keyword_section")).lower() or "general",
                "best_faculty_keyword_value": _clean_text(row.get("best_faculty_keyword_value")),
                "best_faculty_keyword_section": _clean_text(row.get("best_faculty_keyword_section")).lower() or "general",
            }
        )

    return {
        "faculty": {
            "faculty_id": _safe_int(faculty_identity.get("faculty_id"), default=0),
            "name": _clean_text(faculty_identity.get("name")),
            "email": _clean_text(faculty_identity.get("email")).lower(),
        },
        "top_match": {
            "opportunity_id": _clean_text(top_grant.get("opportunity_id")),
            "opportunity_title": _clean_text(top_grant.get("opportunity_title")),
            "agency_name": _clean_text(top_grant.get("agency_name")),
        },
        "matched_domain_keywords": _trim_evidence_rows(list(matched_domains or []), limit=10),
        "matched_specialization_keywords": trimmed_specs,
        "supporting_evidence": {
            "faculty_domain_chunks": _trim_evidence_rows(list((evidence or {}).get("faculty_domain_chunks") or []), limit=8),
            "faculty_spec_chunks": _trim_evidence_rows(
                list(((evidence or {}).get("faculty_spec_support") or {}).get("chunks") or []),
                limit=8,
            ),
            "faculty_spec_publications": _trim_evidence_rows(
                list(((evidence or {}).get("faculty_spec_support") or {}).get("publications") or []),
                limit=8,
            ),
            "grant_domain_chunks": _trim_evidence_rows(list((evidence or {}).get("grant_domain_chunks") or []), limit=8),
            "grant_spec_chunks": _trim_evidence_rows(list((evidence or {}).get("grant_spec_chunks") or []), limit=8),
        },
    }


def _generate_llm_justification(
    *,
    llm_model: str,
    faculty_identity: Dict[str, Any],
    top_grant: Dict[str, Any],
    matched_domains: List[Dict[str, Any]],
    matched_specs: List[Dict[str, Any]],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    chain, chain_error = _build_llm_chain(llm_model=llm_model)
    if chain is None:
        logger.warning("LLM justification skipped: %s", _clean_text(chain_error))
        return {
            "ok": False,
            "reason": "llm_chain_unavailable",
            "detail": _clean_text(chain_error),
            "model_id": _clean_text(llm_model),
        }

    source_bundle = _build_llm_source_bundle(
        faculty_identity=faculty_identity,
        top_grant=top_grant,
        matched_domains=matched_domains,
        matched_specs=matched_specs,
        evidence=evidence,
    )
    try:
        out = chain.invoke({"source_json": json.dumps(source_bundle, ensure_ascii=False)})
        row = out.model_dump() if hasattr(out, "model_dump") else dict(out or {})
        full_justification = _clean_text(row.get("full_justification")) or _clean_text(row.get("one_paragraph"))
        if not full_justification:
            logger.warning("LLM justification returned empty full_justification")
            return {"ok": False, "reason": "llm_empty_output", "source_bundle": source_bundle}
        full_justification = _sanitize_specialization_mentions(full_justification, matched_specs)
        logger.info("LLM full justification generated")
        return {
            "ok": True,
            "source_bundle": source_bundle,
            "full_justification": full_justification,
            "model_id": _clean_text(llm_model),
        }
    except Exception as e:
        logger.exception("LLM justification invoke failed")
        return {
            "ok": False,
            "reason": f"{type(e).__name__}: {e}",
            "source_bundle": source_bundle,
        }


def _resolve_faculty_identity(
    *,
    driver,
    database: str,
    faculty_id: Optional[int],
    faculty_email: str,
) -> Dict[str, Any]:
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty)
        WHERE
            ($faculty_id IS NULL OR f.faculty_id = $faculty_id)
            AND ($faculty_email = '' OR toLower(f.email) = $faculty_email)
        RETURN
            f.faculty_id AS faculty_id,
            toLower(coalesce(f.email, '')) AS email,
            coalesce(f.name, '') AS name
        LIMIT 1
        """,
        parameters_={
            "faculty_id": faculty_id,
            "faculty_email": _clean_text(faculty_email).lower(),
        },
        database_=database,
    )
    if not records:
        return {
            "faculty_id": int(faculty_id or 0),
            "email": _clean_text(faculty_email).lower(),
            "name": "",
        }
    row = dict(records[0] or {})
    return {
        "faculty_id": int(row.get("faculty_id") or 0),
        "email": _clean_text(row.get("email")).lower(),
        "name": _clean_text(row.get("name")),
    }


def _collect_matched_domains(top_grant: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in list(top_grant.get("matched_domains") or []):
        if not isinstance(row, dict):
            continue
        domain_norm = _clean_text(row.get("domain_norm")).lower() or _clean_text(row.get("domain")).lower()
        domain_value = _clean_text(row.get("domain")) or domain_norm
        if not domain_norm:
            continue
        out.append(
            {
                "domain": domain_value,
                "domain_norm": domain_norm,
                "faculty_domain_weight": _safe_unit_float(row.get("faculty_domain_weight"), default=0.0),
                "grant_domain_weight": _safe_unit_float(row.get("grant_domain_weight"), default=0.0),
                "pair_weight": _safe_unit_float(row.get("pair_weight"), default=0.0),
            }
        )
    out.sort(
        key=lambda x: (
            float(x.get("pair_weight") or 0.0),
            float(x.get("faculty_domain_weight") or 0.0),
            float(x.get("grant_domain_weight") or 0.0),
            _clean_text(x.get("domain")),
        ),
        reverse=True,
    )
    return _unique_rows(out, key_fields=["domain_norm"])


def _collect_matched_specs(top_grant: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in list(top_grant.get("grant_spec_coverage_details") or []):
        if not isinstance(row, dict):
            continue
        grant_spec_value = _clean_text(row.get("grant_keyword_value"))
        grant_spec_section = _clean_text(row.get("grant_keyword_section")).lower() or "general"
        if not grant_spec_value:
            continue
        best_pair = dict(row.get("best_pair") or {})
        out.append(
            {
                "grant_keyword_value": grant_spec_value,
                "grant_keyword_section": grant_spec_section,
                "grant_spec_weight": _safe_unit_float(row.get("grant_spec_weight"), default=0.0),
                "matched_faculty_spec_count": int(row.get("matched_faculty_spec_count") or 0),
                "spec_score_sum": _safe_float(row.get("spec_score_sum"), default=0.0),
                "best_pair_score": _safe_float(row.get("best_pair_score"), default=0.0),
                "best_faculty_keyword_value": _clean_text(best_pair.get("faculty_keyword_value")),
                "best_faculty_keyword_section": _clean_text(best_pair.get("faculty_keyword_section")).lower() or "general",
                "best_similarity": _safe_unit_float(best_pair.get("similarity"), default=0.0),
                "best_domain_overlap": _safe_unit_float(best_pair.get("domain_overlap"), default=0.0),
                "matched_faculty_specs": list(row.get("matched_faculty_specs") or []),
            }
        )
    out.sort(
        key=lambda x: (
            float(x.get("best_pair_score") or 0.0),
            int(x.get("matched_faculty_spec_count") or 0),
            float(x.get("spec_score_sum") or 0.0),
            _clean_text(x.get("grant_keyword_value")),
        ),
        reverse=True,
    )
    return _unique_rows(out, key_fields=["grant_keyword_value", "grant_keyword_section"])


def _fetch_faculty_domain_chunks(
    *,
    driver,
    database: str,
    faculty_id: int,
    domain_norms: List[str],
    evidence_limit: int,
) -> List[Dict[str, Any]]:
    if not domain_norms or faculty_id <= 0:
        return []
    records, _, _ = driver.execute_query(
        """
        UNWIND $domain_norms AS dn
        MATCH (f:Faculty {faculty_id: $faculty_id})-[fr:HAS_DOMAIN_KEYWORD]->(fd:FacultyKeyword {bucket: 'domain'})
        MATCH (fd)-[:MAPS_TO_SHARED_DOMAIN]->(sd:DomainKeywordShared {bucket: 'domain'})
        WHERE sd.value_norm = dn
        MATCH (fd)-[r:DOMAIN_SUPPORTED_BY_FACULTY_CHUNK {scope_faculty_id: $faculty_id}]->(c:FacultyTextChunk)
        RETURN
            dn AS domain_norm,
            coalesce(fd.value, dn) AS faculty_domain_keyword,
            coalesce(fr.weight, 0.0) AS faculty_domain_weight,
            c.chunk_id AS chunk_id,
            c.source_type AS source_type,
            c.source_url AS source_url,
            c.text AS text,
            coalesce(r.score, 0.0) AS support_score
        ORDER BY support_score DESC, chunk_id ASC
        LIMIT $evidence_limit
        """,
        parameters_={
            "faculty_id": int(faculty_id),
            "domain_norms": [_clean_text(x).lower() for x in domain_norms if _clean_text(x)],
            "evidence_limit": int(evidence_limit),
        },
        database_=database,
    )
    out: List[Dict[str, Any]] = []
    for raw in records:
        row = dict(raw or {})
        out.append(
            {
                "domain_norm": _clean_text(row.get("domain_norm")).lower(),
                "faculty_domain_keyword": _clean_text(row.get("faculty_domain_keyword")),
                "faculty_domain_weight": _safe_unit_float(row.get("faculty_domain_weight"), default=0.0),
                "chunk_id": _clean_text(row.get("chunk_id")),
                "source_type": _clean_text(row.get("source_type")),
                "source_url": _clean_text(row.get("source_url")),
                "support_score": _safe_unit_float(row.get("support_score"), default=0.0),
                "text": _clip_text(row.get("text")),
            }
        )
    return _unique_rows(out, key_fields=["domain_norm", "chunk_id"])


def _fetch_faculty_spec_support(
    *,
    driver,
    database: str,
    faculty_id: int,
    faculty_spec_values: List[str],
    evidence_limit: int,
) -> Dict[str, List[Dict[str, Any]]]:
    if not faculty_spec_values or faculty_id <= 0:
        return {"chunks": [], "publications": []}

    spec_values = [_clean_text(x).lower() for x in faculty_spec_values if _clean_text(x)]
    if not spec_values:
        return {"chunks": [], "publications": []}

    chunk_records, _, _ = driver.execute_query(
        """
        MATCH (s:FacultyKeyword {bucket: 'specialization'})
        WHERE toLower(coalesce(s.value, '')) IN $spec_values
        MATCH (c:FacultyTextChunk)-[r:FACULTY_CHUNK_SUPPORTS_SPECIALIZATION {scope_faculty_id: $faculty_id}]->(s)
        RETURN
            s.value AS faculty_keyword_value,
            toLower(coalesce(s.section, 'general')) AS faculty_keyword_section,
            c.chunk_id AS chunk_id,
            c.source_type AS source_type,
            c.source_url AS source_url,
            c.text AS text,
            coalesce(r.score, 0.0) AS support_score
        ORDER BY support_score DESC, chunk_id ASC
        LIMIT $evidence_limit
        """,
        parameters_={
            "faculty_id": int(faculty_id),
            "spec_values": spec_values,
            "evidence_limit": int(evidence_limit),
        },
        database_=database,
    )
    pub_records, _, _ = driver.execute_query(
        """
        MATCH (s:FacultyKeyword {bucket: 'specialization'})
        WHERE toLower(coalesce(s.value, '')) IN $spec_values
        MATCH (p:FacultyPublication)-[r:FACULTY_PUBLICATION_SUPPORTS_SPECIALIZATION {scope_faculty_id: $faculty_id}]->(s)
        RETURN
            s.value AS faculty_keyword_value,
            toLower(coalesce(s.section, 'general')) AS faculty_keyword_section,
            p.publication_id AS publication_id,
            p.title AS title,
            p.abstract AS abstract,
            p.year AS year,
            coalesce(r.score, 0.0) AS support_score
        ORDER BY support_score DESC, publication_id ASC
        LIMIT $evidence_limit
        """,
        parameters_={
            "faculty_id": int(faculty_id),
            "spec_values": spec_values,
            "evidence_limit": int(evidence_limit),
        },
        database_=database,
    )

    chunk_out: List[Dict[str, Any]] = []
    for raw in chunk_records:
        row = dict(raw or {})
        chunk_out.append(
            {
                "faculty_keyword_value": _clean_text(row.get("faculty_keyword_value")),
                "faculty_keyword_section": _clean_text(row.get("faculty_keyword_section")).lower() or "general",
                "chunk_id": _clean_text(row.get("chunk_id")),
                "source_type": _clean_text(row.get("source_type")),
                "source_url": _clean_text(row.get("source_url")),
                "support_score": _safe_unit_float(row.get("support_score"), default=0.0),
                "text": _clip_text(row.get("text")),
            }
        )

    pub_out: List[Dict[str, Any]] = []
    for raw in pub_records:
        row = dict(raw or {})
        pub_out.append(
            {
                "faculty_keyword_value": _clean_text(row.get("faculty_keyword_value")),
                "faculty_keyword_section": _clean_text(row.get("faculty_keyword_section")).lower() or "general",
                "publication_id": int(row.get("publication_id") or 0),
                "title": _clean_text(row.get("title")),
                "year": int(row.get("year") or 0) if row.get("year") is not None else 0,
                "support_score": _safe_unit_float(row.get("support_score"), default=0.0),
                "abstract": _clip_text(row.get("abstract")),
            }
        )

    return {
        "chunks": _unique_rows(chunk_out, key_fields=["faculty_keyword_value", "chunk_id"]),
        "publications": _unique_rows(pub_out, key_fields=["faculty_keyword_value", "publication_id"]),
    }


def _fetch_grant_domain_chunks(
    *,
    driver,
    database: str,
    opportunity_id: str,
    domain_norms: List[str],
    evidence_limit: int,
) -> List[Dict[str, Any]]:
    oid = _clean_text(opportunity_id)
    if not oid or not domain_norms:
        return []
    records, _, _ = driver.execute_query(
        """
        UNWIND $domain_norms AS dn
        MATCH (g:Grant {opportunity_id: $opportunity_id})-[gr:HAS_DOMAIN_KEYWORD]->(gd:GrantKeyword {bucket: 'domain'})
        MATCH (gd)-[:MAPS_TO_SHARED_DOMAIN]->(sd:DomainKeywordShared {bucket: 'domain'})
        WHERE sd.value_norm = dn
        MATCH (gd)-[r:DOMAIN_SUPPORTED_BY_GRANT_CHUNK {scope_opportunity_id: $opportunity_id}]->(c:GrantTextChunk)
        RETURN
            dn AS domain_norm,
            coalesce(gd.value, dn) AS grant_domain_keyword,
            coalesce(gr.weight, 0.0) AS grant_domain_weight,
            c.chunk_id AS chunk_id,
            c.source_type AS source_type,
            c.source_url AS source_url,
            c.source_title AS source_title,
            c.text AS text,
            coalesce(r.score, 0.0) AS support_score
        ORDER BY support_score DESC, chunk_id ASC
        LIMIT $evidence_limit
        """,
        parameters_={
            "opportunity_id": oid,
            "domain_norms": [_clean_text(x).lower() for x in domain_norms if _clean_text(x)],
            "evidence_limit": int(evidence_limit),
        },
        database_=database,
    )
    out: List[Dict[str, Any]] = []
    for raw in records:
        row = dict(raw or {})
        out.append(
            {
                "domain_norm": _clean_text(row.get("domain_norm")).lower(),
                "grant_domain_keyword": _clean_text(row.get("grant_domain_keyword")),
                "grant_domain_weight": _safe_unit_float(row.get("grant_domain_weight"), default=0.0),
                "chunk_id": _clean_text(row.get("chunk_id")),
                "source_type": _clean_text(row.get("source_type")),
                "source_url": _clean_text(row.get("source_url")),
                "source_title": _clean_text(row.get("source_title")),
                "support_score": _safe_unit_float(row.get("support_score"), default=0.0),
                "text": _clip_text(row.get("text")),
            }
        )
    return _unique_rows(out, key_fields=["domain_norm", "chunk_id"])


def _fetch_grant_spec_chunks(
    *,
    driver,
    database: str,
    opportunity_id: str,
    grant_spec_values: List[str],
    evidence_limit: int,
) -> List[Dict[str, Any]]:
    oid = _clean_text(opportunity_id)
    if not oid or not grant_spec_values:
        return []
    spec_values = [_clean_text(x).lower() for x in grant_spec_values if _clean_text(x)]
    if not spec_values:
        return []

    records, _, _ = driver.execute_query(
        """
        MATCH (g:Grant {opportunity_id: $opportunity_id})
        MATCH (s:GrantKeyword {bucket: 'specialization'})
        WHERE toLower(coalesce(s.value, '')) IN $spec_values
        MATCH (c:GrantTextChunk)-[r:GRANT_CHUNK_SUPPORTS_SPECIALIZATION {scope_opportunity_id: $opportunity_id}]->(s)
        RETURN
            s.value AS grant_keyword_value,
            toLower(coalesce(s.section, 'general')) AS grant_keyword_section,
            c.chunk_id AS chunk_id,
            c.source_type AS source_type,
            c.source_url AS source_url,
            c.source_title AS source_title,
            c.text AS text,
            coalesce(r.score, 0.0) AS support_score
        ORDER BY support_score DESC, chunk_id ASC
        LIMIT $evidence_limit
        """,
        parameters_={
            "opportunity_id": oid,
            "spec_values": spec_values,
            "evidence_limit": int(evidence_limit),
        },
        database_=database,
    )

    out: List[Dict[str, Any]] = []
    for raw in records:
        row = dict(raw or {})
        out.append(
            {
                "grant_keyword_value": _clean_text(row.get("grant_keyword_value")),
                "grant_keyword_section": _clean_text(row.get("grant_keyword_section")).lower() or "general",
                "chunk_id": _clean_text(row.get("chunk_id")),
                "source_type": _clean_text(row.get("source_type")),
                "source_url": _clean_text(row.get("source_url")),
                "source_title": _clean_text(row.get("source_title")),
                "support_score": _safe_unit_float(row.get("support_score"), default=0.0),
                "text": _clip_text(row.get("text")),
            }
        )
    return _unique_rows(out, key_fields=["grant_keyword_value", "chunk_id"])


def _abstract_requirement_phrase(chunk_text: str, *, blocked_terms: Optional[List[str]] = None) -> str:
    raw = _clean_text(chunk_text).lower()
    if not raw:
        return "a core technical workstream in the opportunity"
    for term in list(blocked_terms or []):
        t = _clean_text(term)
        if len(t) < 6:
            continue
        raw = re.sub(re.escape(t.lower()), "related capability", raw, flags=re.IGNORECASE)

    raw = re.sub(
        r"^(the project|the award|the program|this grant|the opportunity)\s+(requires|expects|seeks|supports)\s+",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[^a-z0-9\s]", " ", raw)
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return "a core technical workstream in the opportunity"
    phrase = " ".join(tokens[:14]).strip()
    if len(tokens) > 14:
        phrase += "..."
    return phrase


def _sanitize_specialization_mentions(text: str, matched_specs: List[Dict[str, Any]]) -> str:
    out = str(text or "")
    terms: List[str] = []
    for row in matched_specs or []:
        g = _clean_text(row.get("grant_keyword_value"))
        f = _clean_text(row.get("best_faculty_keyword_value"))
        if len(g) >= 6:
            terms.append(g)
        if len(f) >= 6:
            terms.append(f)

    seen = set()
    unique_terms: List[str] = []
    for t in terms:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        unique_terms.append(t)

    for term in sorted(unique_terms, key=len, reverse=True):
        out = re.sub(re.escape(term), "related capability", out, flags=re.IGNORECASE)

    out = re.sub(r"(related capability[\s,;:]*){2,}", "related capability ", out, flags=re.IGNORECASE)

    normalized_lines: List[str] = []
    for ln in out.split("\n"):
        normalized_lines.append(re.sub(r"[ \t]+", " ", ln).strip())

    while normalized_lines and not normalized_lines[-1]:
        normalized_lines.pop()
    return "\n".join(normalized_lines)


def _compose_justification_text(
    *,
    faculty_identity: Dict[str, Any],
    top_grant: Dict[str, Any],
    matched_domains: List[Dict[str, Any]],
    matched_specs: List[Dict[str, Any]],
    evidence: Dict[str, Any],
) -> str:
    faculty_name = _clean_text(faculty_identity.get("name")) or _clean_text(faculty_identity.get("email")) or str(
        faculty_identity.get("faculty_id") or ""
    )
    title = _clean_text(top_grant.get("opportunity_title"))
    oid = _clean_text(top_grant.get("opportunity_id"))
    agency = _clean_text(top_grant.get("agency_name"))

    domain_names = [_clean_text(x.get("domain")) for x in matched_domains if _clean_text(x.get("domain"))]
    domain_text = ", ".join(domain_names) if domain_names else "the core technical areas in the opportunity"

    faculty_spec_support = dict((evidence or {}).get("faculty_spec_support") or {})
    spec_pubs = list(faculty_spec_support.get("publications") or [])
    grant_spec_chunks = list((evidence or {}).get("grant_spec_chunks") or [])

    pubs_by_faculty_spec: Dict[str, List[Dict[str, Any]]] = {}
    for row in spec_pubs:
        key = _clean_text(row.get("faculty_keyword_value")).lower()
        if not key:
            continue
        pubs_by_faculty_spec.setdefault(key, []).append(dict(row))

    grant_chunks_by_spec: Dict[str, List[Dict[str, Any]]] = {}
    for row in grant_spec_chunks:
        key = _clean_text(row.get("grant_keyword_value")).lower()
        if not key:
            continue
        grant_chunks_by_spec.setdefault(key, []).append(dict(row))

    intro = (
        f"{faculty_name} is a compelling fit for '{title}'"
        f"{f' from {agency}' if agency else ''} ({oid}). "
        f"The opportunity sits in {domain_text}, and the faculty profile shows sustained prior work in those same areas. "
        f"This match is practical, not superficial: since {faculty_name} has already delivered closely related work, "
        f"they are positioned to execute the required grant tasks."
    )

    coverage_paragraphs: List[str] = []
    seen_pub_titles = set()
    all_pub_titles: List[str] = []
    for row in list(matched_specs or [])[:8]:
        grant_spec = _clean_text(row.get("grant_keyword_value"))
        faculty_spec = _clean_text(row.get("best_faculty_keyword_value"))

        grant_rows = grant_chunks_by_spec.get(grant_spec.lower(), []) if grant_spec else []
        requirement_phrase = (
            _abstract_requirement_phrase(
                _clean_text((grant_rows[0] or {}).get("text")),
                blocked_terms=[grant_spec, faculty_spec],
            )
            if grant_rows
            else ""
        )
        if requirement_phrase:
            lead = (
                f"A required workstream for this grant focuses on {requirement_phrase}. "
                f"{faculty_name}'s prior portfolio indicates they can carry this part of the project forward."
            )
        else:
            lead = (
                f"Another required workstream aligns with capabilities already present in {faculty_name}'s prior work, "
                f"which supports immediate execution."
            )

        pub_rows = pubs_by_faculty_spec.get(_clean_text(faculty_spec).lower(), []) if faculty_spec else []
        pub_bits: List[str] = []
        for p in pub_rows:
            title_txt = _clean_text(p.get("title"))
            if not title_txt:
                continue
            year = int(p.get("year") or 0)
            pub_key = title_txt.lower()
            if pub_key in seen_pub_titles:
                continue
            seen_pub_titles.add(pub_key)
            all_pub_titles.append(title_txt)
            pub_bits.append(f"{title_txt}{f' ({year})' if year > 0 else ''}")
            if len(pub_bits) >= 2:
                break

        publications_line = ""
        if pub_bits:
            publications_line = " Evidence of prior execution appears in publications such as " + "; ".join(pub_bits) + "."

        source_line = ""
        if grant_rows:
            src_title = _clean_text(grant_rows[0].get("source_title"))
            if src_title:
                source_line = f" The requirement is clearly described in the grant source '{src_title}'."
            else:
                source_line = " The requirement is clearly described in the grant source material."

        coverage_paragraphs.append(f"{lead}{publications_line}{source_line}")

    if not coverage_paragraphs:
        coverage_paragraphs.append(
            f"The matched evidence suggests {faculty_name}'s previous work covers the operational needs of this opportunity, "
            "with support from both faculty-side records and grant-side source chunks."
        )

    publications_paragraph = ""
    if all_pub_titles:
        publications_paragraph = (
            "Taken together, the related publication record includes "
            + "; ".join(all_pub_titles)
            + ", reinforcing that the required work can be executed with existing expertise."
        )

    conclusion = (
        f"The overall case is straightforward: because {faculty_name} has already produced closely aligned research outputs, "
        f"they can perform the grant's key workstreams with a realistic delivery path and evidence-backed readiness."
    )

    sections: List[str] = [intro, "\n\n".join(coverage_paragraphs)]
    if publications_paragraph:
        sections.append(publications_paragraph)
    sections.append(conclusion)
    return _sanitize_specialization_mentions("\n\n".join([x for x in sections if _clean_text(x)]), matched_specs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve grants via domain_gate_spec_coverage and justify top-1 match with matched "
            "domain/spec keywords plus faculty+grant supporting chunk evidence."
        )
    )
    parser.add_argument("--faculty-id", type=int, default=103, help="Faculty ID filter.")
    parser.add_argument("--faculty-email", type=str, default="", help="Faculty email filter.")
    parser.add_argument("--min-domain-weight", type=float, default=0.6, help="Stage1 domain gate threshold.")
    parser.add_argument("--candidate-limit", type=int, default=20, help="Stage1 candidate count.")
    parser.add_argument("--top-k", type=int, default=20, help="Final grants to retrieve before choosing top-1.")
    parser.add_argument("--include-closed", action="store_true", help="Include closed grants.")
    parser.add_argument("--min-pair-similarity", type=float, default=0.0, help="Minimum FACULTY_SPEC_MATCHES_GRANT_SPEC similarity.")
    parser.add_argument("--pairs-per-spec", type=int, default=10, help="Matched faculty specs retained per grant spec keyword.")
    parser.add_argument("--evidence-limit", type=int, default=12, help="Per-side evidence limit.")
    parser.add_argument("--llm-model", type=str, default="", help="Optional LLM model id override for justification generation.")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM justification and use template fallback only.")
    parser.add_argument("--json-only", action="store_true", help="Print JSON only.")

    parser.add_argument("--uri", type=str, default="", help="Neo4j URI. Fallback: NEO4J_URI")
    parser.add_argument("--username", type=str, default="", help="Neo4j username. Fallback: NEO4J_USERNAME")
    parser.add_argument("--password", type=str, default="", help="Neo4j password. Fallback: NEO4J_PASSWORD")
    parser.add_argument("--database", type=str, default="", help="Neo4j database. Fallback: NEO4J_DATABASE or neo4j")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    load_dotenv_if_present()
    logger.info(
        "Starting top1 justification run (faculty_id=%s, faculty_email=%s, top_k=%d, disable_llm=%s)",
        str(int(args.faculty_id or 0) or ""),
        _clean_text(args.faculty_email).lower(),
        int(args.top_k or 20),
        str(bool(args.disable_llm)),
    )

    fid_raw = int(args.faculty_id or 0)
    faculty_id = fid_raw if fid_raw > 0 else None
    faculty_email = _clean_text(args.faculty_email).lower()
    evidence_limit = _safe_limit(args.evidence_limit, default=12, minimum=1, maximum=200)

    retrieval = retrieve_grants_by_domain_gate_spec_coverage(
        faculty_id=faculty_id,
        faculty_email=faculty_email,
        min_domain_weight=float(args.min_domain_weight),
        candidate_limit=int(args.candidate_limit or 20),
        top_k=int(args.top_k or 20),
        include_closed=bool(args.include_closed),
        min_pair_similarity=float(args.min_pair_similarity),
        pairs_per_spec=int(args.pairs_per_spec or 10),
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    grants = list(retrieval.get("grants") or [])
    top_grant = dict(grants[0] or {}) if grants else {}
    matched_domains = _collect_matched_domains(top_grant)
    matched_specs = _collect_matched_specs(top_grant)

    settings = read_neo4j_settings(uri=args.uri, username=args.username, password=args.password, database=args.database)
    faculty_identity: Dict[str, Any] = {
        "faculty_id": int(faculty_id or 0),
        "email": faculty_email,
        "name": "",
    }
    evidence: Dict[str, Any] = {
        "faculty_domain_chunks": [],
        "faculty_spec_support": {"chunks": [], "publications": []},
        "grant_domain_chunks": [],
        "grant_spec_chunks": [],
    }

    if top_grant:
        domain_norms = [_clean_text(x.get("domain_norm")).lower() for x in matched_domains if _clean_text(x.get("domain_norm"))]
        faculty_spec_values = [
            _clean_text(x.get("best_faculty_keyword_value")).lower()
            for x in matched_specs
            if _clean_text(x.get("best_faculty_keyword_value"))
        ]
        grant_spec_values = [
            _clean_text(x.get("grant_keyword_value")).lower()
            for x in matched_specs
            if _clean_text(x.get("grant_keyword_value"))
        ]

        with GraphDatabase.driver(settings.uri, auth=(settings.username, settings.password)) as driver:
            driver.verify_connectivity()
            faculty_identity = _resolve_faculty_identity(
                driver=driver,
                database=settings.database,
                faculty_id=faculty_id,
                faculty_email=faculty_email,
            )
            resolved_faculty_id = int(faculty_identity.get("faculty_id") or 0)
            opportunity_id = _clean_text(top_grant.get("opportunity_id"))

            evidence["faculty_domain_chunks"] = _fetch_faculty_domain_chunks(
                driver=driver,
                database=settings.database,
                faculty_id=resolved_faculty_id,
                domain_norms=domain_norms,
                evidence_limit=evidence_limit,
            )
            evidence["faculty_spec_support"] = _fetch_faculty_spec_support(
                driver=driver,
                database=settings.database,
                faculty_id=resolved_faculty_id,
                faculty_spec_values=faculty_spec_values,
                evidence_limit=evidence_limit,
            )
            evidence["grant_domain_chunks"] = _fetch_grant_domain_chunks(
                driver=driver,
                database=settings.database,
                opportunity_id=opportunity_id,
                domain_norms=domain_norms,
                evidence_limit=evidence_limit,
            )
            evidence["grant_spec_chunks"] = _fetch_grant_spec_chunks(
                driver=driver,
                database=settings.database,
                opportunity_id=opportunity_id,
                grant_spec_values=grant_spec_values,
                evidence_limit=evidence_limit,
            )
        logger.info(
            "Collected evidence for top grant %s (matched_domains=%d, matched_specs=%d)",
            _clean_text(top_grant.get("opportunity_id")),
            len(matched_domains),
            len(matched_specs),
        )
    else:
        logger.warning("No top grant found from retrieval results")

    justification_text = ""
    llm_justification: Dict[str, Any] = {}
    justification_source = "none"
    if top_grant:
        template_justification = _compose_justification_text(
            faculty_identity=faculty_identity,
            top_grant=top_grant,
            matched_domains=matched_domains,
            matched_specs=matched_specs,
            evidence=evidence,
        )
        justification_text = template_justification
        justification_source = "template"

        if not bool(args.disable_llm):
            llm_justification = _generate_llm_justification(
                llm_model=_clean_text(args.llm_model),
                faculty_identity=faculty_identity,
                top_grant=top_grant,
                matched_domains=matched_domains,
                matched_specs=matched_specs,
                evidence=evidence,
            )
            llm_full = _clean_text(llm_justification.get("full_justification"))
            if bool(llm_justification.get("ok")) and llm_full:
                justification_text = llm_full
                justification_source = "llm"
            else:
                justification_source = "template_fallback"
                logger.warning(
                    "LLM fallback to template (reason=%s, detail=%s)",
                    _clean_text(llm_justification.get("reason")),
                    _clean_text(llm_justification.get("detail")),
                )
                if not llm_justification:
                    llm_justification = {"ok": False, "reason": "llm_not_invoked"}

    payload = {
        "params": {
            "faculty_id": faculty_id,
            "faculty_email": faculty_email,
            "min_domain_weight": float(args.min_domain_weight),
            "candidate_limit": int(args.candidate_limit or 20),
            "top_k": int(args.top_k or 20),
            "include_closed": bool(args.include_closed),
            "min_pair_similarity": float(args.min_pair_similarity),
            "pairs_per_spec": int(args.pairs_per_spec or 10),
            "evidence_limit": evidence_limit,
            "llm_model": _clean_text(args.llm_model),
            "disable_llm": bool(args.disable_llm),
        },
        "retrieval_totals": dict(retrieval.get("totals") or {}),
        "faculty": faculty_identity,
        "top_match": {
            "opportunity_id": _clean_text(top_grant.get("opportunity_id")),
            "opportunity_title": _clean_text(top_grant.get("opportunity_title")),
            "agency_name": _clean_text(top_grant.get("agency_name")),
            "final_score": _safe_float(top_grant.get("final_score"), default=0.0),
            "domain_rank_score": _safe_float(top_grant.get("domain_rank_score"), default=0.0),
            "covered_grant_spec_keywords": int(top_grant.get("covered_grant_spec_keywords") or 0),
            "total_grant_spec_keywords": int(top_grant.get("total_grant_spec_keywords") or 0),
            "grant_spec_coverage_ratio": _safe_float(top_grant.get("grant_spec_coverage_ratio"), default=0.0),
            "grant_spec_coverage_sum": _safe_float(top_grant.get("grant_spec_coverage_sum"), default=0.0),
        },
        "matched_domain_keywords": matched_domains,
        "matched_specialization_keywords": matched_specs,
        "supporting_evidence": evidence,
        "justification_source": justification_source,
        "llm_justification": llm_justification,
        "justification": justification_text,
    }
    logger.info(
        "Finished top1 justification run (source=%s, top_match=%s)",
        justification_source,
        _clean_text((payload.get("top_match") or {}).get("opportunity_id")),
    )

    if not bool(args.json_only):
        top = payload.get("top_match") or {}
        print("Top-1 justification built from domain gate + spec coverage retrieval.")
        print(f"  faculty           : {payload.get('faculty', {}).get('name') or payload.get('faculty', {}).get('email')}")
        print(f"  top grant         : {top.get('opportunity_title')} ({top.get('opportunity_id')})")
        print(f"  matched domains   : {len(list(payload.get('matched_domain_keywords') or []))}")
        print(f"  matched specs     : {len(list(payload.get('matched_specialization_keywords') or []))}")
        print(f"  faculty chunks    : {len(list((payload.get('supporting_evidence') or {}).get('faculty_domain_chunks') or []))}")
        print(
            "  faculty spec chunks/publications: "
            f"{len(list((((payload.get('supporting_evidence') or {}).get('faculty_spec_support') or {}).get('chunks') or [])))} / "
            f"{len(list((((payload.get('supporting_evidence') or {}).get('faculty_spec_support') or {}).get('publications') or [])))}"
        )
        print(f"  grant chunks      : {len(list((payload.get('supporting_evidence') or {}).get('grant_domain_chunks') or []))}")
        print(f"  grant spec chunks : {len(list((payload.get('supporting_evidence') or {}).get('grant_spec_chunks') or []))}")
        print(f"  justification src : {payload.get('justification_source')}")
        print()
        print("Justification:")
        print(justification_text or "(no top match found)")
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
