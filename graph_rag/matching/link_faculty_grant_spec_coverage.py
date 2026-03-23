from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from neo4j import GraphDatabase

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from graph_rag.common import json_ready, load_dotenv_if_present, read_neo4j_settings


EDGE_LABEL = "FACULTY_GRANT_SPEC_COVERAGE"
EDGE_METHOD = "domain_gate_passed_domain_spec_chunk_coverage_v1"
EDGE_F2G_SPEC = "FACULTY_COVERS_GRANT_SPEC_KEYWORD"
EDGE_G2F_SPEC = "GRANT_COVERS_FACULTY_SPEC_KEYWORD"


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
    if math.isnan(parsed) or math.isinf(parsed):
        return float(default)
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _coerce_vector(value: Any) -> List[float]:
    if not isinstance(value, (list, tuple)):
        return []
    out: List[float] = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            return []
    return out


def _cosine_vec(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for idx in range(len(a)):
        x = float(a[idx])
        y = float(b[idx])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    sim = dot / ((math.sqrt(na) * math.sqrt(nb)) + 1e-9)
    return _safe_unit_float(sim, default=0.0)


def _mean_vector(vectors: Sequence[Sequence[float]]) -> List[float]:
    vv = [list(v) for v in vectors if v]
    if not vv:
        return []
    dim = len(vv[0])
    keep = [v for v in vv if len(v) == dim]
    if not keep:
        return []
    out = [0.0] * dim
    for v in keep:
        for i in range(dim):
            out[i] += float(v[i])
    inv = 1.0 / float(len(keep))
    return [x * inv for x in out]


def _json_compact(value: Any) -> str:
    return json.dumps(json_ready(value), ensure_ascii=True, separators=(",", ":"))


def _load_faculties(
    *,
    driver,
    database: str,
    faculty_id: Optional[int],
    faculty_email: str,
    all_faculties: bool,
    limit: int,
    offset: int,
) -> List[Dict[str, Any]]:
    if all_faculties:
        records, _, _ = driver.execute_query(
            """
            MATCH (f:Faculty)
            WHERE f.faculty_id IS NOT NULL
            RETURN
                toInteger(f.faculty_id) AS faculty_id,
                toLower(coalesce(f.email, '')) AS faculty_email
            ORDER BY faculty_id ASC
            SKIP $offset
            LIMIT $limit
            """,
            parameters_={
                "offset": int(max(0, offset)),
                "limit": int(max(1, limit)),
            },
            database_=database,
        )
        return [dict(r or {}) for r in records]

    if faculty_id is None and not faculty_email:
        raise ValueError("Provide --faculty-id or --faculty-email, or use --all-faculties.")

    records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty)
        WHERE
            ($faculty_id IS NULL OR f.faculty_id = $faculty_id)
            AND ($faculty_email = '' OR toLower(f.email) = $faculty_email)
        RETURN
            toInteger(f.faculty_id) AS faculty_id,
            toLower(coalesce(f.email, '')) AS faculty_email
        ORDER BY faculty_id ASC
        """,
        parameters_={
            "faculty_id": int(faculty_id) if faculty_id is not None else None,
            "faculty_email": _clean_text(faculty_email).lower(),
        },
        database_=database,
    )
    return [dict(r or {}) for r in records]


def _fetch_domain_gate_candidates(
    *,
    driver,
    database: str,
    faculty_id: int,
    min_domain_weight: float,
    top_candidates: int,
) -> List[Dict[str, Any]]:
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty {faculty_id: $faculty_id})-[fr:HAS_DOMAIN_KEYWORD]->(fk:FacultyKeyword {bucket: 'domain'})
        WHERE coalesce(fr.weight, 0.0) > $min_domain_weight

        MATCH (fk)-[:MAPS_TO_SHARED_DOMAIN]->(d:DomainKeywordShared {bucket: 'domain'})

        MATCH (gk:GrantKeyword {bucket: 'domain'})-[:MAPS_TO_SHARED_DOMAIN]->(d)
        MATCH (g:Grant)-[gr:HAS_DOMAIN_KEYWORD]->(gk)
        WHERE coalesce(gr.weight, 0.0) > $min_domain_weight

        WITH
            g,
            coalesce(d.value_norm, toLower(coalesce(d.value, ''))) AS domain_norm,
            (coalesce(fr.weight, 0.0) * coalesce(gr.weight, 0.0)) AS pair_weight
        WITH
            g,
            collect({
                domain_norm: domain_norm,
                pair_weight: pair_weight
            }) AS passed_domain_pairs,
            sum(pair_weight) AS candidate_rank_score,
            count(DISTINCT domain_norm) AS passed_domain_count
        RETURN
            toString(g.opportunity_id) AS opportunity_id,
            coalesce(g.opportunity_title, g.title, '') AS opportunity_title,
            candidate_rank_score,
            passed_domain_count,
            passed_domain_pairs
        ORDER BY candidate_rank_score DESC, passed_domain_count DESC, opportunity_id ASC
        LIMIT $top_candidates
        """,
        parameters_={
            "faculty_id": int(faculty_id),
            "min_domain_weight": float(min_domain_weight),
            "top_candidates": int(top_candidates),
        },
        database_=database,
    )
    return [dict(r or {}) for r in records]


def _fetch_faculty_specs(
    *,
    driver,
    database: str,
    faculty_id: int,
    faculty_email: str,
    min_domain_weight: float,
) -> List[Dict[str, Any]]:
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty {faculty_id: $faculty_id})-[hs:HAS_SPECIALIZATION_KEYWORD]->(fs:FacultyKeyword {bucket: 'specialization'})
        WHERE fs.embedding IS NOT NULL

        OPTIONAL MATCH (f)-[hd:HAS_DOMAIN_KEYWORD]->(fd:FacultyKeyword {bucket: 'domain'})-[ds:FACULTY_DOMAIN_HAS_SPECIALIZATION]->(fs)
        WHERE
            ds.scope_faculty_id = $faculty_id
            AND coalesce(hd.weight, 0.0) > $min_domain_weight
        OPTIONAL MATCH (fd)-[:MAPS_TO_SHARED_DOMAIN]->(sd:DomainKeywordShared {bucket: 'domain'})

        RETURN
            coalesce(fs.value, '') AS keyword_value,
            coalesce(hs.weight, 0.0) AS keyword_weight,
            fs.embedding AS embedding,
            collect(DISTINCT {
                domain_norm: coalesce(sd.value_norm, toLower(coalesce(fd.value, ''))),
                domain_weight: coalesce(ds.domain_weight, hd.weight, 0.0)
            }) AS domain_links
        ORDER BY keyword_weight DESC, keyword_value ASC
        """,
        parameters_={
            "faculty_id": int(faculty_id),
            "faculty_email": _clean_text(faculty_email).lower(),
            "min_domain_weight": float(min_domain_weight),
        },
        database_=database,
    )

    out: List[Dict[str, Any]] = []
    for row in records:
        item = dict(row or {})
        keyword_value = _clean_text(item.get("keyword_value"))
        emb = _coerce_vector(item.get("embedding"))
        if not keyword_value or not emb:
            continue
        domains: Set[str] = set()
        domain_weights: Dict[str, float] = {}
        for link in list(item.get("domain_links") or []):
            if not isinstance(link, dict):
                continue
            dom = _clean_text(link.get("domain_norm")).lower()
            if not dom:
                continue
            w = _safe_unit_float(link.get("domain_weight"), default=0.0)
            domains.add(dom)
            if w > float(domain_weights.get(dom, 0.0)):
                domain_weights[dom] = w
        out.append(
            {
                "keyword_value": keyword_value,
                "keyword_weight": _safe_unit_float(item.get("keyword_weight"), default=0.0),
                "embedding": emb,
                "domain_norms": domains,
                "domain_weights": domain_weights,
            }
        )
    return out


def _fetch_faculty_domain_evidence_vectors(
    *,
    driver,
    database: str,
    faculty_id: int,
    min_domain_weight: float,
) -> Dict[str, Dict[str, Any]]:
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty {faculty_id: $faculty_id})-[hd:HAS_DOMAIN_KEYWORD]->(fd:FacultyKeyword {bucket: 'domain'})
        WHERE coalesce(hd.weight, 0.0) > $min_domain_weight

        OPTIONAL MATCH (fd)-[:MAPS_TO_SHARED_DOMAIN]->(sd:DomainKeywordShared {bucket: 'domain'})

        OPTIONAL MATCH (fd)-[dc:DOMAIN_SUPPORTED_BY_FACULTY_CHUNK]->(c:FacultyTextChunk)
        WHERE dc.scope_faculty_id = $faculty_id

        RETURN
            coalesce(sd.value_norm, toLower(coalesce(fd.value, ''))) AS domain_norm,
            max(coalesce(hd.weight, 0.0)) AS domain_weight,
            collect(DISTINCT c.embedding) AS chunk_embeddings
        """,
        parameters_={
            "faculty_id": int(faculty_id),
            "min_domain_weight": float(min_domain_weight),
        },
        database_=database,
    )

    out: Dict[str, Dict[str, Any]] = {}
    for row in records:
        item = dict(row or {})
        domain_norm = _clean_text(item.get("domain_norm")).lower()
        if not domain_norm:
            continue
        entry = out.setdefault(
            domain_norm,
            {
                "vectors": [],
                "domain_weight": _safe_unit_float(item.get("domain_weight"), default=0.0),
            },
        )
        d_w = _safe_unit_float(item.get("domain_weight"), default=0.0)
        if d_w > float(entry.get("domain_weight") or 0.0):
            entry["domain_weight"] = d_w
        vectors = list(entry.get("vectors") or [])
        for raw in list(item.get("chunk_embeddings") or []):
            vec = _coerce_vector(raw)
            if vec:
                vectors.append(vec)
        entry["vectors"] = vectors
    return out


def _fetch_grant_specs_for_candidates(
    *,
    driver,
    database: str,
    candidate_ids: List[str],
    min_domain_weight: float,
) -> Dict[str, List[Dict[str, Any]]]:
    if not candidate_ids:
        return {}

    records, _, _ = driver.execute_query(
        """
        UNWIND $candidate_ids AS oid
        MATCH (g:Grant)
        WHERE toString(g.opportunity_id) = oid
        MATCH (g)-[hs:HAS_SPECIALIZATION_KEYWORD]->(gs:GrantKeyword {bucket: 'specialization'})
        WHERE gs.embedding IS NOT NULL

        OPTIONAL MATCH (g)-[hd:HAS_DOMAIN_KEYWORD]->(gd:GrantKeyword {bucket: 'domain'})-[ds:GRANT_DOMAIN_HAS_SPECIALIZATION]->(gs)
        WHERE
            toString(ds.scope_opportunity_id) = oid
            AND coalesce(hd.weight, 0.0) > $min_domain_weight
        OPTIONAL MATCH (gd)-[:MAPS_TO_SHARED_DOMAIN]->(sd:DomainKeywordShared {bucket: 'domain'})

        RETURN
            oid AS opportunity_id,
            coalesce(gs.value, '') AS keyword_value,
            coalesce(hs.weight, 0.0) AS keyword_weight,
            gs.embedding AS embedding,
            collect(DISTINCT {
                domain_norm: coalesce(sd.value_norm, toLower(coalesce(gd.value, ''))),
                domain_weight: coalesce(ds.domain_weight, hd.weight, 0.0)
            }) AS domain_links
        ORDER BY opportunity_id ASC, keyword_weight DESC, keyword_value ASC
        """,
        parameters_={
            "candidate_ids": [str(x) for x in candidate_ids if _clean_text(x)],
            "min_domain_weight": float(min_domain_weight),
        },
        database_=database,
    )

    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in records:
        item = dict(row or {})
        oid = _clean_text(item.get("opportunity_id"))
        keyword_value = _clean_text(item.get("keyword_value"))
        emb = _coerce_vector(item.get("embedding"))
        if not oid or not keyword_value or not emb:
            continue
        domains: Set[str] = set()
        domain_weights: Dict[str, float] = {}
        for link in list(item.get("domain_links") or []):
            if not isinstance(link, dict):
                continue
            dom = _clean_text(link.get("domain_norm")).lower()
            if not dom:
                continue
            w = _safe_unit_float(link.get("domain_weight"), default=0.0)
            domains.add(dom)
            if w > float(domain_weights.get(dom, 0.0)):
                domain_weights[dom] = w
        out.setdefault(oid, []).append(
            {
                "keyword_value": keyword_value,
                "keyword_weight": _safe_unit_float(item.get("keyword_weight"), default=0.0),
                "embedding": emb,
                "domain_norms": domains,
                "domain_weights": domain_weights,
            }
        )
    return out


def _fetch_grant_domain_evidence_vectors_for_candidates(
    *,
    driver,
    database: str,
    candidate_ids: List[str],
    min_domain_weight: float,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not candidate_ids:
        return {}

    records, _, _ = driver.execute_query(
        """
        UNWIND $candidate_ids AS oid
        MATCH (g:Grant)
        WHERE toString(g.opportunity_id) = oid
        MATCH (g)-[hd:HAS_DOMAIN_KEYWORD]->(gd:GrantKeyword {bucket: 'domain'})
        WHERE coalesce(hd.weight, 0.0) > $min_domain_weight

        OPTIONAL MATCH (gd)-[:MAPS_TO_SHARED_DOMAIN]->(sd:DomainKeywordShared {bucket: 'domain'})

        OPTIONAL MATCH (gd)-[dc:DOMAIN_SUPPORTED_BY_GRANT_CHUNK]->(c:GrantTextChunk)
        WHERE toString(dc.scope_opportunity_id) = oid

        RETURN
            oid AS opportunity_id,
            coalesce(sd.value_norm, toLower(coalesce(gd.value, ''))) AS domain_norm,
            max(coalesce(hd.weight, 0.0)) AS domain_weight,
            collect(DISTINCT c.embedding) AS chunk_embeddings
        """,
        parameters_={
            "candidate_ids": [str(x) for x in candidate_ids if _clean_text(x)],
            "min_domain_weight": float(min_domain_weight),
        },
        database_=database,
    )

    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in records:
        item = dict(row or {})
        oid = _clean_text(item.get("opportunity_id"))
        domain_norm = _clean_text(item.get("domain_norm")).lower()
        if not oid or not domain_norm:
            continue
        dom_map = out.setdefault(oid, {})
        entry = dom_map.setdefault(
            domain_norm,
            {
                "vectors": [],
                "domain_weight": _safe_unit_float(item.get("domain_weight"), default=0.0),
            },
        )
        d_w = _safe_unit_float(item.get("domain_weight"), default=0.0)
        if d_w > float(entry.get("domain_weight") or 0.0):
            entry["domain_weight"] = d_w
        vectors = list(entry.get("vectors") or [])
        for raw in list(item.get("chunk_embeddings") or []):
            vec = _coerce_vector(raw)
            if vec:
                vectors.append(vec)
        entry["vectors"] = vectors
    return out


def _normalize_passed_domains(pairs: Iterable[Dict[str, Any]]) -> Tuple[Set[str], Dict[str, float], List[Dict[str, Any]]]:
    passed_domains: Set[str] = set()
    by_domain_weight: Dict[str, float] = {}
    for row in list(pairs or []):
        if not isinstance(row, dict):
            continue
        domain_norm = _clean_text(row.get("domain_norm")).lower()
        if not domain_norm:
            continue
        pair_weight = float(row.get("pair_weight") or 0.0)
        passed_domains.add(domain_norm)
        by_domain_weight[domain_norm] = by_domain_weight.get(domain_norm, 0.0) + max(0.0, pair_weight)
    details = [{"domain_norm": k, "pair_weight_sum": float(v)} for k, v in sorted(by_domain_weight.items())]
    return passed_domains, by_domain_weight, details


def _compute_directional_coverage(
    *,
    target_specs: Sequence[Dict[str, Any]],
    source_domain_vectors: Dict[str, Dict[str, Any]],
    passed_domains: Set[str],
    passed_domain_weights: Dict[str, float],
    hit_threshold: float,
    details_limit: int,
) -> Dict[str, Any]:
    total_specs = len(list(target_specs or []))
    considered = 0
    scored = 0
    hit_count = 0
    sum_sim = 0.0
    sum_weighted = 0.0
    weight_total = 0.0
    all_details: List[Dict[str, Any]] = []

    for spec in list(target_specs or []):
        spec_domains = set(spec.get("domain_norms") or set())
        spec_domain_weights = dict(spec.get("domain_weights") or {})
        active_domains = sorted(spec_domains.intersection(passed_domains))
        if not active_domains:
            continue
        considered += 1

        evidence_vectors: List[List[float]] = []
        source_domain_weight_sum = 0.0
        target_domain_weight_sum = 0.0
        sqrt_pair_weight_sum = 0.0
        for dom in active_domains:
            src_entry = dict(source_domain_vectors.get(dom) or {})
            evidence_vectors.extend(list(src_entry.get("vectors") or []))
            src_w = _safe_unit_float(src_entry.get("domain_weight"), default=0.0)
            tgt_w = _safe_unit_float(spec_domain_weights.get(dom), default=0.0)
            source_domain_weight_sum += src_w
            target_domain_weight_sum += tgt_w
            sqrt_pair_weight_sum += math.sqrt(max(0.0, src_w * tgt_w))

        agg_vec = _mean_vector(evidence_vectors)
        sim = 0.0
        if agg_vec:
            spec_vec = list(spec.get("embedding") or [])
            sim = _cosine_vec(spec_vec, agg_vec)
            scored += 1

        # Penalize only domains that belong to this target specialization but are not
        # present in passed/shared domains for the faculty<->grant pair.
        mismatch_domains = sorted(set(spec_domains).difference(passed_domains))
        penalty_multiplier = 1.0
        penalty_domain_weights: Dict[str, float] = {}
        for dom in mismatch_domains:
            if dom in spec_domain_weights:
                pw = _safe_unit_float(spec_domain_weights.get(dom), default=1.0)
            else:
                pw = _safe_unit_float(passed_domain_weights.get(dom), default=1.0)
            penalty_multiplier *= max(0.0, (1.0 - pw))
            penalty_domain_weights[dom] = pw

        weight = _safe_unit_float(spec.get("keyword_weight"), default=0.0)
        penalty_factor = _safe_unit_float(1.0 - penalty_multiplier, default=0.0)
        weight_penalized = _safe_unit_float(weight * penalty_multiplier, default=0.0)
        sqrt_domain_weight_factor = math.sqrt(max(0.0, source_domain_weight_sum * target_domain_weight_sum))
        coverage_score = _safe_unit_float(sim * sqrt_domain_weight_factor * weight_penalized, default=0.0)
        sum_sim += sim
        sum_weighted += (sim * weight)
        weight_total += weight
        if sim >= hit_threshold:
            hit_count += 1

        all_details.append(
            {
                "spec_keyword": _clean_text(spec.get("keyword_value")),
                "spec_weight": weight,
                "passed_domains_for_spec": active_domains,
                "evidence_vector_count": int(len(evidence_vectors)),
                "coverage_sim": sim,
                "source_domain_weight_sum": float(source_domain_weight_sum),
                "target_domain_weight_sum": float(target_domain_weight_sum),
                "sqrt_pair_weight_sum": float(sqrt_pair_weight_sum),
                "sqrt_domain_weight_factor": float(sqrt_domain_weight_factor),
                "penalty_domains": mismatch_domains,
                "penalty_domain_weights": penalty_domain_weights,
                "penalty_factor": float(penalty_factor),
                "penalty_multiplier": float(penalty_multiplier),
                "spec_weight_penalized": float(weight_penalized),
                "coverage_score": float(coverage_score),
            }
        )

    avg_sim = 0.0 if considered <= 0 else (sum_sim / float(considered))
    weighted_avg_sim = 0.0 if weight_total <= 0.0 else (sum_weighted / weight_total)
    hit_ratio = 0.0 if considered <= 0 else (float(hit_count) / float(considered))
    return {
        "total_specs": int(total_specs),
        "considered_specs": int(considered),
        "scored_specs": int(scored),
        "hit_count": int(hit_count),
        "hit_ratio": float(hit_ratio),
        "coverage_avg": float(avg_sim),
        "coverage_weighted_avg": float(weighted_avg_sim),
        "details": all_details[: int(max(0, details_limit))],
        "all_details": all_details,
    }


def _collapse_keyword_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        key_raw = _clean_text(row.get("spec_keyword"))
        key_norm = key_raw.lower()
        if not key_norm:
            continue
        spec_weight = _safe_unit_float(row.get("spec_weight"), default=0.0)
        coverage_sim = _safe_unit_float(row.get("coverage_sim"), default=0.0)
        coverage_score = _safe_unit_float(row.get("coverage_score"), default=0.0)
        source_domain_weight_sum = _safe_unit_float(row.get("source_domain_weight_sum"), default=0.0)
        target_domain_weight_sum = _safe_unit_float(row.get("target_domain_weight_sum"), default=0.0)
        sqrt_pair_weight_sum = _safe_unit_float(row.get("sqrt_pair_weight_sum"), default=0.0)
        sqrt_domain_weight_factor = _safe_unit_float(row.get("sqrt_domain_weight_factor"), default=0.0)
        penalty_factor = _safe_unit_float(row.get("penalty_factor"), default=0.0)
        penalty_multiplier = _safe_unit_float(row.get("penalty_multiplier"), default=1.0)
        spec_weight_penalized = _safe_unit_float(row.get("spec_weight_penalized"), default=spec_weight)
        evidence_count = int(max(0, int(row.get("evidence_vector_count") or 0)))
        domains = {
            _clean_text(x).lower()
            for x in list(row.get("passed_domains_for_spec") or [])
            if _clean_text(x)
        }
        penalty_domains = {
            _clean_text(x).lower()
            for x in list(row.get("penalty_domains") or [])
            if _clean_text(x)
        }

        prev = by_key.get(key_norm)
        if prev is None:
            by_key[key_norm] = {
                "spec_keyword": key_raw,
                "spec_weight": spec_weight,
                "coverage_sim": coverage_sim,
                "coverage_score": coverage_score,
                "source_domain_weight_sum": source_domain_weight_sum,
                "target_domain_weight_sum": target_domain_weight_sum,
                "sqrt_pair_weight_sum": sqrt_pair_weight_sum,
                "sqrt_domain_weight_factor": sqrt_domain_weight_factor,
                "penalty_factor": penalty_factor,
                "penalty_multiplier": penalty_multiplier,
                "spec_weight_penalized": spec_weight_penalized,
                "evidence_vector_count": evidence_count,
                "passed_domains_for_spec": domains,
                "penalty_domains": penalty_domains,
            }
            continue

        if coverage_score > float(prev.get("coverage_score") or 0.0):
            prev["coverage_score"] = coverage_score
        if coverage_sim > float(prev.get("coverage_sim") or 0.0):
            prev["coverage_sim"] = coverage_sim
        if spec_weight > float(prev.get("spec_weight") or 0.0):
            prev["spec_weight"] = spec_weight
        if source_domain_weight_sum > float(prev.get("source_domain_weight_sum") or 0.0):
            prev["source_domain_weight_sum"] = source_domain_weight_sum
        if target_domain_weight_sum > float(prev.get("target_domain_weight_sum") or 0.0):
            prev["target_domain_weight_sum"] = target_domain_weight_sum
        if sqrt_pair_weight_sum > float(prev.get("sqrt_pair_weight_sum") or 0.0):
            prev["sqrt_pair_weight_sum"] = sqrt_pair_weight_sum
        if sqrt_domain_weight_factor > float(prev.get("sqrt_domain_weight_factor") or 0.0):
            prev["sqrt_domain_weight_factor"] = sqrt_domain_weight_factor
        if penalty_factor > float(prev.get("penalty_factor") or 0.0):
            prev["penalty_factor"] = penalty_factor
        if penalty_multiplier < float(prev.get("penalty_multiplier") or 1.0):
            prev["penalty_multiplier"] = penalty_multiplier
        if spec_weight_penalized < float(prev.get("spec_weight_penalized") or spec_weight):
            prev["spec_weight_penalized"] = spec_weight_penalized
        if evidence_count > int(prev.get("evidence_vector_count") or 0):
            prev["evidence_vector_count"] = evidence_count
        prev_domains = set(prev.get("passed_domains_for_spec") or set())
        prev["passed_domains_for_spec"] = prev_domains.union(domains)
        prev_penalty = set(prev.get("penalty_domains") or set())
        prev["penalty_domains"] = prev_penalty.union(penalty_domains)

    out: List[Dict[str, Any]] = []
    for row in by_key.values():
        out.append(
            {
                "spec_keyword": _clean_text(row.get("spec_keyword")),
                "spec_weight": _safe_unit_float(row.get("spec_weight"), default=0.0),
                "coverage_sim": _safe_unit_float(row.get("coverage_sim"), default=0.0),
                "coverage_score": _safe_unit_float(row.get("coverage_score"), default=0.0),
                "source_domain_weight_sum": _safe_unit_float(row.get("source_domain_weight_sum"), default=0.0),
                "target_domain_weight_sum": _safe_unit_float(row.get("target_domain_weight_sum"), default=0.0),
                "sqrt_pair_weight_sum": _safe_unit_float(row.get("sqrt_pair_weight_sum"), default=0.0),
                "sqrt_domain_weight_factor": _safe_unit_float(row.get("sqrt_domain_weight_factor"), default=0.0),
                "penalty_factor": _safe_unit_float(row.get("penalty_factor"), default=0.0),
                "penalty_multiplier": _safe_unit_float(row.get("penalty_multiplier"), default=1.0),
                "spec_weight_penalized": _safe_unit_float(row.get("spec_weight_penalized"), default=0.0),
                "evidence_vector_count": int(row.get("evidence_vector_count") or 0),
                "passed_domains_for_spec": sorted(set(row.get("passed_domains_for_spec") or set())),
                "penalty_domains": sorted(set(row.get("penalty_domains") or set())),
            }
        )
    out.sort(key=lambda x: (float(x.get("coverage_score") or 0.0), float(x.get("coverage_sim") or 0.0), _clean_text(x.get("spec_keyword"))), reverse=True)
    return out


def _upsert_keyword_coverage_edges(
    *,
    driver,
    database: str,
    faculty_id: int,
    opportunity_id: str,
    coverage_hit_threshold: float,
    f2g_rows: Sequence[Dict[str, Any]],
    g2f_rows: Sequence[Dict[str, Any]],
) -> Dict[str, int]:
    f2g_linked = 0
    g2f_linked = 0

    if f2g_rows:
        records, _, _ = driver.execute_query(
            f"""
            UNWIND $rows AS row
            MATCH (f:Faculty {{faculty_id: $faculty_id}})
            MATCH (g:Grant)
            WHERE toString(g.opportunity_id) = $opportunity_id
            MATCH (g)-[:HAS_SPECIALIZATION_KEYWORD]->(gk:GrantKeyword {{bucket: 'specialization'}})
            WHERE toLower(coalesce(gk.value, '')) = toLower(row.spec_keyword)
            MERGE (f)-[r:{EDGE_F2G_SPEC} {{
                scope_faculty_id: $faculty_id,
                scope_opportunity_id: $opportunity_id,
                grant_keyword_value: row.spec_keyword
            }}]->(gk)
            SET
                r.method = $method,
                r.score = row.coverage_score,
                r.coverage_score = row.coverage_score,
                r.coverage_sim = row.coverage_sim,
                r.grant_specialization_weight = row.spec_weight,
                r.grant_specialization_weight_penalized = row.spec_weight_penalized,
                r.spec_weight = row.spec_weight,
                r.spec_weight_penalized = row.spec_weight_penalized,
                r.faculty_domain_weight_sum = row.source_domain_weight_sum,
                r.grant_domain_weight_sum = row.target_domain_weight_sum,
                r.sqrt_domain_weight_factor = row.sqrt_domain_weight_factor,
                r.penalty_factor = row.penalty_factor,
                r.penalty_multiplier = row.penalty_multiplier,
                r.penalty_domains = row.penalty_domains,
                r.penalty_domain_count = size(row.penalty_domains),
                r.passed_domains = row.passed_domains_for_spec,
                r.passed_domain_count = size(row.passed_domains_for_spec),
                r.evidence_vector_count = toInteger(row.evidence_vector_count),
                r.hit_threshold = $coverage_hit_threshold,
                r.is_hit = (row.coverage_sim >= $coverage_hit_threshold),
                r.updated_at = datetime()
            RETURN count(r) AS linked_count
            """,
            parameters_={
                "faculty_id": int(faculty_id),
                "opportunity_id": str(opportunity_id),
                "coverage_hit_threshold": float(coverage_hit_threshold),
                "method": EDGE_METHOD,
                "rows": list(f2g_rows),
            },
            database_=database,
        )
        if records:
            f2g_linked = int(dict(records[0] or {}).get("linked_count") or 0)

    if g2f_rows:
        records, _, _ = driver.execute_query(
            f"""
            UNWIND $rows AS row
            MATCH (f:Faculty {{faculty_id: $faculty_id}})
            MATCH (g:Grant)
            WHERE toString(g.opportunity_id) = $opportunity_id
            MATCH (f)-[:HAS_SPECIALIZATION_KEYWORD]->(fk:FacultyKeyword {{bucket: 'specialization'}})
            WHERE toLower(coalesce(fk.value, '')) = toLower(row.spec_keyword)
            MERGE (g)-[r:{EDGE_G2F_SPEC} {{
                scope_faculty_id: $faculty_id,
                scope_opportunity_id: $opportunity_id,
                faculty_keyword_value: row.spec_keyword
            }}]->(fk)
            SET
                r.method = $method,
                r.score = row.coverage_score,
                r.coverage_score = row.coverage_score,
                r.coverage_sim = row.coverage_sim,
                r.faculty_specialization_weight = row.spec_weight,
                r.faculty_specialization_weight_penalized = row.spec_weight_penalized,
                r.spec_weight = row.spec_weight,
                r.spec_weight_penalized = row.spec_weight_penalized,
                r.grant_domain_weight_sum = row.source_domain_weight_sum,
                r.faculty_domain_weight_sum = row.target_domain_weight_sum,
                r.sqrt_domain_weight_factor = row.sqrt_domain_weight_factor,
                r.penalty_factor = row.penalty_factor,
                r.penalty_multiplier = row.penalty_multiplier,
                r.penalty_domains = row.penalty_domains,
                r.penalty_domain_count = size(row.penalty_domains),
                r.passed_domains = row.passed_domains_for_spec,
                r.passed_domain_count = size(row.passed_domains_for_spec),
                r.evidence_vector_count = toInteger(row.evidence_vector_count),
                r.hit_threshold = $coverage_hit_threshold,
                r.is_hit = (row.coverage_sim >= $coverage_hit_threshold),
                r.updated_at = datetime()
            RETURN count(r) AS linked_count
            """,
            parameters_={
                "faculty_id": int(faculty_id),
                "opportunity_id": str(opportunity_id),
                "coverage_hit_threshold": float(coverage_hit_threshold),
                "method": EDGE_METHOD,
                "rows": list(g2f_rows),
            },
            database_=database,
        )
        if records:
            g2f_linked = int(dict(records[0] or {}).get("linked_count") or 0)

    return {
        "f2g_keyword_edges_linked": int(f2g_linked),
        "g2f_keyword_edges_linked": int(g2f_linked),
    }


def _upsert_coverage_edge(
    *,
    driver,
    database: str,
    faculty_id: int,
    opportunity_id: str,
    payload: Dict[str, Any],
) -> int:
    records, _, _ = driver.execute_query(
        f"""
        MATCH (f:Faculty {{faculty_id: $faculty_id}})
        MATCH (g:Grant)
        WHERE toString(g.opportunity_id) = $opportunity_id
        MERGE (f)-[r:{EDGE_LABEL}]->(g)
        SET
            r.method = $method,
            r.min_domain_weight = $min_domain_weight,
            r.coverage_hit_threshold = $coverage_hit_threshold,
            r.candidate_rank_score = $candidate_rank_score,
            r.passed_domain_count = $passed_domain_count,
            r.passed_domains = $passed_domains,
            r.passed_domain_weights_json = $passed_domain_weights_json,
            r.faculty_covers_grant_specs_total = $f2g_total,
            r.faculty_covers_grant_specs_considered = $f2g_considered,
            r.faculty_covers_grant_specs_scored = $f2g_scored,
            r.faculty_covers_grant_specs_hit_count = $f2g_hit_count,
            r.faculty_covers_grant_specs_hit_ratio = $f2g_hit_ratio,
            r.faculty_covers_grant_specs_coverage_avg = $f2g_coverage_avg,
            r.faculty_covers_grant_specs_coverage_weighted_avg = $f2g_coverage_weighted_avg,
            r.faculty_covers_grant_specs_details_json = $f2g_details_json,
            r.grant_covers_faculty_specs_total = $g2f_total,
            r.grant_covers_faculty_specs_considered = $g2f_considered,
            r.grant_covers_faculty_specs_scored = $g2f_scored,
            r.grant_covers_faculty_specs_hit_count = $g2f_hit_count,
            r.grant_covers_faculty_specs_hit_ratio = $g2f_hit_ratio,
            r.grant_covers_faculty_specs_coverage_avg = $g2f_coverage_avg,
            r.grant_covers_faculty_specs_coverage_weighted_avg = $g2f_coverage_weighted_avg,
            r.grant_covers_faculty_specs_details_json = $g2f_details_json,
            r.updated_at = datetime()
        RETURN count(r) AS linked_count
        """,
        parameters_={
            "faculty_id": int(faculty_id),
            "opportunity_id": str(opportunity_id),
            "method": str(payload.get("method") or EDGE_METHOD),
            "min_domain_weight": float(payload.get("min_domain_weight") or 0.0),
            "coverage_hit_threshold": float(payload.get("coverage_hit_threshold") or 0.0),
            "candidate_rank_score": float(payload.get("candidate_rank_score") or 0.0),
            "passed_domain_count": int(payload.get("passed_domain_count") or 0),
            "passed_domains": list(payload.get("passed_domains") or []),
            "passed_domain_weights_json": str(payload.get("passed_domain_weights_json") or "[]"),
            "f2g_total": int(payload.get("f2g_total") or 0),
            "f2g_considered": int(payload.get("f2g_considered") or 0),
            "f2g_scored": int(payload.get("f2g_scored") or 0),
            "f2g_hit_count": int(payload.get("f2g_hit_count") or 0),
            "f2g_hit_ratio": float(payload.get("f2g_hit_ratio") or 0.0),
            "f2g_coverage_avg": float(payload.get("f2g_coverage_avg") or 0.0),
            "f2g_coverage_weighted_avg": float(payload.get("f2g_coverage_weighted_avg") or 0.0),
            "f2g_details_json": str(payload.get("f2g_details_json") or "[]"),
            "g2f_total": int(payload.get("g2f_total") or 0),
            "g2f_considered": int(payload.get("g2f_considered") or 0),
            "g2f_scored": int(payload.get("g2f_scored") or 0),
            "g2f_hit_count": int(payload.get("g2f_hit_count") or 0),
            "g2f_hit_ratio": float(payload.get("g2f_hit_ratio") or 0.0),
            "g2f_coverage_avg": float(payload.get("g2f_coverage_avg") or 0.0),
            "g2f_coverage_weighted_avg": float(payload.get("g2f_coverage_weighted_avg") or 0.0),
            "g2f_details_json": str(payload.get("g2f_details_json") or "[]"),
        },
        database_=database,
    )
    if not records:
        return 0
    row = dict(records[0] or {})
    return int(row.get("linked_count") or 0)


def _link_for_faculty(
    *,
    driver,
    database: str,
    faculty_id: int,
    faculty_email: str,
    min_domain_weight: float,
    top_candidates: int,
    coverage_hit_threshold: float,
    details_limit: int,
) -> Dict[str, Any]:
    candidates = _fetch_domain_gate_candidates(
        driver=driver,
        database=database,
        faculty_id=faculty_id,
        min_domain_weight=min_domain_weight,
        top_candidates=top_candidates,
    )
    if not candidates:
        return {
            "faculty_id": faculty_id,
            "faculty_email": faculty_email,
            "candidate_count": 0,
            "edge_linked": 0,
            "skipped_no_specs_or_evidence": 0,
        }

    faculty_specs = _fetch_faculty_specs(
        driver=driver,
        database=database,
        faculty_id=faculty_id,
        faculty_email=faculty_email,
        min_domain_weight=min_domain_weight,
    )
    faculty_domain_evidence = _fetch_faculty_domain_evidence_vectors(
        driver=driver,
        database=database,
        faculty_id=faculty_id,
        min_domain_weight=min_domain_weight,
    )

    candidate_ids = [str(_clean_text(c.get("opportunity_id"))) for c in candidates if _clean_text(c.get("opportunity_id"))]
    grant_specs_map = _fetch_grant_specs_for_candidates(
        driver=driver,
        database=database,
        candidate_ids=candidate_ids,
        min_domain_weight=min_domain_weight,
    )
    grant_domain_evidence_map = _fetch_grant_domain_evidence_vectors_for_candidates(
        driver=driver,
        database=database,
        candidate_ids=candidate_ids,
        min_domain_weight=min_domain_weight,
    )

    linked_edges = 0
    f2g_keyword_edges_linked = 0
    g2f_keyword_edges_linked = 0
    skipped = 0
    per_grant: List[Dict[str, Any]] = []

    for cand in candidates:
        opportunity_id = _clean_text(cand.get("opportunity_id"))
        if not opportunity_id:
            continue
        grant_specs = list(grant_specs_map.get(opportunity_id) or [])
        grant_domain_evidence = dict(grant_domain_evidence_map.get(opportunity_id) or {})

        passed_domains, passed_domain_weights, passed_domain_details = _normalize_passed_domains(cand.get("passed_domain_pairs") or [])
        if not passed_domains or not faculty_specs or not grant_specs:
            skipped += 1
            per_grant.append(
                {
                    "opportunity_id": opportunity_id,
                    "opportunity_title": _clean_text(cand.get("opportunity_title")),
                    "linked": False,
                    "reason": "missing passed domains or specialization keywords",
                }
            )
            continue

        f2g = _compute_directional_coverage(
            target_specs=grant_specs,
            source_domain_vectors=faculty_domain_evidence,
            passed_domains=passed_domains,
            passed_domain_weights=passed_domain_weights,
            hit_threshold=coverage_hit_threshold,
            details_limit=details_limit,
        )
        g2f = _compute_directional_coverage(
            target_specs=faculty_specs,
            source_domain_vectors=grant_domain_evidence,
            passed_domains=passed_domains,
            passed_domain_weights=passed_domain_weights,
            hit_threshold=coverage_hit_threshold,
            details_limit=details_limit,
        )
        f2g_keyword_rows = _collapse_keyword_rows(f2g.get("all_details") or [])
        g2f_keyword_rows = _collapse_keyword_rows(g2f.get("all_details") or [])
        kw_counts = _upsert_keyword_coverage_edges(
            driver=driver,
            database=database,
            faculty_id=faculty_id,
            opportunity_id=opportunity_id,
            coverage_hit_threshold=coverage_hit_threshold,
            f2g_rows=f2g_keyword_rows,
            g2f_rows=g2f_keyword_rows,
        )
        f2g_keyword_edges_linked += int(kw_counts.get("f2g_keyword_edges_linked") or 0)
        g2f_keyword_edges_linked += int(kw_counts.get("g2f_keyword_edges_linked") or 0)

        payload = {
            "method": EDGE_METHOD,
            "min_domain_weight": float(min_domain_weight),
            "coverage_hit_threshold": float(coverage_hit_threshold),
            "candidate_rank_score": float(cand.get("candidate_rank_score") or 0.0),
            "passed_domain_count": int(len(passed_domains)),
            "passed_domains": sorted(passed_domains),
            "passed_domain_weights_json": _json_compact(passed_domain_details),
            "f2g_total": int(f2g["total_specs"]),
            "f2g_considered": int(f2g["considered_specs"]),
            "f2g_scored": int(f2g["scored_specs"]),
            "f2g_hit_count": int(f2g["hit_count"]),
            "f2g_hit_ratio": float(f2g["hit_ratio"]),
            "f2g_coverage_avg": float(f2g["coverage_avg"]),
            "f2g_coverage_weighted_avg": float(f2g["coverage_weighted_avg"]),
            "f2g_details_json": _json_compact(f2g["details"]),
            "g2f_total": int(g2f["total_specs"]),
            "g2f_considered": int(g2f["considered_specs"]),
            "g2f_scored": int(g2f["scored_specs"]),
            "g2f_hit_count": int(g2f["hit_count"]),
            "g2f_hit_ratio": float(g2f["hit_ratio"]),
            "g2f_coverage_avg": float(g2f["coverage_avg"]),
            "g2f_coverage_weighted_avg": float(g2f["coverage_weighted_avg"]),
            "g2f_details_json": _json_compact(g2f["details"]),
        }
        linked_count = _upsert_coverage_edge(
            driver=driver,
            database=database,
            faculty_id=faculty_id,
            opportunity_id=opportunity_id,
            payload=payload,
        )
        linked_edges += int(linked_count)
        per_grant.append(
            {
                "opportunity_id": opportunity_id,
                "opportunity_title": _clean_text(cand.get("opportunity_title")),
                "linked": bool(linked_count),
                "candidate_rank_score": float(cand.get("candidate_rank_score") or 0.0),
                "passed_domain_count": int(len(passed_domains)),
                "f2g_keyword_edges_linked": int(kw_counts.get("f2g_keyword_edges_linked") or 0),
                "g2f_keyword_edges_linked": int(kw_counts.get("g2f_keyword_edges_linked") or 0),
                "faculty_covers_grant_specs_coverage_avg": float(f2g["coverage_avg"]),
                "grant_covers_faculty_specs_coverage_avg": float(g2f["coverage_avg"]),
                "faculty_covers_grant_specs_hit_ratio": float(f2g["hit_ratio"]),
                "grant_covers_faculty_specs_hit_ratio": float(g2f["hit_ratio"]),
            }
        )

    return {
        "faculty_id": int(faculty_id),
        "faculty_email": _clean_text(faculty_email).lower(),
        "candidate_count": int(len(candidates)),
        "edge_linked": int(linked_edges),
        "f2g_keyword_edges_linked": int(f2g_keyword_edges_linked),
        "g2f_keyword_edges_linked": int(g2f_keyword_edges_linked),
        "skipped_no_specs_or_evidence": int(skipped),
        "grants": per_grant,
    }


def run_linker(
    *,
    faculty_id: Optional[int],
    faculty_email: str,
    all_faculties: bool,
    faculty_limit: int,
    faculty_offset: int,
    min_domain_weight: float,
    top_candidates: int,
    coverage_hit_threshold: float,
    details_limit: int,
    stop_on_error: bool,
    uri: str = "",
    username: str = "",
    password: str = "",
    database: str = "",
) -> Dict[str, Any]:
    safe_min_domain_weight = _safe_unit_float(min_domain_weight, default=0.6)
    safe_top_candidates = _safe_limit(top_candidates, default=20, minimum=1, maximum=500)
    safe_faculty_limit = _safe_limit(faculty_limit, default=100000, minimum=1, maximum=1000000)
    safe_faculty_offset = max(0, int(faculty_offset or 0))
    safe_hit_threshold = _safe_unit_float(coverage_hit_threshold, default=0.35)
    safe_details_limit = _safe_limit(details_limit, default=200, minimum=0, maximum=5000)

    load_dotenv_if_present()
    settings = read_neo4j_settings(
        uri=uri,
        username=username,
        password=password,
        database=database,
    )

    with GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
    ) as driver:
        driver.verify_connectivity()

        faculties = _load_faculties(
            driver=driver,
            database=settings.database,
            faculty_id=faculty_id,
            faculty_email=faculty_email,
            all_faculties=all_faculties,
            limit=safe_faculty_limit,
            offset=safe_faculty_offset,
        )
        if not faculties:
            raise RuntimeError("No faculty rows found for requested scope.")

        ok_rows: List[Dict[str, Any]] = []
        error_rows: List[Dict[str, Any]] = []

        for fac in faculties:
            fid = int(fac.get("faculty_id") or 0)
            email = _clean_text(fac.get("faculty_email")).lower()
            if fid <= 0:
                continue
            try:
                res = _link_for_faculty(
                    driver=driver,
                    database=settings.database,
                    faculty_id=fid,
                    faculty_email=email,
                    min_domain_weight=safe_min_domain_weight,
                    top_candidates=safe_top_candidates,
                    coverage_hit_threshold=safe_hit_threshold,
                    details_limit=safe_details_limit,
                )
                ok_rows.append(res)
            except Exception as exc:
                err = {"faculty_id": fid, "faculty_email": email, "error": str(exc)}
                error_rows.append(err)
                if stop_on_error:
                    raise

    return {
        "params": {
            "faculty_id": faculty_id,
            "faculty_email": _clean_text(faculty_email).lower(),
            "all_faculties": bool(all_faculties),
            "faculty_limit": int(safe_faculty_limit),
            "faculty_offset": int(safe_faculty_offset),
            "min_domain_weight": float(safe_min_domain_weight),
            "top_candidates": int(safe_top_candidates),
            "coverage_hit_threshold": float(safe_hit_threshold),
            "details_limit": int(safe_details_limit),
            "edge_label": EDGE_LABEL,
            "edge_label_f2g_keyword": EDGE_F2G_SPEC,
            "edge_label_g2f_keyword": EDGE_G2F_SPEC,
            "edge_method": EDGE_METHOD,
            "pipeline": (
                "domain gate first; "
                "for each spec, evidence chunk set only from linked chunks/publications in passed domains; "
                "store faculty->grant coverage and grant->faculty coverage on Faculty-Grant edge; "
                "also store keyword-level faculty->grant_spec and grant->faculty_spec coverage edges"
            ),
        },
        "totals": {
            "faculties_requested": int(len(ok_rows) + len(error_rows)),
            "faculties_succeeded": int(len(ok_rows)),
            "faculties_failed": int(len(error_rows)),
            "edges_linked": int(sum(int(x.get("edge_linked") or 0) for x in ok_rows)),
            "f2g_keyword_edges_linked": int(sum(int(x.get("f2g_keyword_edges_linked") or 0) for x in ok_rows)),
            "g2f_keyword_edges_linked": int(sum(int(x.get("g2f_keyword_edges_linked") or 0) for x in ok_rows)),
            "candidates_processed": int(sum(int(x.get("candidate_count") or 0) for x in ok_rows)),
        },
        "faculties": ok_rows,
        "errors": error_rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Link Faculty->Grant coverage edges with domain gate first, then "
            "spec coverage using evidence chunks constrained to passed domains."
        )
    )
    parser.add_argument("--faculty-id", type=int, default=0, help="Run for one faculty_id.")
    parser.add_argument("--faculty-email", type=str, default="", help="Run for one faculty email.")
    parser.add_argument("--all-faculties", action="store_true", help="Run for all faculties.")
    parser.add_argument("--faculty-limit", type=int, default=100000, help="Limit faculties in --all-faculties mode.")
    parser.add_argument("--faculty-offset", type=int, default=0, help="Offset faculties in --all-faculties mode.")
    parser.add_argument("--min-domain-weight", type=float, default=0.6, help="Domain gate threshold.")
    parser.add_argument("--top-candidates", type=int, default=20, help="Top grants per faculty after domain gate.")
    parser.add_argument("--coverage-hit-threshold", type=float, default=0.35, help="Similarity threshold for coverage hit ratio.")
    parser.add_argument("--details-limit", type=int, default=200, help="Max per-direction per-grant spec detail rows stored on edge.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on first faculty error.")
    parser.add_argument("--json-only", action="store_true", help="Print only JSON payload.")
    parser.add_argument("--uri", type=str, default="", help="Neo4j URI. Fallback: NEO4J_URI")
    parser.add_argument("--username", type=str, default="", help="Neo4j username. Fallback: NEO4J_USERNAME")
    parser.add_argument("--password", type=str, default="", help="Neo4j password. Fallback: NEO4J_PASSWORD")
    parser.add_argument("--database", type=str, default="", help="Neo4j database. Fallback: NEO4J_DATABASE or neo4j")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    fid_raw = int(args.faculty_id or 0)
    fid = fid_raw if fid_raw > 0 else None

    payload = run_linker(
        faculty_id=fid,
        faculty_email=_clean_text(args.faculty_email).lower(),
        all_faculties=bool(args.all_faculties),
        faculty_limit=int(args.faculty_limit or 100000),
        faculty_offset=int(args.faculty_offset or 0),
        min_domain_weight=float(args.min_domain_weight),
        top_candidates=int(args.top_candidates or 20),
        coverage_hit_threshold=float(args.coverage_hit_threshold),
        details_limit=int(args.details_limit or 200),
        stop_on_error=bool(args.stop_on_error),
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    if not args.json_only:
        print("Faculty-Grant coverage linking complete.")
        print(f"  faculties requested : {payload.get('totals', {}).get('faculties_requested', 0)}")
        print(f"  faculties succeeded : {payload.get('totals', {}).get('faculties_succeeded', 0)}")
        print(f"  faculties failed    : {payload.get('totals', {}).get('faculties_failed', 0)}")
        print(f"  candidates processed: {payload.get('totals', {}).get('candidates_processed', 0)}")
        print(f"  fac->grant edges    : {payload.get('totals', {}).get('edges_linked', 0)}")
        print(f"  fac->grant_spec     : {payload.get('totals', {}).get('f2g_keyword_edges_linked', 0)}")
        print(f"  grant->fac_spec     : {payload.get('totals', {}).get('g2f_keyword_edges_linked', 0)}")
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
