from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from neo4j import GraphDatabase

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from graph_rag.common import json_ready, load_dotenv_if_present, read_neo4j_settings


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


def _safe_nonneg_float(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if math.isnan(parsed) or math.isinf(parsed):
        return float(default)
    if parsed < 0.0:
        return 0.0
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
    for i in range(len(a)):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    sim = dot / ((math.sqrt(na) * math.sqrt(nb)) + 1e-9)
    return _safe_unit_float(sim, default=0.0)


def _parse_domain_links(items: Any) -> List[Dict[str, Any]]:
    by_name: Dict[str, Dict[str, Any]] = {}
    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        name = _clean_text(item.get("domain")).lower()
        if not name:
            continue
        score = _safe_unit_float(item.get("score"), default=0.0)
        domain_weight = _safe_unit_float(item.get("domain_weight"), default=1.0)
        strength = score * domain_weight
        embedding = _coerce_vector(item.get("embedding"))
        prev = by_name.get(name)
        if prev is None:
            by_name[name] = {
                "domain": name,
                "strength": strength,
                "embedding": embedding,
            }
            continue
        if strength > float(prev.get("strength") or 0.0):
            prev["strength"] = strength
        if not prev.get("embedding") and embedding:
            prev["embedding"] = embedding
    return list(by_name.values())


def _best_domain_matches(
    source: Sequence[Dict[str, Any]],
    target: Sequence[Dict[str, Any]],
    *,
    min_cosine: float,
    source_key: str,
    target_key: str,
) -> Tuple[float, List[str], List[Dict[str, Any]]]:
    den = 0.0
    num = 0.0
    missing: List[str] = []
    pairs: List[Dict[str, Any]] = []

    safe_min_cosine = _safe_unit_float(min_cosine, default=0.5)

    for s in list(source or []):
        s_name = _clean_text(s.get("domain")).lower()
        if not s_name:
            continue
        s_strength = _safe_unit_float(s.get("strength"), default=0.0)
        s_vec = list(s.get("embedding") or [])
        if s_strength <= 0.0:
            continue

        den += s_strength
        best_sim = 0.0
        best_t_name = ""
        for t in list(target or []):
            t_name = _clean_text(t.get("domain")).lower()
            t_vec = list(t.get("embedding") or [])
            if not t_name or not t_vec or not s_vec:
                continue
            sim = _cosine_vec(s_vec, t_vec)
            if sim > best_sim:
                best_sim = sim
                best_t_name = t_name

        if best_sim >= safe_min_cosine and best_t_name:
            num += (s_strength * best_sim)
            pairs.append(
                {
                    source_key: s_name,
                    target_key: best_t_name,
                    "cosine_sim": best_sim,
                    "source_strength": s_strength,
                }
            )
        else:
            missing.append(s_name)

    coverage = 0.0 if den <= 0.0 else _safe_unit_float(num / den, default=0.0)
    return coverage, sorted(list(set(missing))), pairs


def _soft_domain_overlap_by_embedding(
    faculty_domains: Sequence[Dict[str, Any]],
    grant_domains: Sequence[Dict[str, Any]],
    *,
    min_cosine: float,
) -> Tuple[float, List[str], List[str], List[Dict[str, Any]]]:
    f_list = list(faculty_domains or [])
    g_list = list(grant_domains or [])
    if not f_list and not g_list:
        return 1.0, [], [], []
    if not f_list:
        g_only = sorted(list({_clean_text(x.get("domain")).lower() for x in g_list if _clean_text(x.get("domain"))}))
        return 0.0, [], g_only, []
    if not g_list:
        f_only = sorted(list({_clean_text(x.get("domain")).lower() for x in f_list if _clean_text(x.get("domain"))}))
        return 0.0, f_only, [], []

    f_cov, f_missing, f_pairs = _best_domain_matches(
        f_list,
        g_list,
        min_cosine=min_cosine,
        source_key="faculty_domain",
        target_key="grant_domain",
    )
    g_cov, g_missing, g_pairs = _best_domain_matches(
        g_list,
        f_list,
        min_cosine=min_cosine,
        source_key="grant_domain",
        target_key="faculty_domain",
    )
    overlap = _safe_unit_float((f_cov + g_cov) / 2.0, default=0.0)
    # Keep short list for debugging.
    domain_pairs = (f_pairs + g_pairs)[:20]
    return overlap, f_missing, g_missing, domain_pairs


def _fetch_v3_pairs(
    *,
    driver,
    database: str,
    faculty_id: Optional[int],
    faculty_email: str,
    min_domain_gate_score: float,
    min_edge_score: float,
    include_closed: bool,
) -> List[Dict[str, Any]]:
    records, _, _ = driver.execute_query(
        """
        MATCH (fk:FacultyKeyword)-[m:FACULTY_SPEC_MATCHES_GRANT_SPEC_V3]->(gk:GrantKeyword)
        WHERE
            ($faculty_id IS NULL OR m.scope_faculty_id = $faculty_id)
            AND ($faculty_email = '' OR toLower(m.scope_faculty_email) = $faculty_email)
            AND m.scope_opportunity_id IS NOT NULL
            AND coalesce(m.domain_gate_score, 0.0) >= $min_domain_gate_score
            AND coalesce(m.score, 0.0) >= $min_edge_score
        MATCH (g:Grant {opportunity_id: m.scope_opportunity_id})
        WITH
            fk, gk, m, g,
            toLower(coalesce(g.opportunity_status, '')) AS status_token,
            coalesce(toString(g.close_date), '') AS close_token
        WITH
            fk, gk, m, g, status_token,
            CASE
                WHEN close_token =~ '^\\d{4}-\\d{2}-\\d{2}.*$' THEN date(substring(close_token, 0, 10))
                ELSE NULL
            END AS close_dt
        WHERE
            $include_closed
            OR (
                NONE(token IN ['closed', 'archived', 'inactive', 'canceled'] WHERE status_token CONTAINS token)
                AND (close_dt IS NULL OR close_dt >= date())
            )
        CALL (fk, m) {
            OPTIONAL MATCH (fd:FacultyKeyword {bucket: 'domain'})-[r:FACULTY_DOMAIN_HAS_SPECIALIZATION]->(fk)
            WHERE r.scope_faculty_id = m.scope_faculty_id
            RETURN
                coalesce(max(coalesce(r.score, 0.0) * coalesce(r.domain_weight, 1.0)), 0.0) AS faculty_domain_spec_strength,
                collect(DISTINCT {
                    domain: toLower(coalesce(fd.value, '')),
                    score: coalesce(r.score, 0.0),
                    domain_weight: coalesce(r.domain_weight, 1.0),
                    embedding: fd.embedding
                }) AS faculty_domain_links
        }
        CALL (fk, m) {
            OPTIONAL MATCH (:FacultyTextChunk)-[r:FACULTY_CHUNK_SUPPORTS_SPECIALIZATION]->(fk)
            WHERE r.scope_faculty_id = m.scope_faculty_id
            RETURN coalesce(max(r.score), 0.0) AS faculty_chunk_support_strength
        }
        CALL (fk, m) {
            OPTIONAL MATCH (:FacultyPublication)-[r:FACULTY_PUBLICATION_SUPPORTS_SPECIALIZATION]->(fk)
            WHERE r.scope_faculty_id = m.scope_faculty_id
            RETURN coalesce(max(r.score), 0.0) AS faculty_publication_support_strength
        }
        CALL (gk, m) {
            OPTIONAL MATCH (gd:GrantKeyword {bucket: 'domain'})-[r:GRANT_DOMAIN_HAS_SPECIALIZATION]->(gk)
            WHERE r.scope_opportunity_id = m.scope_opportunity_id
            RETURN
                coalesce(max(coalesce(r.score, 0.0) * coalesce(r.domain_weight, 1.0)), 0.0) AS grant_domain_spec_strength,
                collect(DISTINCT {
                    domain: toLower(coalesce(gd.value, '')),
                    score: coalesce(r.score, 0.0),
                    domain_weight: coalesce(r.domain_weight, 1.0),
                    embedding: gd.embedding
                }) AS grant_domain_links
        }
        CALL (gk, m) {
            OPTIONAL MATCH (:GrantTextChunk)-[r:GRANT_CHUNK_SUPPORTS_SPECIALIZATION]->(gk)
            WHERE r.scope_opportunity_id = m.scope_opportunity_id
            RETURN coalesce(max(r.score), 0.0) AS grant_chunk_support_strength
        }
        RETURN
            m.scope_opportunity_id AS opportunity_id,
            coalesce(g.opportunity_title, g.title, '') AS opportunity_title,
            coalesce(g.agency_name, '') AS agency_name,
            coalesce(g.opportunity_status, '') AS opportunity_status,
            coalesce(toString(g.close_date), '') AS close_date,
            coalesce(m.faculty_keyword_value, fk.value, '') AS faculty_keyword_value,
            coalesce(m.faculty_keyword_section, fk.section, 'general') AS faculty_keyword_section,
            coalesce(m.grant_keyword_value, gk.value, '') AS grant_keyword_value,
            coalesce(m.grant_keyword_section, gk.section, 'general') AS grant_keyword_section,
            coalesce(m.score, 0.0) AS edge_score,
            coalesce(m.cosine_sim, 0.0) AS cosine_sim,
            coalesce(m.attention_score, 0.0) AS attention_score,
            coalesce(m.hybrid_sim, (coalesce(m.cosine_sim, 0.0) + coalesce(m.attention_score, 0.0)) / 2.0) AS hybrid_sim,
            coalesce(m.domain_gate_score, 0.0) AS domain_gate_score,
            coalesce(m.faculty_keyword_weight, 0.0) AS faculty_keyword_weight,
            coalesce(m.grant_keyword_weight, 0.0) AS grant_keyword_weight,
            coalesce(m.faculty_keyword_confidence, 0.0) AS faculty_keyword_confidence,
            coalesce(m.grant_keyword_confidence, 0.0) AS grant_keyword_confidence,
            faculty_domain_spec_strength,
            faculty_chunk_support_strength,
            faculty_publication_support_strength,
            grant_domain_spec_strength,
            grant_chunk_support_strength,
            faculty_domain_links,
            grant_domain_links
        ORDER BY opportunity_id ASC, edge_score DESC
        """,
        parameters_={
            "faculty_id": int(faculty_id) if faculty_id is not None else None,
            "faculty_email": _clean_text(faculty_email).lower(),
            "min_domain_gate_score": _safe_unit_float(min_domain_gate_score, default=0.3),
            "min_edge_score": _safe_unit_float(min_edge_score, default=0.0),
            "include_closed": bool(include_closed),
        },
        database_=database,
    )
    return [dict(row or {}) for row in records]


def retrieve_grants_by_v3_composite_score(
    *,
    faculty_id: Optional[int],
    faculty_email: str,
    top_k: int,
    top_pairs_for_rank: int,
    pairs_per_grant: int,
    min_domain_gate_score: float,
    domain_link_min_cosine: float,
    min_edge_score: float,
    include_closed: bool,
    missing_domain_penalty: float,
    coverage_bonus: float,
    w_edge: float,
    w_cosine: float,
    w_attention: float,
    w_hybrid: float,
    w_domain_score: float,
    w_fac_domain_spec: float,
    w_grant_domain_spec: float,
    w_fac_support: float,
    w_grant_support: float,
    uri: str = "",
    username: str = "",
    password: str = "",
    database: str = "",
) -> Dict[str, Any]:
    fid = int(faculty_id) if faculty_id is not None else None
    femail = _clean_text(faculty_email).lower()
    if fid is None and not femail:
        raise ValueError("Provide faculty_id or faculty_email.")

    safe_top_k = _safe_limit(top_k, default=20, minimum=1, maximum=2000)
    safe_top_pairs_for_rank = _safe_limit(top_pairs_for_rank, default=5, minimum=3, maximum=7)
    safe_pairs = _safe_limit(pairs_per_grant, default=20, minimum=1, maximum=500)
    safe_min_domain_gate = _safe_unit_float(min_domain_gate_score, default=0.3)
    safe_domain_link_min_cosine = _safe_unit_float(domain_link_min_cosine, default=0.3)
    safe_min_edge = _safe_unit_float(min_edge_score, default=0.0)
    safe_missing_penalty = _safe_unit_float(missing_domain_penalty, default=0.6)
    safe_coverage_bonus = _safe_nonneg_float(coverage_bonus, default=0.0)

    weights = {
        "edge": _safe_nonneg_float(w_edge, default=1.0),
        "cosine": _safe_nonneg_float(w_cosine, default=0.0),
        "attention": _safe_nonneg_float(w_attention, default=0.0),
        "hybrid": _safe_nonneg_float(w_hybrid, default=0.0),
        "domain_score": _safe_nonneg_float(w_domain_score, default=0.25),
        "fac_domain_spec": _safe_nonneg_float(w_fac_domain_spec, default=0.0),
        "grant_domain_spec": _safe_nonneg_float(w_grant_domain_spec, default=0.0),
        "fac_support": _safe_nonneg_float(w_fac_support, default=0.0),
        "grant_support": _safe_nonneg_float(w_grant_support, default=0.0),
    }

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
        pair_rows = _fetch_v3_pairs(
            driver=driver,
            database=settings.database,
            faculty_id=fid,
            faculty_email=femail,
            min_domain_gate_score=safe_min_domain_gate,
            min_edge_score=safe_min_edge,
            include_closed=bool(include_closed),
        )

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in pair_rows:
        oid = _clean_text(row.get("opportunity_id"))
        if not oid:
            continue

        g = grouped.get(oid)
        if g is None:
            g = {
                "opportunity_id": oid,
                "opportunity_title": _clean_text(row.get("opportunity_title")),
                "agency_name": _clean_text(row.get("agency_name")),
                "opportunity_status": _clean_text(row.get("opportunity_status")),
                "close_date": _clean_text(row.get("close_date")),
                "pair_rows": [],
                "sum_pair_score": 0.0,
                "sum_base_combo": 0.0,
                "sum_penalty_loss": 0.0,
                "sum_overlap": 0.0,
                "pair_count": 0,
                "faculty_keywords": set(),
                "grant_keywords": set(),
            }
            grouped[oid] = g

        edge_score = _safe_unit_float(row.get("edge_score"), default=0.0)
        cosine_sim = _safe_unit_float(row.get("cosine_sim"), default=0.0)
        attention_score = _safe_unit_float(row.get("attention_score"), default=0.0)
        hybrid_sim = _safe_unit_float(row.get("hybrid_sim"), default=(cosine_sim + attention_score) / 2.0)
        fac_domain_spec = _safe_unit_float(row.get("faculty_domain_spec_strength"), default=0.0)
        grant_domain_spec = _safe_unit_float(row.get("grant_domain_spec_strength"), default=0.0)
        fac_support = max(
            _safe_unit_float(row.get("faculty_chunk_support_strength"), default=0.0),
            _safe_unit_float(row.get("faculty_publication_support_strength"), default=0.0),
        )
        grant_support = _safe_unit_float(row.get("grant_chunk_support_strength"), default=0.0)

        fac_domains = _parse_domain_links(row.get("faculty_domain_links"))
        grant_domains = _parse_domain_links(row.get("grant_domain_links"))
        has_domain_evidence = bool(fac_domains and grant_domains)
        overlap, f_only, g_only, matched_domain_pairs = _soft_domain_overlap_by_embedding(
            fac_domains,
            grant_domains,
            min_cosine=safe_domain_link_min_cosine,
        )
        domain_score = overlap
        missing_ratio = 1.0 - overlap

        semantic_numer = (
            (weights["edge"] * edge_score)
            + (weights["cosine"] * cosine_sim)
            + (weights["attention"] * attention_score)
            + (weights["hybrid"] * hybrid_sim)
        )
        semantic_weight_sum = (
            float(weights["edge"])
            + float(weights["cosine"])
            + float(weights["attention"])
            + float(weights["hybrid"])
        )
        if semantic_weight_sum > 0.0:
            semantic_score = semantic_numer / semantic_weight_sum
        else:
            semantic_score = 0.0
        semantic_domain_score = semantic_score * domain_score

        support_score = (
            (weights["fac_support"] * fac_support)
            + (weights["grant_support"] * grant_support)
        )
        support_domain_score = support_score * math.sqrt(max(0.0, domain_score))

        base_combo = (
            semantic_domain_score
            + (weights["domain_score"] * domain_score)
            + (weights["fac_domain_spec"] * fac_domain_spec)
            + (weights["grant_domain_spec"] * grant_domain_spec)
            + support_domain_score
        )
        # Hard-zero only when we have domain evidence on both sides.
        domain_hard_blocked = bool(has_domain_evidence and (domain_score <= 1e-9))
        if domain_hard_blocked:
            pair_score = 0.0
        else:
            pair_score = base_combo
        penalty_loss = 0.0

        fac_kw = _clean_text(row.get("faculty_keyword_value"))
        grant_kw = _clean_text(row.get("grant_keyword_value"))
        g["faculty_keywords"].add(fac_kw.lower())
        g["grant_keywords"].add(grant_kw.lower())

        g["pair_rows"].append(
            {
                "faculty_keyword_value": fac_kw,
                "faculty_keyword_section": _clean_text(row.get("faculty_keyword_section")).lower() or "general",
                "grant_keyword_value": grant_kw,
                "grant_keyword_section": _clean_text(row.get("grant_keyword_section")).lower() or "general",
                "edge_score": edge_score,
                "cosine_sim": cosine_sim,
                "attention_score": attention_score,
                "hybrid_sim": hybrid_sim,
                "faculty_domain_spec_strength": fac_domain_spec,
                "grant_domain_spec_strength": grant_domain_spec,
                "faculty_support_strength": fac_support,
                "grant_support_strength": grant_support,
                "semantic_score": semantic_score,
                "semantic_weight_sum": semantic_weight_sum,
                "semantic_domain_score": semantic_domain_score,
                "support_score": support_score,
                "support_domain_score": support_domain_score,
                "domain_score": domain_score,
                "domain_overlap": overlap,
                "missing_domain_ratio": missing_ratio,
                "domain_penalty_factor": 1.0,
                "domain_hard_blocked": domain_hard_blocked,
                "has_domain_evidence": has_domain_evidence,
                "faculty_only_domains": f_only,
                "grant_only_domains": g_only,
                "matched_domain_pairs": matched_domain_pairs,
                "base_combo_score": base_combo,
                "pair_score": pair_score,
                "penalty_loss": penalty_loss,
            }
        )
        g["sum_pair_score"] += pair_score
        g["sum_base_combo"] += base_combo
        g["sum_penalty_loss"] += penalty_loss
        g["sum_overlap"] += overlap
        g["pair_count"] += 1

    grant_rows: List[Dict[str, Any]] = []
    for item in grouped.values():
        pairs_sorted = sorted(
            list(item["pair_rows"]),
            key=lambda x: float(x.get("pair_score") or 0.0),
            reverse=True,
        )

        fac_cov = len(item["faculty_keywords"])
        grant_cov = len(item["grant_keywords"])
        coverage_bonus_score = safe_coverage_bonus * float(fac_cov + grant_cov)
        top_pairs = pairs_sorted[:safe_top_pairs_for_rank]
        rank_core_score = float(sum(float(x.get("pair_score") or 0.0) for x in top_pairs))
        rank_score = rank_core_score + coverage_bonus_score
        avg_overlap = 0.0
        if int(item["pair_count"]) > 0:
            avg_overlap = float(item["sum_overlap"]) / float(item["pair_count"])

        grant_rows.append(
            {
                "opportunity_id": item["opportunity_id"],
                "opportunity_title": item["opportunity_title"],
                "agency_name": item["agency_name"],
                "opportunity_status": item["opportunity_status"],
                "close_date": item["close_date"],
                "rank_score": rank_score,
                "rank_core_score": rank_core_score,
                "rank_pair_count_used": len(top_pairs),
                "sum_pair_score": float(item["sum_pair_score"]),
                "coverage_bonus_score": coverage_bonus_score,
                "sum_base_combo": float(item["sum_base_combo"]),
                "sum_penalty_loss": float(item["sum_penalty_loss"]),
                "avg_domain_overlap": avg_overlap,
                "pair_count": int(item["pair_count"]),
                "faculty_keyword_coverage": fac_cov,
                "grant_keyword_coverage": grant_cov,
                "matched_pairs": pairs_sorted[:safe_pairs],
            }
        )

    grant_rows.sort(
        key=lambda x: (
            float(x.get("rank_score") or 0.0),
            float(x.get("rank_core_score") or 0.0),
            float(x.get("sum_pair_score") or 0.0),
            float(x.get("avg_domain_overlap") or 0.0),
            int(x.get("pair_count") or 0),
        ),
        reverse=True,
    )
    grant_rows = grant_rows[:safe_top_k]

    return {
        "params": {
            "faculty_id": fid,
            "faculty_email": femail,
            "top_k": safe_top_k,
            "top_pairs_for_rank": safe_top_pairs_for_rank,
            "pairs_per_grant": safe_pairs,
            "min_domain_gate_score": safe_min_domain_gate,
            "domain_link_min_cosine": safe_domain_link_min_cosine,
            "min_edge_score": safe_min_edge,
            "include_closed": bool(include_closed),
            "missing_domain_penalty": safe_missing_penalty,
            "missing_domain_penalty_ignored_in_current_formula": True,
            "coverage_bonus": safe_coverage_bonus,
            "weights": weights,
            "domain_link_strength_formula": "domain_link_strength = domain_spec_score * domain_weight",
            "domain_overlap_formula": "match faculty_domain <-> grant_domain when cosine(embedding) >= domain_link_min_cosine",
            "domain_hard_gate": "if has_domain_evidence and domain_score == 0 then pair_score = 0",
            "rank_formula": (
                "semantic_score = (w_edge*edge + w_cosine*cosine + w_attention*attention + w_hybrid*hybrid) "
                "/ (w_edge + w_cosine + w_attention + w_hybrid); "
                "semantic_domain_score = semantic_score * domain_score; "
                "support_score = weighted_sum(faculty_support, grant_support); "
                "support_domain_score = support_score * sqrt(domain_score); "
                "pair_score = semantic_domain_score + w_domain_score*domain_score + "
                "w_fac_domain_spec*faculty_domain_spec + w_grant_domain_spec*grant_domain_spec + support_domain_score; "
                "if has_domain_evidence and domain_score == 0 then pair_score = 0; "
                "grant_rank = sum(top_k(pair_score)) + coverage_bonus * (faculty_keyword_coverage + grant_keyword_coverage), "
                "k in [3..7]"
            ),
        },
        "totals": {
            "pair_rows": len(pair_rows),
            "grants": len(grant_rows),
        },
        "grants": grant_rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve top grants using V3 specialization match edges plus intrinsic "
            "domain/spec/chunk support scores, with domain-mismatch penalty."
        )
    )
    parser.add_argument("--faculty-id", type=int, default=0, help="Faculty ID filter.")
    parser.add_argument("--faculty-email", type=str, default="", help="Faculty email filter.")
    parser.add_argument("--top-k", type=int, default=20, help="Top K grants to return.")
    parser.add_argument(
        "--top-pairs-for-rank",
        type=int,
        default=5,
        help="Use only top-k pair_score values when computing grant rank (clamped to 3..7).",
    )
    parser.add_argument("--pairs-per-grant", type=int, default=5, help="Top matched pairs returned per grant.")
    parser.add_argument("--min-domain-gate-score", type=float, default=0.2, help="Minimum domain gate score.")
    parser.add_argument(
        "--domain-link-min-cosine",
        type=float,
        default=0.3,
        help="Only domain-link matches with cosine >= this threshold contribute to domain_score.",
    )
    parser.add_argument("--min-edge-score", type=float, default=0.0, help="Minimum V3 edge score.")
    parser.add_argument("--include-closed", action="store_true", help="Include closed grants.")
    parser.add_argument(
        "--missing-domain-penalty",
        type=float,
        default=0.99,
        help="Legacy arg (ignored in current domain-aware scoring formula).",
    )
    parser.add_argument(
        "--coverage-bonus",
        type=float,
        default=0.0,
        help="Optional additive bonus per unique matched keyword coverage.",
    )

    parser.add_argument("--w-edge", type=float, default=1.0, help="Weight for V3 edge score.")
    parser.add_argument("--w-cosine", type=float, default=1.0, help="Weight for cosine similarity.")
    parser.add_argument("--w-attention", type=float, default=0.0, help="Weight for attention score.")
    parser.add_argument("--w-hybrid", type=float, default=0.0, help="Weight for hybrid similarity.")
    parser.add_argument(
        "--w-domain-score",
        type=float,
        default=0.00,
        help="Weight for explicit domain overlap score (built from score * domain_weight).",
    )
    parser.add_argument(
        "--w-fac-domain-spec",
        type=float,
        default=0.9,
        help="Weight for faculty domain->specialization link strength.",
    )
    parser.add_argument(
        "--w-grant-domain-spec",
        type=float,
        default=0.9,
        help="Weight for grant domain->specialization link strength.",
    )
    parser.add_argument(
        "--w-fac-support",
        type=float,
        default=0.9,
        help="Weight for faculty chunk/publication->specialization support strength.",
    )
    parser.add_argument(
        "--w-grant-support",
        type=float,
        default=0.9,
        help="Weight for grant chunk->specialization support strength.",
    )

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

    payload = retrieve_grants_by_v3_composite_score(
        faculty_id=faculty_id,
        faculty_email=_clean_text(args.faculty_email).lower(),
        top_k=int(args.top_k or 20),
        top_pairs_for_rank=int(args.top_pairs_for_rank or 5),
        pairs_per_grant=int(args.pairs_per_grant or 20),
        min_domain_gate_score=float(args.min_domain_gate_score),
        domain_link_min_cosine=float(args.domain_link_min_cosine),
        min_edge_score=float(args.min_edge_score),
        include_closed=bool(args.include_closed),
        missing_domain_penalty=float(args.missing_domain_penalty),
        coverage_bonus=float(args.coverage_bonus),
        w_edge=float(args.w_edge),
        w_cosine=float(args.w_cosine),
        w_attention=float(args.w_attention),
        w_hybrid=float(args.w_hybrid),
        w_domain_score=float(args.w_domain_score),
        w_fac_domain_spec=float(args.w_fac_domain_spec),
        w_grant_domain_spec=float(args.w_grant_domain_spec),
        w_fac_support=float(args.w_fac_support),
        w_grant_support=float(args.w_grant_support),
        uri=str(args.uri or "").strip(),
        username=str(args.username or "").strip(),
        password=str(args.password or "").strip(),
        database=str(args.database or "").strip(),
    )

    if not bool(args.json_only):
        print("Grant retrieval from V3 composite scoring complete.")
        print(f"  grants returned : {payload.get('totals', {}).get('grants', 0)}")
        grants = list(payload.get("grants") or [])
        if grants:
            print("  top results:")
            for idx, row in enumerate(grants[:10], start=1):
                print(
                    f"    {idx:02d}. {row.get('opportunity_id', '')} | "
                    f"{float(row.get('rank_score') or 0.0):.4f} | {row.get('opportunity_title', '')}"
                )
        print()

    payload_top3 = dict(payload)
    payload_top3["grants"] = list(payload.get("grants") or [])[:3]
    payload_top3["totals"] = dict(payload.get("totals") or {})
    payload_top3["totals"]["grants"] = len(payload_top3["grants"])

    print(json.dumps(json_ready(payload_top3), indent=2))

    if grants:
        print("  top results:")
        for idx, row in enumerate(grants[:10], start=1):
            print(
                f"    {idx:02d}. {row.get('opportunity_id', '')} | "
                f"{float(row.get('rank_score') or 0.0):.4f} | {row.get('opportunity_title', '')}"
            )
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
