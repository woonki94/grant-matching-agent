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

from graph_rag.common import Neo4jSettings, json_ready, load_dotenv_if_present, read_neo4j_settings

FACULTY_SPEC_RELATIONS = [
    "HAS_SPECIALIZATION_KEYWORD",
    "HAS_RESEARCH_SPECIALIZATION",
    "HAS_APPLICATION_SPECIALIZATION",
]
GRANT_SPEC_RELATIONS = [
    "HAS_SPECIALIZATION_KEYWORD",
    "HAS_RESEARCH_SPECIALIZATION",
    "HAS_APPLICATION_SPECIALIZATION",
]

DEFAULT_MODEL_DIR = str(
    (
        Path(__file__).resolve().parents[2]
        / "train_cross_encoder"
        / "models"
        / "bge-reranker-base-finetuned"
        / "checkpoint-47002"
    ).resolve()
)


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


def _dedupe_nonempty(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values or []:
        token = _clean_text(value)
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(token)
    return out


def _truncate_text(value: Any, max_chars: int) -> str:
    text = _clean_text(value)
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _sigmoid(value: float) -> float:
    x = float(value)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _normalize_model_score(raw_score: float, *, score_transform: str) -> float:
    mode = _clean_text(score_transform).lower() or "clip"
    raw = float(raw_score)
    if mode == "sigmoid":
        return _safe_unit_float(_sigmoid(raw), default=0.0)
    if mode == "raw":
        return raw
    # default: clip to [0, 1] because this model was trained with MSE against [0,1] labels.
    return _safe_unit_float(raw, default=0.0)


def _resolve_model_dir(model_dir: str) -> Path:
    def _has_model_artifacts(p: Path) -> bool:
        if not p.exists() or not p.is_dir():
            return False
        has_config = (p / "config.json").exists()
        has_model = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
        has_tokenizer = (p / "tokenizer.json").exists() or (p / "tokenizer_config.json").exists()
        return bool(has_config and has_model and has_tokenizer)

    def _pick_best_checkpoint_dir(p: Path) -> Optional[Path]:
        cands = [x for x in p.glob("checkpoint-*") if x.is_dir()]
        if not cands:
            return None

        def _score(x: Path) -> Tuple[int, float]:
            name = x.name
            step = -1
            if name.startswith("checkpoint-"):
                try:
                    step = int(name.split("-", 1)[1])
                except Exception:
                    step = -1
            return (step, x.stat().st_mtime)

        cands.sort(key=_score, reverse=True)
        for c in cands:
            if _has_model_artifacts(c):
                return c.resolve()
        return None

    requested = _clean_text(model_dir) or DEFAULT_MODEL_DIR
    p = Path(requested).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Model directory not found: {p}")
    if _has_model_artifacts(p):
        return p
    ckpt = _pick_best_checkpoint_dir(p)
    if ckpt is not None:
        return ckpt
    raise FileNotFoundError(
        f"No loadable model artifacts found in {p}. "
        "Expected config + model + tokenizer files either in the directory or checkpoint-* subdirs."
    )


def _load_reranker_cpu(model_dir: Path):
    try:
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Missing inference dependencies. Install in your venv:\n"
            "pip install torch transformers sentencepiece"
        ) from exc

    tok_last_err: Optional[Exception] = None
    tokenizer = None
    tok_attempts = (
        {"use_fast": True},
        {"use_fast": False},
        {},
    )
    for kwargs in tok_attempts:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
                **kwargs,
            )
            break
        except Exception as e:
            tok_last_err = e

    if tokenizer is None:
        base_name = ""
        try:
            cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
            base_name = _clean_text(getattr(cfg, "_name_or_path", ""))
        except Exception:
            base_name = ""
        if base_name and base_name != str(model_dir):
            for kwargs in tok_attempts:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True, **kwargs)
                    break
                except Exception as e:
                    tok_last_err = e

    if tokenizer is None:
        raise RuntimeError(
            "Failed to load tokenizer for inference. "
            "Install tokenizer deps if needed: `pip install sentencepiece tiktoken`.\n"
            f"Model dir: {model_dir}\n"
            f"Last tokenizer error: {tok_last_err}"
        )

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), trust_remote_code=True)
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return tokenizer, model, device, torch


def _score_pairs_raw(
    *,
    tokenizer,
    model,
    device,
    torch_mod,
    pairs: Sequence[Tuple[str, str]],
    batch_size: int,
    max_length: int,
) -> List[float]:
    if not pairs:
        return []
    out: List[float] = []
    safe_batch = _safe_limit(batch_size, default=64, minimum=1, maximum=8192)
    safe_max_len = _safe_limit(max_length, default=256, minimum=32, maximum=4096)

    pair_list = list(pairs or [])
    for start in range(0, len(pair_list), safe_batch):
        batch = pair_list[start : start + safe_batch]
        queries = [_clean_text(x[0]) for x in batch]
        docs = [_clean_text(x[1]) for x in batch]
        enc = tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=safe_max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch_mod.no_grad():
            logits = model(**enc).logits
            if logits.ndim == 2 and logits.shape[-1] > 1:
                scores = logits[:, -1]
            else:
                scores = logits.squeeze(-1)
            out.extend([float(v) for v in scores.detach().cpu().tolist()])
    return out


def _list_faculty_emails(
    *,
    driver,
    settings: Neo4jSettings,
    limit: int,
    offset: int,
) -> List[str]:
    query = """
        MATCH (f:Faculty)
        WHERE f.email IS NOT NULL
        RETURN f.email AS email
        ORDER BY f.email ASC
        SKIP $offset
    """
    params: Dict[str, Any] = {"offset": max(0, int(offset or 0))}
    if int(limit or 0) > 0:
        query += "\nLIMIT $limit"
        params["limit"] = max(1, int(limit))
    records, _, _ = driver.execute_query(query, parameters_=params, database_=settings.database)
    return _dedupe_nonempty([_clean_text(row.get("email")).lower() for row in records])


def _list_grant_ids(
    *,
    driver,
    settings: Neo4jSettings,
    include_closed: bool,
    limit: int,
    offset: int,
) -> List[str]:
    query = """
        MATCH (g:Grant)
        WHERE g.opportunity_id IS NOT NULL
        WITH
            g,
            toLower(coalesce(g.opportunity_status, "")) AS status_token,
            coalesce(toString(g.close_date), "") AS close_token
        WITH
            g,
            status_token,
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
        RETURN toString(g.opportunity_id) AS opportunity_id
        ORDER BY opportunity_id ASC
        SKIP $offset
    """
    params: Dict[str, Any] = {
        "include_closed": bool(include_closed),
        "offset": max(0, int(offset or 0)),
    }
    if int(limit or 0) > 0:
        query += "\nLIMIT $limit"
        params["limit"] = max(1, int(limit))
    records, _, _ = driver.execute_query(query, parameters_=params, database_=settings.database)
    return _dedupe_nonempty([_clean_text(row.get("opportunity_id")) for row in records])


def _fetch_faculty_identity(
    *,
    driver,
    settings: Neo4jSettings,
    faculty_email: str,
) -> Optional[Dict[str, Any]]:
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})
        RETURN f.faculty_id AS faculty_id, toLower(f.email) AS email
        LIMIT 1
        """,
        parameters_={"email": _clean_text(faculty_email).lower()},
        database_=settings.database,
    )
    if not records:
        return None
    row = dict(records[0] or {})
    try:
        fid = int(row.get("faculty_id"))
    except Exception:
        return None
    return {
        "faculty_id": fid,
        "email": _clean_text(row.get("email")).lower(),
    }


def _fetch_faculty_spec_keywords(
    *,
    driver,
    settings: Neo4jSettings,
    faculty_email: str,
    min_keyword_confidence: float,
) -> List[Dict[str, Any]]:
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Faculty {email: $email})-[r]->(k:FacultyKeyword)
        WHERE type(r) IN $relations
          AND k.bucket = 'specialization'
          AND k.value IS NOT NULL
        CALL (f, k) {
            OPTIONAL MATCH (:FacultyTextChunk)-[s:FACULTY_CHUNK_SUPPORTS_SPECIALIZATION]->(k)
            WHERE s.scope_faculty_id = f.faculty_id
            RETURN coalesce(max(s.score), 0.0) AS chunk_conf
        }
        CALL (f, k) {
            OPTIONAL MATCH (:FacultyPublication)-[s:FACULTY_PUBLICATION_SUPPORTS_SPECIALIZATION]->(k)
            WHERE s.scope_faculty_id = f.faculty_id
            RETURN coalesce(max(s.score), 0.0) AS pub_conf
        }
        RETURN DISTINCT
            k.value AS keyword_value,
            toLower(coalesce(k.section, 'general')) AS keyword_section,
            coalesce(r.weight, 0.5) AS keyword_weight,
            CASE WHEN chunk_conf >= pub_conf THEN chunk_conf ELSE pub_conf END AS keyword_confidence
        ORDER BY keyword_value ASC
        """,
        parameters_={
            "email": _clean_text(faculty_email).lower(),
            "relations": FACULTY_SPEC_RELATIONS,
        },
        database_=settings.database,
    )
    safe_min_conf = _safe_unit_float(min_keyword_confidence, default=0.0)
    out: List[Dict[str, Any]] = []
    for row in records:
        item = dict(row or {})
        value = _clean_text(item.get("keyword_value"))
        if not value:
            continue
        conf = _safe_unit_float(item.get("keyword_confidence"), default=safe_min_conf)
        if conf < safe_min_conf:
            conf = safe_min_conf
        out.append(
            {
                "keyword_value": value,
                "keyword_section": _clean_text(item.get("keyword_section")).lower() or "general",
                "keyword_weight": _safe_unit_float(item.get("keyword_weight"), default=0.5),
                "keyword_confidence": conf,
            }
        )
    return out


def _fetch_grant_spec_keywords(
    *,
    driver,
    settings: Neo4jSettings,
    opportunity_id: str,
    min_keyword_confidence: float,
) -> List[Dict[str, Any]]:
    records, _, _ = driver.execute_query(
        """
        MATCH (g:Grant)
        WHERE toString(g.opportunity_id) = $opportunity_id
        MATCH (g)-[r]->(k:GrantKeyword)
        WHERE type(r) IN $relations
          AND k.bucket = 'specialization'
          AND k.value IS NOT NULL
        CALL (g, k) {
            OPTIONAL MATCH (:GrantTextChunk)-[s:GRANT_CHUNK_SUPPORTS_SPECIALIZATION]->(k)
            WHERE toString(s.scope_opportunity_id) = toString(g.opportunity_id)
            RETURN coalesce(max(s.score), 0.0) AS chunk_conf
        }
        RETURN DISTINCT
            k.value AS keyword_value,
            toLower(coalesce(k.section, 'general')) AS keyword_section,
            coalesce(r.weight, 0.5) AS keyword_weight,
            chunk_conf AS keyword_confidence
        ORDER BY keyword_value ASC
        """,
        parameters_={
            "opportunity_id": _clean_text(opportunity_id),
            "relations": GRANT_SPEC_RELATIONS,
        },
        database_=settings.database,
    )
    safe_min_conf = _safe_unit_float(min_keyword_confidence, default=0.0)
    out: List[Dict[str, Any]] = []
    for row in records:
        item = dict(row or {})
        value = _clean_text(item.get("keyword_value"))
        if not value:
            continue
        conf = _safe_unit_float(item.get("keyword_confidence"), default=safe_min_conf)
        if conf < safe_min_conf:
            conf = safe_min_conf
        out.append(
            {
                "keyword_value": value,
                "keyword_section": _clean_text(item.get("keyword_section")).lower() or "general",
                "keyword_weight": _safe_unit_float(item.get("keyword_weight"), default=0.5),
                "keyword_confidence": conf,
            }
        )
    return out


def _rank_spec_links_with_model(
    *,
    faculty_keywords: List[Dict[str, Any]],
    grant_keywords: List[Dict[str, Any]],
    tokenizer,
    model,
    device,
    torch_mod,
    batch_size: int,
    max_length: int,
    same_section_only: bool,
    max_pair_text_chars: int,
    min_score: float,
    score_transform: str,
    top_k_per_faculty_keyword: int,
    use_weighted_score: bool,
) -> Dict[str, Any]:
    safe_min_score = _safe_float(min_score, default=0.0)
    safe_top_k = int(top_k_per_faculty_keyword or 0)
    safe_max_chars = _safe_limit(max_pair_text_chars, default=300, minimum=50, maximum=12000)

    all_links: List[Dict[str, Any]] = []
    scored_pairs = 0

    for fac in faculty_keywords or []:
        fac_value = _clean_text(fac.get("keyword_value"))
        fac_section = _clean_text(fac.get("keyword_section")).lower() or "general"
        fac_weight = _safe_unit_float(fac.get("keyword_weight"), default=0.5)
        fac_conf = _safe_unit_float(fac.get("keyword_confidence"), default=0.0)
        if not fac_value:
            continue

        candidates: List[Dict[str, Any]] = []
        pair_inputs: List[Tuple[str, str]] = []
        for grant in grant_keywords or []:
            grant_value = _clean_text(grant.get("keyword_value"))
            if not grant_value:
                continue
            grant_section = _clean_text(grant.get("keyword_section")).lower() or "general"
            if bool(same_section_only) and fac_section != grant_section:
                continue
            candidates.append(grant)
            pair_inputs.append(
                (
                    _truncate_text(fac_value, safe_max_chars),
                    _truncate_text(grant_value, safe_max_chars),
                )
            )

        if not candidates:
            continue

        raw_scores = _score_pairs_raw(
            tokenizer=tokenizer,
            model=model,
            device=device,
            torch_mod=torch_mod,
            pairs=pair_inputs,
            batch_size=batch_size,
            max_length=max_length,
        )
        if not raw_scores:
            continue

        one_fac_rows: List[Dict[str, Any]] = []
        for grant, raw in zip(candidates, raw_scores):
            model_score = _normalize_model_score(raw, score_transform=score_transform)
            if model_score < safe_min_score:
                continue
            grant_value = _clean_text(grant.get("keyword_value"))
            grant_section = _clean_text(grant.get("keyword_section")).lower() or "general"
            grant_weight = _safe_unit_float(grant.get("keyword_weight"), default=0.5)
            grant_conf = _safe_unit_float(grant.get("keyword_confidence"), default=0.0)

            weighted_score = model_score * fac_weight * grant_weight * fac_conf * grant_conf
            edge_score = weighted_score if bool(use_weighted_score) else model_score

            one_fac_rows.append(
                {
                    "faculty_keyword_value": fac_value,
                    "faculty_keyword_section": fac_section,
                    "grant_keyword_value": grant_value,
                    "grant_keyword_section": grant_section,
                    "model_raw_score": float(raw),
                    "model_score": float(model_score),
                    "weighted_score": float(weighted_score),
                    "score": float(edge_score),
                    "faculty_keyword_weight": float(fac_weight),
                    "grant_keyword_weight": float(grant_weight),
                    "faculty_keyword_confidence": float(fac_conf),
                    "grant_keyword_confidence": float(grant_conf),
                }
            )

        scored_pairs += len(candidates)
        if not one_fac_rows:
            continue

        one_fac_rows.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        if safe_top_k > 0:
            one_fac_rows = one_fac_rows[:safe_top_k]
        all_links.extend(one_fac_rows)

    return {
        "rows": all_links,
        "stats": {
            "scored_pairs": int(scored_pairs),
            "links_kept": int(len(all_links)),
        },
    }


def _write_edges(
    *,
    driver,
    settings: Neo4jSettings,
    faculty_id: int,
    faculty_email: str,
    opportunity_id: str,
    rows: List[Dict[str, Any]],
    score_transform: str,
    min_score: float,
    use_weighted_score: bool,
    model_dir: str,
) -> None:
    driver.execute_query(
        """
        MATCH (:FacultyKeyword)-[r:FACULTY_SPEC_MATCHES_GRANT_SPEC {
            scope_faculty_id: $faculty_id,
            scope_opportunity_id: $opportunity_id
        }]->(:GrantKeyword)
        DELETE r
        """,
        parameters_={
            "faculty_id": int(faculty_id),
            "opportunity_id": _clean_text(opportunity_id),
        },
        database_=settings.database,
    )

    if not rows:
        return

    driver.execute_query(
        """
        UNWIND $rows AS row

        MATCH (f:Faculty {faculty_id: $faculty_id})
        MATCH (g:Grant)
        WHERE toString(g.opportunity_id) = $opportunity_id

        MATCH (f)-[:HAS_SPECIALIZATION_KEYWORD|HAS_RESEARCH_SPECIALIZATION|HAS_APPLICATION_SPECIALIZATION]->(fk:FacultyKeyword)
        WHERE fk.bucket = 'specialization'
          AND fk.value = row.faculty_keyword_value
          AND toLower(coalesce(fk.section, 'general')) = row.faculty_keyword_section

        MATCH (g)-[:HAS_SPECIALIZATION_KEYWORD|HAS_RESEARCH_SPECIALIZATION|HAS_APPLICATION_SPECIALIZATION]->(gk:GrantKeyword)
        WHERE gk.bucket = 'specialization'
          AND gk.value = row.grant_keyword_value
          AND toLower(coalesce(gk.section, 'general')) = row.grant_keyword_section

        MERGE (fk)-[r:FACULTY_SPEC_MATCHES_GRANT_SPEC {
            scope_faculty_id: $faculty_id,
            scope_faculty_email: $faculty_email,
            scope_opportunity_id: $opportunity_id,
            faculty_keyword_value: row.faculty_keyword_value,
            faculty_keyword_section: row.faculty_keyword_section,
            grant_keyword_value: row.grant_keyword_value,
            grant_keyword_section: row.grant_keyword_section
        }]->(gk)
        SET
            r.score = row.score,
            r.model_score = row.model_score,
            r.weighted_score = row.weighted_score,
            r.model_raw_score = row.model_raw_score,
            r.faculty_keyword_weight = row.faculty_keyword_weight,
            r.grant_keyword_weight = row.grant_keyword_weight,
            r.faculty_keyword_confidence = row.faculty_keyword_confidence,
            r.grant_keyword_confidence = row.grant_keyword_confidence,
            r.score_transform = $score_transform,
            r.min_score = $min_score,
            r.use_weighted_score = $use_weighted_score,
            r.model_dir = $model_dir,
            r.method = 'finetuned_bge_reranker_base',
            r.updated_at = datetime()
        """,
        parameters_={
            "faculty_id": int(faculty_id),
            "faculty_email": _clean_text(faculty_email).lower(),
            "opportunity_id": _clean_text(opportunity_id),
            "rows": rows,
            "score_transform": _clean_text(score_transform).lower() or "clip",
            "min_score": float(min_score),
            "use_weighted_score": bool(use_weighted_score),
            "model_dir": _clean_text(model_dir),
        },
        database_=settings.database,
    )


def run_finetuned_spec_keyword_linker(
    *,
    faculty_emails: Sequence[str],
    grant_ids: Sequence[str],
    all_faculty: bool,
    all_grants: bool,
    include_closed: bool,
    limit: int,
    offset: int,
    model_dir: str,
    batch_size: int,
    max_length: int,
    min_score: float,
    score_transform: str,
    same_section_only: bool,
    top_k_per_faculty_keyword: int,
    min_keyword_confidence: float,
    max_pair_text_chars: int,
    use_weighted_score: bool,
    write_edges: bool,
    uri: str = "",
    username: str = "",
    password: str = "",
    database: str = "",
) -> Dict[str, Any]:
    load_dotenv_if_present()
    settings = read_neo4j_settings(uri=uri, username=username, password=password, database=database)
    resolved_model_dir = _resolve_model_dir(model_dir)
    tokenizer, model, device, torch_mod = _load_reranker_cpu(resolved_model_dir)

    safe_limit = _safe_limit(limit, default=0, minimum=0, maximum=1_000_000)
    safe_offset = _safe_limit(offset, default=0, minimum=0, maximum=1_000_000)

    targets_faculty = _dedupe_nonempty([_clean_text(x).lower() for x in list(faculty_emails or [])])
    targets_grants = _dedupe_nonempty([_clean_text(x) for x in list(grant_ids or [])])

    with GraphDatabase.driver(settings.uri, auth=(settings.username, settings.password)) as driver:
        driver.verify_connectivity()

        if bool(all_faculty):
            targets_faculty = _list_faculty_emails(
                driver=driver,
                settings=settings,
                limit=safe_limit,
                offset=safe_offset,
            )

        if bool(all_grants) or not targets_grants:
            targets_grants = _list_grant_ids(
                driver=driver,
                settings=settings,
                include_closed=bool(include_closed),
                limit=safe_limit,
                offset=safe_offset,
            )

        if not targets_faculty:
            return {
                "status": "skipped",
                "reason": "no_faculty_targets",
            }
        if not targets_grants:
            return {
                "status": "skipped",
                "reason": "no_grant_targets",
            }

        grant_cache: Dict[str, List[Dict[str, Any]]] = {}
        pair_summaries: List[Dict[str, Any]] = []

        for faculty_email in targets_faculty:
            fac = _fetch_faculty_identity(driver=driver, settings=settings, faculty_email=faculty_email)
            if not fac:
                pair_summaries.append(
                    {
                        "faculty_email": _clean_text(faculty_email).lower(),
                        "status": "faculty_not_found",
                    }
                )
                continue

            faculty_keywords = _fetch_faculty_spec_keywords(
                driver=driver,
                settings=settings,
                faculty_email=faculty_email,
                min_keyword_confidence=float(min_keyword_confidence),
            )

            for opportunity_id in targets_grants:
                oid = _clean_text(opportunity_id)
                if not oid:
                    continue
                if oid not in grant_cache:
                    grant_cache[oid] = _fetch_grant_spec_keywords(
                        driver=driver,
                        settings=settings,
                        opportunity_id=oid,
                        min_keyword_confidence=float(min_keyword_confidence),
                    )
                grant_keywords = grant_cache[oid]

                ranked = _rank_spec_links_with_model(
                    faculty_keywords=faculty_keywords,
                    grant_keywords=grant_keywords,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    torch_mod=torch_mod,
                    batch_size=int(batch_size),
                    max_length=int(max_length),
                    same_section_only=bool(same_section_only),
                    max_pair_text_chars=int(max_pair_text_chars),
                    min_score=float(min_score),
                    score_transform=score_transform,
                    top_k_per_faculty_keyword=int(top_k_per_faculty_keyword),
                    use_weighted_score=bool(use_weighted_score),
                )
                rows = list(ranked.get("rows") or [])
                stats = dict(ranked.get("stats") or {})

                if bool(write_edges):
                    _write_edges(
                        driver=driver,
                        settings=settings,
                        faculty_id=int(fac["faculty_id"]),
                        faculty_email=_clean_text(faculty_email).lower(),
                        opportunity_id=oid,
                        rows=rows,
                        score_transform=score_transform,
                        min_score=float(min_score),
                        use_weighted_score=bool(use_weighted_score),
                        model_dir=str(resolved_model_dir),
                    )

                scores = [float(x.get("score") or 0.0) for x in rows]
                pair_summaries.append(
                    {
                        "faculty_id": int(fac["faculty_id"]),
                        "faculty_email": _clean_text(faculty_email).lower(),
                        "opportunity_id": oid,
                        "faculty_keyword_count": len(faculty_keywords),
                        "grant_keyword_count": len(grant_keywords),
                        "scored_pairs": int(stats.get("scored_pairs") or 0),
                        "edge_count": len(rows),
                        "max_score": max(scores) if scores else 0.0,
                        "mean_score": (sum(scores) / len(scores)) if scores else 0.0,
                    }
                )

    return {
        "params": {
            "all_faculty": bool(all_faculty),
            "all_grants": bool(all_grants),
            "include_closed": bool(include_closed),
            "limit": int(safe_limit),
            "offset": int(safe_offset),
            "model_dir": str(resolved_model_dir),
            "batch_size": _safe_limit(batch_size, default=64, minimum=1, maximum=8192),
            "max_length": _safe_limit(max_length, default=256, minimum=32, maximum=4096),
            "min_score": float(min_score),
            "score_transform": _clean_text(score_transform).lower() or "clip",
            "same_section_only": bool(same_section_only),
            "top_k_per_faculty_keyword": int(top_k_per_faculty_keyword or 0),
            "min_keyword_confidence": _safe_unit_float(min_keyword_confidence, default=0.0),
            "max_pair_text_chars": _safe_limit(max_pair_text_chars, default=300, minimum=50, maximum=12000),
            "use_weighted_score": bool(use_weighted_score),
            "write_edges": bool(write_edges),
            "edge_label": "FACULTY_SPEC_MATCHES_GRANT_SPEC",
        },
        "totals": {
            "pairs_processed": len(pair_summaries),
            "scored_pairs": sum(int(x.get("scored_pairs") or 0) for x in pair_summaries),
            "edges_written": sum(int(x.get("edge_count") or 0) for x in pair_summaries),
        },
        "pairs": pair_summaries,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Link Faculty specialization keywords to Grant specialization keywords using "
            "a fine-tuned cross-encoder checkpoint (CPU inference)."
        )
    )
    parser.add_argument("--faculty-email", action="append", default=[], help="Target faculty email (repeatable).")
    parser.add_argument("--grant-id", action="append", default=[], help="Target grant opportunity_id (repeatable).")
    parser.add_argument("--all-faculty", action="store_true", help="Process all faculty in Neo4j.")
    parser.add_argument("--all-grants", action="store_true", help="Process all grants in Neo4j.")
    parser.add_argument("--include-closed", action="store_true", help="Include closed grants when listing all grants.")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows when using --all-* (0 = all).")
    parser.add_argument("--offset", type=int, default=0, help="Offset when using --all-*.")

    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR, help="Fine-tuned model directory or checkpoint path.")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size (CPU).")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max_length.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum normalized model score to keep.")
    parser.add_argument(
        "--score-transform",
        type=str,
        choices=["clip", "sigmoid", "raw"],
        default="clip",
        help="How to normalize model raw output into model_score.",
    )
    parser.add_argument("--same-section-only", action="store_true", help="Only compare same section pairs.")
    parser.add_argument(
        "--top-k-per-faculty-keyword",
        type=int,
        default=0,
        help="Keep top-K links per faculty keyword (0 = keep all).",
    )
    parser.add_argument(
        "--min-keyword-confidence",
        type=float,
        default=0.0,
        help="Lower bound when keyword confidence is missing.",
    )
    parser.add_argument("--max-pair-text-chars", type=int, default=300, help="Max chars per keyword text fed to model.")
    parser.add_argument("--use-weighted-score", action="store_true", help="Set edge score=weighted_score instead of model_score.")

    parser.add_argument("--skip-write", action="store_true", help="Compute only; do not write edges.")
    parser.add_argument("--json-only", action="store_true", help="Print only JSON output.")

    parser.add_argument("--uri", type=str, default="", help="Neo4j URI. Fallback: NEO4J_URI")
    parser.add_argument("--username", type=str, default="", help="Neo4j username. Fallback: NEO4J_USERNAME")
    parser.add_argument("--password", type=str, default="", help="Neo4j password. Fallback: NEO4J_PASSWORD")
    parser.add_argument("--database", type=str, default="", help="Neo4j database. Fallback: NEO4J_DATABASE or neo4j")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if not bool(args.all_faculty) and not list(args.faculty_email or []):
        raise SystemExit("Provide --faculty-email (repeatable) or use --all-faculty.")

    payload = run_finetuned_spec_keyword_linker(
        faculty_emails=list(args.faculty_email or []),
        grant_ids=list(args.grant_id or []),
        all_faculty=bool(args.all_faculty),
        all_grants=bool(args.all_grants),
        include_closed=bool(args.include_closed),
        limit=int(args.limit or 0),
        offset=int(args.offset or 0),
        model_dir=_clean_text(args.model_dir) or DEFAULT_MODEL_DIR,
        batch_size=int(args.batch_size or 64),
        max_length=int(args.max_length or 256),
        min_score=float(args.min_score),
        score_transform=_clean_text(args.score_transform).lower() or "clip",
        same_section_only=bool(args.same_section_only),
        top_k_per_faculty_keyword=int(args.top_k_per_faculty_keyword or 0),
        min_keyword_confidence=float(args.min_keyword_confidence),
        max_pair_text_chars=int(args.max_pair_text_chars or 300),
        use_weighted_score=bool(args.use_weighted_score),
        write_edges=not bool(args.skip_write),
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )

    if not args.json_only:
        totals = payload.get("totals", {})
        print("Fine-tuned spec keyword linking complete.")
        print(f"  pairs processed : {totals.get('pairs_processed', 0)}")
        print(f"  scored pairs    : {totals.get('scored_pairs', 0)}")
        print(f"  edges written   : {totals.get('edges_written', 0)}")
        print()

    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
