from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.embedder import cosine_sim_matrix, embed_texts


def coerce_keyword_sections(kw_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure research/application sections are dicts, parsing JSON-strings when needed."""
    out = dict(kw_dict or {})
    for section in ("research", "application"):
        if isinstance(out.get(section), str):
            out[section] = json.loads(out[section])
    return out


def apply_weighted_specializations(*, keywords: Dict[str, Any], weighted: Any) -> Dict[str, Any]:
    out = dict(keywords or {})
    out["research"] = dict(out.get("research") or {})
    out["application"] = dict(out.get("application") or {})
    out["research"]["specialization"] = [x.model_dump() for x in (getattr(weighted, "research", None) or [])]
    out["application"]["specialization"] = [x.model_dump() for x in (getattr(weighted, "application", None) or [])]
    return out


def extract_domains_from_keywords(kw: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    r = (kw.get("research") or {}).get("domain") or []
    a = (kw.get("application") or {}).get("domain") or []
    return list(r), list(a)


def extract_specializations(kw: dict) -> dict:
    kw = kw or {}
    out = {"research": [], "application": []}

    for sec in ("research", "application"):
        specs = (kw.get(sec) or {}).get("specialization") or []
        for s in specs:
            if isinstance(s, dict) and "t" in s:
                out[sec].append(
                    {
                        "t": str(s["t"]),
                        "w": float(s.get("w", 1.0)),
                    }
                )
            elif isinstance(s, str):
                out[sec].append(
                    {
                        "t": str(s),
                        "w": 1.0,
                    }
                )

    return out


def specialization_text_sections(kw: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Normalize keyword payload into text-only specialization sections.

    Output shape:
    {
      "research": ["spec text", ...],
      "application": ["spec text", ...],
    }
    """
    specs = extract_specializations(kw or {})
    return {
        "research": [str(item.get("t") or "").strip() for item in list(specs.get("research") or []) if str(item.get("t") or "").strip()],
        "application": [str(item.get("t") or "").strip() for item in list(specs.get("application") or []) if str(item.get("t") or "").strip()],
    }


def keywords_for_matching(kw: dict) -> dict:
    specs = extract_specializations(kw)
    return {
        sec: {
            "domain": (kw.get(sec) or {}).get("domain") or [],
            "specialization": [s["t"] for s in specs[sec]],
        }
        for sec in ("research", "application")
    }


def requirements_indexed(kw: dict) -> dict:
    specs = extract_specializations(kw)
    out = {"application": {}, "research": {}}

    for sec in ("application", "research"):
        for i, s in enumerate(specs[sec]):
            out[sec][str(i)] = s["t"]

    return out


def keyword_inventory_for_rerank(kw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a flat keyword inventory for rerank payloads.

    Output shape:
    {
      "domain": [str, ...],
      "specialization": { "<keyword>": "<weight_str_2dp>", ... }
    }

    Rules:
    - Domain has no weights.
    - Specialization missing weights default to 0.0.
    """
    kw_norm = coerce_keyword_sections(dict(kw or {}))
    r_domains, a_domains = extract_domains_from_keywords(kw_norm)

    domain: List[str] = []
    seen_domain = set()
    for item in list(r_domains or []) + list(a_domains or []):
        text = " ".join(str(item or "").split()).strip()
        if not text or text in seen_domain:
            continue
        seen_domain.add(text)
        domain.append(text)

    spec_map: Dict[str, str] = {}
    seen_spec = set()
    for sec in ("research", "application"):
        specs = ((kw_norm.get(sec) or {}).get("specialization") or [])
        for item in list(specs):
            if isinstance(item, dict):
                text = str(item.get("t") if item.get("t") is not None else item.get("text") or "").strip()
                try:
                    weight = float(item.get("w", 0.0))
                except Exception:
                    weight = 0.0
            else:
                text = str(item or "").strip()
                weight = 0.0
            if not text or text in seen_spec:
                continue
            seen_spec.add(text)
            spec_map[text] = f"{weight:.2f}"

    return {
        "domain": domain,
        "specialization": spec_map,
    }



def extract_requirement_specs(opp_ctx: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Extract requirement text/weight by section/index from opportunity keyword payload."""
    out: Dict[str, Dict[int, Dict[str, Any]]] = {"application": {}, "research": {}}
    kw = (opp_ctx.get("keywords") or {}) if isinstance(opp_ctx, dict) else {}

    for sec in ("application", "research"):
        sec_obj = kw.get(sec) if isinstance(kw, dict) else None
        if not isinstance(sec_obj, dict):
            continue
        specs = sec_obj.get("specialization")
        if not isinstance(specs, list):
            continue
        for i, item in enumerate(specs):
            if not isinstance(item, dict):
                continue
            out[sec][i] = {
                "text": str(item.get("t") or f"{sec} requirement {i}"),
                "weight": float(item.get("w") or 0.0),
            }
    return out


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _short_excerpt(text: Any, *, max_chars: int = 260) -> str:
    s = " ".join(str(text or "").split()).strip()
    if not s:
        return ""
    cap = max(80, int(max_chars or 260))
    if len(s) <= cap:
        return s
    clipped = s[:cap].rstrip()
    cut = clipped.rfind(" ")
    if cut >= int(cap * 0.6):
        return clipped[:cut].rstrip()
    return clipped


def build_specialization_source_catalog(
    context: Dict[str, Any],
    *,
    max_items: int = 120,
    max_excerpt_chars: int = 260,
) -> List[Dict[str, Any]]:
    """
    Build a compact source catalog for LLM specialization-source mapping.
    """
    out: List[Dict[str, Any]] = []
    seen_keys = set()
    ctx = context or {}

    def _append_source(
        *,
        source_type: str,
        source_pk: Any,
        text: Any,
    ) -> None:
        try:
            pk = int(source_pk)
        except Exception:
            return
        snippet = _short_excerpt(text, max_chars=max_excerpt_chars)
        if not snippet:
            return
        dedup_key = (
            str(source_type),
            str(pk),
            _normalize_text(snippet),
        )
        if dedup_key in seen_keys:
            return
        seen_keys.add(dedup_key)
        row: Dict[str, Any] = {
            "type": source_type,
            "id": pk,
            "excerpt": snippet,
        }
        out.append(row)

    for item in list(ctx.get("additional_infos") or []):
        _append_source(
            source_type="additional_info_chunk",
            source_pk=item.get("id", item.get("row_id")),
            text=item.get("content"),
        )
        if len(out) >= max_items:
            return out

    for item in list(ctx.get("additional_info_extracted") or []):
        _append_source(
            source_type="additional_info_chunk",
            source_pk=item.get("id", item.get("row_id")),
            text=item.get("content"),
        )
        if len(out) >= max_items:
            return out

    for item in list(ctx.get("attachments_extracted") or []):
        _append_source(
            source_type="attachment_chunk",
            source_pk=item.get("id", item.get("row_id")),
            text=item.get("content"),
        )
        if len(out) >= max_items:
            return out

    for pub in list(ctx.get("publications") or []):
        title = str(pub.get("title") or "").strip()
        abstract = str(pub.get("abstract") or "").strip()
        body = f"{title}. {abstract}".strip()
        _append_source(
            source_type="publication",
            source_pk=pub.get("id"),
            text=body,
        )
        if len(out) >= max_items:
            return out

    return out


def _merge_one_specialization_sources(
    *,
    spec_item: Dict[str, Any],
    llm_item: Dict[str, Any],
    source_by_key: Dict[Tuple[int, str], Dict[str, Any]],
    max_sources: int,
) -> Dict[str, Any]:
    out = dict(spec_item or {})
    out.setdefault("t", str(out.get("text") or ""))
    out.setdefault("w", 1.0)

    refs = list(llm_item.get("sources") or []) if isinstance(llm_item, dict) else []
    sources_out: List[Dict[str, Any]] = []
    seen = set()
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        try:
            rid = int(ref.get("id"))
        except Exception:
            continue
        rtype = str(ref.get("type") or "").strip()
        if not rtype:
            continue
        key = (rid, rtype)
        if key in seen:
            continue
        base = source_by_key.get(key)
        if not base:
            continue
        seen.add(key)
        try:
            score = float(ref.get("score", 0.5))
        except Exception:
            score = 0.5
        score = max(0.0, min(1.0, score))
        sources_out.append(
            {
                "id": rid,
                "type": rtype,
                "score": score,
            }
        )
        if len(sources_out) >= max(1, int(max_sources or 1)):
            break
    out["sources"] = sources_out
    return out


def attach_specialization_sources_from_llm(
    *,
    keywords: Dict[str, Any],
    llm_sources: Dict[str, Any],
    source_catalog: List[Dict[str, Any]],
    max_sources_per_specialization: int = 4,
) -> Dict[str, Any]:
    """
    Merge LLM-produced id/type/score mappings back into keyword payload.
    """
    out = dict(keywords or {})
    source_by_key: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for row in list(source_catalog or []):
        try:
            rid = int(row.get("id"))
        except Exception:
            continue
        rtype = str(row.get("type") or "").strip()
        if not rtype:
            continue
        source_by_key[(rid, rtype)] = dict(row)

    for section in ("research", "application"):
        sec = dict(out.get(section) or {})
        specs = list(sec.get("specialization") or [])
        llm_rows = list((llm_sources.get(section) if isinstance(llm_sources, dict) else None) or [])
        llm_by_t = {
            _normalize_text(row.get("t")): row
            for row in llm_rows
            if isinstance(row, dict) and str(row.get("t") or "").strip()
        }
        merged_specs: List[Dict[str, Any]] = []
        for spec in specs:
            if isinstance(spec, dict):
                spec_text = str(spec.get("t") or spec.get("text") or "").strip()
                base_spec = dict(spec)
            else:
                spec_text = str(spec or "").strip()
                base_spec = {"t": spec_text, "w": 1.0}
            if not spec_text:
                continue
            llm_item = llm_by_t.get(_normalize_text(spec_text), {})
            merged_specs.append(
                _merge_one_specialization_sources(
                    spec_item=base_spec,
                    llm_item=llm_item,
                    source_by_key=source_by_key,
                    max_sources=max_sources_per_specialization,
                )
            )
        sec["specialization"] = merged_specs
        out[section] = sec
    return out


def map_specialization_sources_by_cosine(
    *,
    keywords: Dict[str, Any],
    source_catalog: List[Dict[str, Any]],
    embedding_client: Optional[Any] = None,
    max_sources_per_specialization: int = 4,
    min_similarity: float = 0.10,
) -> Dict[str, Any]:
    """
    Build specialization->sources mapping using cosine similarity over embeddings.

    Output shape matches SpecializationSourcesOut.model_dump():
    {
      "research": [{"t": "...", "sources": [{"id": 1, "type": "publication", "score": 0.42}]}],
      "application": [...]
    }
    """
    spec_sections = specialization_text_sections(keywords or {})
    spec_rows: List[Dict[str, Any]] = []
    for section in ("research", "application"):
        for text in list(spec_sections.get(section) or []):
            t = str(text or "").strip()
            if not t:
                continue
            spec_rows.append({"section": section, "t": t})

    catalog_rows: List[Dict[str, Any]] = []
    for row in list(source_catalog or []):
        try:
            rid = int(row.get("id"))
        except Exception:
            continue
        rtype = str(row.get("type") or "").strip()
        excerpt = str(row.get("excerpt") or "").strip()
        if not rtype or not excerpt:
            continue
        catalog_rows.append(
            {
                "id": rid,
                "type": rtype,
                "excerpt": excerpt,
            }
        )

    out: Dict[str, Any] = {"research": [], "application": []}
    if not spec_rows or not catalog_rows:
        for row in spec_rows:
            out[row["section"]].append({"t": row["t"], "sources": []})
        return out

    spec_texts = [row["t"] for row in spec_rows]
    src_texts = [row["excerpt"] for row in catalog_rows]

    # Naive parallelization: embed specialization texts and source excerpts concurrently.
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_spec = pool.submit(embed_texts, spec_texts, embedding_client=embedding_client)
        fut_src = pool.submit(embed_texts, src_texts, embedding_client=embedding_client)
        spec_vecs = fut_spec.result()
        src_vecs = fut_src.result()
    sims = cosine_sim_matrix(spec_vecs, src_vecs)

    keep_k = max(1, int(max_sources_per_specialization or 1))
    threshold = float(min_similarity)

    for i, spec in enumerate(spec_rows):
        row_sims = np.asarray(sims[i], dtype=np.float32).reshape(-1)
        ranked_idx = np.argsort(row_sims)[::-1]
        refs: List[Dict[str, Any]] = []
        seen = set()
        for j in ranked_idx:
            sim = float(row_sims[j])
            if sim < threshold:
                break
            src = catalog_rows[int(j)]
            key = (int(src["id"]), str(src["type"]))
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                {
                    "id": int(src["id"]),
                    "type": str(src["type"]),
                    "score": max(0.0, min(1.0, float(round(sim, 6)))),
                }
            )
            if len(refs) >= keep_k:
                break
        out[spec["section"]].append(
            {
                "t": spec["t"],
                "sources": refs,
            }
        )
    return out
