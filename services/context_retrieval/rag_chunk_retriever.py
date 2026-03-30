from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from db.models.faculty import Faculty, FacultyAdditionalInfo
from db.models.opportunity import Opportunity
from utils.content_extractor import load_extracted_content
from utils.embedder import embed_texts

WORD_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "to", "with", "using", "use",
}


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _normalize_text_lower(text: Any) -> str:
    return _normalize_text(text).lower()


def _tokenize_for_keyword_match(text: Any) -> List[str]:
    tokens = [t for t in WORD_RE.findall(_normalize_text_lower(text)) if t]
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def _query_match_score(*, query_text: str, chunk_text: str) -> float:
    q_norm = _normalize_text_lower(query_text)
    c_norm = _normalize_text_lower(chunk_text)
    if not q_norm or not c_norm:
        return 0.0

    q_tokens = _tokenize_for_keyword_match(q_norm)
    if not q_tokens:
        return 0.0
    c_tokens = set(_tokenize_for_keyword_match(c_norm))
    if not c_tokens:
        return 0.0

    overlap = sum(1 for tok in q_tokens if tok in c_tokens)
    overlap_ratio = float(overlap) / float(len(q_tokens))

    phrase_bonus = 0.0
    if q_norm in c_norm:
        phrase_bonus = 0.30
    elif len(q_tokens) >= 3:
        first_phrase = " ".join(q_tokens[: min(5, len(q_tokens))])
        if first_phrase and first_phrase in c_norm:
            phrase_bonus = 0.15

    score = overlap_ratio + phrase_bonus
    return max(0.0, min(1.0, float(score)))


def _score_block_against_spec_queries(
    *,
    block: Dict[str, Any],
    spec_queries: List[Dict[str, Any]],
) -> Tuple[float, Optional[Dict[str, Any]]]:
    text = str(block.get("content") or "")
    if not text.strip():
        return 0.0, None
    best_score = 0.0
    best_meta: Optional[Dict[str, Any]] = None
    for q in list(spec_queries or []):
        q_text = str(q.get("text") or "").strip()
        if not q_text:
            continue
        lexical = _query_match_score(query_text=q_text, chunk_text=text)
        if lexical <= 0.0:
            continue
        coverage = float(q.get("coverage_score") or 0.0)
        weighted = lexical * (0.65 + (0.35 * max(0.0, min(1.0, coverage))))
        if weighted > best_score:
            best_score = float(weighted)
            best_meta = {
                "text": q_text,
                "section": str(q.get("section") or ""),
                "idx": int(q.get("idx") or 0),
                "coverage_score": float(round(coverage, 6)),
                "lexical_score": float(round(lexical, 6)),
            }
    return best_score, best_meta


def _build_source_key(block: Dict[str, Any], *, use_title: bool) -> str:
    url = str(block.get("url") or "").strip()
    title = str(block.get("title") or "").strip() if use_title else ""
    key = f"{url}||{title}".strip("|")
    if key:
        return key
    return f"row:{int(block.get('id') or 0)}"


def _rank_blocks_by_specializations(
    *,
    blocks: List[Dict[str, Any]],
    spec_queries: List[Dict[str, Any]],
    top_k_per_source: int,
    max_total: int,
    min_score: float,
    use_title_in_source_key: bool,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for block in list(blocks or []):
        score, match_meta = _score_block_against_spec_queries(
            block=block,
            spec_queries=spec_queries,
        )
        if score < float(min_score):
            continue
        row = dict(block)
        row["specialization_match_score"] = float(round(score, 6))
        if match_meta:
            row["matched_specialization"] = match_meta
        key = _build_source_key(row, use_title=use_title_in_source_key)
        grouped.setdefault(key, []).append(row)

    picked: List[Dict[str, Any]] = []
    per_source = max(1, int(top_k_per_source or 1))
    for _, rows in grouped.items():
        rows.sort(
            key=lambda b: (
                float(b.get("specialization_match_score") or 0.0),
                -int(b.get("chunk_index") or 0),
            ),
            reverse=True,
        )
        picked.extend(rows[:per_source])

    picked.sort(
        key=lambda b: float(b.get("specialization_match_score") or 0.0),
        reverse=True,
    )
    safe_total = max(0, int(max_total or 0))
    if safe_total > 0:
        return picked[:safe_total]
    return picked


def retrieve_opportunity_supporting_chunks_by_specializations(
    opp: Opportunity,
    *,
    specialization_queries: List[Dict[str, Any]],
    top_k_per_additional_source: int = 2,
    top_k_per_attachment_source: int = 2,
    max_total_additional_chunks: int = 8,
    max_total_attachment_chunks: int = 8,
    min_score: float = 0.10,
) -> Dict[str, Any]:
    additional_rows = list(getattr(opp, "additional_info", None) or [])
    attachment_rows = list(getattr(opp, "attachments", None) or [])

    additional_blocks = load_extracted_content(
        additional_rows,
        url_attr="additional_info_url",
        group_chunks=False,
        include_row_meta=True,
    )
    attachment_blocks = load_extracted_content(
        attachment_rows,
        url_attr="file_download_path",
        title_attr="file_name",
        group_chunks=False,
        include_row_meta=True,
    )

    additional_ranked = _rank_blocks_by_specializations(
        blocks=additional_blocks,
        spec_queries=list(specialization_queries or []),
        top_k_per_source=top_k_per_additional_source,
        max_total=max_total_additional_chunks,
        min_score=min_score,
        use_title_in_source_key=False,
    )
    attachment_ranked = _rank_blocks_by_specializations(
        blocks=attachment_blocks,
        spec_queries=list(specialization_queries or []),
        top_k_per_source=top_k_per_attachment_source,
        max_total=max_total_attachment_chunks,
        min_score=min_score,
        use_title_in_source_key=True,
    )
    return {
        "specialization_queries": list(specialization_queries or []),
        "additional_info_chunks": additional_ranked,
        "attachment_chunks": attachment_ranked,
    }


def retrieve_faculty_additional_info_chunks_by_specializations(
    fac: Faculty,
    *,
    specialization_queries: List[Dict[str, Any]],
    top_k_per_source: int = 2,
    max_total_chunks: int = 8,
    min_score: float = 0.10,
) -> Dict[str, Any]:
    rows: List[FacultyAdditionalInfo] = list(getattr(fac, "additional_info", None) or [])
    blocks = load_extracted_content(
        rows,
        url_attr="additional_info_url",
        group_chunks=False,
        include_row_meta=True,
    )
    ranked = _rank_blocks_by_specializations(
        blocks=blocks,
        spec_queries=list(specialization_queries or []),
        top_k_per_source=top_k_per_source,
        max_total=max_total_chunks,
        min_score=min_score,
        use_title_in_source_key=False,
    )
    return {
        "specialization_queries": list(specialization_queries or []),
        "additional_info_chunks": ranked,
    }


def _to_vector(value: Any) -> Optional[np.ndarray]:
    """Convert pgvector/list-like payload to a 1D float32 numpy vector."""
    if value is None:
        return None

    raw = value
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            raw = json.loads(s)
        except Exception:
            return None

    try:
        vec = np.asarray(raw, dtype=np.float32).reshape(-1)
    except Exception:
        return None

    if vec.size == 0:
        return None
    return vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _is_chunk_row_usable(row: Any) -> bool:
    if getattr(row, "extract_status", None) not in ("done", "success"):
        return False
    if getattr(row, "extract_error", None):
        return False
    if not getattr(row, "content_path", None):
        return False
    if getattr(row, "content_embedding", None) is None:
        return False
    return True


def _rank_rows_by_query_vector(
    rows: List[Any],
    *,
    query_vector: np.ndarray,
    top_k: int,
) -> List[Tuple[Any, float]]:
    ranked: List[Tuple[Any, float]] = []
    for row in list(rows or []):
        if not _is_chunk_row_usable(row):
            continue
        vec = _to_vector(getattr(row, "content_embedding", None))
        if vec is None:
            continue
        if vec.shape[0] != query_vector.shape[0]:
            continue
        ranked.append((row, _cosine(query_vector, vec)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    safe_k = max(1, int(top_k or 1))
    return ranked[:safe_k]


def _source_groups(rows: List[Any], *, source_attrs: Tuple[str, ...]) -> Dict[str, List[Any]]:
    groups: Dict[str, List[Any]] = {}
    for row in list(rows or []):
        source_key = ""
        for attr in source_attrs:
            val = str(getattr(row, attr, "") or "").strip()
            if val:
                source_key = val
                break
        if not source_key:
            source_key = f"row:{int(getattr(row, 'id', 0) or 0)}"
        groups.setdefault(source_key, []).append(row)
    return groups


def _rank_rows_per_source(
    rows: List[Any],
    *,
    query_vector: np.ndarray,
    top_k_per_source: int,
    source_attrs: Tuple[str, ...],
) -> List[Tuple[Any, float]]:
    """
    Rank chunk rows by cosine similarity, but cap selections per source
    (e.g., per additional-info URL / per attachment file path).
    """
    grouped = _source_groups(rows, source_attrs=source_attrs)
    per_source_best: List[Tuple[float, str, List[Tuple[Any, float]]]] = []
    safe_per_source = max(1, int(top_k_per_source or 1))

    for source_key, source_rows in grouped.items():
        ranked = _rank_rows_by_query_vector(
            source_rows,
            query_vector=query_vector,
            top_k=safe_per_source,
        )
        if not ranked:
            continue
        best = float(ranked[0][1])
        per_source_best.append((best, source_key, ranked))

    # Keep higher-signal sources earlier while preserving per-source caps.
    per_source_best.sort(key=lambda x: x[0], reverse=True)
    out: List[Tuple[Any, float]] = []
    for _, _, ranked_rows in per_source_best:
        out.extend(ranked_rows)
    return out


def _fallback_rows_per_source(
    rows: List[Any],
    *,
    top_k_per_source: int,
    source_attrs: Tuple[str, ...],
) -> List[Any]:
    grouped = _source_groups(rows, source_attrs=source_attrs)
    safe_per_source = max(1, int(top_k_per_source or 1))
    out: List[Any] = []
    for source_key in sorted(grouped.keys()):
        source_rows = list(grouped[source_key] or [])
        source_rows.sort(key=lambda r: (int(getattr(r, "chunk_index", 0) or 0), int(getattr(r, "id", 0) or 0)))
        out.extend(source_rows[:safe_per_source])
    return out


def _load_ranked_chunk_blocks(
    rows: List[Any],
    *,
    url_attr: str,
    title_attr: Optional[str] = None,
    scores_by_row_id: Optional[Dict[int, float]] = None,
) -> List[Dict[str, Any]]:
    blocks = load_extracted_content(
        rows,
        url_attr=url_attr,
        title_attr=title_attr,
        group_chunks=False,
        include_row_meta=True,
    )
    for b in blocks:
        rid = int(b.get("row_id", 0) or 0)
        if scores_by_row_id and rid in scores_by_row_id:
            b["similarity"] = float(scores_by_row_id[rid])
        if rid > 0:
            b["id"] = rid
        b.pop("row_id", None)
    return blocks


def build_opportunity_rag_query(opp: Opportunity) -> str:
    """Grant-side RAG query: title + summary."""
    title = _normalize_text(getattr(opp, "opportunity_title", None))
    summary = _normalize_text(getattr(opp, "summary_description", None))
    parts = [p for p in [title, summary] if p]
    return "\n".join(parts).strip()


def build_faculty_rag_query(
    fac: Faculty,
    *,
    max_recent_pub_titles: int = 5,
) -> str:
    """Faculty-side RAG query: biography + top recent publication titles."""
    bio = _normalize_text(getattr(fac, "biography", None))
    pubs = sorted(
        list(getattr(fac, "publications", None) or []),
        key=lambda p: (int(getattr(p, "year", 0) or 0), int(getattr(p, "id", 0) or 0)),
        reverse=True,
    )
    pub_titles: List[str] = []
    for p in pubs:
        t = _normalize_text(getattr(p, "title", None))
        if not t:
            continue
        pub_titles.append(t)
        if len(pub_titles) >= max(0, int(max_recent_pub_titles or 0)):
            break

    parts: List[str] = []
    if bio:
        parts.append(bio)
    if pub_titles:
        parts.append("Recent publications: " + "; ".join(pub_titles))
    return "\n".join(parts).strip()


def retrieve_opportunity_supporting_chunks(
    opp: Opportunity,
    *,
    top_k_per_additional_source: int = 4,
    top_k_per_attachment_source: int = 4,
) -> Dict[str, Any]:
    """
    Retrieve top relevant chunk rows for an opportunity:
      - opportunity_additional_info
      - opportunity_attachment
    """
    query_text = build_opportunity_rag_query(opp)
    additional_rows = list(getattr(opp, "additional_info", None) or [])
    attachment_rows = list(getattr(opp, "attachments", None) or [])

    if not query_text:
        add_fallback = _fallback_rows_per_source(
            additional_rows,
            top_k_per_source=top_k_per_additional_source,
            source_attrs=("additional_info_url",),
        )
        att_fallback = _fallback_rows_per_source(
            attachment_rows,
            top_k_per_source=top_k_per_attachment_source,
            source_attrs=("file_download_path", "file_name"),
        )
        return {
            "query_text": query_text,
            "additional_info_chunks": _load_ranked_chunk_blocks(
                add_fallback,
                url_attr="additional_info_url",
            ),
            "attachment_chunks": _load_ranked_chunk_blocks(
                att_fallback,
                url_attr="file_download_path",
                title_attr="file_name",
            ),
        }

    try:
        q = embed_texts([query_text])
        if q.ndim != 2 or q.shape[0] <= 0:
            raise RuntimeError("query_embedding_empty")
        query_vector = q[0].astype(np.float32)
    except Exception:
        add_fallback = _fallback_rows_per_source(
            additional_rows,
            top_k_per_source=top_k_per_additional_source,
            source_attrs=("additional_info_url",),
        )
        att_fallback = _fallback_rows_per_source(
            attachment_rows,
            top_k_per_source=top_k_per_attachment_source,
            source_attrs=("file_download_path", "file_name"),
        )
        return {
            "query_text": query_text,
            "additional_info_chunks": _load_ranked_chunk_blocks(
                add_fallback,
                url_attr="additional_info_url",
            ),
            "attachment_chunks": _load_ranked_chunk_blocks(
                att_fallback,
                url_attr="file_download_path",
                title_attr="file_name",
            ),
        }

    ranked_add = _rank_rows_per_source(
        additional_rows,
        query_vector=query_vector,
        top_k_per_source=top_k_per_additional_source,
        source_attrs=("additional_info_url",),
    )
    ranked_att = _rank_rows_per_source(
        attachment_rows,
        query_vector=query_vector,
        top_k_per_source=top_k_per_attachment_source,
        source_attrs=("file_download_path", "file_name"),
    )

    add_rows = [r for r, _ in ranked_add]
    att_rows = [r for r, _ in ranked_att]
    if not add_rows:
        add_rows = _fallback_rows_per_source(
            additional_rows,
            top_k_per_source=top_k_per_additional_source,
            source_attrs=("additional_info_url",),
        )
    if not att_rows:
        att_rows = _fallback_rows_per_source(
            attachment_rows,
            top_k_per_source=top_k_per_attachment_source,
            source_attrs=("file_download_path", "file_name"),
        )

    add_scores = {int(getattr(r, "id", 0) or 0): float(s) for r, s in ranked_add}
    att_scores = {int(getattr(r, "id", 0) or 0): float(s) for r, s in ranked_att}

    return {
        "query_text": query_text,
        "additional_info_chunks": _load_ranked_chunk_blocks(
            add_rows,
            url_attr="additional_info_url",
            scores_by_row_id=add_scores,
        ),
        "attachment_chunks": _load_ranked_chunk_blocks(
            att_rows,
            url_attr="file_download_path",
            title_attr="file_name",
            scores_by_row_id=att_scores,
        ),
    }


def retrieve_faculty_additional_info_chunks(
    fac: Faculty,
    *,
    top_k_per_source: int = 5,
    max_recent_pub_titles: int = 5,
) -> Dict[str, Any]:
    """
    Retrieve top relevant faculty additional-info chunk rows with a per-source cap.
    Source is grouped by additional_info_url.
    """
    query_text = build_faculty_rag_query(
        fac,
        max_recent_pub_titles=max_recent_pub_titles,
    )
    rows: List[FacultyAdditionalInfo] = list(getattr(fac, "additional_info", None) or [])

    if not query_text:
        fallback_rows = _fallback_rows_per_source(
            rows,
            top_k_per_source=top_k_per_source,
            source_attrs=("additional_info_url",),
        )
        return {
            "query_text": query_text,
            "additional_info_chunks": _load_ranked_chunk_blocks(
                fallback_rows,
                url_attr="additional_info_url",
            ),
        }

    try:
        q = embed_texts([query_text])
        if q.ndim != 2 or q.shape[0] <= 0:
            raise RuntimeError("query_embedding_empty")
        query_vector = q[0].astype(np.float32)
    except Exception:
        fallback_rows = _fallback_rows_per_source(
            rows,
            top_k_per_source=top_k_per_source,
            source_attrs=("additional_info_url",),
        )
        return {
            "query_text": query_text,
            "additional_info_chunks": _load_ranked_chunk_blocks(
                fallback_rows,
                url_attr="additional_info_url",
            ),
        }

    ranked = _rank_rows_per_source(
        rows,
        query_vector=query_vector,
        top_k_per_source=top_k_per_source,
        source_attrs=("additional_info_url",),
    )
    selected_rows = [r for r, _ in ranked] or _fallback_rows_per_source(
        rows,
        top_k_per_source=top_k_per_source,
        source_attrs=("additional_info_url",),
    )
    scores = {int(getattr(r, "id", 0) or 0): float(s) for r, s in ranked}
    return {
        "query_text": query_text,
        "additional_info_chunks": _load_ranked_chunk_blocks(
            selected_rows,
            url_attr="additional_info_url",
            scores_by_row_id=scores,
        ),
    }
