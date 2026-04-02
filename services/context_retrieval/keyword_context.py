from __future__ import annotations

import json
from typing import Any, Dict, List

from utils.keyword_utils import (
    attach_specialization_sources_from_llm,
    build_specialization_source_catalog,
    map_specialization_sources_by_cosine,
)


class KeywordContextBuilder:
    """Formatting helpers for keyword merge-chain inputs/outputs."""

    DEFAULT_MAX_BATCH_KEYWORDS = 20
    DEFAULT_MAX_MERGED_DOMAIN = 10
    DEFAULT_MAX_MERGED_SPECIALIZATION = 15

    @staticmethod
    def _normalize_text_key(value: Any) -> str:
        return " ".join(str(value or "").strip().lower().split())

    @classmethod
    def dedupe_texts(cls, values: List[Any]) -> List[str]:
        out: List[str] = []
        seen = set()
        for raw in list(values or []):
            text = " ".join(str(raw or "").split()).strip()
            if not text:
                continue
            key = cls._normalize_text_key(text)
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out

    @staticmethod
    def payload_len(payload: Dict[str, Any]) -> int:
        return len(json.dumps(payload, ensure_ascii=False))

    @classmethod
    def collect_keyword_contents(cls, *, context: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        ctx = dict(context or {})

        if "faculty_id" in ctx or "publications" in ctx:
            header = {
                "faculty_name": ctx.get("name"),
                "biography": ctx.get("biography"),
                "contents": [],
            }
            contents: List[str] = []
            for row in list(ctx.get("additional_info_extracted") or []):
                if not isinstance(row, dict):
                    continue
                content = str(row.get("content") or "").strip()
                if content:
                    contents.append(content)
            for pub in list(ctx.get("publications") or []):
                if not isinstance(pub, dict):
                    continue
                title = str(pub.get("title") or "").strip()
                abstract = str(pub.get("abstract") or "").strip()
                body = f"{title}. {abstract}".strip(". ").strip()
                if body:
                    contents.append(body)
            if not contents:
                merged = str(ctx.get("merged_content") or "").strip()
                if merged:
                    contents.append(merged)
            return header, cls.dedupe_texts(contents)

        if "opportunity_id" in ctx or "attachments_extracted" in ctx:
            header = {
                "opportunity_title": ctx.get("opportunity_title"),
                "summary_description": ctx.get("summary_description"),
                "contents": [],
            }
            contents = []
            for row in list(ctx.get("additional_info_extracted") or []):
                if not isinstance(row, dict):
                    continue
                content = str(row.get("content") or "").strip()
                if content:
                    contents.append(content)
            for row in list(ctx.get("attachments_extracted") or []):
                if not isinstance(row, dict):
                    continue
                content = str(row.get("content") or "").strip()
                if content:
                    contents.append(content)
            if not contents:
                merged = str(ctx.get("summary_description") or "").strip()
                if merged:
                    contents.append(merged)
            return header, cls.dedupe_texts(contents)

        return {"context": ctx, "contents": []}, []

    @classmethod
    def build_context_batches(
        cls,
        *,
        context: Dict[str, Any],
        max_chars: int,
    ) -> List[Dict[str, Any]]:
        header, contents = cls.collect_keyword_contents(context=context)
        if not contents:
            return [header]

        safe_max = max(2_000, int(max_chars or 0))
        base = dict(header or {})
        base["contents"] = []
        base_len = cls.payload_len(base)

        batches: List[Dict[str, Any]] = []
        current: List[str] = []
        current_len = int(base_len)
        for content in list(contents):
            item = str(content or "").strip()
            if not item:
                continue
            item_len = len(json.dumps(item, ensure_ascii=False)) + 1
            if current and (current_len + item_len > safe_max):
                flush = dict(base)
                flush["contents"] = list(current)
                batches.append(flush)
                current = []
                current_len = int(base_len)
            if base_len + item_len > safe_max:
                # keep oversized content as a standalone batch (no truncation)
                flush = dict(base)
                flush["contents"] = [item]
                batches.append(flush)
                continue
            current.append(item)
            current_len += item_len
        if current:
            flush = dict(base)
            flush["contents"] = list(current)
            batches.append(flush)
        return batches or [header]

    @classmethod
    def build_weight_context_from_batches(
        cls,
        *,
        batches: List[Dict[str, Any]],
        max_chars: int,
    ) -> Dict[str, Any]:
        if not batches:
            return {}
        seed = dict(batches[0] or {})
        header = {k: v for k, v in seed.items() if k != "contents"}
        header["contents"] = []
        safe_max = max(2_000, int(max_chars or 0))
        cur_len = cls.payload_len(header)
        idx = [0] * len(batches)
        progressed = True
        while progressed:
            progressed = False
            for i, batch in enumerate(list(batches)):
                contents = list((batch or {}).get("contents") or [])
                if idx[i] >= len(contents):
                    continue
                item = str(contents[idx[i]] or "").strip()
                idx[i] += 1
                if not item:
                    continue
                item_len = len(json.dumps(item, ensure_ascii=False)) + 1
                if cur_len + item_len > safe_max:
                    continue
                header["contents"].append(item)
                cur_len += item_len
                progressed = True
        return header

    @classmethod
    def format_merge_input_row(
        cls,
        *,
        batch_idx: int,
        candidates: List[Any],
        keyword_bucket: Dict[str, Any],
        max_batch_keywords: int = DEFAULT_MAX_BATCH_KEYWORDS,
    ) -> Dict[str, Any]:
        bucket = dict(keyword_bucket or {})
        cap = max(1, int(max_batch_keywords or cls.DEFAULT_MAX_BATCH_KEYWORDS))
        return {
            "batch_idx": int(batch_idx),
            "candidates": cls.dedupe_texts(list(candidates or [])),
            "domain": cls.dedupe_texts(list(bucket.get("domain") or []))[:cap],
            "specialization": cls.dedupe_texts(list(bucket.get("specialization") or []))[:cap],
        }

    @classmethod
    def normalize_merge_output(
        cls,
        merged: Any,
        *,
        max_domain: int = DEFAULT_MAX_MERGED_DOMAIN,
        max_specialization: int = DEFAULT_MAX_MERGED_SPECIALIZATION,
    ) -> Dict[str, List[str]]:
        payload = merged.model_dump() if hasattr(merged, "model_dump") else dict(merged or {})
        domain_cap = max(1, int(max_domain or cls.DEFAULT_MAX_MERGED_DOMAIN))
        spec_cap = max(1, int(max_specialization or cls.DEFAULT_MAX_MERGED_SPECIALIZATION))
        return {
            "domain": cls.dedupe_texts(list((payload or {}).get("domain") or []))[:domain_cap],
            "specialization": cls.dedupe_texts(list((payload or {}).get("specialization") or []))[:spec_cap],
        }

    @classmethod
    def fallback_merge_from_rows(
        cls,
        rows: List[Dict[str, Any]],
        *,
        max_domain: int = DEFAULT_MAX_MERGED_DOMAIN,
        max_specialization: int = DEFAULT_MAX_MERGED_SPECIALIZATION,
    ) -> Dict[str, List[str]]:
        domains: List[Any] = []
        specializations: List[Any] = []
        for row in list(rows or []):
            if not isinstance(row, dict):
                continue
            domains.extend(list(row.get("domain") or []))
            specializations.extend(list(row.get("specialization") or []))
        domain_cap = max(1, int(max_domain or cls.DEFAULT_MAX_MERGED_DOMAIN))
        spec_cap = max(1, int(max_specialization or cls.DEFAULT_MAX_MERGED_SPECIALIZATION))
        return {
            "domain": cls.dedupe_texts(domains)[:domain_cap],
            "specialization": cls.dedupe_texts(specializations)[:spec_cap],
        }

    @classmethod
    def fallback_merge_bucket(
        cls,
        *,
        batch_domains: List[List[str]],
        batch_specializations: List[List[str]],
        max_domain: int = DEFAULT_MAX_MERGED_DOMAIN,
        max_specialization: int = DEFAULT_MAX_MERGED_SPECIALIZATION,
    ) -> Dict[str, List[str]]:
        rows = [
            {
                "domain": list(d or []),
                "specialization": list(s or []),
            }
            for d, s in zip(list(batch_domains or []), list(batch_specializations or []))
        ]
        return cls.fallback_merge_from_rows(
            rows,
            max_domain=max_domain,
            max_specialization=max_specialization,
        )

    @staticmethod
    def build_source_catalog(context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build source catalog from normalized context for specialization source mapping."""
        return list(build_specialization_source_catalog(context) or [])

    @classmethod
    def attach_sources_by_cosine(
        cls,
        *,
        keywords: Dict[str, Any],
        context: Dict[str, Any],
        embedding_client: Any,
        max_sources_per_specialization: int = 4,
        min_similarity: float = 0.10,
    ) -> Dict[str, Any]:
        """Attach specialization sources using cosine similarity over context-derived catalog."""
        source_catalog = cls.build_source_catalog(context)
        source_map_raw: Dict[str, Any] = {}
        source_error: str | None = None
        kw_with_sources = dict(keywords or {})

        if source_catalog:
            try:
                source_map_raw = map_specialization_sources_by_cosine(
                    keywords=keywords or {},
                    source_catalog=source_catalog,
                    embedding_client=embedding_client,
                    max_sources_per_specialization=int(max_sources_per_specialization),
                    min_similarity=float(min_similarity),
                )
            except Exception as e:
                source_error = f"{type(e).__name__}: {e}"

            kw_with_sources = attach_specialization_sources_from_llm(
                keywords=keywords or {},
                llm_sources=source_map_raw,
                source_catalog=source_catalog,
            )

        return {
            "keywords": kw_with_sources,
            "source_catalog": source_catalog,
            "source_map_raw": source_map_raw,
            "source_error": source_error,
        }
