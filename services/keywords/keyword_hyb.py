from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import get_llm_client
from dto.llm_response_dto import CandidatesOut, KeywordsOut, WeightedSpecsOut
from services.prompts.keyword_prompts import (
    OPP_CANDIDATE_PROMPT,
    OPP_KEYWORDS_PROMPT,
    OPP_SPECIALIZATION_WEIGHT_PROMPT,
)
from utils.keyword_utils import apply_weighted_specializations, coerce_keyword_sections
from utils.payload_sanitizer import sanitize_for_postgres


class GrantKeywordHyb:
    """
    Hybrid single-grant keyword generator.

    Uses v1 prompt flow:
      1) candidate extraction
      2) keyword generation
      3) specialization weighting

    Input context is built from Neo4j Grant node + GrantTextChunk rows only.
    No snippet/source linking is generated.
    """

    def __init__(
        self,
        *,
        max_context_chars: int = 90_000,
        max_neo4j_chunks: int = 4_000,
    ):
        self.max_context_chars = max(5_000, int(max_context_chars or 90_000))
        self.max_neo4j_chunks = max(10, int(max_neo4j_chunks or 4_000))

    @staticmethod
    def build_keyword_chain(
        candidate_prompt: ChatPromptTemplate = OPP_CANDIDATE_PROMPT,
        keywords_prompt: ChatPromptTemplate = OPP_KEYWORDS_PROMPT,
        weight_prompt: ChatPromptTemplate = OPP_SPECIALIZATION_WEIGHT_PROMPT,
    ):
        llm = get_llm_client().build()
        candidates_chain = candidate_prompt | llm.with_structured_output(CandidatesOut)
        keywords_chain = keywords_prompt | llm.with_structured_output(KeywordsOut)
        weight_chain = weight_prompt | llm.with_structured_output(WeightedSpecsOut)
        return candidates_chain, keywords_chain, weight_chain

    def _fetch_neo4j_rows(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            from neo4j import GraphDatabase, RoutingControl
            from graph_rag.common import load_dotenv_if_present, read_neo4j_settings
        except Exception:
            return []

        try:
            load_dotenv_if_present()
            neo4j_settings = read_neo4j_settings()
        except Exception:
            return []

        try:
            with GraphDatabase.driver(
                neo4j_settings.uri,
                auth=(neo4j_settings.username, neo4j_settings.password),
            ) as driver:
                records, _, _ = driver.execute_query(
                    query,
                    parameters_=params,
                    routing_=RoutingControl.READ,
                    database_=neo4j_settings.database,
                )
        except Exception:
            return []

        return [dict(row or {}) for row in records]

    def _fetch_grant_node(self, *, opportunity_id: str) -> Dict[str, Any]:
        rows = self._fetch_neo4j_rows(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            RETURN
                g.opportunity_id AS opportunity_id,
                g.opportunity_title AS opportunity_title,
                g.agency_name AS agency_name,
                g.category AS category,
                g.opportunity_status AS opportunity_status,
                g.summary_description AS summary_description
            LIMIT 1
            """,
            {"opportunity_id": str(opportunity_id)},
        )
        return rows[0] if rows else {}

    def _fetch_grant_chunks(self, *, opportunity_id: str) -> List[Dict[str, Any]]:
        rows = self._fetch_neo4j_rows(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(c:GrantTextChunk)
            WHERE c.chunk_id IS NOT NULL
              AND c.text IS NOT NULL
              AND trim(toString(c.text)) <> ''
            RETURN
                c.chunk_id AS chunk_id,
                c.source_type AS source_type,
                c.source_ref_id AS source_ref_id,
                c.source_url AS source_url,
                c.source_title AS source_title,
                c.chunk_index AS chunk_index,
                c.text AS text
            ORDER BY
                c.source_type ASC,
                c.source_ref_id ASC,
                c.chunk_index ASC,
                c.chunk_id ASC
            """,
            {"opportunity_id": str(opportunity_id)},
        )
        out: List[Dict[str, Any]] = []
        for row in rows[: self.max_neo4j_chunks]:
            text = str(row.get("text") or "").strip()
            chunk_id = str(row.get("chunk_id") or "").strip()
            if not text or not chunk_id:
                continue
            out.append(
                {
                    "chunk_id": chunk_id,
                    "source_type": str(row.get("source_type") or "").strip().lower(),
                    "source_ref_id": str(row.get("source_ref_id") or "").strip(),
                    "source_url": str(row.get("source_url") or "").strip() or None,
                    "source_title": str(row.get("source_title") or "").strip() or None,
                    "chunk_index": row.get("chunk_index"),
                    "text": text,
                }
            )
        return out

    def build_context_from_neo4j(self, *, opportunity_id: str) -> Dict[str, Any]:
        grant = self._fetch_grant_node(opportunity_id=str(opportunity_id))
        chunks = self._fetch_grant_chunks(opportunity_id=str(opportunity_id))

        has_summary_chunk = any((c.get("source_type") or "") == "summary" for c in chunks)

        additional_info_extracted: List[Dict[str, Any]] = []
        attachments_extracted: List[Dict[str, Any]] = []
        summary_chunks: List[Dict[str, Any]] = []

        used_chars = 0
        for c in chunks:
            text = str(c.get("text") or "").strip()
            if not text:
                continue
            if used_chars + len(text) > self.max_context_chars:
                break
            used_chars += len(text)

            source_type = str(c.get("source_type") or "").strip().lower()
            if source_type == "attachment":
                attachments_extracted.append(
                    {
                        "title": c.get("source_title"),
                        "url": c.get("source_url"),
                        "content": text,
                    }
                )
            elif source_type == "additional_info":
                additional_info_extracted.append(
                    {
                        "url": c.get("source_url"),
                        "content": text,
                    }
                )
            elif source_type == "summary":
                summary_chunks.append(
                    {
                        "chunk_id": c.get("chunk_id"),
                        "content": text,
                    }
                )
            else:
                additional_info_extracted.append(
                    {
                        "url": c.get("source_url"),
                        "content": text,
                    }
                )

        context = {
            "opportunity_id": str(grant.get("opportunity_id") or opportunity_id),
            "opportunity_title": grant.get("opportunity_title"),
            "agency_name": grant.get("agency_name"),
            "category": grant.get("category"),
            "opportunity_status": grant.get("opportunity_status"),
            "summary_description": None if has_summary_chunk else grant.get("summary_description"),
            "summary_chunks": summary_chunks,
            "additional_info_extracted": additional_info_extracted,
            "attachments_extracted": attachments_extracted,
        }
        return sanitize_for_postgres(context)

    def generate_for_single_grant(self, *, opportunity_id: str) -> Dict[str, Any]:
        opp_id = str(opportunity_id or "").strip()
        if not opp_id:
            return {}

        candidates_chain, keywords_chain, weight_chain = self.build_keyword_chain()

        context = self.build_context_from_neo4j(opportunity_id=opp_id)
        context_json = json.dumps(context, ensure_ascii=False)
        print(context_json)
        cand_out: CandidatesOut = candidates_chain.invoke({"context_json": context_json})
        candidates = (cand_out.candidates or [])[:50]

        kw_out: KeywordsOut = keywords_chain.invoke(
            {
                "context_json": context_json,
                "candidates": "\n".join(f"- {c}" for c in candidates),
            }
        )
        kw_dict = coerce_keyword_sections(kw_out.model_dump())

        spec_in = {
            "research": (kw_dict.get("research") or {}).get("specialization") or [],
            "application": (kw_dict.get("application") or {}).get("specialization") or [],
        }
        weighted_out: WeightedSpecsOut = weight_chain.invoke(
            {
                "context_json": context_json,
                "spec_json": json.dumps(spec_in, ensure_ascii=False),
            }
        )

        kw_weighted = apply_weighted_specializations(keywords=kw_dict, weighted=weighted_out)
        return sanitize_for_postgres(kw_weighted)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate one grant keyword payload using v1 prompts with Neo4j context only."
    )
    parser.add_argument("--opportunity-id", type=str, default="b09d00a3-4b76-4540-8fea-9de1543e83c1", help="Target grant opportunity_id")
    parser.add_argument("--max-context-chars", type=int, default=90_000)
    parser.add_argument("--max-neo4j-chunks", type=int, default=4_000)
    args = parser.parse_args()

    service = GrantKeywordHyb(
        max_context_chars=args.max_context_chars,
        max_neo4j_chunks=args.max_neo4j_chunks,
    )
    result = service.generate_for_single_grant(opportunity_id=args.opportunity_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

