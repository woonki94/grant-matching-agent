from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.prompts import ChatPromptTemplate

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import get_llm_client, settings
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from db.models.opportunity import Opportunity
from dto.llm_response_dto import KeywordMergeOut, KeywordMentionsOut, WeightedSpecsIdxOut
from logging_setup import setup_logging
from services.prompts.keyword_prompts import FACULTY_KEYWORD_MERGE_PROMPT
from utils.embedder import embed_domain_bucket
from utils.keyword_utils import extract_domains_from_keywords
from utils.thread_pool import parallel_map

setup_logging()
logger = logging.getLogger(__name__)


GRANT_CHUNK_RESEARCH_KEYWORD_LINK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You extract RESEARCH keywords from grant chunk rows and attach specialization evidence ids.\n\n"
        "Input:\n"
        "- JSON list of chunk rows: [{{chunk_id, text}}, ...]\n\n"
        "Output JSON schema:\n"
        "{{\n"
        "  \"research\": {{\n"
        "    \"domain\": [\"...\"],\n"
        "    \"specialization\": [{{\"t\": \"...\", \"e\": {{\"chunk_id\": 0.0}}}}]\n"
        "  }},\n"
        "  \"application\": {{\n"
        "    \"domain\": [],\n"
        "    \"specialization\": []\n"
        "  }}\n"
        "}}\n\n"
        "Rules:\n"
        "- Use ONLY provided chunk text.\n"
        "- domain: 2-4 words.\n"
        "- specialization: 8-25 words, capabilities/requirements the grant expects.\n"
        "- Focus ONLY on research intent: methods/theory/algorithms/modeling/scientific inquiry.\n"
        "- research.domain: 5–10 items\n"
        "- research.specialization: 10–15 items\n"
        "- application must remain empty.\n"
        "- Lowercase unless proper nouns.\n"
        "- Do not invent chunk ids.\n"
        "- For specialization, e must be a non-empty object map chunk_id -> confidence in [0,1].\n"
        "- Domains do not require evidence ids.\n"
        "- Do not output empty keywords.\n"
        "- Keep output compact and deduplicated.\n"
        "- Confidence must be discriminative; do not assign one default value to all chunk ids."
    ),
    ("human", "Chunks JSON:\n{chunks_json}"),
])

GRANT_CHUNK_APPLICATION_KEYWORD_LINK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You extract APPLICATION keywords from grant chunk rows and attach specialization evidence ids.\n\n"
        "Input:\n"
        "- JSON list of chunk rows: [{{chunk_id, text}}, ...]\n\n"
        "Output JSON schema:\n"
        "{{\n"
        "  \"application\": {{\n"
        "    \"domain\": [\"...\"],\n"
        "    \"specialization\": [{{\"t\": \"...\", \"e\": {{\"chunk_id\": 0.0}}}}]\n"
        "  }},\n"
        "  \"research\": {{\n"
        "    \"domain\": [],\n"
        "    \"specialization\": []\n"
        "  }}\n"
        "}}\n\n"
        "Rules:\n"
        "- Use ONLY provided chunk text.\n"
        "- domain: 2-4 words.\n"
        "- specialization: 8-25 words, capabilities/requirements the grant expects.\n"
        "- Focus ONLY on application intent: real-world use, deployment context, operational setting, industry/sector problems, implementation constraints, and practical impact.\n"
        "- application.domain: 5–10 items\n"
        "- application.specialization: 10–15 items\n"
        "- If chunks mention any target population, sector, use-case, deployment setting, operational constraint, or practical outcome, you MUST output application keywords.\n"
        "- Treat these as strong application signals: healthcare, education, agriculture, climate, infrastructure, manufacturing, public safety, energy, policy, workforce, community impact.\n"
        "- If there is technical/problem content but weak sector specificity, still output at least 1 application.domain and 1 application.specialization using the most concrete practical framing supported by text.\n"
        "- Return empty application only when the chunkset is purely administrative/compliance text with no technical/problem/use-case content.\n"
        "- research must remain empty.\n"
        "- Lowercase unless proper nouns.\n"
        "- Do not invent chunk ids.\n"
        "- For specialization, e must be a non-empty object map chunk_id -> confidence in [0,1].\n"
        "- Domains do not require evidence ids.\n"
        "- Do not output empty keywords.\n"
        "- Keep output compact and deduplicated.\n"
        "- Confidence must be discriminative; do not assign one default value to all chunk ids."
    ),
    ("human", "Chunks JSON:\n{chunks_json}"),
])

GRANT_SPECIALIZATION_WEIGHT_FLAT_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You assign importance weights to grant specialization keywords.\n\n"
        "You will receive one JSON object with:\n"
        "- grant_context: grant profile + flat chunk strings\n"
        "- specializations: research/application arrays of items with idx and t\n\n"
        "Input item shape:\n"
        "- {{\"idx\": 0, \"t\": \"specialization text\"}}\n\n"
        "Task:\n"
        "- Return weights for each input item as:\n"
        "{{\n"
        "  \"research\": [{{\"idx\": 0, \"w\": 0.0}}],\n"
        "  \"application\": [{{\"idx\": 0, \"w\": 0.0}}]\n"
        "}}\n\n"
        "Rules:\n"
        "- Use ONLY provided input JSON.\n"
        "- Keep idx unchanged and return each input idx exactly once in its same section.\n"
        "- Do NOT invent or remove idx.\n"
        "- w must be in [0,1].\n"
        "- 0.85-1.00: core requirement; 0.60-0.84: important; 0.35-0.59: supporting; 0.10-0.34: minor.\n"
        "- If uncertain, assign a lower weight instead of omitting.\n"
        "- Return ONLY JSON."
    ),
    ("human", "WEIGHT INPUT JSON:\n{weight_input_json}\n\nReturn weighted idx JSON."),
])


class GrantKeywordGenerator:
    """
    Grant keyword generator mirroring faculty v2 structure.

    Pipeline:
    1) Read grant chunks from Neo4j only.
    2) Pack chunks up to a context-size budget (no per-chunk truncation).
    3) Ask LLM in separate calls to extract research/application keyword links.
    4) Convert linked mentions -> keyword buckets.
    5) Optionally ask LLM to merge near-duplicate keywords.
    6) Ask LLM to weight specialization keywords.
    7) Persist keywords/raw_json + domain embeddings in Postgres.
    """

    def __init__(
        self,
        *,
        force_regenerate: bool = False,
        max_context_chars: int = 180_000,
        max_neo4j_chunks: int = 4_000,
        reserve_prompt_chars: int = 3_000,
    ):
        self.force_regenerate = bool(force_regenerate)
        self.max_context_chars = max(1_000, int(max_context_chars or 180_000))
        self.max_neo4j_chunks = max(10, int(max_neo4j_chunks or 4_000))
        self.reserve_prompt_chars = max(0, int(reserve_prompt_chars or 0))

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

    @staticmethod
    def build_chunk_research_keyword_chain(prompt=GRANT_CHUNK_RESEARCH_KEYWORD_LINK_PROMPT):
        llm = get_llm_client().build()
        return prompt | llm.with_structured_output(KeywordMentionsOut)

    @staticmethod
    def build_chunk_application_keyword_chain(prompt=GRANT_CHUNK_APPLICATION_KEYWORD_LINK_PROMPT):
        llm = get_llm_client().build()
        return prompt | llm.with_structured_output(KeywordMentionsOut)

    @staticmethod
    def build_keyword_merge_chain(prompt=FACULTY_KEYWORD_MERGE_PROMPT):
        llm = get_llm_client(settings.opus or settings.haiku).build()
        return prompt | llm.with_structured_output(KeywordMergeOut)

    @staticmethod
    def build_specialization_weight_flat_context_chain(prompt=GRANT_SPECIALIZATION_WEIGHT_FLAT_CONTEXT_PROMPT):
        llm = get_llm_client().build()
        return prompt | llm.with_structured_output(WeightedSpecsIdxOut)

    def get_grant_chunks_by_opportunity_id(self, *, opportunity_id: str) -> List[Dict[str, Any]]:
        """
        Return all GrantTextChunk rows linked to a Grant node by opportunity_id.
        """
        opp_id = str(opportunity_id or "").strip()
        if not opp_id:
            return []

        rows = self._fetch_neo4j_rows(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})-[r]->(c:GrantTextChunk)
            WHERE c.chunk_id IS NOT NULL
              AND c.text IS NOT NULL
            RETURN
                c.chunk_id AS chunk_id,
                elementId(c) AS chunk_node_id,
                type(r) AS relation,
                c.source_type AS source_type,
                c.source_ref_id AS source_ref_id,
                c.source_url AS source_url,
                c.chunk_index AS chunk_index,
                c.char_count AS char_count,
                c.text AS text,
                c.embedding AS chunk_embedding
            ORDER BY
                c.source_type ASC,
                c.source_ref_id ASC,
                c.chunk_index ASC,
                c.chunk_id ASC
            """,
            {"opportunity_id": opp_id},
        )

        out: List[Dict[str, Any]] = []
        seen_chunk_ids = set()
        for row in rows:
            chunk_id = str(row.get("chunk_id") or "").strip()
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            out.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_node_id": str(row.get("chunk_node_id") or "").strip() or None,
                    "relation": str(row.get("relation") or "").strip() or None,
                    "source_type": str(row.get("source_type") or "").strip() or None,
                    "source_ref_id": str(row.get("source_ref_id") or "").strip() or None,
                    "source_url": str(row.get("source_url") or "").strip() or None,
                    "chunk_index": row.get("chunk_index"),
                    "char_count": row.get("char_count"),
                    "text": str(row.get("text") or ""),
                    "chunk_embedding": row.get("chunk_embedding"),
                }
            )
        return out

    def build_chunk_payload_windows(
        self,
        *,
        chunks: Sequence[Dict[str, Any]],
        max_context_chars: Optional[int] = None,
    ) -> List[List[Dict[str, str]]]:
        """
        Split chunks into chunksets of {"chunk_id", "text"} under a char budget.
        """
        base_budget = int(max_context_chars) if max_context_chars is not None else int(self.max_context_chars)
        budget = max(100, int(base_budget) - int(self.reserve_prompt_chars))

        windows: List[List[Dict[str, str]]] = []
        current: List[Dict[str, str]] = []
        current_chars = 0

        for row in chunks or []:
            chunk_id = str((row or {}).get("chunk_id") or "").strip()
            text = str((row or {}).get("text") or "")
            if not chunk_id or not text.strip():
                continue

            item = {"chunk_id": chunk_id, "text": text}
            item_chars = len(json.dumps(item, ensure_ascii=False))
            join_overhead = 1 if current else 0
            next_chars = current_chars + item_chars + join_overhead

            if current and next_chars > budget:
                windows.append(current)
                current = [item]
                current_chars = item_chars
                continue

            if not current and item_chars > budget:
                windows.append([item])
                continue

            current.append(item)
            current_chars = next_chars

        if current:
            windows.append(current)

        return windows

    def build_chunk_payload(
        self,
        *,
        chunks: Sequence[Dict[str, Any]],
        max_context_chars: Optional[int] = None,
    ) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        """
        Return:
        {
          "grant_chunk": {
            "chunkset1": [...],
            "chunkset2": [...]
          }
        }
        """
        windows = self.build_chunk_payload_windows(
            chunks=chunks,
            max_context_chars=max_context_chars,
        )
        return {
            "grant_chunk": {
                f"chunkset{i}": window
                for i, window in enumerate(windows, start=1)
            }
        }

    def build_grant_chunk_payload(
        self,
        *,
        opportunity_id: str,
        max_context_chars: Optional[int] = None,
    ) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        chunks = self.get_grant_chunks_by_opportunity_id(opportunity_id=str(opportunity_id))
        return self.build_chunk_payload(
            chunks=chunks,
            max_context_chars=max_context_chars,
        )

    def extract_keyword_mentions_from_payload(
        self,
        *,
        grant_chunk_payload: Dict[str, Any],
        chunk_research_keyword_chain=None,
        chunk_application_keyword_chain=None,
    ) -> List[Dict[str, Any]]:
        """
        Run prompt per chunkset and return flat keyword mentions.
        """
        grant_chunk = (grant_chunk_payload or {}).get("grant_chunk") or {}
        if not isinstance(grant_chunk, dict):
            return []

        chunkset_items: List[tuple[str, List[Dict[str, Any]]]] = []
        for chunkset_key in sorted(grant_chunk.keys()):
            rows = grant_chunk.get(chunkset_key) or []
            if not isinstance(rows, list) or not rows:
                continue
            chunkset_items.append((chunkset_key, rows))
        if not chunkset_items:
            return []

        workers = len(chunkset_items)

        def _run_one(item: tuple[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
            chunkset_key, chunk_rows = item
            llm_input = {"chunks_json": json.dumps(chunk_rows, ensure_ascii=False)}
            #print(llm_input)
            research_chain = chunk_research_keyword_chain or self.build_chunk_research_keyword_chain()
            application_chain = chunk_application_keyword_chain or self.build_chunk_application_keyword_chain()
            research_out: KeywordMentionsOut = research_chain.invoke(llm_input)
            application_out: KeywordMentionsOut = application_chain.invoke(llm_input)
            #print(research_out)
            #print(application_out)
            local_out: List[Dict[str, Any]] = []
            valid_chunk_ids = {
                str((row or {}).get("chunk_id") or "").strip()
                for row in chunk_rows
                if str((row or {}).get("chunk_id") or "").strip()
            }

            def _append_section_rows(section: str, sec_obj: Any) -> None:
                if sec_obj is None:
                    return
                for raw_domain in list(getattr(sec_obj, "domain", []) or []):
                    keyword = str(raw_domain or "").strip()
                    if not keyword:
                        continue
                    local_out.append(
                        {
                            "section": section,
                            "bucket": "domain",
                            "keyword": keyword,
                            "chunkset": chunkset_key,
                        }
                    )

                for spec in list(getattr(sec_obj, "specialization", []) or []):
                    keyword = str(getattr(spec, "t", "") or "").strip()
                    e_raw = getattr(spec, "e", {}) or {}
                    if not keyword:
                        continue
                    snippet_ids: Dict[str, float] = {}
                    if isinstance(e_raw, dict):
                        for sid_raw, conf_raw in e_raw.items():
                            sid = str(sid_raw or "").strip()
                            if not sid or sid not in valid_chunk_ids:
                                continue
                            try:
                                conf = float(conf_raw)
                            except Exception:
                                conf = 0.8
                            snippet_ids[sid] = max(0.0, min(1.0, conf))
                    else:
                        for sid_raw in list(e_raw or []):
                            sid = str(sid_raw or "").strip()
                            if not sid or sid not in valid_chunk_ids:
                                continue
                            snippet_ids[sid] = 0.8
                    if not snippet_ids:
                        continue
                    local_out.append(
                        {
                            "section": section,
                            "bucket": "specialization",
                            "keyword": keyword,
                            "snippet_ids": snippet_ids,
                            "chunkset": chunkset_key,
                        }
                    )

            _append_section_rows("research", getattr(research_out, "research", None))
            _append_section_rows("application", getattr(application_out, "application", None))
            return local_out

        def _on_error(_idx: int, item: tuple[str, List[Dict[str, Any]]], _e: Exception) -> List[Dict[str, Any]]:
            logger.exception("[grant_keyword_v2] chunkset=%s failed", item[0])
            return []

        batches = parallel_map(
            chunkset_items,
            max_workers=workers,
            run_item=_run_one,
            on_error=_on_error,
        )
        out: List[Dict[str, Any]] = []
        for rows in batches:
            out.extend(rows)
        return out

    def extract_keyword_mentions_for_grant(
        self,
        *,
        opportunity_id: str,
        max_context_chars: Optional[int] = None,
        chunk_research_keyword_chain=None,
        chunk_application_keyword_chain=None,
    ) -> List[Dict[str, Any]]:
        payload = self.build_grant_chunk_payload(
            opportunity_id=str(opportunity_id),
            max_context_chars=max_context_chars,
        )
        #print(payload)
        return self.extract_keyword_mentions_from_payload(
            grant_chunk_payload=payload,
            chunk_research_keyword_chain=chunk_research_keyword_chain,
            chunk_application_keyword_chain=chunk_application_keyword_chain,
        )

    @staticmethod
    def _norm_keyword(value: Any) -> str:
        return " ".join(str(value or "").split()).strip().lower()

    def merge_keyword_mentions_with_llm(
        self,
        *,
        mentions: List[Dict[str, Any]],
        merge_chain=None,
    ) -> List[Dict[str, Any]]:
        """
        Domain: deterministic exact dedup.
        Specialization: one LLM merge call with compact grouped schema.
        """
        if not mentions:
            return []

        grouped: Dict[str, Dict[str, Any]] = {
            "research": {"domain": [], "specialization": []},
            "application": {"domain": [], "specialization": []},
        }
        section_sid_conf: Dict[str, Dict[str, float]] = {"research": {}, "application": {}}
        spec_acc: Dict[tuple[str, str], set[str]] = {}
        domains_by_section: Dict[str, List[str]] = {"research": [], "application": []}

        domain_seen = set()
        for m in mentions:
            section = str(m.get("section") or "").strip().lower()
            bucket = str(m.get("bucket") or "").strip().lower()
            keyword = self._norm_keyword(m.get("keyword"))
            if section not in {"research", "application"}:
                continue
            if bucket not in {"domain", "specialization"}:
                continue
            if not keyword:
                continue

            if bucket == "domain":
                d_key = (section, keyword)
                if d_key in domain_seen:
                    continue
                domain_seen.add(d_key)
                domains_by_section[section].append(keyword)
                continue

            snippet_ids = dict(m.get("snippet_ids") or {})
            s_key = (section, keyword)
            spec_acc.setdefault(s_key, set())
            for sid_raw, conf_raw in snippet_ids.items():
                sid = str(sid_raw or "").strip()
                if not sid:
                    continue
                spec_acc[s_key].add(sid)
                try:
                    conf = float(conf_raw)
                except Exception:
                    conf = 0.8
                conf = max(0.0, min(1.0, conf))
                prev = float(section_sid_conf[section].get(sid, 0.0))
                section_sid_conf[section][sid] = max(prev, conf)

        for (section, keyword), e_set in spec_acc.items():
            grouped[section]["specialization"].append({"t": keyword, "e": sorted(e_set)})

        chain = merge_chain or self.build_keyword_merge_chain()
        try:
            merge_out: KeywordMergeOut = chain.invoke({"mentions_json": json.dumps(grouped, ensure_ascii=False)})
        except Exception:
            logger.exception("[grant_keyword_v2] merge failed; fallback=original_mentions")
            return mentions

        merged_rows: List[Dict[str, Any]] = []
        for section in ("research", "application"):
            for keyword in domains_by_section[section]:
                merged_rows.append({"section": section, "bucket": "domain", "keyword": keyword})

            sec_obj = getattr(merge_out, section, None)
            if sec_obj is None:
                continue
            seen_spec = set()
            for spec in list(getattr(sec_obj, "specialization", []) or []):
                keyword = self._norm_keyword(getattr(spec, "t", ""))
                if not keyword or keyword in seen_spec:
                    continue
                seen_spec.add(keyword)
                snippet_ids: Dict[str, float] = {}
                for sid_raw in list(getattr(spec, "e", []) or []):
                    sid = str(sid_raw or "").strip()
                    if not sid:
                        continue
                    snippet_ids[sid] = float(section_sid_conf[section].get(sid, 0.8))
                if not snippet_ids:
                    continue
                merged_rows.append(
                    {
                        "section": section,
                        "bucket": "specialization",
                        "keyword": keyword,
                        "snippet_ids": snippet_ids,
                    }
                )

        if not merged_rows:
            logger.warning("[grant_keyword_v2] merge produced empty output; fallback=original_mentions")
            return mentions
        return merged_rows

    @staticmethod
    def _clip_text(value: Any, max_chars: int = 1800) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max(0, int(max_chars) - 3)].rstrip() + "..."

    def _fetch_grant_weight_context_flat_chunks(
        self,
        *,
        opportunity_id: str,
        max_chunks: int = 400,
        max_chunk_chars: int = 1800,
    ) -> Dict[str, Any]:
        opp_id = str(opportunity_id or "").strip()
        if not opp_id:
            return {"opportunity_id": "", "chunks": []}

        rows = self._fetch_neo4j_rows(
            """
            MATCH (g:Grant {opportunity_id: $opportunity_id})
            RETURN
                g.opportunity_id AS opportunity_id,
                g.opportunity_title AS opportunity_title,
                g.agency_name AS agency_name,
                g.opportunity_status AS opportunity_status,
                g.category AS category,
                g.summary_description AS summary_description
            LIMIT 1
            """,
            {"opportunity_id": opp_id},
        )
        row = rows[0] if rows else {}

        chunk_rows = self.get_grant_chunks_by_opportunity_id(opportunity_id=opp_id)
        has_summary_chunks = any(
            str(c.get("source_type") or "").strip().lower() == "summary"
            for c in (chunk_rows or [])
        )
        chunk_lines: List[str] = []
        for c in chunk_rows:
            chunk_id = str(c.get("chunk_id") or "").strip() or "na"
            source_type = str(c.get("source_type") or "").strip() or "na"
            source_ref_id = str(c.get("source_ref_id") or "").strip() or "na"
            chunk_index = str(c.get("chunk_index") or "").strip() or "na"
            text = str(c.get("text") or "").strip()
            if not text:
                continue
            line = f"{chunk_id} | {source_type}/{source_ref_id}/{chunk_index} | text: {text}"
            chunk_lines.append(self._clip_text(line, max_chars=max_chunk_chars))
            if len(chunk_lines) >= max(1, int(max_chunks)):
                break

        return {
            "opportunity_id": str(row.get("opportunity_id") or opp_id),
            "opportunity_title": row.get("opportunity_title"),
            "agency_name": row.get("agency_name"),
            "opportunity_status": row.get("opportunity_status"),
            "category": row.get("category"),
            "summary_description": None if has_summary_chunks else self._clip_text(row.get("summary_description"), max_chars=6000),
            "chunks": chunk_lines,
        }

    def weight_specializations_for_grant_flat_context(
        self,
        *,
        opportunity_id: str,
        mentions: List[Dict[str, Any]],
        discard_below_weight: float = 0.35,
        weight_chain=None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "research": {"domain": [], "specialization": []},
            "application": {"domain": [], "specialization": []},
        }
        if not mentions:
            return out

        seen_domains = set()
        spec_rows: Dict[str, List[Dict[str, Any]]] = {"research": [], "application": []}
        spec_lookup: Dict[str, Dict[int, Dict[str, Any]]] = {"research": {}, "application": {}}
        spec_seen = set()

        for m in mentions:
            section = str(m.get("section") or "").strip().lower()
            bucket = str(m.get("bucket") or "").strip().lower()
            keyword = self._norm_keyword(m.get("keyword"))
            if section not in {"research", "application"} or bucket not in {"domain", "specialization"} or not keyword:
                continue

            if bucket == "domain":
                d_key = (section, keyword)
                if d_key in seen_domains:
                    continue
                seen_domains.add(d_key)
                out[section]["domain"].append(keyword)
                continue

            s_key = (section, keyword)
            if s_key in spec_seen:
                for row in spec_rows[section]:
                    if self._norm_keyword(row.get("t")) == keyword:
                        src = dict(m.get("snippet_ids") or {})
                        dst = row.setdefault("snippet_ids", {})
                        for sid, conf in src.items():
                            sid_s = str(sid or "").strip()
                            if not sid_s:
                                continue
                            try:
                                c = float(conf)
                            except Exception:
                                c = 0.8
                            c = max(0.0, min(1.0, c))
                            prev = float(dst.get(sid_s, 0.0))
                            dst[sid_s] = max(prev, c)
                        break
                continue

            spec_seen.add(s_key)
            spec_rows[section].append({"t": keyword, "snippet_ids": dict(m.get("snippet_ids") or {})})

        spec_payload: Dict[str, List[Dict[str, Any]]] = {"research": [], "application": []}
        for section in ("research", "application"):
            for idx, row in enumerate(spec_rows[section]):
                t = self._norm_keyword(row.get("t"))
                if not t:
                    continue
                spec_payload[section].append({"idx": idx, "t": t})
                spec_lookup[section][idx] = row

        if not spec_payload["research"] and not spec_payload["application"]:
            return out

        weight_input = {
            "grant_context": self._fetch_grant_weight_context_flat_chunks(opportunity_id=str(opportunity_id)),
            "specializations": spec_payload,
        }
        chain = weight_chain or self.build_specialization_weight_flat_context_chain()
        try:
            weighted_out: WeightedSpecsIdxOut = chain.invoke(
                {"weight_input_json": json.dumps(weight_input, ensure_ascii=False)}
            )
        except Exception:
            logger.exception("[grant_keyword_v2] flat_context_weight failed; fallback=default_weight")
            weighted_out = WeightedSpecsIdxOut()

        threshold = max(0.0, min(1.0, float(discard_below_weight)))
        seen_idx = {"research": set(), "application": set()}

        for section in ("research", "application"):
            for item in list(getattr(weighted_out, section, []) or []):
                idx = int(getattr(item, "idx", -1))
                if idx < 0:
                    continue
                base = spec_lookup[section].get(idx)
                if not base:
                    continue
                seen_idx[section].add(idx)
                w = max(0.0, min(1.0, float(getattr(item, "w", 0.0) or 0.0)))
                if w < threshold:
                    continue
                out[section]["specialization"].append(
                    {
                        "t": self._norm_keyword(base.get("t")),
                        "w": w,
                        "snippet_ids": dict(base.get("snippet_ids") or {}),
                    }
                )

            for idx, base in spec_lookup[section].items():
                if idx in seen_idx[section]:
                    continue
                fallback_w = max(threshold, 0.5)
                out[section]["specialization"].append(
                    {
                        "t": self._norm_keyword(base.get("t")),
                        "w": fallback_w,
                        "snippet_ids": dict(base.get("snippet_ids") or {}),
                    }
                )
        return out

    def save_grant_keywords(
        self,
        *,
        opportunity_id: str,
        keywords: Dict[str, Any],
        raw_json: Optional[Dict[str, Any]] = None,
        source_model: Optional[str] = None,
    ) -> None:
        with SessionLocal() as sess:
            opp_dao = OpportunityDAO(sess)
            opp_dao.upsert_keywords_json(
                [
                    {
                        "opportunity_id": str(opportunity_id),
                        "keywords": keywords,
                        "raw_json": raw_json or {},
                        "source": str(source_model or settings.haiku),
                    }
                ]
            )
            r_domains, a_domains = extract_domains_from_keywords(keywords or {})
            r_vec = embed_domain_bucket(r_domains)
            a_vec = embed_domain_bucket(a_domains)
            if r_vec is not None or a_vec is not None:
                opp_dao.upsert_keyword_embedding(
                    {
                        "opportunity_id": str(opportunity_id),
                        "model": settings.bedrock_embed_model_id,
                        "research_domain_vec": r_vec,
                        "application_domain_vec": a_vec,
                    }
                )
            sess.commit()

    def run_grant_keyword_pipeline(
        self,
        *,
        opportunity_id: str,
        max_context_chars: Optional[int] = None,
        discard_below_weight: float = 0.35,
        use_merge: bool = False,
        persist: bool = True,
    ) -> Dict[str, Any]:
        mentions = self.extract_keyword_mentions_for_grant(
            opportunity_id=str(opportunity_id),
            max_context_chars=max_context_chars,
        )
        merged = self.merge_keyword_mentions_with_llm(mentions=mentions) if use_merge else mentions
        keywords = self.weight_specializations_for_grant_flat_context(
            opportunity_id=str(opportunity_id),
            mentions=merged,
            discard_below_weight=discard_below_weight,
        )
        if persist:
            self.save_grant_keywords(
                opportunity_id=str(opportunity_id),
                keywords=keywords,
                raw_json={
                    "mentions": mentions,
                    "merged_mentions": merged,
                    "use_merge": bool(use_merge),
                },
                source_model=settings.haiku,
            )
        return keywords

    def generate_grant_keywords_for_id(
        self,
        opportunity_id: str,
        *,
        max_context_chars: Optional[int] = None,
        discard_below_weight: float = 0.35,
        use_merge: bool = False,
        persist: bool = True,
    ) -> Dict[str, Any]:
        return self.run_grant_keyword_pipeline(
            opportunity_id=str(opportunity_id),
            max_context_chars=max_context_chars,
            discard_below_weight=discard_below_weight,
            use_merge=use_merge,
            persist=persist,
        )

    def list_all_opportunity_ids(self) -> List[str]:
        with SessionLocal() as sess:
            rows = (
                sess.query(Opportunity.opportunity_id)
                .order_by(Opportunity.opportunity_id.asc())
                .all()
            )
        out: List[str] = []
        for row in rows:
            oid = str(getattr(row, "opportunity_id", "") or "").strip()
            if oid:
                out.append(oid)
        return out

    def run_all_grant_keyword_pipelines_parallel(
        self,
        *,
        max_workers: int = 8,
        max_context_chars: Optional[int] = None,
        discard_below_weight: float = 0.35,
        use_merge: bool = False,
        persist: bool = True,
    ) -> Dict[str, Any]:
        opportunity_ids = self.list_all_opportunity_ids()
        if not opportunity_ids:
            return {"total": 0, "succeeded": 0, "failed": 0, "failed_opportunity_ids": []}

        def _run_one(opp_id: str) -> Dict[str, Any]:
            self.run_grant_keyword_pipeline(
                opportunity_id=str(opp_id),
                max_context_chars=max_context_chars,
                discard_below_weight=discard_below_weight,
                use_merge=use_merge,
                persist=persist,
            )
            return {"opportunity_id": str(opp_id), "ok": True}

        def _on_error(_idx: int, opp_id: str, e: Exception) -> Dict[str, Any]:
            logger.exception("[grant_keyword_v2] grant_pipeline_failed opportunity_id=%s", opp_id)
            return {
                "opportunity_id": str(opp_id),
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            }

        results = parallel_map(
            opportunity_ids,
            max_workers=max(1, int(max_workers)),
            run_item=_run_one,
            on_error=_on_error,
        )
        failed_ids = [str(r["opportunity_id"]) for r in results if not bool(r.get("ok"))]
        return {
            "total": len(opportunity_ids),
            "succeeded": len(opportunity_ids) - len(failed_ids),
            "failed": len(failed_ids),
            "failed_opportunity_ids": failed_ids,
        }


if __name__ == "__main__":
    generator = GrantKeywordGenerator(
        max_context_chars=40_000,
        reserve_prompt_chars=2_000,
    )
    '''
    summary = generator.run_all_grant_keyword_pipelines_parallel(
        max_workers=8,
        max_context_chars=40_000,
        discard_below_weight=0.20,
        use_merge=False,
        persist=True,
    )
    '''
    summary = generator.generate_grant_keywords_for_id(
        opportunity_id='b09d00a3-4b76-4540-8fea-9de1543e83c1',
        max_context_chars=40_000,
        discard_below_weight=0.20,
        use_merge=False,
        persist=False,
    )


    print(summary)
