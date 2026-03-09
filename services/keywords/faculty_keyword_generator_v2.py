from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.prompts import ChatPromptTemplate
from pydantic import AliasChoices, BaseModel, Field, field_validator

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from config import get_llm_client, settings
from dao.faculty_dao import FacultyDAO
from db.db_conn import SessionLocal
from db.models.faculty import Faculty
from logging_setup import setup_logging
from utils.thread_pool import parallel_map

setup_logging()
logger = logging.getLogger(__name__)

FACULTY_V2_DOMAIN_MENTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Extract faculty expertise domains from compact chunk rows.\n\n"
        "Input:\n"
        "- chunks_json: JSON list [{{\"i\": 0, \"t\": \"chunk text\"}}, ...]\n\n"

        "Output JSON schema:\n"
        "{{\n"
        "  \"domains\": [\n"
        "    {{\"t\": \"...\", \"e\": {{\"0\": 0.0}}}}\n"
        "  ]\n"
        "}}\n\n"

        "Task:\n"
        "Identify the major research domains represented in the provided chunks.\n\n"

        "Rules:\n"
        "- Use ONLY the provided chunks.\n"
        "- t must be 2–6 words describing a single faculty expertise concept.\n"
        "- Each domain must represent one atomic research concept.\n"
        "- Domains should correspond to reusable research areas that could appear across many faculty.\n"
        "- Prefer broader research areas rather than narrow technical subtopics.\n"
        "- Avoid overly specific algorithmic components or techniques.\n"
        "- Each domain should correspond to a single node in a knowledge graph.\n"
        "- Do NOT combine multiple concepts in one domain.\n"
        "- Avoid phrases containing 'and', 'for', 'with', or multiple fields.\n\n"

        "Examples of good domains:\n"
        "reinforcement learning\n"
        "robotics\n"
        "bipedal locomotion\n"
        "explainable AI\n"
        "anomaly detection\n"
        "automated planning\n\n"

        "Examples of bad domains:\n"
        "reinforcement learning for robotics\n"
        "robotics and locomotion\n"
        "explainable AI systems\n"
        "machine learning for agriculture\n"
        "policy evaluation\n"
        "tree search planning\n\n"

        "Evidence rules:\n"
        "- e must contain at least one chunk index.\n"
        "- e maps chunk_index(string) -> confidence score in [0,1].\n"
        "- chunk indices in e must be from the input i values only.\n\n"

        "Additional constraints:\n"
        "- Deduplicate similar domains.\n"
        "- Keep the list concise (typically 2–5 domains).\n"
        "- Prefer fewer high-quality domains rather than many granular ones.\n\n"
        "- Return JSON only."
    ),
    ("human", "Chunks JSON:\n{chunks_json}"),
])

FACULTY_V2_SPECIALIZATION_MENTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Extract detailed faculty specialization capabilities from compact chunk rows.\n\n"

        "Input:\n"
        "- chunks_json: JSON list [{{\"i\": 0, \"t\": \"chunk text\"}}, ...]\n"
        "\n"

        "Output JSON schema:\n"
        "{{\n"
        "  \"specializations\": [\n"
        "    {{\"t\": \"...\", \"e\": {{\"0\": 0.0}}}}\n"
        "  ]\n"
        "}}\n\n"

        "Rules:\n"
        "- Use ONLY provided chunks.\n"
        "- t: 8–30 words describing a concrete capability demonstrated by the faculty.\n"
        "- Specializations should reflect real research expertise, typically combining:\n"
        "  method + system + task.\n\n"

        "Examples:\n"
        "- sim-to-real reinforcement learning for bipedal robot locomotion\n"
        "- explainable neural network analysis for image classification models\n"
        "- planning algorithms for autonomous robot navigation in dynamic environments\n"
        "- machine learning prediction models for specialty crop agriculture\n\n"

        "- Avoid vague topics like 'machine learning research'.\n"
        "- Prefer concrete capabilities or research methods.\n"

        "- e must be non-empty chunk_index(string) -> confidence [0,1].\n"
        "- chunk indices in e must be from input i values only.\n"
        "- The caller provides chunkset_count and specialization limits.\n"
        "- Produce between specialization_min and specialization_max specializations depending on the available evidence.\n"
        "- If evidence is sparse, you may output fewer than specialization_min.\n"
        "- Return JSON only."
    ),
    (
        "human",
        "Chunkset key: {chunkset_key}\n"
        "Total chunksets: {chunkset_count}\n"
        "Specialization limits: min={specialization_min}, max={specialization_max}\n\n"
        "Chunks JSON:\n{chunks_json}",
    ),
])

FACULTY_V2_DOMAIN_WEIGHT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Assign importance weights for faculty domains.\n\n"
        "Input:\n"
        "- domains_json: JSON list [{{\"i\": 0, \"t\": \"...\", \"ec\": 3}}, ...]\n\n"
        "Output JSON schema:\n"
        "{{\"items\": [{{\"i\": 0, \"w\": 0.0}}]}}\n\n"
        "Rules:\n"
        "- Use ONLY input.\n"
        "- Return each domain i exactly once.\n"
        "- w in [0,1].\n"
        "- Weight should reflect how central the domain is to the faculty's research program.\n"
        "- 0.85-1.00 core faculty expertise, 0.60-0.84 important, 0.35-0.59 supporting, 0.10-0.34 minor.\n"
        "- Return JSON only."
    ),
    (
        "human",
        "Domains JSON:\n{domains_json}",
    ),
])
FACULTY_V2_SPECIALIZATION_WEIGHT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Assign importance weights for faculty specialization capabilities.\n\n"

        "Input:\n"
        "- domains_json: JSON list [{{\"i\": 0, \"t\": \"...\"}}, ...]\n"
        "- specializations_json: JSON list [{{\"i\": 0, \"t\": \"...\", \"ec\": 2}}, ...]\n\n"

        "Output JSON schema:\n"
        "{{\"items\": [{{\"i\": 0, \"w\": 0.0, \"d\": {{\"0\": 0.0}}}}]}}\n\n"

        "Goal:\n"
        "For each specialization, determine:\n"
        "1. how important the capability is for the faculty's research (w)\n"
        "2. which domains it belongs to (d)\n\n"

        "How to evaluate each specialization:\n"
        "1. Read the specialization text.\n"
        "2. Identify the main research topic or method.\n"
        "3. Select the most relevant domains from domains_json.\n"
        "4. Assign a weight w describing how central this capability is.\n\n"

        "Use the evidence count ec as a signal:\n"
        "- ec represents how many supporting text chunks mention the specialization.\n"
        "- Higher ec generally means the specialization is more central to the faculty's work.\n\n"

        "Rules:\n"
        "- Use ONLY the provided inputs.\n"
        "- Every specialization i must appear exactly once in the output.\n"
        "- Do not skip any specialization.\n"
        "- w must be between 0 and 1.\n"
        "- d maps domain_i (as a string) to a relevance score in [0,1].\n"
        "- Each specialization must link to at least one domain.\n"
        "- Most specializations should link to 1–3 domains.\n"
        "- Prefer a few strong domain links instead of many weak ones.\n"
        "- Only assign domains clearly supported by the specialization description.\n\n"

        "Interpretation of weights:\n"
        "w = importance of the capability in the faculty's research.\n\n"

        "w guidelines:\n"
        "0.85–1.00 → signature capability\n"
        "0.60–0.84 → strong capability\n"
        "0.35–0.59 → supporting research direction\n"
        "0.10–0.34 → minor topic\n\n"

        "d guidelines:\n"
        "1.00 → specialization almost entirely belongs to the domain\n"
        "0.80–0.95 → domain is the primary topic or method\n"
        "0.60–0.79 → domain is a major component\n"
        "0.40–0.59 → domain is secondary\n"
        "0.10–0.39 → weak but meaningful relation\n\n"

        "Avoid assigning 1.00 unless the specialization clearly belongs almost entirely to that domain.\n\n"

        "Return JSON only.\n"
    ),
    (
        "human",
        "Domains JSON:\n{domains_json}\n\nSpecializations JSON:\n{specializations_json}",
    ),
])


FACULTY_V2_MERGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Merge near-duplicate domain keywords.\n\n"
        "Input:\n"
        "- domains_json: JSON list [{{\"i\": 0, \"t\": \"...\"}}, ...]\n\n"
        "Output JSON schema:\n"
        "{{\n"
        "  \"domains\": [{{\"t\": \"...\", \"ixs\": [0, 2]}}]\n"
        "}}\n\n"
        "Rules:\n"
        "- Merge only semantically equivalent or near-duplicate phrases.\n"
        "- Keep distinct concepts separate.\n"
        "- For each output item, ixs must reference valid input i values.\n"
        "- Every input domain i should appear in exactly one output item.\n"
        "- Choose concise canonical text for t.\n"
        "- Return JSON only."
    ),
    (
        "human",
        "Domains JSON:\n{domains_json}",
    ),
])


class _EvidenceMixin(BaseModel):
    e: Dict[str, float] = Field(default_factory=dict)

    @field_validator("e", mode="before")
    @classmethod
    def _coerce_evidence(cls, v):
        if isinstance(v, dict):
            out: Dict[str, float] = {}
            for k, val in v.items():
                sid = str(k or "").strip()
                if not sid:
                    continue
                try:
                    conf = float(val)
                except Exception:
                    conf = 0.8
                out[sid] = max(0.0, min(1.0, conf))
            return out
        if isinstance(v, list):
            out: Dict[str, float] = {}
            for x in v:
                sid = str(x or "").strip()
                if sid:
                    out[sid] = 0.8
            return out
        return {}


class _DomainLinkMixin(BaseModel):
    d: Dict[str, float] = Field(default_factory=dict)

    @field_validator("d", mode="before")
    @classmethod
    def _coerce_domain_links(cls, v):
        if isinstance(v, dict):
            out: Dict[str, float] = {}
            for k, val in v.items():
                key = str(k or "").strip()
                if not key:
                    continue
                try:
                    rel = float(val)
                except Exception:
                    rel = 0.0
                out[key] = max(0.0, min(1.0, rel))
            return out
        return {}


class FacultyV2DomainMention(_EvidenceMixin):
    t: str = ""


class FacultyV2DomainMentionOut(BaseModel):
    domains: List[FacultyV2DomainMention] = Field(default_factory=list)


class FacultyV2SpecializationMention(_EvidenceMixin):
    t: str = ""


class FacultyV2SpecializationMentionOut(BaseModel):
    specializations: List[FacultyV2SpecializationMention] = Field(default_factory=list)


class _WeightedIdxItem(BaseModel):
    i: int = Field(default=-1, validation_alias=AliasChoices("i", "idx"))
    w: float = Field(default=0.5, ge=0.0, le=1.0)


class _WeightedIdxOut(BaseModel):
    items: List[_WeightedIdxItem] = Field(default_factory=list)


class _WeightedSpecItem(_DomainLinkMixin):
    i: int = Field(default=-1, validation_alias=AliasChoices("i", "idx"))
    w: float = Field(default=0.5, ge=0.0, le=1.0)


class _WeightedSpecOut(BaseModel):
    items: List[_WeightedSpecItem] = Field(default_factory=list)


class _MergeItem(BaseModel):
    t: str = ""
    ixs: List[int] = Field(default_factory=list, validation_alias=AliasChoices("ixs", "idxs"))


class _MergeOut(BaseModel):
    domains: List[_MergeItem] = Field(default_factory=list)


class FacultyKeywordGeneratorV2:
    """
    Unified faculty keyword generator v2.

    Pipeline:
    1) Read faculty evidence chunks from Neo4j (FacultyTextChunk + publication abstracts as synthetic chunks).
    2) Pack into chunksets up to context-size budget.
    3) Per chunkset (parallel):
       - Extract domain mentions with chunk evidence.
       - Extract specialization mentions with chunk evidence.
    4) Deterministically merge mentions.
    5) Run separate LLM weighting for domains and specializations; specialization weighting also refreshes d-links.
    6) Re-assemble final keyword JSON with chunk_links.
    7) Persist keywords/raw_json to Postgres.
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
    def build_domain_mention_chain(prompt=FACULTY_V2_DOMAIN_MENTION_PROMPT):
        llm = get_llm_client().build()
        return prompt | llm.with_structured_output(FacultyV2DomainMentionOut)

    @staticmethod
    def build_specialization_mention_chain(prompt=FACULTY_V2_SPECIALIZATION_MENTION_PROMPT):
        llm = get_llm_client().build()
        return prompt | llm.with_structured_output(FacultyV2SpecializationMentionOut)

    @staticmethod
    def build_domain_weight_chain(prompt=FACULTY_V2_DOMAIN_WEIGHT_PROMPT):
        llm = get_llm_client().build()
        return prompt | llm.with_structured_output(_WeightedIdxOut)

    @staticmethod
    def build_specialization_weight_chain(prompt=FACULTY_V2_SPECIALIZATION_WEIGHT_PROMPT):
        llm = get_llm_client().build()
        return prompt | llm.with_structured_output(_WeightedSpecOut)

    @staticmethod
    def build_merge_chain(prompt=FACULTY_V2_MERGE_PROMPT):
        llm = get_llm_client().build()
        return prompt | llm.with_structured_output(_MergeOut)

    def get_faculty_chunks_by_id(self, *, faculty_id: int) -> List[Dict[str, Any]]:
        try:
            fid = int(faculty_id)
        except Exception:
            return []
        if fid <= 0:
            return []

        rows = self._fetch_neo4j_rows(
            """
            MATCH (f:Faculty {faculty_id: $faculty_id})
            CALL (f) {
                MATCH (f)-[r]->(c:FacultyTextChunk)
                WHERE c.chunk_id IS NOT NULL
                  AND c.text IS NOT NULL
                  AND trim(toString(c.text)) <> ''
                RETURN
                    c.chunk_id AS chunk_id,
                    elementId(c) AS chunk_node_id,
                    type(r) AS relation,
                    c.source_type AS source_type,
                    c.source_ref_id AS source_ref_id,
                    c.source_url AS source_url,
                    c.chunk_index AS chunk_index,
                    c.char_count AS char_count,
                    c.text AS text
                UNION ALL
                MATCH (f)-[:AUTHORED]->(p:FacultyPublication)
                WHERE p.publication_id IS NOT NULL
                  AND p.abstract IS NOT NULL
                  AND trim(toString(p.abstract)) <> ''
                RETURN
                    ('pub|' + coalesce(toLower(f.email), '') + '|' + toString(f.faculty_id) + '|' + toString(p.publication_id)) AS chunk_id,
                    elementId(p) AS chunk_node_id,
                    'AUTHORED_ABSTRACT_CHUNK' AS relation,
                    'publication_abstract' AS source_type,
                    toString(p.publication_id) AS source_ref_id,
                    null AS source_url,
                    0 AS chunk_index,
                    size(toString(p.abstract)) AS char_count,
                    toString(p.abstract) AS text
            }
            RETURN
                chunk_id,
                chunk_node_id,
                relation,
                source_type,
                source_ref_id,
                source_url,
                chunk_index,
                char_count,
                text
            ORDER BY source_type ASC, source_ref_id ASC, chunk_index ASC, chunk_id ASC
            """,
            {"faculty_id": fid},
        )

        out: List[Dict[str, Any]] = []
        seen_chunk_ids = set()
        for row in rows[: self.max_neo4j_chunks]:
            chunk_id = str(row.get("chunk_id") or "").strip()
            text = str(row.get("text") or "")
            if not chunk_id or not text.strip() or chunk_id in seen_chunk_ids:
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
                    "text": text,
                }
            )
        return out

    def build_chunk_payload_windows(
        self,
        *,
        chunks: Sequence[Dict[str, Any]],
        max_context_chars: Optional[int] = None,
    ) -> List[List[Dict[str, str]]]:
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
        windows = self.build_chunk_payload_windows(
            chunks=chunks,
            max_context_chars=max_context_chars,
        )
        return {
            "fac_chunk": {
                f"chunkset{i}": window
                for i, window in enumerate(windows, start=1)
            }
        }

    def build_faculty_chunk_payload(
        self,
        *,
        faculty_id: int,
        max_context_chars: Optional[int] = None,
    ) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        chunks = self.get_faculty_chunks_by_id(faculty_id=int(faculty_id))
        return self.build_chunk_payload(chunks=chunks, max_context_chars=max_context_chars)

    @staticmethod
    def _norm(value: Any) -> str:
        return " ".join(str(value or "").split()).strip().lower()

    @staticmethod
    def _clip_text(value: Any, max_chars: int = 1600) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max(0, int(max_chars) - 3)].rstrip() + "..."

    @staticmethod
    def _compute_specialization_limits(
        *,
        chunkset_count: int,
        max_total_specializations: int = 24,
        min_per_chunkset: int = 2,
    ) -> Tuple[int, int]:
        count = max(1, int(chunkset_count or 1))
        max_total = max(1, int(max_total_specializations or 24))
        min_per = max(1, int(min_per_chunkset or 2))
        if count == 1:
            return min_per, max_total
        per_chunk_max = max(min_per, max_total // count)
        per_chunk_max = min(6, per_chunk_max)
        return min_per, per_chunk_max

    @staticmethod
    def _to_log_obj(value: Any) -> Any:
        try:
            if hasattr(value, "model_dump"):
                return value.model_dump()
        except Exception:
            pass
        return value

    def _log_llm_io(self, *, step: str, llm_input: Dict[str, Any], llm_output: Any) -> None:
        try:
            logger.info(
                "[faculty_keyword_generator_v2][%s] input=%s",
                step,
                json.dumps(llm_input, ensure_ascii=False),
            )
        except Exception:
            logger.info("[faculty_keyword_generator_v2][%s] input=%s", step, str(llm_input))

        out_obj = self._to_log_obj(llm_output)
        try:
            logger.info(
                "[faculty_keyword_generator_v2][%s] output=%s",
                step,
                json.dumps(out_obj, ensure_ascii=False),
            )
        except Exception:
            logger.info("[faculty_keyword_generator_v2][%s] output=%s", step, str(out_obj))

    def _extract_for_chunkset(
        self,
        *,
        chunkset_key: str,
        chunkset_count: int,
        specialization_min: int,
        specialization_max: int,
        chunk_rows: List[Dict[str, Any]],
        domain_chain,
        specialization_chain,
    ) -> Dict[str, Any]:
        compact_rows: List[Dict[str, Any]] = []
        idx_to_chunk_id: Dict[str, str] = {}
        valid_chunk_ids = set()
        for row in chunk_rows:
            chunk_id = str((row or {}).get("chunk_id") or "").strip()
            text = str((row or {}).get("text") or "")
            if not chunk_id or not text.strip():
                continue
            idx = len(compact_rows)
            compact_rows.append({"i": idx, "t": text})
            idx_to_chunk_id[str(idx)] = chunk_id
            valid_chunk_ids.add(chunk_id)

        if not compact_rows:
            return {"chunkset": chunkset_key, "domains": [], "specializations": []}

        def _resolve_evidence_chunk_id(raw_key: Any) -> str:
            key = str(raw_key or "").strip()
            if not key:
                return ""
            mapped = idx_to_chunk_id.get(key)
            if mapped:
                return mapped
            try:
                mapped = idx_to_chunk_id.get(str(int(float(key))))
                if mapped:
                    return mapped
            except Exception:
                pass
            if key in valid_chunk_ids:
                return key
            return ""

        chunks_json = json.dumps(compact_rows, ensure_ascii=False)
        domain_input = {"chunks_json": chunks_json}
        domain_out: FacultyV2DomainMentionOut = domain_chain.invoke(domain_input)
        self._log_llm_io(step=f"{chunkset_key}.domain_mention", llm_input=domain_input, llm_output=domain_out)

        domains: List[Dict[str, Any]] = []
        seen_domain = set()
        for item in list(getattr(domain_out, "domains", []) or []):
            d = self._norm(getattr(item, "t", ""))
            if not d or d in seen_domain:
                continue
            sid_map: Dict[str, float] = {}
            for sid_raw, conf_raw in dict(getattr(item, "e", {}) or {}).items():
                sid = _resolve_evidence_chunk_id(sid_raw)
                if not sid:
                    continue
                try:
                    conf = float(conf_raw)
                except Exception:
                    conf = 0.8
                sid_map[sid] = max(0.0, min(1.0, conf))
            if not sid_map:
                continue
            seen_domain.add(d)
            domains.append({"domain": d, "snippet_ids": sid_map, "chunkset": chunkset_key})

        if not domains:
            return {"chunkset": chunkset_key, "domains": [], "specializations": []}

        spec_input = {
            "chunkset_key": str(chunkset_key),
            "chunkset_count": int(chunkset_count),
            "specialization_min": int(specialization_min),
            "specialization_max": int(specialization_max),
            "chunks_json": chunks_json,
        }
        spec_out: FacultyV2SpecializationMentionOut = specialization_chain.invoke(spec_input)
        self._log_llm_io(step=f"{chunkset_key}.specialization_mention", llm_input=spec_input, llm_output=spec_out)

        specs: List[Dict[str, Any]] = []
        seen_spec = set()
        for item in list(getattr(spec_out, "specializations", []) or []):
            t = self._norm(getattr(item, "t", ""))
            if not t:
                continue

            domain_weights: Dict[str, float] = {}

            sid_map: Dict[str, float] = {}
            for sid_raw, conf_raw in dict(getattr(item, "e", {}) or {}).items():
                sid = _resolve_evidence_chunk_id(sid_raw)
                if not sid:
                    continue
                try:
                    conf = float(conf_raw)
                except Exception:
                    conf = 0.8
                sid_map[sid] = max(0.0, min(1.0, conf))
            if not sid_map:
                continue

            key = t
            if key in seen_spec:
                continue
            seen_spec.add(key)
            specs.append(
                {
                    "specialization": t,
                    "domain_weights": domain_weights,
                    "snippet_ids": sid_map,
                    "chunkset": chunkset_key,
                }
            )

        return {"chunkset": chunkset_key, "domains": domains, "specializations": specs}

    def extract_domain_spec_mentions_from_payload(
        self,
        *,
        fac_chunk_payload: Dict[str, Any],
        domain_chain=None,
        specialization_chain=None,
    ) -> Dict[str, Any]:
        fac_chunk = (fac_chunk_payload or {}).get("fac_chunk") or {}
        if not isinstance(fac_chunk, dict):
            return {"domains": [], "specializations": []}

        chunkset_items: List[Tuple[str, List[Dict[str, Any]]]] = []
        for chunkset_key in sorted(fac_chunk.keys()):
            rows = fac_chunk.get(chunkset_key) or []
            if not isinstance(rows, list) or not rows:
                continue
            chunkset_items.append((chunkset_key, rows))
        if not chunkset_items:
            return {"domains": [], "specializations": []}

        chunkset_count = len(chunkset_items)
        specialization_min, specialization_max = self._compute_specialization_limits(
            chunkset_count=chunkset_count,
            max_total_specializations=24,
            min_per_chunkset=2,
        )
        #TODO: remove hard coded
        specialization_min = 2
        specialization_max = 12
        logger.info(
            "[faculty_keyword_generator_v2] chunkset_count=%d specialization_limits=min:%d max:%d",
            chunkset_count,
            specialization_min,
            specialization_max,
        )

        workers = len(chunkset_items)

        def _run_one(item: Tuple[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
            c_key, c_rows = item
            d_chain = domain_chain or self.build_domain_mention_chain()
            s_chain = specialization_chain or self.build_specialization_mention_chain()
            return self._extract_for_chunkset(
                chunkset_key=c_key,
                chunkset_count=chunkset_count,
                specialization_min=specialization_min,
                specialization_max=specialization_max,
                chunk_rows=c_rows,
                domain_chain=d_chain,
                specialization_chain=s_chain,
            )

        def _on_error(_idx: int, item: Tuple[str, List[Dict[str, Any]]], _e: Exception) -> Dict[str, Any]:
            logger.exception("[faculty_keyword_generator_v2] chunkset=%s failed", item[0])
            return {"chunkset": item[0], "domains": [], "specializations": []}

        parts = parallel_map(
            chunkset_items,
            max_workers=workers,
            run_item=_run_one,
            on_error=_on_error,
        )

        out_domains: List[Dict[str, Any]] = []
        out_specs: List[Dict[str, Any]] = []
        for part in parts:
            out_domains.extend(list(part.get("domains") or []))
            out_specs.extend(list(part.get("specializations") or []))

        return {"domains": out_domains, "specializations": out_specs}

    def extract_domain_spec_mentions_for_faculty(
        self,
        *,
        faculty_id: int,
        max_context_chars: Optional[int] = None,
        domain_chain=None,
        specialization_chain=None,
    ) -> Dict[str, Any]:
        payload = self.build_faculty_chunk_payload(
            faculty_id=int(faculty_id),
            max_context_chars=max_context_chars,
        )
        return self.extract_domain_spec_mentions_from_payload(
            fac_chunk_payload=payload,
            domain_chain=domain_chain,
            specialization_chain=specialization_chain,
        )

    def merge_mentions_to_structure(self, *, mentions: Dict[str, Any]) -> Dict[str, Any]:
        domain_acc: Dict[str, Dict[str, Any]] = {}
        spec_acc: Dict[str, Dict[str, Any]] = {}

        for row in list((mentions or {}).get("domains") or []):
            d = self._norm(row.get("domain"))
            if not d:
                continue
            current = domain_acc.setdefault(
                d,
                {
                    "t": d,
                    "snippet_ids": {},
                    "w": 0.0,
                },
            )
            for sid_raw, conf_raw in dict(row.get("snippet_ids") or {}).items():
                sid = str(sid_raw or "").strip()
                if not sid:
                    continue
                try:
                    conf = float(conf_raw)
                except Exception:
                    conf = 0.8
                conf = max(0.0, min(1.0, conf))
                prev = float(current["snippet_ids"].get(sid, 0.0))
                current["snippet_ids"][sid] = max(prev, conf)

        for row in list((mentions or {}).get("specializations") or []):
            t = self._norm(row.get("specialization"))
            if not t:
                continue

            current = spec_acc.setdefault(
                t,
                {
                    "t": t,
                    "domains": {},
                    "snippet_ids": {},
                    "w": 0.0,
                },
            )

            for d_raw, rel_raw in dict(row.get("domain_weights") or {}).items():
                d = self._norm(d_raw)
                if not d:
                    continue
                rel = max(0.0, min(1.0, float(rel_raw or 0.0)))
                if rel <= 0.0:
                    continue
                prev = float(current["domains"].get(d, 0.0))
                current["domains"][d] = max(prev, rel)

                if d not in domain_acc:
                    domain_acc[d] = {
                        "t": d,
                        "snippet_ids": {},
                        "w": 0.0,
                    }

            for sid_raw, conf_raw in dict(row.get("snippet_ids") or {}).items():
                sid = str(sid_raw or "").strip()
                if not sid:
                    continue
                try:
                    conf = float(conf_raw)
                except Exception:
                    conf = 0.8
                conf = max(0.0, min(1.0, conf))
                prev = float(current["snippet_ids"].get(sid, 0.0))
                current["snippet_ids"][sid] = max(prev, conf)

        domains_out = sorted(
            list(domain_acc.values()),
            key=lambda x: str(x.get("t") or ""),
        )
        specs_out = sorted(
            list(spec_acc.values()),
            key=lambda x: str(x.get("t") or ""),
        )

        return {
            "domains": domains_out,
            "specializations": specs_out,
        }

    def merge_domains_and_specializations_with_llm(
        self,
        *,
        structure: Dict[str, Any],
        merge_chain=None,
    ) -> Dict[str, Any]:
        domains = list((structure or {}).get("domains") or [])
        specs = list((structure or {}).get("specializations") or [])
        if not domains:
            return {"domains": [], "specializations": specs}

        domains_json = [
            {
                "i": idx,
                "t": str(d.get("t") or "").strip(),
            }
            for idx, d in enumerate(domains)
            if str(d.get("t") or "").strip()
        ]
        if not domains_json:
            return {"domains": domains, "specializations": specs}

        chain = merge_chain or self.build_merge_chain()
        try:
            merge_input = {
                "domains_json": json.dumps(domains_json, ensure_ascii=False),
            }
            out: _MergeOut = chain.invoke(merge_input)
            self._log_llm_io(step="merge_domains", llm_input=merge_input, llm_output=out)
        except Exception:
            logger.exception("[faculty_keyword_generator_v2] merge_llm_failed; fallback=deterministic")
            return {"domains": domains, "specializations": specs}
        used = set()
        merged_domains: List[Dict[str, Any]] = []
        for m in list(getattr(out, "domains", []) or []):
            t = self._norm(getattr(m, "t", ""))
            idxs_raw = list(getattr(m, "ixs", []) or [])
            idxs: List[int] = []
            for x in idxs_raw:
                try:
                    idx = int(x)
                except Exception:
                    continue
                if idx < 0 or idx >= len(domains):
                    continue
                if idx in idxs:
                    continue
                idxs.append(idx)
            if not t or not idxs:
                continue
            sid_map: Dict[str, float] = {}
            max_w = 0.0
            for idx in idxs:
                used.add(idx)
                row = domains[idx]
                max_w = max(max_w, float(row.get("w") or 0.0))
                for sid_raw, conf_raw in dict(row.get("snippet_ids") or {}).items():
                    sid = str(sid_raw or "").strip()
                    if not sid:
                        continue
                    conf = max(0.0, min(1.0, float(conf_raw or 0.0)))
                    prev = float(sid_map.get(sid, 0.0))
                    sid_map[sid] = max(prev, conf)
            merged_domains.append({"t": t, "snippet_ids": sid_map, "w": max_w})

        for idx, row in enumerate(domains):
            if idx in used:
                continue
            merged_domains.append(
                {
                    "t": self._norm(row.get("t")),
                    "snippet_ids": dict(row.get("snippet_ids") or {}),
                    "w": float(row.get("w") or 0.0),
                }
            )

        dedup: Dict[str, Dict[str, Any]] = {}
        for item in merged_domains:
            key = self._norm(item.get("t"))
            if not key:
                continue
            cur = dedup.setdefault(key, {"t": key, "snippet_ids": {}, "w": 0.0})
            cur["w"] = max(float(cur.get("w") or 0.0), float(item.get("w") or 0.0))
            for sid_raw, conf_raw in dict(item.get("snippet_ids") or {}).items():
                sid = str(sid_raw or "").strip()
                if not sid:
                    continue
                conf = max(0.0, min(1.0, float(conf_raw or 0.0)))
                prev = float(cur["snippet_ids"].get(sid, 0.0))
                cur["snippet_ids"][sid] = max(prev, conf)

        out_domains = list(dedup.values())
        out_domains.sort(key=lambda x: str(x.get("t") or ""))
        return {"domains": out_domains, "specializations": specs}

    def _build_weight_context_chunks(
        self,
        *,
        faculty_id: int,
        evidence_chunk_ids: Sequence[str],
        max_chunks: int = 400,
        max_chunk_chars: int = 1600,
    ) -> List[Dict[str, str]]:
        all_chunks = self.get_faculty_chunks_by_id(faculty_id=int(faculty_id))
        by_id = {
            str(c.get("chunk_id") or "").strip(): str(c.get("text") or "").strip()
            for c in all_chunks
            if str(c.get("chunk_id") or "").strip() and str(c.get("text") or "").strip()
        }

        target_ids = [str(x or "").strip() for x in list(evidence_chunk_ids or []) if str(x or "").strip()]
        rows: List[Dict[str, str]] = []
        seen = set()

        for cid in target_ids:
            txt = by_id.get(cid, "")
            if not txt or cid in seen:
                continue
            seen.add(cid)
            rows.append({"chunk_id": cid, "text": self._clip_text(txt, max_chars=max_chunk_chars)})
            if len(rows) >= max(1, int(max_chunks)):
                return rows

        for cid, txt in by_id.items():
            if cid in seen:
                continue
            rows.append({"chunk_id": cid, "text": self._clip_text(txt, max_chars=max_chunk_chars)})
            if len(rows) >= max(1, int(max_chunks)):
                break

        return rows

    def weight_domains_and_specializations(
        self,
        *,
        faculty_id: int,
        structure: Dict[str, Any],
        domain_weight_chain=None,
        specialization_weight_chain=None,
    ) -> Dict[str, Any]:
        domains = list((structure or {}).get("domains") or [])
        specs = list((structure or {}).get("specializations") or [])
        if not domains and not specs:
            return {"domains": [], "specializations": []}

        d_chain = domain_weight_chain or self.build_domain_weight_chain()
        domain_json_obj = [
            {
                "i": idx,
                "t": str(d.get("t") or "").strip(),
                "ec": len(dict(d.get("snippet_ids") or {})),
            }
            for idx, d in enumerate(domains)
            if str(d.get("t") or "").strip()
        ]

        d_weights: Dict[int, float] = {}
        if domain_json_obj:
            try:
                d_input = {
                    "domains_json": json.dumps(domain_json_obj, ensure_ascii=False),
                }
                d_out: _WeightedIdxOut = d_chain.invoke(d_input)
                self._log_llm_io(step="domain_weight", llm_input=d_input, llm_output=d_out)
                for it in list(getattr(d_out, "items", []) or []):
                    idx = int(getattr(it, "i", -1))
                    if idx < 0:
                        continue
                    d_weights[idx] = max(0.0, min(1.0, float(getattr(it, "w", 0.5) or 0.5)))
            except Exception:
                logger.exception("[faculty_keyword_generator_v2] domain_weight_failed; fallback=heuristic")

        for idx, d in enumerate(domains):
            if idx in d_weights:
                d["w"] = d_weights[idx]
            else:
                max_conf = max([float(v) for v in dict(d.get("snippet_ids") or {}).values()] or [0.0])
                d["w"] = max(0.2, min(1.0, 0.35 + 0.65 * max_conf))

        s_chain = specialization_weight_chain or self.build_specialization_weight_chain()
        spec_json_obj = [
            {
                "i": idx,
                "t": str(s.get("t") or "").strip(),
                "ec": len(dict(s.get("snippet_ids") or {})),
            }
            for idx, s in enumerate(specs)
            if str(s.get("t") or "").strip()
        ]

        s_weights: Dict[int, float] = {}
        s_domain_links: Dict[int, Dict[str, float]] = {}
        if spec_json_obj:
            try:
                s_input = {
                    "domains_json": json.dumps(
                        [{"i": idx, "t": str(d.get("t") or "")} for idx, d in enumerate(domains)],
                        ensure_ascii=False,
                    ),
                    "specializations_json": json.dumps(spec_json_obj, ensure_ascii=False),
                }
                s_out: _WeightedSpecOut = s_chain.invoke(s_input)
                self._log_llm_io(step="specialization_weight", llm_input=s_input, llm_output=s_out)
                for it in list(getattr(s_out, "items", []) or []):
                    idx = int(getattr(it, "i", -1))
                    if idx < 0:
                        continue
                    s_weights[idx] = max(0.0, min(1.0, float(getattr(it, "w", 0.5) or 0.5)))
                    d_map: Dict[str, float] = {}
                    for d_idx_raw, rel_raw in dict(getattr(it, "d", {}) or {}).items():
                        try:
                            d_idx = int(str(d_idx_raw).strip())
                        except Exception:
                            continue
                        if d_idx < 0 or d_idx >= len(domains):
                            continue
                        d_name = str(domains[d_idx].get("t") or "").strip()
                        if not d_name:
                            continue
                        rel = max(0.0, min(1.0, float(rel_raw or 0.0)))
                        if rel <= 0.0:
                            continue
                        prev = float(d_map.get(d_name, 0.0))
                        d_map[d_name] = max(prev, rel)
                    if d_map:
                        s_domain_links[idx] = d_map
            except Exception:
                logger.exception("[faculty_keyword_generator_v2] specialization_weight_failed; fallback=heuristic")

        for idx, s in enumerate(specs):
            if idx in s_weights:
                s["w"] = s_weights[idx]
            else:
                max_conf = max([float(v) for v in dict(s.get("snippet_ids") or {}).values()] or [0.0])
                s["w"] = max(0.2, min(1.0, 0.35 + 0.65 * max_conf))
            if idx in s_domain_links:
                s["domains"] = dict(s_domain_links[idx])

        return {
            "domains": domains,
            "specializations": specs,
        }

    def assemble_keywords(self, *, structure: Dict[str, Any]) -> Dict[str, Any]:
        domains = list((structure or {}).get("domains") or [])
        specs = list((structure or {}).get("specializations") or [])

        domain_specs: Dict[str, List[Dict[str, Any]]] = {str(d.get("t") or "").strip(): [] for d in domains}
        for s in specs:
            s_t = str(s.get("t") or "").strip()
            if not s_t:
                continue
            for d_name_raw, rel_raw in dict(s.get("domains") or {}).items():
                d_name = str(d_name_raw or "").strip()
                if d_name not in domain_specs:
                    continue
                domain_specs[d_name].append(
                    {
                        "t": s_t,
                        "domain_weight": max(0.0, min(1.0, float(rel_raw or 0.0))),
                        "weight": max(0.0, min(1.0, float(s.get("w") or 0.0))),
                        "snippet_ids": dict(s.get("snippet_ids") or {}),
                    }
                )

        chunk_map: Dict[str, Dict[str, Any]] = {}

        domains_out: List[Dict[str, Any]] = []
        for d in domains:
            d_name = str(d.get("t") or "").strip()
            if not d_name:
                continue
            sid_map = dict(d.get("snippet_ids") or {})
            d_weight = max(0.0, min(1.0, float(d.get("w") or 0.0)))

            for sid in sid_map.keys():
                sid_s = str(sid or "").strip()
                if not sid_s:
                    continue
                c = chunk_map.setdefault(sid_s, {"chunk_id": sid_s, "domains": {}, "specializations": {}})
                prev = float((c["domains"].get(d_name) or {}).get("weight", 0.0))
                c["domains"][d_name] = {
                    "weight": max(prev, d_weight),
                }

            domains_out.append(
                {
                    "t": d_name,
                    "w": d_weight,
                    "snippet_ids": sid_map,
                    "specializations": sorted(
                        list(domain_specs.get(d_name) or []),
                        key=lambda x: (float(x.get("weight") or 0.0), float(x.get("domain_weight") or 0.0), str(x.get("t") or "")),
                        reverse=True,
                    ),
                }
            )

        specs_out: List[Dict[str, Any]] = []
        for s in specs:
            s_t = str(s.get("t") or "").strip()
            if not s_t:
                continue
            s_weight = max(0.0, min(1.0, float(s.get("w") or 0.0)))
            s_domains = {
                str(k): max(0.0, min(1.0, float(v or 0.0)))
                for k, v in dict(s.get("domains") or {}).items()
                if str(k).strip()
            }
            sid_map = dict(s.get("snippet_ids") or {})

            for sid in sid_map.keys():
                sid_s = str(sid or "").strip()
                if not sid_s:
                    continue
                c = chunk_map.setdefault(sid_s, {"chunk_id": sid_s, "domains": {}, "specializations": {}})
                prev = float((c["specializations"].get(s_t) or {}).get("weight", 0.0))
                c["specializations"][s_t] = {
                    "domains": dict(s_domains),
                    "weight": max(prev, s_weight),
                }

            specs_out.append(
                {
                    "t": s_t,
                    "w": s_weight,
                    "weight": s_weight,
                    "domains": s_domains,
                    "snippet_ids": sid_map,
                }
            )

        domains_out.sort(key=lambda x: (float(x.get("w") or 0.0), str(x.get("t") or "")), reverse=True)
        specs_out.sort(key=lambda x: (float(x.get("w") or 0.0), str(x.get("t") or "")), reverse=True)

        return {
            "domains": domains_out,
            "specializations": specs_out,
            "chunk_links": sorted(list(chunk_map.values()), key=lambda x: str(x.get("chunk_id") or "")),
        }

    def save_faculty_keywords(
        self,
        *,
        faculty_id: int,
        keywords: Dict[str, Any],
        raw_json: Optional[Dict[str, Any]] = None,
        source_model: Optional[str] = None,
    ) -> None:
        with SessionLocal() as sess:
            fac_dao = FacultyDAO(sess)
            fac_dao.upsert_keywords_json(
                [
                    {
                        "faculty_id": int(faculty_id),
                        "keywords": keywords,
                        "raw_json": raw_json or {},
                        "source": str(source_model or settings.haiku),
                    }
                ]
            )
            sess.commit()

    def run_faculty_keyword_pipeline(
        self,
        *,
        faculty_id: int,
        max_context_chars: Optional[int] = None,
        persist: bool = True,
        use_llm_merge: bool = False,
    ) -> Dict[str, Any]:
        mentions = self.extract_domain_spec_mentions_for_faculty(
            faculty_id=int(faculty_id),
            max_context_chars=max_context_chars,
        )
        merged = self.merge_mentions_to_structure(mentions=mentions)
        merged_llm = (
            self.merge_domains_and_specializations_with_llm(structure=merged)
            if bool(use_llm_merge)
            else merged
        )
        weighted = self.weight_domains_and_specializations(
            faculty_id=int(faculty_id),
            structure=merged_llm,
        )
        keywords = self.assemble_keywords(structure=weighted)

        if persist:
            self.save_faculty_keywords(
                faculty_id=int(faculty_id),
                keywords=keywords,
                raw_json={
                    "mentions": mentions,
                    "merged": merged,
                    "merged_llm": merged_llm,
                    "weighted": weighted,
                },
                source_model=settings.haiku,
            )

        return keywords

    def generate_faculty_keywords_for_id(
        self,
        faculty_id: int,
        *,
        max_context_chars: Optional[int] = None,
        persist: bool = True,
        use_llm_merge: bool = False,
    ) -> Dict[str, Any]:
        return self.run_faculty_keyword_pipeline(
            faculty_id=int(faculty_id),
            max_context_chars=max_context_chars,
            persist=persist,
            use_llm_merge=use_llm_merge,
        )

    def list_all_faculty_ids(self) -> List[int]:
        with SessionLocal() as sess:
            rows = (
                sess.query(Faculty.faculty_id)
                .order_by(Faculty.faculty_id.asc())
                .all()
            )
        out: List[int] = []
        for row in rows:
            try:
                fid = int(getattr(row, "faculty_id", None))
            except Exception:
                continue
            if fid > 0:
                out.append(fid)
        return out

    def run_all_faculty_keyword_pipelines_parallel(
        self,
        *,
        max_workers: int = 8,
        max_context_chars: Optional[int] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:
        faculty_ids = self.list_all_faculty_ids()
        if not faculty_ids:
            return {"total": 0, "succeeded": 0, "failed": 0, "failed_faculty_ids": []}

        def _run_one(fid: int) -> Dict[str, Any]:
            self.run_faculty_keyword_pipeline(
                faculty_id=int(fid),
                max_context_chars=max_context_chars,
                persist=persist,
            )
            return {"faculty_id": int(fid), "ok": True}

        def _on_error(_idx: int, fid: int, e: Exception) -> Dict[str, Any]:
            logger.exception("[faculty_keyword_generator_v2] faculty_pipeline_failed faculty_id=%s", fid)
            return {
                "faculty_id": int(fid),
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            }

        results = parallel_map(
            faculty_ids,
            max_workers=max(1, int(max_workers)),
            run_item=_run_one,
            on_error=_on_error,
        )

        failed_ids = [int(r["faculty_id"]) for r in results if not bool(r.get("ok"))]
        return {
            "total": len(faculty_ids),
            "succeeded": len(faculty_ids) - len(failed_ids),
            "failed": len(failed_ids),
            "failed_faculty_ids": failed_ids,
        }


if __name__ == "__main__":
    generator = FacultyKeywordGeneratorV2(
        max_context_chars=40_000,
        reserve_prompt_chars=3_000,
    )

    summary = generator.generate_faculty_keywords_for_id(
        faculty_id=103,
        max_context_chars=40_000,
        persist=False,
        use_llm_merge=True,
    )
    print(summary)
