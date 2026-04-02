from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import get_embedding_client, get_llm_client, settings
from utils.embedder import cosine_sim_matrix, embed_texts
from utils.thread_pool import parallel_map, resolve_pool_size

logger = logging.getLogger(__name__)


class ClusterRenameOut(BaseModel):
    canonical: str = Field(default="")


@dataclass
class Mention:
    text: str
    norm: str
    batch_idx: int
    weight: float


@dataclass
class LexicalConcept:
    text: str
    norm: str
    mention_count: int = 0
    batch_ids: set[int] = field(default_factory=set)
    weights: List[float] = field(default_factory=list)

    def add(self, mention: Mention) -> None:
        self.mention_count += 1
        self.batch_ids.add(int(mention.batch_idx))
        self.weights.append(float(mention.weight))


@dataclass
class SemanticCluster:
    members: List[LexicalConcept] = field(default_factory=list)
    canonical: str = ""
    support_count: int = 0
    global_support: float = 0.0
    batch_coverage: float = 0.0
    semantic_quality: float = 0.0
    specificity_penalty: float = 0.0
    final_weight: float = 0.0


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


class HybridKeywordAggregator:
    """
    Hybrid specialization aggregation over batch-level weighted keywords.

    Design:
    1) Deterministic clustering first (lexical normalization + embedding similarity).
    2) Optional LLM semantic cleanup for cluster labels only.
    3) Deterministic global reweighting:
       final = a*semantic_quality + b*global_support + c*batch_coverage - d*specificity_penalty
    """

    DEFAULT_SIMILARITY_THRESHOLD = 0.84
    DEFAULT_LABEL_WORKERS = 4
    DEFAULT_A = 0.45
    DEFAULT_B = 0.35
    DEFAULT_C = 0.20
    DEFAULT_D = 0.25

    _STOPWORDS = {
        "a",
        "an",
        "and",
        "for",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }

    def __init__(
        self,
        *,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        label_workers: int = DEFAULT_LABEL_WORKERS,
        llm_enabled: bool = True,
        llm_model_id: Optional[str] = None,
        a: float = DEFAULT_A,
        b: float = DEFAULT_B,
        c: float = DEFAULT_C,
        d: float = DEFAULT_D,
        embedding_client: Optional[Any] = None,
    ):
        self.similarity_threshold = max(0.0, min(1.0, float(similarity_threshold)))
        self.label_workers = max(1, int(label_workers))
        self.llm_enabled = bool(llm_enabled)
        self.llm_model_id = (
            str(llm_model_id or "").strip()
            or str(settings.sonnet or settings.haiku or settings.opus or "").strip()
        )
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.embedding_client = embedding_client or get_embedding_client().build()

    # =================================
    # 1) Input normalization
    # =================================
    @staticmethod
    def _norm(text: Any) -> str:
        s = str(text or "").strip().lower()
        s = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", s)
        s = re.sub(r"[^a-z0-9\s\-_/]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _safe_weight(value: Any, default: float = 0.0) -> float:
        try:
            v = float(value)
        except Exception:
            v = float(default)
        return max(0.0, min(1.0, v))

    def _collect_mentions_from_weighted_batches(
        self,
        weighted_batches: List[Dict[str, Any]],
        *,
        section: str,
    ) -> List[Mention]:
        mentions: List[Mention] = []
        for i, batch in enumerate(list(weighted_batches or []), start=1):
            batch_idx = int((batch or {}).get("batch_idx") or i)
            rows = list(((batch or {}).get("weighted") or {}).get(section) or [])
            for row in rows:
                text = str((row or {}).get("t") or "").strip()
                if not text:
                    continue
                mentions.append(
                    Mention(
                        text=text,
                        norm=self._norm(text),
                        batch_idx=batch_idx,
                        weight=self._safe_weight((row or {}).get("w"), default=0.0),
                    )
                )
        return mentions

    def _build_lexical_concepts(self, mentions: Iterable[Mention]) -> List[LexicalConcept]:
        concepts: Dict[str, LexicalConcept] = {}
        for mention in list(mentions):
            if not mention.norm:
                continue
            if mention.norm not in concepts:
                concepts[mention.norm] = LexicalConcept(text=mention.text, norm=mention.norm)
            concepts[mention.norm].add(mention)
        return list(concepts.values())

    # =================================
    # 2) Global clustering
    # =================================
    def _semantic_cluster(self, concepts: List[LexicalConcept]) -> List[SemanticCluster]:
        if not concepts:
            return []
        if len(concepts) == 1:
            return [SemanticCluster(members=[concepts[0]])]

        texts = [c.norm or c.text for c in concepts]
        vectors = embed_texts(texts, embedding_client=self.embedding_client)
        sims = cosine_sim_matrix(vectors, vectors)
        uf = _UnionFind(len(concepts))

        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                if float(sims[i][j]) >= float(self.similarity_threshold):
                    uf.union(i, j)

        grouped: Dict[int, List[LexicalConcept]] = {}
        for idx, concept in enumerate(concepts):
            root = uf.find(idx)
            grouped.setdefault(root, []).append(concept)

        clusters: List[SemanticCluster] = []
        for members in grouped.values():
            members_sorted = sorted(
                members,
                key=lambda m: (-int(m.mention_count), -len(m.batch_ids), -max(m.weights or [0.0])),
            )
            clusters.append(SemanticCluster(members=members_sorted))
        return clusters

    # =================================
    # 3) Optional LLM label cleanup
    # =================================
    def _build_cluster_rename_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You rename a semantic cluster of specialization phrases.\n"
                    "Rules:\n"
                    "- Use only provided aliases.\n"
                    "- Prefer recurring high-level phrasing over one-off details.\n"
                    "- Keep concise technical wording.\n"
                    "- Do not invent new topics or claims.\n"
                    "- Return JSON only: {\"canonical\": string}.\n",
                ),
                ("human", "INPUT JSON:\n{input_json}"),
            ]
        )
        llm = get_llm_client(model_id=self.llm_model_id).build()
        return prompt | llm.with_structured_output(ClusterRenameOut)

    def _token_set(self, text: str) -> set[str]:
        return {
            tok
            for tok in re.findall(r"[a-z0-9]+", str(text or "").lower())
            if tok and tok not in self._STOPWORDS
        }

    def _validate_llm_label(self, *, candidate: str, aliases: List[str], fallback: str) -> str:
        """
        Deterministic safety anchor:
        keep LLM label only if it overlaps source alias vocabulary.
        """
        label = str(candidate or "").strip()
        if not label:
            return fallback
        label_tokens = self._token_set(label)
        if not label_tokens:
            return fallback

        alias_tokens = set()
        for alias in list(aliases or []):
            alias_tokens |= self._token_set(alias)

        if not (label_tokens & alias_tokens):
            return fallback
        return label

    def _rename_clusters_llm(self, *, section: str, clusters: List[SemanticCluster]) -> None:
        if not self.llm_enabled or not clusters:
            return

        pool_size = resolve_pool_size(max_workers=self.label_workers, task_count=len(clusters))
        if pool_size <= 0:
            return
        rename_chain = self._build_cluster_rename_chain()

        def _run_one(item: Tuple[int, SemanticCluster]) -> Tuple[int, str]:
            idx, cluster = item
            anchor = cluster.members[0].text if cluster.members else ""
            aliases = [m.text for m in list(cluster.members or [])[:12]]
            payload = {
                "section": section,
                "anchor": anchor,
                "aliases": aliases,
                "support_count": int(sum(int(m.mention_count) for m in cluster.members)),
                "batch_support_count": int(len({b for m in cluster.members for b in m.batch_ids})),
            }
            out = rename_chain.invoke({"input_json": json.dumps(payload, ensure_ascii=False)})
            if isinstance(out, ClusterRenameOut):
                candidate = str(out.canonical or "").strip()
            elif isinstance(out, dict):
                candidate = str((out or {}).get("canonical") or "").strip()
            else:
                candidate = str(getattr(out, "canonical", "") or "").strip()
            return idx, self._validate_llm_label(candidate=candidate, aliases=aliases, fallback=anchor)

        def _on_error(_idx: int, item: Tuple[int, SemanticCluster], exc: Exception) -> Tuple[int, str]:
            logger.exception("Hybrid keyword cluster rename failed section=%s: %s", section, exc)
            cluster = item[1]
            fallback = cluster.members[0].text if cluster.members else ""
            return int(item[0]), fallback

        rows = parallel_map(
            list(enumerate(list(clusters or []))),
            max_workers=pool_size,
            run_item=_run_one,
            on_error=_on_error,
        )
        by_idx = {int(i): str(label or "").strip() for i, label in rows}
        for i, cluster in enumerate(clusters):
            cluster.canonical = by_idx.get(i) or (cluster.members[0].text if cluster.members else "")

    # =================================
    # 4) Global reweighting
    # =================================
    @staticmethod
    def _specificity_penalty(text: str, support_count: int) -> float:
        label = str(text or "").strip()
        if not label:
            return 1.0
        tokens = re.findall(r"[a-z0-9]+", label.lower())
        token_len = len(tokens)
        has_numeric = any(ch.isdigit() for ch in label)

        penalty = 0.0
        if int(support_count) <= 1:
            penalty += 0.25
        penalty += max(0.0, (float(token_len) - 12.0) / 18.0) * 0.25
        if has_numeric:
            penalty += 0.10
        return max(0.0, min(1.0, penalty))

    def _score_cluster(self, cluster: SemanticCluster, *, total_batches: int, max_support: int) -> None:
        support_count = int(sum(int(m.mention_count) for m in cluster.members))
        batch_ids = {b for m in cluster.members for b in m.batch_ids}
        quality_values = [w for m in cluster.members for w in m.weights]
        semantic_quality = (sum(quality_values) / float(len(quality_values))) if quality_values else 0.0
        global_support = (
            math.log1p(float(support_count)) / math.log1p(float(max(max_support, 1)))
            if support_count > 0
            else 0.0
        )
        batch_coverage = float(len(batch_ids)) / float(max(total_batches, 1))
        canonical = cluster.canonical or (cluster.members[0].text if cluster.members else "")
        specificity_penalty = self._specificity_penalty(canonical, support_count)

        final = (
            (self.a * float(semantic_quality))
            + (self.b * float(global_support))
            + (self.c * float(batch_coverage))
            - (self.d * float(specificity_penalty))
        )

        cluster.canonical = str(canonical or "").strip()
        cluster.support_count = int(support_count)
        cluster.global_support = max(0.0, min(1.0, float(global_support)))
        cluster.batch_coverage = max(0.0, min(1.0, float(batch_coverage)))
        cluster.semantic_quality = max(0.0, min(1.0, float(semantic_quality)))
        cluster.specificity_penalty = max(0.0, min(1.0, float(specificity_penalty)))
        cluster.final_weight = max(0.0, min(1.0, float(final)))

    # =================================
    # 5) Public API
    # =================================
    def aggregate_section_from_weighted_batches(
        self,
        *,
        section: str,
        weighted_batches: List[Dict[str, Any]],
        max_items: int = 15,
    ) -> Dict[str, Any]:
        """
        Aggregate one section from weighted batch outputs.

        Expected input shape:
        weighted_batches = [
          {"batch_idx": 1, "weighted": {"research": [{"t":"...", "w":0.8}], "application": [...]}}
        ]
        """
        sec = str(section or "").strip().lower()
        if sec not in {"research", "application"}:
            raise ValueError("section must be 'research' or 'application'")

        mentions = self._collect_mentions_from_weighted_batches(weighted_batches, section=sec)
        concepts = self._build_lexical_concepts(mentions)
        clusters = self._semantic_cluster(concepts)
        self._rename_clusters_llm(section=sec, clusters=clusters)

        total_batches = len({int((b or {}).get("batch_idx") or 0) for b in list(weighted_batches or [])}) or 1
        max_support = max((sum(m.mention_count for m in c.members) for c in clusters), default=1)

        for cluster in clusters:
            if not cluster.canonical:
                cluster.canonical = cluster.members[0].text if cluster.members else ""
            self._score_cluster(cluster, total_batches=total_batches, max_support=max_support)

        clusters = sorted(
            clusters,
            key=lambda c: (-float(c.final_weight), -int(c.support_count), -float(c.batch_coverage), c.canonical),
        )
        limit = max(1, int(max_items or 1))
        selected = clusters[:limit]

        specialization = [
            {
                "t": str(cluster.canonical),
                "w": float(round(cluster.final_weight, 6)),
                "support_count": int(cluster.support_count),
                "global_support": float(round(cluster.global_support, 6)),
                "batch_coverage": float(round(cluster.batch_coverage, 6)),
                "semantic_quality": float(round(cluster.semantic_quality, 6)),
                "specificity_penalty": float(round(cluster.specificity_penalty, 6)),
                "members": [m.text for m in list(cluster.members or [])],
            }
            for cluster in selected
            if str(cluster.canonical or "").strip()
        ]

        return {
            "section": sec,
            "specialization": specialization,
            "cluster_count": len(clusters),
            "mention_count": len(mentions),
            "total_batches": int(total_batches),
            "scoring_formula": "a*semantic_quality + b*global_support + c*batch_coverage - d*specificity_penalty",
            "scoring_params": {"a": self.a, "b": self.b, "c": self.c, "d": self.d},
        }

    def aggregate_from_weighted_batches(
        self,
        *,
        weighted_batches: List[Dict[str, Any]],
        max_items_per_section: int = 15,
    ) -> Dict[str, Any]:
        """
        Aggregate both research and application sections.
        """
        return {
            "research": self.aggregate_section_from_weighted_batches(
                section="research",
                weighted_batches=weighted_batches,
                max_items=max_items_per_section,
            ),
            "application": self.aggregate_section_from_weighted_batches(
                section="application",
                weighted_batches=weighted_batches,
                max_items=max_items_per_section,
            ),
        }
