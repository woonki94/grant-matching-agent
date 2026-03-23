# GraphRAG-Based Replacement of LLM-Only Matching
**Project Progress Report**  
**Date:** March 11, 2026  
**Repository:** `/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher`

## 1) Objective
Replace expensive, brittle one-shot LLM matching with a graph-native retrieval and scoring pipeline that:
- stores structured evidence in Neo4j,
- preserves traceability from keyword/domain back to source chunks,
- supports reusable edges for fast retrieval,
- reduces LLM dependence in online matching.

The target was to move from prompt-heavy faculty<->grant matching to GraphRAG-first matching with optional learned reranking.

## 2) Starting Problems
Initial constraints and pain points:
- Data split: embeddings existed in relational DB while many filter fields (e.g., close date) were in Neo4j.
- Source traceability gaps: keyword-level outputs did not reliably map back to exact chunks.
- Cosine-only keyword matching produced semantically weak links for short phrases.
- LLM call volume/cost risk for broad facultyxgrant comparisons.
- Prompt fragility (ChatPromptTemplate variable escaping and JSON-shape drift) caused repeated runtime failures.

## 3) High-Level Architecture You Established

### 3.1 Agent interface scaffolding
You introduced explicit interfaces for supervisor/faculty/grant roles:
- Supervisor interface: [graph_rag/agentic_architecture/agents/supervisor_agent.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/agentic_architecture/agents/supervisor_agent.py)
- Faculty interface: [graph_rag/agentic_architecture/agents/faculty_agent.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/agentic_architecture/agents/faculty_agent.py)
- Grant interface: [graph_rag/agentic_architecture/agents/grant_agent.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/agentic_architecture/agents/grant_agent.py)
- Decision vocabulary (expanded): [graph_rag/agentic_architecture/state.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/agentic_architecture/state.py)

### 3.2 Tool layering for search depth
You organized search tools into primary/intermediary/deep behavior:
- primary: domain threshold + hard open-grant filter
- intermediary: specialization embedding comparison
- deep: publication/additional-info/attachment chunk querying

Implemented in [graph_rag/agentic_architecture/tools/supervisor_search_tool.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/agentic_architecture/tools/supervisor_search_tool.py).

## 4) Graph Data Foundation (Faculty + Grant)

### 4.1 Chunk-first ingestion into Neo4j
You moved matching foundation to chunk-level graph evidence:
- Grant chunking and ingestion: [graph_rag/grant/sync_neo4j.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/grant/sync_neo4j.py)
- Faculty chunking and ingestion: [graph_rag/faculty/sync_neo4j.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/faculty/sync_neo4j.py)

Current chunking defaults in sync layer:
- `chunk_size_chars = 1200`
- `chunk_overlap_chars = 150`
- bounded chunks/source (`max_chunks_per_source`)

### 4.2 Key source modeling decisions you enforced
- Faculty biography now participates as chunk evidence (`HAS_BIO_CHUNK` path present in faculty sync).
- Faculty publications modeled as first-class evidence nodes (`FacultyPublication`) and treated in matching pipelines.
- Grant summary/additional-info/attachments are chunked into `GrantTextChunk`.
- Initial graph upload phase was changed to avoid initial keyword-node upload; keyword linking moved to dedicated sync phase.

## 5) Keyword Generator V2 (Faculty + Grant)

### 5.1 Faculty V2 pipeline
Main file: [services/keywords/faculty_keyword_generator_v2.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/services/keywords/faculty_keyword_generator_v2.py)

Implemented behavior:
- Pulls context from Neo4j chunks (not rebuilding chunk content in relational layer).
- Packs chunks into chunksets constrained by `max_context_chars - reserve_prompt_chars`.
- Runs per-chunkset extraction (parallelizable):
  - domain mentions,
  - specialization mentions with snippet/evidence IDs.
- Supports dynamic specialization count bounds based on chunkset count.
- Supports domain merge stage.
- Runs weighting stages (domain + specialization with domain associations).
- Includes LLM I/O logging hooks for each step.
- Includes `run_all_faculty_keyword_pipelines_parallel`.

### 5.2 Grant V2 pipeline
Main file: [services/keywords/grant_keyword_generator_v2.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/services/keywords/grant_keyword_generator_v2.py)

Mirrors faculty process with grant-specific prompts and output semantics:
- chunkset-based extraction,
- domain and specialization generation,
- merge/weighting stages,
- parallel per-chunkset execution,
- full-batch grant pipeline support.

## 6) Direct Neo4j Keyword Sync (V2)
You created direct-to-graph sync paths (without requiring intermediate relational keyword persistence for final linking stage):
- Faculty direct sync: [graph_rag/faculty/sync_keyword_links_direct_v2.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/faculty/sync_keyword_links_direct_v2.py)
- Grant direct sync: [graph_rag/grant/sync_keyword_links_direct_v2.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/grant/sync_keyword_links_direct_v2.py)

Under the hood, sync/link modules create weighted evidence graph structures:

### Faculty link graph
From [graph_rag/faculty/sync_keyword_links_v2.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/faculty/sync_keyword_links_v2.py):
- `FacultyKeyword` nodes (domain + specialization), with embeddings.
- `FacultyDomainGate` node per faculty (`HAS_DOMAIN_GATE`).
- `DomainKeywordShared` shared nodes via `MAPS_TO_SHARED_DOMAIN`.
- Evidence edges:
  - `DOMAIN_SUPPORTED_BY_FACULTY_CHUNK`
  - `DOMAIN_SUPPORTED_BY_FACULTY_PUBLICATION`
  - `FACULTY_DOMAIN_HAS_SPECIALIZATION`
  - `FACULTY_CHUNK_SUPPORTS_SPECIALIZATION`
  - `FACULTY_PUBLICATION_SUPPORTS_SPECIALIZATION`

### Grant link graph
From [graph_rag/grant/sync_keyword_links_v2.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/grant/sync_keyword_links_v2.py):
- `GrantKeyword` nodes (domain + specialization), with embeddings.
- `GrantDomainGate` node per grant (`HAS_DOMAIN_GATE`).
- `DomainKeywordShared` mappings.
- Evidence edges:
  - `DOMAIN_SUPPORTED_BY_GRANT_CHUNK`
  - `GRANT_DOMAIN_HAS_SPECIALIZATION`
  - `GRANT_CHUNK_SUPPORTS_SPECIALIZATION`

## 7) Matching Evolution (What You Tried)
You iterated through multiple matchers in `graph_rag/matching`, including:
- domain prefilter + spec-chunk linkers,
- attention/cross-encoder variants,
- hybrid cosine+attention pipelines,
- spec<->spec edge construction and retrieval strategies,
- candidate ranking variants (`v3`, `v4`, domain-gate rankers, coverage-based rankers).

Representative files:
- [graph_rag/matching/domain_prefilter_spec_chunk_hybrid_linker.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/matching/domain_prefilter_spec_chunk_hybrid_linker.py)
- [graph_rag/matching/domain_gate_spec_keyword_hybrid_linker_v3.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/matching/domain_gate_spec_keyword_hybrid_linker_v3.py)
- [graph_rag/matching/retrieve_grants_by_v4.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/matching/retrieve_grants_by_v4.py)
- [graph_rag/matching/retrieve_grants_by_domain_weight_gate.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/matching/retrieve_grants_by_domain_weight_gate.py)

## 8) Current Retrieval/Scoring Formula (Active)
Current retrieval implementation:
- [graph_rag/matching/retrieve_grants_by_domain_gate_spec_coverage.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/matching/retrieve_grants_by_domain_gate_spec_coverage.py)

Current pipeline:
1. **Stage 1 domain gate candidates** via shared-domain mappings (`min_domain_weight` default 0.6).
2. **Stage 2 spec coverage** over existing `FACULTY_SPEC_MATCHES_GRANT_SPEC` edges (min similarity default 0.4).

Current scoring (as of latest revision):
- Pair: `score(g_i, f_j) = sim(g_i, f_j) * D(g_i, f_j)^2`
- Per-grant-spec best: `best_i = max_j score(g_i, f_j)`
- Coverage aggregate: `coverage_sum = SUM_i best_i`
- Final rank: `final_score = coverage_sum * domain_rank_score`

The script now also prints top-10 with explicit score fields:
- `final_score`
- `grant_spec_coverage_sum`
- `grant_spec_coverage_ratio`
- `domain_rank_score`

## 9) Cross-Encoder Distillation and Finetuning Path

### 9.1 Dataset build (ranking distillation)
Implemented in [train_cross_encoder/build_llm_spec_pair_dataset.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/train_cross_encoder/build_llm_spec_pair_dataset.py):
- Graph pull of grant/faculty specialization + embeddings.
- Candidate sampling (top-k + hard negatives + random negatives).
- Batch LLM ranking labels.
- Pairwise dataset generation for training.
- Parallelized LLM batching via thread pool.

Latest successful dataset artifact:
- `train_cross_encoder/dataset/spec_pair_rankdistill_train_20260311_070746.jsonl`
- row count: **197,980**
- LLM fallback ratio: **0.0** (from corresponding `.meta.json`)

### 9.2 Reranker training + inference
- Training script: [train_cross_encoder/train_bge_reranker_base.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/train_cross_encoder/train_bge_reranker_base.py)
- Inference script: [train_cross_encoder/infer_bge_reranker.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/train_cross_encoder/infer_bge_reranker.py)

Important compatibility work handled during this phase:
- `transformers` argument/version mismatch fixes,
- tokenizer/model loading fallback handling,
- dataloader collation fixes for pair fields,
- CPU/MPS/CUDA device handling paths.

### 9.3 Graph linking with finetuned reranker
Implemented in [graph_rag/matching/link_spec_keywords_with_finetuned_reranker.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/matching/link_spec_keywords_with_finetuned_reranker.py):
- Creates/refreshes `FACULTY_SPEC_MATCHES_GRANT_SPEC` edges.
- Stores:
  - `model_raw_score`
  - `model_score`
  - `weighted_score`
  - `score`
  - weight/confidence metadata and run metadata.
- Supports score transform modes (`clip|sigmoid|raw`).

## 10) Operational and Debugging Work Completed
You resolved or worked around multiple system-level blockers:
- Prompt-template escaping errors (`{}` in schema examples) causing KeyError.
- Empty LLM output in large context windows -> chunkset autosplit + context reserve strategy.
- Cursor/pool lifecycle error during relational iteration (`named cursor isn't valid anymore`) by redesigning flow around chunk retrieval and pipeline isolation.
- Terminal/repo slowness traced to prompt-side git status behavior, not git command failure.
- Process cleanup for stray Python/git helper processes.

## 11) What Is Functionally Replaced vs LLM Today

### Replaced / reduced
- One-shot LLM-only final matching is no longer the only path.
- Retrieval and most candidate ranking are graph+edge+embedding driven.
- Domain gate + spec edge coverage can run deterministically using graph state.

### Still LLM-dependent
- Keyword generation stages (domain/specialization extraction + weighting).
- Distillation data labeling for reranker training (though amortized/offline).

## 12) Current State Summary
You now have:
- A full chunk-evidence graph foundation.
- Faculty/grant V2 keyword generation pipelines with chunkset controls and parallelism.
- Direct Neo4j keyword sync with domain gates and weighted evidence edges.
- Multiple matcher variants, including current domain-gated spec-coverage retrieval.
- Cross-encoder training/inference toolchain and graph linking integration.

Net result: matching has shifted from expensive online LLM reasoning to a GraphRAG-centric system with explainable edges, reusable features, and tunable scoring.

## 13) Recommended Next Steps (Technical)
1. Freeze one canonical online scorer (`retrieve_grants_by_domain_gate_spec_coverage.py`) and archive others as experiments.
2. Add evaluation set (faculty, grant, human label) and report Precision@K / nDCG.
3. Add edge freshness policy and incremental re-sync for changed faculty context windows.
4. Add sanity constraints in scoring (e.g., minimum domain overlap floor before counting coverage).
5. Add regression tests for scoring invariants and parser/prompt JSON schemas.

---

## Appendix A: Key Files
- Faculty keyword pipeline v2: [services/keywords/faculty_keyword_generator_v2.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/services/keywords/faculty_keyword_generator_v2.py)
- Grant keyword pipeline v2: [services/keywords/grant_keyword_generator_v2.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/services/keywords/grant_keyword_generator_v2.py)
- Faculty direct sync v2: [graph_rag/faculty/sync_keyword_links_direct_v2.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/faculty/sync_keyword_links_direct_v2.py)
- Grant direct sync v2: [graph_rag/grant/sync_keyword_links_direct_v2.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/grant/sync_keyword_links_direct_v2.py)
- Faculty keyword link sync internals: [graph_rag/faculty/sync_keyword_links_v2.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/faculty/sync_keyword_links_v2.py)
- Grant keyword link sync internals: [graph_rag/grant/sync_keyword_links_v2.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/grant/sync_keyword_links_v2.py)
- Current retrieval scorer: [graph_rag/matching/retrieve_grants_by_domain_gate_spec_coverage.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/matching/retrieve_grants_by_domain_gate_spec_coverage.py)
- Reranker edge linker: [graph_rag/matching/link_spec_keywords_with_finetuned_reranker.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/graph_rag/matching/link_spec_keywords_with_finetuned_reranker.py)
- Distillation dataset builder: [train_cross_encoder/build_llm_spec_pair_dataset.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/train_cross_encoder/build_llm_spec_pair_dataset.py)
- Reranker trainer: [train_cross_encoder/train_bge_reranker_base.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/train_cross_encoder/train_bge_reranker_base.py)
- Reranker inference: [train_cross_encoder/infer_bge_reranker.py](/Users/kimwoonki/Desktop/OSU/Fall2025/Capstone/GrantFetcher/train_cross_encoder/infer_bge_reranker.py)

