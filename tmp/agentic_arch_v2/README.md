# Agentic Arch V2 (Conversational Faculty <-> Grant)

Independent implementation with iterative back-and-forth between agents.

## What this adds
- LLM-based query planning from faculty profession profile
- Parallel query execution to grant agent
- Grant requirement sub-calls in parallel (domains, specializations, eligibility, deliverables)
- Reciprocal follow-up loop:
  - Grant agent can ask faculty agent for missing evidence
  - Faculty agent can ask grant agent for requirement priority clarification
- Confidence scoring per query
- Stop rules (`max_rounds`, `confidence_converged`, `no_more_queries`)

## Files
- `models.py`: query/answer/round/result models
- `planner.py`: LLM query planner + fallback plan
- `faculty_agent.py`: faculty conversation responder
- `grant_agent.py`: grant conversation responder with parallel calls
- `orchestrator.py`: loop controller and stop rules
- `run_demo.py`: local in-memory demo
- `run_neo4j.py`: Neo4j-backed execution

## Run (demo)
```bash
python tmp/agentic_arch_v2/run_demo.py
```

## Run (Neo4j)
```bash
python tmp/agentic_arch_v2/run_neo4j.py \
  --email "alan.fern@oregonstate.edu" \
  --candidate-grant-k 30 \
  --top-k 5 \
  --max-rounds 3
```

If Bedrock planner is unavailable, run deterministic fallback:
```bash
python tmp/agentic_arch_v2/run_neo4j.py \
  --email "alan.fern@oregonstate.edu" \
  --disable-llm-planner
```
