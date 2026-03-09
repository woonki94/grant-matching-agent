# LangGraph Agentic Flow (Faculty <-> Grant)

Independent LangGraph implementation with explicit nodes, edges, and router-based dispatch.

## Graph shape

`bootstrap -> round_prepare -> router -> (grant_answer | faculty_answer | skip_query) -> integrate_answer -> router`

When a round is exhausted:

`router -> finalize_round -> (round_prepare | rank_and_finish)`

## Node responsibilities

- `bootstrap`: build faculty profile, find candidate grants, plan/expand initial queries.
- `round_prepare`: dedupe and cap per-round queries.
- `router`: pop one query and route to the node that can answer it.
- `grant_answer`: answer a single grant-targeted query.
- `faculty_answer`: answer a single faculty-targeted query.
- `integrate_answer`: store answer, collect follow-up queries, update confidence failure flags.
- `finalize_round`: stop-or-continue decision (`max_rounds`, `confidence_converged`, `no_more_queries`).
- `rank_and_finish`: collect grant snapshots and produce one-to-one ranked matches.

## Run demo

```bash
python tmp/agentic_langgraph/run_demo.py
```

## Run with Neo4j

```bash
python tmp/agentic_langgraph/run_neo4j.py \
  --email "alan.fern@oregonstate.edu" \
  --candidate-grant-k 30 \
  --top-k 5 \
  --max-rounds 3
```

Fallback planner (no Bedrock planner call):

```bash
python tmp/agentic_langgraph/run_neo4j.py \
  --email "alan.fern@oregonstate.edu" \
  --disable-llm-planner
```

